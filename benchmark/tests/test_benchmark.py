"""Offline evaluation regressions, including real subprocess and timeout tests."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark import data, metrics, runner
from benchmark.common import digest, object_hash, read_json, write_json, write_jsonl
from benchmark.report import compare_runs, create_report


FAKE_FRAMEWORK = '''import json, time
from pathlib import Path
QWEN = "test-qwen"
JEV = "test-jev"
def post(url, key, body, folder, name, **kwargs):
    return {"usage": {"prompt_tokens": 7, "completion_tokens": 2}}
def observe(task):
    if task == "sleep": time.sleep(30)
    return post("https://qwen.example/v1", "NEVER_LOG_THIS_KEY", {"model": QWEN, "messages": [{"content": [{"type":"image_url", "image_url":"mock_image"}]}]}, None, "01-observe")
def select_windows(task): return None
def generate_candidates(task): return None
def match_windows(task):
    if task == "fail": raise ValueError("mock matching failure")
def execute_experts(task): return None
def package_evidence(task): return None
def judge_checks(task): return None
def evaluate_final(task):
    return post("https://api.typesafe.ai/v1/systemone", "NEVER_LOG_THIS_KEY", {"model": JEV}, None, "07-evaluate")
def run(args, sample_interval_seconds=2):
    args.out.mkdir()
    for function in (observe,select_windows,generate_candidates,match_windows,execute_experts,package_evidence,judge_checks,evaluate_final): function(args.task)
    result={"status":"scored", "score":4, "jev_choice":"level_4"}
    (args.out/"result.json").write_text(json.dumps(result))
    return result
'''


def metadata_fixture(path, size=60):
    rows = [{"file_name": f"source_{i%3}/episode_{i}_attempt_1_score_{i%5+1}.mp4",
             "task": f"Move object {i}", "reward": i % 5 + 1,
             "gpt5_mini_check": "DO NOT SEND VERIFICATION OR LABELS"} for i in range(size)]
    write_jsonl(path, rows)
    return rows


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.metadata = self.root / "metadata.jsonl"
        metadata_fixture(self.metadata)

    def test_nested_reproducible_subsets_and_no_input_labels(self):
        a = data.prepare(self.metadata, self.root / "a")
        b = data.prepare(self.metadata, self.root / "b")
        self.assertEqual(a, b)
        self.assertEqual(a["pilot_ids"], a["development_ids"][:10])
        self.assertEqual(len(set(a["development_ids"])), 50)
        self.assertEqual(set(a["reference_counts"]), {"1", "2", "3", "4", "5"})
        raw = (self.root / "a" / "inputs.jsonl").read_text()
        for forbidden in ("reward", "score_", "gpt5_mini", "verification", "origin_group"):
            self.assertNotIn(forbidden, raw)

    def test_metadata_change_is_rejected(self):
        data.prepare(self.metadata, self.root / "a")
        with (self.root / "a" / "labels.jsonl").open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            data.load_dataset(self.root / "a")

    def test_validation_only_and_strict_labels(self):
        for edits in ({"split": "test"}, {"reward": True}, {"reward": 6}, {"task": ""}):
            row = {"file_name": "s/video.mp4", "task": "Move block", "reward": 3, **edits}
            write_jsonl(self.metadata, [row])
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                data.load_metadata(self.metadata)

    def test_no_path_traversal_or_mutable_revision(self):
        for value in ("../x.mp4", "/x.mp4", "a/../../x.mp4", "a\\b.mp4", "https://host/x.mp4", "x.mp4?q=1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                data.safe_filename(value)
        with self.assertRaises(ValueError):
            data.hub_url("main")
        self.assertIn("/val/", data.hub_url(data.REVISION))

    def test_duplicate_pairs_rejected_and_same_video_different_tasks_allowed(self):
        row = {"file_name": "s/video.mp4", "task": "Move block", "reward": 3}
        write_jsonl(self.metadata, [row, row])
        with self.assertRaises(ValueError):
            data.load_metadata(self.metadata)
        write_jsonl(self.metadata, [row, {**row, "task": "Move cup"}])
        self.assertEqual(len(data.load_metadata(self.metadata)), 2)

    def test_local_copy_inventory_and_corruption(self):
        plan = data.prepare(self.metadata, self.root / "a", size=12, pilot_size=5)
        _, inputs, labels = data.load_dataset(self.root / "a")
        for row in inputs:
            path = self.root / "local" / labels[row["sample_id"]]["file_name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic video fixture")
        result = data.download(self.root / "a", local_root=self.root / "local")
        self.assertFalse(result["failures"])
        self.assertEqual(result["verified_videos"], 5)
        self.assertEqual(data.download(self.root / "a", local_root=self.root / "local"), result)
        first = self.root / "a" / inputs[0]["video_path"]
        first.write_bytes(b"corrupt")
        result = data.download(self.root / "a", local_root=self.root / "local")
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(first.read_bytes(), b"corrupt")

    def test_sampler_size_and_no_overwrite(self):
        for size, pilot in ((4, 5), (10, 4), (100, 10)):
            with self.assertRaises(ValueError):
                data.prepare(self.metadata, self.root / "a", size=size, pilot_size=pilot)
        data.prepare(self.metadata, self.root / "a")
        with self.assertRaises(FileExistsError):
            data.prepare(self.metadata, self.root / "a")


def metric_row(index, reward, score=None, status=None, source="source_a"):
    return {"sample_id": str(index), "reward": reward, "score": score,
            "status": status or ("scored" if score is not None else "insufficient_evidence"),
            "source": source, "origin_group": "g" + str(index), "duration_bin": "<10s"}


class MetricTests(unittest.TestCase):
    def test_missing_predictions_retained_and_bounds_correct(self):
        rows = [metric_row(0, 5, 4), metric_row(1, 1, 1), metric_row(2, 3),
                metric_row(3, 2, status="failed"), metric_row(4, 5, status="not_run")]
        result = metrics.evaluate(rows, iterations=20)
        overall = result["overall"]
        self.assertEqual(overall["coverage"], .4)
        self.assertEqual(overall["selective_mae"], .5)
        self.assertIsNone(overall["all_sample_mae"])
        self.assertEqual(overall["answered_and_correct_all"], .2)
        self.assertEqual(overall["hypothetical_all_sample_mae_bounds"], [.2, 2.])
        self.assertEqual(result["attempted"], 4)
        self.assertEqual(result["completion_rate_attempted"], .75)
        self.assertEqual(overall["confusion_true_rows_predicted_columns"][4][3], 1)

    def test_source_macro_differs_from_pooled(self):
        rows = [metric_row(0, 1, 1), metric_row(1, 1, 1), metric_row(2, 5, 1, source="source_b")]
        result = metrics.evaluate(rows, iterations=10)
        self.assertAlmostEqual(result["overall"]["all_sample_mae"], 4/3)
        self.assertEqual(result["source_macro_mae_all_covered"], 2.)
        rows[2].update(status="failed", score=None)
        self.assertIsNone(metrics.evaluate(rows, iterations=10)["source_macro_mae_all_covered"])

    def test_all_abstentions_empty_and_bootstrap_reproducibility(self):
        rows = [metric_row(i, 3) for i in range(5)]
        self.assertIsNone(metrics.evaluate(rows, iterations=10)["overall"]["selective_mae"])
        self.assertIsNone(metrics.evaluate([], iterations=0)["overall"]["coverage"])
        rows[0].update(status="scored", score=2)
        self.assertEqual(metrics.bootstrap_mae(rows, iterations=30), metrics.bootstrap_mae(rows, iterations=30))

    def test_costs_and_usage_missing_are_not_zero(self):
        records = [{"calls": [{"provider": "qwen", "transport_attempted": True,
                    "images": 2, "request_bytes": 100, "status": "returned",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
                   {"provider": "qwen", "transport_attempted": True, "images": 4,
                    "request_bytes": 200, "status": "in_flight", "usage": None}]}]
        result = metrics.resources(records, {"qwen": {"input_usd_per_million": 1., "output_usd_per_million": 2.}})
        self.assertAlmostEqual(result["estimated_usd_known_calls"], .00012)
        self.assertEqual(result["calls_not_priced"], 1)
        self.assertEqual(result["providers"]["qwen"]["calls_missing_input_tokens"], 1)
        self.assertEqual(result["image_presentations_attempted"], 6)
        self.assertEqual(result["in_flight_calls_at_stop"], 1)
        with self.assertRaises(ValueError):
            metrics.validate_rates({"qwen": {"per_request_usd": float("nan")}})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.framework = self.root / "framework"
        self.framework.mkdir()
        (self.framework / "evaluate_rollout.py").write_text(FAKE_FRAMEWORK)
        self.model = self.root / "model.onnx"
        self.model.write_bytes(b"fixture, not a real ONNX model")
        self.metadata = self.root / "metadata.jsonl"
        metadata_fixture(self.metadata)
        self.dataset = self.root / "dataset"
        data.prepare(self.metadata, self.dataset, size=12, pilot_size=5)
        _, self.inputs, _ = data.load_dataset(self.dataset)
        for row in self.inputs:
            video = self.dataset / row["video_path"]
            video.parent.mkdir(exist_ok=True)
            video.write_bytes(b"synthetic video")
        self.configuration = {"framework": {"root": str(self.framework)}, "python": sys.executable,
                              "yolo_model": {"path": str(self.model)}, "threshold_path": None,
                              "sample_interval_seconds": 2., "max_calls_per_episode": None,
                              "timeout_seconds": 5}

    def sample(self, row=None, name="one", **config):
        return runner.execute_sample(row or self.inputs[0], self.dataset, self.root / name,
                                     {**self.configuration, **config})

    def test_comparison_uses_full_denominator_and_common_scored_pairs(self):
        rubric = "prompts/roboreward_rubric.json"
        common = {"plan_sha256": digest(self.dataset / "plan.json"),
                  "framework": {"files": {rubric: "fixture-rubric"}}}
        direct_run, framework_run = self.root / "direct", self.root / "original"
        sample_id = self.inputs[0]["sample_id"]
        for run, configuration, score in ((direct_run, {**common, "method": "direct-jev"}, 5),
                                          (framework_run, common, 4)):
            write_json(run / "configuration.json", configuration)
            write_json(run / "runtime.json", {"models": {"jev": "jev-fixture"}})
            write_json(run / "episodes" / sample_id / "record.json",
                       {"configuration_sha256": object_hash(configuration), "status": "scored", "score": score})
        report = self.root / "comparison.json"
        compare_runs(self.dataset, direct_run, framework_run, report)
        result = read_json(report)
        self.assertEqual(result["direct_jev"]["overall"]["n"], 12)
        self.assertEqual(result["framework"]["overall"]["n"], 12)
        self.assertEqual(result["both_scored"]["sample_ids"], [sample_id])
        reward = data.load_dataset(self.dataset)[2][sample_id]["reward"]
        self.assertEqual(result["both_scored"]["direct_minus_framework_mae"], abs(5 - reward) - abs(4 - reward))

    def test_real_subprocess_instruments_calls_and_keeps_secrets_out(self):
        record = self.sample()
        self.assertEqual(record["status"], "scored")
        self.assertEqual(record["score"], 4)
        self.assertEqual(len(record["calls"]), 2)
        self.assertEqual(record["calls"][0]["images"], 1)
        raw = json.dumps(record) + (self.root / "one" / "job.json").read_text()
        self.assertNotIn("NEVER_LOG_THIS_KEY", raw)
        self.assertNotIn('"reward"', raw)
        self.assertNotIn("score_", raw)

    def test_failure_stage_and_zero_retry(self):
        record = self.sample({**self.inputs[0], "task": "fail"})
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failed_stage"], "match_windows")
        self.assertEqual(record["attempt"], 1)
        self.assertEqual(len(record["calls"]), 1)

    def test_timeout_kills_worker_and_saves_partial_journal(self):
        record = self.sample({**self.inputs[0], "task": "sleep"}, timeout_seconds=2.)
        self.assertEqual(record["status"], "timeout")
        self.assertEqual(record["failed_stage"], "observe")
        with self.assertRaises(ProcessLookupError):
            os.kill(record["worker_pid"], 0)

    def test_opt_in_call_limit_does_not_send_extra_request(self):
        record = self.sample(max_calls_per_episode=1)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(len(record["calls"]), 2)
        self.assertFalse(record["calls"][1]["transport_attempted"])

    def test_result_contract_rejects_boolean_or_implicit_abstention(self):
        path = self.root / "result.json"
        for result in ({"status": "scored", "score": True, "jev_choice": "level_1"},
                       {"status": "insufficient_evidence", "score": 1, "jev_choice": "insufficient_evidence"},
                       {"status": "scored", "score": 4, "jev_choice": "level_5"}):
            write_json(path, result)
            with self.assertRaises(ValueError):
                runner.parse_result(path)

    def readiness(self, *a, **kw):
        return {"ready_for_live": True, "checks": {
            "runtime": {"details": {"python": "fixture"}},
            "videos": {"details": {r["sample_id"]: {"duration_s": 1.} for r in self.inputs}}}}

    def test_pilot_to_development_resume_no_duplicate_calls_and_report(self):
        out = self.root / "run"
        with patch.object(runner, "preflight", side_effect=self.readiness), redirect_stdout(io.StringIO()):
            pilot = runner.run(self.dataset, out, self.framework, self.model, execute=True)
            self.assertEqual(pilot["new_episodes"], 5)
            first = (out / "episodes" / self.inputs[0]["sample_id"] / "record.json").read_bytes()
            result = runner.run(self.dataset, out, self.framework, self.model, execute=True, phase="development")
            self.assertEqual(result["new_episodes"], 7)
            self.assertEqual(first, (out / "episodes" / self.inputs[0]["sample_id"] / "record.json").read_bytes())
            result = runner.run(self.dataset, out, self.framework, self.model, execute=True, phase="development")
            self.assertEqual(result["new_episodes"], 0)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                runner.run(self.dataset, out, self.framework, self.model, execute=True, interval=1.)
        report = create_report(self.dataset, out, iterations=10)
        self.assertEqual(report["coverage"], 1.)
        self.assertEqual(report["n"], 12)
        metrics_json = read_json(report["metrics"])
        self.assertEqual(metrics_json["resources"]["transport_attempted_calls"], 24)
        self.assertTrue(Path(report["report"]).is_file())
        sealed = subprocess.run([sys.executable, "-B", "-m", "benchmark", "freeze", "--dataset", str(self.dataset),
                                 "--run", str(out)], capture_output=True, text=True)
        self.assertEqual(sealed.returncode, 0, sealed.stderr)
        self.assertTrue((out / "freeze.json").is_file())
        with self.assertRaisesRegex(ValueError, "sealed"):
            create_report(self.dataset, out, out / "frozen-report", iterations=5)

    def test_failure_not_retried_on_resume_and_unrun_in_denominator(self):
        out = self.root / "run"
        with patch.object(runner, "preflight", side_effect=self.readiness), redirect_stdout(io.StringIO()):
            runner.run(self.dataset, out, self.framework, self.model, execute=True, max_calls=1, max_new_episodes=1)
            result = runner.run(self.dataset, out, self.framework, self.model, execute=True, max_calls=1, max_new_episodes=1)
            self.assertEqual(result["new_episodes"], 1)
        first = read_json(out / "episodes" / self.inputs[0]["sample_id"] / "record.json")
        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["attempt"], 1)
        report = create_report(self.dataset, out, phase="pilot", iterations=10)
        result = read_json(report["metrics"])
        self.assertEqual(result["status_counts"], {"failed": 2, "not_run": 3})
        self.assertIsNone(result["overall"]["selective_mae"])

    def test_dry_run_has_no_output_or_api_calls(self):
        out = self.root / "run"
        result = runner.run(self.dataset, out, self.framework, self.model)
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(out.exists())

    def test_worker_rejects_reference_fields(self):
        directory = self.root / "leaky"
        directory.mkdir()
        write_json(directory / "job.json", {"reward": 5, "split": "validation"})
        result = subprocess.run([sys.executable, "-B", str(runner.WORKER), "--framework-root", str(self.framework),
                                 "--job", str(directory / "job.json")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected worker fields", result.stderr)

    def test_interrupted_journal_is_retained_without_retry(self):
        out = self.root / "run"
        with patch.object(runner, "preflight", side_effect=self.readiness), redirect_stdout(io.StringIO()):
            runner.run(self.dataset, out, self.framework, self.model, execute=True, max_new_episodes=1)
            path = out / "episodes" / self.inputs[0]["sample_id"] / "record.json"
            original = read_json(path)
            write_json(path, {**original, "status": "running", "score": None, "worker_pid": None})
            runner.run(self.dataset, out, self.framework, self.model, execute=True, max_new_episodes=1)
        record = read_json(path)
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual(record["attempt"], 1)
        self.assertEqual(len(record["calls"]), 2)

    def test_runtime_change_and_freeze_prevent_continuation(self):
        out = self.root / "run"
        with patch.object(runner, "preflight", side_effect=self.readiness), redirect_stdout(io.StringIO()):
            runner.run(self.dataset, out, self.framework, self.model, execute=True, max_new_episodes=1)
            write_json(out / "runtime.json", {"python": "different-environment"})
            with self.assertRaisesRegex(ValueError, "runtime environment changed"):
                runner.run(self.dataset, out, self.framework, self.model, execute=True)
            write_json(out / "freeze.json", {"sealed": True})
            with self.assertRaisesRegex(ValueError, "frozen"):
                runner.run(self.dataset, out, self.framework, self.model, execute=True)

    def test_report_escapes_error_text_csv_formulas_and_svg_is_valid_xml(self):
        import xml.etree.ElementTree as ET
        out = self.root / "run"
        with patch.object(runner, "preflight", side_effect=self.readiness), redirect_stdout(io.StringIO()):
            runner.run(self.dataset, out, self.framework, self.model, execute=True, max_new_episodes=1)
        record_path = out / "episodes" / self.inputs[0]["sample_id"] / "record.json"
        record = read_json(record_path)
        write_json(record_path, {**record, "status": "failed", "score": None,
                                 "error": "=</td><script>alert(1)</script>"})
        report = create_report(self.dataset, out, phase="pilot", iterations=5)
        svg = Path(report["report"]).with_name("confusion.svg")
        ET.fromstring(svg.read_text())
        raw = Path(report["report"]).read_text()
        self.assertNotIn("<script", raw)
        self.assertIn("&lt;script&gt;", raw)
        self.assertIn("'=</td><script>", Path(report["report"]).with_name("predictions.csv").read_text())
        self.assertIn("validation", raw)


if __name__ == "__main__":
    unittest.main()
