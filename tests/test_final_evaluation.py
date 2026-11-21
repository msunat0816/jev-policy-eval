"""Text-only final grading with actual serialized HTTP and mocked JEV responses."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import test_check_judgment as judgment_tests
import test_evidence_packaging as packaging_tests
import test_expert_execution as execution_tests
from test_expert_execution import saved_matching
from test_vlm_matching import candidate_bundle, matching_observation, window_matching
from test_vlm_matching_integration import ev
from window_selection import summarize_selection

if ev is not None:
    import final_evaluation as fe


def final_response(choice="level_4"):
    return {"model": "jev-1.14.0", "answers": {"grade": {
        "type": "choice", "choice": choice, "confidence": .01,
        "probabilities": {key: 1/6 for key in (*[f"level_{i}" for i in range(1, 6)], "insufficient_evidence")}}},
        "usage": {"input_tokens": 100, "output_tokens": 10}}


@unittest.skipIf(ev is None, "OpenCV and NumPy required for upstream boundary fixtures")
class FinalEvaluationTests(unittest.TestCase):
    def fixture(self, folder, *, answer="yes", empty_masks=False, empty_selection=False):
        helper = judgment_tests.CheckJudgmentTests()
        if empty_selection:
            obs = matching_observation()
            selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p0", "w1": "p1"})
            times = [0, .09, .25, .4, .6]
            inputs = obs, selection, times, candidate_bundle(selection, times), saved_matching("Task", obs, selection, times)
            execution = execution_tests.ExpertExecutionTests().run_execution(inputs)
            packages = packaging_tests.EvidencePackagingTests().build(inputs, execution)
        else:
            inputs, execution, packages = helper.fixture(folder, empty_masks=empty_masks)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                patch.object(ev, "post", return_value=judgment_tests.jev_response(answer=answer)):
            judgments = helper.judge(folder, inputs, execution, packages)
        return inputs, execution, packages, judgments

    def evaluate(self, folder, inputs, execution, packages, judgments):
        obs, selection, times, candidates, matching = inputs
        return ev.evaluate_final("Task", obs, selection, times, folder, candidates, matching, execution, packages, judgments)

    def test_actual_http_is_one_text_choice_with_complete_program_materials(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, execution, packages, judgments = self.fixture(folder)
            before = copy.deepcopy((inputs, execution, packages, judgments))
            response = final_response()
            response["answers"]["grade"].update(explanation="Invented model explanation", supporting_evidence_ids=["all"])
            sent = []
            def http(request, timeout):
                sent.append(request)
                return io.BytesIO(json.dumps(response).encode())
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                    patch.object(ev, "urlopen", side_effect=http) as urlopen:
                result = self.evaluate(folder, inputs, execution, packages, judgments)
            urlopen.assert_called_once()
            request, = sent
            self.assertEqual(request.full_url, ev.JEV_URL)
            self.assertNotIn("typesafe-danger-zone-images", {k.lower() for k, v in request.header_items()})
            body = json.loads(request.data)
            self.assertEqual(set(body), {"model", "state", "questions"})
            self.assertIsInstance(body["state"], str)
            materials = json.loads(body["state"])
            self.assertEqual(materials["instruction"], "Task")
            self.assertEqual(materials["requirements"], inputs[0]["requirements"])
            self.assertEqual(materials["rubric"], fe.RUBRIC)
            self.assertEqual(materials["episode"], packages["episode"])
            self.assertEqual([c["check"] for c in materials["checks"]], inputs[0]["candidate_checks"])
            self.assertEqual([c["answer"] for c in materials["checks"]], ["yes", "unknown"])
            self.assertEqual(materials["checks"][0]["evidence_summary"]["measurements"], packages["packages"][0]["measurements"])
            self.assertEqual(materials["checks"][0]["evidence_summary"]["image_coverage"]["missing_ranges"],
                             [{"frame_range": [3, 4], "pts_range_seconds": [.4, .6], "frame_count": 2}])
            self.assertEqual(set(body["questions"]), {"grade"})
            self.assertEqual(body["questions"]["grade"]["criteria"], fe.CRITERIA)
            self.assertEqual(json.loads((folder / fe.MATERIALS_FILE).read_text()), materials)
            self.assertEqual(json.loads((folder / fe.EVALUATION_FILE).read_text()), result)
            trace = json.loads((folder / fe.REQUEST_FILE).read_text())
            self.assertEqual(trace["request"], body)
            self.assertEqual(trace["response"], response)
            self.assertNotIn("test-only", json.dumps(trace))
            self.assertNotIn("headers", trace)
            self.assertEqual(result["score"], 4)
            self.assertNotIn("explanation", result)
            self.assertNotIn("supporting_evidence_ids", result)
            self.assertNotIn("Invented model explanation", json.dumps(materials))
            self.assertEqual((inputs, execution, packages, judgments), before)

    def test_unknowns_do_not_force_abstention_and_yes_ratio_does_not_compute_score(self):
        for answer, choice in (("unknown", "level_3"), ("yes", "insufficient_evidence"), ("no", "level_2")):
            with self.subTest(answer=answer, choice=choice), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                fixture = self.fixture(folder, answer=answer)
                with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                        patch.object(ev, "post", return_value=final_response(choice)) as post:
                    result = self.evaluate(folder, *fixture)
                post.assert_called_once()
                self.assertEqual(result["jev_choice"], choice)
                self.assertEqual(result["score"], fe.SCORES.get(choice))

    def test_empty_selection_still_calls_final_once_with_all_program_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture = self.fixture(folder, empty_selection=True)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                    patch.object(ev, "post", return_value=final_response("insufficient_evidence")) as post:
                result = self.evaluate(folder, *fixture)
            post.assert_called_once()
            materials = json.loads(post.call_args.args[2]["state"])
            self.assertTrue(all(c["answer"] == "unknown" and c["answer_source"] == "program_no_evidence" for c in materials["checks"]))
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertIsNone(result["score"])
            self.assertEqual(result["jev_call_count"], 1)
            self.assertFalse((folder / "check_frames").exists())

    def test_missing_measurements_and_unknown_reason_remain_records_not_invented_rationale(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, execution, packages, judgments = self.fixture(folder, answer="unknown", empty_masks=True)
            materials = fe.build_final_materials("Task", inputs[0], packages, judgments)
            unresolved = materials["checks"][0]["unresolved"]
            self.assertEqual(unresolved[0], {"kind": "unknown_check", "answer_source": "jev", "recorded_reason": None})
            gaps = [i for i in unresolved if i["kind"] == "measurement_gap"]
            self.assertEqual(gaps[-1]["execution_status"], "not_executed")
            self.assertEqual(gaps[-1]["missing"][0]["reason"], "empty_mask")
            self.assertEqual(materials["checks"][1]["unresolved"][0]["recorded_reason"], "no_images_or_usable_measurements")

    def test_all_related_windows_keep_nonuniform_pts_and_independent_summaries(self):
        obs = matching_observation()
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p3"})
        times = [0, .09, .25, .4, .6]
        values = [window_matching("w0", 0, 2), window_matching("w1", 2, 4)]
        values[1]["checks"][0].update(relevance="uncertain", limitations=["Identity uncertain."])
        inputs = obs, selection, times, candidate_bundle(selection, times), saved_matching("Task", obs, selection, times, values)
        execution = execution_tests.ExpertExecutionTests().run_execution(inputs)
        packages = packaging_tests.EvidencePackagingTests().build(inputs, execution)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            helper = judgment_tests.CheckJudgmentTests()
            helper.write_images(folder, inputs)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                    patch.object(ev, "post", return_value=judgment_tests.jev_response()):
                judgments = helper.judge(folder, inputs, execution, packages)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                    patch.object(ev, "post", return_value=final_response()) as post:
                self.evaluate(folder, inputs, execution, packages, judgments)
            summary = json.loads(post.call_args.args[2]["state"])["checks"][0]["evidence_summary"]
            self.assertEqual([w["pts_range_seconds"] for w in summary["windows"]], [[0, .25], [.25, .6]])
            self.assertEqual(summary["measurements"], packages["packages"][0]["measurements"])
            self.assertEqual(summary["windows"][1]["relevance"], "uncertain")

    def test_corrupt_or_incomplete_judgments_fail_before_final_request(self):
        for mode in ("missing", "duplicate", "reordered", "context", "protocol", "status", "answer", "source", "no_evidence_source", "reference", "provided", "boolean", "calls"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                fixture = self.fixture(folder)
                judgments = fixture[-1]
                record = judgments["judgments"][0]
                if mode == "missing": judgments["judgments"].pop()
                elif mode == "duplicate": judgments["judgments"][1] = copy.deepcopy(record)
                elif mode == "reordered": judgments["judgments"].reverse()
                elif mode == "context": judgments["instruction"] = "Other task"
                elif mode == "protocol": judgments["protocol"] = "legacy"
                elif mode == "status": judgments["status"] = "partial"
                elif mode == "answer": record["answer"] = "probably"
                elif mode == "source": record["actual_model"] = "other-model"
                elif mode == "no_evidence_source": judgments["judgments"][1]["answer_source"] = "jev"
                elif mode == "reference": record["request_file"] = "06-judge-c0001.json"
                elif mode == "provided": record["provided_evidence"]["measurement_ids"] = []
                elif mode == "boolean": record["package_index"] = False
                else: judgments["jev_call_count"] = True
                with patch.object(ev, "post") as post:
                    with self.assertRaises(ValueError): self.evaluate(folder, *fixture)
                post.assert_not_called()
                self.assertTrue((folder / "07-failed.json").exists())
                self.assertFalse((folder / fe.MATERIALS_FILE).exists())
                self.assertFalse((folder / fe.EVALUATION_FILE).exists())

    def test_corrupt_program_summary_fails_before_call_without_rerunning_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture = self.fixture(folder)
            fixture[2]["packages"][0]["image_coverage"]["includes_episode_end"] = True
            with patch.object(ev, "post") as post, patch.object(ev, "ObjectSpatialExpert") as detector, \
                    patch("expert_execution.ShortWindowMotionExpert") as motion:
                with self.assertRaisesRegex(ValueError, "upstream records"): self.evaluate(folder, *fixture)
            for method in (post, detector, motion): method.assert_not_called()

    def test_no_images_raw_arrays_labels_or_expert_calls_enter_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture = self.fixture(folder)
            fixture[-1].update(reward=5, binary_success=True, feedback="secret label")
            # Removing image files cannot affect an already completed text-only stage.
            with patch.object(Path, "read_bytes", side_effect=AssertionError("no image reads")), \
                    patch.object(ev.cv2, "imread", side_effect=AssertionError("no image reads")), \
                    patch.object(ev, "ObjectSpatialExpert") as detector, patch("expert_execution.ShortWindowMotionExpert") as motion, \
                    patch("expert_execution.pairwise_spatial_evidence") as geometry, \
                    patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                    patch.object(ev, "post", return_value=final_response()) as post:
                self.evaluate(folder, *fixture)
            for method in (detector, motion, geometry): method.assert_not_called()
            state = json.loads(post.call_args.args[2]["state"])
            def inspect(value):
                if isinstance(value, dict):
                    self.assertTrue(set(value).isdisjoint({"assets", "images", "samples", "mask", "counts", "frame_ids",
                                                         "supporting_evidence_ids", "explanation", "reward", "binary_success", "feedback"}))
                    for child in value.values(): inspect(child)
                elif isinstance(value, list):
                    for child in value: inspect(child)
            inspect(state)

    def test_each_choice_maps_directly_and_metadata_does_not_override_choice(self):
        for choice in fe.CRITERIA:
            response = final_response(choice)
            response["answers"]["grade"]["probabilities"] = {k: int(k == "insufficient_evidence") for k in fe.CRITERIA}
            result = fe.parse_final_response(response)
            self.assertEqual(result["jev_choice"], choice)
            self.assertEqual(result["score"], fe.SCORES.get(choice))
            self.assertEqual(result["status"], "scored" if choice in fe.SCORES else "insufficient_evidence")
        response = final_response()
        del response["usage"]
        del response["answers"]["grade"]["confidence"]
        del response["answers"]["grade"]["probabilities"]
        result = fe.parse_final_response(response)
        self.assertTrue(all(result[k] is None for k in ("usage", "confidence", "probabilities")))

    def test_invalid_response_shapes_and_metadata_are_rejected(self):
        for mode in ("model", "missing", "extra", "type", "choice", "number", "confidence", "probabilities", "boolean"):
            with self.subTest(mode=mode):
                value = final_response()
                answer = value["answers"]["grade"]
                if mode == "model": value["model"] = "other-model"
                elif mode == "missing": value["answers"].clear()
                elif mode == "extra": value["answers"]["reason"] = answer
                elif mode == "type": answer["type"] = "score"
                elif mode == "choice": answer["choice"] = "level_6"
                elif mode == "number": answer["choice"] = 5
                elif mode == "confidence": answer["confidence"] = float("nan")
                elif mode == "probabilities": answer["probabilities"].pop("level_1")
                else: answer["probabilities"]["level_1"] = True
                with self.assertRaises(ValueError): fe.parse_final_response(value)

    def test_invalid_response_keeps_trace_and_materials_without_completed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture = self.fixture(folder)
            invalid = final_response("yes")
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                    patch.object(ev, "urlopen", return_value=io.BytesIO(json.dumps(invalid).encode())) as http:
                with self.assertRaisesRegex(ValueError, "rubric level"): self.evaluate(folder, *fixture)
            http.assert_called_once()
            self.assertEqual(json.loads((folder / fe.REQUEST_FILE).read_text())["response"], invalid)
            self.assertTrue((folder / fe.MATERIALS_FILE).exists())
            self.assertTrue((folder / "07-failed.json").exists())
            self.assertTrue((folder / fe.JUDGMENT_FILE).exists())
            self.assertFalse((folder / fe.EVALUATION_FILE).exists())
            self.assertFalse((folder / "result.json").exists())

    def test_http_error_is_retained_without_retry_or_fallback_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture = self.fixture(folder)
            error_body = {"detail": "max_tokens_exceeded"}
            error = HTTPError(ev.JEV_URL, 400, "Bad Request", None, io.BytesIO(json.dumps(error_body).encode()))
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "urlopen", side_effect=error) as http:
                with self.assertRaises(HTTPError): self.evaluate(folder, *fixture)
            http.assert_called_once()
            trace = json.loads((folder / fe.REQUEST_FILE).read_text())
            self.assertEqual(trace["response_error"], error_body)
            self.assertEqual(trace["attempt"], 1)
            self.assertNotIn("score", json.loads((folder / "07-failed.json").read_text()))
            self.assertFalse((folder / fe.EVALUATION_FILE).exists())

    def test_main_keeps_completed_checks_and_releases_video_on_final_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, execution, packages, judgments = self.fixture(root / "fixture-data")
            obs, selection, times, candidates, matching = inputs
            model, video = root / "model.onnx", root / "video.mp4"
            model.write_bytes(b"mocked")
            video.write_bytes(b"mocked")
            args = SimpleNamespace(video=video, yolo_model=model, out=root / "run", task="Task", split="test", episode="synthetic")
            cap = Mock()
            cap.get.side_effect = [16, 12]
            def judge(*arguments):
                ev.save(arguments[4] / fe.JUDGMENT_FILE, judgments)
                return judgments
            with patch.object(ev, "frame_times", return_value=times), patch.object(ev.cv2, "VideoCapture", return_value=cap), \
                    patch.object(ev, "observe", return_value=obs), patch.object(ev, "select_windows", return_value=selection), \
                    patch.object(ev, "generate_candidates", return_value=candidates), patch.object(ev, "match_windows", return_value=matching), \
                    patch.object(ev, "execute_experts", return_value=execution), patch.object(ev, "package_evidence", return_value=packages), \
                    patch.object(ev, "judge_checks", side_effect=judge), patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                    patch.object(ev, "urlopen", side_effect=OSError("offline final error")) as http, patch.object(ev, "score_saved") as legacy:
                with self.assertRaisesRegex(OSError, "offline final error"): ev.run(args)
            http.assert_called_once()
            legacy.assert_not_called()
            cap.release.assert_called_once()
            self.assertTrue((args.out / fe.JUDGMENT_FILE).exists())
            self.assertTrue((args.out / "07-failed.json").exists())
            self.assertFalse((args.out / "result.json").exists())

    def test_prompt_preserves_approved_rubric_unknown_and_attribution_rules(self):
        for phrase in ("end-of-episode", "disregard of", "time limits", "unknown as no", "does not by itself require abstention",
                       "Do not count or average yes", "Earlier success", "not a model-authored rationale", "structured Choice"):
            self.assertIn(phrase, fe.FINAL_PROMPT)
        self.assertEqual(set(fe.RUBRIC["grades"]), {f"level_{i}" for i in range(1, 6)})
        self.assertIn("at least two", fe.CRITERIA["level_3"])
        self.assertIn("exactly one minor", fe.CRITERIA["level_4"])


if __name__ == "__main__":
    unittest.main()
