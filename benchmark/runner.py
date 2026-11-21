"""Serial, zero-retry execution with immutable runs and per-sample journals."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from .common import (digest, framework_fingerprint, inside, now, object_hash,
                     positive, read_json, write_json)
from .data import load_dataset
from .preflight import WORKER, preflight


@contextmanager
def run_lock(out):
    path = Path(out) / ".runner.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another runner is using this output directory") from None
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=5)
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def stop_orphan(record, directory):
    """Terminate only a surviving worker whose exact job path still matches."""
    pid = record.get("worker_pid")
    if type(pid) is not int:
        return
    process = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=5)
    command = process.stdout
    if (process.returncode == 0 and str(WORKER) in command
            and str(directory / "job.json") in command and "--job" in command):
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def parse_result(path):
    result = read_json(path)
    status, score = result.get("status"), result.get("score")
    if status == "scored":
        if type(score) is not int or score not in range(1, 6) or result.get("jev_choice") != f"level_{score}":
            raise ValueError("invalid scored result")
    elif status == "insufficient_evidence":
        if score is not None or result.get("jev_choice") != "insufficient_evidence":
            raise ValueError("abstention must have a null score")
    else:
        raise ValueError("framework did not produce a completed grading result")
    if "final_evaluation_file" in result:
        final = read_json(inside(Path(path).parent, result["final_evaluation_file"]))
        if any(final.get(k) != result[k] for k in ("status", "score", "jev_choice")):
            raise ValueError("final evaluation and result disagree")
    return result


def collect_artifacts(directory):
    audit = directory / "audit"
    calls = [read_json(path) for path in sorted(audit.glob("call-*.json"))]
    stages = [read_json(path) for path in sorted(audit.glob("stage-*.json"))]
    disk = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    failed = [r for r in stages if r["status"] in ("failed", "running")]
    return {"calls": calls, "stages": stages, "disk_bytes": disk,
            "failed_stage": failed[-1]["stage"] if failed else "initialization"}


def execute_sample(row, dataset, directory, configuration, video_info=None):
    directory.mkdir(parents=True, exist_ok=False)
    record_path = directory / "record.json"
    video = inside(dataset, row["video_path"])
    job = {"video": str(video), "task": row["task"], "episode": row["sample_id"],
           "split": "validation", "out": str(directory / "rollout"),
           "yolo_model": configuration["yolo_model"]["path"],
           "threshold_config": configuration["threshold_path"],
           "sample_interval_seconds": configuration["sample_interval_seconds"],
           "max_calls": configuration["max_calls_per_episode"],
           "timeout_seconds": configuration["timeout_seconds"]}
    write_json(directory / "job.json", job)
    record = {"sample_id": row["sample_id"], "configuration_sha256": object_hash(configuration),
              "status": "running", "score": None, "started_at": now(),
              "video_sha256": None, "video_info": video_info,
              "attempt": 1, "automatic_retries": 0}
    # Persist before launching: a parent crash cannot cause an automatic retry.
    write_json(record_path, record)
    started = time.monotonic()
    process = None
    interrupted = False
    try:
        record["video_sha256"] = digest(video)
        write_json(record_path, record)
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen([configuration["python"], "-B", str(WORKER),
                                        "--framework-root", configuration["framework"]["root"],
                                        "--job", str(directory / "job.json")],
                                       cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                                       start_new_session=True)
            record["worker_pid"] = process.pid
            write_json(record_path, record)
            try:
                code = process.wait(timeout=configuration["timeout_seconds"])
            except subprocess.TimeoutExpired:
                stop_process(process)
                record.update(status="timeout", error_type="EpisodeTimeout",
                              error="episode wall-time limit exceeded; no automatic retry")
            else:
                record["worker_returncode"] = code
                if code:
                    failure_path = directory / "worker-failure.json"
                    details = read_json(failure_path) if failure_path.exists() else {}
                    record.update(status="failed", error_type=details.get("error_type", "WorkerExit"),
                                  error=details.get("error", "worker exited without a result"),
                                  failed_stage=details.get("stage", "initialization"))
                else:
                    result = parse_result(directory / "rollout" / "result.json")
                    record.update(status=result["status"], score=result["score"],
                                  jev_choice=result["jev_choice"], framework_result="rollout/result.json")
    except KeyboardInterrupt:
        if process:
            stop_process(process)
        record.update(status="interrupted", error_type="KeyboardInterrupt", error="stopped by user")
        interrupted = True
    except Exception as exc:
        if process:
            stop_process(process)
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        record.update(elapsed_seconds=time.monotonic() - started, finished_at=now())
        try:
            artifacts = collect_artifacts(directory)
            if "failed_stage" in record:
                artifacts.pop("failed_stage", None)
            record.update(artifacts)
        except (OSError, ValueError, KeyError) as exc:
            record["artifact_error"] = str(exc)
        write_json(record_path, record)
    if interrupted:
        raise KeyboardInterrupt
    return record


def run(dataset, out, root, model, *, phase="pilot", python=sys.executable,
        execute=False, timeout=1800, interval=2., threshold_config=None,
        max_calls=None, max_new_episodes=None, batch_seconds=None):
    positive(timeout, "timeout")
    positive(interval, "sample interval")
    for number, name in ((max_calls, "max calls"), (max_new_episodes, "max new episodes")):
        if number is not None and (type(number) is not int or number <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if batch_seconds is not None:
        positive(batch_seconds, "batch seconds")
    dataset, out, root, model = map(lambda p: Path(p).resolve(), (dataset, out, root, model))
    plan, inputs, _ = load_dataset(dataset)
    ids = plan["pilot_ids"] if phase == "pilot" else plan["development_ids"]
    rows = [r for r in inputs if r["sample_id"] in set(ids)]
    if not execute:
        return {"dry_run": True, "phase": phase, "samples": len(rows), "sample_ids": ids,
                "out": str(out), "model_calls": 0,
                "note": "add --execute to run; no weights or videos are downloaded by run"}
    if not model.is_file() or model.suffix.lower() != ".onnx":
        raise ValueError("provide a local segmentation ONNX model")
    fingerprint = framework_fingerprint(root)
    harness_hashes = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    endpoint_override = os.getenv("DASHSCOPE_BASE_URL")
    configuration = {"protocol": "jev-validation-run-v1", "dataset": str(dataset),
                     "plan_sha256": digest(dataset / "plan.json"), "framework": fingerprint,
                     "harness_sha256": object_hash(harness_hashes),
                     # Preserve virtualenv symlinks; resolve() would bypass the venv.
                     "python": str(Path(shutil.which(str(python)) or python).absolute()),
                     "yolo_model": {"path": str(model), "sha256": digest(model)},
                     "timeout_seconds": timeout, "sample_interval_seconds": interval,
                     "max_calls_per_episode": max_calls,
                     "dashscope_endpoint_override": {
                         "sha256": object_hash(endpoint_override) if endpoint_override is not None else None,
                         "hostname": urlsplit(endpoint_override).hostname if endpoint_override else None},
                     "threshold_sha256": digest(threshold_config) if threshold_config else None,
                     "threshold_path": str(out / "frozen-threshold.json") if threshold_config else None,
                     "retries": 0}
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        if (out / "freeze.json").exists():
            raise ValueError("run is frozen; use a new output directory")
        config_path = out / "configuration.json"
        if config_path.exists() and read_json(config_path) != configuration:
            raise ValueError("configuration changed; use a new output directory to preserve the old run")
        if not config_path.exists():
            write_json(config_path, configuration)
            if threshold_config:
                write_json(out / "frozen-threshold.json", read_json(threshold_config))
        elif threshold_config and digest(out / "frozen-threshold.json") != digest(threshold_config):
            # Compare parsed JSON too, because the frozen copy is formatted.
            if read_json(out / "frozen-threshold.json") != read_json(threshold_config):
                raise ValueError("frozen threshold configuration changed")
        check = preflight(root, python=configuration["python"], model=model,
                          dataset=dataset, stage=phase, out=out / f"preflight-{phase}")
        if not check["ready_for_live"]:
            raise ValueError(f"preflight failed; inspect {out / ('preflight-' + phase) / 'preflight.json'}")
        runtime_path = out / "runtime.json"
        runtime = check["checks"]["runtime"]["details"]
        if runtime_path.exists() and read_json(runtime_path) != runtime:
            raise ValueError("runtime environment changed; use a new output directory")
        if not runtime_path.exists():
            write_json(runtime_path, runtime)
        started, new_count = time.monotonic(), 0
        for row in rows:
            if max_new_episodes is not None and new_count >= max_new_episodes:
                break
            if batch_seconds is not None and time.monotonic() - started >= batch_seconds:
                break
            directory = out / "episodes" / row["sample_id"]
            record_path = directory / "record.json"
            if directory.exists():
                if not record_path.exists():
                    raise ValueError(f"incomplete episode directory without journal: {directory}; preserve it and use a new run")
                record = read_json(record_path)
                if record["configuration_sha256"] != object_hash(configuration):
                    raise ValueError("episode was created under a different configuration")
                if record["status"] == "running":
                    stop_orphan(record, directory)
                    record.update(status="interrupted", score=None,
                                  error_type="ParentInterrupted", error="previous runner stopped; not retried",
                                  finished_at=now(), **collect_artifacts(directory))
                    write_json(record_path, record)
                continue
            if framework_fingerprint(root)["code_sha256"] != fingerprint["code_sha256"]:
                raise ValueError("framework code changed during the run")
            if digest(model) != configuration["yolo_model"]["sha256"]:
                raise ValueError("ONNX weights changed during the run")
            info = check["checks"]["videos"]["details"][row["sample_id"]]
            record = execute_sample(row, dataset, directory, configuration, info)
            new_count += 1
            print(json.dumps({"sample_id": row["sample_id"], "status": record["status"],
                              "new_episodes": new_count}, ensure_ascii=False), flush=True)
        write_json(out / f"phase-{phase}.json", {"phase": phase, "target_ids": ids,
                                                "new_episodes": new_count, "finished_at": now()})
    return {"phase": phase, "target_samples": len(ids), "new_episodes": new_count, "out": str(out)}
