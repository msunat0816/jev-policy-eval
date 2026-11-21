"""Label-free subprocess boundary. Instrument calls without changing model outputs."""
import argparse
import importlib
import importlib.metadata
import json
import os
import signal
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

# Executed by absolute path from a neutral working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import now, read_json, write_json


STAGES = ("observe", "select_windows", "generate_candidates", "match_windows",
          "execute_experts", "package_evidence", "judge_checks", "evaluate_final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--yolo-model", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.framework_root.resolve()))
    ev = importlib.import_module("evaluate_rollout")
    if Path(ev.__file__).resolve() != args.framework_root.resolve() / "evaluate_rollout.py":
        raise ValueError("imported a different framework entry point")
    if args.probe:
        versions = {}
        for package in ("numpy", "opencv-python-headless", "onnxruntime", "onnx"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        # onnx is a test-only dependency; runtime requires ONNX Runtime.
        importlib.import_module("onnxruntime")
        if args.yolo_model:
            ev.ObjectSpatialExpert(args.yolo_model)
        print(json.dumps({"python": sys.version, "versions": versions,
                          "models": {"qwen": ev.QWEN, "jev": ev.JEV},
                          "entry_point": str(Path(ev.__file__).resolve()),
                          "onnx_model_initialized": bool(args.yolo_model)}))
        return
    job = read_json(args.job)
    expected = {"video", "task", "episode", "split", "out", "yolo_model",
                "threshold_config", "sample_interval_seconds", "max_calls", "timeout_seconds"}
    if set(job) != expected or job["split"] != "validation":
        raise ValueError("unexpected worker fields or non-validation split")
    def timeout_handler(signum, frame):
        raise TimeoutError("worker wall-time limit reached")
    signal.signal(signal.SIGALRM, timeout_handler)
    # Also limits an orphaned worker if its parent is killed abruptly.
    signal.setitimer(signal.ITIMER_REAL, job["timeout_seconds"])
    audit = args.job.parent / "audit"
    audit.mkdir(exist_ok=False)
    current_stage, count = "initialization", 0
    original_post = ev.post

    def post(url, key, body, folder, name, **kwargs):
        nonlocal count
        index = count
        count += 1
        assets = body.get("assets", {})
        message_images = sum(part.get("type") == "image_url"
                             for message in body.get("messages", [])
                             for part in message.get("content", []) if isinstance(part, dict))
        request_bytes = len(json.dumps(body, ensure_ascii=False, allow_nan=False).encode())
        info = {"index": index, "stage": current_stage, "trace_name": name,
                "provider": "typesafe" if urlsplit(url).hostname == "api.typesafe.ai" else "qwen",
                "model": body.get("model"), "started_at": now(), "finished_at": None,
                "images": message_images + sum(a.get("type") == "image" for a in assets.values()),
                "request_bytes": request_bytes, "usage": None,
                "transport_attempted": False, "status": "prepared"}
        path = audit / f"call-{index:05d}.json"
        write_json(path, info)
        started = time.monotonic()
        try:
            if job["max_calls"] is not None and index >= job["max_calls"]:
                raise ValueError("configured per-episode API call limit reached; no request sent")
            body_limit = kwargs.get("max_body_bytes")
            if body_limit is not None and request_bytes > body_limit:
                # The original function still writes its native failure trace.
                info["status"] = "body_limit"
            else:
                info["transport_attempted"] = True
                info["status"] = "in_flight"
            write_json(path, info)
            response = original_post(url, key, body, folder, name, **kwargs)
            info.update(status="returned", usage=response.get("usage"))
            return response
        except Exception as exc:
            info.update(status="error", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            info.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
            write_json(path, info)

    ev.post = post
    for stage_name in STAGES:
        original = getattr(ev, stage_name)
        def wrapped(*a, _original=original, _stage=stage_name, **kw):
            nonlocal current_stage
            current_stage = _stage
            path = audit / f"stage-{_stage}.json"
            info = {"stage": _stage, "started_at": now(), "status": "running"}
            write_json(path, info)
            started = time.monotonic()
            try:
                result = _original(*a, **kw)
                info["status"] = "completed"
                return result
            except Exception as exc:
                info.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                info.update(elapsed_seconds=time.monotonic() - started, finished_at=now())
                write_json(path, info)
        setattr(ev, stage_name, wrapped)
    from types import SimpleNamespace
    namespace = SimpleNamespace(video=Path(job["video"]), task=job["task"], episode=job["episode"],
                                split="validation", out=Path(job["out"]),
                                yolo_model=Path(job["yolo_model"]),
                                threshold_config=Path(job["threshold_config"]) if job["threshold_config"] else None)
    try:
        ev.run(namespace, sample_interval_seconds=job["sample_interval_seconds"])
    except Exception as exc:
        write_json(args.job.parent / "worker-failure.json",
                   {"stage": current_stage, "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
