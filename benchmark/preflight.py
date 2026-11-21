"""Readiness checks and upstream regression tests. Never invokes model APIs."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .common import framework_fingerprint, inside, read_json, write_json
from .data import load_dataset

WORKER = Path(__file__).with_name("worker.py").resolve()


def probe_video(path, ffprobe="ffprobe"):
    process = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height,nb_frames,duration:format=duration",
                              "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
    if process.returncode:
        raise ValueError("ffprobe could not read video: " + process.stderr[-500:])
    import json
    import math
    info = json.loads(process.stdout)
    stream = info.get("streams", [{}])[0]
    if not stream.get("width") or not stream.get("height"):
        raise ValueError("video has no readable video stream")
    duration = stream.get("duration", info.get("format", {}).get("duration"))
    duration = float(duration) if duration not in (None, "N/A") else None
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("invalid video duration")
    return {"width": stream["width"], "height": stream["height"], "duration_s": duration,
            "nb_frames": stream.get("nb_frames")}


def preflight(root, *, python=sys.executable, model=None, dataset=None, stage="pilot",
              run_tests=False, out=None):
    result = {"framework": framework_fingerprint(root), "checks": {}, "test_result": None,
              "scope": "validation development only; no model API calls"}
    checks = result["checks"]
    checks["ffprobe"] = {"ok": bool(shutil.which("ffprobe"))}
    checks["credentials"] = {"ok": all(os.getenv(k) for k in ("DASHSCOPE_API_KEY", "TYPESAFE_API_KEY")),
                              "present": {k: bool(os.getenv(k)) for k in ("DASHSCOPE_API_KEY", "TYPESAFE_API_KEY")}}
    valid_model = model is not None and Path(model).is_file() and Path(model).suffix.lower() == ".onnx"
    checks["model_file"] = {"ok": valid_model}
    command = [str(python), "-B", str(WORKER), "--framework-root", str(Path(root).resolve()), "--probe"]
    if valid_model:
        command += ["--yolo-model", str(Path(model).resolve())]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        import json
        checks["runtime"] = {"ok": process.returncode == 0,
                              "details": json.loads(process.stdout) if process.returncode == 0 else process.stderr[-3000:]}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        checks["runtime"] = {"ok": False, "error": str(exc)}
    if dataset:
        try:
            from .common import digest
            plan, inputs, _ = load_dataset(dataset)
            inventory_path = Path(dataset) / "video-inventory.json"
            inventory = read_json(inventory_path) if inventory_path.exists() else {}
            ids = set(plan["pilot_ids"] if stage == "pilot" else plan["development_ids"])
            videos, errors = {}, []
            for row in inputs:
                if row["sample_id"] not in ids:
                    continue
                try:
                    path = inside(dataset, row["video_path"])
                    if row["video_id"] not in inventory or digest(path) != inventory[row["video_id"]]["sha256"]:
                        raise ValueError("video missing or SHA256 differs from download inventory")
                    videos[row["sample_id"]] = probe_video(path)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    errors.append({"sample_id": row["sample_id"], "error": str(exc)})
            checks["videos"] = {"ok": not errors, "details": videos, "errors": errors}
        except (OSError, ValueError, KeyError) as exc:
            checks["videos"] = {"ok": False, "error": str(exc)}
    else:
        checks["videos"] = {"ok": False, "error": "dataset not supplied"}
    if run_tests:
        if out is None:
            raise ValueError("test output directory is required")
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "framework-tests.log"
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        with log_path.open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run([str(python), "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                                         cwd=Path(root).resolve(), stdout=log, stderr=subprocess.STDOUT,
                                         env=environment, timeout=300)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        log = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Ran (\d+) tests?", log)
        skipped = re.search(r"skipped=(\d+)", log)
        count = int(match.group(1)) if match else 0
        skip_count = int(skipped.group(1)) if skipped else 0
        result["test_result"] = {"return_code": code, "count": count, "skipped": skip_count,
                                 "fully_passed": code == 0 and count > 0 and skip_count == 0,
                                 "log": str(log_path.resolve())}
    result["ready_for_live"] = all(c["ok"] for c in checks.values())
    if result["test_result"] is not None:
        result["ready_for_live"] &= result["test_result"]["fully_passed"]
    if out:
        write_json(Path(out) / "preflight.json", result)
    return result
