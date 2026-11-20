"""Evaluate one local RoboReward video using Qwen, text-only JEV and CPU experts."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np

from coarse_observation import OBSERVE, SCHEMA_VERSION, parse_observation


GEOMETRY = {"image_left", "image_right", "image_above", "image_below"}
MOTION = {"relative_move", "relative_still"}
PREDICATES = GEOMETRY | MOTION | {"unsupported"}
CONFIG = {"match_min": 0.82, "peak_margin": 0.08, "min_gray_std": 8,
          "still_max_diagonal": 0.01, "move_min_diagonal": 0.03}


def seed(gray, normalized_box):
    if not isinstance(normalized_box, list) or len(normalized_box) != 4:
        return None
    if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000
               for v in normalized_box):
        return None
    height, width = gray.shape
    x, y, x2, y2 = np.rint(np.array(normalized_box) * [width, height, width, height] / 1000).astype(int)
    if x2 - x < 6 or y2 - y < 6:
        return None
    patch = gray[y:y2, x:x2].copy()
    return patch if patch.std() >= CONFIG["min_gray_std"] else None


def locate(frame, template):
    """Reject weak matches and competing, spatially distinct peaks."""
    h, w = template.shape
    field = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, (x, y) = cv2.minMaxLoc(field)
    if best < CONFIG["match_min"]:
        return None
    field[max(0, y-h//2):y+h//2+1, max(0, x-w//2):x+w//2+1] = -1
    if best - cv2.minMaxLoc(field)[1] < CONFIG["peak_margin"]:
        return None
    return [x, y, x+w, y+h]


def track_pair(cap, first, last, boxes, subject, reference):
    if subject == reference or subject not in boxes or reference not in boxes:
        return [], "missing_distinct_object_or_reference_box"
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    rows = []
    for index in range(first, last + 1):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"video decode failed at frame {index}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if index == first:
            templates = [seed(gray, boxes[name]) for name in (subject, reference)]
            if any(t is None for t in templates):
                return rows, "invalid_or_untextured_box"
        found = [locate(gray, template) for template in templates]
        if any(b is None for b in found):
            return rows, f"ambiguous_or_lost_at_frame_{index}"
        a, b = found
        if a == b:
            return rows, "two_bindings_match_same_region"
        rows.append({"frame": index, "subject_xyxy": a, "reference_xyxy": b})
    return rows, None


def evaluate(expert, predicate, rows, frame_size):
    """A verdict applies to the stated 2D predicate at the declared scope only."""
    a = np.array([r["subject_xyxy"] for r in rows])
    b = np.array([r["reference_xyxy"] for r in rows])
    if expert == "layout":
        x, y, x2, y2 = a[-1]
        u, v, u2, v2 = b[-1]
        gaps = dict(zip(("image_left", "image_right", "image_above", "image_below"),
                        map(int, (u-x2, x-u2, v-y2, y-v2))))
        measurements = {"end_box_gaps_px": gaps}
        if predicate not in GEOMETRY:
            return measurements, "measurement_only"
        gap = gaps[predicate]
        opposite = {"image_left": "image_right", "image_right": "image_left",
                    "image_above": "image_below", "image_below": "image_above"}
        if gap > 2:
            return measurements, "supported"
        if gaps[opposite[predicate]] > 2:
            return measurements, "contradicted"
        return measurements, "unknown"

    offsets = ((a[:, :2] + a[:, 2:]) - (b[:, :2] + b[:, 2:])) / 2
    displacement = float(np.linalg.norm(offsets - offsets[0], axis=1).max())
    fraction = displacement / math.hypot(*frame_size)
    measurements = {"max_relative_excursion_px": displacement,
                    "fraction_of_image_diagonal": fraction}
    if predicate not in MOTION:
        return measurements, "measurement_only"
    if len(rows) < 2:
        return measurements, "unknown"
    if fraction <= CONFIG["still_max_diagonal"]:
        observed = "relative_still"
    elif fraction >= CONFIG["move_min_diagonal"]:
        observed = "relative_move"
    else:
        return measurements, "unknown"
    return measurements, "supported" if observed == predicate else "contradicted"


QWEN = "qwen3.8-flash"
JEV = "jev-1.14.0"
DEFAULT_SAMPLE_INTERVAL_SECONDS = 2.0
PRIORITY = {
    "p0": "No relevant change or open question.",
    "p1": "Relevant but redundant. Checking this interval adds no decisive evidence.",
    "p2": "A specific requirement is unresolved and evidence here could settle it.",
    "p3": "A core ambiguity, contradiction or possible reversal could change the final grade.",
}
ROUTES = {"layout": "Check signed image-plane box separation at interval end.",
          "motion": "Measure 2D movement relative to a visible reference across the interval.",
          "both": "Both measurements bear on the question.", "none": "Neither expert can help."}
GRADES = {
    "level_1": "No goal-relevant end-state change.",
    "level_2": "Small but insufficient goal-relevant end-state change.",
    "level_3": "Good progress but a major requirement or multiple requirements are violated.",
    "level_4": "Correct primary relation and intent; only one minor requirement is missed.",
    "level_5": "All requirements are satisfied at the end of the episode.",
    "insufficient_evidence": "The admissible evidence cannot distinguish the possible grades.",
}


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def post(url, key, body, folder, name):
    raw = json.dumps(body, ensure_ascii=False)
    # Persist the exact prompt; JPEG files are stored separately from the text trace.
    body_log = json.loads(raw)
    for msg in body_log.get("messages", []):
        for part in msg["content"]:
            if part.get("type") == "image_url":
                part["image_url"] = "JPEGs in frames/, in prompt order"
    call = {"url": url, "request": body_log, "attempt": 1, "cache": "no_client_cache"}
    path = folder / (name + ".json")
    start = time.monotonic()
    try:
        req = Request(url, raw.encode(), {"Authorization": "Bearer " + key,
                                         "Content-Type": "application/json"})
        with urlopen(req, timeout=120) as response:
            result = json.load(response)
        call["response"] = result
        call["usage"] = result.get("usage")
        return result
    except Exception as exc:
        call["error"] = str(exc)
        raise
    finally:
        call["seconds"] = time.monotonic() - start
        save(path, call)


def choice(prompt, options):
    return {"type": "choice", "instructions": prompt, "criteria": options}


def ask_jev(state, questions, folder, name):
    result = post("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"],
                  {"model": JEV, "state": json.dumps(state, ensure_ascii=False),
                   "questions": questions}, folder, name)
    answers = {}
    for key, question in questions.items():
        answer = result["answers"][key]["choice"]
        if answer not in question["criteria"]:
            raise ValueError(f"invalid JEV choice for {key}")
        answers[key] = answer
    return answers


def frame_times(video):
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                   "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
                                   "-of", "json", str(video)])
    times = [float(f["best_effort_timestamp_time"]) for f in json.loads(raw)["frames"]]
    if not times or not np.isfinite(times).all() or any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("video has no usable ordered presentation timestamps")
    return times


def overview_indices(times, interval_seconds):
    """Pick the first frame in each occupied time bin and include the final frame."""
    if not math.isfinite(interval_seconds) or interval_seconds < 0.000001:
        raise ValueError("sample interval must be finite and at least one microsecond")
    # ffprobe reports timestamps to microsecond precision. Integer bins avoid drift.
    interval_us = round(interval_seconds * 1_000_000)
    start_us = round(times[0] * 1_000_000)
    selected = [0]
    last_bin = 0
    for index, timestamp in enumerate(times[1:-1], start=1):
        bin_id = (round(timestamp * 1_000_000) - start_us) // interval_us
        if bin_id > last_bin:
            selected.append(index)
            last_bin = bin_id
    if len(times) > 1:
        selected.append(len(times) - 1)
    return selected


def observe(task, indices, times, cap, folder):
    (folder / "frames").mkdir()
    content = [{"type": "text", "text": OBSERVE + "\nTask: " + task}]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"missing frame {idx}")
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("JPEG encoding failed")
        (folder / "frames" / f"{idx}.jpg").write_bytes(jpeg.tobytes())
        content.append({"type": "text", "text": f"frame {idx}, PTS {times[idx]} seconds"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
    base = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    result = post(base.rstrip("/") + "/chat/completions", os.environ["DASHSCOPE_API_KEY"],
                  {"model": QWEN, "enable_thinking": False, "response_format": {"type": "json_object"},
                   "messages": [{"role": "user", "content": content}]}, folder, "01-observe")
    obs = parse_observation(result["choices"][0]["message"]["content"], indices)
    save(folder / "01-observation.json", obs)
    return obs


def route(task, obs, windows, folder):
    questions = {}
    for i, (first, last) in enumerate(windows):
        scope = f"For frames {first} through {last} inclusive"
        questions[f"w{i}"] = choice(scope + ", rate the value of checking this window for the final grade.", PRIORITY)
        for j, req in enumerate(obs["requirements"]):
            questions[f"w{i}r{j}"] = choice(scope + ", choose all applicable measurement types for: " + req["text"], ROUTES)
    return ask_jev({"task": task, "observations": obs}, questions, folder, "02-route")


def collect(cap, obs, windows, routes, times, size):
    evidence, decisions = [], []
    bindings = {f["frame"]: f["boxes"] for f in obs["frames"]}
    for i, (first, last) in enumerate(windows):
        priority = int(routes[f"w{i}"][1:])
        terminal = i == len(windows)-1
        decisions.append({"frame_range": [first, last], "priority": priority,
                          "over_threshold": priority >= 2, "terminal_window": terminal})
        if priority < 2:
            continue
        cache = {}
        for j, req in enumerate(obs["requirements"]):
            selection = routes[f"w{i}r{j}"]
            experts = ["layout", "motion"] if selection == "both" else [selection]
            for expert in experts:
                record = {"id": f"w{i}r{j}-{expert}", "requirement": j, "expert": expert,
                          "terminal_window": terminal, "frames": [first, last],
                          "pts_seconds": [times[first], times[last]], "status": "unknown",
                          "binding_source": "qwen_sparse_proposal", "binding_context_until": times[-1]}
                evidence.append(record)
                if expert == "none":
                    record["reason"] = "no_applicable_program"
                    continue
                key = (req["subject"], req["reference"])
                started = time.monotonic()
                record["reused_track"] = key in cache
                if key not in cache:
                    cache[key] = track_pair(cap, first, last, bindings[first], *key)
                rows, error = cache[key]
                record["tracks"] = rows
                if error:
                    record["reason"] = error
                else:
                    # V2 Qwen describes task semantics; it no longer invents tool predicates.
                    # Until the reviewed semantic adapter exists, retain measurements only.
                    predicate = req.get("predicate", "unsupported")
                    if "predicate" not in req:
                        record["semantic_unknown_reason"] = "requirement_to_measurement_adapter_pending"
                    record["measurement"], record["status"] = evaluate(expert, predicate, rows, size)
                    if expert == "layout":
                        initial, initial_state = evaluate(expert, predicate, rows[:1], size)
                        record["initial_measurement"] = initial
                        record["initial_status"] = initial_state
                    record["measurement_available_at_pts"] = times[last]
                record["seconds"] = time.monotonic() - started
    return decisions, evidence


def score_saved(path):
    bundle = json.loads(path.read_text())
    records = [{k: v for k, v in e.items() if k != "tracks"} for e in bundle["evidence"]]
    state = {"task": bundle["task"], "requirements": bundle["requirements"], "evidence": records,
             "episode_end_frame": bundle["windows"][-1]["frame_range"][1]}
    prompt = ("Assign the RoboReward end-state grade. Use only documented program results. "
              "Groundings are learned proposals. A 2D gap does not prove release, support, depth, "
              "or physical stability. Earlier achievements may be reversed. A skipped window is "
              "unobserved. Judge whether the selected evidence establishes the end state. Unsupported or missing "
              "requirements remain unknown. Abstain if missing evidence can change the grade.")
    answer = ask_jev(state, {"grade": choice(prompt, GRADES)}, path.parent, "04-judge")["grade"]
    usable = [r for r in records if r["status"] in ("supported", "contradicted")]
    reason = None
    if not usable:
        reason = "no_decisive_program_evidence"
    if reason is None and answer == "level_5":
        for index in range(len(bundle["requirements"])):
            checks = [r for r in records if r["requirement"] == index and r["status"] != "measurement_only"]
            if not checks:
                reason = "full_completion_lacks_evidence_for_each_requirement"
                break
            last_checked = max(r["frames"][1] for r in checks)
            latest = {r["status"] for r in checks if r["frames"][1] == last_checked}
            if latest != {"supported"}:
                reason = "full_completion_lacks_evidence_for_each_requirement"
                break
    score = None if reason or answer == "insufficient_evidence" else int(answer[-1])
    return {"score": score, "status": "insufficient_evidence" if score is None else "scored",
            "jev_choice": answer, "guard_reason": reason, "evidence_file": path.name}


def run(args, *, sample_interval_seconds=DEFAULT_SAMPLE_INTERVAL_SECONDS):
    started = time.monotonic()
    times = frame_times(args.video)
    indices = overview_indices(times, sample_interval_seconds)
    windows = list(zip(indices, indices[1:])) or [(0, 0)]
    max_gap = max(times[last] - times[first] for first, last in windows)
    folder = args.out
    folder.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(args.video))
    try:
        size = tuple(int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        with args.video.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        save(folder / "run.json", {"dataset": "RoboReward", "split": args.split, "episode": args.episode,
             "task": args.task, "video": str(args.video.resolve()), "sha256": digest, "image_size": size,
             "frame_pts_seconds": times, "observed_indices": indices, "opencv": cv2.__version__,
             "coarse_sampling": {"method": "fixed_interval_pts_v1", "interval_seconds": sample_interval_seconds,
                                 "n_selected": len(indices), "max_actual_gap_seconds": max_gap},
             "qwen": QWEN, "jev": JEV, "expert_config": CONFIG, "window_threshold": 2,
             "coarse_observation_schema": SCHEMA_VERSION,
             "protocol": "roboreward-end-state-v1; cpu-experts-v1; jev-threshold-all-windows-v1"})
        obs = observe(args.task, indices, times, cap, folder)
        routes = route(args.task, obs, windows, folder)
        decisions, evidence = collect(cap, obs, windows, routes, times, size)
        path = folder / "03-evidence.json"
        save(path, {"task": args.task, "requirements": obs["requirements"],
                    "candidate_checks": obs["candidate_checks"],
                    "windows": decisions, "evidence": evidence})
        result = score_saved(path)
        result["elapsed_seconds"] = time.monotonic() - started
        save(folder / "result.json", result)
        return result
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--out", type=Path, required=True, help="new run directory")
    parser.add_argument("--sample-interval-seconds", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
                        help="seconds between coarse time samples (default: 2; include endpoints)")
    args = parser.parse_args()
    missing = [k for k in ("DASHSCOPE_API_KEY", "TYPESAFE_API_KEY") if not os.getenv(k)]
    if missing:
        parser.error("missing environment variables: " + ", ".join(missing))
    print(json.dumps(run(args, sample_interval_seconds=args.sample_interval_seconds),
                     ensure_ascii=False))
