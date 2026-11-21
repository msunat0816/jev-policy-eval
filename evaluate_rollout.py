"""Run coarse VLM observation, JEV window selection and VLM check/expert matching."""
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
from measurement_adapter import (ADAPTER_VERSION, LAYOUT_MARGIN_PX, adapt_requirement, requirement_states,
                                 requirement_verdict)
from window_selection import (SELECTION_PROTOCOL, WINDOW_THRESHOLD, build_selection_request,
                              summarize_selection)
from vlm_matching import (EXPERT_CATALOG, MATCHING_MODEL, MATCHING_PROMPT, MATCHING_PROTOCOL,
                          matching_context, parse_window_matching, validate_matching_inputs)


CONFIG = {"match_min": 0.82, "peak_margin": 0.08, "min_gray_std": 8,
          "layout_margin_px": LAYOUT_MARGIN_PX}


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


def measure(expert, rows, frame_size):
    """Return quantities only; task verdicts require a separate semantic adapter."""
    a = np.array([r["subject_xyxy"] for r in rows])
    b = np.array([r["reference_xyxy"] for r in rows])
    if expert == "layout":
        x, y, x2, y2 = a[-1]
        u, v, u2, v2 = b[-1]
        gaps = dict(zip(("image_left", "image_right", "image_above", "image_below"),
                        map(int, (u-x2, x-u2, v-y2, y-v2))))
        return {"end_box_gaps_px": gaps}
    if expert != "motion":
        raise ValueError(f"unknown expert: {expert}")
    offsets = ((a[:, :2] + a[:, 2:]) - (b[:, :2] + b[:, 2:])) / 2
    displacement = float(np.linalg.norm(offsets - offsets[0], axis=1).max())
    fraction = displacement / math.hypot(*frame_size)
    return {"max_relative_excursion_px": displacement,
            "fraction_of_image_diagonal": fraction}


QWEN = "qwen3.8-flash"
JEV = "jev-1.14.0"
DEFAULT_SAMPLE_INTERVAL_SECONDS = 2.0
# Legacy collection helpers below retain their existing measurements. They are not
# called by the current entry point, which stops after matching, before experts.
MEASUREMENTS = {
    "layout": "Signed image-plane box separation at the window's final frame.",
    "motion": "Maximum 2D relative-center excursion from the window's first frame, over the window.",
}
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


def post(url, key, body, folder, name, *, image_directory="frames"):
    raw = json.dumps(body, ensure_ascii=False)
    # Persist the exact prompt; JPEG files are stored separately from the text trace.
    body_log = json.loads(raw)
    for msg in body_log.get("messages", []):
        for part in msg["content"]:
            if part.get("type") == "image_url":
                part["image_url"] = f"JPEGs in {image_directory}/, in prompt order"
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


def select_windows(task, obs, windows, folder):
    state, questions = build_selection_request(task, obs, windows)
    answers = ask_jev(state, questions, folder, "02-select-windows") if questions else {}
    selection = summarize_selection(windows, answers)
    save(folder / "02-selected-windows.json", selection)
    return selection


def match_windows(task, obs, selection, times, cap, folder):
    selected = validate_matching_inputs(task, obs, selection, times)
    matched = []
    for window in selected:
        first, last = window["frame_range"]
        frame_ids = list(range(first, last + 1))
        image_directory = f"matching_frames/{window['id']}"
        (folder / image_directory).mkdir(parents=True, exist_ok=False)
        content = [{"type": "text", "text": MATCHING_PROMPT + "\nContext: "
                    + json.dumps(matching_context(task, obs, window), ensure_ascii=False)}]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for index in frame_ids:
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"matching video decode failed at frame {index}")
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise ValueError(f"matching JPEG encoding failed at frame {index}")
            (folder / image_directory / f"{index}.jpg").write_bytes(jpeg.tobytes())
            content.append({"type": "text", "text": f"frame {index}, PTS {times[index]} seconds"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
        name = f"03-match-{window['id']}"
        base = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        response = post(base.rstrip("/") + "/chat/completions", os.environ["DASHSCOPE_API_KEY"],
                        {"model": MATCHING_MODEL, "enable_thinking": False,
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "user", "content": content}]}, folder, name,
                        image_directory=image_directory)
        result = parse_window_matching(response["choices"][0]["message"]["content"], window,
                                       obs["candidate_checks"], frame_ids)
        record = {"window_id": window["id"], "frame_range": [first, last],
                  "frame_ids": frame_ids, "pts_seconds": [times[i] for i in frame_ids],
                  "image_directory": image_directory, "request_file": name + ".json",
                  "checks": result["checks"]}
        save(folder / f"03-matching-{window['id']}.json", record)
        matched.append(record)
    matching = {"protocol": MATCHING_PROTOCOL, "model": MATCHING_MODEL,
                "status": "proposals_only", "task": task, "requirements": obs["requirements"],
                "checklist": obs["candidate_checks"], "expert_catalog": EXPERT_CATALOG, "windows": matched}
    save(folder / "03-window-matching.json", matching)
    return matching


def collect(cap, task, obs, windows, routes, times, size):
    evidence, decisions = [], []
    bindings = {f["frame"]: f["boxes"] for f in obs["frames"]}
    requirements = {req["id"]: (j, req) for j, req in enumerate(obs["requirements"])}
    for i, (first, last) in enumerate(windows):
        terminal = i == len(windows)-1
        pairs = []
        for j, check in enumerate(obs["candidate_checks"]):
            key = f"w{i}c{j}"
            priority = int(routes[key + "_priority"][1:])
            pairs.append({"id": key, "check_id": check["id"],
                          "requirement_ids": check["requirement_ids"], "entity_keys": check["entity_keys"],
                          "related_frames": check["related_frames"], "priority": priority,
                          "over_threshold": priority >= 2,
                          "applicability": {e: routes[key + "_" + e] for e in MEASUREMENTS}})
        priority = max((p["priority"] for p in pairs), default=None)
        decisions.append({"id": f"w{i}", "frame_range": [first, last], "priority": priority,
                          "priority_source": "max_pair_priority" if pairs else "no_candidate_checks",
                          "over_threshold": priority is not None and priority >= 2,
                          "terminal_window": terminal, "pairs": pairs})
        cache = {}
        for pair in pairs:
            if not pair["over_threshold"]:
                continue
            for requirement_id in pair["requirement_ids"]:
                req_index, req = requirements[requirement_id]
                for expert, applicability in pair["applicability"].items():
                    record = {"id": f"{pair['id']}r{req_index}-{expert}",
                              "check_id": pair["check_id"], "pair_id": pair["id"],
                              "requirement": req_index, "requirement_id": requirement_id, "expert": expert,
                              "priority": pair["priority"], "applicability": applicability,
                              "terminal_window": terminal, "frames": [first, last],
                              "pts_seconds": [times[first], times[last]], "status": "unknown",
                              "requirement_status": "unknown", "measurement_status": "unknown",
                              "semantic_adapter": adapt_requirement(task, req, expert, [first, last], len(times)-1),
                              "binding_source": "qwen_sparse_proposal", "binding_context_until": times[-1]}
                    evidence.append(record)
                    if applicability != "useful":
                        record["reason"] = ("expert_not_applicable" if applicability == "not_applicable"
                                            else "expert_applicability_unknown")
                        continue
                    key = (req["subject"], req["reference"])
                    record["measurement_entities"] = {"subject": key[0], "reference": key[1]}
                    record["measurement_target_source"] = "requirement_subject_reference"
                    # Do not invent an object pair or replace a missing start-frame box.
                    if (not all(key) or key[0] == key[1] or len(pair["entity_keys"]) != 2
                            or set(pair["entity_keys"]) != set(key)):
                        record["reason"] = "unsupported_check_entity_binding"
                        continue
                    started = time.monotonic()
                    record["reused_track"] = key in cache
                    if key not in cache:
                        cache[key] = track_pair(cap, first, last, bindings[first], *key)
                    rows, error = cache[key]
                    record["tracks"] = rows
                    if error or len(rows) != last-first+1 or [r["frame"] for r in rows] != list(range(first, last+1)):
                        error = error or "incomplete_or_unordered_track"
                        record["reason"] = error
                    else:
                        record["measurement"] = measure(expert, rows, size)
                        record["measurement_status"] = "available"
                        record["measurement_scope"] = "window_end" if expert == "layout" else "window_from_first_frame"
                        verdict = requirement_verdict(record["semantic_adapter"], record["measurement"], len(rows))
                        record["requirement_status"] = verdict
                        record["status"] = ("measurement_only" if record["semantic_adapter"]["coverage"] != "direct"
                                            else verdict)
                        if verdict == "unknown":
                            record["semantic_unknown_reason"] = (record["semantic_adapter"]["reason"]
                                if record["semantic_adapter"]["coverage"] != "direct" else "measurement_not_decisive")
                        if expert == "layout":
                            record["initial_measurement"] = measure(expert, rows[:1], size)
                            record["initial_measurement_frame"] = first
                        record["measurement_available_at_pts"] = times[last]
                    record["seconds"] = time.monotonic() - started
    return decisions, evidence


def score_saved(path):
    bundle = json.loads(path.read_text())
    # Selection priorities, applicability proposals and coarse descriptions are not scoring evidence.
    records = [{k: v for k, v in e.items() if k not in ("tracks", "priority", "applicability")}
               for e in bundle["evidence"]]
    states = requirement_states(bundle["requirements"], records)
    state = {"task": bundle["task"], "requirements": bundle["requirements"], "evidence": records,
             "requirement_states": states,
             "episode_end_frame": bundle["windows"][-1]["frame_range"][1]}
    prompt = ("Assign the RoboReward end-state grade. Use only documented program results. "
              "Groundings are learned proposals. A 2D gap does not prove release, support, depth, "
              "or physical stability. Earlier achievements may be reversed. A skipped window is "
              "unobserved. Judge whether the selected evidence establishes the end state. Unsupported or missing "
              "requirements remain unknown. measurement_status=available is not requirement satisfaction. "
              "Only requirement_status and direct semantic contracts may establish a requirement. "
              "A local or partial measurement does not establish an episode-wide or terminal requirement. "
              "Abstain if missing evidence can change the grade.")
    answer = ask_jev(state, {"grade": choice(prompt, GRADES)}, path.parent, "04-judge")["grade"]
    usable = [s for s in states if s["status"] in ("supported", "contradicted")]
    reason = None
    if not usable:
        reason = "no_decisive_program_evidence"
    if reason is None and answer == "level_5":
        if any(s["status"] != "supported" for s in states):
            reason = "full_completion_lacks_evidence_for_each_requirement"
    score = None if reason or answer == "insufficient_evidence" else int(answer[-1])
    return {"score": score, "status": "insufficient_evidence" if score is None else "scored",
            "jev_choice": answer, "guard_reason": reason, "evidence_file": path.name,
            "requirement_states": states,
            "supporting_evidence_ids": sorted({i for s in usable for i in s["evidence_ids"]})}


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
             "qwen": QWEN, "jev": JEV,
             "coarse_observation_schema": SCHEMA_VERSION,
             "window_selection": {"protocol": SELECTION_PROTOCOL, "unit": "window",
                                  "choices_per_window": ["priority"],
                                  "context": "full_checklist_and_all_coarse_observations",
                                  "threshold": WINDOW_THRESHOLD},
             "vlm_matching": {"protocol": MATCHING_PROTOCOL, "model": MATCHING_MODEL,
                              "frames": "all_original_frames_inclusive",
                              "calls": "one_per_selected_window", "expert_execution": False},
             "execution_scope": "coarse_observation_window_selection_and_matching",
             "protocol": SCHEMA_VERSION + "; " + SELECTION_PROTOCOL + "; " + MATCHING_PROTOCOL})
        obs = observe(args.task, indices, times, cap, folder)
        selection = select_windows(args.task, obs, windows, folder)
        matching = match_windows(args.task, obs, selection, times, cap, folder)
        # Expert execution and new JEV check judgments/scoring await authorization.
        # The new expert proposals must not be fed to the legacy collector.
        result = {"status": "awaiting_expert_execution", "score": None,
                  "selection_file": "02-selected-windows.json",
                  "matching_file": "03-window-matching.json",
                  "matched_window_count": len(matching["windows"]),
                  "selected_windows": selection["selected_windows"]}
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
