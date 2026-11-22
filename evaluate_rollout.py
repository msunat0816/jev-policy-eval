"""Run observation, bound experts, per-check judgments and text-only final grading."""
import argparse
import base64
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from candidate_presentation import render_candidates
from coarse_observation import OBSERVE, SCHEMA_VERSION, parse_observation
from check_judgment import (CRITERIA as CHECK_CRITERIA, EVIDENCE_FILE, IMAGE_HEADERS, JEV_URL,
                            JUDGMENT_MODEL, JUDGMENT_PROMPT, JUDGMENT_PROTOCOL, MAX_BODY_BYTES,
                            build_check_request, has_usable_measurements, parse_check_response,
                            validate_judgment_inputs)
from experts import ObjectSpatialExpert, VideoFrame
from expert_execution import EXECUTION_PROTOCOL, execute_proposals
from evidence_packaging import (PACKAGING_PROTOCOL, build_evidence_packages,
                                validate_threshold_config)
from final_evaluation import (CRITERIA as FINAL_CRITERIA, EVALUATION_FILE, FINAL_MODEL, FINAL_PROMPT,
                              FINAL_PROTOCOL, JUDGMENT_FILE, MATERIALS_FILE, REQUEST_FILE,
                              RUBRIC, RUBRIC_PATH, build_final_materials, build_final_request,
                              parse_final_response)
from measurement_adapter import (ADAPTER_VERSION, LAYOUT_MARGIN_PX, adapt_requirement, requirement_states,
                                 requirement_verdict)
from window_selection import (SELECTION_PROTOCOL, WINDOW_THRESHOLD, build_selection_request,
                              summarize_selection)
from vlm_matching import (CANDIDATE_FRAME_POLICY, CANDIDATE_PROTOCOL, EMPTY_CANDIDATE_BINDING_METHOD, EXPERT_CATALOG, MATCHING_MODEL,
                          MATCHING_PROMPT, MATCHING_PROTOCOL, matching_context, parse_window_matching,
                          validate_candidate_inputs, validate_matching_inputs)


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
# called by the current entry point, which uses the new evidence/judgment chain.
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


def post(url, key, body, folder, name, *, image_directory="frames", headers=None, image_assets=None, max_body_bytes=None):
    raw = json.dumps(body, ensure_ascii=False, allow_nan=False)
    # Persist the exact prompt; JPEG files are stored separately from the text trace.
    body_log = json.loads(raw)
    for msg in body_log.get("messages", []):
        for part in msg["content"]:
            if part.get("type") == "image_url":
                part["image_url"] = f"JPEGs in {image_directory}/, in prompt order"
    for reference, asset in body_log.get("assets", {}).items():
        metadata = (image_assets or {}).get(reference, {})
        if "data_base64" in asset:
            asset["data_base64"] = f"Image bytes in {metadata.get('prepared_file', image_directory)}; see image_assets manifest"
    call = {"url": url, "request": body_log, "attempt": 1, "cache": "no_client_cache"}
    if headers:
        call["headers"] = dict(headers)
    if image_assets is not None:
        call["image_assets"] = image_assets
    path = folder / (name + ".json")
    start = time.monotonic()
    try:
        data = raw.encode()
        if max_body_bytes is not None:
            call["request_bytes"] = len(data)
            if len(data) > max_body_bytes:
                raise ValueError("JEV check request exceeds the 32 MiB body limit; no frames dropped and no request sent")
        req = Request(url, data, {"Authorization": "Bearer " + key,
                                 "Content-Type": "application/json", **(headers or {})})
        with urlopen(req, timeout=120) as response:
            result = json.load(response)
        call["response"] = result
        call["usage"] = result.get("usage")
        return result
    except Exception as exc:
        call["error"] = str(exc)
        if isinstance(exc, HTTPError):
            call["status_code"] = exc.code
            try:
                message = exc.read().decode("utf-8", errors="replace")
                try:
                    call["response_error"] = json.loads(message)
                except json.JSONDecodeError:
                    call["response_error"] = message
            except Exception as read_error:
                call["response_error_unavailable"] = str(read_error)
            finally:
                exc.close()
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


def generate_candidates(task, obs, selection, times, cap, folder, model_path):
    """Detect every selected original frame once, without task-entity binding."""
    # Reuse the existing selection/checklist/PTS boundary; this makes no VLM call.
    selected = validate_matching_inputs(task, obs, selection, times)
    windows = [{"window_id": w["id"], "frame_range": list(w["frame_range"]),
                "frame_ids": list(range(w["frame_range"][0], w["frame_range"][1] + 1))}
               for w in selected]
    frame_ids = sorted({index for window in windows for index in window["frame_ids"]})
    expert = ObjectSpatialExpert(model_path) if frame_ids else None
    records, last_result = [], None
    previous = None
    for index in frame_ids:
        # Sequential reads within each selected interval; skip unselected gaps.
        if previous is None or index != previous + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"candidate video decode failed at frame {index}")
        # Process one original-resolution image at a time, retaining only JSON data.
        last_result = expert.infer([VideoFrame(index, times[index], image)])
        records.extend(last_result["frames"])
        previous = index
    result = {"protocol": CANDIDATE_PROTOCOL, "expert_id": "object_spatial",
              "status": "candidates_generated" if frame_ids else "no_selected_windows",
              "frame_policy": CANDIDATE_FRAME_POLICY, "binding_status": "unbound",
              "expert_protocol": last_result["protocol"] if last_result else None,
              "provenance": last_result["provenance"] if last_result else None,
              "limitations": last_result["limitations"] if last_result else [],
              "windows": windows, "frames": records}
    save(folder / "03-yolo-candidates.json", result)
    return result


def match_windows(task, obs, selection, times, cap, folder, candidates):
    """Associate checks/experts and bind supplied candidates, without executing experts."""
    selected = validate_matching_inputs(task, obs, selection, times)
    by_frame = validate_candidate_inputs(candidates, selected, times)
    matched = []
    for window in selected:
        first, last = window["frame_range"]
        frame_ids = list(range(first, last + 1))
        candidate_frames = [by_frame[index] for index in frame_ids]
        image_directory = f"matching_frames/{window['id']}"
        (folder / image_directory).mkdir(parents=True, exist_ok=False)
        content = [{"type": "text", "text": MATCHING_PROMPT + "\nContext: "
                    + json.dumps(matching_context(task, obs, window, candidate_frames), ensure_ascii=False)}]
        images = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for index in frame_ids:
            ok, original = cap.read()
            if not ok:
                raise ValueError(f"matching video decode failed at frame {index}")
            annotated = render_candidates(original, by_frame[index])
            image_record = {"frame_id": index, "pts_seconds": times[index]}
            for kind, image, filename in (("original", original, f"{index}.jpg"),
                                          ("candidates", annotated, f"{index}-candidates.jpg")):
                quality = 88 if kind == "original" else 95
                ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    raise ValueError(f"matching JPEG encoding failed at frame {index}")
                relative = f"{image_directory}/{filename}"
                (folder / relative).write_bytes(jpeg.tobytes())
                image_record[kind + "_file"] = relative
                content.append({"type": "text", "text": f"frame {index}, PTS {times[index]} seconds; {kind}"})
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
            images.append(image_record)
        name = f"03-match-{window['id']}"
        base = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        response = post(base.rstrip("/") + "/chat/completions", os.environ["DASHSCOPE_API_KEY"],
                        {"model": MATCHING_MODEL, "enable_thinking": False,
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "user", "content": content}]}, folder, name,
                        image_directory=image_directory)
        cleanup_audit = []
        result = parse_window_matching(response["choices"][0]["message"]["content"], window,
                                       obs["candidate_checks"], frame_ids, candidate_frames,
                                       cleanup_audit=cleanup_audit)
        cleanup_file = None
        if cleanup_audit:
            cleanup_file = f"03-matching-{window['id']}-binding-cleanup.json"
            save(folder / cleanup_file, {"protocol": "matching-binding-cleanup-v1",
                                        "request_file": name + ".json", "candidate_file": "03-yolo-candidates.json",
                                        "warnings": cleanup_audit,
                                        "normalized_response": result})
            logging.getLogger(__name__).warning(
                "matching %s: removed %d unused bindings, added %d missing bindings for empty frames; audit: %s",
                window["id"], sum(len(event.get("removed_bindings", [])) for event in cleanup_audit),
                sum(len(event.get("added_bindings", [])) for event in cleanup_audit), cleanup_file)
        source = {"method": "matching_vlm_candidate_selection", "model": MATCHING_MODEL,
                  "request_file": name + ".json", "candidate_file": "03-yolo-candidates.json"}
        added_keys = {(b["entity_key"], b["frame_id"]) for event in cleanup_audit
                      for b in event.get("added_bindings", [])}
        program_source = {**source, "method": EMPTY_CANDIDATE_BINDING_METHOD, "model": None}
        bindings = [{**binding, "pts_seconds": times[binding["frame_id"]],
                     "source": dict(program_source if (binding["entity_key"], binding["frame_id"]) in added_keys else source)}
                    for binding in result["bindings"]]
        record = {"window_id": window["id"], "frame_range": [first, last],
                  "frame_ids": frame_ids, "pts_seconds": [times[i] for i in frame_ids],
                  "image_directory": image_directory, "images": images, "request_file": name + ".json",
                  "candidate_file": "03-yolo-candidates.json", "checks": result["checks"], "bindings": bindings}
        if cleanup_file:
            record["binding_cleanup_file"] = cleanup_file
        save(folder / f"03-matching-{window['id']}.json", record)
        matched.append(record)
    matching = {"protocol": MATCHING_PROTOCOL, "model": MATCHING_MODEL,
                "status": "proposals_and_bindings_only", "task": task, "requirements": obs["requirements"],
                "checklist": obs["candidate_checks"], "expert_catalog": EXPERT_CATALOG,
                "candidate_file": "03-yolo-candidates.json", "windows": matched}
    save(folder / "03-window-matching.json", matching)
    return matching


def execute_experts(task, obs, selection, times, cap, folder, candidates, matching):
    """Execute validated proposals using saved candidates; never rerun segmentation."""
    execution = execute_proposals(task, obs, selection, times, cap, candidates, matching)
    save(folder / "04-expert-execution.json", execution)
    return execution


def package_evidence(task, obs, selection, times, folder, candidates, matching, execution, threshold_config=None):
    """Save per-check summaries/references without expert reruns or model calls."""
    packages = build_evidence_packages(task, obs, selection, times, candidates, matching, execution, threshold_config)
    save(folder / "05-check-evidence.json", packages)
    return packages


def judge_checks(task, obs, selection, times, folder, candidates, matching, execution, packages):
    """One image/text Choice per check; empty evidence is explicitly program unknown."""
    validate_judgment_inputs(task, obs, selection, times, candidates, matching, execution, packages)
    judgments = []
    for index, packet in enumerate(packages["packages"]):
        started = time.monotonic()
        name = f"06-judge-c{index:04d}"
        record = {"check_id": packet["check_id"], "check": packet["check"],
                  "evidence_file": EVIDENCE_FILE, "package_index": index,
                  "provided_evidence": {"window_ids": [w["window_id"] for w in packet["windows"]],
                                        "measurement_ids": [m["id"] for m in packet["measurements"]],
                                        "image_coverage": packet["image_coverage"]},
                  "request_file": None, "image_assets": {}}
        try:
            if not packet["images"] and not has_usable_measurements(packet):
                record.update(answer="unknown", answer_source="program_no_evidence",
                              reason="no_images_or_usable_measurements", actual_model=None,
                              confidence=None, probabilities=None, usage=None)
            else:
                body, manifest = build_check_request(packet, packages["episode"], folder, index)
                record.update(request_file=name + ".json", image_assets=manifest)
                response = post(JEV_URL, os.environ["TYPESAFE_API_KEY"], body, folder, name,
                                headers=IMAGE_HEADERS if body.get("assets") else None,
                                image_assets=manifest, max_body_bytes=MAX_BODY_BYTES)
                record.update(parse_check_response(response, packet["check_id"]), answer_source="jev", reason=None)
            record["elapsed_seconds"] = time.monotonic() - started
            save(folder / f"06-check-c{index:04d}.json", record)
            judgments.append(record)
        except Exception as exc:
            save(folder / f"06-failed-c{index:04d}.json",
                 {"check_id": packet["check_id"], "evidence_file": EVIDENCE_FILE, "package_index": index,
                  "request_file": record["request_file"], "error_type": type(exc).__name__, "error": str(exc),
                  "elapsed_seconds": time.monotonic() - started})
            raise
    result = {"protocol": JUDGMENT_PROTOCOL, "model": JUDGMENT_MODEL, "status": "checks_judged",
              "instruction": task, "requirements": obs["requirements"], "checklist": obs["candidate_checks"],
              "evidence_file": EVIDENCE_FILE, "judgments": judgments,
              "jev_call_count": sum(j["answer_source"] == "jev" for j in judgments)}
    save(folder / "06-check-judgments.json", result)
    return result


def evaluate_final(task, obs, selection, times, folder, candidates, matching, execution, packages, judgments):
    """One text-only JEV rubric selection after every check has been judged."""
    started = time.monotonic()
    try:
        validate_judgment_inputs(task, obs, selection, times, candidates, matching, execution, packages)
        materials = build_final_materials(task, obs, packages, judgments)
        save(folder / MATERIALS_FILE, materials)
        response = post(JEV_URL, os.environ["TYPESAFE_API_KEY"], build_final_request(materials),
                        folder, Path(REQUEST_FILE).stem)
        result = {"protocol": FINAL_PROTOCOL, "model": FINAL_MODEL,
                  **parse_final_response(response), "materials_file": MATERIALS_FILE,
                  "evidence_file": EVIDENCE_FILE, "judgment_file": JUDGMENT_FILE,
                  "request_file": REQUEST_FILE, "rubric_id": RUBRIC["id"],
                  "jev_call_count": 1, "elapsed_seconds": time.monotonic() - started}
        save(folder / EVALUATION_FILE, result)
        return result
    except Exception as exc:
        save(folder / "07-failed.json", {"protocol": FINAL_PROTOCOL,
             "materials_file": MATERIALS_FILE if (folder / MATERIALS_FILE).is_file() else None,
             "judgment_file": JUDGMENT_FILE,
             "request_file": REQUEST_FILE if (folder / REQUEST_FILE).is_file() else None,
             "error_type": type(exc).__name__, "error": str(exc),
             "elapsed_seconds": time.monotonic() - started})
        raise


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
    model_path = Path(args.yolo_model)
    if model_path.suffix.lower() != ".onnx" or not model_path.is_file():
        raise ValueError("provide a local .onnx segmentation model; no automatic downloads")
    threshold_path = getattr(args, "threshold_config", None)
    threshold_bytes = Path(threshold_path).read_bytes() if threshold_path is not None else None
    threshold_config = validate_threshold_config(json.loads(threshold_bytes) if threshold_bytes is not None else None)
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
             "candidate_generation": {"protocol": CANDIDATE_PROTOCOL,
                                      "model_path": str(model_path.resolve()),
                                      "frame_policy": CANDIDATE_FRAME_POLICY,
                                      "entity_binding": False},
             "vlm_matching": {"protocol": MATCHING_PROTOCOL, "model": MATCHING_MODEL,
                              "enabled": True, "frames": "all_original_frames_inclusive",
                              "candidate_views": "original_and_annotated_per_frame",
                              "calls": "one_per_selected_window", "downstream_execution": True},
             "expert_execution": {"protocol": EXECUTION_PROTOCOL, "enabled": True,
                                  "segmentation": "reuse_saved_candidates",
                                  "motion_frames": "all_continuous_proposed_original_frames",
                                  "missing_reference": "track_target_keep_relative_missing",
                                  "check_judgments": False, "grading": False},
             "evidence_packaging": {"protocol": PACKAGING_PROTOCOL, "enabled": True,
                                    "window_relevance": ["related", "uncertain"],
                                    "images": "all_related_window_original_frames_deduplicated",
                                    "raw_measurements": "file_references_only",
                                    "threshold_config": threshold_config,
                                    "threshold_source": ({"path": str(Path(threshold_path).resolve()),
                                                          "sha256": hashlib.sha256(threshold_bytes).hexdigest()}
                                                         if threshold_bytes is not None else None),
                                    "check_judgments": False, "grading": False},
             "check_judgment": {"protocol": JUDGMENT_PROTOCOL, "model": JUDGMENT_MODEL, "enabled": True,
                                "images": "all_package_frames_original_and_candidate_views",
                                "image_preparation": "shrink_max_1568_pad_min_280_preserve_full_canvas",
                                "calls": "one_per_check_with_evidence", "empty_evidence": "program_unknown",
                                "choices": list(CHECK_CRITERIA), "retries": 0, "grading": False,
                                "prompt_sha256": hashlib.sha256(JUDGMENT_PROMPT.encode()).hexdigest()},
             "final_evaluation": {"protocol": FINAL_PROTOCOL, "model": FINAL_MODEL, "enabled": True,
                                  "input": "text_only_program_assembled_materials",
                                  "calls": "one_after_all_checks_including_all_unknown", "retries": 0,
                                  "choices": list(FINAL_CRITERIA), "rubric_id": RUBRIC["id"],
                                  "rubric_source": RUBRIC["source"],
                                  "rubric_sha256": hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest(),
                                  "prompt_sha256": hashlib.sha256(FINAL_PROMPT.encode()).hexdigest()},
             "execution_scope": "coarse_observation_selection_candidates_matching_experts_packages_check_judgments_and_final_evaluation",
             "protocol": "; ".join((SCHEMA_VERSION, SELECTION_PROTOCOL, CANDIDATE_PROTOCOL,
                                       MATCHING_PROTOCOL, EXECUTION_PROTOCOL, PACKAGING_PROTOCOL, JUDGMENT_PROTOCOL, FINAL_PROTOCOL))})
        obs = observe(args.task, indices, times, cap, folder)
        selection = select_windows(args.task, obs, windows, folder)
        candidates = generate_candidates(args.task, obs, selection, times, cap, folder, model_path)
        matching = match_windows(args.task, obs, selection, times, cap, folder, candidates)
        execution = execute_experts(args.task, obs, selection, times, cap, folder, candidates, matching)
        packages = package_evidence(args.task, obs, selection, times, folder, candidates, matching, execution, threshold_config)
        judgments = judge_checks(args.task, obs, selection, times, folder, candidates, matching, execution, packages)
        final = evaluate_final(args.task, obs, selection, times, folder, candidates, matching, execution, packages, judgments)
        result = {"status": final["status"], "score": final["score"], "jev_choice": final["jev_choice"],
                  "final_evaluation_file": EVALUATION_FILE, "final_materials_file": MATERIALS_FILE,
                  "jev_final_call_count": final["jev_call_count"],
                  "selection_file": "02-selected-windows.json",
                  "candidate_file": "03-yolo-candidates.json",
                  "matching_file": "03-window-matching.json",
                  "execution_file": "04-expert-execution.json",
                  "evidence_file": "05-check-evidence.json",
                  "check_evidence_count": len(packages["packages"]),
                  "judgment_file": "06-check-judgments.json",
                  "check_judgment_count": len(judgments["judgments"]),
                  "jev_check_call_count": judgments["jev_call_count"],
                  "expert_execution_count": len(execution["records"]),
                  "matched_window_count": len(matching["windows"]),
                  "candidate_frame_count": len(candidates["frames"]),
                  "candidate_count": sum(len(frame["candidates"]) for frame in candidates["frames"]),
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
    parser.add_argument("--yolo-model", type=Path, required=True,
                        help="local end-to-end segmentation ONNX model (e.g. YOLO26n-seg); CPU only")
    parser.add_argument("--threshold-config", type=Path,
                        help="optional explicit numeric comparison JSON; no inferred or default task thresholds")
    parser.add_argument("--sample-interval-seconds", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
                        help="seconds between coarse time samples (default: 2; include endpoints)")
    args = parser.parse_args()
    missing = [k for k in ("DASHSCOPE_API_KEY", "TYPESAFE_API_KEY") if not os.getenv(k)]
    if missing:
        parser.error("missing environment variables: " + ", ".join(missing))
    print(json.dumps(run(args, sample_interval_seconds=args.sample_interval_seconds),
                     ensure_ascii=False))
