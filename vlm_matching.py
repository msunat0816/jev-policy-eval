"""VLM window/check/expert matching contracts; no video, model or expert execution."""
import copy
import json
import math
import re
from pathlib import Path

from window_selection import build_selection_request, summarize_selection


CANDIDATE_PROTOCOL = "yolo-window-candidates-v1"
CANDIDATE_FRAME_POLICY = "all_selected_window_frames_inclusive_deduplicated"
MATCHING_PROTOCOL = "vlm-window-check-expert-bindings-v2"
EMPTY_CANDIDATE_BINDING_METHOD = "program_empty_candidate_inventory"
MATCHING_MODEL = "qwen3.8-flash"
MATCHING_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "qwen_window_matching.txt"
MATCHING_PROMPT = MATCHING_PROMPT_PATH.read_text(encoding="utf-8")

# Candidate masks already exist; geometry and motion are proposals for the next stage.
EXPERT_CATALOG = {
    "object_spatial": {
        "name": "Object and spatial evidence",
        "execution_status": "candidates_precomputed_geometry_not_scheduled",
        "implementation": "YOLO nano instance segmentation (e.g. YOLO26n-seg), ONNX Runtime CPU",
        "evidence": "Use supplied candidate classes, scores, boxes and masks; propose 2D gaps/overlap without rerunning YOLO.",
        "limitations": [
            "Fixed pretrained class vocabulary; apple is supported, robot grippers, pegs and many lab props are not.",
            "Entity binding and actual detection must be checked; do not assume every named object has a mask.",
            "2D overlap does not establish 3D contact, support, grasp, release or physical stability.",
        ],
    },
    "short_window_motion": {
        "name": "Short-window motion evidence",
        "execution_status": "implemented_not_scheduled",
        "implementation": "OpenCV pyramidal Lucas-Kanade calcOpticalFlowPyrLK with forward-backward checks",
        "evidence": "Timestamped 2D displacement/direction, valid point counts and tracking gaps.",
        "prerequisites": [
            "Reliable object masks and enough textured feature points; continuous window frames.",
            "A reliable reference region is additionally needed for relative motion.",
        ],
        "limitations": [
            "Textureless objects or gripper occlusion can cause missing tracks; insufficient points mean missing data.",
            "Camera movement contributes image displacement; do not assume a stationary camera or reference.",
            "Local image motion does not independently establish grasp, release or episode-wide task success.",
        ],
    },
}


def validate_matching_inputs(task, obs, selection, times):
    """Validate the stage boundary before any video reads or paid matching requests."""
    if (not isinstance(times, list) or not times
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or any(a > b for a, b in zip(times, times[1:]))):
        raise ValueError("matching: expected finite chronological presentation timestamps")
    if not isinstance(selection, dict) or set(selection) != {"protocol", "threshold", "windows", "selected_windows"}:
        raise ValueError("matching: invalid selection bundle")
    windows = selection["windows"]
    if not isinstance(windows, list) or any(
            not isinstance(w, dict) or set(w) != {"id", "frame_range", "priority", "selected"}
            or type(w["priority"]) is not int or type(w["selected"]) is not bool for w in windows):
        raise ValueError("matching: invalid window decisions")
    ranges = [w["frame_range"] for w in windows]
    for window in windows:
        _text(window["id"], "matching.window_id")
    canonical = summarize_selection(ranges, {w["id"]: f"p{w['priority']}" for w in windows})
    if selection != canonical:
        raise ValueError("matching: selected windows do not match the selection decisions")
    build_selection_request(task, obs, ranges)  # Reuse the existing checklist/endpoint contract; no model call.
    if any(not 0 <= frame["frame"] < len(times) for frame in obs["frames"]):
        raise ValueError("matching: coarse frame ID has no presentation timestamp")
    return canonical["selected_windows"]


def matching_context(task, obs, window, candidate_frames):
    return {
        "instruction": task,
        "requirements": copy.deepcopy(obs["requirements"]),
        "checklist": copy.deepcopy(obs["candidate_checks"]),
        "window": {"id": window["id"], "frame_range": list(window["frame_range"])},
        "expert_catalog": copy.deepcopy(EXPERT_CATALOG),
        "candidate_frames": [{key: copy.deepcopy(frame[key])
                              for key in ("frame_id", "pts_seconds", "image_size_hw", "status")}
                             | {"candidates": [{key: copy.deepcopy(candidate[key])
                                                for key in ("candidate_id", "frame_id", "pts_seconds", "class_id",
                                                            "class_name", "confidence", "box_xyxy_px", "mask_status")}
                                               for candidate in frame["candidates"]]}
                             for frame in candidate_frames],
    }


def _fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{label}: expected exactly {', '.join(names)}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected nonempty string")


def _strings(value, label, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label}: expected {'nonempty ' if nonempty else ''}array")
    for item in value:
        _text(item, label)
    if len(value) != len(set(value)):
        raise ValueError(f"{label}: duplicate entry")


def validate_candidate_inputs(candidates, selected, times):
    """Validate the saved stage-2 references before any reads or matching requests."""
    if (not isinstance(candidates, dict) or candidates.get("protocol") != CANDIDATE_PROTOCOL
            or candidates.get("expert_id") != "object_spatial"
            or candidates.get("frame_policy") != CANDIDATE_FRAME_POLICY
            or candidates.get("binding_status") != "unbound"):
        raise ValueError("matching: expected unbound stage-2 YOLO candidates")
    windows = [{"window_id": w["id"], "frame_range": list(w["frame_range"]),
                "frame_ids": list(range(w["frame_range"][0], w["frame_range"][1] + 1))}
               for w in selected]
    if candidates.get("windows") != windows:
        raise ValueError("matching: candidate windows must match the selected windows")
    expected = sorted({i for w in windows for i in w["frame_ids"]})
    frames = candidates.get("frames")
    if (not isinstance(frames, list) or any(not isinstance(f, dict) for f in frames)
            or any(type(f.get("frame_id")) is not int for f in frames)
            or [f["frame_id"] for f in frames] != expected):
        raise ValueError("matching: candidate frames must cover selected frames exactly once in order")
    for frame in frames:
        index = frame["frame_id"]
        pts = frame.get("pts_seconds")
        if (type(pts) not in (int, float) or not math.isfinite(pts)
                or not 0 <= index < len(times) or pts != times[index]):
            raise ValueError("matching: candidate frame PTS must match the original video")
        size = frame.get("image_size_hw")
        if (not isinstance(size, list) or len(size) != 2
                or any(type(n) is not int or n < 1 for n in size)
                or frame.get("coordinate_system") != "original_image_pixels_xyxy"):
            raise ValueError("matching: expected original image dimensions and pixel coordinates")
        height, width = size
        detections = frame.get("candidates")
        if (not isinstance(detections, list)
                or frame.get("status") != ("candidates" if detections else "no_detections")):
            raise ValueError("matching: invalid candidate frame status")
        seen = set()
        for candidate in detections:
            if not isinstance(candidate, dict):
                raise ValueError("matching: invalid candidate record")
            cid = candidate.get("candidate_id")
            if (not isinstance(cid, str) or not re.fullmatch(rf"f{index}:d(0|[1-9][0-9]*)", cid)
                    or cid in seen or type(candidate.get("frame_id")) is not int
                    or candidate["frame_id"] != index
                    or type(candidate.get("pts_seconds")) not in (int, float)
                    or candidate["pts_seconds"] != pts):
                raise ValueError("matching: duplicate or inconsistent candidate ID/frame/PTS")
            seen.add(cid)
            _text(candidate.get("class_name"), "candidate.class_name")
            score = candidate.get("confidence")
            if (type(candidate.get("class_id")) is not int or candidate["class_id"] < 0
                    or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("matching: invalid detection class or score")
            box = candidate.get("box_xyxy_px")
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(n) not in (int, float) or not math.isfinite(n) for n in box)
                    or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                raise ValueError("matching: invalid original-pixel box")
            mask = candidate.get("mask")
            if (not isinstance(mask, dict) or set(mask) != {"encoding", "size", "counts"}
                    or mask["encoding"] != "rle_row_major" or mask["size"] != size
                    or any(type(n) is not int for n in mask["size"])
                    or not isinstance(mask["counts"], list) or not mask["counts"]
                    or any(type(n) is not int or n < 0 for n in mask["counts"])
                    or sum(mask["counts"]) != height * width):
                raise ValueError("matching: invalid original-resolution RLE mask")
            mask_status = "available" if sum(mask["counts"][1::2]) else "empty"
            if candidate.get("mask_status") != mask_status:
                raise ValueError("matching: mask status does not match the saved mask")
    return {frame["frame_id"]: frame for frame in frames}


def parse_window_matching(response_text, window, checklist, frame_ids, candidate_frames, *, cleanup_audit=None):
    """Validate matching; an audit collector permits conservative normalization.

    The default remains strict for saved downstream inputs. All supplied bindings
    must be valid and unique. Omitted required bindings can only be supplied as
    missing when the validated candidate inventory explicitly has no detections.
    """
    value = json.loads(response_text)
    _fields(value, ("window_id", "checks", "bindings"), "window matching")
    if value["window_id"] != window["id"]:
        raise ValueError("matching: wrong window ID")
    if not isinstance(value["checks"], list):
        raise ValueError("matching.checks: expected array")
    checks = {check["id"]: check for check in checklist}
    entities = {entity for check in checklist for entity in check["entity_keys"]}
    candidates = {c["candidate_id"]: c for frame in candidate_frames for c in frame["candidates"]}
    if not isinstance(value["bindings"], list):
        raise ValueError("matching.bindings: expected array")
    bindings = {}
    for binding in value["bindings"]:
        _fields(binding, ("entity_key", "frame_id", "status", "candidate_id", "reason"), "entity binding")
        _text(binding["entity_key"], "binding.entity_key")
        _text(binding["reason"], "binding.reason")
        frame = binding["frame_id"]
        if (binding["entity_key"] not in entities or type(frame) is not int or frame not in frame_ids):
            raise ValueError("matching: binding entity/frame not supplied in this window")
        key = (binding["entity_key"], frame)
        if key in bindings:
            raise ValueError("matching: duplicate entity/frame binding")
        bindings[key] = binding
        cid = binding["candidate_id"]
        if binding["status"] == "bound":
            if not isinstance(cid, str) or cid not in candidates or candidates[cid]["frame_id"] != frame:
                raise ValueError("matching: bound candidate must exist in the specified frame")
        elif binding["status"] == "missing":
            if cid is not None:
                raise ValueError("matching: missing binding must have null candidate_id")
        else:
            raise ValueError("matching: binding status must be bound or missing")
    seen, needed = set(), set()
    for item in value["checks"]:
        _fields(item, ("check_id", "relevance", "reason", "experts", "limitations"), "check matching")
        _text(item["check_id"], "check_id")
        if item["check_id"] not in checks or item["check_id"] in seen:
            raise ValueError("matching: unknown or duplicate check ID")
        seen.add(item["check_id"])
        if item["relevance"] not in ("related", "unrelated", "uncertain"):
            raise ValueError("matching: invalid relevance")
        _text(item["reason"], "matching.reason")
        _strings(item["limitations"], "matching.limitations")
        if not isinstance(item["experts"], list):
            raise ValueError("matching.experts: expected array")
        if item["relevance"] == "unrelated" and item["experts"]:
            raise ValueError("matching: unrelated checks cannot request experts")
        if (item["relevance"] == "uncertain" or
                (item["relevance"] == "related" and not item["experts"])) and not item["limitations"]:
            raise ValueError("matching: explain uncertain relevance or missing expert coverage")
        requested = set()
        for expert in item["experts"]:
            if not isinstance(expert, dict):
                raise ValueError("matching: expected expert proposal")
            expert_id = expert.get("expert_id")
            _text(expert_id, "expert_id")
            if expert_id not in EXPERT_CATALOG or expert_id in requested:
                raise ValueError("matching: unknown or duplicate expert ID")
            requested.add(expert_id)
            if expert_id == "object_spatial":
                _fields(expert, ("expert_id", "entity_keys", "frame_ids", "reason"), "spatial proposal")
                _strings(expert["entity_keys"], "expert.entity_keys", nonempty=True)
                expert_entities = expert["entity_keys"]
            else:
                _fields(expert, ("expert_id", "target_entity_key", "reference_entity_key", "motion_mode",
                                 "frame_ids", "reason"), "motion proposal")
                target, reference = expert["target_entity_key"], expert["reference_entity_key"]
                _text(target, "motion.target_entity_key")
                if expert["motion_mode"] == "absolute":
                    if reference is not None:
                        raise ValueError("matching: absolute motion must have null reference_entity_key")
                    expert_entities = [target]
                elif expert["motion_mode"] == "relative":
                    _text(reference, "motion.reference_entity_key")
                    if target == reference:
                        raise ValueError("matching: motion target and reference must be distinct")
                    expert_entities = [target, reference]
                else:
                    raise ValueError("matching: motion_mode must be absolute or relative")
            if not set(expert_entities) <= set(checks[item["check_id"]]["entity_keys"]):
                raise ValueError("matching: expert entity not declared by this check")
            _text(expert["reason"], "expert.reason")
            frames = expert["frame_ids"]
            if (not isinstance(frames, list) or not frames or any(type(f) is not int for f in frames)
                    or frames != sorted(set(frames)) or not set(frames) <= set(frame_ids)):
                raise ValueError("matching: expert frames must be unique, chronological supplied window frames")
            if expert_id == "short_window_motion":
                if len(frames) < 2 or frames != list(range(frames[0], frames[-1] + 1)):
                    raise ValueError("matching: motion requires at least two continuous frames")
                needed.update((entity, frames[0]) for entity in expert_entities)
            else:
                needed.update((entity, frame) for entity in expert_entities for frame in frames)
    if seen != set(checks):
        raise ValueError("matching: every checklist check needs exactly one relevance entry")
    original_count = len(bindings)
    added = []
    if cleanup_audit is not None:
        # Absence of an inventory record is not evidence of an empty candidate list.
        empty_frames = {frame["frame_id"] for frame in candidate_frames
                        if frame.get("status") == "no_detections" and frame["candidates"] == []}
        for entity, frame in sorted(needed - set(bindings)):
            if frame in empty_frames:
                binding = {"entity_key": entity, "frame_id": frame, "status": "missing",
                           "candidate_id": None,
                           "reason": "No candidates were detected in this frame; missing binding added from the candidate inventory."}
                bindings[(entity, frame)] = binding
                added.append(binding)
    if needed - set(bindings) or (cleanup_audit is None and set(bindings) != needed):
        raise ValueError("matching: exactly one bound/missing binding is required per proposed entity/frame")
    for item in value["checks"]:
        for expert in item["experts"]:
            if expert["expert_id"] == "short_window_motion" and expert["motion_mode"] == "relative":
                first = expert["frame_ids"][0]
                a, b = (bindings[(expert[role], first)] for role in ("target_entity_key", "reference_entity_key"))
                if a["status"] == b["status"] == "bound" and a["candidate_id"] == b["candidate_id"]:
                    raise ValueError("matching: motion target and reference cannot bind the same candidate")
    removed = [binding for key, binding in bindings.items() if key not in needed]
    if removed or added:
        value["bindings"] = [binding for key, binding in bindings.items() if key in needed]
        # Recheck the normalized response using the same strict downstream contract.
        parse_window_matching(json.dumps(value), window, checklist, frame_ids, candidate_frames)
    if removed:
        cleanup_audit.append({"warning": "unused_bindings_removed", "window_id": value["window_id"],
                              "reason": "Bindings are not required by any expert proposal in this window.",
                              "original_count": original_count, "retained_count": len(value["bindings"]),
                              "removed_bindings": removed})
    if added:
        cleanup_audit.append({"warning": "missing_bindings_added_for_empty_frames", "window_id": value["window_id"],
                              "method": EMPTY_CANDIDATE_BINDING_METHOD,
                              "reason": "Required bindings were omitted for frames with no detected candidates.",
                              "original_count": original_count, "retained_count": len(value["bindings"]),
                              "added_bindings": added,
                              "evidence": [{"frame_id": frame, "status": "no_detections", "candidate_count": 0}
                                           for frame in sorted({b["frame_id"] for b in added})]})
    return value
