"""VLM window/check/expert matching contracts; no video, model or expert execution."""
import copy
import json
import math
from pathlib import Path

from window_selection import build_selection_request, summarize_selection


MATCHING_PROTOCOL = "vlm-window-check-expert-v1"
MATCHING_MODEL = "qwen3.8-flash"
MATCHING_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "qwen_window_matching.txt"
MATCHING_PROMPT = MATCHING_PROMPT_PATH.read_text(encoding="utf-8")

# Capability contracts for matching proposals, not installed/runnable tools.
# The legacy layout/motion implementations do not implement these contracts.
EXPERT_CATALOG = {
    "object_spatial": {
        "name": "Object and spatial evidence",
        "execution_status": "not_implemented",
        "implementation": "YOLO nano instance segmentation (e.g. YOLO26n-seg), ONNX Runtime CPU",
        "evidence": "Candidate classes, confidence scores, boxes, masks and frame IDs; derived 2D gaps/overlap.",
        "limitations": [
            "Fixed pretrained class vocabulary; apple is supported, robot grippers, pegs and many lab props are not.",
            "Entity binding and actual detection must be checked; do not assume every named object has a mask.",
            "2D overlap does not establish 3D contact, support, grasp, release or physical stability.",
        ],
    },
    "short_window_motion": {
        "name": "Short-window motion evidence",
        "execution_status": "not_implemented",
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


def matching_context(task, obs, window):
    return {
        "instruction": task,
        "requirements": copy.deepcopy(obs["requirements"]),
        "checklist": copy.deepcopy(obs["candidate_checks"]),
        "window": {"id": window["id"], "frame_range": list(window["frame_range"])},
        "expert_catalog": copy.deepcopy(EXPERT_CATALOG),
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


def parse_window_matching(response_text, window, checklist, frame_ids):
    """Reject missing checks, invented references/tools and completion verdicts."""
    value = json.loads(response_text)
    _fields(value, ("window_id", "checks"), "window matching")
    if value["window_id"] != window["id"]:
        raise ValueError("matching: wrong window ID")
    if not isinstance(value["checks"], list):
        raise ValueError("matching.checks: expected array")
    checks = {check["id"]: check for check in checklist}
    seen = set()
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
            _fields(expert, ("expert_id", "entity_keys", "frame_ids", "reason"), "expert proposal")
            _text(expert["expert_id"], "expert_id")
            if expert["expert_id"] not in EXPERT_CATALOG or expert["expert_id"] in requested:
                raise ValueError("matching: unknown or duplicate expert ID")
            requested.add(expert["expert_id"])
            _text(expert["reason"], "expert.reason")
            _strings(expert["entity_keys"], "expert.entity_keys", nonempty=True)
            if not set(expert["entity_keys"]) <= set(checks[item["check_id"]]["entity_keys"]):
                raise ValueError("matching: expert entity not declared by this check")
            frames = expert["frame_ids"]
            if (not isinstance(frames, list) or not frames or any(type(f) is not int for f in frames)
                    or frames != sorted(set(frames)) or not set(frames) <= set(frame_ids)):
                raise ValueError("matching: expert frames must be unique, chronological supplied window frames")
            if expert["expert_id"] == "short_window_motion" and (
                    len(frames) < 2 or frames != list(range(frames[0], frames[-1] + 1))):
                raise ValueError("matching: motion requires at least two continuous frames")
    if seen != set(checks):
        raise ValueError("matching: every checklist check needs exactly one relevance entry")
    return value
