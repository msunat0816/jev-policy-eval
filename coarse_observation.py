"""Approved task-guided prompt and structural validation; no API or video dependencies."""
import json
import math
from pathlib import Path


SCHEMA_VERSION = "task-guided-coarse-v2"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "qwen_coarse_observation.txt"
OBSERVE = PROMPT_PATH.read_text(encoding="utf-8")


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label}: expected fields {', '.join(expected)}")


def _text(value, label, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{label}: expected {'possibly empty ' if allow_empty else ''}string")


def _strings(value, label, *, nonempty=False, unique=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label}: expected {'nonempty ' if nonempty else ''}array")
    for item in value:
        _text(item, label)
    if unique and len(value) != len(set(value)):
        raise ValueError(f"{label}: duplicate reference")


def validate_observation(obs, indices):
    """Reject invalid structure/references, without claiming semantic or visual correctness."""
    _fields(obs, ("requirements", "frames", "candidate_checks"), "observation")
    if not isinstance(obs["requirements"], list) or not obs["requirements"]:
        raise ValueError("requirements: expected nonempty array")
    requirement_ids = set()
    for req in obs["requirements"]:
        _fields(req, ("id", "text", "subject", "reference"), "requirement")
        for key in ("id", "text", "subject"):
            _text(req[key], f"requirement.{key}")
        _text(req["reference"], "requirement.reference", allow_empty=True)
        if req["id"] in requirement_ids:
            raise ValueError("requirement: duplicate id")
        requirement_ids.add(req["id"])

    if not isinstance(obs["frames"], list):
        raise ValueError("frames: expected array")
    frame_ids = []
    for frame in obs["frames"]:
        _fields(frame, ("frame", "visible_facts", "uncertainties", "boxes"), "frame")
        if type(frame["frame"]) is not int:
            raise ValueError("frame: expected integer id")
        frame_ids.append(frame["frame"])
        _strings(frame["visible_facts"], "visible_facts")
        _strings(frame["uncertainties"], "uncertainties")
        if not isinstance(frame["boxes"], dict):
            raise ValueError("boxes: expected object")
        for entity, box in frame["boxes"].items():
            _text(entity, "box entity key")
            if (not isinstance(box, list) or len(box) != 4 or
                    not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000
                            for v in box)):
                raise ValueError("box: expected four finite coordinates in 0..1000")
            if not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError("box: expected x1 < x2 and y1 < y2")
    if frame_ids != list(indices):
        raise ValueError("frames: missing, duplicate, unordered or unprovided frame ids")

    if not isinstance(obs["candidate_checks"], list):
        raise ValueError("candidate_checks: expected array")
    check_ids = set()
    for check in obs["candidate_checks"]:
        _fields(check, ("id", "requirement_ids", "entity_keys", "related_frames",
                        "question", "rationale"), "candidate_check")
        for key in ("id", "question", "rationale"):
            _text(check[key], f"candidate_check.{key}")
        if check["id"] in check_ids:
            raise ValueError("candidate_check: duplicate id")
        check_ids.add(check["id"])
        _strings(check["requirement_ids"], "requirement_ids", nonempty=True, unique=True)
        if not set(check["requirement_ids"]) <= requirement_ids:
            raise ValueError("candidate_check: unknown requirement id")
        _strings(check["entity_keys"], "entity_keys", unique=True)
        # An uncertain/occluded actor may legitimately have no box in any frame.
        # Entity identity and instruction coverage need semantic review, not string matching.
        related = check["related_frames"]
        if (not isinstance(related, list) or not related or
                not all(type(v) is int for v in related) or
                len(set(related)) != len(related) or not set(related) <= set(frame_ids)):
            raise ValueError("candidate_check: invalid or unprovided related frame ids")
    return obs


def parse_observation(response_text, indices):
    return validate_observation(json.loads(response_text), indices)
