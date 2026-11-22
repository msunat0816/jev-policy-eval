"""One JEV image/text Choice per check; no final evaluation or evidence attribution."""
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import uuid

import cv2
import numpy as np

from evidence_packaging import build_evidence_packages


JUDGMENT_PROTOCOL = "jev-per-check-image-text-v1"
JUDGMENT_MODEL = "jev-1.14.0"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
EVIDENCE_FILE = "05-check-evidence.json"
IMAGE_HEADERS = {
    "TypeSafe-Danger-Zone-Images": "true",
    "TypeSafe-Danger-Zone-Long-Context": "true",
}
MIN_SIDE, MAX_SIDE = 280, 1568
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_BODY_BYTES = 32 * 1024 * 1024
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "jev_check_judgment.txt"
JUDGMENT_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
CRITERIA = {
    "yes": "The combined evidence sufficiently supports the check in its required time scope.",
    "no": "The combined evidence sufficiently supports that the check is not satisfied in its required time scope.",
    "unknown": "Evidence is insufficient, identity/time scope/coverage is inadequate, or a material conflict remains unresolved.",
}


def validate_judgment_inputs(task, obs, selection, times, candidates, matching, execution, packages):
    """Rebuild only arithmetic packages to check the stage boundary; no expert runs."""
    if not isinstance(packages, dict):
        raise ValueError("judgment: expected saved evidence packages")
    expected = build_evidence_packages(task, obs, selection, times, candidates, matching, execution,
                                       packages.get("threshold_config"))
    # JSON comparison also rejects bool-for-number edits that Python dict
    # equality would otherwise accept (False == 0, True == 1).
    if (json.dumps(packages, sort_keys=True, allow_nan=False)
            != json.dumps(expected, sort_keys=True, allow_nan=False)):
        raise ValueError("judgment: evidence packages differ from their upstream records")


def _stored_file(folder, relative):
    path = Path(relative)
    root = folder.resolve()
    if path.is_absolute() or not (root / path).resolve().is_relative_to(root):
        raise ValueError("judgment: image reference must be inside the run directory")
    path = root / path
    if not path.is_file():
        raise ValueError(f"judgment: missing image file {relative}")
    return path


def prepare_image(folder, relative, output_relative, *, kind):
    """Preserve the full stored canvas; shrink large images and pad small sides."""
    source = _stored_file(folder, relative)
    original = source.read_bytes()
    image = cv2.imdecode(np.frombuffer(original, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"judgment: cannot decode image {relative}")
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    if (new_width, new_height) != (width, height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    extra_x, extra_y = max(0, MIN_SIDE - new_width), max(0, MIN_SIDE - new_height)
    left, top = extra_x // 2, extra_y // 2
    right, bottom = extra_x - left, extra_y - top
    if extra_x or extra_y:
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    if image.shape[:2] == (height, width) and scale == 1 and not (extra_x or extra_y):
        # Existing stage-3 files are baseline JPEGs. Keep their exact bytes when
        # no adaptation is needed; an invalid format is not silently relabeled.
        if not original.startswith(b"\xff\xd8\xff"):
            raise ValueError(f"judgment: expected stored JPEG {relative}")
        prepared = original
    else:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95 if kind == "candidates" else 88])
        if not ok:
            raise ValueError(f"judgment: cannot encode prepared image {relative}")
        prepared = encoded.tobytes()
    if len(prepared) > MAX_IMAGE_BYTES:
        raise ValueError(f"judgment: prepared image exceeds 16 MiB: {relative}")
    destination = folder / output_relative
    with destination.open("xb") as handle:
        handle.write(prepared)
    metadata = {"kind": kind, "source_file": relative, "prepared_file": output_relative,
                "source_sha256": hashlib.sha256(original).hexdigest(),
                "prepared_sha256": hashlib.sha256(prepared).hexdigest(), "bytes": len(prepared),
                "source_size_hw": [height, width], "prepared_size_hw": list(image.shape[:2]),
                "resize_scale_xy": [new_width / width, new_height / height],
                "padding_ltrb_px": [left, top, right, bottom]}
    return prepared, metadata


def has_usable_measurements(packet):
    for measurement in packet["measurements"]:
        summary = measurement["summary"]
        if summary is None:
            continue
        if measurement["expert_id"] == "object_spatial":
            if any(metric["last_available"] is not None for pair in summary["pairs"] for metric in pair["metrics"].values()):
                return True
        elif any(summary[role] is not None and summary[role]["last_available_from_seed"] is not None
                 for role in ("target", "reference", "relative_motion")):
            return True
    return False


def build_check_request(packet, episode, folder, index):
    """Put every asset reference exactly once in state; no raw measurement arrays."""
    directory = f"check_frames/c{index:04d}"
    if packet["images"]:
        (folder / directory).mkdir(parents=True, exist_ok=False)
    frames, assets, manifest = [], {}, {}
    for frame in packet["images"]:
        item = {"frame_id": frame["frame_id"], "pts_seconds": frame["pts_seconds"],
                "sources": copy.deepcopy(frame["sources"]), "view_transforms": {}}
        # All stage-3 windows share this frame's precomputed YOLO candidates.
        # A single numbered view suffices; all window/binding sources stay in text.
        for kind, relative in (("original", frame["original_file"]),
                               ("candidates", frame["sources"][0]["candidates_file"])):
            data, metadata = prepare_image(folder, relative, f"{directory}/{frame['frame_id']}-{kind}.jpg", kind=kind)
            reference = str(uuid.uuid4())
            assets[reference] = {"type": "image", "format": "jpg", "data_base64": base64.b64encode(data).decode("ascii")}
            manifest[reference] = {"frame_id": frame["frame_id"], "pts_seconds": frame["pts_seconds"], **metadata}
            item[kind] = reference
            item["view_transforms"][kind] = {key: copy.deepcopy(metadata[key]) for key in
                                           ("source_size_hw", "prepared_size_hw", "resize_scale_xy", "padding_ltrb_px")}
        frames.append(item)
    state = {"instruction": packet["instruction"], "requirements": copy.deepcopy(packet["requirements"]),
             "current_check": copy.deepcopy(packet["check"]), "episode": copy.deepcopy(episode),
             "evidence": {key: copy.deepcopy(packet[key]) for key in
                          ("windows", "bindings", "measurements", "threshold_comparisons", "image_coverage", "limitations")}}
    state["evidence"]["frames"] = frames
    state["evidence"]["coordinate_note"] = "Measurements use original-frame coordinates; view transforms describe only the displayed stored image canvas."
    body = {"model": JUDGMENT_MODEL, "state": state,
            "questions": {packet["check_id"]: {"type": "choice", "instructions": JUDGMENT_PROMPT, "criteria": dict(CRITERIA)}}}
    if assets:
        body["assets"] = assets
    return body, manifest


def parse_check_response(response, check_id):
    if (not isinstance(response, dict) or response.get("model") != JUDGMENT_MODEL
            or not isinstance(response.get("answers"), dict) or set(response["answers"]) != {check_id}):
        raise ValueError("judgment: expected the pinned model and exactly the requested check answer")
    answer = response["answers"][check_id]
    if (not isinstance(answer, dict) or answer.get("type") != "choice"
            or not isinstance(answer.get("choice"), str) or answer["choice"] not in CRITERIA):
        raise ValueError("judgment: expected yes/no/unknown Choice")
    for key in ("confidence",):
        value = answer.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("judgment: invalid confidence metadata")
    probabilities = answer.get("probabilities")
    if probabilities is not None and (not isinstance(probabilities, dict) or set(probabilities) != set(CRITERIA)
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())):
        raise ValueError("judgment: invalid probability metadata")
    return {"answer": answer["choice"], "actual_model": response["model"],
            "confidence": answer.get("confidence"), "probabilities": copy.deepcopy(probabilities),
            "usage": copy.deepcopy(response.get("usage"))}
