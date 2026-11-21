"""Execute validated, candidate-bound proposals; no check verdicts or scoring."""
import copy
import itertools
import json

import cv2

from experts import MaskSeed, ShortWindowMotionExpert, VideoFrame
from experts.geometry import pairwise_spatial_evidence
from experts.masks import decode_mask
from vlm_matching import (MATCHING_MODEL, MATCHING_PROTOCOL, parse_window_matching,
                          validate_candidate_inputs, validate_matching_inputs)


EXECUTION_PROTOCOL = "bound-expert-execution-v1"
CANDIDATE_FILE = "03-yolo-candidates.json"
MATCHING_FILE = "03-window-matching.json"


def validate_execution_inputs(task, obs, selection, times, candidates, matching):
    """Revalidate the entire stage boundary before any geometry, reads or LK calls."""
    selected = validate_matching_inputs(task, obs, selection, times)
    by_frame = validate_candidate_inputs(candidates, selected, times)
    if (not isinstance(matching, dict) or matching.get("protocol") != MATCHING_PROTOCOL
            or matching.get("model") != MATCHING_MODEL
            or matching.get("status") != "proposals_and_bindings_only"
            or matching.get("task") != task or matching.get("requirements") != obs["requirements"]
            or matching.get("checklist") != obs["candidate_checks"]
            or matching.get("candidate_file") != CANDIDATE_FILE):
        raise ValueError("execution: matching context/protocol does not match the upstream inputs")
    windows = matching.get("windows")
    if not isinstance(windows, list) or len(windows) != len(selected):
        raise ValueError("execution: matching must cover every selected window exactly once")
    for window, record in zip(selected, windows):
        first, last = window["frame_range"]
        ids = list(range(first, last + 1))
        request_file = f"03-match-{window['id']}.json"
        image_directory = f"matching_frames/{window['id']}"
        if (not isinstance(record, dict) or record.get("window_id") != window["id"]
                or record.get("frame_range") != [first, last]
                or record.get("frame_ids") != ids
                or any(type(i) is not int for i in record["frame_ids"])
                or record.get("pts_seconds") != [times[i] for i in ids]
                or record.get("candidate_file") != CANDIDATE_FILE
                or record.get("request_file") != request_file
                or record.get("image_directory") != image_directory):
            raise ValueError("execution: invalid matching window/frame/PTS/source reference")
        images = [{"frame_id": i, "pts_seconds": times[i],
                   "original_file": f"{image_directory}/{i}.jpg",
                   "candidates_file": f"{image_directory}/{i}-candidates.jpg"} for i in ids]
        if record.get("images") != images:
            raise ValueError("execution: invalid matching image references")
        expected_source = {"method": "matching_vlm_candidate_selection", "model": MATCHING_MODEL,
                           "request_file": request_file, "candidate_file": CANDIDATE_FILE}
        bindings = record.get("bindings")
        if not isinstance(bindings, list):
            raise ValueError("execution: expected saved bindings")
        raw_bindings = []
        fields = ("entity_key", "frame_id", "status", "candidate_id", "reason")
        for binding in bindings:
            if (not isinstance(binding, dict) or set(binding) != {*fields, "pts_seconds", "source"}
                    or type(binding.get("frame_id")) is not int or binding["frame_id"] not in ids
                    or type(binding.get("pts_seconds")) not in (int, float)
                    or binding["pts_seconds"] != times[binding["frame_id"]]
                    or binding.get("source") != expected_source):
                raise ValueError("execution: invalid binding PTS/source reference")
            raw_bindings.append({field: binding[field] for field in fields})
        raw = {"window_id": record["window_id"], "checks": record.get("checks"), "bindings": raw_bindings}
        # Reuse exactly the matching contract: checks, experts, roles, candidate IDs,
        # required binding keys and continuous in-window motion frame references.
        parse_window_matching(json.dumps(raw, allow_nan=False), window, obs["candidate_checks"],
                              ids, [by_frame[i] for i in ids])
    return by_frame


def _status(statuses):
    return ("complete" if all(s == "complete" for s in statuses) else
            "missing" if all(s == "missing" for s in statuses) else "partial")


def _spatial(proposal, bindings, by_frame):
    frames = []
    for index in proposal["frame_ids"]:
        inventory = {c["candidate_id"]: c for c in by_frame[index]["candidates"]}
        objects, resolved = [], {}
        for entity in proposal["entity_keys"]:
            binding = bindings[(entity, index)]
            candidate = inventory[binding["candidate_id"]] if binding["status"] == "bound" else None
            resolved[entity] = candidate
            item = {"entity_key": entity, "candidate_id": binding["candidate_id"],
                    "status": "missing", "reason": "binding_missing", "candidate": None}
            if candidate is not None:
                item.update(status="complete" if candidate["mask_status"] == "available" else "partial",
                            reason=None if candidate["mask_status"] == "available" else "empty_mask",
                            candidate={key: copy.deepcopy(candidate[key]) for key in
                                       ("candidate_id", "class_id", "class_name", "confidence",
                                        "box_xyxy_px", "mask_status")})
                item["mask_ref"] = {"file": CANDIDATE_FILE, "frame_id": index,
                                    "candidate_id": candidate["candidate_id"]}
            objects.append(item)
        pairs = []
        for a, b in itertools.combinations(proposal["entity_keys"], 2):
            ca, cb = resolved[a], resolved[b]
            pair = {"entity_keys": [a, b], "status": "missing", "reason": "binding_missing", "measurement": None}
            if ca is not None and cb is not None:
                measurement = pairwise_spatial_evidence(ca, cb)
                pair.update(status="complete" if measurement["mask_status"] == "available" else "partial",
                            reason=None if measurement["mask_status"] == "available" else "empty_mask",
                            measurement=measurement)
                if ca["candidate_id"] == cb["candidate_id"]:
                    pair["limitations"] = ["Both supplied entity bindings select the same candidate; geometry cannot distinguish their identities."]
            pairs.append(pair)
        frames.append({"frame_id": index, "pts_seconds": by_frame[index]["pts_seconds"],
                       "status": _status([o["status"] for o in objects] + [p["status"] for p in pairs]),
                       "objects": objects, "pairs": pairs})
    return {"protocol": "bound-object-spatial-evidence-v1", "expert_id": "object_spatial",
            "coordinate_system": "original_image_pixels_xyxy",
            "status": _status([f["status"] for f in frames]), "frames": frames,
            "limitations": ["Measurements are conditional on VLM candidate bindings, not independently verified identities.",
                            "2D image measurements do not establish contact, support, grasp, release or physical stability."]}


def _seed(entity, first, bindings, by_frame, role):
    binding = bindings[(entity, first)]
    reason = "binding_missing" if binding["status"] == "missing" else None
    candidate = None
    if reason is None:
        candidate = next(c for c in by_frame[first]["candidates"] if c["candidate_id"] == binding["candidate_id"])
        if candidate["mask_status"] != "available":
            reason = "empty_mask"
    if reason:
        return None, {"role": role, "entity_key": entity, "frame_id": first,
                      "pts_seconds": by_frame[first]["pts_seconds"], "candidate_id": binding["candidate_id"],
                      "reason": reason, "binding_reason": binding["reason"]}
    return MaskSeed(entity, first, decode_mask(candidate["mask"]), candidate["candidate_id"]), None


def _motion(proposal, bindings, by_frame, times, cap):
    ids = proposal["frame_ids"]
    target, missing_target = _seed(proposal["target_entity_key"], ids[0], bindings, by_frame, "target")
    reference, missing_reference = None, None
    if proposal["motion_mode"] == "relative":
        reference, missing_reference = _seed(proposal["reference_entity_key"], ids[0], bindings, by_frame, "reference")
    missing = [item for item in (missing_target, missing_reference) if item is not None]
    if target is None:
        return {"execution_status": "not_executed", "status": "missing",
                "reason": "target_mask_unavailable", "missing": missing, "result": None}
    cap.set(cv2.CAP_PROP_POS_FRAMES, ids[0])
    frames = []
    for index in ids:
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"execution video decode failed at frame {index}")
        if list(image.shape[:2]) != by_frame[index]["image_size_hw"]:
            raise ValueError(f"execution image size differs from candidates at frame {index}")
        frames.append(VideoFrame(index, times[index], image))
    # A missing reference never blocks a reliable target track, and never creates
    # a synthetic reference. The original expert marks relative samples missing.
    result = ShortWindowMotionExpert().track(frames, target, reference=reference)
    if proposal["motion_mode"] == "absolute":
        status = result["target"]["status"]
    else:
        status = _status(["complete" if s["status"] == "available" else "missing"
                          for s in result["relative_motion"]])
    return {"execution_status": "executed", "status": status, "reason": None,
            "missing": missing, "result": result}


def execute_proposals(task, obs, selection, times, cap, candidates, matching):
    """Use existing candidates and raw continuous video; retain each check association.

    Results remain individual executions. Cross-window evidence packages, summaries
    and check/grade decisions belong to subsequent stages.
    """
    by_frame = validate_execution_inputs(task, obs, selection, times, candidates, matching)
    records = []
    for window in matching["windows"]:
        bindings = {(b["entity_key"], b["frame_id"]): b for b in window["bindings"]}
        for check in window["checks"]:
            for proposal in check["experts"]:
                ids = proposal["frame_ids"]
                spatial = proposal["expert_id"] == "object_spatial"
                entities = (proposal["entity_keys"] if spatial else [proposal["target_entity_key"]] +
                            ([proposal["reference_entity_key"]] if proposal["motion_mode"] == "relative" else []))
                keys = [(entity, i) for i in (ids if spatial else ids[:1]) for entity in entities]
                record = {"id": f"execution-{len(records):04d}", "window_id": window["window_id"],
                          "check_id": check["check_id"], "expert_id": proposal["expert_id"],
                          "entity_keys": list(entities), "frame_ids": list(ids),
                          "frame_range": [ids[0], ids[-1]], "pts_seconds": [times[i] for i in ids],
                          "proposal": copy.deepcopy(proposal),
                          "association": {key: copy.deepcopy(check[key]) for key in ("relevance", "reason", "limitations")},
                          "bindings": [copy.deepcopy(bindings[key]) for key in keys],
                          "source": {"matching_file": MATCHING_FILE,
                                     "window_matching_file": f"03-matching-{window['window_id']}.json",
                                     "request_file": window["request_file"], "candidate_file": CANDIDATE_FILE}}
                if spatial:
                    result = _spatial(proposal, bindings, by_frame)
                    record.update(execution_status="executed", status=result["status"], reason=None, result=result)
                else:
                    record.update(_motion(proposal, bindings, by_frame, times, cap))
                records.append(record)
    return {"protocol": EXECUTION_PROTOCOL, "status": "executions_recorded",
            "candidate_file": CANDIDATE_FILE, "matching_file": MATCHING_FILE,
            "candidate_provenance": copy.deepcopy(candidates.get("provenance")), "records": records}
