"""Conservative, explicit-language adapter; never infer physical task success.

Only full instructions in the documented grammar are eligible. A paraphrase,
extra constraint, negation, identity mismatch or scope mismatch stays unknown.
The grammar is deliberately narrow, not a general natural-language interpreter.
"""
import math
import re


ADAPTER_VERSION = "explicit-2d-requirements-v1"
LAYOUT_MARGIN_PX = 2
_NAME = r"[a-z0-9_ -]+?"
_LAYOUT = re.compile(
    rf"at episode end, the bounding box of (?P<subject>{_NAME}) must be entirely "
    rf"(?P<relation>to the left of|to the right of|above|below) the bounding box of "
    rf"(?P<reference>{_NAME}) in the image plane")
_MOTION = re.compile(
    rf"over the entire episode, the maximum image-plane relative-center excursion of "
    rf"(?P<subject>{_NAME}) relative to (?P<reference>{_NAME}), measured from the first frame, "
    r"must be (?P<comparison>at most|at least) (?P<percent>[0-9]+(?:\.[0-9]+)?)% of the image diagonal")


def _normalize(text):
    return " ".join(text.lower().strip().removesuffix(".").split())


def _entity(text):
    # Preserve all attributes; do not resolve aliases, synonyms or pronouns.
    return " ".join(text.lower().replace("_", " ").split())


def _parse(text):
    text = _normalize(text)
    match = _LAYOUT.fullmatch(text)
    if match:
        fields = match.groupdict()
        return {"expert": "layout", "scope": "episode_end",
                "predicate": {"to the left of": "image_left", "to the right of": "image_right",
                              "above": "image_above", "below": "image_below"}[fields["relation"]],
                "subject": _entity(fields["subject"]), "reference": _entity(fields["reference"]),
                "margin_px": LAYOUT_MARGIN_PX}
    match = _MOTION.fullmatch(text)
    if match:
        fields = match.groupdict()
        threshold = float(fields["percent"]) / 100
        if not math.isfinite(threshold):
            return None
        return {"expert": "motion", "scope": "entire_episode_from_first_frame",
                "predicate": "relative_excursion", "comparison": fields["comparison"],
                "threshold_fraction": threshold,
                "subject": _entity(fields["subject"]), "reference": _entity(fields["reference"])}
    return None


def adapt_requirement(task, requirement, expert, frames, episode_end_frame):
    """Link a measurement to an original requirement, without using Qwen predicates."""
    result = {"version": ADAPTER_VERSION, "coverage": "not_established",
              "reason": "original_instruction_outside_supported_language"}
    original = _parse(task)
    if original is None:
        return result
    if _parse(requirement["text"]) != original:
        result["reason"] = "requirement_does_not_match_explicit_instruction"
        return result
    if (original["subject"] != _entity(requirement["subject"])
            or original["reference"] != _entity(requirement["reference"])
            or original["subject"] == original["reference"]):
        result["reason"] = "instruction_entity_binding_mismatch"
        return result
    result["contract"] = original
    if expert != original["expert"]:
        result["reason"] = "expert_does_not_measure_required_quantity"
        return result
    scope_matches = (frames[1] == episode_end_frame and
                     (expert == "layout" or frames[0] == 0))
    result.update(coverage="direct" if scope_matches else "partial",
                  reason="explicit_quantity_and_scope_match" if scope_matches else "temporal_scope_mismatch")
    return result


def requirement_verdict(adapter, measurement, n_frames):
    """Return a requirement verdict only when the complete explicit contract is covered."""
    if adapter["coverage"] != "direct":
        return "unknown"
    contract = adapter["contract"]
    if contract["expert"] == "layout":
        gap = measurement["end_box_gaps_px"][contract["predicate"]]
        if gap > contract["margin_px"]:
            return "supported"
        if gap < -contract["margin_px"]:
            return "contradicted"
        return "unknown"
    if n_frames < 2:
        return "unknown"
    value = measurement["fraction_of_image_diagonal"]
    limit = contract["threshold_fraction"]
    satisfied = value <= limit if contract["comparison"] == "at most" else value >= limit
    return "supported" if satisfied else "contradicted"


def requirement_states(requirements, records):
    """Retain latest direct evidence, conflicts and unknowns; never accumulate achievements."""
    states = []
    for req in requirements:
        linked = [r for r in records if r["requirement_id"] == req["id"]]
        direct = [r for r in linked if r["semantic_adapter"]["coverage"] == "direct"]
        latest = max((r["frames"][1] for r in direct), default=None)
        current = [r for r in direct if r["frames"][1] == latest]
        decisive = [r for r in current if r["requirement_status"] in ("supported", "contradicted")]
        verdicts = {r["requirement_status"] for r in decisive}
        status = next(iter(verdicts)) if len(verdicts) == 1 else "unknown"
        states.append({"requirement_id": req["id"], "status": status,
                       "latest_direct_frame": latest, "evidence_ids": [r["id"] for r in decisive],
                       "unknown_evidence_ids": [r["id"] for r in linked if r["requirement_status"] == "unknown"],
                       "reason": ("conflicting_direct_evidence" if len(verdicts) > 1 else
                                  "latest_direct_evidence" if decisive else "no_decisive_direct_evidence")})
    return states
