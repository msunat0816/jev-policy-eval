"""Checklist-aware window selection contracts; no model, video or expert dependencies."""
import json

from coarse_observation import validate_observation


SELECTION_PROTOCOL = "jev-window-selection-v2"
WINDOW_THRESHOLD = 2
WINDOW_PRIORITY = {
    "p0": "This window has no relevance to any criterion in the complete task checklist.",
    "p1": "Relevant but redundant; closer inspection would add no meaningful task-evaluation evidence.",
    "p2": "Closer inspection is worthwhile for evidence about at least one task criterion.",
    "p3": "A core ambiguity, contradiction or possible reversal here could change the final grade.",
}


def _candidates(windows):
    candidates = []
    seen = set()
    for i, bounds in enumerate(windows):
        if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                or not all(type(frame) is int for frame in bounds)
                or not 0 <= bounds[0] <= bounds[1] or tuple(bounds) in seen):
            raise ValueError("windows: expected unique, nonnegative integer frame ranges")
        seen.add(tuple(bounds))
        candidates.append({"id": f"w{i}", "frame_range": list(bounds)})
    return candidates


def build_selection_request(task, obs, windows):
    """Share the whole checklist and coarse descriptions across one Choice per window."""
    observed = [frame["frame"] for frame in obs["frames"]]
    validate_observation(obs, observed)
    if observed != sorted(set(observed)):
        raise ValueError("coarse observations: expected unique chronological frame IDs")
    covered = {req_id for check in obs["candidate_checks"] for req_id in check["requirement_ids"]}
    missing = {req["id"] for req in obs["requirements"]} - covered
    if missing:
        raise ValueError("complete checklist required; uncovered requirements: " + ", ".join(sorted(missing)))
    candidates = _candidates(windows)
    observed_set = set(observed)
    if any(not set(window["frame_range"]) <= observed_set for window in candidates):
        raise ValueError("window endpoints must refer to supplied coarse frames")
    state = {
        "task": task,
        "requirements": obs["requirements"],
        "checklist": obs["candidate_checks"],
        "coarse_observations": [
            {key: frame[key] for key in ("frame", "visible_facts", "uncertainties")}
            for frame in obs["frames"]
        ],
        "candidate_windows": candidates,
    }
    questions = {}
    for window in candidates:
        prompt = (
            "Use the original task instruction as authoritative. The shared state contains the complete "
            "checklist and all coarse observations; use all of them as context for this decision. "
            "Judge only whether this candidate window is worth inspecting more closely: "
            + json.dumps(window) + ". Consider the complete checklist, rather than a single preassigned check. "
            "Coarse descriptions are model-generated proposals, not verified task-completion evidence. "
            "Retain the value of checking apparent success, apparent failure, required events, final states "
            "and possible later reversals. related_frames are observation references, not restrictions on "
            "eligible windows. Missing boxes or unavailable experts are not reasons by themselves to skip "
            "a task-relevant window. Sparse samples do not guarantee that intervening frames will resolve "
            "an uncertainty. Rate this window independently; do not impose a selection quota or force the "
            "last window to be selected. Return only this window's inspection priority. Do not match the "
            "window to checks, select experts, answer checklist questions, or assign a completion score."
        )
        questions[window["id"]] = {
            "type": "choice", "instructions": prompt, "criteria": dict(WINDOW_PRIORITY),
        }
    return state, questions


def summarize_selection(windows, answers):
    """Keep every decision for audit and return selected windows in their original order."""
    candidates = _candidates(windows)
    if not isinstance(answers, dict) or set(answers) != {w["id"] for w in candidates}:
        raise ValueError("selection answers must contain exactly one decision per candidate window")
    decisions = []
    for window in candidates:
        answer = answers[window["id"]]
        if not isinstance(answer, str) or answer not in WINDOW_PRIORITY:
            raise ValueError(f"invalid selection priority for {window['id']}")
        priority = int(answer[1:])
        decisions.append({**window, "priority": priority, "selected": priority >= WINDOW_THRESHOLD})
    return {
        "protocol": SELECTION_PROTOCOL,
        "threshold": WINDOW_THRESHOLD,
        "windows": decisions,
        "selected_windows": [window for window in decisions if window["selected"]],
    }
