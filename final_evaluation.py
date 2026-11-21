"""Program-assembled text materials and one structured final rubric selection."""
import copy
import json
import math
from pathlib import Path

from check_judgment import EVIDENCE_FILE, JUDGMENT_MODEL, JUDGMENT_PROTOCOL, has_usable_measurements


FINAL_PROTOCOL = "jev-text-final-evaluation-v1"
FINAL_MODEL = JUDGMENT_MODEL
JUDGMENT_FILE = "06-check-judgments.json"
MATERIALS_FILE = "07-final-materials.json"
EVALUATION_FILE = "07-final-evaluation.json"
REQUEST_FILE = "07-evaluate.json"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "jev_final_evaluation.txt"
RUBRIC_PATH = PROMPT_PATH.with_name("roboreward_rubric.json")
FINAL_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
RUBRIC = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
CRITERIA = {**RUBRIC["grades"], "insufficient_evidence":
            "The supplied evidence leaves unresolved information that prevents distinguishing the applicable rubric level."}
SCORES = {f"level_{i}": i for i in range(1, 6)}


def _require(condition, message):
    if not condition:
        raise ValueError("final evaluation: " + message)


def _same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def validate_final_inputs(task, obs, packages, judgments):
    """Validate the complete stage-6 boundary; no image reads or model/expert runs."""
    _require(isinstance(judgments, dict) and judgments.get("protocol") == JUDGMENT_PROTOCOL
             and judgments.get("model") == JUDGMENT_MODEL and judgments.get("status") == "checks_judged"
             and judgments.get("evidence_file") == EVIDENCE_FILE, "expected completed check judgments")
    for key, value in (("instruction", task), ("requirements", obs["requirements"]),
                       ("checklist", obs["candidate_checks"])):
        _require(_same(judgments.get(key), value), "judgment context differs from upstream records")
    records = judgments.get("judgments")
    _require(isinstance(records, list) and len(records) == len(packages["packages"]),
             "all checks must have exactly one completed judgment")
    for index, (record, packet) in enumerate(zip(records, packages["packages"])):
        _require(isinstance(record, dict), "invalid check judgment")
        expected = {"check_id": packet["check_id"], "check": packet["check"],
                    "evidence_file": EVIDENCE_FILE, "package_index": index,
                    "provided_evidence": {"window_ids": [w["window_id"] for w in packet["windows"]],
                                          "measurement_ids": [m["id"] for m in packet["measurements"]],
                                          "image_coverage": packet["image_coverage"]}}
        _require(all(_same(record.get(k), v) for k, v in expected.items()),
                 "check identity/order/evidence differs from upstream records")
        _require(isinstance(record.get("answer"), str) and record["answer"] in ("yes", "no", "unknown"),
                 "invalid check answer")
        empty = not packet["images"] and not has_usable_measurements(packet)
        if empty:
            _require(record.get("answer_source") == "program_no_evidence" and record["answer"] == "unknown"
                     and record.get("reason") == "no_images_or_usable_measurements"
                     and record.get("actual_model") is None and record.get("request_file") is None,
                     "empty evidence must retain program unknown and its source")
        else:
            _require(record.get("answer_source") == "jev" and record.get("actual_model") == JUDGMENT_MODEL
                     and record.get("request_file") == f"06-judge-c{index:04d}.json" and record.get("reason") is None,
                     "nonempty evidence must retain its JEV judgment source")
    _require(type(judgments.get("jev_call_count")) is int
             and judgments["jev_call_count"] == sum(r["answer_source"] == "jev" for r in records),
             "invalid check judgment call count")


def _unresolved(packet, judgment):
    """Index recorded unknowns/limitations; do not infer why a model chose unknown."""
    items = []
    if judgment["answer"] == "unknown":
        items.append({"kind": "unknown_check", "answer_source": judgment["answer_source"],
                      "recorded_reason": judgment["reason"]})
    if packet["reason"] is not None:
        items.append({"kind": "package_limitation", "reason": packet["reason"]})
    if packet["image_coverage"]["missing_ranges"]:
        items.append({"kind": "unobserved_image_ranges",
                      "ranges": copy.deepcopy(packet["image_coverage"]["missing_ranges"])})
    for window in packet["windows"]:
        if window["relevance"] == "uncertain" or window["limitations"]:
            items.append({"kind": "window_limitation", **copy.deepcopy(window)})
    for binding in packet["bindings"]:
        if binding["status"] == "missing":
            items.append({"kind": "missing_binding", **copy.deepcopy(binding)})
    for measurement in packet["measurements"]:
        if measurement["status"] != "complete" or measurement["missing"]:
            items.append({"kind": "measurement_gap", **{key: copy.deepcopy(measurement[key]) for key in
                          ("id", "window_id", "expert_id", "status", "execution_status", "reason", "missing", "raw_result_ref")}})
    for comparison in packet["threshold_comparisons"]["comparisons"]:
        if comparison["status"] != "compared":
            items.append({"kind": "unavailable_comparison", **copy.deepcopy(comparison)})
    return items


def build_final_materials(task, obs, packages, judgments):
    """Copy audited summaries with references; never label all input as support."""
    validate_final_inputs(task, obs, packages, judgments)
    checks = []
    for index, (packet, judgment) in enumerate(zip(packages["packages"], judgments["judgments"])):
        checks.append({"check_id": packet["check_id"], "check": copy.deepcopy(packet["check"]),
                       "answer": judgment["answer"], "answer_source": judgment["answer_source"],
                       "recorded_reason": judgment["reason"],
                       "judgment_ref": {"file": JUDGMENT_FILE, "index": index,
                                        "record_file": f"06-check-c{index:04d}.json", "request_file": judgment["request_file"]},
                       "evidence_ref": {"file": EVIDENCE_FILE, "package_index": index},
                       "evidence_summary": {key: copy.deepcopy(packet[key]) for key in
                                            ("status", "reason", "windows", "image_coverage", "bindings",
                                             "measurements", "threshold_comparisons", "limitations")},
                       "unresolved": _unresolved(packet, judgment)})
    return {"protocol": FINAL_PROTOCOL, "status": "materials_assembled", "instruction": task,
            "requirements": copy.deepcopy(obs["requirements"]), "rubric": copy.deepcopy(RUBRIC),
            "episode": copy.deepcopy(packages["episode"]), "checks": checks,
            "source": {"evidence_file": EVIDENCE_FILE, "judgment_file": JUDGMENT_FILE},
            "summary_origin": "program_copy_of_recorded_answers_summaries_ranges_and_limitations",
            "evidence_note": "References identify supplied records, not model-selected supporting evidence. No free-text JEV rationale is available."}


def build_final_request(materials):
    # A serialized state and no assets keep this call entirely text-only.
    return {"model": FINAL_MODEL, "state": json.dumps(materials, ensure_ascii=False, allow_nan=False),
            "questions": {"grade": {"type": "choice", "instructions": FINAL_PROMPT, "criteria": dict(CRITERIA)}}}


def parse_final_response(response):
    _require(isinstance(response, dict) and response.get("model") == FINAL_MODEL
             and isinstance(response.get("answers"), dict) and set(response["answers"]) == {"grade"},
             "expected the pinned model and exactly one grade answer")
    answer = response["answers"]["grade"]
    _require(isinstance(answer, dict) and answer.get("type") == "choice"
             and isinstance(answer.get("choice"), str) and answer["choice"] in CRITERIA,
             "expected a rubric level or insufficient_evidence Choice")
    confidence, probabilities = answer.get("confidence"), answer.get("probabilities")
    def probability(value):
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    _require(confidence is None or probability(confidence), "invalid confidence metadata")
    _require(probabilities is None or (isinstance(probabilities, dict) and set(probabilities) == set(CRITERIA)
             and all(probability(p) for p in probabilities.values())), "invalid probability metadata")
    selected = answer["choice"]
    return {"jev_choice": selected, "score": SCORES.get(selected),
            "status": "insufficient_evidence" if selected == "insufficient_evidence" else "scored",
            "actual_model": response["model"], "confidence": confidence,
            "probabilities": copy.deepcopy(probabilities), "usage": copy.deepcopy(response.get("usage"))}
