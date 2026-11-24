"""Per-check image references, arithmetic summaries and explicit numeric comparisons."""
import copy
import itertools
import math
import operator

from expert_execution import (CANDIDATE_FILE, EXECUTION_PROTOCOL, MATCHING_FILE,
                              validate_execution_inputs)


PACKAGING_PROTOCOL = "per-check-evidence-packages-v1"
THRESHOLD_PROTOCOL = "explicit-measurement-thresholds-v1"
EXECUTION_FILE = "04-expert-execution.json"
SPATIAL_UNITS = {
    "center_distance_px": "px", "box_edge_distance_px": "px", "box_iou": "ratio",
    "signed_box_gaps_px.a_left_of_b": "px", "signed_box_gaps_px.a_above_b": "px",
    "mask_intersection_px": "px^2", "mask_iou": "ratio",
}
MOTION_UNITS = {f"{role}.from_seed.magnitude_px": "px" for role in ("target", "reference", "relative_motion")}
OPERATORS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


def _require(condition, message):
    if not condition:
        raise ValueError("evidence: " + message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_threshold_config(config):
    """Validate the optional JSON before video/model calls; no text threshold extraction."""
    if config is None:
        return {"protocol": THRESHOLD_PROTOCOL, "comparisons": []}
    _require(isinstance(config, dict) and set(config) == {"protocol", "comparisons"}
             and config["protocol"] == THRESHOLD_PROTOCOL and isinstance(config["comparisons"], list),
             "invalid threshold configuration protocol/fields")
    seen = set()
    fields = {"id", "window_id", "check_id", "expert_id", "entity_keys", "metric", "unit",
              "aggregation", "frame_range", "operator", "threshold"}
    for item in config["comparisons"]:
        _require(isinstance(item, dict) and set(item) == fields, "invalid threshold comparison fields")
        _require(all(_text(item[k]) for k in ("id", "window_id", "check_id", "metric")), "invalid threshold IDs/metric")
        _require(item["id"] not in seen, "duplicate threshold ID")
        seen.add(item["id"])
        units = SPATIAL_UNITS if item["expert_id"] == "object_spatial" else MOTION_UNITS
        _require(item["expert_id"] in ("object_spatial", "short_window_motion")
                 and item["metric"] in units and item["unit"] == units[item["metric"]], "threshold metric/unit mismatch")
        entities = item["entity_keys"]
        _require(isinstance(entities, list) and entities and all(_text(e) for e in entities)
                 and len(entities) == len(set(entities)), "invalid threshold entity keys")
        count = 2 if item["expert_id"] == "object_spatial" or item["metric"].startswith("relative_motion.") else 1
        _require(len(entities) == count, "threshold requires explicit entity roles/pair")
        bounds = item["frame_range"]
        _require(isinstance(bounds, list) and len(bounds) == 2
                 and all(type(i) is int and i >= 0 for i in bounds) and bounds[0] <= bounds[1], "invalid threshold frame range")
        aggregations = ("at_frame", "min", "max") if item["expert_id"] == "object_spatial" else ("at_frame", "max")
        _require(item["aggregation"] in aggregations
                 and (item["aggregation"] != "at_frame" or bounds[0] == bounds[1]), "invalid threshold aggregation/scope")
        _require(isinstance(item["operator"], str) and item["operator"] in OPERATORS
                 and _number(item["threshold"]), "invalid threshold operator/value")
    return copy.deepcopy(config)


def _span(ids, times):
    return {"frame_range": [ids[0], ids[-1]], "pts_range_seconds": [times[ids[0]], times[ids[-1]]],
            "frame_count": len(ids)}


def _ranges(ids, times):
    groups = []
    for index in sorted(set(ids)):
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [_span(group, times) for group in groups]


def _coverage(requested, available, times):
    requested, available = set(requested), set(available)
    return {"requested_frame_count": len(requested), "available_frame_count": len(available),
            "missing_frame_count": len(requested - available),
            "available_ranges": _ranges(available, times), "missing_ranges": _ranges(requested - available, times)}


def _point(frame, value, times):
    return {"frame_id": frame, "pts_seconds": times[frame], "value": copy.deepcopy(value)}


def _extrema(values, times):
    # Equal extrema select the first chronological occurrence, without tolerance.
    if not values:
        return {"min": None, "max": None, "last_available": None}
    return {"min": _point(*min(values, key=lambda p: p[1]), times),
            "max": _point(*max(values, key=lambda p: p[1]), times),
            "last_available": _point(*values[-1], times)}


def _get(value, path):
    for key in path.split("."):
        value = value[key]
    return value


def _coverage_status(statuses):
    return ("complete" if all(s == "complete" for s in statuses) else
            "missing" if all(s == "missing" for s in statuses) else "partial")


def _validate_samples(samples, ids, times, *, relative=False):
    _require(isinstance(samples, list) and len(samples) == len(ids), "motion samples must cover proposed frames exactly once")
    lost = False
    for index, sample in zip(ids, samples):
        _require(isinstance(sample, dict) and type(sample.get("frame_id")) is int
                 and sample["frame_id"] == index and _number(sample.get("pts_seconds"))
                 and sample["pts_seconds"] == times[index], "motion sample frame/PTS mismatch")
        _require(sample.get("status") in ("available", "missing"), "invalid motion sample status")
        count_fields = ("target_valid_points", "reference_valid_points") if relative else ("valid_points",)
        _require(all(type(sample.get(k)) is int and sample[k] >= 0 for k in count_fields), "invalid motion point count")
        if sample["status"] == "missing":
            lost = True
            _require(_text(sample.get("reason")) and sample.get("step") is None and sample.get("from_seed") is None,
                     "missing motion must retain a reason and null measurements")
            continue
        _require(not lost, "motion cannot resume after a missing sample without a new execution/seed")
        _require(sample.get("reason") is None and sample.get("from_seed") is not None,
                 "available motion requires from-seed measurement")
        for name in ("step", "from_seed"):
            value = sample.get(name)
            if name == "step" and index == ids[0]:
                _require(value is None, "seed frame has no preceding motion step")
                continue
            _require(isinstance(value, dict) and set(value) == {"displacement_xy_px", "magnitude_px", "direction_degrees_image"},
                     "invalid motion vector")
            xy = value["displacement_xy_px"]
            _require(isinstance(xy, list) and len(xy) == 2 and all(_number(x) for x in xy)
                     and _number(value["magnitude_px"]) and value["magnitude_px"] >= 0,
                     "nonfinite/invalid motion vector")
            _require(math.isclose(math.hypot(*xy), value["magnitude_px"], rel_tol=1e-9, abs_tol=1e-9),
                     "motion magnitude disagrees with vector")
            angle = value["direction_degrees_image"]
            _require((angle is None and value["magnitude_px"] <= 1e-6)
                     or (_number(angle) and -180 <= angle <= 180), "invalid motion direction")


def validate_packaging_inputs(task, obs, selection, times, candidates, matching, execution, thresholds):
    """Check provenance and measurement coverage without rerunning either expert."""
    by_frame = validate_execution_inputs(task, obs, selection, times, candidates, matching)
    _require(isinstance(execution, dict) and execution.get("protocol") == EXECUTION_PROTOCOL
             and execution.get("status") == "executions_recorded"
             and execution.get("candidate_file") == CANDIDATE_FILE and execution.get("matching_file") == MATCHING_FILE
             and execution.get("candidate_provenance") == candidates.get("provenance")
             and isinstance(execution.get("records"), list), "invalid execution bundle/source")
    checks = {c["id"]: c for c in obs["candidate_checks"]}
    window_ids = {w["id"] for w in selection["windows"]}
    for item in thresholds["comparisons"]:
        _require(item["check_id"] in checks and item["window_id"] in window_ids, "unknown threshold check/window")
        _require(set(item["entity_keys"]) <= set(checks[item["check_id"]]["entity_keys"])
                 and item["frame_range"][1] < len(times), "threshold entity/frame outside supplied task/video")
    expected = [(w, c, p) for w in matching["windows"] for c in w["checks"] for p in c["experts"]]
    _require(len(execution["records"]) == len(expected), "execution records must cover all proposals exactly once")
    for i, (record, (window, check, proposal)) in enumerate(zip(execution["records"], expected)):
        _require(isinstance(record, dict), "invalid execution record")
        ids = proposal["frame_ids"]
        spatial = proposal["expert_id"] == "object_spatial"
        entities = (proposal["entity_keys"] if spatial else [proposal["target_entity_key"]] +
                    ([proposal["reference_entity_key"]] if proposal["motion_mode"] == "relative" else []))
        bindings = {(b["entity_key"], b["frame_id"]): b for b in window["bindings"]}
        used = [bindings[(entity, f)] for f in (ids if spatial else ids[:1]) for entity in entities]
        metadata = {"id": f"execution-{i:04d}", "window_id": window["window_id"], "check_id": check["check_id"],
                    "expert_id": proposal["expert_id"], "entity_keys": entities, "frame_ids": ids,
                    "frame_range": [ids[0], ids[-1]], "pts_seconds": [times[f] for f in ids], "proposal": proposal,
                    "association": {k: check[k] for k in ("relevance", "reason", "limitations")}, "bindings": used,
                    "source": {"matching_file": MATCHING_FILE, "window_matching_file": f"03-matching-{window['window_id']}.json",
                               "request_file": window["request_file"], "candidate_file": CANDIDATE_FILE}}
        _require(all(record.get(k) == value for k, value in metadata.items()), "execution identity/binding/source differs from proposal")
        _require(record.get("status") in ("complete", "partial", "missing"), "invalid execution coverage status")
        unresolved = [b for b in used if b["status"] == "unresolved"]
        if unresolved:
            missing = [{"entity_key": b["entity_key"], "frame_id": b["frame_id"], "pts_seconds": b["pts_seconds"],
                        "candidate_id": None, "reason": "binding_unresolved", "binding_reason": b["reason"]} for b in unresolved]
            _require(record.get("execution_status") == "not_executed" and record["status"] == "missing"
                     and record.get("reason") == "binding_unresolved" and record.get("result") is None
                     and record.get("missing") == missing, "unresolved binding must not have expert measurements")
            continue
        if spatial:
            _validate_spatial(record, by_frame, times)
        else:
            _validate_motion(record, by_frame, times)
    return by_frame


def _validate_spatial(record, by_frame, times):
    result, ids = record.get("result"), record["frame_ids"]
    _require(record.get("execution_status") == "executed" and isinstance(result, dict)
             and result.get("protocol") == "bound-object-spatial-evidence-v1"
             and isinstance(result.get("frames"), list) and len(result["frames"]) == len(ids), "invalid spatial result")
    bindings = {(b["entity_key"], b["frame_id"]): b for b in record["bindings"]}
    frame_statuses = []
    for index, frame in zip(ids, result["frames"]):
        _require(isinstance(frame, dict) and frame.get("frame_id") == index
                 and frame.get("pts_seconds") == times[index], "spatial frame/PTS mismatch")
        objects = frame.get("objects")
        _require(isinstance(objects, list) and all(isinstance(o, dict) for o in objects)
                 and [o.get("entity_key") for o in objects] == record["entity_keys"], "spatial entities mismatch")
        inventory = {c["candidate_id"]: c for c in by_frame[index]["candidates"]}
        for obj in objects:
            binding = bindings[(obj["entity_key"], index)]
            _require(obj.get("candidate_id") == binding["candidate_id"], "spatial candidate differs from binding")
            if binding["status"] == "missing":
                _require(obj.get("candidate") is None and obj.get("status") == "missing", "missing object contains measurements")
            else:
                candidate = inventory[binding["candidate_id"]]
                _require(obj.get("candidate") == {k: candidate[k] for k in
                         ("candidate_id", "class_id", "class_name", "confidence", "box_xyxy_px", "mask_status")}, "spatial object differs from candidate")
                _require(obj.get("status") == ("complete" if candidate["mask_status"] == "available" else "partial"),
                         "spatial object coverage status mismatch")
        pairs = frame.get("pairs")
        expected_pairs = list(itertools.combinations(record["entity_keys"], 2))
        _require(isinstance(pairs, list) and len(pairs) == len(expected_pairs), "spatial pair coverage mismatch")
        for pair, entities in zip(pairs, expected_pairs):
            _require(isinstance(pair, dict) and pair.get("entity_keys") == list(entities), "spatial pair roles mismatch")
            measurement = pair.get("measurement")
            bound = all(bindings[(entity, index)]["status"] == "bound" for entity in entities)
            if not bound:
                _require(measurement is None and pair.get("status") == "missing", "missing binding has spatial measurement")
                continue
            _require(isinstance(measurement, dict) and measurement.get("frame_id") == index
                     and measurement.get("pts_seconds") == times[index]
                     and measurement.get("candidate_ids") == [bindings[(entity, index)]["candidate_id"] for entity in entities],
                     "spatial measurement source/frame mismatch")
            masks_available = all(inventory[bindings[(e, index)]["candidate_id"]]["mask_status"] == "available" for e in entities)
            _require(measurement.get("mask_status") == ("available" if masks_available else "missing"), "spatial mask status mismatch")
            _require(pair.get("status") == ("complete" if masks_available else "partial"), "spatial pair coverage status mismatch")
            for metric, unit in SPATIAL_UNITS.items():
                try:
                    value = _get(measurement, metric)
                except (KeyError, TypeError):
                    raise ValueError("evidence: missing spatial metric " + metric) from None
                if metric.startswith("mask_") and not masks_available:
                    _require(value is None, "missing mask has non-null measurement")
                else:
                    _require(_number(value), "nonfinite/invalid spatial measurement")
                    if unit == "ratio":
                        _require(0 <= value <= 1, "overlap ratio outside 0..1")
                    elif not metric.startswith("signed_"):
                        _require(value >= 0, "negative spatial distance/area")
        status = _coverage_status([o["status"] for o in objects] + [p["status"] for p in pairs])
        _require(frame.get("status") == status, "spatial frame coverage status mismatch")
        frame_statuses.append(status)
    _require(result.get("status") == record["status"] == _coverage_status(frame_statuses), "spatial result coverage status mismatch")


def _validate_motion(record, by_frame, times):
    ids, proposal = record["frame_ids"], record["proposal"]
    bindings = {b["entity_key"]: b for b in record["bindings"]}
    def has_mask(entity):
        binding = bindings[entity]
        return binding["status"] == "bound" and next(c for c in by_frame[ids[0]]["candidates"]
                if c["candidate_id"] == binding["candidate_id"])["mask_status"] == "available"
    if not has_mask(proposal["target_entity_key"]):
        _require(record.get("execution_status") == "not_executed" and record.get("result") is None
                 and record["status"] == "missing" and record.get("reason") == "target_mask_unavailable",
                 "unavailable target must not have a motion result")
        return
    result = record.get("result")
    _require(record.get("execution_status") == "executed" and isinstance(result, dict)
             and result.get("protocol") == "short-window-motion-evidence-v1"
             and result.get("frame_range") == [ids[0], ids[-1]], "invalid motion result")
    for role in ("target", "reference"):
        region = result.get(role)
        entity = proposal[role + "_entity_key"]
        if entity is None or not has_mask(entity):
            _require(region is None, "unavailable/unrequested reference has track")
            continue
        _require(isinstance(region, dict) and region.get("entity_key") == entity
                 and region.get("mask_source") == bindings[entity]["candidate_id"]
                 and region.get("seed_frame_id") == ids[0], "motion region identity/seed mismatch")
        _validate_samples(region.get("samples"), ids, times)
        count = sum(s["status"] == "available" for s in region["samples"])
        status = "complete" if count == len(ids) else "partial" if count else "missing"
        _require(region.get("status") == status and isinstance(region.get("interruptions"), list), "motion region coverage mismatch")
        for gap in region["interruptions"]:
            bounds = gap.get("frame_range")
            _require(isinstance(bounds, list) and len(bounds) == 2 and all(type(f) is int and f in ids for f in bounds)
                     and bounds[0] <= bounds[1] and gap.get("pts_range_seconds") == [times[f] for f in bounds]
                     and _text(gap.get("reason")), "invalid motion interruption")
    _validate_samples(result.get("relative_motion"), ids, times, relative=True)
    for i, sample in enumerate(result["relative_motion"]):
        available = (result["reference"] is not None and result["target"]["samples"][i]["status"] == "available"
                     and result["reference"]["samples"][i]["status"] == "available")
        _require((sample["status"] == "available") == available, "relative motion coverage differs from target/reference")
        if available:
            for field in ("from_seed", "step"):
                if field == "step" and i == 0:
                    continue
                target = result["target"]["samples"][i][field]["displacement_xy_px"]
                reference = result["reference"]["samples"][i][field]["displacement_xy_px"]
                _require(all(math.isclose(value, a - b, rel_tol=1e-9, abs_tol=1e-9)
                             for value, a, b in zip(sample[field]["displacement_xy_px"], target, reference)),
                         "relative motion vector differs from target minus reference")
    samples = (result["target"]["samples"] if proposal["motion_mode"] == "absolute" else result["relative_motion"])
    count = sum(s["status"] == "available" for s in samples)
    _require(record["status"] == ("complete" if count == len(ids) else "partial" if count else "missing"),
             "execution motion coverage status mismatch")


def _spatial_summary(record, times):
    result = record["result"]
    pairs = []
    for entities in itertools.combinations(record["entity_keys"], 2):
        metrics = {}
        for metric, unit in SPATIAL_UNITS.items():
            values = []
            for frame in result["frames"]:
                pair = next(p for p in frame["pairs"] if p["entity_keys"] == list(entities))
                measurement = pair["measurement"]
                value = _get(measurement, metric) if measurement is not None else None
                if value is not None:
                    values.append((frame["frame_id"], value))
            metrics[metric] = {"unit": unit, **_extrema(values, times),
                               "coverage": _coverage(record["frame_ids"], [f for f, _ in values], times)}
        pairs.append({"entity_keys": list(entities), "metrics": metrics,
                      "limitations": list(dict.fromkeys(limit for frame in result["frames"] for pair in frame["pairs"]
                                                        if pair["entity_keys"] == list(entities) for limit in pair.get("limitations", [])))})
    return {"scope": "proposed_frames_only_no_interpolation", "pairs": pairs,
            "objects": [{"entity_key": entity,
                         "binding_coverage": _coverage(record["frame_ids"],
                             [b["frame_id"] for b in record["bindings"] if b["entity_key"] == entity and b["status"] == "bound"], times)}
                        for entity in record["entity_keys"]]}


def _motion_region_summary(samples, ids, times, interruptions):
    available = [s for s in samples if s["status"] == "available"]
    def point(sample):
        return {"frame_id": sample["frame_id"], "pts_seconds": sample["pts_seconds"], **copy.deepcopy(sample["from_seed"])}
    missing = []
    for sample in samples:
        if sample["status"] == "missing":
            if missing and missing[-1]["reason"] == sample["reason"] and missing[-1]["frame_range"][1] + 1 == sample["frame_id"]:
                missing[-1]["frame_range"][1] = sample["frame_id"]
                missing[-1]["pts_range_seconds"][1] = sample["pts_seconds"]
            else:
                missing.append({"frame_range": [sample["frame_id"]] * 2,
                                "pts_range_seconds": [sample["pts_seconds"]] * 2, "reason": sample["reason"]})
    return {"unit": "px", "seed_frame_id": ids[0], "seed_pts_seconds": times[ids[0]],
            "coverage": _coverage(ids, [s["frame_id"] for s in available], times),
            "requested_end_available": samples[-1]["status"] == "available",
            "first_available_from_seed": point(available[0]) if available else None,
            "last_available_from_seed": point(available[-1]) if available else None,
            "max_from_seed": point(max(available, key=lambda s: s["from_seed"]["magnitude_px"])) if available else None,
            "missing_spans": missing, "interruptions": copy.deepcopy(interruptions)}


def _summarize_record(record, times):
    item = {k: copy.deepcopy(record[k]) for k in ("id", "window_id", "expert_id", "entity_keys", "frame_range",
                                                 "execution_status", "status", "reason", "source")}
    item["proposal"] = {k: copy.deepcopy(v) for k, v in record["proposal"].items() if k != "frame_ids"}
    item["proposed_frame_ranges"] = _ranges(record["frame_ids"], times)
    item["raw_result_ref"] = {"file": EXECUTION_FILE, "record_id": record["id"]}
    item["missing"] = copy.deepcopy(record.get("missing", []))
    if record["result"] is None:
        item["summary"] = None
        item["limitations"] = (["Required entity bindings remain unresolved after completion; this expert was not executed. Images remain usable and this is not evidence of task failure."]
                               if record["reason"] == "binding_unresolved" else
                               ["No reliable target mask; tracking was not executed."])
        return item
    item["limitations"] = copy.deepcopy(record["result"]["limitations"])
    if record["expert_id"] == "object_spatial":
        item["summary"] = _spatial_summary(record, times)
    else:
        raw, ids = record["result"], record["frame_ids"]
        summary = {"motion_mode": record["proposal"]["motion_mode"], "coordinate_system": raw["coordinate_system"]}
        for role in ("target", "reference"):
            region = raw[role]
            summary[role] = (_motion_region_summary(region["samples"], ids, times, region["interruptions"])
                             if region is not None else None)
        summary["relative_motion"] = (_motion_region_summary(raw["relative_motion"], ids, times, [])
                                       if record["proposal"]["motion_mode"] == "relative" else None)
        item["summary"] = summary
    return item


def _threshold_comparison(contract, records, times):
    out = {"contract": copy.deepcopy(contract), "status": "not_comparable", "reason": None,
           "measurement": None, "satisfied": None, "raw_result_ref": None}
    record = next((r for r in records if r["window_id"] == contract["window_id"]
                   and r["expert_id"] == contract["expert_id"]), None)
    if record is None:
        out["reason"] = "no_matching_expert_execution"
        return out
    out["raw_result_ref"] = {"file": EXECUTION_FILE, "record_id": record["id"]}
    if record["result"] is None:
        out["reason"] = record["reason"]
        return out
    values = {}
    if record["expert_id"] == "object_spatial":
        for frame in record["result"]["frames"]:
            pair = next((p for p in frame["pairs"] if set(p["entity_keys"]) == set(contract["entity_keys"])), None)
            if pair is not None and pair["measurement"] is not None:
                value = _get(pair["measurement"], contract["metric"])
                # Symmetric distances/overlap are order-independent. For a signed
                # gap, a/b are the explicitly configured entities, not list roles.
                if pair["entity_keys"] != contract["entity_keys"] and contract["metric"].startswith("signed_box_gaps_px."):
                    boxes = {o["entity_key"]: o["candidate"]["box_xyxy_px"] for o in frame["objects"]
                             if o["candidate"] is not None}
                    a, b = [boxes[e] for e in contract["entity_keys"]]
                    value = b[0] - a[2] if contract["metric"].endswith("a_left_of_b") else b[1] - a[3]
                if value is not None:
                    values[frame["frame_id"]] = value
    else:
        role = contract["metric"].split(".", 1)[0]
        proposal, raw = record["proposal"], record["result"]
        entities = ([proposal["target_entity_key"], proposal["reference_entity_key"]] if role == "relative_motion"
                    else [proposal[role + "_entity_key"]])
        if entities != contract["entity_keys"] or (role == "relative_motion" and proposal["motion_mode"] != "relative"):
            out["reason"] = "motion_role_or_entity_mismatch"
            return out
        region = raw.get(role)
        samples = region if role == "relative_motion" else region["samples"] if region is not None else []
        values = {s["frame_id"]: s["from_seed"]["magnitude_px"] for s in samples if s["status"] == "available"}
        out["seed_frame_id"] = record["frame_ids"][0]
        out["seed_pts_seconds"] = times[record["frame_ids"][0]]
    first, last = contract["frame_range"]
    required = list(range(first, last + 1))
    covered = [f for f in required if f in values]
    out["coverage"] = _coverage(required, covered, times)
    if len(covered) != len(required):
        out["reason"] = "required_measurement_coverage_incomplete"
        return out
    chosen = (first if contract["aggregation"] == "at_frame" else
              min(required, key=values.get) if contract["aggregation"] == "min" else max(required, key=values.get))
    value = values[chosen]
    out.update(status="compared", reason=None, measurement={**_point(chosen, value, times), "unit": contract["unit"],
               "scope": _span(required, times)}, satisfied=OPERATORS[contract["operator"]](value, contract["threshold"]))
    return out


def build_evidence_packages(task, obs, selection, times, candidates, matching, execution, threshold_config=None):
    """One package per original check. No images/experts/models are executed here."""
    thresholds = validate_threshold_config(threshold_config)
    validate_packaging_inputs(task, obs, selection, times, candidates, matching, execution, thresholds)
    packages = []
    for check in obs["candidate_checks"]:
        relevant = [(w, next(c for c in w["checks"] if c["check_id"] == check["id"])) for w in matching["windows"]]
        relevant = [(w, c) for w, c in relevant if c["relevance"] in ("related", "uncertain")]
        relevant.sort(key=lambda pair: (pair[0]["frame_range"][0], pair[0]["frame_range"][1], pair[0]["window_id"]))
        records = [r for r in execution["records"] if r["check_id"] == check["id"]]
        records.sort(key=lambda r: (r["frame_ids"][0], r["frame_ids"][-1], r["id"]))
        images, bindings, windows = {}, [], []
        for window, association in relevant:
            windows.append({"window_id": window["window_id"], "frame_range": list(window["frame_range"]),
                            "pts_range_seconds": [times[f] for f in window["frame_range"]],
                            "relevance": association["relevance"], "reason": association["reason"],
                            "limitations": copy.deepcopy(association["limitations"]),
                            "execution_ids": [r["id"] for r in records if r["window_id"] == window["window_id"]]})
            for image in window["images"]:
                index = image["frame_id"]
                entry = images.setdefault(index, {"frame_id": index, "pts_seconds": times[index],
                                                   "original_file": image["original_file"], "sources": []})
                entry["sources"].append({"window_id": window["window_id"], "original_file": image["original_file"],
                                          "candidates_file": image["candidates_file"]})
            bindings.extend({"window_id": window["window_id"], **copy.deepcopy(b)} for b in window["bindings"]
                            if b["entity_key"] in check["entity_keys"])
        contracts = [c for c in thresholds["comparisons"] if c["check_id"] == check["id"]]
        packages.append({"check_id": check["id"], "check": copy.deepcopy(check), "instruction": task,
                         "requirements": [copy.deepcopy(r) for r in obs["requirements"] if r["id"] in check["requirement_ids"]],
                         "status": "evidence_collected" if relevant else "no_related_windows",
                         "reason": None if relevant else "no_related_or_uncertain_selected_window",
                         "windows": windows, "images": [images[f] for f in sorted(images)],
                         "image_coverage": {**_coverage(range(len(times)), images, times),
                                            "includes_episode_start": 0 in images, "includes_episode_end": len(times) - 1 in images},
                         "bindings": sorted(bindings, key=lambda b: (b["frame_id"], b["window_id"], b["entity_key"])),
                         "measurements": [_summarize_record(r, times) for r in records],
                         "threshold_comparisons": {"status": "configured" if contracts else "not_configured",
                             "reason": None if contracts else "no_explicit_threshold_contract",
                             "comparisons": [_threshold_comparison(c, records, times) for c in contracts]},
                         "limitations": ["Window relevance and candidate bindings are VLM proposals, not check verdicts.",
                                         "Image/measurement coverage does not establish the semantic time scope of a check.",
                                         "Extrema summarize available measurements only; missing intervals are not interpolated.",
                                         "Tracks have independent seed origins; displacements are not joined across executions.",
                                         "2D geometry and local point motion do not establish 3D physical conclusions.",
                                         "Numeric threshold comparisons are conditional measurements, not yes/no check answers."]})
    return {"protocol": PACKAGING_PROTOCOL, "status": "evidence_packaged", "instruction": task,
            "requirements": copy.deepcopy(obs["requirements"]), "checklist": copy.deepcopy(obs["candidate_checks"]),
            "episode": {"frame_range": [0, len(times) - 1], "pts_range_seconds": [times[0], times[-1]]},
            "source": {"candidate_file": CANDIDATE_FILE, "matching_file": MATCHING_FILE, "execution_file": EXECUTION_FILE},
            "threshold_config": thresholds, "packages": packages}
