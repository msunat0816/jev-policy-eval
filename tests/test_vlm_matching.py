"""Pure matching contracts; synthetic proposals do not validate model judgments."""
import copy
import json
import unittest

from test_window_selection import selection_observation
from vlm_matching import (EXPERT_CATALOG, matching_context, parse_window_matching,
                          CANDIDATE_FRAME_POLICY, CANDIDATE_PROTOCOL, validate_candidate_inputs,
                          validate_matching_inputs)
from window_selection import summarize_selection


def matching_observation():
    obs = selection_observation()
    obs["requirements"][0].update(text="Place the apple in the bowl.", subject="apple", reference="bowl")
    obs["candidate_checks"][0].update(
        entity_keys=["apple", "bowl"], question="Is the apple in the bowl at episode end?",
        rationale="Sampled image overlap is not yet verified containment.")
    obs["frames"][0]["visible_facts"] = ["The apple is beside the bowl."]
    obs["frames"][1]["visible_facts"] = ["The apple and bowl overlap in the image."]
    obs["frames"][2]["uncertainties"] = ["The apple is obscured."]
    return obs


def candidate_frames(frame_ids, times=None, *, empty=False, empty_masks=False, size=None):
    size = size or [12, 16]
    height, width = size
    frames = []
    for frame_id in frame_ids:
        pts = times[frame_id] if times is not None else frame_id / 10
        detections = []
        if not empty:
            for i, (name, box) in enumerate((("apple", [1, 2, 6, 8]), ("bowl", [8, 2, 14, 8]))):
                # Standard-library RLE fixture; pure contract tests need no NumPy/OpenCV.
                counts, current, length = [], False, 0
                for y in range(height):
                    for x in range(width):
                        pixel = not empty_masks and box[0] <= x < box[2] and box[1] <= y < box[3]
                        if pixel == current:
                            length += 1
                        else:
                            counts.append(length)
                            length, current = 1, pixel
                counts.append(length)
                detections.append({"candidate_id": f"f{frame_id}:d{i}", "frame_id": frame_id,
                                   "pts_seconds": pts, "class_id": i, "class_name": name, "confidence": .9,
                                   "box_xyxy_px": box, "mask_status": "empty" if empty_masks else "available",
                                   "mask": {"encoding": "rle_row_major", "size": list(size), "counts": counts}})
        frames.append({"frame_id": frame_id, "pts_seconds": pts, "image_size_hw": list(size),
                       "coordinate_system": "original_image_pixels_xyxy",
                       "status": "no_detections" if empty else "candidates", "candidates": detections})
    return frames


def candidate_bundle(selection, times, **kwargs):
    windows = [{"window_id": w["id"], "frame_range": w["frame_range"],
                "frame_ids": list(range(w["frame_range"][0], w["frame_range"][1] + 1))}
               for w in selection["selected_windows"]]
    ids = sorted({i for w in windows for i in w["frame_ids"]})
    return {"protocol": CANDIDATE_PROTOCOL, "expert_id": "object_spatial",
            "status": "candidates_generated" if ids else "no_selected_windows",
            "frame_policy": CANDIDATE_FRAME_POLICY, "binding_status": "unbound",
            "expert_protocol": "object-spatial-evidence-v1" if ids else None,
            "provenance": {"providers": ["CPUExecutionProvider"]} if ids else None,
            "limitations": [], "windows": windows, "frames": candidate_frames(ids, times, **kwargs)}


def window_matching(window_id="w0", first=0, last=2, *, motion=True):
    experts = [{"expert_id": "object_spatial", "entity_keys": ["apple", "bowl"],
                "frame_ids": sorted({first, last}), "reason": "Use supplied candidates for 2D overlap."}]
    if motion:
        experts.append({"expert_id": "short_window_motion", "target_entity_key": "apple",
                        "reference_entity_key": None, "motion_mode": "absolute",
                        "frame_ids": list(range(first, last + 1)),
                        "reason": "Propose local image displacement with the explicit apple seed."})
    bindings = [{"entity_key": entity, "frame_id": frame, "status": "bound",
                 "candidate_id": f"f{frame}:d{i}", "reason": "Visible shape and instruction identify this candidate."}
                for frame in sorted({first, last}) for i, entity in enumerate(("apple", "bowl"))]
    return {"window_id": window_id, "bindings": bindings, "checks": [
        {"check_id": "c0", "relevance": "related", "reason": "The apple and bowl are relevant to containment.",
         "experts": experts, "limitations": ["2D overlap does not establish containment; mask reliability and texture need checking."]},
        {"check_id": "c1", "relevance": "unrelated", "reason": "No drawer evidence is identified in this window.",
         "experts": [], "limitations": []},
    ]}


class VlmMatchingTests(unittest.TestCase):
    def parse(self, value, **kwargs):
        return parse_window_matching(json.dumps(value), {"id": "w0"},
                                     matching_observation()["candidate_checks"], [0, 1, 2], candidate_frames([0, 1, 2]),
                                     **kwargs)

    def test_cleanup_removes_valid_extras_with_audit_and_preserves_required_bindings(self):
        for no_experts in (True, False):
            with self.subTest(no_experts=no_experts):
                value = window_matching()
                if no_experts:
                    value["checks"][0]["experts"] = []
                else:
                    value["bindings"].append(dict(value["bindings"][0], frame_id=1, candidate_id="f1:d0"))
                before = copy.deepcopy(value)
                audit = []
                result = self.parse(value, cleanup_audit=audit)
                expected = [] if no_experts else window_matching()["bindings"]
                self.assertEqual(result["bindings"], expected)
                self.assertEqual(result["checks"], before["checks"])
                self.assertEqual(value, before)
                self.assertEqual(audit[0]["warning"], "unused_bindings_removed")
                self.assertEqual(audit[0]["window_id"], "w0")
                self.assertEqual(audit[0]["original_count"], len(before["bindings"]))
                self.assertEqual(audit[0]["retained_count"], len(expected))
                self.assertEqual(audit[0]["removed_bindings"], before["bindings"] if no_experts else before["bindings"][-1:])
                # The normalized response must pass the downstream strict contract.
                self.assertEqual(self.parse(result), result)

    def test_cleanup_does_not_repair_missing_duplicate_or_invalid_bindings(self):
        for mode in ("missing", "duplicate", "invalid_required", "invalid_extra", "same_relative_candidate"):
            with self.subTest(mode=mode):
                value = window_matching()
                value["bindings"].append(dict(value["bindings"][0], frame_id=1, candidate_id="f1:d0"))
                if mode == "missing": value["bindings"].pop(0)
                elif mode == "duplicate": value["bindings"].append(copy.deepcopy(value["bindings"][0]))
                elif mode == "invalid_required": value["bindings"][0]["candidate_id"] = "f0:d999"
                elif mode == "invalid_extra": value["bindings"][-1]["candidate_id"] = "f1:d999"
                else:
                    value["checks"][0]["experts"][1].update(motion_mode="relative", reference_entity_key="bowl")
                    value["bindings"][1]["candidate_id"] = "f0:d0"
                audit = []
                with self.assertRaises(ValueError): self.parse(value, cleanup_audit=audit)
                self.assertEqual(audit, [])

    def test_cleanup_leaves_valid_empty_and_shared_bindings_unchanged(self):
        for no_experts in (True, False):
            with self.subTest(no_experts=no_experts):
                value = window_matching()
                if no_experts:
                    value["checks"][0]["experts"] = []
                    value["bindings"] = []
                audit = []
                self.assertEqual(self.parse(value, cleanup_audit=audit), value)
                self.assertEqual(audit, [])

    def test_empty_frames_fill_only_required_omissions_and_preserve_model_bindings(self):
        value = window_matching()
        supplied = copy.deepcopy(value["bindings"][:2])
        value["bindings"] = supplied
        frames = candidate_frames([0, 1]) + candidate_frames([2], empty=True)
        audit = []
        result = parse_window_matching(json.dumps(value), {"id": "w0"}, matching_observation()["candidate_checks"],
                                       [0, 1, 2], frames, cleanup_audit=audit)
        self.assertEqual(result["bindings"][:2], supplied)
        self.assertEqual(result["checks"], value["checks"])
        self.assertEqual([(b["entity_key"], b["frame_id"], b["status"], b["candidate_id"])
                          for b in result["bindings"][2:]], [("apple", 2, "missing", None), ("bowl", 2, "missing", None)])
        self.assertEqual(audit[0]["evidence"], [{"frame_id": 2, "status": "no_detections", "candidate_count": 0}])
        self.assertEqual(audit[0]["added_bindings"], result["bindings"][2:])
        self.assertEqual(parse_window_matching(json.dumps(result), {"id": "w0"}, matching_observation()["candidate_checks"],
                                               [0, 1, 2], frames), result)

    def test_empty_frame_repairs_are_atomic_and_do_not_relax_invalid_inputs(self):
        for mode in ("strict", "nonempty", "absent_inventory", "duplicate", "invalid_candidate", "bad_status"):
            with self.subTest(mode=mode):
                value = window_matching()
                value["bindings"] = []
                frames = candidate_frames([0, 1, 2], empty=True)
                if mode == "nonempty": frames[2] = candidate_frames([2])[0]
                elif mode == "absent_inventory": frames.pop()
                elif mode == "duplicate":
                    binding = {"entity_key": "apple", "frame_id": 0, "status": "missing", "candidate_id": None, "reason": "No detections."}
                    value["bindings"] = [binding, dict(binding)]
                elif mode == "invalid_candidate": value["bindings"] = window_matching()["bindings"][:1]
                elif mode == "bad_status": frames[2]["status"] = "candidates"
                audit = []
                with self.assertRaises(ValueError):
                    parse_window_matching(json.dumps(value), {"id": "w0"}, matching_observation()["candidate_checks"],
                                          [0, 1, 2], frames, cleanup_audit=None if mode == "strict" else audit)
                self.assertEqual(audit, [])

    def test_motion_repairs_only_first_frame_and_deduplicates_shared_requirements(self):
        obs = matching_observation()
        obs["candidate_checks"][1]["entity_keys"] = ["apple", "bowl"]
        value = window_matching()
        motion = value["checks"][0]["experts"][1]
        motion.update(motion_mode="relative", reference_entity_key="bowl")
        value["checks"][0]["experts"] = [motion]
        value["checks"][1].update(relevance="related", experts=[copy.deepcopy(motion)])
        value["bindings"] = []
        audit = []
        result = parse_window_matching(json.dumps(value), {"id": "w0"}, obs["candidate_checks"], [0, 1, 2],
                                       candidate_frames([0], empty=True) + candidate_frames([1, 2]), cleanup_audit=audit)
        self.assertEqual([(b["entity_key"], b["frame_id"]) for b in result["bindings"]], [("apple", 0), ("bowl", 0)])
        self.assertEqual(len(audit[0]["added_bindings"]), 2)

    def test_empty_frame_additions_and_unused_removals_share_one_normalized_response(self):
        value = window_matching()
        value["bindings"] = [{"entity_key": "apple", "frame_id": 1, "status": "missing", "candidate_id": None,
                              "reason": "No detections."}]
        audit = []
        result = parse_window_matching(json.dumps(value), {"id": "w0"}, matching_observation()["candidate_checks"],
                                       [0, 1, 2], candidate_frames([0, 1, 2], empty=True), cleanup_audit=audit)
        self.assertEqual(len(result["bindings"]), 4)
        self.assertEqual([event["warning"] for event in audit],
                         ["unused_bindings_removed", "missing_bindings_added_for_empty_frames"])
        self.assertEqual(audit[0]["removed_bindings"], value["bindings"])
        self.assertEqual(len(audit[1]["added_bindings"]), 4)

    def test_preserves_complete_checklist_and_proposals_without_completion_verdicts(self):
        value = window_matching()
        self.assertEqual(self.parse(value), value)
        context = matching_context("Original instruction", matching_observation(), {"id": "w0", "frame_range": [0, 2]},
                                   candidate_frames([0, 1, 2]))
        self.assertEqual(context["instruction"], "Original instruction")
        self.assertEqual(context["checklist"], matching_observation()["candidate_checks"])
        self.assertEqual(set(context["expert_catalog"]), {"object_spatial", "short_window_motion"})
        self.assertEqual(context["expert_catalog"]["object_spatial"]["execution_status"],
                         "candidates_precomputed_geometry_not_scheduled")
        self.assertEqual(context["expert_catalog"]["short_window_motion"]["execution_status"], "implemented_not_scheduled")
        self.assertNotIn("mask", context["candidate_frames"][0]["candidates"][0])
        context["expert_catalog"].clear()
        self.assertEqual(len(EXPERT_CATALOG), 2)

    def test_related_checks_without_tool_coverage_are_retained(self):
        value = window_matching()
        value["checks"][0]["experts"] = []
        value["bindings"] = []
        value["checks"][0]["limitations"] = ["No supplied tool can establish physical containment."]
        self.assertEqual(self.parse(value), value)

    def test_uncertain_or_uncovered_checks_need_an_explicit_limitation(self):
        for relevance, experts in (("uncertain", window_matching()["checks"][0]["experts"]), ("related", [])):
            with self.subTest(relevance=relevance):
                value = window_matching()
                value["checks"][0].update(relevance=relevance, experts=experts, limitations=[])
                if not experts:
                    value["bindings"] = []
                with self.assertRaises(ValueError):
                    self.parse(value)
                value["checks"][0]["limitations"] = ["Visibility or tool coverage is insufficient."]
                self.assertEqual(self.parse(value), value)

    def test_missing_duplicate_and_invented_check_ids_are_rejected(self):
        for mode in ("missing", "duplicate", "invented"):
            with self.subTest(mode=mode):
                value = window_matching()
                if mode == "missing":
                    value["checks"].pop()
                elif mode == "duplicate":
                    value["checks"].append(copy.deepcopy(value["checks"][0]))
                else:
                    value["checks"][0]["check_id"] = "c999"
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_old_or_invented_expert_names_and_duplicate_calls_are_rejected(self):
        for expert_id in ("layout", "motion", "depth_oracle", "duplicate"):
            with self.subTest(expert_id=expert_id):
                value = window_matching()
                if expert_id == "duplicate":
                    value["checks"][0]["experts"].append(copy.deepcopy(value["checks"][0]["experts"][0]))
                else:
                    value["checks"][0]["experts"][0]["expert_id"] = expert_id
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_expert_entities_must_be_distinct_declared_check_entities(self):
        for entities in ([], ["apple", "apple"], ["invented_gripper"], ["drawer"]):
            with self.subTest(entities=entities):
                value = window_matching()
                value["checks"][0]["experts"][0]["entity_keys"] = entities
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_frames_must_be_supplied_and_motion_interval_must_be_continuous(self):
        for frames in ([], [3], [True, 2], [2, 1], [1, 1], [0, 2], [1]):
            with self.subTest(frames=frames):
                value = window_matching()
                value["checks"][0]["experts"][1]["frame_ids"] = frames
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_unrelated_checks_cannot_schedule_experts(self):
        value = window_matching()
        value["checks"][0]["relevance"] = "unrelated"
        with self.assertRaises(ValueError):
            self.parse(value)

    def test_wrong_window_prose_wrapped_json_or_completion_fields_are_rejected(self):
        value = window_matching()
        value["window_id"] = "w1"
        with self.assertRaises(ValueError):
            self.parse(value)
        with self.assertRaises(ValueError):
            parse_window_matching("```json\n" + json.dumps(window_matching()) + "\n```", {"id": "w0"},
                                  matching_observation()["candidate_checks"], [0, 1, 2], candidate_frames([0, 1, 2]))
        for location in ("top", "check", "expert"):
            with self.subTest(location=location):
                value = window_matching()
                target = value if location == "top" else value["checks"][0]
                if location == "expert":
                    target = target["experts"][0]
                target["verdict"] = "yes"
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_selected_windows_are_validated_against_all_original_decisions(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        self.assertEqual(validate_matching_inputs("Task", matching_observation(), selection, [i/10 for i in range(5)]),
                         selection["selected_windows"])
        for mode in ("unselected", "wrong_id", "missing_check"):
            with self.subTest(mode=mode):
                obs, changed = matching_observation(), copy.deepcopy(selection)
                if mode == "unselected":
                    changed["selected_windows"].append(changed["windows"][1])
                elif mode == "wrong_id":
                    changed["windows"][0]["id"] = "../unsafe"
                else:
                    obs["candidate_checks"].pop()
                with self.assertRaises(ValueError):
                    validate_matching_inputs("Task", obs, changed, [i/10 for i in range(5)])

    def test_missing_nonfinite_or_unordered_timestamps_are_rejected(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        for times in ([], [0, 1], [0, 1, float("nan"), 3, 4], [0, 2, 1, 3, 4], [0, True, 2, 3, 4]):
            with self.subTest(times=times):
                with self.assertRaises(ValueError):
                    validate_matching_inputs("Task", matching_observation(), selection, times)

    def test_missing_binding_retains_expert_and_specific_reason(self):
        value = window_matching()
        value["bindings"][0].update(status="missing", candidate_id=None, reason="Two apples are visually indistinguishable.")
        result = self.parse(value)
        self.assertEqual(result, value)
        self.assertEqual(len(result["checks"][0]["experts"]), 2)

    def test_bindings_reject_unknown_cross_frame_duplicate_and_omitted_references(self):
        for mode in ("invented_candidate", "wrong_frame", "unknown_entity", "duplicate", "omitted", "extra",
                     "missing_with_candidate", "bound_without_candidate", "blank_reason", "boolean_frame", "invalid_status"):
            with self.subTest(mode=mode):
                value = window_matching()
                binding = value["bindings"][0]
                if mode == "invented_candidate": binding["candidate_id"] = "f0:d999"
                elif mode == "wrong_frame": binding["candidate_id"] = "f2:d0"
                elif mode == "unknown_entity": binding["entity_key"] = "new_apple"
                elif mode == "duplicate": value["bindings"].append(copy.deepcopy(binding))
                elif mode == "omitted": value["bindings"].pop()
                elif mode == "extra": value["bindings"].append(dict(binding, frame_id=1, candidate_id="f1:d0"))
                elif mode == "missing_with_candidate": binding["status"] = "missing"
                elif mode == "bound_without_candidate": binding["candidate_id"] = None
                elif mode == "blank_reason": binding["reason"] = " "
                elif mode == "boolean_frame": binding["frame_id"] = False
                else: binding["status"] = "unknown"
                with self.assertRaises(ValueError): self.parse(value)

    def test_relative_motion_requires_explicit_distinct_target_and_reference(self):
        value = window_matching()
        motion = value["checks"][0]["experts"][1]
        motion.update(motion_mode="relative", reference_entity_key="bowl")
        self.assertEqual(self.parse(value), value)
        for field, bad in (("reference_entity_key", None), ("reference_entity_key", "apple"),
                           ("reference_entity_key", "drawer"), ("target_entity_key", ""),
                           ("motion_mode", "auto")):
            with self.subTest(field=field, bad=bad):
                changed = copy.deepcopy(value)
                changed["checks"][0]["experts"][1][field] = bad
                with self.assertRaises(ValueError): self.parse(changed)
        value["bindings"][1]["candidate_id"] = "f0:d0"
        with self.assertRaisesRegex(ValueError, "same candidate"): self.parse(value)

    def test_motion_binds_only_explicit_seed_frame_with_no_entity_list_role_guess(self):
        value = window_matching()
        value["checks"][0]["experts"].pop(0)
        value["bindings"] = value["bindings"][:1]
        self.assertEqual(self.parse(value), value)
        motion = value["checks"][0]["experts"][0]
        motion["entity_keys"] = ["apple", "bowl"]
        with self.assertRaises(ValueError): self.parse(value)
        del motion["entity_keys"]
        del motion["target_entity_key"]
        with self.assertRaises(ValueError): self.parse(value)

    def test_absolute_motion_rejects_reference_and_relative_missing_seed_stays_explicit(self):
        value = window_matching()
        motion = value["checks"][0]["experts"][1]
        motion["reference_entity_key"] = "bowl"
        with self.assertRaises(ValueError): self.parse(value)
        motion["motion_mode"] = "relative"
        value["bindings"][1].update(status="missing", candidate_id=None, reason="Reference is occluded.")
        self.assertEqual(self.parse(value), value)

    def test_shared_binding_is_reused_across_checks_without_duplicates(self):
        obs = matching_observation()
        obs["candidate_checks"][1]["entity_keys"] = ["apple", "bowl"]
        value = window_matching()
        value["checks"][1].update(relevance="related", experts=copy.deepcopy(value["checks"][0]["experts"]),
                                  limitations=["Partial image evidence only."])
        result = parse_window_matching(json.dumps(value), {"id": "w0"}, obs["candidate_checks"],
                                       [0, 1, 2], candidate_frames([0, 1, 2]))
        self.assertEqual(len(result["bindings"]), 4)

    def test_no_detection_records_explicit_missing_and_empty_mask_can_still_bind_identity(self):
        value = window_matching()
        for binding in value["bindings"]:
            binding.update(status="missing", candidate_id=None, reason="No candidate was detected in this frame.")
        for kwargs in ({"empty": True}, {"empty_masks": True}):
            parsed = parse_window_matching(json.dumps(value), {"id": "w0"}, matching_observation()["candidate_checks"],
                                            [0, 1, 2], candidate_frames([0, 1, 2], **kwargs))
            self.assertEqual(parsed, value)
        value = window_matching()
        self.assertEqual(parse_window_matching(json.dumps(value), {"id": "w0"}, matching_observation()["candidate_checks"],
                                                [0, 1, 2], candidate_frames([0, 1, 2], empty_masks=True)), value)

    def test_candidate_inventory_rejects_invalid_window_frame_pts_mask_and_identifier(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        times = [0, .1, .2, .3, .4]
        valid = candidate_bundle(selection, times)
        self.assertEqual(set(validate_candidate_inputs(valid, selection["selected_windows"], times)), {0, 1, 2})
        for mode in ("window", "missing_frame", "duplicate_frame", "frame_pts", "candidate_pts", "id", "duplicate_id",
                     "mask_size", "mask_counts", "mask_status", "box", "score", "frame_id_bool"):
            with self.subTest(mode=mode):
                changed = copy.deepcopy(valid)
                frame = changed["frames"][0]
                candidate = frame["candidates"][0]
                if mode == "window": changed["windows"].pop()
                elif mode == "missing_frame": changed["frames"].pop()
                elif mode == "duplicate_frame": changed["frames"].append(copy.deepcopy(frame))
                elif mode == "frame_pts": frame["pts_seconds"] = .9
                elif mode == "candidate_pts": candidate["pts_seconds"] = .9
                elif mode == "id": candidate["candidate_id"] = "f1:d0"
                elif mode == "duplicate_id": frame["candidates"].append(copy.deepcopy(candidate))
                elif mode == "mask_size": candidate["mask"]["size"] = [1, 1]
                elif mode == "mask_counts": candidate["mask"]["counts"] = [1]
                elif mode == "mask_status": candidate["mask_status"] = "empty"
                elif mode == "box": candidate["box_xyxy_px"] = [-1, 0, 10, 10]
                elif mode == "score": candidate["confidence"] = float("nan")
                else: frame["frame_id"] = False
                with self.assertRaises(ValueError):
                    validate_candidate_inputs(changed, selection["selected_windows"], times)


if __name__ == "__main__":
    unittest.main()
