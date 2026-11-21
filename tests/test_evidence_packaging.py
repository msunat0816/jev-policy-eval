"""Offline per-check packaging and exact numeric contracts; never paid models."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_expert_execution as execution_tests
from test_expert_execution import saved_matching
from test_vlm_matching import candidate_bundle, matching_observation, window_matching
from test_vlm_matching_integration import ev
from window_selection import summarize_selection

if ev is not None:
    import evidence_packaging as ep
    from experts.masks import encode_mask


@unittest.skipIf(ev is None, "OpenCV and NumPy required for stage-4 fixtures")
class EvidencePackagingTests(unittest.TestCase):
    def inputs(self, *, full_spatial=False, **kwargs):
        helper = execution_tests.ExpertExecutionTests()
        inputs = helper.inputs(**kwargs)
        if full_spatial:
            value = window_matching(motion=False)
            value["checks"][0]["experts"][0]["frame_ids"] = [0, 1, 2]
            value["bindings"] += [{**b, "frame_id": 1, "candidate_id": b["candidate_id"].replace("f0:", "f1:")}
                                  for b in value["bindings"] if b["frame_id"] == 0]
            obs, selection, times, candidates, _ = inputs
            inputs = obs, selection, times, candidates, saved_matching("Task", obs, selection, times, [value])
        return inputs, helper.run_execution(inputs)

    def build(self, inputs, execution, comparisons=None):
        obs, selection, times, candidates, matching = inputs
        config = {"protocol": ep.THRESHOLD_PROTOCOL, "comparisons": comparisons or []}
        return ep.build_evidence_packages("Task", obs, selection, times, candidates, matching, execution, config)

    def contract(self, **overrides):
        return {"id": "t0", "window_id": "w0", "check_id": "c0", "expert_id": "object_spatial",
                "entity_keys": ["apple", "bowl"], "metric": "box_edge_distance_px", "unit": "px",
                "aggregation": "at_frame", "frame_range": [2, 2], "operator": "<=", "threshold": 2, **overrides}

    def test_one_package_per_check_keeps_images_pts_sources_and_empty_check(self):
        inputs, execution = self.inputs()
        before = copy.deepcopy((inputs, execution))
        with patch.object(ev, "post") as post, patch.object(ev, "ask_jev") as jev, \
                patch.object(ev, "ObjectSpatialExpert") as detector, patch("expert_execution.ShortWindowMotionExpert") as motion:
            result = self.build(inputs, execution)
        for method in (post, jev, detector, motion):
            method.assert_not_called()
        self.assertEqual((inputs, execution), before)
        first, second = result["packages"]
        self.assertEqual([p["check_id"] for p in result["packages"]], ["c0", "c1"])
        self.assertEqual(first["check"], inputs[0]["candidate_checks"][0])
        self.assertEqual([f["frame_id"] for f in first["images"]], [0, 1, 2])
        self.assertEqual([f["pts_seconds"] for f in first["images"]], [0, .09, .25])
        self.assertEqual(first["images"][1]["sources"][0]["candidates_file"], "matching_frames/w0/1-candidates.jpg")
        self.assertEqual(first["bindings"][0]["source"]["request_file"], "03-match-w0.json")
        self.assertEqual(first["image_coverage"]["missing_ranges"],
                         [{"frame_range": [3, 4], "pts_range_seconds": [.4, .6], "frame_count": 2}])
        self.assertFalse(first["image_coverage"]["includes_episode_end"])
        self.assertEqual(second["status"], "no_related_windows")
        self.assertEqual(second["reason"], "no_related_or_uncertain_selected_window")
        self.assertEqual(second["images"], [])
        self.assertEqual(second["measurements"], [])
        self.assertNotIn("answer", first)
        self.assertNotIn("score", result)
        json.dumps(result, allow_nan=False)

    def test_spatial_extrema_have_exact_frames_units_and_sparse_coverage(self):
        inputs, _ = self.inputs(full_spatial=True)
        for index, left in enumerate((8, 7, 9)):
            candidate = inputs[3]["frames"][index]["candidates"][1]
            candidate["box_xyxy_px"] = [left, 2, left + 6, 8]
            mask = ev.np.zeros((12, 16), bool)
            mask[2:8, left:left + 6] = True
            candidate["mask"] = encode_mask(mask)
        execution = execution_tests.ExpertExecutionTests().run_execution(inputs)
        pair = self.build(inputs, execution)["packages"][0]["measurements"][0]["summary"]["pairs"][0]
        metric = pair["metrics"]["box_edge_distance_px"]
        self.assertEqual(metric["unit"], "px")
        self.assertEqual(metric["min"], {"frame_id": 1, "pts_seconds": .09, "value": 1})
        self.assertEqual(metric["max"], {"frame_id": 2, "pts_seconds": .25, "value": 3})
        self.assertEqual(metric["last_available"], metric["max"])
        self.assertEqual(pair["metrics"]["box_iou"]["unit"], "ratio")
        self.assertEqual(pair["metrics"]["mask_intersection_px"]["unit"], "px^2")
        sparse_inputs, sparse_execution = self.inputs()
        measurement = self.build(sparse_inputs, sparse_execution)["packages"][0]["measurements"][0]
        self.assertEqual([r["frame_range"] for r in measurement["proposed_frame_ranges"]], [[0, 0], [2, 2]])
        self.assertEqual(measurement["summary"]["scope"], "proposed_frames_only_no_interpolation")

    def test_no_raw_motion_arrays_masks_or_frame_arrays_leak_into_summaries(self):
        inputs, execution = self.inputs()
        packet = self.build(inputs, execution)["packages"][0]
        def inspect(value):
            if isinstance(value, dict):
                self.assertTrue(set(value).isdisjoint({"samples", "counts", "mask", "frames", "frame_ids"}))
                for child in value.values(): inspect(child)
            elif isinstance(value, list):
                for child in value: inspect(child)
        inspect(packet)
        self.assertEqual(packet["measurements"][0]["raw_result_ref"],
                         {"file": "04-expert-execution.json", "record_id": "execution-0000"})

    def test_missing_masks_keep_box_summary_and_null_mask_summary(self):
        inputs, execution = self.inputs(empty_masks=True)
        packet = self.build(inputs, execution)["packages"][0]
        metrics = packet["measurements"][0]["summary"]["pairs"][0]["metrics"]
        self.assertEqual(metrics["box_edge_distance_px"]["max"]["value"], 2)
        self.assertIsNone(metrics["mask_iou"]["max"])
        self.assertEqual(metrics["mask_iou"]["coverage"]["available_frame_count"], 0)
        motion = packet["measurements"][1]
        self.assertEqual(motion["execution_status"], "not_executed")
        self.assertIsNone(motion["summary"])
        self.assertEqual(motion["missing"][0]["reason"], "empty_mask")

    def test_actual_motion_summary_is_per_role_and_from_its_original_seed(self):
        helper = execution_tests.ExpertExecutionTests()
        inputs, cap, _ = helper.motion_inputs()
        execution = helper.run_execution(inputs, cap)
        summary = self.build(inputs, execution)["packages"][0]["measurements"][0]["summary"]
        self.assertEqual(summary["target"]["seed_frame_id"], 20)
        self.assertEqual(summary["target"]["coverage"]["available_frame_count"], 5)
        self.assertEqual(summary["target"]["max_from_seed"]["frame_id"], 24)
        self.assertAlmostEqual(summary["target"]["max_from_seed"]["magnitude_px"], 12, delta=.3)
        self.assertAlmostEqual(summary["reference"]["max_from_seed"]["magnitude_px"], 4, delta=.3)
        self.assertAlmostEqual(summary["relative_motion"]["max_from_seed"]["magnitude_px"], 8, delta=.4)
        self.assertEqual(summary["target"]["first_available_from_seed"]["magnitude_px"], 0)
        self.assertTrue(summary["target"]["requested_end_available"])

    def test_tracking_loss_does_not_turn_last_available_into_window_end(self):
        helper = execution_tests.ExpertExecutionTests()
        inputs, cap, images = helper.motion_inputs(relative=False)
        for index in (22, 23, 24): images[index][:] = 0
        execution = helper.run_execution(inputs, cap)
        summary = self.build(inputs, execution)["packages"][0]["measurements"][0]["summary"]["target"]
        self.assertFalse(summary["requested_end_available"])
        self.assertLess(summary["last_available_from_seed"]["frame_id"], 24)
        self.assertEqual(summary["interruptions"], execution["records"][0]["result"]["target"]["interruptions"])
        self.assertEqual(summary["coverage"]["missing_ranges"][-1]["frame_range"][-1], 24)
        self.assertTrue(summary["missing_spans"])

    def test_missing_reference_keeps_absolute_summary_and_relative_missing(self):
        helper = execution_tests.ExpertExecutionTests()
        inputs, cap, _ = helper.motion_inputs()
        inputs[-1]["windows"][0]["bindings"][1].update(status="missing", candidate_id=None, reason="Reference identity unclear.")
        execution = helper.run_execution(inputs, cap)
        contract = self.contract(expert_id="short_window_motion", metric="relative_motion.from_seed.magnitude_px",
                                 aggregation="max", frame_range=[20, 24], threshold=10)
        packet = self.build(inputs, execution, [contract])["packages"][0]
        summary = packet["measurements"][0]["summary"]
        self.assertEqual(summary["target"]["coverage"]["available_frame_count"], 5)
        self.assertIsNone(summary["reference"])
        self.assertEqual(summary["relative_motion"]["coverage"]["available_frame_count"], 0)
        self.assertIsNone(summary["relative_motion"]["max_from_seed"])
        self.assertEqual(packet["threshold_comparisons"]["comparisons"][0]["status"], "not_comparable")

    def test_overlap_images_are_deduplicated_but_binding_sources_are_preserved(self):
        obs = matching_observation()
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p3"})
        times = [0, .09, .25, .4, .6]
        values = [window_matching("w0", 0, 2), window_matching("w1", 2, 4)]
        values[1]["checks"][0].update(relevance="uncertain", limitations=["Identity uncertain."])
        values[1]["bindings"][0]["candidate_id"] = "f2:d1"
        inputs = obs, selection, times, candidate_bundle(selection, times), saved_matching("Task", obs, selection, times, values)
        execution = execution_tests.ExpertExecutionTests().run_execution(inputs)
        packet = self.build(inputs, execution)["packages"][0]
        self.assertEqual([i["frame_id"] for i in packet["images"]], [0, 1, 2, 3, 4])
        self.assertEqual(len(packet["images"][2]["sources"]), 2)
        self.assertEqual(packet["windows"][1]["relevance"], "uncertain")
        bindings = [b for b in packet["bindings"] if b["frame_id"] == 2 and b["entity_key"] == "apple"]
        self.assertEqual([b["candidate_id"] for b in bindings], ["f2:d0", "f2:d1"])
        self.assertEqual(len(packet["measurements"]), 4)
        tracks = [m for m in packet["measurements"] if m["expert_id"] == "short_window_motion"]
        self.assertEqual([m["summary"]["target"]["seed_frame_id"] for m in tracks], [0, 2])
        self.assertTrue(packet["image_coverage"]["includes_episode_end"])

    def test_related_photo_only_check_is_retained_without_expert_measurements(self):
        inputs, _ = self.inputs()
        association = inputs[-1]["windows"][0]["checks"][1]
        association.update(relevance="related", reason="Images may show the drawer.", limitations=["No expert covers drawer state."])
        execution = execution_tests.ExpertExecutionTests().run_execution(inputs)
        packet = self.build(inputs, execution)["packages"][1]
        self.assertEqual(len(packet["images"]), 3)
        self.assertEqual(packet["measurements"], [])
        self.assertEqual(packet["status"], "evidence_collected")
        self.assertEqual(packet["windows"][0]["limitations"], ["No expert covers drawer state."])

    def test_empty_selection_still_outputs_every_original_check(self):
        obs = matching_observation()
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p0", "w1": "p1"})
        times = [i/10 for i in range(5)]
        inputs = obs, selection, times, candidate_bundle(selection, times), saved_matching("Task", obs, selection, times)
        execution = execution_tests.ExpertExecutionTests().run_execution(inputs)
        result = self.build(inputs, execution)
        self.assertEqual(len(result["packages"]), len(obs["candidate_checks"]))
        self.assertTrue(all(p["status"] == "no_related_windows" and p["images"] == [] for p in result["packages"]))

    def test_threshold_comparisons_use_exact_boundaries_and_all_four_operators(self):
        inputs, execution = self.inputs()
        for op, expected in (("<", False), ("<=", True), (">", False), (">=", True)):
            with self.subTest(operator=op):
                packet = self.build(inputs, execution, [self.contract(operator=op)])["packages"][0]
                comparison = packet["threshold_comparisons"]["comparisons"][0]
                self.assertEqual(comparison["status"], "compared")
                self.assertIs(comparison["satisfied"], expected)
                self.assertEqual(comparison["measurement"]["value"], 2)
                self.assertNotIn("answer", packet)

    def test_interval_extrema_require_all_frames_not_just_proposed_endpoints(self):
        contract = self.contract(aggregation="max", frame_range=[0, 2])
        sparse, sparse_execution = self.inputs()
        comparison = self.build(sparse, sparse_execution, [contract])["packages"][0]["threshold_comparisons"]["comparisons"][0]
        self.assertEqual(comparison["reason"], "required_measurement_coverage_incomplete")
        self.assertIsNone(comparison["satisfied"])
        self.assertEqual(comparison["coverage"]["missing_ranges"][0]["frame_range"], [1, 1])
        full, full_execution = self.inputs(full_spatial=True)
        comparison = self.build(full, full_execution, [contract])["packages"][0]["threshold_comparisons"]["comparisons"][0]
        self.assertTrue(comparison["satisfied"])
        self.assertEqual(comparison["measurement"]["scope"]["frame_range"], [0, 2])

    def test_threshold_units_and_metrics_are_checked_without_implicit_conversion(self):
        inputs, execution = self.inputs()
        contracts = [self.contract(id="ratio", metric="box_iou", unit="ratio", threshold=0),
                     self.contract(id="area", metric="mask_intersection_px", unit="px^2", threshold=0),
                     self.contract(id="gap", metric="signed_box_gaps_px.a_left_of_b", threshold=2)]
        comparisons = self.build(inputs, execution, contracts)["packages"][0]["threshold_comparisons"]["comparisons"]
        self.assertTrue(all(c["status"] == "compared" and c["satisfied"] for c in comparisons))
        for overrides in ({"unit": "mm"}, {"metric": "physical_stability"}, {"threshold": float("nan")},
                          {"threshold": True}, {"operator": "=="}, {"aggregation": "mean"}, {"frame_range": [0, 2]}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.build(inputs, execution, [self.contract(**overrides)])

    def test_motion_threshold_is_from_single_seed_and_does_not_switch_role(self):
        helper = execution_tests.ExpertExecutionTests()
        inputs, cap, _ = helper.motion_inputs()
        execution = helper.run_execution(inputs, cap)
        contracts = [self.contract(id="target", expert_id="short_window_motion", entity_keys=["apple"],
                                  metric="target.from_seed.magnitude_px", aggregation="max", frame_range=[20, 24], threshold=13),
                     self.contract(id="relative", expert_id="short_window_motion", metric="relative_motion.from_seed.magnitude_px",
                                   aggregation="max", frame_range=[20, 24], threshold=9)]
        comparisons = self.build(inputs, execution, contracts)["packages"][0]["threshold_comparisons"]["comparisons"]
        self.assertTrue(all(c["satisfied"] for c in comparisons))
        self.assertTrue(all(c["seed_frame_id"] == 20 for c in comparisons))
        mismatch = self.contract(expert_id="short_window_motion", entity_keys=["bowl"], metric="target.from_seed.magnitude_px",
                                 frame_range=[24, 24])
        comparison = self.build(inputs, execution, [mismatch])["packages"][0]["threshold_comparisons"]["comparisons"][0]
        self.assertEqual(comparison["reason"], "motion_role_or_entity_mismatch")

    def test_spatial_threshold_uses_explicit_entity_order_for_signed_gaps(self):
        inputs, execution = self.inputs()
        contracts = [self.contract(id="symmetric", entity_keys=["bowl", "apple"]),
                     self.contract(id="signed", entity_keys=["bowl", "apple"],
                                   metric="signed_box_gaps_px.a_left_of_b", threshold=-13)]
        comparisons = self.build(inputs, execution, contracts)["packages"][0]["threshold_comparisons"]["comparisons"]
        self.assertEqual([c["measurement"]["value"] for c in comparisons], [2, -13])
        self.assertTrue(all(c["satisfied"] for c in comparisons))

    def test_unconfigured_or_unavailable_threshold_never_becomes_false(self):
        inputs, execution = self.inputs()
        packet = self.build(inputs, execution)["packages"][0]
        self.assertEqual(packet["threshold_comparisons"]["reason"], "no_explicit_threshold_contract")
        no_expert = self.contract(window_id="w1", frame_range=[4, 4])
        comparison = self.build(inputs, execution, [no_expert])["packages"][0]["threshold_comparisons"]["comparisons"][0]
        self.assertEqual(comparison["reason"], "no_matching_expert_execution")
        self.assertIsNone(comparison["satisfied"])
        empty, empty_execution = self.inputs(empty_masks=True)
        comparison = self.build(empty, empty_execution, [self.contract(metric="mask_iou", unit="ratio")])["packages"][0]["threshold_comparisons"]["comparisons"][0]
        self.assertIsNone(comparison["satisfied"])

    def test_motion_threshold_never_joins_independent_overlapping_tracks(self):
        helper = execution_tests.ExpertExecutionTests()
        base, cap, _ = helper.motion_inputs(relative=False)
        obs, _, times, candidates, _ = base
        obs["frames"].insert(1, dict(obs["frames"][0], frame=22))
        selection = summarize_selection([(20, 22), (22, 24)], {"w0": "p2", "w1": "p3"})
        candidates["windows"] = [{"window_id": w["id"], "frame_range": w["frame_range"],
                                  "frame_ids": list(range(w["frame_range"][0], w["frame_range"][1] + 1))}
                                 for w in selection["selected_windows"]]
        mask = ev.np.zeros((96, 160), bool)
        mask[16:48, 21:53] = True
        candidates["frames"][2]["candidates"][0].update(mask=encode_mask(mask), box_xyxy_px=[21, 16, 53, 48])
        values = [window_matching("w0", 20, 22), window_matching("w1", 22, 24)]
        for value, first in zip(values, (20, 22)):
            value["checks"][0]["experts"] = value["checks"][0]["experts"][1:]
            value["bindings"] = [b for b in value["bindings"] if b["frame_id"] == first and b["entity_key"] == "apple"]
        inputs = obs, selection, times, candidates, saved_matching("Task", obs, selection, times, values)
        execution = helper.run_execution(inputs, cap)
        contract = self.contract(expert_id="short_window_motion", metric="target.from_seed.magnitude_px",
                                 entity_keys=["apple"], aggregation="max", frame_range=[20, 24], threshold=13)
        packet = self.build(inputs, execution, [contract])["packages"][0]
        self.assertEqual([m["summary"]["target"]["seed_frame_id"] for m in packet["measurements"]], [20, 22])
        self.assertTrue(all(m["summary"]["target"]["requested_end_available"] for m in packet["measurements"]))
        comparison = packet["threshold_comparisons"]["comparisons"][0]
        self.assertEqual(comparison["reason"], "required_measurement_coverage_incomplete")
        self.assertEqual(comparison["coverage"]["available_frame_count"], 3)
        self.assertIsNone(comparison["satisfied"])

    def test_corrupt_execution_references_frames_numbers_and_coverage_are_rejected(self):
        for mode in ("check", "source", "record_count", "frame", "pts", "nan", "status"):
            with self.subTest(mode=mode):
                inputs, execution = self.inputs()
                record = execution["records"][0]
                if mode == "check": record["check_id"] = "c1"
                elif mode == "source": record["bindings"][0]["source"]["request_file"] = "wrong.json"
                elif mode == "record_count": execution["records"].pop()
                elif mode == "frame": record["result"]["frames"][0]["frame_id"] = 99
                elif mode == "pts": record["result"]["frames"][0]["pts_seconds"] = 99
                elif mode == "nan": record["result"]["frames"][0]["pairs"][0]["measurement"]["box_iou"] = float("nan")
                else: record["status"] = "missing"
                with self.assertRaises(ValueError): self.build(inputs, execution)

    def test_motion_corruption_cannot_fabricate_complete_or_resumed_tracks(self):
        helper = execution_tests.ExpertExecutionTests()
        for mode in ("seed", "missing_sample", "status", "vector"):
            with self.subTest(mode=mode):
                inputs, cap, _ = helper.motion_inputs(relative=False)
                execution = helper.run_execution(inputs, cap)
                target = execution["records"][0]["result"]["target"]
                if mode == "seed": target["seed_frame_id"] = 21
                elif mode == "missing_sample": target["samples"].pop()
                elif mode == "status": execution["records"][0]["status"] = "missing"
                else: target["samples"][1]["from_seed"]["magnitude_px"] = 500
                with self.assertRaises(ValueError): self.build(inputs, execution)

    def test_invalid_threshold_config_fails_before_video_or_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, config = root / "model.onnx", root / "thresholds.json"
            model.write_bytes(b"Model loading must not occur.")
            config.write_text(json.dumps({"protocol": ep.THRESHOLD_PROTOCOL, "comparisons": [self.contract(unit="mm")]}))
            with patch.object(ev, "frame_times") as times, patch.object(ev, "observe") as observe:
                with self.assertRaises(ValueError): ev.run(SimpleNamespace(yolo_model=model, threshold_config=config))
            times.assert_not_called()
            observe.assert_not_called()

    def test_package_failure_keeps_execution_file_and_releases_capture_without_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, video = root / "fixture.onnx", root / "fixture.mp4"
            model.write_bytes(b"mocked")
            video.write_bytes(b"mocked")
            args = SimpleNamespace(yolo_model=model, video=video, out=root / "run", task="Task", split="test", episode="synthetic")
            inputs, execution = self.inputs()
            obs, selection, times, candidates, matching = inputs
            cap = Mock()
            cap.get.side_effect = [16, 12]
            def executed(*args):
                ev.save(args[5] / "04-expert-execution.json", execution)
                return execution
            with patch.object(ev, "frame_times", return_value=times), patch.object(ev.cv2, "VideoCapture", return_value=cap), \
                    patch.object(ev, "observe", return_value=obs), patch.object(ev, "select_windows", return_value=selection), \
                    patch.object(ev, "generate_candidates", return_value=candidates), patch.object(ev, "match_windows", return_value=matching), \
                    patch.object(ev, "execute_experts", side_effect=executed), \
                    patch.object(ev, "package_evidence", side_effect=ValueError("bad execution")), patch.object(ev, "score_saved") as score:
                with self.assertRaisesRegex(ValueError, "bad execution"): ev.run(args)
            cap.release.assert_called_once()
            score.assert_not_called()
            self.assertTrue((args.out / "04-expert-execution.json").exists())
            self.assertFalse((args.out / "05-check-evidence.json").exists())
            self.assertFalse((args.out / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
