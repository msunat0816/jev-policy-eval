"""Candidate-bound execution with actual geometry/LK and synthetic video, no APIs."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_vlm_matching import candidate_bundle, matching_observation, window_matching
from test_vlm_matching_integration import ev, FrameCapture
from window_selection import summarize_selection

if ev is not None:
    import expert_execution as execution
    from experts.masks import encode_mask
    from test_experts import motion_fixture


def saved_matching(task, obs, selection, times, values=None):
    """The exact saved stage-3 format, with caller-selected proposals/bindings."""
    windows = []
    for i, window in enumerate(selection["selected_windows"]):
        first, last = window["frame_range"]
        value = values[i] if values is not None else window_matching(window["id"], first, last)
        request = f"03-match-{window['id']}.json"
        directory = f"matching_frames/{window['id']}"
        ids = list(range(first, last + 1))
        source = {"method": "matching_vlm_candidate_selection", "model": ev.MATCHING_MODEL,
                  "request_file": request, "candidate_file": "03-yolo-candidates.json"}
        windows.append({"window_id": window["id"], "frame_range": [first, last], "frame_ids": ids,
                        "pts_seconds": [times[f] for f in ids], "image_directory": directory,
                        "images": [{"frame_id": f, "pts_seconds": times[f],
                                    "original_file": f"{directory}/{f}.jpg",
                                    "candidates_file": f"{directory}/{f}-candidates.jpg"} for f in ids],
                        "request_file": request, "candidate_file": "03-yolo-candidates.json",
                        "checks": copy.deepcopy(value["checks"]),
                        "bindings": [{**b, "pts_seconds": times[b["frame_id"]], "source": dict(source)}
                                     for b in value["bindings"]]})
    return {"protocol": ev.MATCHING_PROTOCOL, "model": ev.MATCHING_MODEL,
            "status": "proposals_and_bindings_only", "task": task,
            "requirements": copy.deepcopy(obs["requirements"]), "checklist": copy.deepcopy(obs["candidate_checks"]),
            "expert_catalog": ev.EXPERT_CATALOG, "candidate_file": "03-yolo-candidates.json", "windows": windows}


@unittest.skipIf(ev is None, "OpenCV and NumPy required")
class ExpertExecutionTests(unittest.TestCase):
    def inputs(self, values=None, **kwargs):
        obs = matching_observation()
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        times = [0, .09, .25, .4, .6]
        candidates = candidate_bundle(selection, times, **kwargs)
        matching = saved_matching("Task", obs, selection, times, values)
        return obs, selection, times, candidates, matching

    def run_execution(self, inputs, cap=None, folder=None):
        obs, selection, times, candidates, matching = inputs
        if folder is None:
            return execution.execute_proposals("Task", obs, selection, times, cap or FrameCapture(), candidates, matching)
        return ev.execute_experts("Task", obs, selection, times, cap or FrameCapture(), folder, candidates, matching)

    def test_spatial_reuses_candidates_keeps_provenance_and_does_not_read_video(self):
        inputs = self.inputs([window_matching(motion=False)])
        before = copy.deepcopy(inputs)
        cap = FrameCapture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, "ObjectSpatialExpert") as detector, \
                patch.object(ev, "post") as post, patch.object(ev, "ask_jev") as jev:
            result = self.run_execution(inputs, cap, Path(tmp))
            self.assertEqual(json.loads((Path(tmp) / "04-expert-execution.json").read_text()), result)
        detector.assert_not_called()
        post.assert_not_called()
        jev.assert_not_called()
        self.assertEqual(cap.read_indices, [])
        self.assertEqual(inputs, before)
        record, = result["records"]
        self.assertEqual((record["window_id"], record["check_id"], record["entity_keys"]), ("w0", "c0", ["apple", "bowl"]))
        self.assertEqual(record["source"]["window_matching_file"], "03-matching-w0.json")
        self.assertEqual(record["bindings"][0]["source"]["request_file"], "03-match-w0.json")
        self.assertEqual(record["frame_ids"], [0, 2])
        self.assertEqual(record["pts_seconds"], [0, .25])
        self.assertEqual(record["status"], "complete")
        pair = record["result"]["frames"][0]["pairs"][0]
        self.assertEqual(pair["entity_keys"], ["apple", "bowl"])
        self.assertEqual(pair["measurement"]["box_edge_distance_px"], 2)
        self.assertEqual(pair["measurement"]["mask_iou"], 0)
        self.assertNotIn("answer", record)
        self.assertNotIn("score", result)

    def test_empty_masks_preserve_box_geometry_but_not_mask_measurements(self):
        inputs = self.inputs([window_matching(motion=False)], empty_masks=True)
        result = self.run_execution(inputs)["records"][0]
        self.assertEqual(result["status"], "partial")
        for frame in result["result"]["frames"]:
            pair = frame["pairs"][0]
            self.assertEqual(pair["measurement"]["box_edge_distance_px"], 2)
            self.assertIsNone(pair["measurement"]["mask_iou"])
            self.assertIsNone(pair["measurement"]["mask_intersection_px"])

    def test_spatial_missing_binding_keeps_other_object_and_explicit_missing_pair(self):
        inputs = self.inputs([window_matching(motion=False)])
        binding = inputs[-1]["windows"][0]["bindings"][0]
        binding.update(status="missing", candidate_id=None, reason="Two indistinguishable apples.")
        record = self.run_execution(inputs)["records"][0]
        first, last = record["result"]["frames"]
        self.assertEqual(first["objects"][0]["status"], "missing")
        self.assertEqual(first["objects"][1]["status"], "complete")
        self.assertIsNone(first["pairs"][0]["measurement"])
        self.assertEqual(last["status"], "complete")
        self.assertEqual(record["status"], "partial")
        self.assertEqual(record["bindings"][0]["reason"], "Two indistinguishable apples.")

    def test_single_entity_spatial_proposal_does_not_invent_a_reference(self):
        value = window_matching(motion=False)
        value["checks"][0]["experts"][0]["entity_keys"] = ["apple"]
        value["bindings"] = [b for b in value["bindings"] if b["entity_key"] == "apple"]
        record, = self.run_execution(self.inputs([value]))["records"]
        self.assertEqual(record["entity_keys"], ["apple"])
        self.assertEqual(record["result"]["frames"][0]["pairs"], [])

    def test_no_detections_preserve_all_missing_bindings_without_tracking(self):
        inputs = self.inputs(empty=True)
        for binding in inputs[-1]["windows"][0]["bindings"]:
            binding.update(status="missing", candidate_id=None, reason="No candidates detected.")
        cap = FrameCapture()
        with patch.object(execution, "ShortWindowMotionExpert") as expert:
            records = self.run_execution(inputs, cap)["records"]
        expert.assert_not_called()
        self.assertEqual(cap.read_indices, [])
        self.assertTrue(all(r["status"] == "missing" for r in records))
        self.assertIsNone(records[0]["result"]["frames"][0]["pairs"][0]["measurement"])

    def test_program_missing_provenance_requires_empty_inventory_and_null_model(self):
        for mode in ("valid", "nonempty", "bound", "model", "omitted"):
            with self.subTest(mode=mode):
                inputs = self.inputs(empty=mode != "nonempty")
                for binding in inputs[-1]["windows"][0]["bindings"]:
                    binding.update(status="missing", candidate_id=None, reason="No candidates detected.")
                    binding["source"].update(method="program_empty_candidate_inventory", model=None)
                first = inputs[-1]["windows"][0]["bindings"][0]
                if mode == "bound": first.update(status="bound", candidate_id="f0:d0")
                elif mode == "model": first["source"]["model"] = ev.MATCHING_MODEL
                elif mode == "omitted": inputs[-1]["windows"][0]["bindings"].pop()
                if mode == "valid":
                    self.assertTrue(all(r["status"] == "missing" for r in self.run_execution(inputs)["records"]))
                else:
                    with self.assertRaises(ValueError): self.run_execution(inputs)

    def test_single_frame_window_executes_only_spatial_geometry(self):
        obs = matching_observation()
        obs["frames"] = [obs["frames"][0]]
        for check in obs["candidate_checks"]:
            check["related_frames"] = [0]
        selection = summarize_selection([(0, 0)], {"w0": "p2"})
        times = [1.25]
        matching = saved_matching("Task", obs, selection, times, [window_matching(last=0, motion=False)])
        cap = FrameCapture(count=1)
        record, = self.run_execution((obs, selection, times, candidate_bundle(selection, times), matching), cap)["records"]
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["frame_ids"], [0])
        self.assertEqual(record["pts_seconds"], [1.25])
        self.assertEqual(cap.read_indices, [])

    def test_missing_target_binding_or_empty_mask_never_calls_lk_or_reads_frames(self):
        for mode in ("binding", "mask"):
            with self.subTest(mode=mode):
                inputs = self.inputs(empty_masks=mode == "mask")
                if mode == "binding":
                    inputs[-1]["windows"][0]["bindings"][0].update(status="missing", candidate_id=None,
                                                                 reason="Cannot identify the task apple.")
                cap = FrameCapture()
                with patch.object(execution, "ShortWindowMotionExpert") as expert:
                    motion = self.run_execution(inputs, cap)["records"][1]
                expert.assert_not_called()
                self.assertEqual(cap.read_indices, [])
                self.assertEqual(motion["execution_status"], "not_executed")
                self.assertEqual(motion["status"], "missing")
                self.assertIsNone(motion["result"])
                self.assertEqual(motion["missing"][0]["reason"], "binding_missing" if mode == "binding" else "empty_mask")

    def test_insufficient_features_are_recorded_by_actual_lk_as_missing(self):
        motion = self.run_execution(self.inputs())["records"][1]
        self.assertEqual(motion["execution_status"], "executed")
        self.assertEqual(motion["status"], "missing")
        self.assertEqual(motion["result"]["target"]["interruptions"][0]["reason"], "insufficient_feature_points")
        self.assertIsNone(motion["result"]["target"]["samples"][0]["from_seed"])

    def test_bad_references_in_any_window_fail_before_any_execution(self):
        for mode in ("window", "check", "expert", "candidate", "candidate_frame", "motion_gap", "outside_window",
                     "source", "pts", "target", "context", "duplicate_binding", "candidate_mask"):
            with self.subTest(mode=mode):
                inputs = self.inputs()
                obs, selection, times, candidates, matching = inputs
                record = matching["windows"][0]
                if mode == "window": record["window_id"] = "w99"
                elif mode == "check": record["checks"][0]["check_id"] = "c99"
                elif mode == "expert": record["checks"][0]["experts"][0]["expert_id"] = "new_expert"
                elif mode == "candidate": record["bindings"][0]["candidate_id"] = "f0:d999"
                elif mode == "candidate_frame": record["bindings"][0]["candidate_id"] = "f2:d0"
                elif mode == "motion_gap": record["checks"][0]["experts"][1]["frame_ids"] = [0, 2]
                elif mode == "outside_window": record["checks"][0]["experts"][1]["frame_ids"] = [2, 3]
                elif mode == "source": record["bindings"][0]["source"]["request_file"] = "other-request.json"
                elif mode == "pts": record["bindings"][0]["pts_seconds"] = 999
                elif mode == "target": del record["checks"][0]["experts"][1]["target_entity_key"]
                elif mode == "context": matching["task"] = "Other task"
                elif mode == "duplicate_binding": record["bindings"].append(copy.deepcopy(record["bindings"][0]))
                else: candidates["frames"][0]["candidates"][0]["mask"]["counts"] = [1]
                cap = FrameCapture()
                with tempfile.TemporaryDirectory() as tmp, \
                        patch.object(execution, "pairwise_spatial_evidence") as geometry, \
                        patch.object(execution, "ShortWindowMotionExpert") as expert:
                    with self.assertRaises(ValueError):
                        self.run_execution(inputs, cap, Path(tmp))
                    self.assertFalse((Path(tmp) / "04-expert-execution.json").exists())
                geometry.assert_not_called()
                expert.assert_not_called()
                self.assertEqual(cap.read_indices, [])

    def test_later_invalid_proposal_cannot_leave_earlier_expert_executed(self):
        inputs = self.inputs()
        inputs[-1]["windows"][0]["checks"][0]["experts"][1]["motion_mode"] = "guess"
        with patch.object(execution, "pairwise_spatial_evidence") as geometry:
            with self.assertRaises(ValueError):
                self.run_execution(inputs)
        geometry.assert_not_called()

    def test_empty_selection_has_no_reads_models_or_executions(self):
        obs = matching_observation()
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p0", "w1": "p1"})
        times = [i/10 for i in range(5)]
        cap = FrameCapture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(execution, "ShortWindowMotionExpert") as motion, \
                patch.object(ev, "ObjectSpatialExpert") as detector:
            result = self.run_execution((obs, selection, times, candidate_bundle(selection, times),
                                         saved_matching("Task", obs, selection, times)), cap, Path(tmp))
        self.assertEqual(result["records"], [])
        self.assertEqual(cap.read_indices, [])
        motion.assert_not_called()
        detector.assert_not_called()

    def test_shared_bindings_and_multiple_windows_keep_each_check_link(self):
        inputs = self.inputs([window_matching(motion=False)])
        obs, selection, times, candidates, _ = inputs
        obs["frames"].append(dict(obs["frames"][-1], frame=6))
        selection = summarize_selection([(0, 2), (2, 4), (4, 6)], {"w0": "p2", "w1": "p1", "w2": "p3"})
        times = [i/10 for i in range(7)]
        # A second check can share one window binding; it remains a separate execution association.
        obs["candidate_checks"][1]["entity_keys"] = ["apple", "bowl"]
        values = [window_matching("w0", 0, 2, motion=False), window_matching("w2", 4, 6, motion=False)]
        for value in values:
            value["checks"][1] = {**copy.deepcopy(value["checks"][0]), "check_id": "c1"}
        result = self.run_execution((obs, selection, times, candidate_bundle(selection, times),
                                     saved_matching("Task", obs, selection, times, values)))
        self.assertEqual([(r["window_id"], r["check_id"]) for r in result["records"]],
                         [("w0", "c0"), ("w0", "c1"), ("w2", "c0"), ("w2", "c1")])
        self.assertEqual(len({r["id"] for r in result["records"]}), 4)
        self.assertEqual(result["records"][0]["bindings"], result["records"][1]["bindings"])

    def motion_inputs(self, relative=True):
        frames, target, reference = motion_fixture(3, 1)
        obs = matching_observation()
        obs["frames"] = [dict(obs["frames"][0], frame=20), dict(obs["frames"][-1], frame=24)]
        for check in obs["candidate_checks"]:
            check["related_frames"] = [20, 24]
        selection = summarize_selection([(20, 24)], {"w0": "p2"})
        times = [i*.06 + i*i*.001 for i in range(25)]  # Original, nonuniform PTS.
        candidates = candidate_bundle(selection, times, size=[96, 160])
        value = window_matching("w0", 20, 24)
        value["checks"][0]["experts"] = [value["checks"][0]["experts"][1]]
        proposal = value["checks"][0]["experts"][0]
        proposal.update(motion_mode="relative" if relative else "absolute",
                        reference_entity_key="bowl" if relative else None)
        value["bindings"] = [b for b in value["bindings"] if b["frame_id"] == 20 and
                             (b["entity_key"] == "apple" or relative)]
        for candidate, seed, box in zip(candidates["frames"][0]["candidates"], (target, reference),
                                        ([15, 16, 47, 48], [95, 56, 127, 88])):
            candidate.update(mask=encode_mask(seed.mask), box_xyxy_px=box)
        images = {f.frame_id: f.image.copy() for f in frames}
        cap = FrameCapture(count=25)
        def read():
            index = cap.position
            cap.position += 1
            cap.read_indices.append(index)
            return True, images[index].copy()
        cap.read = read
        return (obs, selection, times, candidates, saved_matching("Task", obs, selection, times, [value])), cap, images

    def test_actual_lk_uses_explicit_roles_first_frame_masks_and_all_original_pts(self):
        inputs, cap, _ = self.motion_inputs()
        # Entity ordering in a check must not assign target/reference roles.
        inputs[0]["candidate_checks"][0]["entity_keys"].reverse()
        inputs[-1]["checklist"] = copy.deepcopy(inputs[0]["candidate_checks"])
        result, = self.run_execution(inputs, cap)["records"]
        self.assertEqual(cap.read_indices, [20, 21, 22, 23, 24])
        self.assertEqual(result["status"], "complete")
        target = result["result"]["target"]
        reference = result["result"]["reference"]
        self.assertEqual((target["entity_key"], reference["entity_key"]), ("apple", "bowl"))
        self.assertEqual((target["mask_source"], reference["mask_source"]), ("f20:d0", "f20:d1"))
        self.assertEqual([s["pts_seconds"] for s in target["samples"]], inputs[2][20:25])
        ev.np.testing.assert_allclose(target["samples"][-1]["from_seed"]["displacement_xy_px"], [12, 0], atol=.3)
        ev.np.testing.assert_allclose(result["result"]["relative_motion"][-1]["from_seed"]["displacement_xy_px"], [8, 0], atol=.4)
        json.dumps(result, allow_nan=False)

    def test_missing_reference_retains_target_absolute_track_and_relative_missing(self):
        for mode in ("binding", "mask"):
            with self.subTest(mode=mode):
                inputs, cap, _ = self.motion_inputs()
                if mode == "binding":
                    inputs[-1]["windows"][0]["bindings"][1].update(status="missing", candidate_id=None,
                                                                 reason="Reference identity ambiguous.")
                else:
                    inputs[3]["frames"][0]["candidates"][1].update(mask=encode_mask(ev.np.zeros((96, 160), bool)), mask_status="empty")
                result, = self.run_execution(inputs, cap)["records"]
                self.assertEqual(result["execution_status"], "executed")
                self.assertEqual(result["status"], "missing")  # Requested relative measurement is missing.
                self.assertEqual(result["result"]["target"]["status"], "complete")
                self.assertIsNone(result["result"]["reference"])
                self.assertTrue(all(s["status"] == "missing" and s["from_seed"] is None
                                    for s in result["result"]["relative_motion"]))
                self.assertEqual(result["missing"][0]["role"], "reference")

    def test_absolute_motion_never_uses_an_unrequested_reference(self):
        inputs, cap, _ = self.motion_inputs(relative=False)
        record, = self.run_execution(inputs, cap)["records"]
        self.assertEqual(record["status"], "complete")
        self.assertIsNone(record["result"]["reference"])
        self.assertEqual(record["entity_keys"], ["apple"])
        self.assertEqual(len(record["bindings"]), 1)

    def test_motion_subinterval_uses_proposed_start_mask_not_window_start(self):
        inputs, cap, _ = self.motion_inputs(relative=False)
        window = inputs[-1]["windows"][0]
        window["checks"][0]["experts"][0]["frame_ids"] = [21, 22, 23]
        binding = window["bindings"][0]
        binding.update(frame_id=21, pts_seconds=inputs[2][21], candidate_id="f21:d0")
        seed_candidate = inputs[3]["frames"][1]["candidates"][0]
        mask = ev.np.zeros((96, 160), bool)
        mask[16:48, 18:50] = True
        seed_candidate.update(mask=encode_mask(mask), box_xyxy_px=[18, 16, 50, 48])
        record, = self.run_execution(inputs, cap)["records"]
        self.assertEqual(cap.read_indices, [21, 22, 23])
        self.assertEqual(record["result"]["target"]["seed_frame_id"], 21)
        self.assertEqual(record["result"]["target"]["mask_source"], "f21:d0")
        ev.np.testing.assert_allclose(record["result"]["target"]["samples"][-1]["from_seed"]["displacement_xy_px"],
                                     [6, 0], atol=.3)

    def test_tracking_loss_preserves_interruption_and_does_not_reseed(self):
        inputs, cap, images = self.motion_inputs(relative=False)
        for i in (22, 23, 24):
            images[i][:] = 0
        record, = self.run_execution(inputs, cap)["records"]
        self.assertEqual(record["status"], "partial")
        target = record["result"]["target"]
        self.assertTrue(target["interruptions"])
        self.assertEqual(target["interruptions"][0]["frame_range"][-1], 24)
        self.assertIsNone(target["samples"][-1]["from_seed"])
        self.assertEqual(target["samples"][-1]["reason"], "track_unavailable_after_loss")

    def test_decode_or_size_failure_never_saves_completed_execution(self):
        for mode in ("decode", "size"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                inputs, cap, images = self.motion_inputs()
                if mode == "decode":
                    real_read = cap.read
                    cap.read = lambda: (False, None) if cap.position == 22 else real_read()
                else:
                    images[22] = images[22][:-1]
                with self.assertRaisesRegex(ValueError, "frame 22"):
                    self.run_execution(inputs, cap, Path(tmp))
                self.assertFalse((Path(tmp) / "04-expert-execution.json").exists())

    def test_main_releases_capture_and_does_not_save_result_on_execution_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, video = root / "fixture.onnx", root / "fixture.mp4"
            model.write_bytes(b"Model initialization mocked.")
            video.write_bytes(b"Video decode mocked.")
            args = SimpleNamespace(yolo_model=model, video=video, out=root / "run", task="Task", split="test", episode="synthetic")
            inputs = self.inputs()
            obs, selection, times, candidates, matching = inputs
            cap = FrameCapture()
            cap.get = Mock(side_effect=[16, 12])
            cap.release = Mock()
            with patch.object(ev, "frame_times", return_value=times), patch.object(ev.cv2, "VideoCapture", return_value=cap), \
                    patch.object(ev, "observe", return_value=obs), patch.object(ev, "select_windows", return_value=selection), \
                    patch.object(ev, "generate_candidates", return_value=candidates), patch.object(ev, "match_windows", return_value=matching), \
                    patch.object(ev, "execute_experts", side_effect=ValueError("execution decode failed")), \
                    patch.object(ev, "score_saved") as score:
                with self.assertRaisesRegex(ValueError, "execution decode failed"):
                    ev.run(args)
            cap.release.assert_called_once()
            score.assert_not_called()
            self.assertTrue((args.out / "run.json").exists())
            self.assertFalse((args.out / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
