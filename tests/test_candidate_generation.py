"""Selected-window extraction and actual ORT CPU contract fixtures; no paid models."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_experts as expert_tests
from test_check_judgment import jev_response
from test_final_evaluation import final_response
from test_vlm_matching import matching_observation, window_matching
from test_vlm_matching_integration import FrameCapture, ev, qwen_response
from window_selection import summarize_selection


@unittest.skipIf(ev is None, "OpenCV and NumPy required")
class CandidateBoundaryTests(unittest.TestCase):
    def test_empty_selection_saves_empty_bundle_without_model_or_frame_reads(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p0", "w1": "p1"})
        cap = FrameCapture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, "ObjectSpatialExpert") as factory:
            result = ev.generate_candidates("Task", matching_observation(), selection,
                                            [0, .1, .2, .3, .4], cap, Path(tmp), "unused.onnx")
            self.assertEqual(json.loads((Path(tmp) / "03-yolo-candidates.json").read_text()), result)
        factory.assert_not_called()
        self.assertEqual(cap.read_indices, [])
        self.assertEqual((result["windows"], result["frames"]), ([], []))
        self.assertEqual(result["status"], "no_selected_windows")
        self.assertIsNone(result["provenance"])

    def test_invalid_selection_or_pts_fails_before_loading_model_or_reading_video(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        invalid = copy.deepcopy(selection)
        invalid["selected_windows"].append(invalid["windows"][1])
        for bundle, times in ((invalid, [0, .1, .2, .3, .4]),
                              (selection, [0, .1, float("nan"), .3, .4]),
                              (selection, [0, .1])):
            cap = FrameCapture()
            with tempfile.TemporaryDirectory() as tmp, patch.object(ev, "ObjectSpatialExpert") as factory:
                with self.assertRaises(ValueError):
                    ev.generate_candidates("Task", matching_observation(), bundle, times,
                                           cap, Path(tmp), "unused.onnx")
                self.assertFalse((Path(tmp) / "03-yolo-candidates.json").exists())
            factory.assert_not_called()
            self.assertEqual(cap.read_indices, [])

    def test_missing_local_model_fails_before_any_video_or_paid_requests(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, "frame_times") as times, \
                patch.object(ev, "observe") as observe, patch.object(ev, "ask_jev") as ask:
            for model in (Path(tmp) / "missing.onnx", Path(tmp)):
                with self.assertRaisesRegex(ValueError, "local .onnx"):
                    ev.run(SimpleNamespace(yolo_model=model))
        times.assert_not_called()
        observe.assert_not_called()
        ask.assert_not_called()


@unittest.skipIf(ev is None or not expert_tests.HAS_ONNX, "OpenCV, NumPy, ONNX Runtime and ONNX required")
class CandidateGenerationIntegrationTests(unittest.TestCase):
    def make_expert(self, folder, **kwargs):
        return expert_tests.ObjectSpatialExpertTests().make_expert(folder, **kwargs)

    def test_all_selected_frames_preserve_ids_pts_masks_and_reuse_overlapping_frames(self):
        obs = matching_observation()
        obs["frames"].extend(dict(obs["frames"][-1], frame=i) for i in (6, 8))
        original = copy.deepcopy(obs)
        selection = summarize_selection([(0, 2), (2, 4), (4, 6), (6, 8)],
                                        {"w0": "p2", "w1": "p3", "w2": "p1", "w3": "p2"})
        times = [3, 3.02, 3.09, 3.09, 3.2, 3.23, 3.5, 3.55, 3.8]
        cap = FrameCapture(count=9)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            expert = self.make_expert(folder)
            with patch.object(ev, "ObjectSpatialExpert", return_value=expert) as factory, \
                    patch.object(expert, "infer", wraps=expert.infer) as infer, \
                    patch.object(expert.session, "run", wraps=expert.session.run) as run, \
                    patch.object(ev, "post") as post:
                result = ev.generate_candidates("Task", obs, selection, times, cap, folder, folder / "fixture.onnx")
            factory.assert_called_once_with(folder / "fixture.onnx")
            expected = [0, 1, 2, 3, 4, 6, 7, 8]
            self.assertEqual(cap.read_indices, expected)
            self.assertEqual([call.args[0][0].frame_id for call in infer.call_args_list], expected)
            self.assertEqual(run.call_count, len(expected))
            post.assert_not_called()
            self.assertEqual([f["frame_id"] for f in result["frames"]], expected)
            self.assertEqual([f["pts_seconds"] for f in result["frames"]], [times[i] for i in expected])
            self.assertEqual([w["window_id"] for w in result["windows"]], ["w0", "w1", "w3"])
            self.assertEqual([w["frame_ids"] for w in result["windows"]], [[0, 1, 2], [2, 3, 4], [6, 7, 8]])
            self.assertEqual(result["binding_status"], "unbound")
            self.assertEqual(result["provenance"]["providers"], ["CPUExecutionProvider"])
            self.assertEqual(result["provenance"]["model_sha256"], expert.provenance["model_sha256"])
            for frame in result["frames"]:
                for i, candidate in enumerate(frame["candidates"]):
                    self.assertEqual(candidate["candidate_id"], f"f{frame['frame_id']}:d{i}")
                    self.assertEqual(candidate["pts_seconds"], frame["pts_seconds"])
                    self.assertEqual(candidate["frame_id"], frame["frame_id"])
                    self.assertEqual(expert_tests.decode_mask(candidate["mask"]).shape, (12, 16))
                    self.assertNotIn("entity_key", candidate)
            self.assertEqual(json.loads((folder / "03-yolo-candidates.json").read_text()), result)
        self.assertEqual(obs, original)

    def test_two_apple_candidates_remain_unbound_and_numbering_is_frame_local(self):
        pred = ev.np.array([[[8, 20, 24, 36, .9, 0, 1, 0],
                             [16, 24, 32, 40, .8, 0, 1, 0]]], ev.np.float32)
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p1", "w1": "p2"})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            expert = self.make_expert(folder, pred=pred)
            with patch.object(ev, "ObjectSpatialExpert", return_value=expert):
                result = ev.generate_candidates("Task", matching_observation(), selection,
                                                [0, .1, .2, .3, .4], FrameCapture(), folder, folder / "fixture.onnx")
        self.assertEqual([f["frame_id"] for f in result["frames"]], [2, 3, 4])
        for frame in result["frames"]:
            self.assertEqual([c["class_name"] for c in frame["candidates"]], ["apple", "apple"])
            self.assertEqual([c["candidate_id"] for c in frame["candidates"]],
                             [f"f{frame['frame_id']}:d0", f"f{frame['frame_id']}:d1"])
            self.assertTrue(all("entity_key" not in c for c in frame["candidates"]))

    def test_no_detections_and_empty_masks_are_persisted_without_fabricated_binding(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        for mode in ("no_detections", "empty_mask"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                kwargs = {"pred": ev.np.zeros((1, 3, 8), ev.np.float32)} if mode == "no_detections" else {
                    "proto": -ev.np.ones((1, 2, 16, 16), ev.np.float32)}
                expert = self.make_expert(folder, **kwargs)
                with patch.object(ev, "ObjectSpatialExpert", return_value=expert):
                    result = ev.generate_candidates("Task", matching_observation(), selection,
                                                    [0, .1, .2, .3, .4], FrameCapture(), folder, folder / "fixture.onnx")
                for frame in result["frames"]:
                    if mode == "no_detections":
                        self.assertEqual((frame["status"], frame["candidates"]), ("no_detections", []))
                    else:
                        self.assertTrue(all(c["mask_status"] == "empty" for c in frame["candidates"]))
                self.assertEqual(result["binding_status"], "unbound")
                self.assertTrue(result["limitations"])
                self.assertEqual(json.loads((folder / "03-yolo-candidates.json").read_text()), result)

    def test_single_frame_window_runs_segmentation_once(self):
        selection = summarize_selection([(0, 0)], {"w0": "p2"})
        obs = matching_observation()
        obs["frames"] = obs["frames"][:1]
        for check in obs["candidate_checks"]:
            check["related_frames"] = [0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            expert = self.make_expert(folder)
            with patch.object(ev, "ObjectSpatialExpert", return_value=expert), \
                    patch.object(expert.session, "run", wraps=expert.session.run) as run:
                result = ev.generate_candidates("Task", obs, selection, [1.25], FrameCapture(count=1),
                                                folder, folder / "fixture.onnx")
            run.assert_called_once()
        self.assertEqual(result["windows"][0]["frame_ids"], [0])
        self.assertEqual(result["frames"][0]["pts_seconds"], 1.25)

    def test_decode_failure_cannot_save_a_completed_candidate_bundle(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            expert = self.make_expert(folder)
            with patch.object(ev, "ObjectSpatialExpert", return_value=expert), patch.object(ev, "post") as post:
                with self.assertRaisesRegex(ValueError, "candidate video decode failed at frame 1"):
                    ev.generate_candidates("Task", matching_observation(), selection,
                                           [0, .1, .2, .3, .4], FrameCapture(fail_at=1), folder, folder / "fixture.onnx")
            post.assert_not_called()
            self.assertFalse((folder / "03-yolo-candidates.json").exists())

    def test_run_uses_actual_candidate_stage_and_releases_capture_on_failure(self):
        for fail_at in (None, 1):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                expert = self.make_expert(root)
                video = root / "mocked-video.mp4"
                video.write_bytes(b"Decoding uses a synthetic capture.")
                args = SimpleNamespace(video=video, yolo_model=root / "fixture.onnx", out=root / "run",
                                       task="Task", split="test", episode="synthetic")
                args.threshold_config = root / "thresholds.json"
                args.threshold_config.write_text(json.dumps({"protocol": "explicit-measurement-thresholds-v1", "comparisons": [
                    {"id": "fixture-gap", "window_id": "w0", "check_id": "c0", "expert_id": "object_spatial",
                     "entity_keys": ["apple", "bowl"], "metric": "box_edge_distance_px", "unit": "px",
                     "aggregation": "at_frame", "frame_range": [2, 2], "operator": "<=", "threshold": 2}]}))
                cap = FrameCapture(count=5, fail_at=fail_at)
                cap.get = Mock(side_effect=[16, 12])
                cap.release = Mock()
                selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
                with patch.object(ev, "frame_times", return_value=[0, .1, .2, .3, .4]), \
                        patch.object(ev.cv2, "VideoCapture", return_value=cap), \
                        patch.object(ev, "observe", return_value=matching_observation()), \
                        patch.object(ev, "select_windows", return_value=selection), \
                        patch.object(ev, "ObjectSpatialExpert", return_value=expert) as factory, \
                        patch.object(ev, "match_windows", wraps=ev.match_windows) as match, \
                        patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only", "TYPESAFE_API_KEY": "test-only"}), \
                        patch.object(ev, "post", side_effect=[qwen_response(window_matching()), jev_response(), final_response("level_3")]) as post, \
                        patch.object(expert.session, "run", wraps=expert.session.run) as infer:
                    if fail_at is not None:
                        with self.assertRaises(ValueError):
                            ev.run(args, sample_interval_seconds=.2)
                        self.assertFalse((args.out / "result.json").exists())
                        match.assert_not_called()
                        post.assert_not_called()
                    else:
                        result = ev.run(args, sample_interval_seconds=.2)
                        self.assertEqual((result["candidate_frame_count"], result["candidate_count"]), (3, 6))
                        self.assertEqual(result["status"], "scored")
                        self.assertEqual(result["score"], 3)
                        execution = json.loads((args.out / result["execution_file"]).read_text())
                        self.assertEqual(result["expert_execution_count"], 2)
                        self.assertEqual([r["expert_id"] for r in execution["records"]],
                                         ["object_spatial", "short_window_motion"])
                        self.assertEqual(execution["records"][0]["status"], "complete")
                        self.assertEqual(execution["records"][1]["status"], "missing")  # Untextured fixture.
                        packages = json.loads((args.out / result["evidence_file"]).read_text())
                        self.assertEqual(result["check_evidence_count"], 2)
                        self.assertEqual(packages["packages"][0]["threshold_comparisons"]["comparisons"][0]["status"], "compared")
                        metadata = json.loads((args.out / "run.json").read_text())
                        self.assertEqual(metadata["evidence_packaging"]["threshold_source"]["path"], str(args.threshold_config.resolve()))
                        self.assertEqual(metadata["evidence_packaging"]["threshold_source"]["sha256"],
                                         ev.hashlib.sha256(args.threshold_config.read_bytes()).hexdigest())
                        self.assertEqual(infer.call_count, 3)
                        match.assert_called_once()
                        self.assertEqual(post.call_count, 3)
                        self.assertEqual(result["jev_final_call_count"], 1)
                        self.assertEqual(result["jev_check_call_count"], 1)
                        self.assertEqual(result["check_judgment_count"], 2)
                        judgments = json.loads((args.out / result["judgment_file"]).read_text())
                        self.assertEqual([r["answer"] for r in judgments["judgments"]], ["yes", "unknown"])
                        self.assertEqual(json.loads((args.out / "result.json").read_text()), result)
                    cap.release.assert_called_once()
                    factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
