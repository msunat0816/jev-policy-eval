"""Offline evidence checks: actual OpenCV and tiny ONNX graphs, no paid APIs."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    from experts import (MaskSeed, MotionConfig, ObjectSpatialConfig, ObjectSpatialExpert,
                         ShortWindowMotionExpert, VideoFrame)
    from experts.geometry import pairwise_spatial_evidence
    from experts.masks import decode_mask, encode_mask
except ModuleNotFoundError as exc:
    if exc.name not in ("cv2", "numpy"):
        raise
    cv2 = None

HAS_ONNX = all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime"))


@unittest.skipIf(cv2 is None, "OpenCV and NumPy required")
class MaskAndGeometryTests(unittest.TestCase):
    def test_lossless_row_major_roundtrip_and_json(self):
        for mask in (np.zeros((3, 4), bool), np.ones((3, 4), bool),
                     np.array([[1, 0, 0], [0, 1, 1]], bool)):
            record = json.loads(json.dumps(encode_mask(mask), allow_nan=False))
            np.testing.assert_array_equal(decode_mask(record), mask)
            self.assertEqual(record["encoding"], "rle_row_major")
        with self.assertRaises(ValueError):
            decode_mask({"encoding": "rle_row_major", "size": [2, 2], "counts": [3]})
        with self.assertRaises(ValueError):
            encode_mask(np.array([[0.5]]))

    def test_geometry_gaps_overlap_and_missing_mask(self):
        def candidate(name, box):
            mask = np.zeros((40, 60), bool)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = True
            return {"candidate_id": name, "frame_id": 7, "pts_seconds": 1.2,
                    "box_xyxy_px": box, "mask": encode_mask(mask)}
        a, b = candidate("a", [2, 4, 12, 14]), candidate("b", [22, 4, 32, 14])
        result = pairwise_spatial_evidence(a, b)
        self.assertEqual(result["box_edge_distance_px"], 10)
        self.assertEqual(result["center_distance_px"], 20)
        self.assertEqual(result["mask_iou"], 0)
        overlap = pairwise_spatial_evidence(a, candidate("c", [7, 4, 17, 14]))
        self.assertAlmostEqual(overlap["mask_iou"], 1 / 3)
        b["mask"] = encode_mask(np.zeros((40, 60), bool))
        self.assertIsNone(pairwise_spatial_evidence(a, b)["mask_iou"])
        b["frame_id"] = 8
        with self.assertRaises(ValueError):
            pairwise_spatial_evidence(a, b)


def motion_fixture(target_step=2, reference_step=0):
    rng = random.Random(17)
    textures = [np.array([rng.randrange(256) for _ in range(32 * 32 * 3)], dtype=np.uint8).reshape(32, 32, 3)
                for _ in range(2)]
    frames = []
    for i in range(5):
        image = np.zeros((96, 160, 3), np.uint8)
        x, rx = 15 + i * target_step, 95 + i * reference_step
        image[16:48, x:x + 32] = textures[0]
        image[56:88, rx:rx + 32] = textures[1]
        frames.append(VideoFrame(20 + i, 1.0 + i * 0.1, image))
    masks = [np.zeros((96, 160), np.uint8) for _ in range(2)]
    masks[0][16:48, 15:47] = 255
    masks[1][56:88, 95:127] = 255
    return frames, MaskSeed("apple", 20, masks[0], "verified:f20:d0"), MaskSeed("tray", 20, masks[1], "manual:reference")


@unittest.skipIf(cv2 is None, "OpenCV and NumPy required")
class MotionExpertTests(unittest.TestCase):
    def test_actual_lk_translation_pts_counts_and_relative_motion(self):
        frames, target, ref = motion_fixture(3, 1)
        result = ShortWindowMotionExpert().track(frames, target, reference=ref)
        self.assertEqual(result["target"]["status"], "complete")
        self.assertEqual(result["reference"]["status"], "complete")
        sample = result["target"]["samples"][-1]
        self.assertEqual(sample["frame_id"], 24)
        self.assertAlmostEqual(sample["pts_seconds"], 1.4)
        self.assertGreaterEqual(sample["valid_points"], 6)
        np.testing.assert_allclose(sample["from_seed"]["displacement_xy_px"], [12, 0], atol=0.2)
        np.testing.assert_allclose(result["relative_motion"][-1]["from_seed"]["displacement_xy_px"], [8, 0], atol=0.3)
        self.assertAlmostEqual(sample["from_seed"]["direction_degrees_image"], 0, delta=1)
        json.dumps(result, allow_nan=False)

    def test_stationary_points_and_no_reference_do_not_claim_relative_stability(self):
        frames, seed, _ = motion_fixture(0, 0)
        result = ShortWindowMotionExpert().track(frames, seed)
        self.assertEqual(result["target"]["status"], "complete")
        self.assertAlmostEqual(result["target"]["samples"][-1]["from_seed"]["magnitude_px"], 0, delta=0.01)
        self.assertTrue(all(r["reason"] == "reference_not_supplied" for r in result["relative_motion"]))

    def test_missing_empty_and_textureless_masks_produce_missing_not_zero_motion(self):
        frames, seed, _ = motion_fixture()
        for target in (None, MaskSeed("apple", 20, np.zeros((96, 160), bool), "empty"),
                       MaskSeed("apple", 20, np.pad(np.ones((10, 10), bool), ((0, 86), (0, 150))), "smooth")):
            result = ShortWindowMotionExpert().track(frames, target)
            self.assertEqual(result["target"]["status"], "missing")
            self.assertTrue(all(s["from_seed"] is None for s in result["target"]["samples"]))
            self.assertEqual(result["target"]["interruptions"][0]["frame_range"], [20, 24])

    def test_forward_backward_rejection_records_loss_through_window_end(self):
        frames, target, ref = motion_fixture()
        actual = cv2.calcOpticalFlowPyrLK
        calls = 0

        def inconsistent(*args, **kwargs):
            nonlocal calls
            calls += 1
            points, status, error = actual(*args, **kwargs)
            if calls == 2:  # Forward succeeds; backward returns inconsistent points.
                points = points + 5
            return points, status, error

        with patch.object(cv2, "calcOpticalFlowPyrLK", side_effect=inconsistent):
            result = ShortWindowMotionExpert().track(frames, target, reference=ref)
        track = result["target"]
        self.assertEqual(track["status"], "partial")
        self.assertEqual(track["samples"][1]["valid_points"], 0)
        self.assertIsNone(track["samples"][1]["step"])
        self.assertEqual(track["interruptions"][0]["frame_range"], [20, 24])
        self.assertTrue(all(s["status"] == "missing" for s in result["relative_motion"][1:]))

    def test_actual_occlusion_is_not_reported_as_stationary(self):
        frames, seed, _ = motion_fixture()
        for frame in frames[2:]:
            frame.image[0:52] = 0
        result = ShortWindowMotionExpert().track(frames, seed)["target"]
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["samples"][-1]["from_seed"])
        self.assertTrue(result["interruptions"])

    def test_continuity_mask_provenance_and_config_are_validated(self):
        frames, seed, _ = motion_fixture()
        expert = ShortWindowMotionExpert()
        for invalid in (frames[:1], frames[::2], frames[::-1]):
            with self.assertRaises(ValueError):
                expert.track(invalid, seed)
        for invalid in (MaskSeed("apple", 21, seed.mask, "bad frame"),
                        MaskSeed("apple", 20, seed.mask[:10], "bad size"),
                        MaskSeed("apple", 20, seed.mask, "")):
            with self.assertRaises(ValueError):
                expert.track(frames, invalid)
        for config in ({"min_points": 201}, {"max_fb_error_px": float("nan")}, {"window_size_px": 20}):
            with self.assertRaises(ValueError):
                MotionConfig(**config)


@unittest.skipIf(cv2 is None or not HAS_ONNX, "OpenCV, NumPy, ONNX Runtime and ONNX required")
class ObjectSpatialExpertTests(unittest.TestCase):
    def make_expert(self, folder, *, pred=None, proto=None, task="segment", names="{0: 'apple', 1: 'cup'}"):
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        if pred is None:
            pred = np.array([[[8, 20, 24, 36, .9, 0, 1, 0],
                              [16, 24, 32, 40, .8, 1, 1, 0],
                              [0, 0, 1, 1, .1, 0, 0, 0]]], np.float32)
        if proto is None:
            proto = np.ones((1, 2, 16, 16), np.float32)
        graph = helper.make_graph([
            helper.make_node("Constant", [], ["pred"], value=numpy_helper.from_array(pred)),
            helper.make_node("Constant", [], ["proto"], value=numpy_helper.from_array(proto)),
        ], "segmentation-contract-fixture", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
            [helper.make_tensor_value_info("pred", TensorProto.FLOAT, list(pred.shape)),
             helper.make_tensor_value_info("proto", TensorProto.FLOAT, list(proto.shape))])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        helper.set_model_props(model, {"task": task, "names": names})
        path = Path(folder) / "fixture.onnx"
        onnx.save(model, str(path))
        return ObjectSpatialExpert(path)

    def frame(self):
        image = np.zeros((32, 64, 3), np.uint8)
        image[:, :] = [10, 20, 30]
        return VideoFrame(37, 2.7, image)

    def test_actual_ort_cpu_preprocessing_original_boxes_masks_and_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            expert = self.make_expert(tmp)
            with patch.object(expert.session, "run", wraps=expert.session.run) as run:
                result = expert.infer([self.frame()])
            tensor = run.call_args.args[1]["images"]
            np.testing.assert_allclose(tensor[0, :, 16, 0], np.array([30, 20, 10]) / 255, atol=1e-6)
            np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255, atol=1e-6)
            self.assertEqual(result["provenance"]["providers"], ["CPUExecutionProvider"])
            self.assertEqual(len(result["provenance"]["model_sha256"]), 64)
            record = result["frames"][0]
            self.assertEqual(record["frame_id"], 37)
            a, b = record["candidates"]
            self.assertEqual(a["class_name"], "apple")
            self.assertEqual(a["box_xyxy_px"], [8, 4, 24, 20])
            self.assertEqual(decode_mask(a["mask"]).shape, (32, 64))
            self.assertEqual(int(decode_mask(a["mask"]).sum()), 256)
            self.assertEqual(pairwise_spatial_evidence(a, b)["mask_intersection_px"], 96)
            json.dumps(result, allow_nan=False)

    def test_no_detections_and_empty_mask_preserve_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            expert = self.make_expert(tmp, pred=np.zeros((1, 3, 8), np.float32))
            self.assertEqual(expert.infer([self.frame()])["frames"][0]["status"], "no_detections")
            expert = self.make_expert(tmp, proto=-np.ones((1, 2, 16, 16), np.float32))
            self.assertEqual(expert.infer([self.frame()])["frames"][0]["candidates"][0]["mask_status"], "empty")

    def test_reject_raw_outputs_wrong_task_invalid_class_and_nonfinite_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            for pred in (np.zeros((1, 8, 40), np.float32),
                         np.array([[[1, 1, 9, 9, .9, 9, 1, 0]]], np.float32),
                         np.full((1, 1, 8), np.nan, np.float32)):
                with self.assertRaises(ValueError):
                    self.make_expert(tmp, pred=pred).infer([self.frame()])
            with self.assertRaises(ValueError):
                self.make_expert(tmp, task="detect")
            with self.assertRaises(ValueError):
                self.make_expert(tmp, names="{4: 'apple'}")

    def test_original_frame_ids_pts_and_confidence_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            expert = self.make_expert(tmp)
            with self.assertRaises(ValueError):
                expert.infer([self.frame(), self.frame()])
            with self.assertRaises(ValueError):
                expert.infer([VideoFrame(3, float("nan"), self.frame().image)])
            with self.assertRaises(ValueError):
                ObjectSpatialConfig(confidence_threshold=0)


if __name__ == "__main__":
    unittest.main()
