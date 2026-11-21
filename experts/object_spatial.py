"""YOLO26 end-to-end segmentation via ONNX Runtime's CPU provider only.

Supported export: static batch-1 FP32 NCHW; outputs [1,N,6+M] and [1,M,H,W].
Raw one-to-many YOLO outputs are rejected instead of guessed or silently decoded.
"""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .masks import encode_mask
from .types import validate_frames


@dataclass(frozen=True)
class ObjectSpatialConfig:
    confidence_threshold: float = 0.25

    def __post_init__(self):
        if (type(self.confidence_threshold) not in (int, float)
                or not math.isfinite(self.confidence_threshold)
                or not 0 < self.confidence_threshold <= 1):
            raise ValueError("confidence_threshold must be in (0, 1]")


def _class_names(raw):
    try:
        names = json.loads(raw)
    except (ValueError, TypeError):
        try:
            names = ast.literal_eval(raw)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError("model metadata must contain a valid class vocabulary") from exc
    if isinstance(names, dict):
        if set(map(str, names)) != {str(i) for i in range(len(names))}:
            raise ValueError("model class IDs must be contiguous from zero")
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("model metadata must contain a nonempty class vocabulary")
    return names


class ObjectSpatialExpert:
    expert_id = "object_spatial"

    def __init__(self, model_path, config=None):
        self.config = config or ObjectSpatialConfig()
        path = Path(model_path)
        if path.suffix.lower() != ".onnx" or not path.is_file():
            raise ValueError("provide a local .onnx segmentation model; no automatic downloads")
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("object_spatial requires CPUExecutionProvider exclusively")
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("expected exactly one image input")
        inp = inputs[0]
        if (inp.type != "tensor(float)" or len(inp.shape) != 4 or inp.shape[:2] != [1, 3]
                or any(type(v) is not int or v <= 0 for v in inp.shape[2:])):
            raise ValueError("expected static batch-1 FP32 NCHW image input")
        self.input_name = inp.name
        self.input_hw = tuple(inp.shape[2:])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("task") != "segment":
            raise ValueError("model metadata task must be segment")
        self.names = _class_names(metadata.get("names"))
        self.provenance = {
            "model_file": path.name, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "runtime": "onnxruntime", "runtime_version": ort.__version__,
            "providers": self.session.get_providers(), "opencv_version": cv2.__version__,
            "numpy_version": np.__version__, "class_names": self.names,
            "output_format": "end2end_xyxy_score_class_mask_coefficients",
            "config": asdict(self.config),
        }

    def infer(self, frames):
        """Run only the supplied necessary frames; IDs and PTS remain original."""
        frames = validate_frames(frames)
        records = [self._infer_frame(frame) for frame in frames]
        return {"protocol": "object-spatial-evidence-v1", "expert_id": self.expert_id,
                "provenance": self.provenance, "frames": records,
                "limitations": [
                    "Fixed model vocabulary; absent detections do not prove object absence.",
                    "Candidates have no task entity identity or persistent cross-frame ID.",
                    "2D boxes and masks do not establish physical contact, support, grasp or release.",
                ]}

    def _infer_frame(self, frame):
        height, width = frame.image.shape[:2]
        input_h, input_w = self.input_hw
        scale = min(input_w / width, input_h / height)
        resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (input_w - resized_w) // 2, (input_h - resized_h) // 2
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(
            frame.image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 2:
            raise ValueError("expected detection rows and mask prototypes from an end-to-end segment export")
        predictions = [out for out in outputs if out.ndim == 3]
        prototypes = [out for out in outputs if out.ndim == 4]
        if len(predictions) != 1 or len(prototypes) != 1:
            raise ValueError("unsupported segmentation output layout")
        pred, proto = predictions[0], prototypes[0]
        if (pred.shape[0] != 1 or proto.shape[0] != 1 or min(proto.shape) < 1
                or pred.shape[2] != 6 + proto.shape[1]
                or not all(np.isfinite(out).all() for out in outputs)):
            raise ValueError("invalid end-to-end segmentation tensors; raw YOLO outputs are unsupported")
        proto = proto[0]
        candidates = []
        rejected = 0
        for row_index, row in enumerate(pred[0]):
            score = float(row[4])
            if not 0 <= score <= 1:
                raise ValueError("invalid model confidence")
            if score < self.config.confidence_threshold:
                continue
            class_id = int(row[5])
            if row[5] != class_id or not 0 <= class_id < len(self.names):
                raise ValueError("invalid model class ID")
            box = row[:4].astype(float).copy()
            box[[0, 2]] = np.clip((box[[0, 2]] - left) / scale, 0, width)
            box[[1, 3]] = np.clip((box[[1, 3]] - top) / scale, 0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                rejected += 1
                continue
            logits = (row[6:] @ proto.reshape(proto.shape[0], -1)).reshape(proto.shape[1:])
            # Undo precisely the preprocessing padding before scaling to original pixels.
            logits = cv2.resize(logits, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            logits = cv2.resize(logits[top:top + resized_h, left:left + resized_w],
                                (width, height), interpolation=cv2.INTER_LINEAR)
            yy, xx = np.ogrid[:height, :width]
            mask = ((logits > 0) & (xx >= box[0]) & (xx < box[2])
                    & (yy >= box[1]) & (yy < box[3]))
            candidates.append({
                "candidate_id": f"f{frame.frame_id}:d{row_index}", "frame_id": frame.frame_id,
                "pts_seconds": frame.pts_seconds, "class_id": class_id, "class_name": self.names[class_id],
                "confidence": score, "box_xyxy_px": box.tolist(), "mask": encode_mask(mask),
                "mask_status": "available" if mask.any() else "empty",
            })
        return {"frame_id": frame.frame_id, "pts_seconds": frame.pts_seconds,
                "image_size_hw": [height, width], "coordinate_system": "original_image_pixels_xyxy",
                "status": "candidates" if candidates else "no_detections",
                "rejected_degenerate_boxes": rejected, "candidates": candidates}
