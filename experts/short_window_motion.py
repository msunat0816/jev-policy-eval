"""Mask-seeded pyramidal LK, forward/backward validation and explicit gaps."""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from .types import MaskSeed, validate_frames, validate_mask


@dataclass(frozen=True)
class MotionConfig:
    max_points: int = 200
    min_points: int = 6
    quality_level: float = 0.01
    min_distance_px: float = 3.0
    window_size_px: int = 21
    pyramid_levels: int = 3
    max_fb_error_px: float = 1.0
    max_lk_error: float = 30.0

    def __post_init__(self):
        for name in ("max_points", "min_points", "window_size_px", "pyramid_levels"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if (not 1 <= self.min_points <= self.max_points or self.window_size_px < 3
                or self.window_size_px % 2 == 0 or self.pyramid_levels < 0):
            raise ValueError("invalid feature count or LK window/pyramid configuration")
        for name in ("quality_level", "min_distance_px", "max_fb_error_px", "max_lk_error"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.quality_level > 1:
            raise ValueError("quality_level must be <= 1")


def _motion(vector):
    dx, dy = map(float, vector)
    length = math.hypot(dx, dy)
    return {"displacement_xy_px": [dx, dy], "magnitude_px": length,
            "direction_degrees_image": math.degrees(math.atan2(dy, dx)) if length > 1e-6 else None}


def _sample(frame, valid, reason=None, step=None, total=None):
    return {"frame_id": frame.frame_id, "pts_seconds": frame.pts_seconds,
            "status": "available" if reason is None else "missing", "reason": reason,
            "valid_points": valid, "step": _motion(step) if step is not None else None,
            "from_seed": _motion(total) if total is not None else None}


class ShortWindowMotionExpert:
    expert_id = "short_window_motion"

    def __init__(self, config=None):
        self.config = config or MotionConfig()

    def track(self, frames, target=None, *, reference=None):
        """Caller supplies a reliable first-frame mask and its entity binding/source.

        No box fallback, implicit identity matching, interpolation or automatic reseeding.
        A reference mask permits relative image motion; it is not camera calibration.
        """
        frames = validate_frames(frames, continuous=True)
        if len(frames) < 2:
            raise ValueError("short-window motion requires at least two continuous frames")
        grays = [cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY) for frame in frames]
        target_result = self._track_region(frames, grays, target)
        reference_result = self._track_region(frames, grays, reference) if reference is not None else None
        relative = []
        for i, frame in enumerate(frames):
            subject = target_result["samples"][i]
            anchor = reference_result["samples"][i] if reference_result else None
            valid = anchor is not None and subject["status"] == anchor["status"] == "available"
            relative.append({
                "frame_id": frame.frame_id, "pts_seconds": frame.pts_seconds,
                "status": "available" if valid else "missing",
                "reason": None if valid else ("reference_not_supplied" if anchor is None else "region_track_missing"),
                "target_valid_points": subject["valid_points"],
                "reference_valid_points": anchor["valid_points"] if anchor else 0,
                "step": _motion(np.subtract(subject["step"]["displacement_xy_px"],
                                            anchor["step"]["displacement_xy_px"])) if valid and i else None,
                "from_seed": _motion(np.subtract(subject["from_seed"]["displacement_xy_px"],
                                                 anchor["from_seed"]["displacement_xy_px"])) if valid else None,
            })
        return {"protocol": "short-window-motion-evidence-v1", "expert_id": self.expert_id,
                "frame_range": [frames[0].frame_id, frames[-1].frame_id],
                "coordinate_system": "original_image_pixels_x_right_y_down",
                "provenance": {"opencv_version": cv2.__version__, "numpy_version": np.__version__,
                               "algorithm": "pyramidal_lk_forward_backward", "config": asdict(self.config)},
                "target": target_result, "reference": reference_result, "relative_motion": relative,
                "limitations": [
                    "Caller-supplied mask/entity binding is a prerequisite, not verified by optical flow.",
                    "Texture loss, occlusion or drift can invalidate tracks even with forward/backward checks.",
                    "Image displacement includes camera motion; a reference does not remove general perspective effects.",
                    "Point motion is not an object center or a proof of grasp, release or physical stability.",
                ]}

    def _track_region(self, frames, grays, seed):
        cfg = self.config
        points = None
        reason = "mask_not_supplied"
        if seed is not None:
            if (not isinstance(seed, MaskSeed)
                    or not isinstance(seed.entity_key, str) or not seed.entity_key.strip()
                    or not isinstance(seed.source, str) or not seed.source.strip()
                    or type(seed.frame_id) is not int or seed.frame_id != frames[0].frame_id):
                raise ValueError("mask seed requires an entity, source and the first original frame ID")
            mask = validate_mask(seed.mask, grays[0].shape).astype(np.uint8) * 255
            points = cv2.goodFeaturesToTrack(grays[0], mask=mask, maxCorners=cfg.max_points,
                                            qualityLevel=cfg.quality_level, minDistance=cfg.min_distance_px)
            reason = "insufficient_feature_points"
        count = 0 if points is None else len(points)
        live = count >= cfg.min_points
        origins = points.copy() if live else None
        samples = [_sample(frames[0], count, None if live else reason,
                           total=[0, 0] if live else None)]
        interruptions = []
        if not live:
            interruptions.append({"frame_range": [frames[0].frame_id, frames[-1].frame_id],
                                  "pts_range_seconds": [frames[0].pts_seconds, frames[-1].pts_seconds],
                                  "reason": reason})
        params = {"winSize": (cfg.window_size_px, cfg.window_size_px), "maxLevel": cfg.pyramid_levels,
                  "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
        for i in range(1, len(frames)):
            if not live:
                samples.append(_sample(frames[i], 0, "track_unavailable_after_loss" if origins is not None else reason))
                continue
            q, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(grays[i - 1], grays[i], points, None, **params)
            if q is None or forward_status is None or forward_error is None:
                keep = np.zeros(len(points), dtype=bool)
            else:
                finite_forward = np.isfinite(q).all(axis=(1, 2))
                # Never pass nonfinite coordinates into the backward OpenCV call.
                safe_q = np.where(np.isfinite(q), q, points).astype(np.float32)
                back, back_status, back_error = cv2.calcOpticalFlowPyrLK(grays[i], grays[i - 1], safe_q, None, **params)
                keep = np.zeros(len(points), dtype=bool)
                if back is not None and back_status is not None and back_error is not None:
                    xy, old = q.reshape(-1, 2), points.reshape(-1, 2)
                    height, width = grays[i].shape
                    keep = (finite_forward & np.isfinite(back).all(axis=(1, 2))
                            & forward_status.ravel().astype(bool) & back_status.ravel().astype(bool)
                            & (forward_error.ravel() <= cfg.max_lk_error)
                            & (back_error.ravel() <= cfg.max_lk_error)
                            & (np.linalg.norm(back.reshape(-1, 2) - old, axis=1) <= cfg.max_fb_error_px)
                            & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height))
            count = int(keep.sum())
            if count < cfg.min_points:
                live = False
                samples.append(_sample(frames[i], count, "insufficient_consistent_points"))
                interruptions.append({"frame_range": [frames[i - 1].frame_id, frames[-1].frame_id],
                                      "pts_range_seconds": [frames[i - 1].pts_seconds, frames[-1].pts_seconds],
                                      "reason": "insufficient_consistent_points"})
                continue
            new = q[keep].reshape(-1, 2)
            old = points[keep].reshape(-1, 2)
            origins = origins[keep]
            samples.append(_sample(frames[i], count, step=np.median(new - old, axis=0),
                                   total=np.median(new - origins.reshape(-1, 2), axis=0)))
            points = new.reshape(-1, 1, 2)
        available = sum(sample["status"] == "available" for sample in samples)
        return {"entity_key": seed.entity_key if seed else None, "mask_source": seed.source if seed else None,
                "seed_frame_id": frames[0].frame_id,
                "status": "complete" if available == len(frames) else "partial" if available else "missing",
                "samples": samples, "interruptions": interruptions}
