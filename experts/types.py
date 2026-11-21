"""Explicit original-frame coordinates and caller-supplied mask provenance."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    frame_id: int
    pts_seconds: float
    image: np.ndarray  # OpenCV BGR uint8, original resolution


@dataclass(frozen=True)
class MaskSeed:
    entity_key: str
    frame_id: int
    mask: np.ndarray
    source: str  # E.g. an externally verified candidate ID; never inferred from a class name.


def validate_frames(frames, *, continuous=False):
    frames = list(frames)
    if not frames:
        raise ValueError("at least one video frame is required")
    if any(not isinstance(frame, VideoFrame) or not isinstance(frame.image, np.ndarray) for frame in frames):
        raise ValueError("expected VideoFrame records containing NumPy images")
    shape = frames[0].image.shape
    for i, frame in enumerate(frames):
        if (type(frame.frame_id) is not int or frame.frame_id < 0
                or type(frame.pts_seconds) not in (int, float) or not math.isfinite(frame.pts_seconds)
                or not isinstance(frame.image, np.ndarray) or frame.image.dtype != np.uint8
                or frame.image.ndim != 3 or frame.image.shape[2] != 3
                or min(frame.image.shape[:2]) < 1 or frame.image.shape != shape):
            raise ValueError("expected original frame IDs, finite PTS and equal-sized BGR uint8 images")
        if i:
            prev = frames[i - 1]
            if frame.frame_id <= prev.frame_id or frame.pts_seconds < prev.pts_seconds:
                raise ValueError("frames must have increasing IDs and chronological PTS")
            if continuous and frame.frame_id != prev.frame_id + 1:
                raise ValueError("motion requires continuous original frames; gaps cannot be interpolated")
    return frames


def validate_mask(mask, shape):
    mask = np.asarray(mask)
    if (mask.shape != shape or mask.dtype.kind not in "buif" or not np.isfinite(mask).all()
            or not np.isin(mask, [0, 1, 255]).all()):
        raise ValueError("expected an original-resolution binary mask (0/1 or 0/255)")
    return mask.astype(bool)
