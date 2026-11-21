"""Image-plane measurements between two candidates in the same frame."""
import math

import numpy as np

from .masks import decode_mask


def pairwise_spatial_evidence(a, b):
    """No entity binding, physical relation or completion verdict is inferred."""
    if a["frame_id"] != b["frame_id"] or a["pts_seconds"] != b["pts_seconds"]:
        raise ValueError("spatial measurements require candidates from the same frame/PTS")
    boxes = np.asarray([a["box_xyxy_px"], b["box_xyxy_px"]], dtype=float)
    if (boxes.shape != (2, 4) or not np.isfinite(boxes).all()
            or (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("invalid pixel boxes")
    lo = np.maximum(boxes[0, :2], boxes[1, :2])
    hi = np.minimum(boxes[0, 2:], boxes[1, 2:])
    intersection = float(np.maximum(hi - lo, 0).prod())
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    masks = [decode_mask(item["mask"]) for item in (a, b)]
    if masks[0].shape != masks[1].shape:
        raise ValueError("masks must use the same original-resolution canvas")
    mask_areas = [int(mask.sum()) for mask in masks]
    overlap = int(np.logical_and(*masks).sum())
    union = mask_areas[0] + mask_areas[1] - overlap
    return {
        "candidate_ids": [a["candidate_id"], b["candidate_id"]],
        "frame_id": a["frame_id"], "pts_seconds": a["pts_seconds"],
        "center_distance_px": float(np.linalg.norm(centers[0] - centers[1])),
        "box_edge_distance_px": math.hypot(*np.maximum(lo - hi, 0)),
        "box_iou": intersection / float(areas.sum() - intersection),
        "signed_box_gaps_px": {"a_left_of_b": float(boxes[1, 0] - boxes[0, 2]),
                               "a_above_b": float(boxes[1, 1] - boxes[0, 3])},
        "mask_status": "available" if all(mask_areas) else "missing",
        "mask_intersection_px": overlap if all(mask_areas) else None,
        "mask_iou": overlap / union if all(mask_areas) else None,
        "limitations": ["2D image measurements do not establish contact, support, grasp or release."],
    }
