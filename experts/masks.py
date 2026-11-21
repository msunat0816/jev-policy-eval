"""Lossless JSON masks: uncompressed row-major RLE, starting with a zero run."""
import numpy as np

from .types import validate_mask


def encode_mask(mask):
    mask = validate_mask(mask, np.asarray(mask).shape)
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("expected a nonempty 2D mask canvas")
    flat = mask.ravel(order="C")
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, boundaries, flat.size]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"encoding": "rle_row_major", "size": list(mask.shape), "counts": counts}


def decode_mask(value):
    if not isinstance(value, dict) or set(value) != {"encoding", "size", "counts"}:
        raise ValueError("invalid mask record")
    shape, counts = value["size"], value["counts"]
    if (value["encoding"] != "rle_row_major" or not isinstance(shape, list) or len(shape) != 2
            or any(type(x) is not int or x < 1 for x in shape)
            or not isinstance(counts, list) or not counts
            or any(type(x) is not int or x < 0 for x in counts)
            or sum(counts) != shape[0] * shape[1]):
        raise ValueError("invalid row-major mask RLE")
    flat = np.zeros(sum(counts), dtype=bool)
    offset = 0
    for i, count in enumerate(counts):
        if i % 2:
            flat[offset:offset + count] = True
        offset += count
    return flat.reshape(shape)
