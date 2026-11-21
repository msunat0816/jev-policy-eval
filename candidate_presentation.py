"""Deterministic views of existing candidates; no detection or entity binding."""
import cv2
import numpy as np

from experts.masks import decode_mask


COLORS = ((0, 220, 255), (255, 180, 40), (100, 230, 70), (220, 90, 240),
          (60, 140, 255), (255, 220, 100), (170, 240, 220), (210, 150, 120))


def render_candidates(image, frame):
    """Keep original pixels in a separate image; annotate a copy with a full-ID ledger."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or list(image.shape[:2]) != frame["image_size_hw"]):
        raise ValueError("matching: decoded image size/type does not match candidate frame")
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = [(candidate["candidate_id"],
             f"{candidate['class_name']}  score={candidate['confidence']:.3f}  mask={candidate['mask_status']}")
            for candidate in frame["candidates"]]
    header = f"Frame {frame['frame_id']}, PTS {frame['pts_seconds']:.6f} s"
    labels = [header, "No detections"] + [text for row in rows for text in row]
    ledger_width = max(360, max(cv2.getTextSize(text, font, .5, 1)[0][0] for text in labels) + 32)
    height, width = image.shape[:2]
    canvas = np.full((max(height, 100, 44 + 50 * len(rows)), width + ledger_width, 3), 24, np.uint8)
    canvas[:height, :width] = image
    cv2.putText(canvas, header, (width + 12, 24), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
    if not rows:
        cv2.putText(canvas, "No detections", (width + 12, 58), font, .5, (220, 220, 220), 1, cv2.LINE_AA)
    for i, candidate in enumerate(frame["candidates"]):
        color = COLORS[i % len(COLORS)]
        mask = decode_mask(candidate["mask"])
        if mask.shape != (height, width):
            raise ValueError("matching: mask dimensions do not match the decoded frame")
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas[:height, :width], contours, -1, color, 1)
        x1, y1, x2, y2 = candidate["box_xyxy_px"]
        a = (min(width - 1, int(x1)), min(height - 1, int(y1)))
        b = (min(width - 1, int(np.ceil(x2))), min(height - 1, int(np.ceil(y2))))
        cv2.rectangle(canvas, a, b, color, 2)
        # Full IDs are repeated in the ledger, even when image labels overlap.
        label = candidate["candidate_id"]
        (tw, th), baseline = cv2.getTextSize(label, font, .5, 1)
        label_x = max(0, min(a[0], width - tw - 4))
        label_y = max(th + 3, a[1])
        cv2.rectangle(canvas, (label_x, label_y - th - 3),
                      (label_x + tw + 4, label_y + baseline), (0, 0, 0), -1)
        cv2.putText(canvas, label, (label_x + 2, label_y), font, .5, color, 1, cv2.LINE_AA)
        row_y = 52 + 50 * i
        cv2.putText(canvas, rows[i][0], (width + 12, row_y), font, .5, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, rows[i][1], (width + 12, row_y + 20), font, .5, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas
