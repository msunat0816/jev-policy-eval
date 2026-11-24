"""Prepare a single rubric Choice from fixed whole-episode image samples."""
import base64
import json
from pathlib import Path
import uuid

from .common import inside, read_json, write_json


METHOD = "direct-jev"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "jev_direct_images.txt"
SAMPLE_INTERVAL_SECONDS = 2.0
MAX_FRAMES = 128


def select_pilot(inputs, video_info, size):
    if not 2 <= size <= len(inputs):
        raise ValueError("duration pilot must contain at least two samples")
    if any(video_info[row["sample_id"]]["duration_s"] is None for row in inputs):
        raise ValueError("duration pilot requires a measured duration for every validation sample")
    ordered = sorted((video_info[row["sample_id"]]["duration_s"], row["sample_id"]) for row in inputs)
    return [ordered[i * (len(ordered) - 1) // (size - 1)][1] for i in range(size)]


def prepare_request(ev, job, folder):
    import cv2
    from check_judgment import prepare_image

    times = ev.frame_times(job["video"])
    indices = ev.overview_indices(times, SAMPLE_INTERVAL_SECONDS)
    frames = [{"frame_id": index, "pts_seconds": times[index]} for index in indices]
    sampling = {"interval_seconds": SAMPLE_INTERVAL_SECONDS, "video_frame_count": len(times),
                "first_pts_seconds": times[0], "last_pts_seconds": times[-1],
                "frames": frames, "image_count": len(frames), "max_frames": MAX_FRAMES}
    write_json(folder / "sampling.json", sampling)
    if len(frames) > MAX_FRAMES:
        raise ValueError(f"sampled {len(frames)} frames, above the {MAX_FRAMES}-image context guard")

    (folder / "frames").mkdir()
    (folder / "prepared").mkdir()
    state = {"instruction": job["task"], "rubric": ev.RUBRIC,
             "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
             "episode": {"first_frame_id": 0, "last_frame_id": len(times) - 1},
             "frames": []}
    assets, manifest = {}, {}
    cap = cv2.VideoCapture(job["video"])
    try:
        if not cap.isOpened():
            raise ValueError("video could not be opened")
        next_frame = 0
        for frame in frames:
            frame_id = frame["frame_id"]
            # Advance in presentation order so the ffprobe frame IDs also work
            # for variable-frame-rate videos and decoders with inexact seeks.
            for skipped in range(next_frame, frame_id):
                if not cap.grab():
                    raise ValueError(f"video decode failed at frame {skipped}")
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"video decode failed at frame {frame_id}")
            next_frame = frame_id + 1
            source = f"frames/{frame_id:07d}.jpg"
            if not cv2.imwrite(str(folder / source), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise ValueError(f"JPEG encode failed at frame {frame_id}")
            data, metadata = prepare_image(folder, source, f"prepared/{frame_id:07d}.jpg", kind="original")
            ref = str(uuid.uuid4())
            assets[ref] = {"type": "image", "format": "jpg", "data_base64": base64.b64encode(data).decode("ascii")}
            manifest[ref] = {**frame, **metadata}
            state["frames"].append({**frame, "image": ref})
        if cap.grab():
            raise ValueError("decoded video has more frames than the presentation timestamps")
    finally:
        cap.release()

    body = {"model": ev.FINAL_MODEL, "state": state, "assets": assets,
            "questions": {"grade": ev.choice(PROMPT_PATH.read_text(encoding="utf-8"), dict(ev.FINAL_CRITERIA))}}
    return body, manifest


def score_request(ev, folder, body, manifest, key):
    response = ev.post(ev.JEV_URL, key, body, folder, "evaluate",
                       headers=ev.IMAGE_HEADERS, image_assets=manifest, max_body_bytes=ev.MAX_BODY_BYTES)
    return ev.parse_final_response(response)


def inspect_requests(ev, inputs, dataset, out, video_info):
    summaries = []
    out.mkdir(parents=True, exist_ok=False)
    for row in inputs:
        sample_id = row["sample_id"]
        folder = out / sample_id
        folder.mkdir()
        summary = {"sample_id": sample_id, "duration_s": video_info[sample_id]["duration_s"]}
        try:
            body, manifest = prepare_request(ev, {"video": str(inside(dataset, row["video_path"])),
                                                  "task": row["task"]}, folder)
            request_bytes = len(json.dumps(body, ensure_ascii=False, allow_nan=False).encode())
            summary["request_bytes"] = request_bytes
            if request_bytes > ev.MAX_BODY_BYTES:
                raise ValueError("prepared request exceeds the 32 MiB body limit")
            summary["status"] = "prepared"
            for asset in body["assets"].values():
                del asset["data_base64"]
            write_json(folder / "request.json", {"body": body, "image_manifest": manifest,
                                                 "headers": ev.IMAGE_HEADERS})
        except Exception as exc:
            summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if (folder / "sampling.json").exists():
            sampling = read_json(folder / "sampling.json")
            summary.update(image_count=sampling["image_count"],
                           frame_ids=[frame["frame_id"] for frame in sampling["frames"]])
        summaries.append(summary)
    result = {"model": ev.FINAL_MODEL, "model_calls": 0, "samples": summaries}
    write_json(out / "inspection.json", result)
    return result
