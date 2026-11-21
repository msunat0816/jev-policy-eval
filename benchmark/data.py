"""Prepare nested validation subsets and keep reference labels outside workers."""
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
from urllib.parse import quote
from urllib.request import urlopen

from .common import digest, inside, object_hash, read_json, read_jsonl, write_json, write_jsonl

DATASET = "teetone/RoboReward"
REVISION = "d69d5b92b8f3caee9a616b615b300bdf0d37aef2"


def check_revision(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("use an immutable 40-character Hub commit, not main")


def safe_filename(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or "?" in value or "#" in value):
        raise ValueError("invalid metadata file_name")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".mp4":
        raise ValueError("file_name must be a relative MP4 path")
    return path.as_posix()


def hub_url(revision, filename="metadata.jsonl"):
    check_revision(revision)
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/val/{quote(filename, safe='/')}"


def transfer(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    if not partial.stat().st_size:
        raise ValueError("empty download")
    partial.replace(destination)


def load_metadata(path):
    result, seen = [], set()
    for row in read_jsonl(path):
        filename = safe_filename(row.get("file_name"))
        task, reward = row.get("task"), row.get("reward")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be nonempty text")
        if type(reward) is not int or reward not in range(1, 6):
            raise ValueError("reward must be an integer from 1 through 5")
        if row.get("split", "validation") not in ("val", "validation"):
            raise ValueError("only validation data is supported in this development harness")
        identity = object_hash(["validation", filename, task])[:20]
        if identity in seen:
            raise ValueError("duplicate file_name/task pair in metadata")
        seen.add(identity)
        source = filename.split("/")[0] if "/" in filename else "unspecified"
        # A heuristic lineage key, not a claim of complete semantic deduplication.
        original = re.sub(r"_attempt_\d+_score_[1-5]$", "", PurePosixPath(filename).stem)
        result.append({"sample_id": "s" + identity, "file_name": filename, "task": task,
                       "reward": reward, "source": source,
                       "origin_group": source + "/" + original,
                       "task_group": object_hash([source, " ".join(task.lower().split())])[:20],
                       "duration_s": None})
    if not result:
        raise ValueError("metadata is empty")
    return result


def duration_bin(seconds):
    if seconds is None:
        return "unavailable"
    return "<10s" if seconds < 10 else "10–30s" if seconds < 30 else "30–60s" if seconds < 60 else ">=60s"


def prepare(metadata, out, *, revision=REVISION, size=50, pilot_size=10, seed=20261121,
            durations=None):
    check_revision(revision)
    if type(size) is not int or type(pilot_size) is not int or not 5 <= pilot_size <= 10 or size < pilot_size:
        raise ValueError("pilot size must be 5–10 and development size must include the pilot")
    rows = load_metadata(metadata)
    if size > len(rows):
        raise ValueError("requested subset is larger than metadata")
    if durations:
        import math
        with Path(durations).open(newline="", encoding="utf-8") as handle:
            times = {r["file_name"]: float(r["duration_s"]) for r in csv.DictReader(handle)}
        if any(not math.isfinite(t) or t <= 0 for t in times.values()):
            raise ValueError("duration_s must be finite and positive")
        for row in rows:
            row["duration_s"] = times.get(row["file_name"])
    rng = random.Random(seed)
    # Round-robin labels, then sources (and duration bins when available), so the
    # pilot is a prefix of the development set without reading model outcomes.
    strata = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strata[row["reward"]][(row["source"], duration_bin(row["duration_s"]))].append(row)
    queues = {}
    for label in sorted(strata):
        groups = []
        for key in sorted(strata[label]):
            candidates = sorted(strata[label][key], key=lambda r: r["sample_id"])
            rng.shuffle(candidates)
            groups.append(deque(candidates))
        rng.shuffle(groups)
        ordered = []
        while groups:
            for group in groups:
                ordered.append(group.popleft())
            groups = [g for g in groups if g]
        queues[label] = deque(ordered)
    selected = []
    while len(selected) < size:
        labels = sorted(k for k, q in queues.items() if q)
        rng.shuffle(labels)
        for label in labels:
            if len(selected) == size:
                break
            selected.append(queues[label].popleft())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    inputs, references = [], []
    for row in selected:
        video_id = "v" + hashlib.sha256(row["file_name"].encode()).hexdigest()[:20]
        inputs.append({"sample_id": row["sample_id"], "video_id": video_id,
                       "video_path": f"videos/{video_id}.mp4", "task": row["task"], "split": "validation"})
        references.append({k: v for k, v in row.items() if k != "task"})
    write_jsonl(out / "inputs.jsonl", inputs)
    write_jsonl(out / "labels.jsonl", references)
    plan = {"protocol": "jev-validation-development-v1", "dataset": DATASET,
            "revision": revision, "hub_split": "val", "framework_split": "validation",
            "metadata_sha256": digest(metadata), "input_sha256": digest(out / "inputs.jsonl"),
            "labels_sha256": digest(out / "labels.jsonl"), "seed": seed,
            "size": size, "pilot_size": pilot_size,
            "pilot_ids": [r["sample_id"] for r in inputs[:pilot_size]],
            "development_ids": [r["sample_id"] for r in inputs],
            "sampling": "label/source round-robin; add duration strata only when supplied; exploratory, not population weighted",
            "duration_metadata_sha256": digest(durations) if durations else None,
            "reference_counts": dict(Counter(str(r["reward"]) for r in selected)),
            "source_count": len({r["source"] for r in selected}),
            "scope": "offline preflight, real pilot, validation development only; no baselines or long-horizon audit"}
    write_json(out / "plan.json", plan)
    return plan


def load_dataset(root):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    if plan.get("protocol") != "jev-validation-development-v1" or plan.get("framework_split") != "validation":
        raise ValueError("expected a validation development plan")
    for name, key in (("inputs.jsonl", "input_sha256"), ("labels.jsonl", "labels_sha256")):
        if digest(root / name) != plan[key]:
            raise ValueError(f"dataset changed after preparation: {name}")
    inputs, labels = read_jsonl(root / "inputs.jsonl"), read_jsonl(root / "labels.jsonl")
    if [r["sample_id"] for r in inputs] != plan["development_ids"] or len(inputs) != plan["size"]:
        raise ValueError("plan and input IDs disagree")
    references = {r["sample_id"]: r for r in labels}
    if len(references) != len(inputs) or set(references) != set(plan["development_ids"]):
        raise ValueError("references and input IDs disagree")
    for row in inputs:
        if set(row) != {"sample_id", "video_id", "video_path", "task", "split"} or row["split"] != "validation":
            raise ValueError("input records must contain only the approved label-free fields")
        inside(root, row["video_path"])
    return plan, inputs, references


def download(root, *, stage="pilot", local_root=None):
    root = Path(root).resolve()
    plan, inputs, references = load_dataset(root)
    ids = set(plan["pilot_ids"] if stage == "pilot" else plan["development_ids"])
    inventory_path = root / "video-inventory.json"
    inventory = read_json(inventory_path) if inventory_path.exists() else {}
    failures = []
    for row in inputs:
        if row["sample_id"] not in ids:
            continue
        path = inside(root, row["video_path"])
        video_id = row["video_id"]
        try:
            previous = inventory.get(video_id)
            if previous and path.is_file() and digest(path) == previous["sha256"]:
                continue
            if path.exists():
                raise ValueError("existing video has no matching verified inventory; will not overwrite")
            name = safe_filename(references[row["sample_id"]]["file_name"])
            if local_root:
                source = inside(local_root, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, path.with_suffix(".mp4.partial"))
                path.with_suffix(".mp4.partial").replace(path)
            else:
                transfer(hub_url(plan["revision"], name), path)
            if not path.stat().st_size:
                raise ValueError("empty video")
            inventory[video_id] = {"sha256": digest(path), "bytes": path.stat().st_size,
                                   "video_path": row["video_path"]}
            write_json(inventory_path, inventory)
        except (OSError, ValueError) as exc:
            failures.append({"video_id": video_id, "error_type": type(exc).__name__, "error": str(exc)})
    write_json(root / f"download-{stage}.json", {"stage": stage, "failures": failures,
                                                "verified_videos": len(inventory)})
    return {"failures": failures, "verified_videos": len(inventory)}
