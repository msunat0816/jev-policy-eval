"""Small, dependency-free persistence and provenance helpers."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                                  for r in rows), encoding="utf-8")


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("path must stay inside its dataset/run directory")
    return path


def framework_fingerprint(root):
    root = Path(root).resolve()
    if not (root / "evaluate_rollout.py").is_file():
        raise ValueError("framework root must contain evaluate_rollout.py")
    files = list(root.glob("*.py")) + list(root.glob("requirements*.txt"))
    for directory in ("experts", "prompts"):
        files += [p for p in (root / directory).rglob("*")
                  if p.is_file() and p.suffix in (".py", ".json", ".txt")]
    hashes = {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)}
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                    text=True, timeout=10, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"root": str(root), "code_sha256": object_hash(hashes), "files": hashes,
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short")}
