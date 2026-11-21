"""Run with python -m benchmark --help from the workspace root."""
import argparse
import json
from pathlib import Path
import sys

from .common import digest, now, read_json, write_json
from .data import REVISION, download, hub_url, prepare, transfer
from .preflight import preflight
from .report import create_report, joined_rows
from .runner import run, run_lock


def main():
    parser = argparse.ArgumentParser(description="Validation-only JEV evaluation: preflight → pilot → development")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-metadata", help="download the pinned official val metadata; no model calls")
    fetch.add_argument("--revision", default=REVISION)
    fetch.add_argument("--out", type=Path, required=True)
    prep = commands.add_parser("prepare", help="create label-separated nested subsets; offline")
    prep.add_argument("--metadata", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--revision", default=REVISION)
    prep.add_argument("--size", type=int, default=50)
    prep.add_argument("--pilot-size", type=int, default=10)
    prep.add_argument("--seed", type=int, default=20261121)
    prep.add_argument("--durations", type=Path, help="optional CSV: file_name,duration_s for validation")
    dl = commands.add_parser("download", help="download only selected videos with neutral names; no model calls")
    dl.add_argument("--dataset", type=Path, required=True)
    dl.add_argument("--local-root", type=Path, help="copy from a local val directory instead of downloading")
    dl.add_argument("--phase", choices=("pilot", "development"), default="pilot")
    check = commands.add_parser("preflight", help="readiness checks and optional upstream tests; no model calls")
    check.add_argument("--framework-root", type=Path, required=True)
    check.add_argument("--python", default=sys.executable)
    check.add_argument("--yolo-model", type=Path)
    check.add_argument("--dataset", type=Path)
    check.add_argument("--phase", choices=("pilot", "development"), default="pilot")
    check.add_argument("--run-tests", action="store_true")
    check.add_argument("--out", type=Path, required=True)
    runner = commands.add_parser("run", help="dry-run by default; --execute enables paid APIs")
    runner.add_argument("--dataset", type=Path, required=True)
    runner.add_argument("--framework-root", type=Path, required=True)
    runner.add_argument("--yolo-model", type=Path, required=True)
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--python", default=sys.executable)
    runner.add_argument("--phase", choices=("pilot", "development"), default="pilot")
    runner.add_argument("--execute", action="store_true")
    runner.add_argument("--timeout-seconds", type=float, default=1800.)
    runner.add_argument("--sample-interval-seconds", type=float, default=2.)
    runner.add_argument("--threshold-config", type=Path)
    runner.add_argument("--max-calls-per-episode", type=int)
    runner.add_argument("--max-new-episodes", type=int)
    runner.add_argument("--batch-seconds", type=float, help="stop between episodes after this wall time; not a billing cap")
    report = commands.add_parser("report", help="join labels after predictions and build offline reports")
    report.add_argument("--dataset", type=Path, required=True)
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--out", type=Path)
    report.add_argument("--phase", choices=("pilot", "development"), default="development")
    report.add_argument("--bootstrap-iterations", type=int, default=2000)
    report.add_argument("--rates", type=Path, help="optional user-supplied provider USD rate JSON")
    freeze = commands.add_parser("freeze", help="seal the completed development configuration and report")
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fetch-metadata":
            if args.out.exists():
                raise ValueError("metadata destination already exists; preserve the old snapshot")
            transfer(hub_url(args.revision), args.out)
            result = {"path": str(args.out.resolve()), "sha256": digest(args.out), "revision": args.revision}
        elif args.command == "prepare":
            result = prepare(args.metadata, args.out, revision=args.revision, size=args.size,
                             pilot_size=args.pilot_size, seed=args.seed, durations=args.durations)
        elif args.command == "download":
            result = download(args.dataset, stage=args.phase, local_root=args.local_root)
            if result["failures"]:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 2
        elif args.command == "preflight":
            result = preflight(args.framework_root, python=args.python, model=args.yolo_model,
                               dataset=args.dataset, stage=args.phase, run_tests=args.run_tests, out=args.out)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            # Offline tests can pass while live inputs/keys remain unavailable.
            return 0 if result["ready_for_live"] else 2
        elif args.command == "run":
            try:
                result = run(args.dataset, args.out, args.framework_root, args.yolo_model,
                             phase=args.phase, python=args.python, execute=args.execute,
                             timeout=args.timeout_seconds, interval=args.sample_interval_seconds,
                             threshold_config=args.threshold_config, max_calls=args.max_calls_per_episode,
                             max_new_episodes=args.max_new_episodes, batch_seconds=args.batch_seconds)
            except KeyboardInterrupt:
                if (args.out / "configuration.json").exists():
                    create_report(args.dataset, args.out, phase=args.phase)
                print("Stopped; journals and partial report retained. No automatic retries.", file=sys.stderr)
                return 130
            if args.execute:
                result["report"] = create_report(args.dataset, args.out, phase=args.phase)
        elif args.command == "report":
            if args.bootstrap_iterations < 0:
                raise ValueError("bootstrap iterations must be nonnegative")
            result = create_report(args.dataset, args.run, args.out, phase=args.phase,
                                   iterations=args.bootstrap_iterations,
                                   rates=read_json(args.rates) if args.rates else None)
        elif args.command == "freeze":
            with run_lock(args.run):
                if (args.run / "freeze.json").exists():
                    raise ValueError("this run is already frozen")
                _, configuration, rows = joined_rows(args.dataset, args.run, "development")
                if any(r["status"] in ("not_run", "interrupted") for r in rows):
                    raise ValueError("development contains unrun/interrupted samples; cannot seal a completed development run")
                result = create_report(args.dataset, args.run, args.run / "frozen-report", phase="development")
                write_json(args.run / "freeze.json", {"frozen_at": now(),
                           "configuration_sha256": digest(args.run / "configuration.json"),
                           "metrics_sha256": digest(result["metrics"]), "scope": "validation development only"})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
