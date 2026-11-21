"""Metrics with explicit missing predictions and exploratory grouped intervals."""
from collections import Counter, defaultdict
import math
import random
import statistics


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def quality(rows):
    n = len(rows)
    scored = [r for r in rows if r["status"] == "scored"]
    errors = [abs(r["score"] - r["reward"]) for r in scored]
    matrix = [[0] * 5 for _ in range(5)]
    for row in scored:
        matrix[row["reward"] - 1][row["score"] - 1] += 1
    total_error = sum(errors)
    unknown_worst = sum(max(r["reward"] - 1, 5 - r["reward"])
                        for r in rows if r["status"] != "scored")
    return {"n": n, "scored": len(scored), "coverage": len(scored) / n if n else None,
            "selective_mae": statistics.mean(errors) if errors else None,
            "exact_accuracy_scored": sum(e == 0 for e in errors) / len(errors) if errors else None,
            "within_one_scored": sum(e <= 1 for e in errors) / len(errors) if errors else None,
            "large_error_rate_scored": sum(e >= 2 for e in errors) / len(errors) if errors else None,
            "answered_and_correct_all": sum(e == 0 for e in errors) / n if n else None,
            "all_sample_mae": total_error / n if n and len(scored) == n else None,
            "hypothetical_all_sample_mae_bounds": [total_error / n, (total_error + unknown_worst) / n] if n else None,
            "confusion_true_rows_predicted_columns": matrix,
            "absolute_error_counts_scored": dict(Counter(str(e) for e in errors))}


def grouped(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: quality(value) for key, value in sorted(groups.items())}


def bootstrap_mae(rows, *, iterations=2000, seed=20261121):
    groups = defaultdict(list)
    for row in rows:
        groups[row["origin_group"]].append(row)
    if len(groups) < 2 or iterations < 1:
        return {"ci95": None, "groups": len(groups), "iterations": iterations,
                "reason": "at least two origin groups and one iteration required"}
    rng = random.Random(seed)
    values = list(groups.values())
    estimates = []
    for _ in range(iterations):
        sampled = [row for group in rng.choices(values, k=len(values)) for row in group]
        errors = [abs(r["score"] - r["reward"]) for r in sampled if r["status"] == "scored"]
        if errors:
            estimates.append(statistics.mean(errors))
    return {"ci95": [quantile(estimates, .025), quantile(estimates, .975)] if estimates else None,
            "groups": len(groups), "iterations": iterations, "valid_resamples": len(estimates),
            "seed": seed, "unit": "heuristic origin_group",
            "scope": "conditional on scored predictions; exploratory validation subset"}


def token_counts(usage):
    usage = usage if isinstance(usage, dict) else {}
    def count(primary, alternative):
        value = usage.get(primary, usage.get(alternative))
        return value if type(value) is int and value >= 0 else None
    return count("input_tokens", "prompt_tokens"), count("output_tokens", "completion_tokens")


def validate_rates(rates):
    if rates is None:
        return None
    if not isinstance(rates, dict) or not rates:
        raise ValueError("rates must be a nonempty provider mapping")
    allowed = {"input_usd_per_million", "output_usd_per_million", "per_request_usd"}
    for provider, values in rates.items():
        if provider not in ("qwen", "typesafe") or not isinstance(values, dict) or not values or set(values) - allowed:
            raise ValueError("invalid provider or pricing fields")
        for number in values.values():
            if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
                raise ValueError("rates must be finite nonnegative numbers")
    return rates


def resources(records, rates=None):
    validate_rates(rates)
    calls = [call for record in records for call in record.get("calls", [])]
    attempted = [c for c in calls if c.get("transport_attempted")]
    providers = {}
    for provider in sorted({c["provider"] for c in attempted}):
        subset = [c for c in attempted if c["provider"] == provider]
        tokens = [token_counts(c.get("usage")) for c in subset]
        providers[provider] = {"calls": len(subset),
                               "reported_input_tokens": sum(a for a, _ in tokens if a is not None),
                               "reported_output_tokens": sum(b for _, b in tokens if b is not None),
                               "calls_missing_input_tokens": sum(a is None for a, _ in tokens),
                               "calls_missing_output_tokens": sum(b is None for _, b in tokens)}
    estimate, unpriced = 0., 0
    for call in attempted:
        rate = (rates or {}).get(call["provider"])
        if rate is None:
            unpriced += 1
            continue
        input_tokens, output_tokens = token_counts(call.get("usage"))
        if (("input_usd_per_million" in rate and input_tokens is None)
                or ("output_usd_per_million" in rate and output_tokens is None)):
            unpriced += 1
            continue
        estimate += rate.get("per_request_usd", 0)
        estimate += (input_tokens or 0) * rate.get("input_usd_per_million", 0) / 1e6
        estimate += (output_tokens or 0) * rate.get("output_usd_per_million", 0) / 1e6
    latencies = [r["elapsed_seconds"] for r in records if r.get("elapsed_seconds") is not None]
    stage_rows = defaultdict(list)
    for record in records:
        for stage in record.get("stages", []):
            stage_rows[stage["stage"]].append(stage)
    by_stage = {}
    for name, stages in sorted(stage_rows.items()):
        times = [s["elapsed_seconds"] for s in stages if s.get("elapsed_seconds") is not None]
        by_stage[name] = {"episodes": len(stages), "p50_seconds": quantile(times, .5),
                          "p95_seconds": quantile(times, .95),
                          "calls_attempted": sum(c.get("transport_attempted", False) for c in calls if c["stage"] == name)}
    return {"transport_attempted_calls": len(attempted), "prepared_calls": len(calls),
            "in_flight_calls_at_stop": sum(c.get("status") == "in_flight" for c in calls),
            "image_presentations_attempted": sum(c["images"] for c in attempted),
            "request_bytes_attempted": sum(c["request_bytes"] for c in attempted),
            "providers": providers, "by_stage": by_stage, "latency_p50_seconds": quantile(latencies, .5),
            "latency_p95_seconds": quantile(latencies, .95),
            "total_episode_wall_seconds": sum(latencies),
            "episode_disk_bytes": sum(r.get("disk_bytes", 0) for r in records),
            "estimated_usd_known_calls": estimate if rates else None,
            "calls_not_priced": unpriced,
            "pricing_note": "user-supplied rate estimate, not a provider invoice; missing usage is never treated as zero"}


def evaluate(rows, *, iterations=2000, rates=None):
    overall = quality(rows)
    by_source = grouped(rows, "source")
    available_group_maes = [v["selective_mae"] for v in by_source.values() if v["selective_mae"] is not None]
    all_complete = bool(rows) and all(r["status"] == "scored" for r in rows)
    attempted = [r for r in rows if r["status"] != "not_run"]
    complete = sum(r["status"] in ("scored", "insufficient_evidence") for r in rows)
    return {"scope": "exploratory validation subset, not RoboRewardBench test/leaderboard replication",
            "overall": overall, "status_counts": dict(Counter(r["status"] for r in rows)),
            "attempted": len(attempted), "completed": complete,
            "completion_rate_attempted": complete / len(attempted) if attempted else None,
            "abstention_rate_attempted": sum(r["status"] == "insufficient_evidence" for r in rows) / len(attempted) if attempted else None,
            "source_macro_mae_all_covered": statistics.mean(available_group_maes) if all_complete else None,
            "source_macro_selective_mae_available_groups": statistics.mean(available_group_maes) if available_group_maes else None,
            "sources_with_scores": len(available_group_maes), "sources_total": len(by_source),
            "by_source": by_source, "by_reward": grouped(rows, "reward"),
            "by_duration": grouped(rows, "duration_bin"),
            "failed_stage_counts": dict(Counter(r.get("failed_stage", "unknown") for r in rows
                                                 if r["status"] in ("failed", "timeout", "interrupted"))),
            "mae_cluster_bootstrap": bootstrap_mae(rows, iterations=iterations),
            "resources": resources(rows, rates),
            "missing_prediction_policy": "keep all planned IDs; do not impute scores or drop failures/abstentions; bounds describe hypothetical completions, not observed scores"}
