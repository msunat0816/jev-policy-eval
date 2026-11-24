"""Offline JSON/CSV and self-contained HTML reports; no external assets."""
import csv
import html
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from .common import digest, read_json, write_json
from .data import duration_bin, load_dataset
from .direct_jev import METHOD
from .metrics import evaluate, quality, token_counts


def joined_rows(dataset, run, phase):
    plan, inputs, references = load_dataset(dataset)
    configuration = read_json(Path(run) / "configuration.json")
    if configuration["plan_sha256"] != digest(Path(dataset) / "plan.json"):
        raise ValueError("run was not created from this dataset plan")
    from .common import object_hash
    pilot_ids = configuration.get("pilot_ids", plan["pilot_ids"])
    if (len(pilot_ids) != plan["pilot_size"] or len(set(pilot_ids)) != len(pilot_ids)
            or not set(pilot_ids).issubset(plan["development_ids"])):
        raise ValueError("run pilot IDs do not match the validation plan")
    ids = set(pilot_ids if phase == "pilot" else plan["development_ids"])
    rows = []
    for sample in inputs:
        if sample["sample_id"] not in ids:
            continue
        path = Path(run) / "episodes" / sample["sample_id"] / "record.json"
        prediction = read_json(path) if path.exists() else {"status": "not_run", "score": None}
        if path.exists() and prediction.get("configuration_sha256") != object_hash(configuration):
            raise ValueError("prediction configuration does not match run")
        if prediction["status"] == "running":
            prediction = {**prediction, "status": "interrupted", "score": None,
                          "error": "journal was still running when report was created; inspect before resuming"}
        if prediction["status"] not in {"scored", "insufficient_evidence", "failed", "timeout", "interrupted", "not_run"}:
            raise ValueError("invalid prediction status")
        score = prediction.get("score")
        if prediction["status"] == "scored":
            if type(score) is not int or score not in range(1, 6):
                raise ValueError("invalid numeric score in journal")
        elif score is not None:
            raise ValueError("unscored journal has a numeric score")
        reference = references[sample["sample_id"]]
        duration = (prediction.get("video_info") or {}).get("duration_s")
        if duration is None:
            duration = reference["duration_s"]
        # Reference labels are joined here, after per-episode predictions exist.
        rows.append({**prediction, "sample_id": sample["sample_id"], "task": sample["task"],
                     "reward": reference["reward"], "source": reference["source"],
                     "origin_group": reference["origin_group"], "task_group": reference["task_group"],
                     "duration_s": duration, "duration_bin": duration_bin(duration)})
    return plan, configuration, rows


def compare_runs(dataset, direct_run, framework_run, out):
    _, direct_config, direct_rows = joined_rows(dataset, direct_run, "development")
    _, framework_config, framework_rows = joined_rows(dataset, framework_run, "development")
    if direct_config.get("method") != METHOD or framework_config.get("method", "framework") != "framework":
        raise ValueError("expected a direct-jev run and a framework run")
    rubric_file = "prompts/roboreward_rubric.json"
    rubric = direct_config["framework"]["files"][rubric_file]
    if rubric != framework_config["framework"]["files"][rubric_file]:
        raise ValueError("direct-jev and framework rubric files differ")
    direct_model = read_json(Path(direct_run) / "runtime.json")["models"]["jev"]
    framework_model = read_json(Path(framework_run) / "runtime.json")["models"]["jev"]
    if direct_model != framework_model:
        raise ValueError("direct-jev and framework JEV model versions differ")
    paired_direct, paired_framework = [], []
    for direct, framework in zip(direct_rows, framework_rows, strict=True):
        if direct["sample_id"] != framework["sample_id"]:
            raise ValueError("runs use different validation IDs")
        if (direct.get("video_sha256") and framework.get("video_sha256")
                and direct["video_sha256"] != framework["video_sha256"]):
            raise ValueError(f"runs have different videos for {direct['sample_id']}")
        if direct["status"] == framework["status"] == "scored":
            paired_direct.append(direct)
            paired_framework.append(framework)
    direct_mae = quality(paired_direct)["selective_mae"]
    framework_mae = quality(paired_framework)["selective_mae"]
    result = {"plan_sha256": direct_config["plan_sha256"], "rubric_sha256": rubric, "jev_model": direct_model,
              "direct_run_configuration": direct_config, "framework_run_configuration": framework_config,
              "direct_jev": evaluate(direct_rows), "framework": evaluate(framework_rows),
              "both_scored": {"n": len(paired_direct), "sample_ids": [row["sample_id"] for row in paired_direct],
                              "direct_mae": direct_mae, "framework_mae": framework_mae,
                              "direct_minus_framework_mae": None if direct_mae is None else direct_mae - framework_mae}}
    write_json(out, result)
    return {"comparison": str(Path(out).resolve()), "both_scored": result["both_scored"]}


def display(value):
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def table(headers, rows):
    esc = lambda value: html.escape(display(value))
    return ("<table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row)
                                               + "</tr>" for row in rows) + "</tbody></table>")


def confusion_svg(matrix):
    peak = max((n for row in matrix for n in row), default=1) or 1
    content = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 360" role="img" aria-label="混淆矩阵，行为真实等级，列为预测等级">',
               '<rect width="420" height="360" fill="white"/>',
               '<text x="210" y="24" text-anchor="middle" font-size="16">预测等级</text>']
    for i in range(5):
        content.append(f'<text x="{100+i*55}" y="49" text-anchor="middle">{i+1}</text>')
        content.append(f'<text x="50" y="{85+i*55}" text-anchor="middle">{i+1}</text>')
        for j in range(5):
            value = matrix[i][j]
            alpha = .08 + .80 * value / peak
            content.append(f'<rect x="{73+j*55}" y="{58+i*55}" width="52" height="52" fill="rgba(35,92,170,{alpha})"/>')
            content.append(f'<text x="{100+j*55}" y="{89+i*55}" text-anchor="middle" fill="{"white" if alpha>.55 else "#18314f"}">{value}</text>')
    content.append('<text x="16" y="205" transform="rotate(-90 16 205)" text-anchor="middle">真实等级</text></svg>')
    return "".join(content)


def create_report(dataset, run, out=None, *, phase="development", iterations=2000, rates=None):
    _, configuration, rows = joined_rows(dataset, run, phase)
    out = Path(out) if out else Path(run) / "reports" / phase
    if (Path(run) / "freeze.json").exists() and out.resolve().is_relative_to((Path(run) / "frozen-report").resolve()):
        raise ValueError("cannot overwrite the sealed development report")
    out.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(rows, iterations=iterations, rates=rates)
    metrics["phase"] = phase
    metrics["generated_at_hong_kong"] = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
    metrics["plan_sha256"] = digest(Path(dataset) / "plan.json")
    metrics["configuration"] = configuration
    metrics["pricing_rates"] = rates
    write_json(out / "metrics.json", metrics)
    direct = configuration.get("method") == METHOD
    columns = ("sample_id", "source", "reward", "status", "score", "absolute_error", "duration_s",
               "elapsed_seconds", "failed_stage", "error_type", "error", "origin_group", "task")
    if direct:
        columns += ("sampled_frame_ids", "sampled_images", "request_attempted", "request_bytes",
                    "request_elapsed_seconds", "input_tokens", "output_tokens")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            values = {k: row.get(k) for k in columns}
            values["absolute_error"] = abs(row["score"] - row["reward"]) if row["status"] == "scored" else None
            if direct:
                sampling = row.get("sampling")
                if sampling:
                    values.update(sampled_images=sampling["image_count"],
                                  sampled_frame_ids=" ".join(str(frame["frame_id"]) for frame in sampling["frames"]))
                calls = row.get("calls", [])
                if calls:
                    call = calls[0]
                    input_tokens, output_tokens = token_counts(call.get("usage"))
                    values.update(request_attempted=call["transport_attempted"], request_bytes=call["request_bytes"],
                                  request_elapsed_seconds=call.get("elapsed_seconds"),
                                  input_tokens=input_tokens, output_tokens=output_tokens)
            # Prevent spreadsheet formula execution from task or error strings.
            for key, value in values.items():
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                    values[key] = "'" + value
            writer.writerow(values)
    q, res = metrics["overall"], metrics["resources"]
    svg = confusion_svg(q["confusion_true_rows_predicted_columns"])
    (out / "confusion.svg").write_text(svg, encoding="utf-8")
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
             '<title>JEV validation 开发评估</title><style>body{font:16px/1.6 system-ui;margin:36px auto;max-width:1200px;padding:0 20px;color:#18314f;background:#f7f9fc}table{border-collapse:collapse;background:white;margin:16px 0;width:100%}th,td{border:1px solid #dce3ed;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#eaf0f8}svg{max-width:440px;width:100%}.note{padding:14px;background:#fff3d6}h1{font-size:28px}h2{margin-top:32px}code{overflow-wrap:anywhere}</style>',
             '<h1>JEV 直接图像评分 · validation 开发评估</h1>' if direct else '<h1>JEV validation 开发评估</h1>',
             '<p class="note">这是 validation 子集的探索性结果，不能作为 RoboRewardBench test 榜单成绩。未运行、失败和弃答均保留；已评分 MAE 必须结合覆盖率阅读。</p>',
             f'<p>阶段：{html.escape(phase)} · 样本：{len(rows)} · 来源：{len({r["source"] for r in rows})} · 生成时间：{html.escape(metrics["generated_at_hong_kong"])}</p>',
             table(["指标", "值"], [("计划样本", q["n"]), ("已尝试", metrics["attempted"]),
                   ("完成（含弃答）", metrics["completed"]), ("数值评分覆盖率", q["coverage"]),
                   ("已评分 MAE", q["selective_mae"]), ("全覆盖 MAE（仅全部有分时）", q["all_sample_mae"]),
                   ("全覆盖来源宏平均 MAE", metrics["source_macro_mae_all_covered"]),
                   ("已评分 exact accuracy", q["exact_accuracy_scored"]), ("已评分误差≤1", q["within_one_scored"]),
                   ("已评分误差≥2", q["large_error_rate_scored"]), ("全样本答对比例", q["answered_and_correct_all"]),
                   ("假设补齐缺失预测后的 MAE 界", q["hypothetical_all_sample_mae_bounds"]),
                   ("已评分 MAE 的 origin-group bootstrap 95% CI", metrics["mae_cluster_bootstrap"]["ci95"])]),
             '<p>MAE 界是假设为全部缺失项补上 1–5 预测时的可行范围，不是已观察结果。置信区间仅反映该开发子集，并条件于已有数值评分；origin-group 由文件名推断，不能保证找全近重复。</p>',
             '<h2>执行状态</h2>', table(["状态", "数量"], sorted(metrics["status_counts"].items())),
             '<h2>混淆矩阵</h2>', svg,
             '<h2>调用与成本</h2>', table(["指标", "值"], [("尝试的 API 调用", res["transport_attempted_calls"]),
                 ("停止时仍在途的调用", res["in_flight_calls_at_stop"]),
                 ("图片呈现次数（含重复）", res["image_presentations_attempted"]),
                 ("请求总字节", res["request_bytes_attempted"]),
                 ("单条耗时 p50 / 秒", res["latency_p50_seconds"]), ("单条耗时 p95 / 秒", res["latency_p95_seconds"]),
                 ("episode 文件字节", res["episode_disk_bytes"]), ("可计价调用估算 USD", res["estimated_usd_known_calls"]),
                 ("未能计价调用", res["calls_not_priced"])]),
             '<p>费用仅按用户提供费率估算；缺失 token 不视为零。超时或网络异常的服务端消耗可能无法完全恢复。</p>']
    parts += [table(["服务", "调用", "已报告输入 token", "已报告输出 token", "缺输入用量的调用", "缺输出用量的调用"],
              [(name, v["calls"], v["reported_input_tokens"], v["reported_output_tokens"],
                v["calls_missing_input_tokens"], v["calls_missing_output_tokens"])
               for name, v in res["providers"].items()]),
              table(["阶段", "样本数", "p50 秒", "p95 秒", "尝试调用数"],
              [(name, v["episodes"], v["p50_seconds"], v["p95_seconds"], v["calls_attempted"])
               for name, v in res["by_stage"].items()])]
    for field, title in (("by_source", "按来源"), ("by_reward", "按真实等级"), ("by_duration", "按时长")):
        parts += [f"<h2>{title}</h2>", table(["分组", "计划数量", "评分数", "覆盖率", "已评分 MAE", "大错率"],
                  [(key, value["n"], value["scored"], value["coverage"], value["selective_mae"], value["large_error_rate_scored"])
                   for key, value in metrics[field].items()])]
    parts += ['<h2>逐样本</h2>', table(["ID", "真值", "状态", "预测", "失败阶段", "错误"],
               [(r["sample_id"], r["reward"], r["status"], r.get("score"), r.get("failed_stage") if r["status"] in ("failed", "timeout", "interrupted") else None, r.get("error")) for r in rows]),
              '<p>完整导出：metrics.json、predictions.csv、confusion.svg。配置和代码哈希保存在 metrics.json。</p></html>']
    (out / "report.html").write_text("\n".join(parts), encoding="utf-8")
    return {"report": str((out / "report.html").resolve()), "metrics": str((out / "metrics.json").resolve()),
            "n": q["n"], "coverage": q["coverage"], "selective_mae": q["selective_mae"]}
