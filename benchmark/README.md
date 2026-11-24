# JEV validation 开发评测工具

框架评测覆盖离线预检、validation 5–10 条真实链路试点和约 50 条 validation 开发评估。直接 JEV 图像评分 baseline 使用同一 validation 开发清单和报告口径，运行方式见下方。

评测工具通过 `--framework-root` 指向待测框架。在本仓库运行时，将它设为仓库根目录；也可以指向另一份独立的交付目录。

## 环境

适用于 macOS/Linux，要求 Python 3.11+、ffprobe，以及满足交付版合约的本地分割 ONNX 权重。原 framework 的 YOLO 合约是静态 batch=1、FP32、NCHW、end-to-end segmentation 输出；不自动下载或转换权重。

在工作区根目录运行以下命令：

```sh
cd /path/to/jev-policy-eval
python3 -m venv .venv-benchmark
.venv-benchmark/bin/python -m pip install -r benchmark/requirements.txt
```

下文 `PY`、`FW`、`MODEL` 都是普通的 shell 变量；把 `MODEL` 替换为实际权重路径。

```sh
PY="$PWD/.venv-benchmark/bin/python"
FW="$PWD"
MODEL='/absolute/path/yolo26n-seg.onnx'
```

模型名称和 prompt 来自交付代码，不会自动换模型。`DASHSCOPE_BASE_URL` 沿用框架配置；API 的实时可用性由真实试点验证。密钥不需要发到聊天里；在本地 **zsh** 中用隐藏输入设置，避免把明文放进命令历史：

```sh
read -rs 'DASHSCOPE_API_KEY?DashScope API key: '; echo
export DASHSCOPE_API_KEY
read -rs 'TYPESAFE_API_KEY?TypeSafe API key: '; echo
export TYPESAFE_API_KEY
```

密钥只由 worker 继承，工具不序列化环境变量或 Authorization 头。Codex 已运行的进程不会继承另一个终端后来设置的变量；从上述同一个终端启动 CLI 即可。实现和离线测试不需要密钥。

## 1. 离线预检

```sh
"$PY" -B -m unittest discover -s benchmark/tests -v
"$PY" -B -m benchmark preflight --framework-root "$FW" \
  --python "$PY" --yolo-model "$MODEL" --run-tests \
  --out benchmark-output/offline-preflight
```

preflight 不调用模型 API，检查导入环境、ffprobe、本地 ONNX 初始化、密钥是否存在，并可运行交付仓库的全部测试。`preflight.json` 区分测试通过与真实运行准备就绪；依赖导致的 skip 不算完整测试通过。缺视频/密钥/权重时退出码为 2，即使离线测试已经通过。完整日志在 `framework-tests.log`。

## 数据清单与视频

官方 validation 在 Hub 中实际叫 `val/`，映射到框架参数 `validation`。默认固定到不可变快照 `d69d5b92b8f3caee9a616b615b300bdf0d37aef2`；另选快照时必须使用完整的 40 位 commit。官方 JSONL 的字段为 `file_name/task/reward/gpt5_mini_check`。

```sh
"$PY" -B -m benchmark fetch-metadata --out benchmark-data/val-metadata.jsonl
"$PY" -B -m benchmark prepare --metadata benchmark-data/val-metadata.jsonl \
  --out benchmark-data/validation-50 --size 50 --pilot-size 10 --seed 20261121
"$PY" -B -m benchmark download --dataset benchmark-data/validation-50 --phase pilot
```

如果本地已生成 `benchmark-data/validation-50/plan.json`，直接复用，不要再次 prepare 同一个目录。数据、视频、虚拟环境和运行日志不随源码提交；新克隆需按上述命令准备。离线视频可以用 `download --local-root /path/to/val` 拷贝；该目录下面应直接包含 metadata 中的相对路径。

- `inputs.jsonl` 仅包含 sample/video ID、中性视频路径、原始 task 和 validation 标记；worker 接口拒绝额外字段。
- `labels.jsonl` 单独保存参考分、来源、原文件名与分组信息；不交给 worker，`gpt5_mini_check` 不进入任何推理输入。
- 视频另存为 `videos/v<hash>.mp4`，不会把文件名中的分数传给模型。相同原视频复用同一个中性视频文件。
- `plan.json` 固定抽样种子、哈希和试点/开发 ID；试点是开发清单的前缀。默认按等级及来源轮转抽样，优先检查覆盖，不代表总体比例。
- 可选 `prepare --durations /path/to/validation-durations.csv`，列为 `file_name,duration_s`，加入时长分层；未提供时不伪造时长。运行后 ffprobe 提供已下载视频的实际时长。
- `video-inventory.json` 保存 SHA256。视频缺失、被修改或未登记都会阻止真实运行；下载器不会覆盖无法验证的现有文件。

官方数据和标签说明：[RoboReward](https://huggingface.co/datasets/teetone/RoboReward)。validation 未标为人工核验的 RoboRewardBench test，因此报告明确为探索性开发结果。

## 2. 真实链路试点

先检查试点视频与环境。准备完成后，使用 `--execute` 启动实际 API 调用；不加时仅打印清单，零模型调用。

```sh
"$PY" -B -m benchmark preflight --framework-root "$FW" --python "$PY" \
  --yolo-model "$MODEL" --dataset benchmark-data/validation-50 --phase pilot \
  --out benchmark-output/pilot-preflight

"$PY" -B -m benchmark run --framework-root "$FW" --python "$PY" \
  --yolo-model "$MODEL" --dataset benchmark-data/validation-50 \
  --out benchmark-runs/delivery-01 --phase pilot

"$PY" -B -m benchmark run --framework-root "$FW" --python "$PY" \
  --yolo-model "$MODEL" --dataset benchmark-data/validation-50 \
  --out benchmark-runs/delivery-01 --phase pilot --execute
```

可以先加 `--max-new-episodes 1` 观察单条成本，再用相同命令继续余下试点。默认每条 wall timeout 为 1800 秒。`--max-calls-per-episode` 是可选的硬调用数限制；更改它会改变运行配置。`--batch-seconds` 只在样本之间检查是否停止，不是账单上限。原框架的 32 MiB JEV 请求限制保持不变；不通过删帧、补分或重试掩盖超限。

## 3. validation 开发评估

试点确认可运行后下载其余开发视频，复用同一配置和输出目录：

```sh
"$PY" -B -m benchmark download --dataset benchmark-data/validation-50 --phase development
"$PY" -B -m benchmark run --framework-root "$FW" --python "$PY" \
  --yolo-model "$MODEL" --dataset benchmark-data/validation-50 \
  --out benchmark-runs/delivery-01 --phase development --execute
```

已完成、弃答、失败和中断样本都会跳过，不自动重试。试点 10 条与开发 50 条使用同一配置时，总共只尝试 50 条。若修复 framework、修改 prompt/权重/间隔/每条限额/超时或运行环境，使用新输出目录；保留旧结果，不能混算。需要重试失败样本时同样使用新目录，明确视为另一次运行。

每个 episode 有独立 job、运行日志、预测 journal、原框架完整 artifacts、逐调用 audit 与逐阶段耗时。job 中没有参考分；key 不写入 audit。Ctrl-C 终止该 episode 的进程组并保存局部结果；硬杀父进程后，下次恢复会将 running 状态标为 interrupted，检查并终止匹配到的旧 worker，不再重复调用。

## 报告和冻结

run 在正常返回或 Ctrl-C 后生成对应阶段报告。也可以纯离线重建：

```sh
"$PY" -B -m benchmark report --dataset benchmark-data/validation-50 \
  --run benchmark-runs/delivery-01 --phase development
"$PY" -B -m benchmark freeze --dataset benchmark-data/validation-50 \
  --run benchmark-runs/delivery-01
```

报告在 `reports/<phase>/report.html`，可直接打开；同时输出 `metrics.json`、`predictions.csv`、`confusion.svg`。freeze 要求开发清单中没有未运行或中断项，保留独立的 `frozen-report/` 和配置/指标哈希，之后阻止追加运行。

统计包含：

- MAE、exact/within-one/大错率、混淆矩阵，按来源、等级和时长分组。
- 全部计划样本始终在分母中；单独报告运行完成率、数值覆盖率、弃答、失败/超时/中断/未运行。
- 全覆盖 pooled/source-macro MAE 仅在全部有数值评分时给出；否则明确报告 selective MAE。另给假设补齐缺失预测的 MAE 可行界，不把它当实际成绩。
- 以文件名推断的 origin group 做 2000 次 bootstrap；区间条件于数值评分，仅用于开发子集。重复/截断/任务近似组不能保证找全，不作独立性或总体代表性保证。
- 所有调用次数、图像呈现次数（含重复）、请求字节、可获得的 input/output token、单条 wall p50/p95、episode 文件占用。
- 未提供费率时费用显示不可用。可用 `report --rates /path/to/rates.json` 输入经核实的费率，键为 `qwen/typesafe`，值可包含 `input_usd_per_million/output_usd_per_million/per_request_usd`。这只是估算；缺失 usage 不计为零费用，报告未计价调用数。

同一框架同时使用 Qwen 和 JEV；本阶段没有对照实验，因此这些结果只能说明链路与开发集表现，不能证明比直接评分更好，或证明长程机制有效。

## 直接 JEV 图像评分 baseline

复用 first delivery 的 `validation-50/plan.json`。如果需要恢复清单，使用上面的固定快照、种子和默认参数，核对 `input_sha256` 为 `8022bcaca064411137f99397af65c2e3eef50a9a5a663f046d0ea44cffb52067`。开发评估保留原 50 条；baseline 试点按实测时长均匀选取 10 个排序位置，包含最短和最长视频，所以先下载全部 50 条视频。

每段视频按原始 PTS 分为相对首帧的 2 秒时间桶，选每个非空桶的首帧并保留末帧。短于 2 秒的视频保留首末帧，单帧视频只发送一次。一个请求包含原始任务、排序后的图像与原始帧号/PTS，以及 first delivery 的同一 rubric。固定使用 `jev-1.14.0`，一次 Choice 返回 `level_1` 到 `level_5` 或 `insufficient_evidence`。真实标签只在报告时合并。

图像适配与 first delivery 的图文 JEV 一致。每次请求使用图像和长上下文请求头，限制 128 张图和 32 MiB JSON 请求体，单图限制 16 MiB。图片数是本地资源保护上限；TypeSafe 判定实际 token 数与处理后的请求尺寸。超限按错误记录，固定采样帧完整保留，模型请求最多尝试一次。[TypeSafe 图像限制](https://docs.typesafe.ai/images#limits) · [TypeSafe 长上下文](https://docs.typesafe.ai/models#long-context)

运行需要 Python 3.11+、`ffprobe`、NumPy 和 OpenCV。真实请求使用 `TYPESAFE_API_KEY`。使用同一运行目录先验收试点，再继续开发评估：

```sh
DATA=benchmark-data/validation-50
DIRECT=benchmark-runs/direct-jev-01
"$PY" -B -m benchmark download --dataset "$DATA" --phase development
"$PY" -B -m benchmark inspect-direct --dataset "$DATA" \
  --out benchmark-output/direct-jev-pilot-inspection
"$PY" -B -m benchmark run --method direct-jev --framework-root "$PWD" \
  --dataset "$DATA" --out "$DIRECT" --phase pilot --execute
"$PY" -B -m benchmark run --method direct-jev --framework-root "$PWD" \
  --dataset "$DATA" --out "$DIRECT" --phase development --execute
```

`inspect-direct` 在本地生成试点 JPEG、帧号和请求清单，供调用前核对。真实运行的 `reports/<phase>/` 使用框架报告的 MAE、覆盖率、分组、混淆矩阵和资源指标；`predictions.csv` 增加采样帧号、图片数、请求字节、token 和请求时间。逐条记录含实际 PTS、图片 SHA256、服务端用量、错误及阶段耗时。开发阶段复用已尝试的试点样本。

两个开发运行目录齐备后，生成同一批视频的比较报告。报告保留各自的完整 50 条分母，也计算双方均有数值评分样本上的配对 MAE 差：

```sh
"$PY" -B -m benchmark compare --dataset "$DATA" --direct-run "$DIRECT" \
  --framework-run benchmark-runs/delivery-01 \
  --out benchmark-output/direct-vs-framework.json
```

## 已完成的本地验收

本次重新运行交付版 153 项离线测试，全部通过、无跳过；benchmark 有真实子进程、超时、断点恢复、标签隔离及统计回归测试。新增评测工具的 23 项离线测试也全部通过、无跳过，共 176 项；本地完整记录保存在不纳入版本控制的 `benchmark-output/implementation-verification.json`。

已解析固定官方快照的 6,232 条 validation 元数据，并生成 50 条开发清单（每等级 10 条、18 个来源），前 10 条为试点。尚未进行真实 Qwen/JEV 推理；模型服务可用性和评分质量待密钥与本地 ONNX 权重就绪后验证。
