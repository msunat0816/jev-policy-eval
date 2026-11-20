# jev-policy-eval

离线机器人 rollout 的证据驱动评价原型。输入视频和原始 instruction，目标链路为：

**一次 Qwen 粗观察 → 一次 JEV 批量选窗/工具选择 → CPU expert 取证 → 写入证据文件 → 一次 JEV 终局评分或弃答。**

第一轮只完成至少一条官方 RoboRewardBench test 样本的真实链路验收。不做 baseline 对比、全测试集评分、逐帧 reward、student 训练或蒸馏。

## 工作位置与协作方式

- 交付分支：[`codex/first-delivery`](https://github.com/msunat0816/jev-policy-eval/tree/codex/first-delivery)，尚未合并 main。
- 最初交接点：[`bc7d5af`](https://github.com/msunat0816/jev-policy-eval/commit/bc7d5af728ae470a2757eb4da8e255c91aeecdb1)。2026-11-20 本次编辑前核对，main 仍指向它。
- 模型更新：[`b8106df`](https://github.com/msunat0816/jev-policy-eval/commit/b8106df5f1dd1187340aa48f1bd5f229ffcc9745)，代码及当前规格改为 `qwen3.8-flash`。
- 优先读 [FYP_FRAMEWORK.md 开头的当前执行规格](FYP_FRAMEWORK.md)，历史方案仅供参考。
- 用户要求从头一步步检查，给选项由用户选择；不会选时再给建议。不要跳过待确认的设计直接调用付费模型。

## 已确认的职责分工

**Qwen 的核心职责是结合原始 instruction 和稀疏采样视频，给出具体、与任务相关的视觉描述。instruction 引导关注对象和变化，视频使描述具体化。**

用户确认保留 Qwen 候选框作为定位提示，逐帧拆开可见内容与未知，并由 Qwen 从具体描述和疑问提出待核验点。原始 instruction 始终是任务依据，模型清单不能替代它。

| 组件 | 职责 |
|---|---|
| Qwen | 忠实整理自然语言任务要求；逐采样帧描述可见内容、未知和候选框；提出关联要求、实体和采样帧的待核验点。 |
| 程序 | 构造窗口与待核验点的有限组合，维护 expert 能力，执行选择并记录证据。 |
| JEV | 从给定选项判断核验价值与适用工具；不要求它自由生成 requirements 或核验清单。 |
| CPU expert | 在原视频跟踪和测量，缺框、丢失或超出能力时记录未知。 |
| JEV 终评 | 从已保存的证据按原始终局 rubric 选择 1–5 档或弃答。 |

Qwen 不选窗、不设优先级、不指派 expert、不输出 reward。二维重叠或相对运动不能独立证明三维支撑、抓取、释放或放稳。稀疏帧看不清，也不能断言窗口内部一定能看清。

## 本次已落地

- 用户确认的完整英文 prompt：[prompts/qwen_coarse_observation.txt](prompts/qwen_coarse_observation.txt)。运行入口读取它，再附原始 instruction、帧号、PTS 和所有采样图片。
- 解析与结构校验：[coarse_observation.py](coarse_observation.py)，schema 为 `task-guided-coarse-v2`。
- 模型：`qwen3.8-flash`，OpenAI 兼容 Chat Completions，`enable_thinking=false`；JEV 为 `jev-1.14.0`。
- 采样仍从第一帧原始 PTS 开始，每 2 秒取每个非空时间段的第一帧，额外保留末帧，按原帧号去重，无 8 帧上限。沿用 `--sample-interval-seconds`。

| 输出数组 | 每项字段 |
|---|---|
| `requirements` | `id, text, subject, reference`；自然语言任务语义，不输出工具 `predicate`。 |
| `frames` | `frame, visible_facts, uncertainties, boxes`；框为 0–1000 的 `[x1,y1,x2,y2]`，可缺框。 |
| `candidate_checks` | `id, requirement_ids, entity_keys, related_frames, question, rationale`；无具体问题时可为空。 |

校验字段类型、帧覆盖/顺序、框合法性、ID 唯一性及要求/帧引用。结构校验不能证明描述真实、身份正确、任务要求完整或问题值得核验。非法响应直接报错，保留已有原始 API trace，不自动付费重试修复。

有效输出另存 `01-observation.json`；`run.json` 记录 schema 版本；`03-evidence.json` 保留候选问题。

## 未完成的适配与验收

本阶段完成 prompt、解析及安全衔接，尚未完成新 schema 下游适配：

1. 当前 JEV 仍主要按“窗口 × requirement”提问，state 已包含 `candidate_checks`；审核后的“窗口 × 待核验点”配对尚未实现。下一步先和用户检查此规则及原子 Choice 设计。
2. 新 Qwen 不再输出旧 `predicate`。CPU 暂保留 `measurement_only` 和 `requirement_to_measurement_adapter_pending`，没有完成“自然语言要求 → 程序测量判定”适配，不能把测量升级成任务要求满足。
3. 现有保护会在只有测量、没有决定性判定时转为弃答。这是适配未完成的限制，不能包装成完整真实验收或真实模型的合理弃答。
4. 香港端点和 Secrets 尚未核对。旧代码仍有缺 `DASHSCOPE_BASE_URL` 时回退国际站地址的路径，必须在真实调用前移除并补前置检查。
5. 仍需核对 RoboReward 原文 rubric、固定真实样本及标签隔离、完整版本/调用/费用留档、终分证据引用及成果撤销。

因此，当前先完成适配和模拟检查，再进行付费验收。

## 验证状态与复跑

| 项目 | 状态与边界 |
|---|---|
| 2,831 条 test 视频时长调查 | 已有结果可复用：[逐视频 CSV](notes/roborewardbench_video_durations.csv)。 |
| 历史真实视频解码/采样 | [已有 Actions run](https://github.com/msunat0816/jev-policy-eval/actions/runs/44531285937)，工作流在 `codex/verify-interval-sampling`；只证明采样组件。 |
| 历史选窗/弃答保护 | [已有 Actions run](https://github.com/msunat0816/jev-policy-eval/actions/runs/44517117399)，工作流在 `codex/verify-jev-windows`。旧流程可在缺密钥时跳过真实评分，绿色状态不能证明真实模型调用。 |
| 新 schema 与观察请求 | [tests](tests) 使用合成图片和模拟 API。[CPU 检查工作流](.github/workflows/check-coarse-observation.yml) 不读取模型密钥；运行状态以 Actions 为准。 |
| 香港 Qwen + JEV 真实端到端 | 尚未验收；本聊天内真实模型 API 调用为 0。 |
| 有证据支持的数值终分 | 尚未展示。 |

在仓库根目录执行不付费检查：

```sh
python3 -m unittest discover -s tests -v
```

结构测试只依赖标准库；没有 OpenCV/NumPy 时集成测试明确跳过。CPU Actions 安装依赖，执行模拟集成检查并上传 `coarse-observation-checks` artifact，包含日志及批准的 prompt。

2026-11-20 验证记录：[代码/工作流提交 `435b9d7`](https://github.com/msunat0816/jev-policy-eval/commit/435b9d767118bc3601dc38172dcce66d2764e825) 的 [CPU Actions](https://github.com/msunat0816/jev-policy-eval/actions/runs/44686307873) 已通过全部 11 项测试，无跳过项；[artifact](https://github.com/msunat0816/jev-policy-eval/actions/runs/44686307873/artifacts/14606552201) 已下载并打开，内含测试日志与 prompt，prompt 与批准的源文件逐字节一致。ZIP SHA256 为 `126a66fa8ee7edd5ca764d2ffdf893d01f406afa5f5353fc4ff6349a746e92d7`。这只验证结构和模拟集成，不验证真实 Qwen/JEV 的行为。

此前 [首次检查运行](https://github.com/msunat0816/jev-policy-eval/actions/runs/44685402241) 在系统依赖安装阶段达到 10 分钟上限后结束为 `cancelled`，测试步骤未执行；保留该记录，并将工作流改为 Python 3.11 + headless OpenCV wheels；没有改动测试判据或调用真实模型。

## 后续真实验收配置和要求

先完成上述设计检查和适配，再核对：

- [Repository Secrets](https://github.com/msunat0816/jev-policy-eval/settings/secrets/actions)：`DASHSCOPE_API_KEY`、`TYPESAFE_API_KEY`。
- [Repository Variables](https://github.com/msunat0816/jev-policy-eval/settings/variables/actions)：`DASHSCOPE_BASE_URL`，使用百炼香港工作空间提供的完整兼容 base URL；不猜地址、不静默回退新加坡。
- 不打印、提交密钥，不要求用户在聊天粘贴密钥。

选择至少一条 [官方 RoboReward test](https://huggingface.co/datasets/teetone/RoboReward) 固定版本真实样本。记录 instruction、revision、稳定 ID、视频哈希和选样理由。参考分、解释和泄露结果的文件名不得进入模型输入；预测保存后再关联参考分。

相邻采样帧组成全部候选窗口。统一按 0–3 判断，所有 ≥2 的窗口核验，数量不限且可为零；末段没有特殊强制规则。粗看末帧不等于核验末段。全部未选或没有有效证据应正确弃答；后来的反证不能让早期成果继续支持满分。

真实验收跟进 CPU Actions 至结束并打开 artifact，分别报告“真实链路是否跑通”和“是否展示有证据的数值分”。保留合理弃答；实现故障修复后重跑；不因模型分与参考分不同改规则，也不为得到数字反复调用或删除失败记录。

最终交付 commit、run、artifact 链接及复跑步骤，汇总样本/任务、时长、粗观察帧数、入选窗口数、关键证据时间、预测或弃答原因、参考分、调用次数、耗时、可得成本。保留实际 prompt、模型/代码版本、帧号/PTS/分辨率、所有窗口/工具/证据/未知、每次调用用量/耗时/重试/复用及 rollout 总耗时。未返回用量标为缺失；费用估算注明香港实际部署范围、费率和币种。保存判断实际看过的时间范围，便于未来排除未来信息；未检查帧保持未知。

新聊天从本分支 README 和当前执行规格开始，重新核对远端 main 与分支 HEAD，保留已确认决策，继续下一步检查。
