# jev-policy-eval

离线机器人 rollout 的证据驱动评价原型。输入视频和原始 instruction，目标链路为：

**VLM 粗看与完整 checklist → JEV 逐窗口选窗 → VLM 匹配 check 与 expert → 工具取证后 JEV 逐 check 判断与终局评分。**

2026-11-21 用户已批准加入第三步 VLM 匹配，当前入口执行前三个阶段，匹配模型暂定 `qwen3.8-flash`。保存匹配结果后返回 `awaiting_expert_execution`；expert 执行与新版判断/评分须先获用户允许再修改。旧取证和评分函数保留，但当前入口不调用，也不产生最终分数。用户随后授权将本轮改动提交并推送至 `first-delivery` 分支。

第一轮只完成至少一条官方 RoboRewardBench test 样本的真实链路验收。不做 baseline 对比、全测试集评分、逐帧 reward、student 训练或蒸馏。

## 工作位置与协作方式

- 交付分支：[`first-delivery`](https://github.com/msunat0816/jev-policy-eval/tree/first-delivery)，远端已从 `codex/first-delivery` 更名，尚未合并 main。
- 最初交接点：[`bc7d5af`](https://github.com/msunat0816/jev-policy-eval/commit/bc7d5af728ae470a2757eb4da8e255c91aeecdb1)。2026-11-20 本次编辑前核对，main 仍指向它。
- 模型更新：[`b8106df`](https://github.com/msunat0816/jev-policy-eval/commit/b8106df5f1dd1187340aa48f1bd5f229ffcc9745)，代码及当前规格改为 `qwen3.8-flash`。
- 优先读 [FYP_FRAMEWORK.md 开头的当前执行规格](FYP_FRAMEWORK.md)，历史方案仅供参考。
- 用户要求从头一步步检查，给选项由用户选择；不会选时再给建议。不要跳过待确认的设计直接调用付费模型。

## 已确认的职责分工

**Qwen 结合原始 instruction 和稀疏采样帧，逐帧描述可见操作、相关对象及未知，并生成评估全部任务要求所需的完整 checklist。** 即使画面看似已完成或失败，也保留对应问题；不可见或工具无法核验的要求也不能省略。

保留现有候选框字段兼容旧数据格式。原始 instruction 始终是任务依据，模型清单不能替代它。粗看不做窗口匹配。

| 组件 | 职责 |
|---|---|
| VLM 粗看 | 整理自然语言任务要求，逐帧描述操作、对象及未知，输出完整 checklist。 |
| 程序 | 以相邻采样帧构造候选窗口；校验输入和选窗答案，保存全部判断与选中列表。 |
| JEV 选窗 | 以完整 checklist 和全部粗观察为共同背景，每个窗口独立判断是否值得细看；不做 check 匹配或 expert 选择。 |
| VLM 匹配 | 查看选中窗口的连续原始帧，结合完整 checklist 和两类 expert 的能力契约，输出相关性和 expert 调用提议。执行器尚未接入，不代表工具已运行。 |
| Expert（待更新） | 对象与空间、短窗口运动取证；本轮未替换旧实现。 |
| JEV 判断与评分（待更新） | 综合工具证据对每个 check 给一次 yes/no/unknown，再按 rubric 评分；证据不足时返回 insufficient_evidence。 |

粗看阶段的 Qwen 不选窗、不设优先级、不指派 expert、不输出 reward；匹配阶段的 Qwen 可以提出 expert 调用，但不回答 check 或评分。二维重叠或相对运动不能独立证明三维支撑、抓取、释放或放稳。稀疏帧看不清，也不能断言窗口内部一定能看清。

## 本次已落地

- 用户确认的完整英文 prompt：[prompts/qwen_coarse_observation.txt](prompts/qwen_coarse_observation.txt)。运行入口读取它，再附原始 instruction、帧号、PTS 和所有采样图片。
- 解析与结构校验：[coarse_observation.py](coarse_observation.py)，schema 为 `task-guided-coarse-v2`。
- 模型：`qwen3.8-flash`，OpenAI 兼容 Chat Completions，`enable_thinking=false`；JEV 为 `jev-1.14.0`。
- 采样仍从第一帧原始 PTS 开始，每 2 秒取每个非空时间段的第一帧，额外保留末帧，按原帧号去重，无 8 帧上限。沿用 `--sample-interval-seconds`。
- 选窗请求与输出契约：[window_selection.py](window_selection.py)，协议 `jev-window-selection-v2`；入口调用 `select_windows()`，替换旧 `route()`。
- VLM 匹配契约与校验：[vlm_matching.py](vlm_matching.py)，协议 `vlm-window-check-expert-v1`；独立 prompt 为 [prompts/qwen_window_matching.txt](prompts/qwen_window_matching.txt)，入口调用 `match_windows()`。

| 输出数组 | 每项字段 |
|---|---|
| `requirements` | `id, text, subject, reference`；自然语言任务语义，不输出工具 `predicate`。 |
| `frames` | `frame, visible_facts, uncertainties, boxes`；框为 0–1000 的 `[x1,y1,x2,y2]`，可缺框。 |
| `candidate_checks` | `id, requirement_ids, entity_keys, related_frames, question, rationale`；沿用字段名，但含义为完整 checklist。 |

校验字段类型、帧覆盖/顺序、框合法性、ID 唯一性及要求/帧引用。结构校验不能证明描述真实、身份正确、任务要求完整或问题值得核验。非法响应直接报错，保留已有原始 API trace，不自动付费重试修复。

有效粗观察仍另存 `01-observation.json`。旧粗观察解析器仍兼容空 checks；选窗入口额外要求 checklist 非空且覆盖所有 requirement ID，否则在 JEV 调用前报错。这只是引用覆盖检查，不能证明 checklist 在语义上完整。

## 当前 JEV 选窗

1. 原始 instruction、所有 requirements、完整 checklist、全部逐帧 `visible_facts / uncertainties` 只在共享 state 中提交一次；候选框和 expert 测量能力不进入选窗背景。`related_frames` 只作观察引用，不限制可选窗口。
2. 每个窗口一个独立 Choice，所有窗口在一次 JEV 请求中批量提交。沿用 0–3 档：0 无任务相关性，1 相关但重复，2 值得细看，3 可能影响终分的核心歧义、矛盾或成果撤销。所有 ≥2 的窗口入选，无数量上限，也可全不选；末段采用同一规则。
3. 不再构造“窗口 × check”配对，不再取配对最大值，不再询问 `layout / motion` 适用性，也不提前输出窗口关联哪些 check。看似已完成、缺框或缺工具，均不能单独作为跳过依据。
4. JEV 原始请求/响应 trace 为 `02-select-windows.json`。`02-selected-windows.json` 保存 `protocol, threshold, windows, selected_windows`；每条窗口记录含 `id, frame_range, priority, selected`。`frame_range` 保留原始帧号和两端，不对入选窗口重新编号。全部判断保留，选中列表保持候选窗口的输入顺序。
5. `run.json` 保留 `window_selection` 协议、决策单位、共享背景、每窗口问题数和阈值。选窗结果交给下面的 VLM 匹配；它本身不做窗口/check 关联或工具选择。

## 当前 VLM 匹配

1. 匹配模型使用独立常量 `MATCHING_MODEL=qwen3.8-flash`，非思考模式、JSON 输出，沿用现有 DashScope 密钥和 base URL 解析方式。粗看模型和采样规则不变。每个选中窗口单独请求一次，不重新选窗，也不处理未选中的窗口。
2. 读取选中窗口内全部连续原始帧，包括两端，不额外设采样上限。每帧标注原帧号和 PTS，JPEG 保存至 `matching_frames/<window_id>/`，不覆盖粗看图片。每次请求携带原 instruction、全部 requirements、完整 checklist、当前窗口及 expert 契约；选窗优先级不传入匹配模型。
3. 两个 expert ID 为 `object_spatial`（对象与空间）和 `short_window_motion`（短窗口运动）。契约描述用户指定的 YOLO nano 分割/ONNX CPU 与金字塔 LK 光流及其限制，均明确 `execution_status=not_implemented`。本轮只输出调用提议，不安装或运行这些工具，也不映射到旧 `layout / motion` 实现。固定检测类别、可靠 mask/纹理点、参考区域和相机运动限制均写入 prompt。
4. 每个窗口必须返回每个 check 的一条记录：`check_id, relevance, reason, experts, limitations`；相关性为 `related / unrelated / uncertain`，不是任务完成判断。expert 提议包含 `expert_id, entity_keys, frame_ids, reason`。不相关项不得提议 expert；相关但无工具覆盖，或相关性不确定时，必须解释限制。实体须来自该 check，帧号须来自当前窗口；运动提议至少指定两帧并覆盖连续区间。
5. 调用前校验完整选窗记录与选中列表一致、checklist 引用覆盖、帧端点和时间戳。返回后校验窗口/check/expert/实体/帧引用、完整 check 覆盖和字段结构，拒绝新增 check、旧工具名、遗漏/重复项及 completion verdict。结构校验不证明相关性、工具适用性或视觉描述正确。
6. `03-match-<window_id>.json` 保存原始调用 trace；`03-matching-<window_id>.json` 保存该窗口的有效匹配与原帧号/PTS；`03-window-matching.json` 汇总完整 checklist、expert 契约和所有窗口提议，标记 `status=proposals_only`。HTTP 日志新增可选图片目录参数，使匹配日志能定位到实际 JPEG。非法响应直接报错，保留已有 trace 和已完成窗口记录，不自动重试或补造结果。
7. 空选中列表不读匹配图片、不调用匹配 API，仍保存空汇总。`run.json` 新增匹配模型、协议、全帧策略、每窗口一次请求和未执行 expert 的配置。`result.json` 返回 `status=awaiting_expert_execution`、`score=null`、选窗/匹配文件路径、选中窗口、匹配窗口数与耗时。当前入口不调用旧取证/评分，不写 `03-evidence.json`。

## 保留的旧取证与评分实现（当前入口不调用）

以下记录先前实现的能力边界。`collect()` 仍接收旧配对/工具格式，`score_saved()` 仍读取旧证据格式；它们未被改造为新四阶段设计，也不接受新的选窗结果。回归测试保留，不能据此宣称新下游已实现。

- 旧执行器仅处理优先级 ≥2 且 expert 为 `useful` 的配对；缺框、跟踪失败、额外 actor、缺参照物或实体有歧义时保留未知，同窗口同对象对共享跟踪。
- [measurement_adapter.py](measurement_adapter.py) 的 `explicit-2d-requirements-v1` 仍只支持以下受限二维句式；普通自然语言任务即使有测量也可能只有 `measurement_only`，要求状态仍为未知。

### 首版受限适配的精确边界

只有**整个原始 instruction** 和 requirement 文本都能完整匹配同一条白名单句式、实体名称与键一致、expert 与时间范围一致，才允许直接判定要求。允许大小写、空白和句末句号变化，以及实体键中下划线对应空格；不推断别名、代词、属性或同义改写。当前白名单示例：

```text
At episode end, the bounding box of red block must be entirely to the left of the bounding box of blue target in the image plane.
Over the entire episode, the maximum image-plane relative-center excursion of red block relative to blue target, measured from the first frame, must be at most 1% of the image diagonal.
```

布局也支持 `to the right of / above / below`；运动支持 `at most / at least` 和显式非负百分比。布局只在 episode 末帧直接覆盖终局要求，以 2 像素死区保留边界未知。整段相对位移要求必须完整跟踪第 0 帧至末帧；多个分别重置起点的局部位移不能拼成整段位移。旧的“静止/运动”默认阈值不再用来解释普通动作词。一般自然语言、多约束句式、抓取、释放、容纳、支撑和放稳均不在此白名单内；不能改写官方 instruction 来套模板。对象身份与框仍为 Qwen 提议，所有二维结论受该绑定条件限制。

旧 `03-evidence.json` 格式保存 `measurement_status`、`requirement_status`、`semantic_adapter`、实体、帧/PTS、测量和未知；旧终评重新从已保存记录计算要求状态，保留证据 ID。粗观察叙述、路由优先级和适用性选择不进入旧终评 state。没有决定性的要求级证据则弃答；满分还须每项要求具有最新直接支持。模板外任务得到的弃答反映此受限适配的能力边界，不足以证明通用评价能力。

## 尚未完成的真实验收

当前入口只交付粗观察、选窗及 VLM 匹配提议。以下端到端验收须等 expert 执行及 JEV 逐 check 判断/评分分别获准并实现后再继续。

1. 受限适配尚待用户查看覆盖范围；官方样本中的自然语言和物理要求可能仍无法得到要求级判定，真实数值终分不能保证。
2. 香港端点和 Secrets 尚未核对。旧代码仍有缺 `DASHSCOPE_BASE_URL` 时回退国际站地址的路径，必须在真实调用前移除并补前置检查。
3. 仍需核对 RoboReward 原文 rubric、固定真实样本及标签隔离、完整版本/调用/费用留档、终分证据引用和真实成果撤销案例。

先检查本地改动与模拟结果，再由用户确认下一步；当前不进行付费验收。

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

结构、纯选窗与纯匹配契约测试只依赖标准库；没有 OpenCV/NumPy 时视频入口等集成测试明确跳过。CPU Actions 安装依赖，执行模拟集成检查并上传 `coarse-observation-checks` artifact，包含日志及批准的粗看 prompt。

2026-11-21 VLM 匹配检查：复用本机已有 Python 3.14.6、NumPy 2.4.6、headless OpenCV 4.14.0，51 项离线测试全部通过，无跳过。新增 11 项纯匹配契约测试与 5 项模拟集成测试，并将原入口停止检查更新为匹配后停止。覆盖连续帧读取、完整 checklist、expert/实体/帧引用、空选择、解码失败、非法响应 trace、单帧窗口和不运行 expert/评分。所有模型响应均为模拟，未调用真实 API、未验证真实匹配质量，未运行新 Actions。

2026-11-21 选窗阶段历史检查（接入匹配前）：在本机已有 Python 3.14.6、NumPy 2.4.6、headless OpenCV 4.14.0 环境中，35 项离线测试全部通过，无跳过。新增 13 项纯选窗/模拟集成检查，替换 1 项旧全量路由测试，并保留其余 22 项回归检查。覆盖共享完整 checklist、每窗口一个 Choice、阈值与空选择、输入/回答校验、结果保存，以及当时入口在选窗后停止。所有 JEV/VLM 响应均为模拟；未调用真实模型、未验证真实选窗质量，也未运行新 Actions。

2026-11-20 新增本地检查：23 项通过，无跳过。使用 Python 3.14.6、NumPy 2.5.5、headless OpenCV 4.14.0，覆盖全量配对、三个 Choice、阈值、缺框/丢失/不完整跟踪、共享跟踪、语义/时间边界、相反证据和弃答。包含合成纹理图像上的实际 OpenCV 跟踪；所有模型回答均为模拟，测试中的数值分不是官方真实样本成绩。首次本地尝试在 NumPy 随机数模块加载时被中断；测试纹理改由标准库的固定随机种子构造后完成，未放宽判定。尚无针对这些改动的新 Actions run。

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
