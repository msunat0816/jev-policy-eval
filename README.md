# jev-policy-eval

离线机器人 rollout 的证据驱动评价原型。输入视频和原始 instruction，目标链路为：

**VLM 粗看与完整 checklist → JEV 选窗 → YOLO 候选前置 → 匹配 VLM 关联 check/expert 并绑定候选 → 使用候选与运动 expert → 逐 check 证据包 → JEV 图文判断与纯文字终评。**

2026-11-21 本次接续实现计划第 7 步，仅本地修改，不 push。全部 check 判断完成后，程序汇总原始 instruction、完整任务要求、RoboReward 五档终局 rubric、全部答案及对应证据摘要/时间范围/未解决记录，一次纯文字调用 JEV 选择 `level_1…level_5` 或 `insufficient_evidence`。程序仅将等级映射为 1–5，弃答为 null；不按 yes 比例计分、不将 unknown 当 no、不因任意 unknown 强制弃答。空选择/全 unknown 仍交给 JEV；接口/解析失败留档停止，不重试、不自动改分。第 1–6 步行为保持不变。

本次已确认的接口：每帧原图与独立框/mask 轮廓/完整编号展示，附类别/分数清单；窗口级绑定按 `(entity_key, frame_id)` 唯一，可跨 check 复用；绑定为 `bound` 或带原因的 `missing`。运动提议显式指定目标、绝对/相对模式及相对模式的参考实体，不能从实体列表顺序猜角色。只要求实际提议所需帧的绑定；缺失时保留提议与原因，不假装执行成功。

现有 [`experts/`](experts/README.md) 算法保持原样。YOLO ONNX Runtime CPU 只在第 2 步运行；第 3、4 步只复用这些候选，原候选文件保持 `unbound`，VLM 绑定另存到匹配记录。`expert_execution.py` 将绑定转换为二维几何参数和 `MaskSeed`，调用既有接口。用户确认：参考绑定/mask 缺失时仍跟踪可靠目标、保留绝对运动，相对运动明确缺测；目标 mask 缺失时不调用跟踪。旧 `collect()`、`track_pair()`、`score_saved()` 保留但不调用。依赖不变，运行/完整测试依赖分别见 `requirements-experts.txt`、`requirements-experts-test.txt`。

当前新增的开发评测入口为 [`benchmark/`](benchmark/README.md)：离线预检、RoboReward validation 5–10 条真实链路试点、约 50 条 validation 开发评估（包含试点）。本阶段暂不做 baseline、消融、长程专项审核或 test 集正式评估。工具包含标签隔离、固定清单、零自动重试的批量运行、失败与中断留档、覆盖率/误差/成本统计和离线报告；153 项框架测试与 23 项评测工具测试均已离线通过，尚未进行真实模型推理。

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
| YOLO 候选（已接入） | 仅在已选窗口的全部原始帧上执行分割，重叠帧复用；候选不绑定任务实体。 |
| VLM 匹配（已接入） | 同时关联窗口/check/expert 并选择候选编号绑定实体，显式保留绑定缺失及运动目标/参考角色；不回答 check、不评分。 |
| Expert 使用（已接入） | 校验全部提议与绑定后复用候选做二维几何；以提议起始帧的目标/参考 mask 调用连续帧 LK；不重复分割，缺失与中断保留。 |
| 逐 check 证据包（已接入） | 程序按 check 汇集图片/绑定、几何与独立运动摘要、覆盖/缺测和显式阈值比较；不重跑 expert，不回答 check。 |
| JEV 逐 check 判断（已接入） | 对每个 check 的全部相关窗口图文综合判断一次 yes/no/unknown；空证据由程序明确记 unknown；不按窗口投票、不按置信度覆盖选择。 |
| JEV 纯文字终评（已接入） | 全部 check 完成后一次纯文字 Choice，按完整 rubric 选择等级或 insufficient_evidence；证据摘要由程序整理，未知仅在影响等级区分时导致弃答。 |

粗看阶段的 Qwen 不选窗、不设优先级、不指派 expert、不输出 reward；匹配阶段的 Qwen 可以提出 expert 调用，但不回答 check 或评分。二维重叠或相对运动不能独立证明三维支撑、抓取、释放或放稳。稀疏帧看不清，也不能断言窗口内部一定能看清。

## 本次已落地

- 用户确认的完整英文 prompt：[prompts/qwen_coarse_observation.txt](prompts/qwen_coarse_observation.txt)。运行入口读取它，再附原始 instruction、帧号、PTS 和所有采样图片。
- 解析与结构校验：[coarse_observation.py](coarse_observation.py)，schema 为 `task-guided-coarse-v2`。
- 模型：`qwen3.8-flash`，OpenAI 兼容 Chat Completions，`enable_thinking=false`；JEV 为 `jev-1.14.0`。
- 采样仍从第一帧原始 PTS 开始，每 2 秒取每个非空时间段的第一帧，额外保留末帧，按原帧号去重，无 8 帧上限。沿用 `--sample-interval-seconds`。
- 选窗请求与输出契约：[window_selection.py](window_selection.py)，协议 `jev-window-selection-v2`；入口调用 `select_windows()`，替换旧 `route()`。
- VLM 匹配契约与校验：[vlm_matching.py](vlm_matching.py)，协议 `vlm-window-check-expert-bindings-v3`；独立 prompt 为 [prompts/qwen_window_matching.txt](prompts/qwen_window_matching.txt)，入口在候选生成后调用 `match_windows()`，保存绑定后交给 `execute_experts()`。

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
5. `run.json` 保留 `window_selection` 协议、决策单位、共享背景、每窗口问题数和阈值。选窗结果先交给下面的 YOLO 候选生成；选窗本身不做窗口/check 关联或工具选择。

## 当前 YOLO 候选前置（计划第 2 步）

1. `evaluate_rollout.py::generate_candidates()` 复用已有选窗/checklist/PTS 边界校验，只处理 `selected_windows`。每个窗口保留原窗口 ID、两端帧号和完整 `frame_ids`；对这些帧号取并集并按原帧号执行一次分割，未入选区间不检测。逐帧解码和推理，内存中不累积全部原始图片。
2. 复用 `ObjectSpatialExpert`，限定 ONNX Runtime `CPUExecutionProvider`，沿用现有检测阈值和输出合约，不改两个 expert 的算法。新增必填参数 `--yolo-model` 指定本地 ONNX 文件；在读取视频或调用模型前检查路径，不自动下载权重。模型输入/输出合约在实际初始化和推理时由 expert 校验。
3. `03-yolo-candidates.json` 保存 `protocol=yolo-window-candidates-v1`、全帧去重策略、`binding_status=unbound`、模型 SHA256/类别表/运行库/阈值等来源，以及 `windows` 和去重后的 `frames`。窗口通过 `frame_ids` 引用共享帧记录；每个候选保留 `candidate_id, frame_id, pts_seconds, class_id, class_name, confidence, box_xyxy_px, mask, mask_status`。候选编号如 `f120:d1` 只表示该原始帧内的候选，不是跨帧身份。
4. 无检测仍保存帧记录和 `status=no_detections`；空 mask 保留 `mask_status=empty`，不制造任务实体绑定。空选择保存空候选汇总，不初始化分割器、不读候选帧。解码/推理失败直接报错，不写完成态候选汇总或 `result.json`，不补造检测；主入口仍释放视频句柄。
5. `run.json` 记录候选协议、本地模型路径和帧策略。候选文件本身不写任务实体身份；生成后进入匹配与第 4 步执行。最终 `result.json` 含各阶段文件、检测帧数、候选总数、匹配窗口数、执行记录数与耗时；候选只供后续匹配/执行使用；评分在第 6、7 步完成。

运行示例（先准备符合 [expert 合约](experts/README.md) 的本地分割模型）：

```sh
python -m pip install -r requirements-experts.txt
python evaluate_rollout.py --video "$ROLLOUT_VIDEO" --task "$ROLLOUT_TASK" \
  --episode "$ROLLOUT_SAMPLE_ID" --split test --out run-001 \
  --yolo-model /absolute/path/yolo26n-seg.onnx
```

## 当前 VLM 匹配与候选绑定（计划第 3 步）

1. 模型仍为 `MATCHING_MODEL=qwen3.8-flash`、非思考模式、JSON 输出，每个选中窗口一次请求。沿用 DashScope 密钥/base URL 解析方式，不修改粗观察、选窗或两个 expert 算法。输入为原 instruction、全部 requirements、完整 checklist、当前窗口原 ID/范围、expert 能力和已有候选。空选择保存空匹配汇总，不读图片、不调用匹配 API。
2. `candidate_presentation.py` 在原图副本上画框、mask 轮廓、完整 `f帧号:d编号`，并添加类别/分数/mask 状态侧栏；每帧单独提供原图和候选图。连续原帧和 PTS 完整保留，不裁剪、不重新分割。图片保存为 `matching_frames/<window_id>/<frame_id>.jpg` 与 `<frame_id>-candidates.jpg`；有效匹配记录中的 `images` 保存确切文件/帧/PTS 对应。候选元数据不包含长 RLE 数组，完整 mask 仍保存在第 2 步文件。
3. 调用前验证完整选窗/checklist/PTS 边界，并校验候选覆盖所有已选原帧且无重复，窗口映射、帧内编号、候选帧号/PTS、原始像素框、mask 尺寸/RLE/状态相符。实际解码图片尺寸也须与候选一致。非法输入在对应请求前报错；不填补或重新检测候选。
4. 响应协议升级为 `vlm-window-check-expert-bindings-v3`，严格返回 `window_id, checks, bindings`。每个 check 恰有一条 `check_id, relevance, reason, experts, limitations`，相关性仍为 `related/unrelated/uncertain`。不相关项无 expert；相关但无 expert 覆盖或相关性不确定时须解释限制。禁止新增/遗漏 check、旧工具名及答案/评分字段。
5. 窗口级 `bindings` 每项严格为 `entity_key, frame_id, status, candidate_id, reason`。`bound` 必须引用该帧真实存在的候选；找不到或身份不明时为 `missing`、`candidate_id=null` 且原因非空。同一 `(entity_key, frame_id)` 只出现一次，跨 check 复用；不凭类别或帧内编号建立跨帧身份。空 mask 不禁止清楚的对象身份绑定，但也不因此获得可靠运动 seed。
6. `object_spatial` 提议字段为 `expert_id, entity_keys, frame_ids, reason`，只提议使用已生成候选。`short_window_motion` 提议字段为 `expert_id, target_entity_key, reference_entity_key, motion_mode, frame_ids, reason`：绝对模式参考为 null；相对模式须有明确、不同的参考实体，双方不得绑定同一候选。实体须来自当前 check，运动至少两帧且连续，目标/参考角色不明确时不猜测、不提出不完整角色提议。角色明确但 seed 绑定缺失时保留提议及缺失原因。
7. 只要求提议必需的绑定：空间提议的每个实体×每个帧；运动提议仅需首帧目标及相对模式的参考实体。必须恰好覆盖这些绑定键，不额外要求逐帧跟踪身份，也不接受重复或无用途的绑定。程序在保存记录时补充 PTS 及来源（匹配模型、请求 trace、`03-yolo-candidates.json`），不把模型理由写成已执行的工具证据。
8. 原始 trace、逐窗口结果与汇总分别为 `03-match-<window_id>.json`、`03-matching-<window_id>.json`、`03-window-matching.json`，原响应不改写。已有的保守清理继续生效：仅删除无用途的合法绑定，仅在明确的空候选帧补 `missing`；清理留档 `03-matching-<window_id>-binding-cleanup.json`。重复绑定、非法候选或不合法专家提案仍直接报错。
9. **必需绑定遗漏恢复（第 1 项）**：程序根据已校验提案生成去重的必填 `(entity_key, frame_id)` 清单。正常响应完整时无额外调用；剩余遗漏最多触发一次定向视觉补问，仅发送所需帧的原图/候选图和现有上下文。补问 prompt 为 `prompts/qwen_binding_completion.txt`，返回仅含 `window_id, bindings`，不得改变已有绑定、check、相关性或提案。合法的部分补答可以保留；非法补答整体拒绝，不根据最终 score 再试。原请求、补问请求、审计分别留档 `03-match-<window_id>.json`、`03-bind-<window_id>.json`、`03-matching-<window_id>-binding-completion.json`。
10. 补问后仍未解决的键记录为 `status: unresolved, candidate_id: null`，来源 `program_unresolved_binding, model: null`；这表示漏答未解决，不表示目标不存在。模型明确找不到/身份不清的候选仍是 `missing`；补问获得的绑定来源为 `matching_vlm_binding_completion`。逐窗口 `binding_completion` 保存程序必填清单、补问清单、未解决清单、状态、错误和请求引用；下游严格验证清单、候选、来源和 PTS。空间或运动提案使用了 unresolved 键时，仅该提案记录 `not_executed / missing / binding_unresolved`，不调用测量算法、不制造结果；该窗口相关/不确定的原图继续进入证据包和 JEV。已有明确 `missing` 的处理保持原语义（例如可靠目标仍可做绝对运动）。JEV 保留 `yes/no/unknown` 与最终 `insufficient_evidence`，不会由程序补出 score。
11. 定向补问计入 API 调用审计和每视频调用预算；不增加其他阶段的网络重试，调用预算、整段超时、解码与存储错误仍可终止运行。`run.json` 记录 v3 匹配协议和补问策略。修改后的 framework 使用新运行目录，保留 validation50-08 原记录。

## 当前绑定后的 expert 执行（计划第 4 步）

1. `expert_execution.py::validate_execution_inputs()` 在任何几何或跟踪调用前重新校验所有输入：选窗、完整 checklist、候选/frame/PTS/mask、匹配上下文与窗口覆盖、check/expert/实体引用、绑定来源、显式运动角色及连续的提议帧区间。复用匹配解析契约；无效引用直接报错，不补绑定、不选新窗口或 check。原观察、候选和匹配记录不被改写。
2. `object_spatial` 对提议的每个帧读取已有绑定候选，保留类别/分数/像素框及原 mask 的文件引用；在同帧提议实体之间两两调用既有 `pairwise_spatial_evidence()`，记录中心距离、框间距/IoU、带方向框间隙、mask 交集/IoU。输出的 `entity_keys=[a,b]` 明确对应几何字段的 a/b，不推测任务角色。单实体只返回对象资料；绑定缺失时保留对象/配对缺测。空 mask 仍可计算框几何，mask 测量为 null；两个实体选中同一候选时注明身份无法区分，不将结果当成身份核验。
3. `short_window_motion` 只取提议区间**首帧**绑定候选的非空 mask 创建 `MaskSeed`，source 为候选编号；目标和参考来自显式字段。目标绑定缺失或 mask 为空时 `execution_status=not_executed`、测量 `status=missing`，不读取跟踪帧、不调用 LK、不回退到框。参考缺失时仍跟踪目标，原始 `target` 保存绝对运动，`relative_motion` 全部缺测，记录具体参考实体、候选及缺失原因；不将绝对结果当作相对结果。
4. 从原视频读取提议区间全部连续原始帧，含两端，使用原帧号/PTS并校验尺寸；调用未改动的 `ShortWindowMotionExpert.track()`。不重新分割、不重选点、不插值或拼接缺口。点不足、跟踪中断和参考轨迹丢失保留在原始 `samples/interruptions` 中。绝对提议的 `status` 反映目标覆盖；相对提议的 `status` 反映双方同时可用的相对覆盖，即使相对缺测也可保留有效目标轨迹。`complete/partial/missing` 均为测量覆盖，绝不是 check 答案。
5. `evaluate_rollout.py::execute_experts()` 保存 `04-expert-execution.json`，协议 `bound-expert-execution-v1`。每个实际提议一条独立 `execution-XXXX` 记录，含 `window_id/check_id/expert_id/entity_keys`、原始帧/PTS、完整提议与关联理由/限制、所用绑定及其来源、候选/匹配/请求文件引用、执行状态和原始结果。几何 mask 用文件引用避免复制长 RLE；运动保留 expert 原始输出。不同 check 的执行保持各自关联，不将模型提议理由标为支持证据。没有提议的相关 check 继续保留在匹配记录，不能据此当成已核验。
6. 空选择或零提议保存空 `records`，不读取运动帧或加载运动 expert。解码/尺寸/接口错误直接报错，不写完成态执行汇总或 `result.json`，保留前面阶段文件并释放视频句柄。缺测本身会正常保存，不伪造测量成功。
7. `run.json` 保留执行协议、候选复用、连续帧及缺参考策略；`result.json` 保留 `execution_file/expert_execution_count`。第 4 步原始输出现交给下面的第 5 步整理。两个 expert 算法未改，没有新增模型调用或依赖。

## 当前逐 check 证据包与阈值比较（计划第 5 步）

1. `evidence_packaging.py::build_evidence_packages()` 对原 checklist 的每个 check 恰好生成一个包，保留原 instruction、该 check 及关联 requirements；不新增/删减 check。仅收集匹配中 relevance 为 `related` 或 `uncertain` 的已选窗口，保留 uncertain、原理由与限制。没有相关窗口也保存空包、`status=no_related_windows` 和原因；没有 expert 的相关 check 仍保留图片，缺测不预先转换成 check 的 unknown。
2. 收集这些窗口的全部连续原图文件引用，按原始帧号去重、按时间排序，并保留各窗口原图/候选标注图的所有来源；不再抽帧、裁剪、复制或解码图片。记录原帧号/PTS、相对整个视频的图片覆盖/未覆盖区间及是否包含首末帧。覆盖仅表示可供后续读取的图片范围，不声称任何语义要求已全程满足。
3. 包内绑定保留窗口、实体、原帧/PTS、候选编号、bound/missing/unresolved、理由和 VLM/请求/候选文件来源；同一帧在不同窗口的绑定分别保存，不用某一条覆盖其他绑定。原始候选、匹配和执行输出不改写。
4. 几何按每次执行、每个实体对、每个既有测量分别计算最小、最大、最后有效值，记录单位、对应帧/PTS与有效/缺测区间。极值相同时选最早出现帧。空间提议可只含离散帧，摘要明确 `proposed_frames_only_no_interpolation`；不会据此声称中间帧都被测量。单实体仍有绑定覆盖记录；空 mask 保留框测量，mask 测量摘要为空。
5. 运动分别对目标、实际参考和提议的相对运动整理首个/最后有效 from-seed 位移、最大 from-seed 位移（含向量/像素模长/图像方向及对应帧/PTS）、有效/缺测区间、最后一帧是否有测量、缺失原因和原始 interruptions。不同执行的 seed 起点分别保留，不拼接、不重置起点、不把特征点位移称为中心轨迹；不计算未经确认的速度、路径长度或“放稳”阈值。参考缺失时保留目标绝对摘要，相对摘要明确缺测；目标 mask 缺失时保留未执行记录。
6. 原始逐帧 `samples`、mask/RLE 和几何数组均不放入测量摘要；通过 `raw_result_ref={file,record_id}` 回溯 `04-expert-execution.json`。提议帧列表压缩为连续帧区间。每条摘要继续关联执行 ID、窗口、expert、实体与来源。这些材料现交给下面的第 6 步；第 5 步自身只构造材料。
7. 整理前重新校验上游契约和执行记录的完整覆盖、check/实体/绑定/来源、候选、帧/PTS、数值有限性、运动 seed/有效状态/中断等；不重新运行 YOLO、几何 expert 或 LK。非法输入不写完成态证据包或 `result.json`，已保存上游文件保留，主入口释放视频句柄。合法缺测正常保存。
8. `evaluate_rollout.py::package_evidence()` 保存 `05-check-evidence.json`，协议 `per-check-evidence-packages-v1`。`run.json` 增加证据范围/摘要策略和显式阈值配置及来源文件 SHA256；`result.json` 增加 `evidence_file/check_evidence_count`，第 5 步输出交给第 6、7 步后，由终评产生最终状态和分数。真实 API 验收尚未运行。

### 用户确认的显式阈值配置

可选参数：`--threshold-config /absolute/path/thresholds.json`。不提供时，每个 check 保存 `not_configured/no_explicit_threshold_contract`，不会推断阈值。可先用以下合法空配置；实际比较项须自行明确填写，没有预置的任务阈值：

```json
{
  "protocol": "explicit-measurement-thresholds-v1",
  "comparisons": []
}
```

每个 comparison 严格包含以下字段，不接受隐式单位转换、自由文本条件或额外字段：

| 字段 | 含义 |
|---|---|
| `id` | 本配置内唯一的比较 ID。 |
| `window_id`, `check_id`, `expert_id` | 原窗口、原 check 和 `object_spatial` / `short_window_motion`；共同定位一次执行，防止混用不同起点。 |
| `entity_keys` | 空间为明确的 `[a,b]`；对称量与顺序无关，带方向间隙严格按此 a/b 计算。运动 target/reference 为对应单实体，relative_motion 为显式 `[目标,参考]`，必须与提议角色一致。 |
| `metric`, `unit` | 只能是下面表中的固定测量及其精确单位。 |
| `aggregation` | 空间为 `at_frame` / `min` / `max`；运动为 `at_frame` / `max`。 |
| `frame_range` | 原始帧号 `[first,last]`，含两端且在视频内；`at_frame` 要求两端相等。帧号是零起始索引。 |
| `operator` | `<`, `<=`, `>`, `>=`，直接比较，不附加容差或死区。 |
| `threshold` | 明确的有限数值；不接受布尔值、NaN 或 Infinity。 |

| `metric` | `unit` |
|---|---|
| `center_distance_px`, `box_edge_distance_px` | `px` |
| `signed_box_gaps_px.a_left_of_b`, `signed_box_gaps_px.a_above_b` | `px`（可为负值） |
| `box_iou`, `mask_iou` | `ratio`（测量值范围 0–1） |
| `mask_intersection_px` | `px^2` |
| `target.from_seed.magnitude_px`, `reference.from_seed.magnitude_px`, `relative_motion.from_seed.magnitude_px` | `px` |

配置基础结构/单位/范围格式在读取视频或调用模型前校验；动态 check/窗口/实体及原始帧引用在第 5 步校验。运动量始终相对**该次执行的 seed**，即使配置区间从更晚帧开始也不重置 seed。空间区间极值需要区间内逐帧都有该测量，运动区间极值需要同一次连续跟踪覆盖全部指定帧；不借其他窗口或另一条轨迹补齐。未提议、未选择、空 mask、跟踪中断或角色不匹配等导致没有完整对应测量时返回 `not_comparable`、原因和 `satisfied=null`，不写成 false。已有局部摘要仍保留。完整测量才返回 `compared`、所用数值/单位/帧/PTS/范围及 `satisfied=true/false`。这只是以已给定绑定为前提的数值比较，不等同于 check 的 yes/no，也不自动证明三维接触、抓住或放稳。

## 当前 JEV 图文逐 check 判断（计划第 6 步）

1. `check_judgment.py` 定义协议 `jev-per-check-image-text-v1`、请求组装、完整图片适配与 Choice 响应解析；prompt 独立保存在 `prompts/jev_check_judgment.txt`。`evaluate_rollout.py::judge_checks()` 在第 5 步之后接入。阶段边界通过程序重新整理第 5 步的纯算术结果，与输入证据包严格核对；不重新运行 YOLO、几何 expert 或 LK，不修改已有阶段输出。
2. 固定模型仍为 `jev-1.14.0`，调用既有 `https://api.typesafe.ai/v1/systemone`。每个有图片或可用测量的 check 单独一个请求、一个 Choice（原 check ID），将原 instruction、相关 requirements、当前 check 和该 check 的所有窗口/绑定/测量摘要/阈值比较/覆盖/缺测一起提交；不包含其他 check 的上下文，不按窗口分别判断或投票。没有图片也没有有效测量的空包不调用 API，由程序明确记录 `answer=unknown`、`answer_source=program_no_evidence`、原因与 `actual_model=null`，保留在完整判断列表中。
3. 用户确认每个去重原帧同时发送原图和一张已有候选编号图，按原帧号/PTS 排序；原图/编号图是同一观察的两种展示。所有窗口来源和绑定保留，重叠原帧只发送这一对图。编号只标识帧内候选，实体身份仍依据显式绑定及图片核对；不凭类别或列表位置猜身份。
4. 依据 [TypeSafe 官方图像接口](https://docs.typesafe.ai/images)，使用顶层 `assets`（JPEG/base64）和 `TypeSafe-Danger-Zone-Images: true` 请求头。每张图使用独立 UUID，在 state 中只引用一次；不把 VLM 的 `image_url` 格式或文件路径当作已发送图片。既有文字选窗 helper 的请求格式保持不变。
5. 官方图像能力仍为实验功能。用户确认：保持完整 canvas，超大图片等比例缩小至最长边 ≤1568，短边不足 280 时居中补灰边；另存为 `check_frames/cXXXX/<原帧号>-original.jpg` 与 `-candidates.jpg`。尺寸已合规的既有 JPEG 保留原字节。每个附件记录原/适配尺寸、实际 xy 缩放比、四边补边、来源/适配文件及 SHA256、原帧号/PTS。原图和已有候选图不被覆盖；测量与 mask 坐标仍是原视频坐标，适配仅作用于显示 canvas。
6. 单张准备图超过 16 MiB 即报错；按实际 JSON/UTF-8 字节检查 32 MiB 请求体上限。服务端重编码后的图片总量、展开大小或模型上下文限制由服务端判定，错误保留到 trace。不自动丢帧、另行抽帧、分窗投票、切换模型或上下文模式，也不自动重试。坏图/缺文件同样是实现失败，不伪装成 unknown。
7. prompt 明确区分“至少曾经发生”“episode 最终成立”和“要求区间持续成立”；早期成果可能撤销，最后有效测量不代表末帧。以原 instruction 与当前 check 决定所需语义范围，不足或冲突无法解决时选择 unknown。expert 缺测不强制 unknown（图片可能足够）；数值比较或工具完整也不强制 yes/no。二维重叠/局部点运动本身不证明三维物理状态，不新增“放稳/抓住”等默认阈值。
8. 响应只读取固定模型、原 check 对应的结构化 Choice，严格接受 `yes/no/unknown`；错误 model/check/type/choice 报错。confidence/probabilities/usage 仅留档，未返回则保留 null；不从这些元数据推导新答案或阈值，不把 Choice 解释成自由文本理由。每条判断的 `provided_evidence` 和图片 manifest 表示实际提供的材料，不标成“支持证据”，也不声称 JEV 选择了具体证据。
9. 原始请求/响应 trace：`06-judge-cXXXX.json`；有效逐 check 记录（含程序空包 unknown）：`06-check-cXXXX.json`；失败：`06-failed-cXXXX.json`；全部完成后才保存 `06-check-judgments.json`。XXXX 是 checklist 输入顺序索引，原 check ID 独立保留，不用任意 check 文本构造路径。trace 单独保存图片 manifest 并引用实际适配 JPEG，不重复存 base64；记录请求头（不含 Authorization）、字节数、API 用量/耗时与 HTTP 错误正文。非法响应仍保留原 trace；失败时保留此前 check 文件，不写完成态总判断或 `result.json`，主入口释放视频句柄。
10. `run.json` 增加第 6 步协议/模型/图片策略/空包规则/零重试/prompt SHA256；`result.json` 增加 `judgment_file/check_judgment_count/jev_check_call_count`，这些判断继续交给第 7 步；第 6 步不产生分数，真实付费模型验收尚未运行。

## 当前 JEV 纯文字终评（计划第 7 步）

1. `final_evaluation.py` 定义协议 `jev-text-final-evaluation-v1`、阶段边界校验、程序汇总、请求及结构化响应解析；`evaluate_rollout.py::evaluate_final()` 在全部 check 判断完成后接入。复核第 5 步与上游记录一致、第 6 步上下文/check 顺序/数量/答案/来源/引用一致；缺少、重复、错位或未完成判断会在终评请求前报错。不重跑两个 expert、不读图片、不改变原 checklist。
2. 固定 `jev-1.14.0`，每条 rollout 一次纯文字 `grade` Choice，`state` 为序列化文字，无图片附件或图像请求头。空选择、全部程序 unknown 或全部 JEV unknown 仍调用一次终评。独立 prompt 为 `prompts/jev_final_evaluation.txt`；固定 rubric 为 `prompts/roboreward_rubric.json`，完整保留 [RoboReward 论文 A.2.2](https://arxiv.org/html/2601.00675v2) 五档定义与“评价最终状态、忽略时间限制”的语义，注明来源并以独立措辞整理，不增加评分规则。
3. `07-final-materials.json` 保存原 instruction、完整 requirements/rubric、episode 原帧/PTS 范围和全部原 check/答案/来源。每项包含第 5 步已有窗口、绑定及来源、测量摘要/单位/范围、覆盖/缺测/中断、阈值比较、限制与原始结果引用；未观察范围、unknown、缺绑定和不可比较项按记录整理到 `unresolved`。不填造 JEV unknown 的理由，不传原始长数组或图片；check 的问题文本保留其要求时间范围，覆盖范围不冒充语义成立范围。
4. JEV 综合全部记录按 rubric 选 `level_1…level_5` 或 `insufficient_evidence`。不将 yes 比例换分、不将 unknown 当 no、不因任意 unknown 强制弃答；判断未解决信息是否影响等级区分。prompt 区分曾经/终局/持续，提示早期成果撤销、局部测量/独立 seed/二维物理限制，不重新回答 check 或请求窗口。
5. 程序严格读取固定模型、唯一 `grade`、`type=choice` 和允许的选择；等级仅映射为整数 1–5，弃答为 `score=null`。confidence/probabilities/usage 仅留档，缺失为 null，不作阈值或改分依据。不将结构化 Choice 当自由文本解释，也不接纳模型额外字段作为解释/证据选择；材料引用表示实际提供的记录，不自动标为支持证据。
6. 原始请求/响应 trace 为 `07-evaluate.json`；有效终评为 `07-final-evaluation.json`，含所选等级、分数、状态、模型/元数据/耗时及材料/判断/证据/请求引用。校验、接口或解析失败保存 `07-failed.json`，保留已有材料/trace/第 6 步记录，不重试、不回退旧评分、不写完成态终评或 `result.json`；主入口释放视频句柄。
7. `run.json` 增加终评模型/协议/一次调用/零重试/纯文字策略、rubric ID/来源/SHA256 与 prompt SHA256；`result.json` 增加 `jev_choice/final_evaluation_file/final_materials_file/jev_final_call_count`，最终为 `status=scored, score=1…5` 或 `status=insufficient_evidence, score=null`。已有阶段路径/计数保留。旧 `collect()/track_pair()/score_saved()` 继续保留但不调用，没有新增依赖或 CLI 参数。

## 保留的旧取证与评分实现（当前入口不调用）

以下记录先前实现的能力边界。`collect()` 仍接收旧配对/工具格式，`score_saved()` 仍读取旧证据格式；它们未被改造为新版链路，也不接受新的选窗结果。回归测试保留，不能据此宣称新下游已实现。

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

当前入口已接入第 1–7 步。纯文字终评已获用户确认并完成本地实现；以下真实端到端评分验收仍需另行确认后进行。

1. 受限适配尚待用户查看覆盖范围；官方样本中的自然语言和物理要求可能仍无法得到要求级判定，真实数值终分不能保证。
2. 香港端点和 Secrets 尚未核对。旧代码仍有缺 `DASHSCOPE_BASE_URL` 时回退国际站地址的路径，必须在真实调用前移除并补前置检查。
3. RoboReward A.2.2 rubric 与 Choice 接口已核对；仍需固定真实样本及标签隔离、完整版本/调用/费用留档、真实终评分级及成果撤销案例验收。

先检查本地改动与模拟结果，再由用户确认下一步；当前不进行付费验收。

## 验证状态与复跑

2026-11-21 第 7 步检查：153 项离线测试全部通过，无跳过（原 139 项，新增 14 项并更新两处主入口集成回归）。模拟实际 HTTP 序列化验证单次纯文字请求、完整要求/rubric/答案/程序摘要、非均匀 PTS/多窗口/缺测与来源、全 unknown 仍调用、未知不强制弃答/yes 比例不计分、结构化选择与元数据隔离、非法或未完成上游记录拒绝、非法响应与 HTTP 上下文错误留档/零重试、终评失败保留第 6 步且释放视频。既有实际 LK/合成 ONNX CPU 回归通过；不增加分割次数。未调用真实 VLM/JEV，未运行 Actions，未 push；真实终评分级质量尚未验收。

2026-11-21 第 6 步检查：139 项离线测试全部通过，无跳过（原 123 项，新增 16 项并更新主入口/候选集成回归）。实际生成 JPEG、缩放/补边并目视检查合成编号图；模拟 HTTP 层验证图片字节/专用请求头/UUID 单次引用、帧/PTS 与来源哈希、逐 check 独立上下文及多窗口单次判断、空包程序 unknown、纯图片/工具缺测仍交给 JEV、结构化 Choice/元数据边界、非法后续响应保留前序记录、损坏图片/请求体/服务上下文错误停止、无重试/无 expert 重跑及资源释放。既有合成 ONNX CPU 与实际 LK 回归仍通过。环境沿用 Python 3.14.6、OpenCV 4.14.0、NumPy 2.5.5、ONNX Runtime 1.32.0、ONNX 1.23.2。**API 均为模拟，没有验证真实 JEV 视觉质量、时间语义判断或真实接口端到端；未调用真实 VLM/JEV，未运行新 Actions，未 push。**

2026-11-21 第 5 步检查：123 项离线测试全部通过，无跳过（原 102 项，新增 21 项并更新主入口/候选集成回归）。覆盖逐 check 完整保留、related/uncertain/纯图片窗口、原图去重与绑定来源、几何/实际 LK 摘要、非均匀 PTS、空选择/空 mask、跟踪丢失、不同 seed 不拼接、全部比较符号、单位/范围/方向角色、局部覆盖拒绝比较、显式 JSON 前置校验及来源哈希、原始数组只留引用、失败后资源释放。既有合成 ONNX Runtime CPU 集成确认不增加分割或模型调用次数。环境沿用 Python 3.14.6、OpenCV 4.14.0、NumPy 2.5.5、ONNX Runtime 1.32.0、ONNX 1.23.2。**仅离线合成输入/模拟 API，未验收真实模型/视频质量，未调用真实 VLM/JEV，未运行新 Actions，未 push。**

2026-11-21 第 4 步检查：102 项离线测试全部通过，无跳过（原 83 项，新增 19 项并更新入口/ONNX 集成回归）。实际运行已有 OpenCV LK 与合成 ONNX Runtime CPU 图，覆盖候选复用、全部引用预校验、绑定来源/PTS、同帧几何、空 mask/无检测、单实体/单帧窗口、显式目标/参考、窗口内部跟踪起点、参考缺失时绝对结果保留、点不足/中断、空选择、解码/尺寸失败及资源释放。主入口验证第 4 步不增加 YOLO 推理次数。环境沿用 Python 3.14.6、OpenCV 4.14.0、NumPy 2.5.5、ONNX Runtime 1.32.0、ONNX 1.23.2。**仅离线合成输入/模拟 API；未调用真实 VLM/JEV、未验收真实权重或视频绑定/跟踪质量，未运行新 Actions，未 push。**

2026-11-21 第 3 步检查：83 项离线测试全部通过，无跳过（此前 72 项，新增 11 项并更新匹配/入口回归）。新增覆盖窗口共享绑定、缺失原因、候选跨帧/伪造/重复引用、相对运动显式角色、首帧 seed 绑定、候选边界及 mask 一致性、原图＋编号图输入与文件映射、无检测/空 mask、非法响应 trace、空选择和入口停止。合成 ONNX CPU 分割＋模拟匹配响应的集成检查确认匹配不重复推理。实际绘制并目视检查候选编号展示。环境沿用 Python 3.14.6、OpenCV 4.14.0、NumPy 2.5.5、ONNX Runtime 1.32.0、ONNX 1.23.2。**未验收真实 VLM 绑定质量或真实 YOLO26 权重，未调用真实 VLM/JEV，未运行新 Actions，未 push。**

2026-11-21 候选前置检查：72 项离线测试全部通过，无跳过（原有 63 项，新增 9 项并更新入口停止检查）。Python 3.14.6、OpenCV 4.14.0、NumPy 2.5.5、ONNX Runtime 1.32.0、ONNX 1.23.2。新增检查实际运行 ONNX Runtime CPU 的合成分割合约图，覆盖已选窗口全帧、跨窗口去重与跳过未选帧、原帧号/非均匀 PTS、候选/mask/来源保存、同类候选保持未绑定、空选择、无检测、空 mask、单帧窗口、非法边界、解码失败及主入口停止/释放句柄。**尚未验收真实 YOLO26 权重和真实视频检测质量；未调用真实 VLM/JEV，未运行新 Actions，未 push。**

本次 expert 本地验证：63 项离线检查全部通过，无跳过（原有 51 项、新增 12 项）；环境为 Python 3.14.6、OpenCV 4.14.0、NumPy 2.5.5、ONNX Runtime 1.32.0、ONNX 1.23.2。实际运行 OpenCV 合成平移/遮挡及前后向不一致检查，实际运行 ONNX Runtime CPU 的小型合成分割合约图，覆盖 RGB/letterbox、原图坐标与 mask 还原、无检测、空 mask、格式错误和 JSON 序列化。**未下载或验收 YOLO26 预训练权重，未验证真实视频检测/跟踪质量，未调用真实 VLM/JEV，未运行新 Actions。** 前三步和入口停止行为的既有模拟回归保持通过。

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

完整离线检查先安装 `requirements-experts-test.txt`。结构、纯选窗与纯匹配契约测试只依赖标准库；缺少 OpenCV/NumPy 或 ONNX/ONNX Runtime 时相应集成测试明确跳过。CPU Actions 安装依赖，执行模拟集成检查并上传 `coarse-observation-checks` artifact，包含日志及批准的粗看 prompt。

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
