# 独立 expert 取证接口

本包实现用户指定的两个 expert。**`object_spatial` 已接入候选前置，其既有二维几何接口和 `short_window_motion` 已在第 4 步接入主流程。** expert 本身不会调用 VLM/JEV、选择窗口、修改 checklist、回答 check 或评分；也不会自动下载权重。实体绑定已由主流程的匹配 VLM 完成；本次由 `expert_execution.py` 按绑定执行既有几何/运动接口；第 5 步已接入逐 check 证据包，第 6 步已接入 JEV 图文逐 check 判断，纯文字终评仍属后续步骤。

## 主流程候选前置

用户已确认对已选窗口的全部连续原始帧运行分割，包含两端；窗口重叠处同一原始帧只检测一次。`evaluate_rollout.py::generate_candidates()` 逐帧调用本包，保存 `03-yolo-candidates.json`，窗口通过原始 `frame_ids` 引用共享候选帧。原帧号/PTS、框/mask、类别/分数及模型来源保持本包输出格式，候选仍未绑定任务实体。空选择不加载分割器；无检测或空 mask 如实保存；解码或推理失败直接报错，不写完成态汇总。

主入口必填 `--yolo-model /absolute/path/yolo26n-seg.onnx`，候选保存后进入匹配，匹配保存后执行第 4 步，保存 `04-expert-execution.json` 后由第 5 步生成 `05-check-evidence.json`，再进入第 6 步图文判断，保存 `06-check-judgments.json` 后返回 `awaiting_final_evaluation`。第 3 步现已接入：连续原图及候选标注图交给匹配 VLM，输出共享的 bound/missing 绑定和显式运动角色；绑定单独保存在匹配记录，不改本包的候选。第 4 步先校验所有引用，再将运动区间首帧的明确绑定候选转成 MaskSeed；两个 expert 的独立算法接口保持不变。目标绑定/mask 缺失时不调用跟踪；参考缺失时仍跟踪目标、保留绝对运动，相对运动明确缺测。空 mask 不阻断框几何，mask 重叠测量保持 null。每条执行关联窗口/check/实体/绑定及来源，保留原始结果；不重新运行分割、不增加阈值解释或任务结论。

## 文件职责

| 文件 | 职责 |
|---|---|
| `types.py` | 原始帧号、PTS、BGR 图像及带来源的对象 mask；输入边界校验。 |
| `object_spatial.py` | ONNX Runtime CPU 分割、letterbox 坐标还原、候选对象与模型来源。 |
| `masks.py` | 原始分辨率 mask 的无损 JSON RLE 编解码。 |
| `geometry.py` | 同帧两个候选的二维距离、框间距、框/mask 重叠。 |
| `short_window_motion.py` | mask 内选点、金字塔 LK、前后向一致性、绝对/相对图像位移和缺测区间。 |

## 依赖与模型

Python 3.11+；运行依赖与完整离线测试依赖分别安装：

```sh
python -m pip install -r requirements-experts.txt
# 若需运行 ONNX 合约测试，再安装：
python -m pip install -r requirements-experts-test.txt
python -m unittest discover -s tests -v
```

分割器接收本地 YOLO26n-seg ONNX 文件。支持静态 batch=1、FP32、NCHW 输入及 end-to-end 输出：检测 `[1,N,6+M]`（`xyxy, score, class_id, M 个 mask 系数`），原型 `[1,M,H,W]`。读取 ONNX 内的 `task=segment` 和 `names` 元数据；不猜类别表。原始 one-to-many 输出、动态输入、FP16 输入及不匹配模型会报错。

在**单独的模型导出环境**准备权重，例如：

```python
from ultralytics import YOLO

YOLO("yolo26n-seg.pt").export(
    format="onnx", imgsz=640, batch=1, dynamic=False, nms=False, device="cpu"
)
```

以上准备命令需要自行安装 Ultralytics，首次加载可能下载权重；expert 运行不依赖 Ultralytics/PyTorch。按实际导出文件检查输出合约，不将其他 YOLO 输出布局强行当成该格式。依据：[分割说明](https://docs.ultralytics.com/tasks/segment/)、[ONNX 导出](https://docs.ultralytics.com/integrations/onnx/)、[模型输出合约](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)、[ONNX 后端](https://onnxruntime.ai/docs/execution-providers/)。

## 调用与证据

```python
from experts import VideoFrame, MaskSeed, ObjectSpatialExpert, ShortWindowMotionExpert
from experts.geometry import pairwise_spatial_evidence
from experts.masks import decode_mask

# necessary_frames / continuous_frames 由调用方提供，均为 VideoFrame 列表。
# VideoFrame(frame_id=原始帧号, pts_seconds=原始PTS, image=原始分辨率BGR uint8图像)
spatial = ObjectSpatialExpert("/absolute/path/yolo26n-seg.onnx")
objects = spatial.infer(necessary_frames)
candidates = objects["frames"][0]["candidates"]
# 至少两个候选时，可计算同帧二维关系：
if len(candidates) >= 2:
    geometry = pairwise_spatial_evidence(candidates[0], candidates[1])
```

输出保留原帧号/PTS，像素坐标 `box_xyxy_px`，候选类别与检测置信度，原分辨率 mask（`encoding=rle_row_major`、`size=[H,W]`、零值起始游程 `counts`），以及模型 SHA256、完整类别表、运行库版本、阈值和 CPU provider。RLE 按行展开，**不是 COCO 的列优先 RLE**。没有检测时返回 `no_detections`，检测存在但 mask 为空时返回 `mask_status=empty`，不能推断对象不存在。

候选 ID 是帧内 ID，不是跨帧身份。类别 `apple` 不能自行确定它就是 instruction 中指定的苹果，也不支持自动为夹爪/插销生成 mask。调用方须先确认候选与任务实体的对应；这里不增加自动绑定规则。二维关系只表达图像平面几何，不作三维物理结论。

```python
# 调用方已确认 candidate 对应目标实体，且 mask 来自 continuous_frames 的首帧。
# 不得仅依据类别或列表位置自动做这个确认。
seed = MaskSeed(
    entity_key="task_apple", frame_id=candidate["frame_id"],
    mask=decode_mask(candidate["mask"]), source=candidate["candidate_id"],
)
motion = ShortWindowMotionExpert().track(continuous_frames, seed)
# 有额外可靠参考区域时：
# motion = ShortWindowMotionExpert().track(continuous_frames, seed, reference=reference_seed)
```

运动输入必须至少两帧，原帧号连续，PTS 有限且按时间排序；不可把稀疏图片冒充连续帧。初始 mask、实体键、原始帧号和来源由调用方明确给出；空 mask/无 mask/纹理点不足均记录为 `missing`。有可用参考 mask 时才计算相对运动；否则相对证据为 `reference_not_supplied`。

`target/reference.samples` 逐帧保存 PTS、有效点数、`step`（上一帧到当前帧）及 `from_seed`（首帧到当前帧），包含二维位移、模长及图像方向角（右=0°、下=90°；零位移没有方向）。位移是通过一致性检查的特征点位移中位数，不是物体中心轨迹；没有点的时刻为 `null`，不会写成零。`relative_motion` 仅在双方同时可用时对位移相减。

跟踪失败后不自动重选点或跨缺口拼接位移。`interruptions` 从失败的帧间区间起点记录到窗口末尾，含原始帧号与 PTS。`status=complete/partial/missing` 表示跟踪覆盖程度，不是任务完成状态。前后向检查不能保证物体身份或排除所有漂移。相机运动会贡献图像位移，相对参考也不能消除一般透视/深度效应。算法依据：[OpenCV 光流](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)。

所有返回记录可以 `json.dumps(result, allow_nan=False)`。expert 算法本身不写文件；主流程保存 YOLO 未绑定候选，随后在 `04-expert-execution.json` 中保存独立的绑定后几何/运动执行记录（含窗口/check/实体、绑定来源、原始帧/PTS、缺测/中断及原始输出）。第 5 步已按执行生成几何/运动摘要、覆盖与显式数值比较，并汇入逐 check 证据包；原始长数组只留执行文件引用，不重跑 expert。几何极值只描述已测帧，运动保留各自 seed，缺测/中断不拼接。可选阈值 JSON 的量/单位/范围见主 README，阈值比较不等于 check 答案。第 6 步现在读取摘要及全部相关原图/编号图作每个 check 一次 yes/no/unknown 判断；expert 缺测不强制 unknown。尺寸适配只作用于另存的展示图片，不改变原 mask/测量坐标；不重跑本包接口。纯文字终评尚未实现。
