# 第一阶段 SANPO 五类检测 baseline — 标签与数据设计方案

> 基于 `dataset_analysis_round1.md` 的结论，本文档为第一阶段提供可直接指导编码的数据方案。
> 范围：仅 SANPO 室外数据，仅 5 类检测，仅感知层。

---

## 1. 第一阶段 5 类标签体系

### 1.1 类别定义

| 统一 ID | 类别名 | SANPO 原始类名（labelmap ID） | 保留理由 |
|---------|--------|------------------------------|----------|
| 0 | **person** | pedestrian (12) | 安全关键目标，盲人场景最高优先级。行人是所有避障场景中最需要提前检测的目标 |
| 1 | **vehicle** | vehicle (21) | 安全关键目标，室外高频。包括汽车、卡车、自行车等移动载具 |
| 2 | **pole** | pole (24) | 室外常见固定障碍，盲人行走时极易碰撞的细长物体（路灯杆、电线杆、路标杆） |
| 3 | **stairs** | stairs (15) | 盲人场景核心障碍，跌落风险最高。台阶检测是辅助行走的核心需求 |
| 4 | **obstacle** | obstacle (20) + bike rack (26) + animal (14) + rider (13) | 兜底类，覆盖所有不适合作为独立类别的障碍物 |

### 1.2 被排除的类及理由

| SANPO 类名 | ID | 类型 | 排除理由 |
|------------|-----|------|----------|
| rider | 13 | panoptic | 样本量少，且骑车人的避让策略与 vehicle 相似。**合并入 obstacle** |
| animal | 14 | panoptic | 室外场景中动物出现频率低。**合并入 obstacle** |
| traffic sign | 22 | panoptic | 与避障无直接关系（标志牌通常在路边，不阻挡行进路线）。**第一阶段忽略** |
| traffic light | 23 | panoptic | 同上。**第一阶段忽略** |
| bike rack | 26 | panoptic | 静态小障碍物，出现频率低。**合并入 obstacle** |
| tree | 28 | panoptic | 树干可视为障碍，但树冠区域的 bbox 会非常大且不规则。**第一阶段忽略** |
| bus stop | 25 | panoptic | 出现频率低，且通常是行人活动区域而非障碍。**第一阶段忽略** |
| crosswalk | 5 | panoptic | 地面标线，非障碍物。**忽略** |
| opening-door | 10 | panoptic | 室内概念，在室外数据中极少出现。**忽略** |
| opening-gate | 11 | panoptic | 同上。**忽略** |
| road | 1 | semantic | 区域类，无实例，非障碍物。**忽略** |
| curb | 2 | semantic | 区域类，无实例。**忽略** |
| sidewalk | 3 | semantic | 区域类，无实例。**忽略** |
| building | 7 | semantic | 区域类，无实例（建筑物整体不是可避让的障碍物）。**忽略** |
| wall/fence | 8 | semantic | 区域类，无实例。**忽略** |
| sky/vegetation/terrain 等 | 27,29,30 | semantic | 背景类。**忽略** |

### 1.3 修正点（相对 dataset_analysis_round1.md）

**修正 1：rider 不独立成类，合并入 obstacle**

上一轮分析中 rider 被归入 obstacle 但未明确说明理由。本次明确：rider 在 SANPO 中是 panoptic 类，可以提取实例，但样本量远少于 pedestrian/vehicle。在 5 类方案中独立成类会导致：
- 类别严重不平衡（rider 样本可能不到 pedestrian 的 5%）
- 浪费一个类别位
- 对盲人避障而言，rider 的避让策略与 vehicle 相似（都是移动目标，需要让路）

因此 rider 合并入 obstacle 兜底类，而非被丢弃。

**修正 2：tree 不合并入 obstacle**

上一轮分析未明确讨论 tree。tree 是 panoptic 类（有实例），但树冠的 bbox 会覆盖图像中很大区域，对检测器学习"可避让障碍物"的概念产生干扰。tree 的避让策略也与小型障碍物不同。第一阶段排除。

---

## 2. SANPO 原始类别到 5 类的完整映射表

### 2.1 映射总表（31 类 → 5 类）

| SANPO ID | SANPO 类名 | 标注类型 | 映射目标 | 映射方式 | 理由 |
|----------|-----------|----------|----------|----------|------|
| 0 | unlabeled | semantic | **忽略** | 不提取 | 背景/未标注区域 |
| 1 | road | semantic | **忽略** | 不提取 | 道路表面，非障碍物 |
| 2 | curb | semantic | **忽略** | 不提取 | 路缘，区域类无实例 |
| 3 | sidewalk | semantic | **忽略** | 不提取 | 人行道，区域类无实例 |
| 4 | guard rail/road barrier | semantic | **忽略** | 不提取 | 区域类无实例，无法提取单个护栏 |
| 5 | crosswalk | panoptic | **忽略** | 不提取 | 地面标线，非障碍物 |
| 6 | paved trail | semantic | **忽略** | 不提取 | 区域类无实例 |
| 7 | building | semantic | **忽略** | 不提取 | 区域类无实例 |
| 8 | wall/fence | semantic | **忽略** | 不提取 | 区域类无实例 |
| 9 | hand rail | semantic | **忽略** | 不提取 | 区域类无实例 |
| 10 | opening-door | panoptic | **忽略** | 不提取 | 室外场景极少出现 |
| 11 | opening-gate | panoptic | **忽略** | 不提取 | 室外场景极少出现 |
| 12 | pedestrian | panoptic | **→ person (0)** | 直接映射 | 安全关键目标 |
| 13 | rider | panoptic | **→ obstacle (4)** | 合并 | 样本少，避让策略与 vehicle 相似 |
| 14 | animal | panoptic | **→ obstacle (4)** | 合并 | 样本少，室外低频 |
| 15 | stairs | panoptic | **→ stairs (3)** | 直接映射 | 盲人核心障碍 |
| 16 | water body | semantic | **忽略** | 不提取 | 区域类无实例 |
| 17 | other walkable surface | semantic | **忽略** | 不提取 | 区域类无实例 |
| 18 | inaccessible surface | semantic | **忽略** | 不提取 | 区域类无实例 |
| 19 | railway track | semantic | **忽略** | 不提取 | 区域类无实例 |
| 20 | obstacle | panoptic | **→ obstacle (4)** | 直接映射 | 泛化障碍物 |
| 21 | vehicle | panoptic | **→ vehicle (1)** | 直接映射 | 安全关键目标 |
| 22 | traffic sign | panoptic | **忽略** | 不提取 | 非阻挡行进路线的障碍 |
| 23 | traffic light | panoptic | **忽略** | 不提取 | 非阻挡行进路线的障碍 |
| 24 | pole | panoptic | **→ pole (2)** | 直接映射 | 室外高频固定障碍 |
| 25 | bus stop | panoptic | **忽略** | 不提取 | 低频，非典型障碍物 |
| 26 | bike rack | panoptic | **→ obstacle (4)** | 合并 | 低频小障碍，归入兜底类 |
| 27 | sky | semantic | **忽略** | 不提取 | 背景类 |
| 28 | tree | panoptic | **忽略** | 不提取 | bbox 过大且不规则，干扰训练 |
| 29 | vegetation | semantic | **忽略** | 不提取 | 区域类无实例 |
| 30 | terrain | semantic | **忽略** | 不提取 | 区域类无实例 |

### 2.2 映射规则总结

```
直接映射（4 类）：
  pedestrian(12)  → person(0)
  vehicle(21)     → vehicle(1)
  pole(24)        → pole(2)
  stairs(15)      → stairs(3)

合并到 obstacle（4 类）：
  obstacle(20)    → obstacle(4)  [主类]
  bike rack(26)   → obstacle(4)  [合并]
  animal(14)      → obstacle(4)  [合并]
  rider(13)       → obstacle(4)  [合并]

第一阶段忽略（23 类）：
  所有 semantic 类（17 类）→ 无实例，无法提取 bbox
  crosswalk, opening-door, opening-gate → 非障碍物或极低频
  traffic sign, traffic light → 非阻挡行进路线
  bus stop, tree → 低频或 bbox 不适用
```

### 2.3 从 SANPO 原始 mask ID 到 5 类 ID 的直接映射表

供转换脚本直接查表使用：

```python
SANPO_MASK_ID_TO_PHASE1 = {
    12: 0,  # pedestrian → person
    21: 1,  # vehicle → vehicle
    24: 2,  # pole → pole
    15: 3,  # stairs → stairs
    20: 4,  # obstacle → obstacle
    26: 4,  # bike rack → obstacle
    14: 4,  # animal → obstacle
    13: 4,  # rider → obstacle
}
# 不在此表中的 ID 全部跳过
```

### 2.4 从现有 10 类 YOLO 到 5 类的映射表

如果选择在已有的 10 类 YOLO 基础上转换（而非从原始 mask 重新提取）：

```python
YOLO_10_TO_PHASE1 = {
    0: 4,  # obstacle → obstacle
    1: 1,  # vehicle → vehicle
    2: 0,  # pedestrian → person
    3: 4,  # rider → obstacle (合并)
    4: 4,  # animal → obstacle (合并)
    5: 3,  # stairs → stairs
    6: -1, # traffic sign → 忽略
    7: -1, # traffic light → 忽略
    8: 2,  # pole → pole
    9: 4,  # bike rack → obstacle (合并)
}
# -1 表示丢弃该 bbox
```

---

## 3. bbox 提取规则

### 3.1 提取原理

SANPO mask 的编码方式：
- **R 通道**：语义类别 ID（对应 labelmap.json 中的值，0-30）
- **B 通道**：实例 ID（仅 panoptic 类有效，semantic 类的 B 通道为 0）

提取流程：
```
对每个 mask PNG：
  1. 读取图像，分离 R 通道（semantic）和 B 通道（instance）
  2. 对每个目标类别（12, 21, 24, 15, 20, 26, 14, 13）：
     a. 创建二值掩码：semantic == target_class_id
     b. 如果该类无像素，跳过
     c. 提取该区域内的所有唯一 instance_id
     d. 对每个 instance_id：
        - 创建实例掩码：semantic == target_class_id AND instance == instance_id
        - 计算像素面积
        - 如果面积 < 最小阈值，跳过
        - 找到所有像素的 (y, x) 坐标
        - 计算外接矩形：xmin, ymin, xmax, ymax
        - 转换为 YOLO 格式：cx, cy, w, h（归一化到 [0,1]）
     e. 将 SANPO class_id 映射到 5 类 ID
  3. 写入 YOLO .txt 标签文件
```

### 3.2 semantic id 和 instance id 的使用规则

| 场景 | semantic id | instance id | 处理方式 |
|------|-------------|-------------|----------|
| panoptic 类有实例 | 有效（如 12=pedestrian） | 有效（如 1, 2, 3...） | 每个 instance_id 提取一个 bbox |
| panoptic 类无实例 | 有效 | 0 | 整个连通域作为一个 bbox |
| semantic 类 | 有效 | 0 | **跳过**，不提取 bbox |

### 3.3 哪些类别必须要求 instance 级目标

| 类别 | 要求 instance 级？ | 理由 |
|------|-------------------|------|
| person (pedestrian) | **是** | 画面中可能有多个行人，必须区分每个个体 |
| vehicle | **是** | 画面中可能有多辆车，必须区分每辆 |
| pole | **是** | 画面中可能有多根杆子 |
| stairs | **否** | 楼梯通常是连续区域，一个实例即可 |
| obstacle | **是** | 可能有多个障碍物 |

**如果某类的 instance_id 全为 0（无实例区分）：**
- 将整个语义区域视为一个连通域
- 使用连通域分析（connected component labeling）自动分割
- 但这种情况在 panoptic 类中不应出现（panoptic 类天然有实例 ID）

### 3.4 最小面积阈值

**当前值：** 80 像素（在 2208×1242 图像上）

**是否保留：** 建议修改

**分析：**
- 2208×1242 = 2,742,336 总像素
- 80 像素 ≈ 图像面积的 0.0029%
- 对应 bbox 尺寸约 8×10 像素
- 这个阈值可能过滤掉远处的行人（在高分辨率图像中，远处行人可能只有 10-20 像素高）

**建议：**
- 将阈值提高到 **200 像素**，理由：
  - 80 像素的 bbox 在 resize 到 300×300 输入后会变成约 2.6×3.2 像素，几乎无法被检测器识别
  - 200 像素对应约 10×20 像素，resize 后约 5.4×8.7 像素，至少有 2-3 个 anchor 可以覆盖
  - 减少噪声标签（极小区域可能是标注噪声而非真实物体）
- 如果后续需要检测远处小目标，可以降低阈值，但第一阶段优先保证标注质量

### 3.5 破碎区域、遮挡区域、极小目标的处理

| 情况 | 处理方式 | 理由 |
|------|----------|------|
| **破碎区域**（同一 instance 被遮挡分成多块） | 按 instance_id 合并：同一 instance_id 的所有像素合并为一个 bbox | instance_id 已经标识了"这是同一个物体"，破碎是视觉遮挡的结果 |
| **遮挡区域**（物体被部分遮挡） | 保留被遮挡物体的完整 bbox（基于可见像素推断的外接矩形） | 检测任务的 ground truth 应该反映物体的实际位置，而非仅可见部分 |
| **极小目标**（面积 < 阈值） | 直接丢弃 | 极小目标无法被有效检测，保留会引入噪声 |
| **bbox 超出图像边界** | clip 到 [0, width] 和 [0, height] | SANPO mask 不应有超出边界的像素，但作为防御性编程 |
| **同一像素属于多个类别** | 不应发生（mask 是语义分割，每个像素只有一个类别） | 如果发生，取 R 通道值即可 |

### 3.6 特殊处理：obstacle 类的合并逻辑

obstacle(4) 类由 4 个 SANPO 原始类合并而来：
```
obstacle(20) + bike rack(26) + animal(14) + rider(13)
```

处理时：
1. 分别从 mask 中提取每个原始类的实例
2. 所有实例统一标记为 obstacle(4)
3. 不需要合并重叠的 bbox（这 4 个类在 mask 中不会重叠，因为每个像素只有一个语义 ID）

---

## 4. 第一阶段数据转换产物格式

### 4.1 数据组织结构

```
data/phase1_sanpo_5class/
├── images/
│   ├── train/
│   │   ├── -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000.png
│   │   ├── -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000059.png
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── configs/
│   └── classes.txt
└── meta/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── stats.json
```

### 4.2 文件格式规范

**classes.txt：**
```
person
vehicle
pole
stairs
obstacle
```
（5 行，每行一个类名，ID 按行号 0-indexed）

**标签 .txt 文件（YOLO 格式）：**
```
<class_id> <cx> <cy> <w> <h>
```
- class_id: 0-4（整数）
- cx, cy, w, h: 归一化到 [0, 1] 的浮点数（6 位小数）
- 每行一个 bbox
- 无 bbox 的图像对应空 .txt 文件

示例：
```
0 0.450000 0.600000 0.080000 0.200000
1 0.750000 0.500000 0.150000 0.300000
4 0.300000 0.700000 0.050000 0.100000
```

**train.txt / val.txt / test.txt：**
```
-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000
-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000059
```
（每行一个图像名，无扩展名，无路径前缀）

**stats.json：**
```json
{
  "total_images": 35000,
  "train": 24500,
  "val": 5250,
  "test": 5250,
  "class_counts": {
    "person": 12000,
    "vehicle": 18000,
    "pole": 8000,
    "stairs": 2000,
    "obstacle": 6000
  },
  "bbox_counts": {
    "person": 15000,
    "vehicle": 22000,
    "pole": 9500,
    "stairs": 2200,
    "obstacle": 8000
  },
  "images_with_no_bbox": 3000
}
```

### 4.3 是否继续沿用 YOLO 风格标注

**是。理由：**

| 因素 | 分析 |
|------|------|
| 格式通用性 | YOLO 格式是检测任务的事实标准，几乎所有框架都支持 |
| 与现有管线兼容 | 项目已有 YOLO → SSD 的数据加载逻辑 |
| 存储效率 | 纯文本格式，每行仅约 30 字节，远小于 XML/JSON |
| 调试便利性 | 可直接 cat 查看，无需解析 |

### 4.4 YOLO 标注与 SSD 训练的适配问题

**潜在问题：**

| 问题 | 详情 | 解决方案 |
|------|------|----------|
| **anchor 匹配** | 当前 SSD 配置的 anchor 是为 8 类设计的，5 类方案需要重新评估 anchor 尺寸 | 第一阶段先沿用现有 anchor 配置，观察检测效果后再调整 |
| **类别 ID 偏移** | YOLO 格式中 class_id 从 0 开始，SSD 训练时 background 通常是 class 0 | 在数据加载时将 YOLO class_id +1，0 留给 background |
| **输入尺寸** | YOLO 标注基于原始图像尺寸（2208×1242），SSD 输入为 300×300 | 数据加载时自动归一化，YOLO 格式已经是归一化的，无需额外处理 |
| **空标签文件** | 没有 bbox 的图像需要对应空 .txt 文件 | SSD 训练时需要处理空标签（跳过或作为 hard negative） |
| **类别不平衡** | obstacle 类由 4 个原始类合并，可能比其他类样本多 | 使用 focal loss 或类别权重调整 |

**特别注意：当前 SSD 模型的 num_classes 配置**

当前 `ssd_default.yaml` 中 `model.num_classes: 9`（8 类 + background）。改为 5 类后需要修改为 `model.num_classes: 6`（5 类 + background）。这会导致：
- 检测头输出维度变化
- 需要重新初始化检测头权重（backbone 可以保留预训练权重）

---

## 5. 第一阶段数据划分策略

### 5.1 为什么必须按 session 划分

| 理由 | 详情 |
|------|------|
| **防止数据泄漏** | 同一 session 的连续帧高度相似（相邻帧可能只差几毫秒），随机划分会导致 train/val/test 中出现几乎相同的图像 |
| **评估真实性** | 按 session 划分确保测试集包含训练中未见过的行走路线，更接近真实部署场景 |
| **避免过拟合** | 如果同一 session 的帧分散在 train 和 val 中，模型可能记住特定场景的特征而非学习通用检测能力 |

### 5.2 推荐比例

```
train : val : test = 70 : 15 : 15
```

按 session 级别划分（不是按帧级别）：
1. 列出所有有数据的 session（94 个）
2. 随机打乱 session 列表（固定种子）
3. 前 66 个 session → train
4. 接下来 14 个 session → val
5. 最后 14 个 session → test

### 5.3 如何避免数据泄漏

```
规则 1: 同一 session 的所有帧必须在同一个集合中
  → 不允许 session A 的帧 000000-000100 在 train，帧 000101-000200 在 val

规则 2: 划分基于 session ID，不基于帧号
  → session ID 从图像文件名中提取：文件名格式为 {session_id}_{frame_number}

规则 3: 使用固定随机种子
  → 确保每次划分结果可复现

规则 4: 验证无泄漏
  → 划分后检查：train 中的 session ID 集合与 val/test 无交集
```

### 5.4 空 session 的处理

当前 146 个 session 目录中有 52 个为空（无图像文件）。

**处理方式：**
- 在划分前直接过滤掉空 session
- 仅对 94 个有数据的 session 进行划分
- 在 stats.json 中记录：`"total_sessions": 146, "active_sessions": 94, "empty_sessions": 52`

### 5.5 类别不平衡时的划分注意事项

| 策略 | 是否采用 | 理由 |
|------|----------|------|
| **分层抽样**（确保每个集合中各类别比例相似） | ❌ 不采用 | 按 session 划分时无法保证（一个 session 可能没有某些类） |
| **按 session 划分后统计各类别分布** | ✅ 采用 | 划分后检查各类别在 train/val/test 中的分布，如果严重不平衡则重新随机 |
| **少数类过采样** | 留到训练阶段 | 不在数据划分阶段处理 |
| **确保 val/test 中每个类别至少有 N 个样本** | ✅ 采用 | 如果某个类别在 val 或 test 中样本过少（如 < 50），评估指标不可靠 |

**建议的验证步骤：**
1. 划分后统计每个集合中每个类别的 bbox 数量
2. 如果某个类别在 val 或 test 中 bbox 数 < 100，发出警告
3. 如果 stairs 类在某些 session 中完全不存在，确保它在 train/val/test 中都有分布

---

## 6. 第一批代码任务清单

### 任务 1：编写 `convert_sanpo_phase1.py`

**产出文件：** `blind-assist-detection/scripts/convert/convert_sanpo_phase1.py`

**功能：**
- 从原始 SANPO mask 提取 5 类 bbox
- 输入：`data/SANPO-Real-Labeled-Full/`（原始 mask + 图像）
- 输出：`data/phase1_sanpo_5class/`（YOLO 格式）

**关键实现：**
```python
# 核心映射表
SANPO_MASK_ID_TO_PHASE1 = {
    12: 0,  # pedestrian → person
    21: 1,  # vehicle → vehicle
    24: 2,  # pole → pole
    15: 3,  # stairs → stairs
    20: 4,  # obstacle → obstacle
    26: 4,  # bike rack → obstacle
    14: 4,  # animal → obstacle
    13: 4,  # rider → obstacle
}
MIN_AREA = 200  # 像素面积阈值

# 处理流程
for each session in active_sessions:
    for each frame in session:
        mask = load_mask(frame)
        semantic = mask[:, :, 0]  # R 通道
        instance = mask[:, :, 2]  # B 通道
        boxes = []
        for sanpo_id, phase1_id in SANPO_MASK_ID_TO_PHASE1.items():
            class_pixels = (semantic == sanpo_id)
            if not class_pixels.any():
                continue
            instance_ids = np.unique(instance[class_pixels])
            for iid in instance_ids:
                component = class_pixels & (instance == iid)
                area = component.sum()
                if area < MIN_AREA:
                    continue
                ys, xs = np.where(component)
                # 计算归一化 bbox
                ...
                boxes.append((phase1_id, cx, cy, w, h))
        write_yolo_label(frame, boxes)
        copy_image(frame)
```

**必须处理的边界情况：**
- 空 session（跳过）
- mask 文件不存在（跳过该帧，记录警告）
- instance_id 为 0 的 panoptic 类像素（整体作为一个 bbox）
- bbox 超出图像边界（clip）

### 任务 2：编写 `split_phase1.py`

**产出文件：** `blind-assist-detection/scripts/convert/split_phase1.py`

**功能：**
- 按 session 划分 train/val/test
- 输入：`data/phase1_sanpo_5class/`（转换后的数据）
- 输出：`data/phase1_sanpo_5class/meta/train.txt`, `val.txt`, `test.txt`

**关键实现：**
```python
SEED = 42
RATIO = (0.70, 0.15, 0.15)

# 从文件名提取 session ID
# 文件名格式: {session_id}_{frame_number:06d}.png
# session_id 可能包含下划线，所以从右侧分割
def extract_session_id(filename):
    parts = filename.rsplit('_', 1)
    return parts[0]

# 按 session 分组
sessions = group_by_session(all_filenames)
random.seed(SEED)
session_keys = list(sessions.keys())
random.shuffle(session_keys)

# 按比例分配
train_sessions = session_keys[:66]
val_sessions = session_keys[66:80]
test_sessions = session_keys[80:]

# 验证无泄漏
assert len(set(train_sessions) & set(val_sessions)) == 0
assert len(set(train_sessions) & set(test_sessions)) == 0
```

### 任务 3：编写 `verify_phase1.py`

**产出文件：** `blind-assist-detection/scripts/convert/verify_phase1.py`

**功能：**
- 验证转换结果的正确性
- 统计每个类别的 bbox 数量
- 检查 train/val/test 无数据泄漏
- 生成 stats.json

**检查项：**
```
1. 图像数量 == 标签文件数量
2. train/val/test 的 session 无交集
3. 每个类别在 train/val/test 中都有分布
4. 所有 bbox 的 cx, cy, w, h 都在 [0, 1] 范围内
5. 所有 class_id 都在 [0, 4] 范围内
6. 无空标签文件意外产生（允许空文件，但需要统计）
7. 输出 stats.json
```

### 任务 4：修改 `ssd_default.yaml`

**产出文件：** `blind-assist-detection/src/configs/ssd_default.yaml`（修改）

**修改内容：**
```yaml
model:
  backbone: "mobilenet_v2"
  pretrained: true
  input_size: [300, 300]
  num_classes: 6    # 从 9 改为 6（5 + background）

data:
  num_classes: 5    # 从 8 改为 5
  classes_file: "data/phase1_sanpo_5class/configs/classes.txt"
  train_list: "data/phase1_sanpo_5class/meta/train.txt"
  val_list: "data/phase1_sanpo_5class/meta/val.txt"
```

### 任务 5：修改数据加载代码

**产出文件：** `blind-assist-detection/src/datasets/` 下的数据加载模块（需确认具体文件名）

**修改内容：**
- 读取 phase1_sanpo_5class 目录下的图像和标签
- 适配 5 类标签（class_id 0-4）
- 处理空标签文件
- 图像 resize 到 300×300

### 任务 6：端到端验证

**产出文件：** 无（运行验证）

**步骤：**
1. 运行 convert_sanpo_phase1.py → 检查输出目录结构
2. 运行 split_phase1.py → 检查划分结果
3. 运行 verify_phase1.py → 检查 stats.json
4. 用小数据集（100 张）运行训练脚本 → 检查 loss 是否正常下降
5. 用全量数据训练 → 建立 baseline

### 任务顺序

```
任务 1 (convert_sanpo_phase1.py)  →  任务 2 (split_phase1.py)  →  任务 3 (verify_phase1.py)
                                                                        ↓
任务 4 (修改 ssd_default.yaml)  →  任务 5 (修改数据加载)  →  任务 6 (端到端验证)
```

任务 1-3 是数据准备，任务 4-5 是模型适配，任务 6 是验证。
任务 1-3 可以独立于任务 4-5 完成（数据准备不需要修改模型代码）。

---

## 7. 可直接进入编码的最小结论区

### 第一阶段类别集合

```
5 类：person(0), vehicle(1), pole(2), stairs(3), obstacle(4)
```

### 映射原则

```
从 SANPO 31 类中选择 8 个 panoptic 类映射到 5 类：
  pedestrian(12)  → person(0)      [直接映射]
  vehicle(21)     → vehicle(1)     [直接映射]
  pole(24)        → pole(2)        [直接映射]
  stairs(15)      → stairs(3)      [直接映射]
  obstacle(20)    → obstacle(4)    [直接映射]
  bike rack(26)   → obstacle(4)    [合并]
  animal(14)      → obstacle(4)    [合并]
  rider(13)       → obstacle(4)    [合并]

其余 23 类全部忽略（17 个 semantic 类无实例，6 个 panoptic 类不相关或低频）。
```

### 数据格式

```
YOLO 风格：<class_id> <cx> <cy> <w> <h>（归一化到 [0,1]）
目录结构：images/{train,val,test}/ + labels/{train,val,test}/
文件命名：{session_id}_{frame_number:06d}.png/.txt
最小面积阈值：200 像素
```

### 划分原则

```
按 session 划分（不按帧），防止数据泄漏。
比例：train:val:test = 70:15:15（session 级别）。
固定随机种子 42。
过滤空 session（52 个），仅使用 94 个有效 session。
```

### 第一批代码任务

```
1. 编写 convert_sanpo_phase1.py — 从原始 mask 提取 5 类 bbox
2. 编写 split_phase1.py — 按 session 划分 train/val/test
3. 编写 verify_phase1.py — 验证转换结果 + 生成 stats.json
4. 修改 ssd_default.yaml — num_classes 从 9 改为 6
5. 修改数据加载代码 — 适配新的 5 类数据目录
6. 端到端验证 — 小数据集训练测试 pipeline 通畅
```

---

当前任务已完成
