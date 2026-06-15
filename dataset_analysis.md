# 标注与任务适配分析报告

## 目标：MobileNet backbone 轻量化障碍物检测

---

## 1. 每个数据集当前有什么标注

### SANPO-Real-Labeled-Full（室外）

| 维度 | 内容 |
|------|------|
| **标注格式** | 像素级语义/全景分割 mask（PNG） |
| **编码方式** | R 通道 = 语义类别 ID，B 通道 = 实例 ID（仅 panoptic 类） |
| **类别数** | 31 类（17 个 semantic-only + 14 个带实例的 panoptic） |
| **图像数** | 38,609 帧（94 个有效 session，52 个 session 目录为空） |
| **图像格式** | RGB PNG，胸戴左相机 |
| **bbox 标注** | ❌ **原生无 bbox**，需要从 mask 提取 |

关键 panoptic 类（带实例分离，可提取独立物体）：
`pedestrian(12), rider(13), animal(14), stairs(15), obstacle(20), vehicle(21), traffic sign(22), traffic light(23), pole(24), bus stop(25), bike rack(26), tree(28), crosswalk(5), opening-door(10), opening-gate(11)`

### SUNRGBD（室内）

| 维度 | 内容 |
|------|------|
| **标注格式** | JSON 多边形（2D）+ 3D 长方体（3D） |
| **annotation2Dfinal** | 像素坐标多边形顶点 + 每顶点 3D 坐标 |
| **annotation3Dfinal** | 相机坐标系下的 3D bbox（米制） |
| **annotation2D3D** | 同时包含 2D 矩形 bbox + 3D bbox |
| **类别数** | 无全局类表，每场景局部定义，原始名 14,596 个（大量拼写变体） |
| **图像数** | ~10,335 个场景（JPEG，约 640×480） |
| **bbox 标注** | ⚠️ **有 2D bbox，但格式为多边形，需提取外接矩形** |

关键发现：`annotation2Dfinal/index.json` 中的多边形可以**直接计算** axis-aligned bbox：
```
xmin = min(x[]), ymin = min(y[]), xmax = max(x[]), ymax = max(y[])
```
但需要：裁剪负坐标到图像边界、归一化到 YOLO 格式。

---

## 2. 哪些标注可以直接支持 detection

### SANPO：不能直接支持

| 问题 | 详情 |
|------|------|
| 格式不匹配 | 原始是 pixel mask，检测需要 bbox |
| 已有解决方案 | `tools/sanpo_masks_to_yolo_obstacles.py` 已实现从 mask 提取 bbox |
| 提取原理 | 按 semantic class + instance id 定义连通域 → 计算外接矩形 → 归一化 |
| 最小面积阈值 | 80 像素（过滤噪声碎片） |
| **结论** | ✅ 有现成管线，可直接生成 YOLO bbox |

### SUNRGBD：部分直接支持

| 问题 | 详情 |
|------|------|
| 格式半匹配 | `annotation2D3D` 中已有矩形 bbox（x[left,right], y[top,bottom]） |
| 但需处理 | 负坐标裁剪、归一化、类名标准化、格式转换 |
| 更好的选择 | 用 `annotation2Dfinal` 多边形提取 bbox（标注更精细，是最终版） |
| **结论** | ⚠️ 需要写转换脚本（项目中目前**没有** SUNRGBD 转换脚本） |

---

## 3. 哪些标注需要转换成 bbox

### SANPO 需要转换的内容

| 转换路径 | 状态 |
|----------|------|
| segmentation mask → YOLO bbox | ✅ `sanpo_masks_to_yolo_obstacles.py` 已完成 |
| 10 类 YOLO → 8 类统一格式 | ✅ `convert_sanpo_yolo.py` 已完成 |
| **待做** | 无，管线已完备 |

### SUNRGBD 需要转换的内容

| 转换步骤 | 详情 | 状态 |
|----------|------|------|
| ① 解析 annotation2Dfinal JSON | 读取多边形顶点 | ❌ 无脚本 |
| ② 多边形 → 外接矩形 bbox | `min(x), min(y), max(x), max(y)` | ❌ 无脚本 |
| ③ 裁剪负坐标 | `xmin = max(0, xmin)`, `ymin = max(0, ymin)` | ❌ 无脚本 |
| ④ 归一化到 YOLO 格式 | `cx, cy, w, h` 除以图像宽高 | ❌ 无脚本 |
| ⑤ 类名标准化 | 14,596 个原始名 → 统一类表 | ❌ 无映射表 |
| ⑥ 生成 train/val/test split | 按场景划分 | ❌ 无脚本 |
| **整体** | **需要从零编写完整转换管线** | ❌ |

---

## 4. 每个数据集更适合做什么角色

### SANPO → 室外障碍物检测主训练集

| 理由 | 详情 |
|------|------|
| 标注覆盖 | 38,609 帧，量级足够支撑 MobileNet 检测器训练 |
| 类别匹配度 | pedestrian, vehicle, rider, animal, stairs, pole, obstacle 全部直接覆盖 |
| 场景多样性 | 146 个独立行走路线，覆盖街道、人行道、公园等 |
| 管线成熟度 | mask→bbox→统一类 的管线已全部就绪 |
| 标注质量 | Google Research 出品，全景分割标注质量高 |
| **角色** | **主训练集，提供 ~38k 个带 bbox 标注的室外样本** |

### SUNRGBD → 室内障碍物检测补充集

| 理由 | 详情 |
|------|------|
| 标注覆盖 | ~10,335 场景，量级适合作为补充 |
| 类别匹配度 | chair, table, sofa, desk, bed 等室内家具/物体需要映射到统一类表 |
| 标注格式 | 多边形标注需要转换，但转换逻辑简单 |
| 场景价值 | 提供 MobileNet 在室内环境的泛化能力 |
| **角色** | **室内补充集，提供 ~10k 个室内场景的 bbox 标注** |

---

## 5. 室内外是否应该直接混合训练

### 结论：不应该直接混合，应分阶段策略

| 因素 | 分析 |
|------|------|
| **域差异** | 室外（自然光照、远景、动态物体）vs 室内（人工光照、近景、静态家具）差异极大 |
| **类别语义冲突** | SANPO 的 "obstacle" 是泛化的室外障碍物，SUNRGBD 的物体类别是具体的家具名 |
| **图像尺寸差异** | SANPO 高分辨率 PNG vs SUNRGBD ~640×480 JPEG |
| **标注精度差异** | SANPO 从像素级 mask 提取的 bbox 精度高，SUNRGBD 多边形外接矩形会有冗余区域 |
| **直接混合风险** | 模型在两类场景间震荡，室内小物体被室外大物体淹没 |

### 推荐策略

```
阶段 1: 先用 SANPO 单独训练室外检测器（验证 MobileNet backbone 可行性）
阶段 2: 用 SUNRGBD 单独微调室内检测能力
阶段 3: 冻结 backbone，混合 fine-tune 检测头（domain-adaptive 混合）
```

混合时应使用**域感知采样**：每个 batch 内按比例分配室内外样本（如 7:3），而非完全随机混合。

---

## 6. 建议的统一标签空间

### 分析：当前 8 类方案的问题

当前 `classes.txt` 的 8 类方案：

```
0: pedestrian    ← SANPO ✅ / SUNRGBD ❌（室内几乎没有）
1: vehicle       ← SANPO ✅ / SUNRGBD ❌
2: rider         ← SANPO ✅ / SUNRGBD ❌
3: animal        ← SANPO ✅ / SUNRGBD ❌
4: stairs        ← SANPO ✅ / SUNRGBD ⚠️（有但少）
5: pole          ← SANPO ✅ / SUNRGBD ❌
6: obstacle      ← SANPO ✅ / SUNRGBD ⚠️（太泛化）
7: furniture     ← SANPO ❌ / SUNRGBD ✅（但需映射）
```

**问题**：室内场景中 chair/table/sofa/desk 是最常见的障碍物，但被笼统归为 "furniture" 一类，丢失了细粒度信息。同时 pedestrian/vehicle/rider/animal 在室内几乎不存在，浪费了 4 个类别容量。

### 推荐的统一标签空间（8 类，适配 MobileNet 轻量化）

| ID | 类别名 | SANPO 来源 | SUNRGBD 来源 | 说明 |
|----|--------|------------|-------------|------|
| 0 | **person** | pedestrian(12) | person | 室内外共通，最高优先级 |
| 1 | **vehicle** | vehicle(21) | （无） | 室外专用 |
| 2 | **chair** | （无） | chair, sofa_chair, stool | 室内最高频障碍物 |
| 3 | **table** | （无） | desk, table, sofa_table, night_stand | 室内第二大类 |
| 4 | **large_furniture** | （无） | sofa, bed, bookshelf, cabinet, wardrobe, refrigerator | 室内大型障碍物 |
| 5 | **pole** | pole(24) | （无） | 室外杆状物 |
| 6 | **stairs** | stairs(15) | stairs（少量） | 室内外共通 |
| 7 | **obstacle** | obstacle(20), bike rack(26), animal(14), rider(13) | box, bag, objects 等兜底类 | 泛化兜底类 |

### 关键设计决策

**为什么拆 furniture 而不保留 rider/animal？**
- rider/animal 在实际盲人辅助场景中出现频率远低于 person/vehicle
- 室内 chair/table 是真正的高频障碍物，值得独立类别
- MobileNet 8 类输出层已经很轻量，类别数不应再加

**为什么 rider 归入 obstacle 而不是独立？**
- rider(骑车人)在 SANPO 中样本量本身就不多
- 对于避障任务，rider 和 vehicle 的避让策略相似
- 合并后节省一个类别位给室内高频类

---

## 7. 第一阶段最合理的检测类别设计

### 阶段 1 目标：验证 MobileNet backbone 可行性

| 决策 | 选择 | 理由 |
|------|------|------|
| 数据源 | **仅 SANPO** | 管线就绪，无需额外开发 |
| 类别数 | **5 类**（最小可行集） | 快速验证，避免类别不平衡 |
| backbone | MobileNetV2/V3 | 轻量化主干 |
| 检测头 | SSD-Lite 或 YOLOv8n | 适配 MobileNet 的轻量检测头 |
| 输入尺寸 | 320×320 或 416×416 | 移动端友好 |

### 阶段 1 的 5 类设计

| ID | 类别 | SANPO 来源 | 选择理由 |
|----|------|------------|----------|
| 0 | **person** | pedestrian(12) | 安全关键，最高优先 |
| 1 | **vehicle** | vehicle(21) | 安全关键，室外高频 |
| 2 | **pole** | pole(24) | 室外常见固定障碍 |
| 3 | **stairs** | stairs(15) | 盲人场景核心障碍 |
| 4 | **obstacle** | obstacle(20) + bike rack(26) + animal(14) | 兜底类，覆盖长尾 |

**为什么是这 5 类而不是更多？**
- 去掉了 rider（样本少，可后期加）
- 去掉了 animal（已并入 obstacle 兜底）
- 5 类足够验证 backbone 特征提取能力
- 类别越少，初期调参越容易收敛

### 阶段 2 扩展路径

```
阶段 1 (5类, SANPO only)
  ↓ 验证通过
阶段 2 (8类, SANPO + SUNRGBD 混合)
  ├── 加入: chair, table, large_furniture
  ├── 重新平衡采样策略
  └── 冻结 backbone, 仅 fine-tune 检测头
```

---

## 8. 数据预处理方案决策矩阵

| 预处理任务 | SANPO | SUNRGBD | 优先级 |
|-----------|-------|---------|--------|
| mask → YOLO bbox | ✅ 已有脚本 | 不适用 | — |
| JSON polygon → YOLO bbox | 不适用 | ❌ 需新写 | P1（阶段 2） |
| 类名标准化映射 | ✅ 已有 10→8 映射 | ❌ 需新写（14596→统一类） | P1（阶段 2） |
| train/val/test split | ✅ 已有（按 session 70/15/15） | ❌ 需新写 | P1（阶段 2） |
| 图像尺寸统一 | 需确认 SANPO 原始分辨率 | ~640×480 已固定 | P2 |
| 数据增强 | 需设计（flip, scale, color） | 同左 | P2（训练时） |

### 阶段 1 需要做的

```
1. 运行 sanpo_masks_to_yolo_obstacles.py（选择 5 类子集）
2. 运行 convert_sanpo_yolo.py（映射到统一 5 类）
3. 确认 train/val/test 划分无数据泄漏
4. 验证 bbox 质量（随机抽样可视化）
```

### 阶段 2 需要新写的

```
1. sunrgbd_polygon_to_yolo.py
   - 遍历所有 annotation2Dfinal/index.json
   - 多边形 → 外接矩形 → 裁剪 → 归一化
   - 14596 原始名 → 8 统一类映射表
2. sunrgbd_split.py
   - 按传感器/场景划分 train/val/test
3. merge_datasets.py
   - 合并 SANPO + SUNRGBD 为统一格式
   - 域感知采样配置
```

---

## 9. 风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SUNRGBD 类名脏数据 | 映射错误导致标签噪声 | 建立人工审核的映射表，对高频类逐一确认 |
| SANPO 52 个空 session | 实际可用数据比预期少 | 以 94 个有效 session 为准（~38k 帧足够） |
| SUNRGBD 多边形负坐标 | bbox 超出图像范围 | 转换时 clip 到 [0, width/height] |
| 室内外 bbox 尺度差异 | 室外物体 bbox 小，室内物体 bbox 大 | 训练时使用多尺度 anchor 或 FPN |
| 类别不平衡 | person/vehicle 样本远多于 stairs/animal | 使用 focal loss 或过采样少数类 |

---

**这份分析可以直接指导下一步：先跑通 SANPO 的 5 类转换管线做阶段 1 验证，再按需开发 SUNRGBD 转换脚本进入阶段 2。**
