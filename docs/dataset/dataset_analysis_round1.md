# 数据集标注与检测任务适配分析（Round 1）

> 目标：为 MobileNet backbone 轻量化障碍物检测 baseline 提供数据层决策依据。
> 约束：仅做感知层（输入图像 → 输出类别+位置+置信度），不做提醒/路径规划/App/整图分类。

---

## 1. 两个数据集各自拥有什么标注

### 1.1 SANPO-Real-Labeled-Full（室外）

| 属性 | 值 |
|------|-----|
| 标注格式 | 像素级全景分割 mask（PNG） |
| mask 编码 | R 通道 = 语义类别 ID（0-30），B 通道 = 实例 ID（仅 panoptic 类） |
| 原始类别数 | 31 类 |
| 图像分辨率 | 2208 × 1242 |
| 图像格式 | PNG，8-bit RGB |
| 相机 | 胸戴左相机（camera_chest/left） |
| 总 session 数 | 146 个目录 |
| 有图像的 session | 94 个（52 个目录为空） |
| 总图像帧数 | 38,609 |
| 总 mask 帧数 | 38,622 |
| 帧命名 | `<session_id>/NNNNNN.png`（6 位零填充） |

**31 类完整列表：**

| ID | 类名 | 标注类型 | 能否提取独立实例 |
|----|------|----------|-----------------|
| 0 | unlabeled | semantic | ❌ |
| 1 | road | semantic | ❌ |
| 2 | curb | semantic | ❌ |
| 3 | sidewalk | semantic | ❌ |
| 4 | guard rail/road barrier | semantic | ❌ |
| 5 | crosswalk | panoptic | ✅ |
| 6 | paved trail | semantic | ❌ |
| 7 | building | semantic | ❌ |
| 8 | wall/fence | semantic | ❌ |
| 9 | hand rail | semantic | ❌ |
| 10 | opening-door | panoptic | ✅ |
| 11 | opening-gate | panoptic | ✅ |
| 12 | pedestrian | panoptic | ✅ |
| 13 | rider | panoptic | ✅ |
| 14 | animal | panoptic | ✅ |
| 15 | stairs | panoptic | ✅ |
| 16 | water body | semantic | ❌ |
| 17 | other walkable surface | semantic | ❌ |
| 18 | inaccessible surface | semantic | ❌ |
| 19 | railway track | semantic | ❌ |
| 20 | obstacle | panoptic | ✅ |
| 21 | vehicle | panoptic | ✅ |
| 22 | traffic sign | panoptic | ✅ |
| 23 | traffic light | panoptic | ✅ |
| 24 | pole | panoptic | ✅ |
| 25 | bus stop | panoptic | ✅ |
| 26 | bike rack | panoptic | ✅ |
| 27 | sky | semantic | ❌ |
| 28 | tree | panoptic | ✅ |
| 29 | vegetation | semantic | ❌ |
| 30 | terrain | semantic | ❌ |

**关键结论：** 15 个 panoptic 类拥有实例 ID，可以通过连通域分析提取独立物体的 bbox。17 个 semantic 类只有区域级标注，无法区分单个物体实例。

### 1.2 SUNRGBD（室内）

| 属性 | 值 |
|------|-----|
| 标注格式 | JSON 多边形（2D）+ JSON 3D 长方体 |
| 标注版本 | annotation2Dfinal（推荐）、annotation3Dfinal、annotation2D3D |
| 原始类别数 | 14,596 个原始名（极度脏，大量拼写变体） |
| 标注场景数 | 10,335 个（有 annotation2Dfinal） |
| 图像格式 | JPEG |
| 图像分辨率 | 因传感器而异（561×427 ~ 730×530） |
| 传感器 | kv1、kv2、realsense、xtion |
| 深度图 | 16-bit PNG（毫米），有 depth 和 depth_bfx 两版 |
| 场景标签 | 45 类（如 bedroom、office、classroom 等） |

**annotation2Dfinal/index.json 结构：**
```json
{
  "frames": [{
    "polygon": [
      {
        "x": [477, 591, 597, ...],     // 像素 x 坐标
        "y": [-37, -36, 167, ...],      // 像素 y 坐标
        "XYZ": [[0.64, -0.81, 1.68], ...], // 每顶点 3D 坐标（米）
        "object": 1                     // 索引到 objects 数组
      }
    ]
  }],
  "objects": [
    {"name": "shelf"},   // index 0
    {"name": "chair"},   // index 3
    ...
  ]
}
```

**annotation2D3D/index.json 额外特征：**
- 2D 矩形标注带 `"rect": 1` 标志，4 个顶点构成矩形
- 同时包含 3D bbox 数据

**类名脏数据示例：**
- `chair` vs `Chair` vs `CHAIR` vs `2chair` vs `2ndchair` vs `3rdchair`
- `bookshelf` vs `Bookshelf` vs `bookshelf `（尾部空格）
- `A/C` vs `AC` vs `A/CUnit` vs `AirConditioner`

### 1.3 标注适配性总结

| 维度 | SANPO | SUNRGBD |
|------|-------|---------|
| 原生格式 | 分割 mask | 多边形 JSON |
| 天然适合 detection？ | ❌ 不适合，需从 mask 提取 bbox | ⚠️ 部分适合，需从多边形提取外接矩形 |
| 已有转换脚本？ | ✅ 两个脚本已就绪 | ❌ 无任何转换脚本 |
| 类名标准化？ | ✅ 已有 31→10→8 映射链 | ❌ 14,596 个原始名，无映射表 |
| 数据划分？ | ✅ 按 session 70/15/15 | ❌ 无划分 |

---

## 2. 两个数据集分别如何转换为检测任务

### 2.1 SANPO：mask → bbox 转换

**转换原理：**

```
对每个 mask PNG：
  1. 读取 R 通道 → semantic class ID
  2. 读取 B 通道 → instance ID
  3. 对每个目标类别：
     a. 找出该类别所有像素
     b. 按 instance ID 分组（每个实例一个连通域）
     c. 对每个实例：计算外接矩形 → 归一化为 YOLO 格式
     d. 过滤面积 < 80 像素的碎片
```

**已有的两步管线：**

```
步骤 1: tools/sanpo_masks_to_yolo_obstacles.py
  输入: data/SANPO-Real-Labeled-Full (原始 mask)
  输出: data/SANPO-Real-YOLO-obstacles (10 类 YOLO bbox)
  选中的 10 类: obstacle, vehicle, pedestrian, rider, animal,
                stairs, traffic sign, traffic light, pole, bike rack

步骤 2: blind-assist-detection/scripts/convert/convert_sanpo_yolo.py
  输入: data/raw/SANPO-Real-YOLO-obstacles (10 类)
  输出: data/processed (8 类统一格式 + train/val/test split)
  映射:
    obstacle(0)    → obstacle(6)
    vehicle(1)     → vehicle(1)
    pedestrian(2)  → pedestrian(0)
    rider(3)       → rider(2)
    animal(4)       → animal(3)
    stairs(5)      → stairs(4)
    traffic_sign(6) → 忽略
    traffic_light(7) → 忽略
    pole(8)        → pole(5)
    bike_rack(9)   → obstacle(6)（合并）
```

**当前处理结果：**
- processed 目录中有 **1,401 张图像**（仅占原始 38,609 帧的 3.6%）
- train/val/test = 989 / 214 / 198

**已知问题：**

| 问题 | 影响 | 严重度 |
|------|------|--------|
| 处理量极低 | 仅 1,401/38,609 帧已转换，96.4% 数据未利用 | 🔴 高 |
| 52 个空 session | 实际可用 session 从 146 降至 94 | 🟡 中 |
| 最小面积阈值 80px | 可能过滤掉远处小目标（如远处行人） | 🟡 中 |
| 同一实例可能被遮挡断裂 | 一个行人被遮挡后可能被分成两个 bbox | 🟡 中 |
| traffic_sign/light 被丢弃 | 盲人辅助场景中这些可能是有用信号 | 🟢 低（可后期加回） |

### 2.2 SUNRGBD：多边形 → bbox 转换

**推荐的数据源：** `annotation2Dfinal/index.json`（最终版 2D 标注，而非 annotation2D3D）

**转换原理：**

```
对每个场景的 annotation2Dfinal/index.json：
  1. 读取 image/ 目录下的 JPEG 获取图像尺寸 (W, H)
  2. 遍历 frames[0].polygon[] 中的每个多边形：
     a. xmin = max(0, min(x[]))
     b. ymin = max(0, min(y[]))
     c. xmax = min(W, max(x[]))
     d. ymax = min(H, max(y[]))
     e. 转换为 YOLO 格式: cx=(xmin+xmax)/2/W, cy=(ymin+ymax)/2/H,
                           w=(xmax-xmin)/W, h=(ymax-ymin)/H
  3. 通过 polygon.object 索引查 objects[] 得到类名
  4. 类名 → 统一类别 ID 映射
  5. 写入 YOLO .txt 标签文件
```

**必须解决的问题：**

| 问题 | 详情 | 优先级 |
|------|------|--------|
| 类名标准化 | 14,596 个原始名需映射到统一类表。大量拼写变体、大小写不一致、数字前缀 | 🔴 P0 |
| 负坐标裁剪 | 多边形顶点可为负值（物体超出图像边界），必须 clip | 🔴 P0 |
| 图像路径解析 | 每个场景的图像路径格式不统一，需从 index.json 的 fileList 或硬编码路径推导 | 🔴 P0 |
| 图像尺寸不一致 | 不同传感器分辨率不同（561×427 ~ 730×530），每张图需单独读取尺寸 | 🟡 P1 |
| 类名映射表缺失 | 项目中无任何 SUNRGBD 类名映射文件 | 🔴 P0 |
| 转换脚本缺失 | 项目中无任何 SUNRGBD 转换脚本 | 🔴 P0 |
| 非矩形多边形 | annotation2Dfinal 的多边形不一定是矩形，外接矩形会有冗余区域 | 🟡 P1 |
| null 对象 | annotation3Dfinal 中有 null 条目（已删除的标注），需跳过 | 🟢 P2 |

### 2.3 转换问题优先级排序

```
必须最先解决（阻塞后续一切工作）：
  1. 编写 SUNRGBD → YOLO bbox 转换脚本
  2. 建立 SUNRGBD 类名 → 统一类的映射表
  3. 解析图像路径（找到每张 JPEG 的实际位置）

其次解决：
  4. 负坐标裁剪逻辑
  5. 图像尺寸自适应读取
  6. train/val/test 划分策略

最后解决：
  7. 非矩形多边形的 bbox 冗余问题
  8. 与 SANPO 数据的合并策略
```

---

## 3. 检测任务角度下的数据集角色分工

### 3.1 SANPO 的角色：室外主训练集

| 依据 | 详情 |
|------|------|
| 数据量 | 38,609 帧，足够训练轻量级检测器 |
| 标注质量 | Google Research 全景分割标注，像素级精度 |
| bbox 精度 | 从像素级 mask 提取的 bbox 边界精确 |
| 类别覆盖 | pedestrian, vehicle, rider, animal, stairs, pole, obstacle 全部直接可用 |
| 管线就绪 | mask→bbox→统一类 的完整管线已有 |
| 场景多样性 | 94 个独立行走路线，覆盖街道、人行道、公园 |
| **定位** | **主训练集，提供室外障碍物检测的全部训练数据** |

### 3.2 SUNRGBD 的角色：室内补充集

| 依据 | 详情 |
|------|------|
| 数据量 | 10,335 标注场景，量级适合作为补充 |
| 标注质量 | 多边形标注，外接矩形会有冗余但可用 |
| 类别覆盖 | chair, table, sofa, desk, bed 等室内物体需映射 |
| 管线状态 | 完全缺失，需从零开发 |
| 场景价值 | 提供室内环境泛化能力 |
| **定位** | **室内补充集，阶段 2 引入** |

### 3.3 是否建议一开始直接混合训练

**不建议。理由：**

| 因素 | 分析 |
|------|------|
| 域差异 | 室外（自然光、远景、动态）vs 室内（人工光、近景、静态）差异巨大 |
| 图像尺寸 | SANPO 2208×1242 vs SUNRGBD ~640×480，目标尺度完全不同 |
| 标注精度 | SANPO bbox 从像素 mask 提取（精确）vs SUNRGBD 从多边形外接矩形（有冗余） |
| 管线成熟度 | SANPO 管线就绪 vs SUNRGBD 管线不存在 |
| 类别语义 | SANPO 的 "obstacle" 是泛化室外障碍 vs SUNRGBD 有具体家具名 |
| 直接混合风险 | 模型在两类域间震荡，室内小目标被室外大目标淹没 |

### 3.4 推荐推进顺序

```
第一阶段：SANPO 单独训练
  → 验证 MobileNet backbone 在室外检测的可行性
  → 建立 baseline 性能指标

第二阶段：SUNRGBD 转换 + 单独验证
  → 开发 SUNRGBD → YOLO 转换脚本
  → 在 SUNRGBD 上单独验证室内检测能力

第三阶段：域感知混合 fine-tune
  → 冻结 backbone，仅训练检测头
  → 每个 batch 按比例混合室内外样本（如 7:3）
  → 或使用域自适应技术
```

---

## 4. 推荐的第一阶段检测类别设计

### 4.1 第一阶段最小可用 baseline

**数据源：** 仅 SANPO
**类别数：** 5 类（最小可行集）

| ID | 类别 | SANPO 原始类 | 选择理由 |
|----|------|-------------|----------|
| 0 | person | pedestrian(12) | 安全关键，盲人场景最高优先级 |
| 1 | vehicle | vehicle(21) | 安全关键，室外高频 |
| 2 | pole | pole(24) | 室外常见固定障碍，盲人易撞 |
| 3 | stairs | stairs(15) | 盲人场景核心障碍，跌落风险 |
| 4 | obstacle | obstacle(20) + bike rack(26) + animal(14) | 兜底类，覆盖长尾障碍物 |

**被排除的类及理由：**

| 被排除 | 理由 |
|--------|------|
| rider(13) | 样本量少，且避让策略与 vehicle 相似 |
| traffic sign(22) | 与避障无直接关系，可后期加 |
| traffic light(23) | 同上 |
| animal(14) | 已合并入 obstacle 兜底类 |
| bike rack(26) | 已合并入 obstacle 兜底类 |

### 4.2 为什么这样选

**为什么先只做室外？**
- SANPO 管线就绪，零开发成本即可开始训练
- SUNRGBD 转换脚本不存在，开发需要时间
- 先验证 backbone 可行性，再扩展数据源，是最低风险路径

**为什么先减少类别数？**
- 5 类 vs 8 类：减少类别可以更快收敛，更容易调参
- 消除类别不平衡的干扰（rider/animal 样本极少）
- 如果 5 类 baseline 性能好，扩展到 8 类是增量工作
- 如果 5 类 baseline 性能差，说明需要换 backbone 或检测头，而不是加数据

### 4.3 与现有模型配置的关系

当前模型配置（`ssd_default.yaml`）：
```yaml
model:
  backbone: "mobilenet_v2"
  input_size: [300, 300]
  num_classes: 9    # 8 classes + background
data:
  num_classes: 8
```

**如果采用 5 类方案，需要修改：**
- `model.num_classes` → 6（5 + background）
- `data.num_classes` → 5
- 检测头输出维度相应调整
- anchor 配置可能需要重新优化（当前 anchor 是为 8 类设计的）

### 4.4 阶段扩展路径

```
阶段 1 (当前): 5 类，仅 SANPO
  目标: 验证 MobileNetV2 + SSD-Lite 在室外检测的可行性
  期望: mAP@0.5 ≥ 0.3 即可认为 backbone 可行

阶段 2: 8 类，SANPO + SUNRGBD
  新增: chair, table, large_furniture
  变更: 重新训练检测头（类别数变化）

阶段 3: 精细化
  选项 A: 按场景切换模型（室外模型 / 室内模型）
  选项 B: 统一模型 + 域自适应
  选项 C: 级联方案（先检测 person/vehicle，再细分类）
```

---

## 5. 下一步代码工作的优先级建议

### 优先级排序

```
P0（阻塞项，必须最先完成）：
  ① 重新运行 sanpo_masks_to_yolo_obstacles.py
     → 选择 5 类子集: pedestrian, vehicle, pole, stairs, obstacle
     → 处理全部 38,609 帧（当前仅处理了 1,401 帧）
     → 输出到 data/SANPO-Real-YOLO-5class/

  ② 修改 convert_sanpo_yolo.py 的映射表
     → 从 10→8 映射改为 5 类直接映射
     → 重新划分 train/val/test
     → 输出到 data/processed_v2/

  ③ 验证转换结果
     → 统计每个类别的 bbox 数量
     → 随机抽样可视化 bbox 是否正确
     → 检查 train/val/test 是否有数据泄漏

P1（模型适配，紧接 P0）：
  ④ 修改 ssd_default.yaml
     → num_classes: 6 (5 + background)
     → 重新评估 anchor 配置是否需要调整

  ⑤ 修改 train_ssd.py 数据加载
     → 适配新的 5 类标签格式
     → 确认 loss 函数的 num_classes 参数正确

  ⑥ 运行第一轮训练
     → 先用小数据集（如 1,000 张）快速验证 pipeline 端到端通畅
     → 再用全量数据训练

P2（SUNRGBD 转换，可与 P1 并行）：
  ⑦ 编写 sunrgbd_to_yolo.py
     → 解析 annotation2Dfinal/index.json
     → 多边形 → 外接矩形 → 归一化
     → 负坐标裁剪
     → 类名映射（需先建映射表）

  ⑧ 建立 SUNRGBD 类名映射表
     → 分析 all_object_names.txt 中的高频类
     → 映射到统一 5 类或 8 类
     → 人工审核映射结果

  ⑨ SUNRGBD 数据划分
     → 按传感器/场景划分 train/val/test
     → 确保同一场景不跨集合

P3（优化，baseline 建立后）：
  ⑩ 数据增强策略
     → 随机翻转、缩放、颜色抖动
     → 针对 MobileNet 300×300 输入的 resize 策略

  ⑪ 类别平衡
     → 分析各类别样本比例
     → 使用 focal loss 或过采样少数类

  ⑫ 室内外混合训练
     → 开发 merge_datasets.py
     → 设计域感知采样策略
```

### 为什么这样安排

```
①②③ 优先的原因：
  → 当前仅处理了 1,401/38,609 帧（3.6%），数据严重不足
  → 补全数据是训练的前提，不解决则无法训练

④⑤ 紧接的原因：
  → 模型配置必须与数据类别数一致
  → 不修改则训练代码无法运行

⑦⑧⑨ 可并行的原因：
  → SUNRGBD 转换不阻塞 SANPO 训练
  → 可以在等 SANPO 训练的同时开发 SUNRGBD 转换脚本

⑩⑪ 延后的原因：
  → 先建立 baseline，再做优化
  → 没有 baseline 就无法评估优化效果
```

---

## 附录 A：当前项目关键文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `data/SANPO-Real-Labeled-Full/metadata/labelmap.json` | SANPO 31 类映射 | ✅ 存在 |
| `tools/sanpo_masks_to_yolo_obstacles.py` | mask → 10 类 YOLO | ✅ 存在 |
| `blind-assist-detection/scripts/convert/convert_sanpo_yolo.py` | 10 类 → 8 类统一 | ✅ 存在 |
| `blind-assist-detection/data/configs/classes.txt` | 统一类表 | ✅ 存在（8 类） |
| `blind-assist-detection/data/processed/` | 已处理数据 | ⚠️ 仅 1,401 帧 |
| `blind-assist-detection/src/configs/ssd_default.yaml` | 模型配置 | ✅ 存在（num_classes=9） |
| `blind-assist-detection/src/models/ssd_mobilenet.py` | 模型定义 | ✅ 存在 |
| `blind-assist-detection/scripts/train/train_ssd.py` | 训练脚本 | ✅ 存在 |
| SUNRGBD 转换脚本 | — | ❌ 不存在 |
| SUNRGBD 类名映射表 | — | ❌ 不存在 |

## 附录 B：数据规模对比

| 维度 | SANPO | SUNRGBD |
|------|-------|---------|
| 图像总数 | 38,609 | 10,335（标注子集） |
| 已处理数量 | 1,401 | 0 |
| 图像分辨率 | 2208×1242 | ~561×427 ~ 730×530 |
| 标注格式 | 分割 mask（PNG） | 多边形 JSON |
| bbox 提取 | ✅ 已有脚本 | ❌ 需新写 |
| 类名标准化 | ✅ 已完成 | ❌ 需新做 |
| 数据划分 | ✅ 已完成 | ❌ 需新做 |
| 可直接训练？ | ⚠️ 需重跑转换（当前量不足） | ❌ 完全不可 |

---

当前任务已完成
