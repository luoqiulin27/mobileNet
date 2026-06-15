# convert_sanpo_phase1.py 实现结果

## 1. 创建/修改的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `blind-assist-detection/scripts/convert/convert_sanpo_phase1.py` | **新建** | 主脚本 |

未修改任何其他文件。

## 2. 脚本实现要点

### 5 类映射

```python
SANPO_MASK_TO_PHASE1 = {
    12: 0,  # pedestrian -> person
    21: 1,  # vehicle -> vehicle
    24: 2,  # pole -> pole
    15: 3,  # stairs -> stairs
    20: 4,  # obstacle -> obstacle
    26: 4,  # bike rack -> obstacle
    14: 4,  # animal -> obstacle
    13: 4,  # rider -> obstacle
}
```

### bbox 提取逻辑

1. 读取 mask PNG，分离 R 通道（semantic）和 B 通道（instance）
2. 对每个目标类别，找到所有像素
3. 按 instance_id 分组，每个实例提取一个 bbox
4. 最小面积阈值：200 像素
5. 转换为 YOLO 归一化格式：`cx, cy, w, h` ∈ [0,1]
6. 过滤 `w <= 0.01` 或 `h <= 0.01` 的框（与 DetectionDataset 一致）
7. clip 到 [0,1] 范围

### 文件命名

```
原始: images/<session_id>/000000.png
输出: images/all/<session_id>_000000.png
      labels/all/<session_id>_000000.txt
```

## 3. SANPO 实际目录结构适配

| 实际路径 | 脚本中的处理 |
|----------|-------------|
| `images/<session_id>/NNNNNN.png` | 直接遍历，按 .png glob |
| `labels_segmentation_masks/<session_id>/NNNNNN.png` | 与图像同名，同目录结构 |
| `metadata/labelmap.json` | 未直接读取（映射硬编码在常量中） |
| 52 个空 session 目录 | 自动跳过，统计中记录 |
| mask 文件缺失的帧 | 跳过提取，写空标签，复制图像 |

## 4. 运行命令

```bash
# 默认运行（使用项目相对路径自动推导）
cd D:\project\mobileNet\blind-assist-detection
python scripts/convert/convert_sanpo_phase1.py

# 指定路径运行
python scripts/convert/convert_sanpo_phase1.py \
    --dataset D:/project/mobileNet/data/SANPO-Real-Labeled-Full \
    --output D:/project/mobileNet/blind-assist-detection/data/phase1_sanpo_5class \
    --min-area 200 \
    --copy-images

# 不复制图像（仅生成标签）
python scripts/convert/convert_sanpo_phase1.py --no-copy-images
```

## 5. 输出目录结构

```
data/phase1_sanpo_5class/
├── images/all/          ← 约 38,609 个 .png 文件
├── labels/all/          ← 与图像一一对应的 .txt 文件
├── configs/classes.txt  ← 5 类名
└── meta/conversion_report.json ← 转换统计报告
```

## 6. 已知风险与后续建议

| 风险 | 影响 | 建议 |
|------|------|------|
| 处理 38k 帧需要较长时间 | 每帧需要读取 mask + 图像 + 复制 | 可考虑用 `--no-copy-images` 先验证标签，再复制图像 |
| obstacle 类由 4 个原始类合并 | 样本量可能远多于其他类 | 训练时需关注类别平衡 |
| 部分帧 mask 缺失 | 会生成空标签文件 | verify 脚本应检查空标签比例 |
| 文件名中的 session_id 包含下划线 | split 脚本需从右侧最后一个 `_` 分割 | 已在接口文档中约定 |
| 未验证 mask 编码的 R/B 通道 | 如果实际编码不同会全部漏提取 | 建议先用 `--no-copy-images` 跑 1 个 session 验证 |

## 7. 后续任务

1. **split_phase1.py** — 按 session 划分 train/val/test
2. **verify_phase1.py** — 验证转换结果 + 生成 stats.json
3. **修改 ssd_default.yaml** — num_classes 从 9 改为 6
