# Blind Assist Detection - 视障人士障碍物感知系统

基于 MobileNetV2 + SSD-Lite 的轻量化障碍物检测系统，用于盲人辅助导航。

## 功能特性

- 🚶 **行人检测** - 安全关键目标
- 🚗 **车辆检测** - 室外高频障碍
- 🪧 **杆状物检测** - 路灯杆、电线杆等
- 🪜 **台阶检测** - 盲人场景核心障碍
- 📦 **通用障碍物** - 兜底类，覆盖长尾

## 环境要求

- Windows 10/11
- [Anaconda](https://www.anaconda.com/download) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU（推荐，支持 CUDA 12.4）

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/你的用户名/blind-assist-detection.git
cd blind-assist-detection
```

### 2. 安装环境

双击运行 `setup.bat`，或手动执行：

```bash
conda env create -f environment.yml
conda activate blind-assist
```

### 3. 准备数据

将数据集放在以下目录：

```
blind-assist-detection/
└── data/
    └── phase1_sanpo_5class/
        ├── images/
        │   └── all/          # 所有图像 (.png)
        ├── labels/
        │   └── all/          # 所有标签 (.txt)
        ├── configs/
        │   └── classes.txt   # 类别配置
        └── meta/
            ├── train.txt     # 训练集列表
            ├── val.txt       # 验证集列表
            └── test.txt      # 测试集列表
```

**数据来源：**
- 室外数据：SANPO-Real-Labeled-Full
- 室内数据：SUNRGBD（Phase 2）

### 4. 启动项目

双击运行 `start.bat`，或手动执行：

```bash
conda activate blind-assist
cd blind-assist-detection
python web_demo/app_detect.py
```

访问 http://localhost:5000 打开 Web Demo。

## 项目结构

```
blind-assist-detection/
├── scripts/
│   ├── convert/          # 数据转换脚本
│   │   ├── convert_sanpo_phase1.py   # SANPO 5类转换
│   │   ├── split_phase1.py           # 数据划分
│   │   └── verify_phase1.py          # 数据验证
│   ├── train/
│   │   └── train_ssd.py              # 训练脚本
│   └── eval/
│       └── eval_map.py               # 评估脚本
├── src/
│   ├── configs/
│   │   └── ssd_default.yaml          # 模型配置
│   ├── models/
│   │   ├── ssd_mobilenet.py          # SSD-MobileNet 模型
│   │   └── box_utils.py              # Anchor 工具
│   ├── datasets/
│   │   └── detection_dataset.py      # 数据加载
│   └── losses/
│       └── multibox_loss.py          # 损失函数
├── web_demo/
│   ├── app_detect.py                 # Web 演示
│   └── templates/                    # 前端模板
├── data/                             # 数据集（不上传）
├── outputs/                          # 训练输出
└── exports/                          # 模型导出
```

## 训练模型

### 数据准备

```bash
# 1. 转换 SANPO 数据（5类）
python scripts/convert/convert_sanpo_phase1.py

# 2. 按 session 划分 train/val/test
python scripts/convert/split_phase1.py

# 3. 验证数据完整性
python scripts/convert/verify_phase1.py
```

### 开始训练

```bash
# 短程验证（10 epochs）
python scripts/train/train_ssd.py --gpu 0 --epochs 10 --num-workers 4

# 全量训练（100 epochs）
python scripts/train/train_ssd.py --gpu 0 --epochs 100 --num-workers 4
```

### 评估模型

```bash
# 在验证集上评估
python scripts/eval/eval_map.py --checkpoint outputs/checkpoints/best.pth --split val --gpu 0

# 在测试集上评估
python scripts/eval/eval_map.py --checkpoint outputs/checkpoints/best.pth --split test --gpu 0
```

## Baseline 性能（10 epochs）

| 类别 | AP | 训练样本数 |
|------|-----|-----------|
| vehicle | 0.3545 | 75,608 |
| person | 0.1803 | 54,614 |
| obstacle | 0.1768 | 104,070 |
| pole | 0.0909 | 27,373 |
| stairs | 0.0606 | 2,183 |
| **mAP@0.5** | **0.1726** | |

## 配置说明

### 模型配置 (`src/configs/ssd_default.yaml`)

```yaml
model:
  backbone: "mobilenet_v2"
  pretrained: true
  input_size: [300, 300]
  num_classes: 6  # 5类 + 背景

data:
  num_classes: 5
  classes_file: "data/phase1_sanpo_5class/configs/classes.txt"
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--gpu` | 0 | GPU 编号（-1 为 CPU） |
| `--num-workers` | 4 | 数据加载线程数 |

## 常见问题

### Q: CUDA 内存不足？

```bash
# 减小 batch_size
python scripts/train/train_ssd.py --batch-size 8
```

### Q: 训练很慢？

```bash
# 增加 num_workers
python scripts/train/train_ssd.py --num-workers 8

# 或使用混合精度（自动启用）
```

### Q: 如何使用 CPU 训练？

```bash
python scripts/train/train_ssd.py --gpu -1
```

## 后续计划

- [ ] Phase 2: 集成 SUNRGBD 室内数据
- [ ] 模型导出：ONNX / TFLite
- [ ] Android 部署
- [ ] 语音提醒功能

## 许可证

MIT License

## 致谢

- [SANPO Dataset](https://google.github.io/sanpo/) - 室外场景数据集
- [SUNRGBD Dataset](https://rgbd.cs.princeton.edu/) - 室内场景数据集
- [MobileNetV2](https://arxiv.org/abs/1801.04381) - 轻量化骨干网络
- [SSD](https://arxiv.org/abs/1512.02325) - 目标检测框架
