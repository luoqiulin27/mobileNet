"""
目标检测数据集 - 加载 YOLO 格式标注

数据格式:
  images/xxx.jpg
  labels/xxx.txt  每行: class_id cx cy w h (归一化)
"""
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DetectionDataset(Dataset):
    """
    YOLO 格式检测数据集

    Args:
        image_dir: 图像目录
        label_dir: 标签目录
        image_list: 图像文件名列表 (不含扩展名)
        input_size: 输入尺寸
        classes: 类别名称列表
        transform: 数据增强
        augment: 是否增强
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        image_list: List[str],
        input_size: int = 300,
        classes: List[str] = None,
        augment: bool = False,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_list = image_list
        self.input_size = input_size
        self.classes = classes or []
        self.num_classes = len(classes) if classes else 0
        self.augment = augment

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]

        # 加载图像
        img_path = os.path.join(self.image_dir, name + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, name + ".jpg")

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # 加载标签
        label_path = os.path.join(self.label_dir, name + ".txt")
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        # 过滤无效框
                        if w > 0.01 and h > 0.01:
                            boxes.append([cx, cy, w, h])
                            labels.append(cls_id + 1)  # 0 留给背景

        # 数据增强
        if self.augment:
            image, boxes, labels = self._augment(image, boxes, labels)

        # Resize
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # 转 tensor
        img_tensor = transforms.ToTensor()(image)
        img_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img_tensor)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)

        return {
            "image": img_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "name": name,
        }

    def _augment(self, image, boxes, labels):
        """简单数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[1 - cx, cy, w, h] for cx, cy, w, h in boxes]

        # 颜色抖动
        if random.random() > 0.5:
            img_np = np.array(image).astype(np.float32) / 255.0
            # HSV 增强
            delta = random.uniform(-0.1, 0.1)
            img_np = np.clip(img_np + delta, 0, 1)
            image = Image.fromarray((img_np * 255).astype(np.uint8))

        return image, boxes, labels


def collate_fn(batch):
    """自定义 collate，处理不同数量的 boxes"""
    images = torch.stack([item["image"] for item in batch])
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    names = [item["name"] for item in batch]
    return {"images": images, "boxes": boxes, "labels": labels, "names": names}


def load_image_list(list_file: str) -> List[str]:
    """加载图像列表文件"""
    names = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                # 去掉扩展名
                name = os.path.splitext(os.path.basename(line))[0]
                names.append(name)
    return names
