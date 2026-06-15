"""
SSD Box 工具: Anchor 生成、编码解码、NMS
"""
import torch
import torch.nn as nn
import numpy as np


def generate_anchors(feature_maps, min_sizes, max_sizes, aspect_ratios, clip=True):
    """
    生成所有尺度的 anchor boxes

    Args:
        feature_maps: 特征图尺寸列表 [19, 10, 5, 3, 1]
        min_sizes: 每个尺度的最小 anchor 尺寸
        max_sizes: 每个尺度的最大 anchor 尺寸
        aspect_ratios: 每个尺度的长宽比
        clip: 是否裁剪到 [0, 1]

    Returns:
        anchors: [num_anchors, 4] (cx, cy, w, h) 归一化坐标
    """
    anchors = []
    for k, f in enumerate(feature_maps):
        for i in range(f):
            for j in range(f):
                # 中心坐标 (归一化)
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f

                # 最小尺寸 anchor
                s_min = min_sizes[k]
                anchors.append([cx, cy, s_min / 300, s_min / 300])

                # 最大尺寸 anchor
                s_max = max_sizes[k]
                anchors.append([cx, cy, s_max / 300, s_max / 300])

                # 不同长宽比的 anchor
                for ar in aspect_ratios[k]:
                    anchors.append([cx, cy, s_min / 300 * np.sqrt(ar), s_min / 300 / np.sqrt(ar)])
                    anchors.append([cx, cy, s_min / 300 / np.sqrt(ar), s_min / 300 * np.sqrt(ar)])

    anchors = np.array(anchors, dtype=np.float32)
    if clip:
        anchors = np.clip(anchors, 0, 1)
    return torch.from_numpy(anchors)


def encode_boxes(boxes, anchors, variance=[0.1, 0.2]):
    """
    将 ground truth boxes 编码为相对于 anchor 的偏移

    Args:
        boxes: [N, 4] (cx, cy, w, h) 归一化
        anchors: [M, 4] (cx, cy, w, h) 归一化
        variance: [中心方差, 尺寸方差]

    Returns:
        targets: [M, 4] 编码后的偏移
    """
    # 匹配的 gt box
    g_cx = (boxes[:, 0] - anchors[:, 0]) / (anchors[:, 2] * variance[0])
    g_cy = (boxes[:, 1] - anchors[:, 1]) / (anchors[:, 3] * variance[0])
    g_w = torch.log(boxes[:, 2] / anchors[:, 2]) / variance[1]
    g_h = torch.log(boxes[:, 3] / anchors[:, 3]) / variance[1]
    return torch.stack([g_cx, g_cy, g_w, g_h], dim=1)


def decode_boxes(loc, anchors, variance=[0.1, 0.2]):
    """
    将模型输出解码为 bounding box

    Args:
        loc: [N, 4] 模型预测的偏移
        anchors: [N, 4] anchor boxes
        variance: [中心方差, 尺寸方差]

    Returns:
        boxes: [N, 4] (cx, cy, w, h) 归一化
    """
    cx = loc[:, 0] * anchors[:, 2] * variance[0] + anchors[:, 0]
    cy = loc[:, 1] * anchors[:, 3] * variance[0] + anchors[:, 1]
    w = anchors[:, 2] * torch.exp(loc[:, 2] * variance[1])
    h = anchors[:, 3] * torch.exp(loc[:, 3] * variance[1])
    return torch.stack([cx, cy, w, h], dim=1)


def cxcywh_to_xyxy(boxes):
    """(cx, cy, w, h) → (x1, y1, x2, y2)"""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def xyxy_to_cxcywh(boxes):
    """(x1, y1, x2, y2) → (cx, cy, w, h)"""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return torch.stack([cx, cy, w, h], dim=1)


def jaccard(boxes_a, boxes_b):
    """
    计算 IoU (Intersection over Union)

    Args:
        boxes_a: [N, 4] (x1, y1, x2, y2)
        boxes_b: [M, 4] (x1, y1, x2, y2)

    Returns:
        iou: [N, M]
    """
    N = boxes_a.size(0)
    M = boxes_b.size(0)

    # 计算交集
    left = torch.max(boxes_a[:, 0].unsqueeze(1), boxes_b[:, 0].unsqueeze(0))
    top = torch.max(boxes_a[:, 1].unsqueeze(1), boxes_b[:, 1].unsqueeze(0))
    right = torch.min(boxes_a[:, 2].unsqueeze(1), boxes_b[:, 2].unsqueeze(0))
    bottom = torch.min(boxes_a[:, 3].unsqueeze(1), boxes_b[:, 3].unsqueeze(0))

    inter = torch.clamp(right - left, min=0) * torch.clamp(bottom - top, min=0)

    # 计算并集
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter

    return inter / (union + 1e-6)


def match_anchors(anchors, gt_boxes, gt_labels, iou_threshold=0.5, anchors_xyxy=None):
    """
    将 anchor 与 ground truth 匹配

    Args:
        anchors: [num_anchors, 4] (cx, cy, w, h)
        gt_boxes: [num_gt, 4] (cx, cy, w, h)
        gt_labels: [num_gt] 类别标签
        iou_threshold: 匹配阈值

    Returns:
        matched_labels: [num_anchors] 匹配的类别 (0=背景)
        matched_boxes: [num_anchors, 4] 匹配的 gt box
        pos_mask: [num_anchors] 正样本掩码
    """
    num_anchors = anchors.size(0)
    num_gt = gt_boxes.size(0)

    if num_gt == 0:
        device = anchors.device
        return (torch.zeros(num_anchors, dtype=torch.long, device=device),
                torch.zeros(num_anchors, 4, device=device),
                torch.zeros(num_anchors, dtype=torch.bool, device=device))

    # Reuse precomputed anchor corners during training to avoid repeating this
    # conversion once per image in every batch.
    if anchors_xyxy is None:
        anchors_xyxy = cxcywh_to_xyxy(anchors)
    gt_xyxy = cxcywh_to_xyxy(gt_boxes)
    iou = jaccard(anchors_xyxy, gt_xyxy)  # [num_anchors, num_gt]

    # 每个 anchor 匹配 IoU 最大的 gt
    best_gt_iou, best_gt_idx = iou.max(dim=1)  # [num_anchors]
    matched_labels = gt_labels[best_gt_idx]  # [num_anchors]
    matched_boxes = gt_boxes[best_gt_idx]    # [num_anchors, 4]

    # 背景: IoU < 阈值
    matched_labels[best_gt_iou < iou_threshold] = 0

    # 正样本掩码
    pos_mask = best_gt_iou >= iou_threshold

    return matched_labels, matched_boxes, pos_mask


def nms(boxes, scores, iou_threshold=0.45, top_k=200):
    """
    非极大值抑制

    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        scores: [N] 置信度
        iou_threshold: NMS 阈值
        top_k: 保留的最大数量

    Returns:
        keep: 保留的索引
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    # 按分数排序
    order = scores.argsort(descending=True)[:top_k]

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        # 计算当前框与剩余框的 IoU
        remaining = order[1:]
        iou = jaccard(boxes[i].unsqueeze(0), boxes[remaining]).squeeze(0)

        # 保留 IoU < 阈值的
        mask = iou < iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long)
