"""
SSD MultiBox 损失函数

包含:
  - 分类损失: CrossEntropyLoss (正负样本)
  - 回归损失: SmoothL1Loss (仅正样本)
  - Hard Negative Mining: 按置信度排序取 top-k 负样本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.box_utils import (
    ciou_loss,
    diou_loss,
    cxcywh_to_xyxy,
    decode_boxes,
    encode_boxes,
    iou_loss,
    match_anchors,
)


class MultiBoxLoss(nn.Module):
    """
    SSD MultiBox Loss

    Args:
        neg_pos_ratio: 负样本与正样本的比例
        loc_weight: 回归损失权重
    """

    def __init__(self, neg_pos_ratio=3, loc_weight=1.0, box_loss="smooth_l1"):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.loc_weight = loc_weight
        self.box_loss = box_loss.lower()

    def forward(self, conf_pred, loc_pred, gt_boxes, gt_labels, anchors, anchors_xyxy=None):
        """
        计算损失

        Args:
            conf_pred: [B, num_anchors, num_classes] 分类 logits
            loc_pred: [B, num_anchors, 4] 回归偏移
            gt_boxes: list of [num_gt, 4] 每张图的 gt boxes (cx,cy,w,h 归一化)
            gt_labels: list of [num_gt] 每张图的 gt 类别
            anchors: [num_anchors, 4] anchor boxes

        Returns:
            total_loss: 标量
            loc_loss: 回归损失
            cls_loss: 分类损失
        """
        B = conf_pred.size(0)
        num_anchors = anchors.size(0)
        num_classes = conf_pred.size(2)

        # 存储每张图的匹配结果
        all_conf_targets = []
        all_loc_targets = []
        all_pos_mask = []

        for b in range(B):
            gt_b = gt_boxes[b].to(anchors.device)
            gl_b = gt_labels[b].to(anchors.device)

            # 匹配 anchor 与 gt
            matched_labels, matched_boxes, pos_mask = match_anchors(
                anchors, gt_b, gl_b, iou_threshold=0.5, anchors_xyxy=anchors_xyxy
            )

            # 编码回归目标
            if pos_mask.sum() > 0:
                loc_target = encode_boxes(matched_boxes[pos_mask], anchors[pos_mask])
            else:
                loc_target = torch.zeros(0, 4, device=conf_pred.device)

            all_conf_targets.append(matched_labels)
            all_loc_targets.append(loc_target)
            all_pos_mask.append(pos_mask)

        # ---- 分类损失 (Hard Negative Mining) ----
        conf_loss = torch.tensor(0.0, device=conf_pred.device)
        for b in range(B):
            # 所有 anchor 的分类预测
            conf_b = conf_pred[b]  # [num_anchors, num_classes]
            target_b = all_conf_targets[b]  # [num_anchors]
            pos_mask_b = all_pos_mask[b]    # [num_anchors]

            # 计算每个 anchor 的损失
            loss_per_anchor = F.cross_entropy(conf_b, target_b, reduction='none')  # [num_anchors]

            # 正样本损失
            pos_loss = loss_per_anchor[pos_mask_b].sum()

            # 负样本 Hard Negative Mining
            neg_mask = ~pos_mask_b
            neg_loss = loss_per_anchor[neg_mask]

            num_pos = pos_mask_b.sum().item()
            num_neg = min(neg_loss.size(0), num_pos * self.neg_pos_ratio)

            if num_neg > 0:
                # 取损失最大的负样本
                neg_loss_sorted, _ = neg_loss.sort(descending=True)
                neg_loss = neg_loss_sorted[:num_neg].sum()
            else:
                neg_loss = torch.tensor(0.0, device=conf_pred.device)

            conf_loss = conf_loss + (pos_loss + neg_loss) / max(num_pos, 1)

        conf_loss = conf_loss / B

        # ---- 回归损失 (仅正样本) ----
        loc_loss = torch.tensor(0.0, device=conf_pred.device)
        for b in range(B):
            pos_mask_b = all_pos_mask[b]
            if pos_mask_b.sum() == 0:
                continue

            loc_pred_b = loc_pred[b][pos_mask_b]  # [num_pos, 4]
            loc_target_b = all_loc_targets[b]      # [num_pos, 4]

            if self.box_loss == "ciou":
                decoded_pred = cxcywh_to_xyxy(decode_boxes(loc_pred_b, anchors[pos_mask_b]))
                decoded_target = cxcywh_to_xyxy(decode_boxes(loc_target_b, anchors[pos_mask_b]))
                loc_loss = loc_loss + ciou_loss(decoded_pred, decoded_target).sum()
            elif self.box_loss == "diou":
                decoded_pred = cxcywh_to_xyxy(decode_boxes(loc_pred_b, anchors[pos_mask_b]))
                decoded_target = cxcywh_to_xyxy(decode_boxes(loc_target_b, anchors[pos_mask_b]))
                loc_loss = loc_loss + diou_loss(decoded_pred, decoded_target).sum()
            elif self.box_loss == "iou":
                decoded_pred = cxcywh_to_xyxy(decode_boxes(loc_pred_b, anchors[pos_mask_b]))
                decoded_target = cxcywh_to_xyxy(decode_boxes(loc_target_b, anchors[pos_mask_b]))
                loc_loss = loc_loss + iou_loss(decoded_pred, decoded_target).sum()
            else:
                loc_loss = loc_loss + F.smooth_l1_loss(loc_pred_b, loc_target_b, reduction='sum')

        num_pos_total = sum(p.sum().item() for p in all_pos_mask)
        loc_loss = loc_loss / max(num_pos_total, 1)

        total_loss = self.loc_weight * loc_loss + conf_loss

        return total_loss, loc_loss, conf_loss
