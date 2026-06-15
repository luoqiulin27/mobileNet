"""
MobileNetV2-SSD300 轻量检测模型

结构:
  MobileNetV2 Backbone → 多尺度特征 → SSD 检测头 → NMS
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from .box_utils import generate_anchors, decode_boxes, cxcywh_to_xyxy, nms


class SSDMobileNetV2(nn.Module):
    """
    基于 MobileNetV2 的 SSD 检测模型

    Args:
        num_classes: 类别数 (含背景)
        pretrained: 是否使用 ImageNet 预训练
        input_size: 输入尺寸
    """

    def __init__(self, num_classes=9, pretrained=True, input_size=300):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # ---- Backbone: MobileNetV2 ----
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features  # 到 layer13 → 19×19×960

        # MobileNetV2 features 层通道数:
        # layer0-10: → 64 channels
        # layer11-13: → 96 channels (stride 8, 38×38)
        # layer14-16: → 160 channels (stride 16, 19×19)
        # layer17: → 320 channels
        # layer18: Conv1x1 → 1280 channels (stride 32, 10×10)

        # 分段提取多尺度特征
        self.backbone_stage1 = nn.Sequential(*list(backbone.features)[:14])  # → 19×19×96
        self.backbone_stage2 = nn.Sequential(*list(backbone.features)[14:])  # → 10×10×1280

        # ---- 额外特征层 (逐步下采样) ----
        self.extra_layers = nn.ModuleList([
            # 10×10×1280 → 5×5×512
            self._make_extra_layer(1280, 512, stride=2),
            # 5×5×512 → 3×3×256
            self._make_extra_layer(512, 256, stride=2),
            # 3×3×256 → 1×1×128
            self._make_extra_layer(256, 128, stride=2),
        ])

        # 特征图通道数: 19×19→96, 10×10→1280, 5×5→512, 3×3→256, 1×1→128
        self.feat_channels = [96, 1280, 512, 256, 128]

        # ---- 检测头 ----
        # 每个尺度: 分类头 + 回归头
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        # 每个尺度的 anchor 数量
        self.num_anchors = [4, 6, 6, 6, 4]  # 与 anchor 配置对应

        for ch, na in zip(self.feat_channels, self.num_anchors):
            self.cls_heads.append(nn.Conv2d(ch, na * num_classes, 3, padding=1))
            self.reg_heads.append(nn.Conv2d(ch, na * 4, 3, padding=1))

        # ---- Anchor ----
        self.anchors = None  # 延迟生成
        self.anchor_feature_maps = [19, 10, 5, 3, 2]
        self.anchor_min_sizes = [30, 60, 111, 162, 213]
        self.anchor_max_sizes = [60, 111, 162, 213, 264]
        self.anchor_aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2]]

        # 初始化检测头
        self._init_heads()

    def _make_extra_layer(self, in_ch, out_ch, stride=2):
        """创建额外卷积层"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
        )

    def _init_heads(self):
        """初始化检测头权重"""
        for m in self.cls_heads:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.reg_heads:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def configure_anchors(self, feature_maps, min_sizes, max_sizes, aspect_ratios):
        """设置推理/解码时使用的 anchor 配置。"""
        self.anchor_feature_maps = feature_maps
        self.anchor_min_sizes = min_sizes
        self.anchor_max_sizes = max_sizes
        self.anchor_aspect_ratios = aspect_ratios
        self.anchors = None

    def get_anchors(self, device, feature_map_sizes=None):
        """获取 anchor boxes (懒初始化)"""
        if self.anchors is None or feature_map_sizes is not None:
            if feature_map_sizes is None:
                feature_map_sizes = self.anchor_feature_maps
            self.anchors = generate_anchors(
                feature_map_sizes,
                self.anchor_min_sizes,
                self.anchor_max_sizes,
                self.anchor_aspect_ratios,
            )
        return self.anchors.to(device)

    def freeze_backbone(self):
        """冻结 backbone"""
        for param in self.backbone_stage1.parameters():
            param.requires_grad = False
        for param in self.backbone_stage2.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻 backbone"""
        for param in self.backbone_stage1.parameters():
            param.requires_grad = True
        for param in self.backbone_stage2.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, 3, 300, 300]

        Returns:
            conf: [B, num_anchors, num_classes] 分类 logits
            loc: [B, num_anchors, 4] 回归偏移
        """
        # 多尺度特征提取
        features = []

        # 19×19×96
        f1 = self.backbone_stage1(x)
        features.append(f1)

        # 10×10×1280
        f2 = self.backbone_stage2(f1)
        features.append(f2)

        # 额外层: 5×5×512, 3×3×256, 1×1×128
        f = f2
        for layer in self.extra_layers:
            f = layer(f)
            features.append(f)

        # 检测头
        conf_list = []
        loc_list = []

        for i, (feat, cls_head, reg_head) in enumerate(zip(features, self.cls_heads, self.reg_heads)):
            cls = cls_head(feat)  # [B, na*nc, H, W]
            reg = reg_head(feat)  # [B, na*4, H, W]

            B, _, H, W = cls.shape
            na = self.num_anchors[i]
            nc = self.num_classes

            cls = cls.permute(0, 2, 3, 1).contiguous().view(B, -1, nc)  # [B, H*W*na, nc]
            reg = reg.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)  # [B, H*W*na, 4]

            conf_list.append(cls)
            loc_list.append(reg)

        conf = torch.cat(conf_list, dim=1)  # [B, total_anchors, nc]
        loc = torch.cat(loc_list, dim=1)    # [B, total_anchors, 4]

        return conf, loc

    @torch.no_grad()
    def detect(self, x, conf_threshold=0.3, nms_threshold=0.45, max_detections=50):
        """
        检测推理 (带 NMS)

        Args:
            x: [B, 3, 300, 300]
            conf_threshold: 置信度阈值
            nms_threshold: NMS 阈值
            max_detections: 最大检测数

        Returns:
            results: list of {boxes, scores, labels}
        """
        self.eval()
        conf, loc = self.forward(x)  # [B, num_anchors, nc], [B, num_anchors, 4]
        anchors = self.get_anchors(x.device)

        B = x.size(0)
        results = []

        for b in range(B):
            # 解码 bbox
            boxes = decode_boxes(loc[b], anchors)  # [num_anchors, 4]
            boxes = cxcywh_to_xyxy(boxes)           # [num_anchors, 4]
            boxes = torch.clamp(boxes, 0, 1)

            # 分类分数
            scores = torch.softmax(conf[b], dim=1)  # [num_anchors, nc]

            # 去掉背景类
            scores = scores[:, 1:]  # [num_anchors, num_classes-1]

            # 每个类别分别 NMS
            all_boxes = []
            all_scores = []
            all_labels = []

            for cls_idx in range(scores.size(1)):
                cls_scores = scores[:, cls_idx]
                mask = cls_scores > conf_threshold
                if mask.sum() == 0:
                    continue

                cls_boxes = boxes[mask]
                cls_scores = cls_scores[mask]

                keep = nms(cls_boxes, cls_scores, nms_threshold)
                all_boxes.append(cls_boxes[keep])
                all_scores.append(cls_scores[keep])
                all_labels.append(torch.full((len(keep),), cls_idx + 1, dtype=torch.long, device=boxes.device))

            if all_boxes:
                all_boxes = torch.cat(all_boxes)
                all_scores = torch.cat(all_scores)
                all_labels = torch.cat(all_labels)

                # 取 top-k
                topk = min(max_detections, all_scores.size(0))
                _, topk_idx = all_scores.topk(topk)

                results.append({
                    "boxes": all_boxes[topk_idx],   # [K, 4] (x1,y1,x2,y2)
                    "scores": all_scores[topk_idx], # [K]
                    "labels": all_labels[topk_idx],  # [K]
                })
            else:
                results.append({
                    "boxes": torch.zeros(0, 4),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                })

        return results


def build_model(config):
    """根据配置构建模型"""
    model_cfg = config["model"]
    return SSDMobileNetV2(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        input_size=model_cfg["input_size"][0],
    )
