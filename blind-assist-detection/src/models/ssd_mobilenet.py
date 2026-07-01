"""
SSD300 检测模型

结构:
  Backbone → 多尺度特征 → SSD 检测头 → NMS
"""
import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet50_Weights,
    VGG16_BN_Weights,
    mobilenet_v2,
    resnet50,
    vgg16_bn,
)

from .box_utils import generate_anchors, decode_boxes, cxcywh_to_xyxy, nms


class ECABlock(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, channels: int, gamma: int = 2, bias: int = 1):
        super().__init__()
        kernel_size = int(abs((torch.log2(torch.tensor(float(channels))).item() + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel_size = max(kernel_size, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        weights = self.pool(x)
        weights = weights.squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights)
        weights = self.activation(weights.transpose(-1, -2).unsqueeze(-1))
        return x * weights.expand_as(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        channel_attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        x = x * channel_attn

        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.amax(x, dim=1, keepdim=True)
        spatial_attn = self.spatial(torch.cat([avg_spatial, max_spatial], dim=1))
        return x * spatial_attn


class SSDMobileNetV2(nn.Module):
    """
    基于 MobileNetV2 的 SSD 检测模型

    Args:
        num_classes: 类别数 (含背景)
        pretrained: 是否使用 ImageNet 预训练
        input_size: 输入尺寸
    """

    # MobileNetV2 has 19 feature blocks, split at index 14:
    #   features[0:14] → backbone_stage1 (indices 0-13)
    #   features[14:19] → backbone_stage2 (indices 14-18)
    _OLD_BACKBONE_MAPPING = None

    @classmethod
    def _get_backbone_remap(cls):
        if cls._OLD_BACKBONE_MAPPING is not None:
            return cls._OLD_BACKBONE_MAPPING
        remap = {}
        # features.{i}.X → backbone_stage1.{i}.X  for i=0..13
        for i in range(14):
            remap[f"features.{i}."] = f"backbone_stage1.{i}."
        # features.{i}.X → backbone_stage2.{i-14}.X  for i=14..18
        for i in range(14, 19):
            remap[f"features.{i}."] = f"backbone_stage2.{i - 14}."
        cls._OLD_BACKBONE_MAPPING = remap
        return remap

    def load_state_dict(self, state_dict, strict=True):
        """Load state_dict with backward compatibility for old checkpoint keys."""
        remap = self._get_backbone_remap()
        new_state = {}
        for key, value in state_dict.items():
            mapped = key
            for old_prefix, new_prefix in remap.items():
                if key.startswith(old_prefix):
                    mapped = new_prefix + key[len(old_prefix):]
                    break
            new_state[mapped] = value

        # Also handle the inverse: new model expects features.* but gets backbone_stage*
        if "features.0.0.weight" in state_dict or any(k.startswith("features.") for k in state_dict):
            pass  # already handled above

        return super().load_state_dict(new_state, strict=strict)

    def __init__(
        self,
        num_classes=9,
        pretrained=True,
        input_size=300,
        backbone="mobilenet_v2",
        attention_type=None,
        use_eca=False,
        eca_stages=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.backbone_name = backbone
        self.attention_type = (attention_type or ("eca" if use_eca else "")).lower()

        self.backbone_stage1, self.backbone_stage2, stage_channels = self._build_backbone(
            backbone=backbone,
            pretrained=pretrained,
        )

        # ---- 额外特征层 (逐步下采样) ----
        self.extra_layers = nn.ModuleList([
            # 10×10 → 5×5
            self._make_extra_layer(stage_channels[1], 512, stride=2),
            # 5×5×512 → 3×3×256
            self._make_extra_layer(512, 256, stride=2),
            # 3×3×256 → 2×2×128
            self._make_extra_layer(256, 128, stride=2),
        ])

        # 特征图通道数: 19×19, 10×10, 5×5, 3×3, 2×2
        self.feat_channels = [stage_channels[0], stage_channels[1], 512, 256, 128]
        requested_stages = set(eca_stages or range(len(self.feat_channels)))
        self.attention_layers = nn.ModuleList(
            [self._build_attention_layer(ch, idx in requested_stages) for idx, ch in enumerate(self.feat_channels)]
        )

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

    def _build_attention_layer(self, channels: int, enabled: bool) -> nn.Module:
        if not enabled or not self.attention_type:
            return nn.Identity()
        if self.attention_type == "eca":
            return ECABlock(channels)
        if self.attention_type == "se":
            return SEBlock(channels)
        if self.attention_type == "cbam":
            return CBAMBlock(channels)
        raise ValueError(f"unsupported attention type: {self.attention_type}")

    def _build_backbone(self, backbone: str, pretrained: bool) -> tuple[nn.Module, nn.Module, tuple[int, int]]:
        backbone = backbone.lower()
        if backbone == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = mobilenet_v2(weights=weights)
            stage1 = nn.Sequential(*list(model.features)[:14])   # 19x19x96
            stage2 = nn.Sequential(*list(model.features)[14:])   # 10x10x1280
            return stage1, stage2, (96, 1280)

        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            stage1 = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
            )  # 19x19x1024
            stage2 = model.layer4  # 10x10x2048
            return stage1, stage2, (1024, 2048)

        if backbone == "vgg16":
            weights = VGG16_BN_Weights.DEFAULT if pretrained else None
            model = vgg16_bn(weights=weights)
            # Use ceil_mode to align 300x300 inputs with SSD-style 38/19/10 feature maps.
            for pool_idx in (23, 33, 43):
                if hasattr(model.features[pool_idx], "ceil_mode"):
                    model.features[pool_idx].ceil_mode = True
            stage1 = nn.Sequential(*list(model.features)[:34])   # 19x19x512
            stage2 = nn.Sequential(*list(model.features)[34:44]) # 10x10x512
            return stage1, stage2, (512, 512)

        raise ValueError(f"unsupported backbone: {backbone}")

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

        for i, (feat, attention_layer, cls_head, reg_head) in enumerate(
            zip(features, self.attention_layers, self.cls_heads, self.reg_heads)
        ):
            feat = attention_layer(feat)
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
        backbone=model_cfg.get("backbone", "mobilenet_v2"),
        attention_type=model_cfg.get("attention_type"),
        use_eca=model_cfg.get("use_eca", False),
        eca_stages=model_cfg.get("eca_stages"),
    )
