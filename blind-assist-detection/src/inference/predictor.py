from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from src.configs.runtime import load_config, resolve_project_path
from src.inference.risk_assessment import summarize_risks
from src.models.ssd_mobilenet import SSDMobileNetV2


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["box"] = [round(value, 2) for value in self.box]
        payload["score"] = round(self.score, 4)
        return payload


class DetectionPredictor:
    def __init__(self, config_path: str | Path, checkpoint_path: str | Path, gpu: int = 0):
        self.config_path = Path(config_path)
        self.checkpoint_path = resolve_project_path(checkpoint_path)
        self.config = load_config(self.config_path)

        self.device = torch.device(
            f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu"
        )
        self.input_size = self.config["model"]["input_size"][0]

        classes_path = resolve_project_path(self.config["data"]["classes_file"])
        with open(classes_path, "r", encoding="utf-8") as file:
            self.classes = [line.strip() for line in file if line.strip()]

        self.model = SSDMobileNetV2(
            num_classes=self.config["model"]["num_classes"],
            pretrained=False,
            input_size=self.input_size,
            backbone=self.config["model"].get("backbone", "mobilenet_v2"),
            use_eca=self.config["model"].get("use_eca", False),
            eca_stages=self.config["model"].get("eca_stages"),
        ).to(self.device)
        self.model.configure_anchors(
            self.config["anchors"]["feature_maps"],
            self.config["anchors"]["min_sizes"],
            self.config["anchors"]["max_sizes"],
            self.config["anchors"]["aspect_ratios"],
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.font = self._load_font()

    @staticmethod
    def _load_font() -> ImageFont.ImageFont:
        font_candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        for font_path in font_candidates:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size=16)
        return ImageFont.load_default()

    @staticmethod
    def _measure_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
    ) -> tuple[int, int]:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(text)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            return font.getsize(text)

    def predict(self, image: Image.Image, conf_threshold: float | None = None) -> list[Detection]:
        source = image.convert("RGB")
        tensor = self.transform(source).unsqueeze(0).to(self.device)
        infer_cfg = self.config["inference"]
        threshold = infer_cfg["conf_threshold"] if conf_threshold is None else conf_threshold

        with torch.no_grad():
            result = self.model.detect(
                tensor,
                conf_threshold=threshold,
                nms_threshold=infer_cfg["nms_threshold"],
                max_detections=infer_cfg["max_detections"],
            )[0]

        width, height = source.size
        detections: list[Detection] = []
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            class_idx = int(label.item()) - 1
            if class_idx < 0 or class_idx >= len(self.classes):
                continue

            x1, y1, x2, y2 = box.tolist()
            detections.append(
                Detection(
                    label=self.classes[class_idx],
                    score=float(score.item()),
                    box=(x1 * width, y1 * height, x2 * width, y2 * height),
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        return detections

    def assess_risks(self, image: Image.Image, detections: list[Detection]) -> dict:
        return summarize_risks(detections, image.convert("RGB").size)

    def render(self, image: Image.Image, detections: list[Detection]) -> Image.Image:
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            label_text = f"{detection.label} {detection.score:.2f}"

            draw.rectangle((x1, y1, x2, y2), outline=(220, 38, 38), width=4)
            text_width, text_height = self._measure_text(draw, label_text, self.font)
            text_x = x1 + 6
            text_y = max(4, y1 - text_height - 10)
            draw.rounded_rectangle(
                (text_x - 4, text_y - 3, text_x + text_width + 6, text_y + text_height + 3),
                radius=6,
                fill=(160, 20, 20),
            )
            draw.text((text_x, text_y), label_text, fill=(255, 248, 240), font=self.font)

        return canvas
