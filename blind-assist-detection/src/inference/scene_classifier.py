from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2

from src.configs.runtime import resolve_project_path


@dataclass
class ScenePrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class SceneClassifier:
    def __init__(self, checkpoint_path: str | Path, gpu: int = 0):
        self.checkpoint_path = resolve_project_path(checkpoint_path)
        self.device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.classes = checkpoint.get("classes", ["indoor", "outdoor"])
        self.image_size = checkpoint.get("image_size", 224)

        self.model = mobilenet_v2(weights=None, num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image: Image.Image) -> ScenePrediction:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor)[0], dim=0).detach().cpu()
        best_idx = int(torch.argmax(probs).item())
        probabilities = {label: float(probs[idx].item()) for idx, label in enumerate(self.classes)}
        return ScenePrediction(
            label=self.classes[best_idx],
            confidence=float(probs[best_idx].item()),
            probabilities=probabilities,
        )
