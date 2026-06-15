from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_scene_manifest(path: str | Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            image_path, label = line.split("\t")
            samples.append((Path(image_path), int(label)))
    return samples


class SceneDataset(Dataset):
    def __init__(self, manifest_path: str | Path, image_size: int = 224, augment: bool = False):
        self.samples = read_scene_manifest(manifest_path)
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size + 32, image_size + 32)),
                    transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label
