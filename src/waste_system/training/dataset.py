from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = ["glass", "metal", "paper", "plastic", "unknown"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


class WasteCsvDataset(Dataset):
    def __init__(
        self,
        annotations: Path,
        project_root: Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.project_root = project_root
        self.transform = transform
        with annotations.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        self.rows = [row for row in rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"No rows found for split={split!r} in {annotations}")
        unknown_classes = sorted({row["target_class"] for row in self.rows} - set(CLASS_NAMES))
        if unknown_classes:
            raise ValueError(f"Unsupported target classes: {unknown_classes}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.project_root / row["image_path"]
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = CLASS_TO_INDEX[row["target_class"]]
        return image, label, row["image_path"]

    def class_counts(self) -> Counter[str]:
        return Counter(row["target_class"] for row in self.rows)


def build_transform(training: bool):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if training:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def balanced_class_weights(dataset: WasteCsvDataset) -> torch.Tensor:
    counts = dataset.class_counts()
    total = sum(counts.values())
    return torch.tensor(
        [total / (len(CLASS_NAMES) * counts[name]) for name in CLASS_NAMES],
        dtype=torch.float32,
    )
