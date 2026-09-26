from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from waste_system.schemas import WasteClass


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class ModelUnavailableError(RuntimeError):
    pass


class TorchscriptClassifier:
    def __init__(self, model_path: Path, classes: Sequence[str], device: str = "cpu") -> None:
        self.model_path = model_path
        self.classes = [WasteClass(value) for value in classes]
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)
        self._model: torch.jit.ScriptModule | None = None

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def _load(self) -> torch.jit.ScriptModule:
        if not self.available:
            raise ModelUnavailableError(f"Model file not found: {self.model_path}")
        if self._model is None:
            self._model = torch.jit.load(str(self.model_path), map_location=self.device)
            self._model.eval()
        return self._model

    def predict(self, rgb_image: np.ndarray) -> dict[WasteClass, float]:
        model = self._load()
        normalized = (rgb_image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        if len(probabilities) != len(self.classes):
            raise RuntimeError("Model output size does not match configured classes")
        return {label: float(value) for label, value in zip(self.classes, probabilities, strict=True)}

    @staticmethod
    def average(predictions: Sequence[dict[WasteClass, float]]) -> dict[WasteClass, float]:
        if not predictions:
            raise ValueError("At least one prediction is required")
        return {
            label: sum(item[label] for item in predictions) / len(predictions)
            for label in WasteClass
        }
