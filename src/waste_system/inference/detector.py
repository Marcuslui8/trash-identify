from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch


class DetectorUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class WasteDetector(Protocol):
    @property
    def available(self) -> bool: ...

    def predict(self, image: np.ndarray) -> list[Detection]: ...


class UltralyticsWasteDetector:
    """Lazy Ultralytics adapter. The application can start without the optional package."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "auto",
        confidence: float = 0.3,
        iou: float = 0.5,
        maximum_objects: int = 20,
    ) -> None:
        self.model_path = model_path
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.confidence = confidence
        self.iou = iou
        self.maximum_objects = maximum_objects
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def _load(self) -> Any:
        if not self.available:
            raise DetectorUnavailableError(f"Detector file not found: {self.model_path}")
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise DetectorUnavailableError(
                    "Ultralytics is not installed; install the detection extra with "
                    "python -m pip install -e '.[detection]'"
                ) from exc
            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(self, image: np.ndarray) -> list[Detection]:
        model = self._load()
        results = model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None:
            return []

        height, width = image.shape[:2]
        detections: list[Detection] = []
        for coordinates, score in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), strict=True):
            x1, y1, x2, y2 = coordinates
            detection = Detection(
                x1=max(0, min(width - 1, int(round(x1)))),
                y1=max(0, min(height - 1, int(round(y1)))),
                x2=max(1, min(width, int(round(x2)))),
                y2=max(1, min(height, int(round(y2)))),
                confidence=float(score),
            )
            if detection.x2 > detection.x1 and detection.y2 > detection.y1:
                detections.append(detection)

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[: self.maximum_objects]
