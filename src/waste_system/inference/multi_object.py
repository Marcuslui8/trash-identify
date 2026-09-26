from __future__ import annotations

from typing import Any

import numpy as np

from waste_system.inference.classifier import TorchscriptClassifier
from waste_system.inference.decision_engine import DecisionEngine
from waste_system.inference.detector import WasteDetector
from waste_system.inference.preprocessing import prepare_rgb
from waste_system.llm.advisor import TemplateAdvisor
from waste_system.schemas import (
    BoundingBox,
    DecisionRequest,
    MultiPredictionResponse,
    ObjectPrediction,
)


class MultiObjectPipeline:
    def __init__(
        self,
        detector: WasteDetector,
        classifier: TorchscriptClassifier,
        settings: dict[str, Any],
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.settings = settings
        self.decision_engine = DecisionEngine(settings["decision"])
        self.advisor = TemplateAdvisor()

    def predict(
        self,
        image: np.ndarray,
        *,
        image_quality: float,
        region: str,
    ) -> MultiPredictionResponse:
        height, width = image.shape[:2]
        predictions: list[ObjectPrediction] = []
        image_settings = self.settings["image"]

        for object_id, detection in enumerate(self.detector.predict(image), start=1):
            crop = image[detection.y1 : detection.y2, detection.x1 : detection.x2]
            if crop.size == 0:
                continue
            rgb = prepare_rgb(crop, image_settings["width"], image_settings["height"])
            probabilities = self.classifier.predict(rgb)
            request = DecisionRequest(
                probabilities=probabilities,
                image_quality=image_quality,
                multi_view_agreement=1.0,
                region=region,
            )
            result = self.decision_engine.decide(
                request,
                minimum_image_quality=image_settings["minimum_quality"],
            )
            predictions.append(
                ObjectPrediction(
                    object_id=object_id,
                    box=BoundingBox(
                        x1=detection.x1,
                        y1=detection.y1,
                        x2=detection.x2,
                        y2=detection.y2,
                    ),
                    detection_confidence=detection.confidence,
                    classification=result,
                    probabilities=probabilities,
                    advice=self.advisor.advise(result, region),
                )
            )

        return MultiPredictionResponse(
            image_width=width,
            image_height=height,
            image_quality=image_quality,
            objects=predictions,
            region=region,
        )
