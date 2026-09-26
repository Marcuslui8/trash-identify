import numpy as np

from waste_system.config import get_settings
from waste_system.inference.detector import Detection
from waste_system.inference.multi_object import MultiObjectPipeline
from waste_system.schemas import DecisionStatus, WasteClass


class FakeDetector:
    available = True

    def predict(self, image):
        return [
            Detection(0, 0, 40, 60, 0.95),
            Detection(45, 10, 90, 70, 0.88),
        ]


class FakeClassifier:
    available = True

    def predict(self, image):
        return {
            WasteClass.GLASS: 0.02,
            WasteClass.METAL: 0.90,
            WasteClass.PAPER: 0.02,
            WasteClass.PLASTIC: 0.04,
            WasteClass.UNKNOWN: 0.02,
        }


def test_pipeline_classifies_every_detected_object():
    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    pipeline = MultiObjectPipeline(FakeDetector(), FakeClassifier(), get_settings())
    response = pipeline.predict(image, image_quality=0.95, region="Taiwan")

    assert len(response.objects) == 2
    assert response.objects[0].box.x2 == 40
    assert response.objects[0].classification.final_class == WasteClass.METAL
    assert response.objects[0].classification.status == DecisionStatus.ACCEPTED
