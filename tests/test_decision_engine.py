from waste_system.config import get_settings
from waste_system.inference.decision_engine import DecisionEngine
from waste_system.schemas import DecisionRequest, DecisionStatus, WasteClass


def make_request(**overrides):
    values = {
        "probabilities": {
            "glass": 0.03,
            "metal": 0.88,
            "paper": 0.02,
            "plastic": 0.05,
            "unknown": 0.02,
        },
        "image_quality": 0.95,
        "multi_view_agreement": 0.9,
        "metal_detected": True,
        "magnetic": False,
    }
    values.update(overrides)
    return DecisionRequest(**values)


def engine():
    return DecisionEngine(get_settings()["decision"])


def test_accepts_high_confidence_consistent_result():
    result = engine().decide(make_request(), minimum_image_quality=0.6)
    assert result.final_class == WasteClass.METAL
    assert result.status == DecisionStatus.ACCEPTED
    assert result.subtype == "non_ferrous_candidate"


def test_requests_rescan_for_bad_image():
    result = engine().decide(make_request(image_quality=0.2), minimum_image_quality=0.6)
    assert result.final_class == WasteClass.UNKNOWN
    assert result.status == DecisionStatus.RESCAN


def test_flags_sensor_conflict_for_review():
    probabilities = {
        "glass": 0.02,
        "metal": 0.03,
        "paper": 0.03,
        "plastic": 0.9,
        "unknown": 0.02,
    }
    result = engine().decide(make_request(probabilities=probabilities), minimum_image_quality=0.6)
    assert result.final_class == WasteClass.PLASTIC
    assert result.status == DecisionStatus.REVIEW
    assert "metal_sensor_conflict" in result.reason_codes
