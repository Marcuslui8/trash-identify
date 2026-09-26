import cv2
import numpy as np
from fastapi.testclient import TestClient

import waste_system.api.main as api_main
from waste_system.api.main import app


client = TestClient(app)


def test_health_reports_model_state():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["classifier_available"], bool)
    assert isinstance(response.json()["detector_available"], bool)
    assert isinstance(response.json()["multi_object_ready"], bool)


def test_scanner_page_is_available():
    response = client.get("/scanner")
    assert response.status_code == 200
    assert "智慧垃圾掃描" in response.text
    assert "getUserMedia" in response.text
    assert "即時辨識" in response.text
    assert "LIVE_INTERVAL_MS" in response.text


def test_local_ca_certificate_can_be_downloaded(monkeypatch, tmp_path):
    certificate = tmp_path / "trash-identify-ca.crt"
    certificate.write_bytes(b"public-ca-certificate")
    monkeypatch.setattr(api_main, "project_path", lambda _path: certificate)

    response = client.get("/local-ca.crt")

    assert response.status_code == 200
    assert response.content == b"public-ca-certificate"
    assert "trash-identify-ca.crt" in response.headers["content-disposition"]


def test_multi_predict_rejects_low_quality_before_models_are_needed():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    response = client.post(
        "/v1/predict-multiple",
        files={"file": ("dark.jpg", buffer.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["objects"] == []
    assert payload["image_quality"] < 0.6


def test_single_predict_returns_classification(monkeypatch):
    class FakeClassifier:
        def predict(self, _image):
            return {
                "glass": 0.02,
                "metal": 0.03,
                "paper": 0.9,
                "plastic": 0.03,
                "unknown": 0.02,
            }

    monkeypatch.setattr(api_main, "classifier", lambda: FakeClassifier())
    image = np.full((100, 100, 3), 127, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded

    response = client.post(
        "/v1/predict",
        files={"file": ("paper.jpg", buffer.tobytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["probabilities"]["paper"] == 0.9
    assert "result" in response.json()


def test_decide_returns_grounded_result():
    response = client.post(
        "/v1/decide",
        json={
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
            "weight_grams": 18,
            "region": "Taiwan",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["final_class"] == "metal"
    assert payload["result"]["status"] == "accepted"
    assert "鐵鋁類" in payload["advice"]
