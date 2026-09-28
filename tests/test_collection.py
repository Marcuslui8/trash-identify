import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import waste_system.api.main as api_main
from waste_system.api.main import app
from waste_system.collection import CollectionError, collection_status, save_capture


def encoded_image(seed: int = 42) -> bytes:
    image = np.random.default_rng(seed).integers(30, 225, (480, 640, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def test_save_capture_and_status(tmp_path: Path):
    root = tmp_path / "local_capture"
    record = save_capture(
        encoded_image(),
        category="positive",
        session="desk_day_01",
        collection_root=root,
        notes="two bottles",
    )

    image_path = root / record["image_path"]
    assert image_path.is_file()
    assert record["annotation_status"] == "pending"
    assert collection_status(root)["counts"]["positive"] == 1
    saved_record = json.loads((root / "records" / f"{record['capture_id']}.json").read_text("utf-8"))
    assert saved_record["session"] == "desk_day_01"


def test_duplicate_capture_is_rejected(tmp_path: Path):
    root = tmp_path / "local_capture"
    content = encoded_image()
    save_capture(content, category="negative", session="floor_01", collection_root=root)
    with pytest.raises(CollectionError, match="相同|太相似"):
        save_capture(content, category="negative", session="floor_01", collection_root=root)


def test_invalid_session_is_rejected(tmp_path: Path):
    with pytest.raises(CollectionError, match="Session"):
        save_capture(
            encoded_image(),
            category="positive",
            session="../outside",
            collection_root=tmp_path / "local_capture",
        )


def test_collection_api_and_page(monkeypatch, tmp_path: Path):
    root = tmp_path / "local_capture"
    original_project_path = api_main.project_path

    def fake_project_path(value: str) -> Path:
        return root if value == "data/local_capture" else original_project_path(value)

    monkeypatch.setattr(api_main, "project_path", fake_project_path)
    client = TestClient(app)
    page = client.get("/collector")
    assert page.status_code == 200
    assert "手機垃圾資料蒐集" in page.text

    response = client.post(
        "/v1/collection/images",
        files={"file": ("capture.jpg", encoded_image(9), "image/jpeg")},
        data={"category": "hard", "session": "night_01", "notes": "dark"},
    )
    assert response.status_code == 201
    assert response.json()["accepted"] is True
    assert client.get("/v1/collection/status").json()["counts"]["hard"] == 1
