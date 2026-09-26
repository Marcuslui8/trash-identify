from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from scripts.import_coxpwd import (
    coco_bbox_to_yolo,
    find_annotations,
    import_split,
)


def test_coco_bbox_is_clipped_to_image() -> None:
    assert coco_bbox_to_yolo([-10, 20, 60, 40], 100, 100) == (0.25, 0.4, 0.5, 0.4)


def make_jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(output, format="JPEG")
    return output.getvalue()


def test_import_split_converts_all_categories_to_waste(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.zip"
    coco = {
        "images": [{"id": 7, "file_name": "sample.jpg", "width": 100, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 3, "bbox": [10, 8, 40, 32]}
        ],
        "categories": [{"id": 3, "name": "bottle"}],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        for split in ("train", "valid", "test"):
            archive.writestr(f"plastic/{split}/_annotations.coco.json", json.dumps(coco))
            archive.writestr(f"plastic/{split}/sample.jpg", make_jpeg())

    output_root = tmp_path / "detection"
    with zipfile.ZipFile(archive_path) as archive:
        annotations = find_annotations(archive)
        rows = import_split(archive, annotations["train"], "train", output_root, None)

    assert len(rows) == 1
    label = output_root / "labels" / "train" / "coxpwd_train_000007.txt"
    assert label.read_text(encoding="utf-8") == "0 0.30000000 0.30000000 0.40000000 0.40000000\n"
    assert json.loads(rows[0]["source_categories"]) == {"bottle": 1}
