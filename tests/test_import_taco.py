import io

from PIL import Image

from scripts.import_taco import assign_batch_splits, coco_bbox_to_yolo, write_jpeg


def test_coco_bbox_is_clipped_and_normalized():
    converted = coco_bbox_to_yolo([-10, 20, 60, 40], 100, 100)
    assert converted == (0.25, 0.4, 0.5, 0.4)


def test_batch_split_keeps_every_batch_together():
    images = [
        {"file_name": f"batch_{batch}/image_{index}.jpg"}
        for batch in range(1, 16)
        for index in range(3)
    ]
    mapping = assign_batch_splits(images, seed=42)
    assert set(mapping.values()) == {"train", "validation", "test"}
    assert len(mapping) == 15
    assert sum(value == "train" for value in mapping.values()) == 11


def test_write_jpeg_applies_exif_orientation(tmp_path):
    source = Image.new("RGB", (4, 2), color=(20, 40, 60))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    encoded = io.BytesIO()
    source.save(encoded, format="JPEG", exif=exif)

    destination = tmp_path / "rotated.jpg"
    width, height, digest = write_jpeg(encoded.getvalue(), destination, 2, 4)

    assert (width, height) == (2, 4)
    assert len(digest) == 64
    with Image.open(destination) as saved:
        assert saved.size == (2, 4)
