"""Import the official MJU-Waste Pascal VOC release into the YOLO dataset.

The image archive is intentionally not downloaded by this repository. Download
the archive from the official MJU-Waste page, extract it, then pass the
extracted directory to this script.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import shutil
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import MJU-Waste VOC data into YOLO")
    parser.add_argument("source_root", type=Path, help="Extracted MJU-Waste directory")
    parser.add_argument("--output-root", type=Path, default=Path("data/detection"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/annotations/mju_waste_import.csv")
    )
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def stem_from_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    name = Path(value.replace("\\", "/")).name
    stem = Path(name).stem
    return stem or None


def collect_split_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, str):
            stem = stem_from_value(value)
            if stem:
                values.add(stem)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(payload)
    return values


def official_splits(source_root: Path) -> dict[str, str]:
    split_names: dict[str, str] = {}
    filenames = {
        "train.json": "train",
        "val.json": "validation",
        "validation.json": "validation",
        "test.json": "test",
    }
    for split_file in source_root.rglob("*.json"):
        split = filenames.get(split_file.name.lower())
        if split is None:
            continue
        for stem in collect_split_names(split_file):
            split_names.setdefault(stem, split)
    return split_names


def fallback_split(stem: str) -> str:
    bucket = int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def find_files(source_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    images: dict[str, Path] = {}
    annotations: dict[str, Path] = {}
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(path.stem, path)
        elif path.suffix.lower() == ".xml":
            annotations.setdefault(path.stem, path)
    return images, annotations


def parse_boxes(annotation: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    root = ET.parse(annotation).getroot()
    boxes: list[tuple[float, float, float, float]] = []
    for obj in root.findall(".//object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            x1 = max(0.0, min(float(width), float(box.findtext("xmin", "0"))))
            y1 = max(0.0, min(float(height), float(box.findtext("ymin", "0"))))
            x2 = max(0.0, min(float(width), float(box.findtext("xmax", "0"))))
            y2 = max(0.0, min(float(height), float(box.findtext("ymax", "0"))))
        except ValueError:
            continue
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def find_supervisely_pairs(
    source_root: Path,
) -> list[tuple[str, Path, Path, str]]:
    """Find color-image/JSON pairs from a Supervisely export."""
    pairs: list[tuple[str, Path, Path, str]] = []
    for folder_name, split in (("train", "train"), ("val", "validation"), ("validation", "validation"), ("test", "test")):
        image_dir = source_root / folder_name / "img"
        annotation_dir = source_root / folder_name / "ann"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if "_depth" in image_path.stem.lower():
                continue
            annotation_path = annotation_dir / f"{image_path.name}.json"
            if annotation_path.is_file():
                pairs.append((image_path.stem, image_path, annotation_path, split))
    return pairs


def parse_supervisely_boxes(annotation: Path) -> list[tuple[float, float, float, float]]:
    """Convert Supervisely bitmap masks to pixel bounding boxes."""
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    boxes: list[tuple[float, float, float, float]] = []
    for obj in payload.get("objects", []):
        bitmap = obj.get("bitmap") or {}
        encoded = bitmap.get("data")
        origin = bitmap.get("origin") or [0, 0]
        if not encoded or len(origin) != 2:
            continue
        try:
            png_bytes = zlib.decompress(base64.b64decode(encoded))
            with Image.open(io.BytesIO(png_bytes)) as mask:
                binary = mask.convert("L").point(lambda value: 255 if value else 0)
                bounds = binary.getbbox()
        except (ValueError, OSError, zlib.error):
            continue
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        ox, oy = float(origin[0]), float(origin[1])
        boxes.append((ox + x1, oy + y1, ox + x2, oy + y2))
    return boxes


def yolo_line(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return "0 %.6f %.6f %.6f %.6f" % (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def save_image(source: Path, destination: Path) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.save(destination, format="JPEG", quality=95, optimize=True)
        return rgb.width, rgb.height


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_root}")

    supervisely_pairs = find_supervisely_pairs(source_root)
    split_names = official_splits(source_root)
    if supervisely_pairs:
        records = [
            (stem, image, annotation, split, "supervisely")
            for stem, image, annotation, split in supervisely_pairs
        ]
    else:
        images, annotations = find_files(source_root)
        records = [
            (stem, images[stem], annotations[stem], split_names.get(stem), "voc")
            for stem in sorted(images)
            if stem in annotations
        ]
    if args.max_images is not None:
        records = records[: args.max_images]
    if not records:
        raise SystemExit("No image/XML pairs found. Expected JPEG images and Pascal VOC XML files.")

    manifest_rows: list[dict[str, object]] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    for index, (stem, image_path, annotation_path, known_split, annotation_format) in enumerate(records, start=1):
        split = known_split or fallback_split(stem)
        output_stem = f"mju_{stem}"
        image_dest = args.output_root / "images" / split / f"{output_stem}.jpg"
        label_dest = args.output_root / "labels" / split / f"{output_stem}.txt"
        width, height = save_image(image_path, image_dest)
        boxes = (
            parse_supervisely_boxes(annotation_path)
            if annotation_format == "supervisely"
            else parse_boxes(annotation_path, width, height)
        )
        if not boxes:
            image_dest.unlink(missing_ok=True)
            continue
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.write_text(
            "\n".join(yolo_line(box, width, height) for box in boxes) + "\n",
            encoding="utf-8",
        )
        counts[split] += 1
        manifest_rows.append(
            {
                "image": str(image_dest).replace("\\", "/"),
                "source_image": str(image_path),
                "source_annotation": str(annotation_path),
                "split": split,
                "boxes": len(boxes),
                "split_source": "official" if stem in split_names else "sha256_fallback",
            }
        )
        if index % 250 == 0 or index == len(records):
            print(f"Imported {index}/{len(records)} pairs", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Imported {len(manifest_rows)} images: {counts}")
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
