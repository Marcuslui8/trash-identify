from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DATASET_ID = "BowerApp/bower-waste-annotations"
DATASET_PAGE = f"https://huggingface.co/datasets/{DATASET_ID}"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
TOTAL_ROWS = 2546
PAGE_SIZE = 100
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Bower phone-camera boxes into YOLO and CNN datasets"
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/detection"))
    parser.add_argument(
        "--crop-root", type=Path, default=Path("data/processed/bower")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/annotations/bower_import.csv")
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path("data/annotations/bower_classification.csv"),
    )
    parser.add_argument(
        "--combined-classification",
        type=Path,
        default=Path("data/annotations/combined_classification.csv"),
    )
    parser.add_argument(
        "--realwaste",
        type=Path,
        default=Path("data/annotations/realwaste_split.csv"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--metadata-cache", type=Path, default=Path("data/bower/rows.json")
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.20,
        help="Fraction of images eligible for one non-overlapping background crop",
    )
    parser.add_argument("--limit", type=int, default=None, help="Import N unique images")
    return parser.parse_args()


def download_bytes(url: str, *, timeout: int, retries: int) -> bytes:
    last_error: Exception | None = None
    request = urllib.request.Request(
        url, headers={"User-Agent": "trash-identify/0.7 Bower importer"}
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if attempt < retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = min(60, int(retry_after) if retry_after and retry_after.isdigit() else 15)
                print(f"HTTP {exc.code}; retrying in {delay}s ...", flush=True)
                time.sleep(delay)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def fetch_rows(
    *, timeout: int, retries: int, metadata_cache: Path
) -> list[dict[str, Any]]:
    if metadata_cache.is_file():
        cached = json.loads(metadata_cache.read_text(encoding="utf-8"))
        if isinstance(cached, list) and len(cached) == TOTAL_ROWS:
            print(f"Using cached annotations: {metadata_cache}", flush=True)
            return cached
    rows: list[dict[str, Any]] = []
    for offset in range(0, TOTAL_ROWS, PAGE_SIZE):
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET_ID,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        page = json.loads(
            download_bytes(f"{ROWS_ENDPOINT}?{query}", timeout=timeout, retries=retries)
        )
        rows.extend(entry["row"] for entry in page["rows"])
        print(f"Fetched annotations: {len(rows)}/{page['num_rows_total']}", flush=True)
    if len(rows) != TOTAL_ROWS:
        raise ValueError(f"Expected {TOTAL_ROWS} Bower rows, received {len(rows)}")
    metadata_cache.parent.mkdir(parents=True, exist_ok=True)
    metadata_cache.write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    return rows


def split_for_image(image_id: str) -> str:
    bucket = int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def map_material(material: str | None) -> str:
    normalized = (material or "").strip().lower()
    if normalized.startswith("glass"):
        return "glass"
    if normalized == "metal" or normalized.startswith("metal "):
        return "metal"
    if normalized in {
        "fiber - paper",
        "fiber - cardboard / paper packaging",
        "paper",
        "cardboard",
    }:
        return "paper"
    if "plastic" in normalized:
        return "plastic"
    return "unknown"


def clip_box(
    row: dict[str, Any], width: int, height: int
) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(width), float(row["x1"])))
    y1 = max(0.0, min(float(height), float(row["y1"])))
    x2 = max(0.0, min(float(width), float(row["x2"])))
    y2 = max(0.0, min(float(height), float(row["y2"])))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def yolo_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def remove_contained_boxes(
    boxes: list[tuple[float, float, float, float]], threshold: float = 0.90
) -> list[tuple[float, float, float, float]]:
    kept: list[tuple[float, float, float, float]] = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        contained = False
        for other_index, other in enumerate(boxes):
            if index == other_index:
                continue
            ox1, oy1, ox2, oy2 = other
            other_area = (ox2 - ox1) * (oy2 - oy1)
            if other_area < area:
                continue
            intersection = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(
                0.0, min(y2, oy2) - max(y1, oy1)
            )
            if intersection / max(area, 1.0) >= threshold:
                if other_area > area or other_index < index:
                    contained = True
                    break
        if not contained:
            kept.append(box)
    return kept


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_image(content: bytes, destination: Path) -> tuple[int, int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(content)) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        width, height = rgb.size
        temporary = destination.with_suffix(".part.jpg")
        rgb.save(temporary, format="JPEG", quality=95, optimize=True)
        temporary.replace(destination)
    return width, height, sha256_file(destination)


def intersects(
    first: tuple[int, int, int, int], second: tuple[float, float, float, float]
) -> bool:
    return not (
        first[2] <= second[0]
        or first[0] >= second[2]
        or first[3] <= second[1]
        or first[1] >= second[3]
    )


def background_crop(
    width: int,
    height: int,
    boxes: list[tuple[float, float, float, float]],
) -> tuple[int, int, int, int] | None:
    crop_width = max(160, int(width * 0.28))
    crop_height = max(160, int(height * 0.28))
    if crop_width > width or crop_height > height:
        return None
    margin_x = width * 0.04
    margin_y = height * 0.04
    expanded = [
        (
            max(0.0, x1 - margin_x),
            max(0.0, y1 - margin_y),
            min(float(width), x2 + margin_x),
            min(float(height), y2 + margin_y),
        )
        for x1, y1, x2, y2 in boxes
    ]
    x_positions = sorted({0, (width - crop_width) // 2, width - crop_width})
    y_positions = sorted({0, (height - crop_height) // 2, height - crop_height})
    for y1 in y_positions:
        for x1 in x_positions:
            candidate = (x1, y1, x1 + crop_width, y1 + crop_height)
            if not any(intersects(candidate, box) for box in expanded):
                return candidate
    return None


def eligible_for_negative(image_id: str, ratio: float) -> bool:
    value = int(hashlib.sha256(f"negative:{image_id}".encode()).hexdigest()[:8], 16)
    return value / 0xFFFFFFFF < ratio


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not 0 <= args.negative_ratio <= 1:
        raise ValueError("--negative-ratio must be within 0..1")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    raw_rows = fetch_rows(
        timeout=args.timeout,
        retries=args.retries,
        metadata_cache=args.metadata_cache,
    )
    rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        rows_by_image[str(row["image_id"])].append(row)
    image_ids = sorted(rows_by_image)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    manifest_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def import_image(image_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        source_rows = rows_by_image[image_id]
        first = source_rows[0]
        split = split_for_image(image_id)
        image_path = args.output_root / "images" / split / f"bower_{image_id}.jpg"
        if image_path.is_file():
            with Image.open(image_path) as existing:
                width, height = existing.size
            digest = sha256_file(image_path)
        else:
            content = download_bytes(
                str(first["image"]["src"]), timeout=args.timeout, retries=args.retries
            )
            width, height, digest = save_image(content, image_path)

        valid_entries: list[
            tuple[dict[str, Any], tuple[float, float, float, float]]
        ] = []
        for row in source_rows:
            box = clip_box(row, width, height)
            if box is not None:
                valid_entries.append((row, box))
        boxes = [box for _, box in valid_entries]
        detection_boxes = remove_contained_boxes(boxes)
        label_lines = [
            "0 " + " ".join(f"{value:.8f}" for value in yolo_box(box, width, height))
            for box in detection_boxes
        ]
        local_classification: list[dict[str, Any]] = []
        materials: Counter[str] = Counter()
        objects: Counter[str] = Counter()
        with Image.open(image_path) as image_file:
            rgb = image_file.convert("RGB")
            for index, (row, box) in enumerate(valid_entries):
                material = str(row.get("material") or "UNSPECIFIED")
                object_name = str(row.get("object") or "UNSPECIFIED")
                materials[material] += 1
                objects[object_name] += 1

                x1, y1, x2, y2 = box
                pad_x = (x2 - x1) * 0.08
                pad_y = (y2 - y1) * 0.08
                crop_box = (
                    int(max(0, x1 - pad_x)),
                    int(max(0, y1 - pad_y)),
                    int(min(width, x2 + pad_x)),
                    int(min(height, y2 + pad_y)),
                )
                crop_path = args.crop_root / split / f"bower_{image_id}_{index:02d}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop = rgb.crop(crop_box)
                crop.save(crop_path, format="JPEG", quality=95, optimize=True)
                local_classification.append(
                    {
                        "image_path": crop_path.as_posix(),
                        "source_dataset": "Bower waste annotations",
                        "source_category": material,
                        "target_class": map_material(row.get("material")),
                        "item_id": f"bower_{image_id}_{index:02d}",
                        "width": crop.width,
                        "height": crop.height,
                        "sha256": sha256_file(crop_path),
                        "split": split,
                    }
                )

            negative_created = False
            if eligible_for_negative(image_id, args.negative_ratio):
                crop_box = background_crop(width, height, boxes)
                if crop_box is not None:
                    negative_path = (
                        args.output_root / "images" / split / f"bower_bg_{image_id}.jpg"
                    )
                    rgb.crop(crop_box).save(
                        negative_path, format="JPEG", quality=95, optimize=True
                    )
                    negative_label = (
                        args.output_root / "labels" / split / f"bower_bg_{image_id}.txt"
                    )
                    negative_label.parent.mkdir(parents=True, exist_ok=True)
                    negative_label.write_text("", encoding="utf-8")
                    negative_created = True

        label_path = args.output_root / "labels" / split / f"bower_{image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
        )
        local_manifest = [
            {
                "bower_image_id": image_id,
                "split": split,
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "width": width,
                "height": height,
                "box_count": len(label_lines),
                "materials": json.dumps(materials, ensure_ascii=False, sort_keys=True),
                "objects": json.dumps(objects, ensure_ascii=False, sort_keys=True),
                "negative_crop_created": negative_created,
                "sha256": digest,
                "source_url": DATASET_PAGE,
                "source_license": "MIT",
            }
        ]
        return local_manifest, local_classification

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {image_id: executor.submit(import_image, image_id) for image_id in image_ids}
        for number, image_id in enumerate(image_ids, start=1):
            try:
                local_manifest, local_classification = futures[image_id].result()
                manifest_rows.extend(local_manifest)
                classification_rows.extend(local_classification)
            except Exception as exc:
                failures.append({"bower_image_id": image_id, "error": str(exc)})
            if number % 50 == 0 or number == len(image_ids):
                print(
                    f"Processed {number}/{len(image_ids)}; successful={len(manifest_rows)}; "
                    f"failed={len(failures)}",
                    flush=True,
                )

    manifest_fields = [
        "bower_image_id",
        "split",
        "image_path",
        "label_path",
        "width",
        "height",
        "box_count",
        "materials",
        "objects",
        "negative_crop_created",
        "sha256",
        "source_url",
        "source_license",
    ]
    classification_fields = [
        "image_path",
        "source_dataset",
        "source_category",
        "target_class",
        "item_id",
        "width",
        "height",
        "sha256",
        "split",
    ]
    write_csv(args.manifest, manifest_rows, manifest_fields)
    write_csv(args.classification, classification_rows, classification_fields)

    if not args.realwaste.is_file():
        raise FileNotFoundError(args.realwaste)
    with args.realwaste.open("r", encoding="utf-8-sig", newline="") as csv_file:
        realwaste_rows = list(csv.DictReader(csv_file))
    write_csv(
        args.combined_classification,
        realwaste_rows + classification_rows,
        classification_fields,
    )

    failure_path = args.manifest.with_name("bower_import_failures.csv")
    if failures:
        write_csv(failure_path, failures, ["bower_image_id", "error"])
    elif failure_path.exists():
        failure_path.unlink()

    print(f"Manifest: {args.manifest}")
    print(f"Imported phone images: {len(manifest_rows)}")
    print(f"Imported boxes/crops: {len(classification_rows)}")
    print(
        "Background negatives: "
        f"{sum(str(row['negative_crop_created']).lower() == 'true' for row in manifest_rows)}"
    )
    print(f"Image splits: {dict(Counter(str(row['split']) for row in manifest_rows))}")
    print(
        "CNN classes: "
        f"{dict(Counter(str(row['target_class']) for row in classification_rows))}"
    )
    if failures:
        print(f"Failures: {len(failures)} ({failure_path})")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
