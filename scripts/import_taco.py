from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps


OFFICIAL_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official TACO images and convert COCO boxes to one-class YOLO labels"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/taco/annotations.json"),
        help="Cached official COCO annotations; downloaded when missing",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/detection"))
    parser.add_argument("--manifest", type=Path, default=Path("data/annotations/taco_import.csv"))
    parser.add_argument("--source-copy", type=Path, default=Path("data/taco/annotations_source.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Import only N images for a smoke test")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--workers", type=int, default=8, help="Concurrent image downloads (default: 8)"
    )
    parser.add_argument("--prefer-original", action="store_true")
    parser.add_argument(
        "--local-image-root",
        type=Path,
        default=None,
        help="Optional extracted TACO data folder containing batch_1, batch_2, ...",
    )
    return parser.parse_args()


def download_bytes(url: str, *, timeout: int, retries: int) -> bytes:
    last_error: Exception | None = None
    request = urllib.request.Request(url, headers={"User-Agent": "trash-identify/0.4 TACO importer"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # urllib exposes several transport exception types
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def load_annotations(path: Path, *, timeout: int, retries: int) -> dict[str, Any]:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        content = download_bytes(OFFICIAL_ANNOTATIONS_URL, timeout=timeout, retries=retries)
        path.write_bytes(content)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    required = {"images", "annotations", "categories"}
    if not required.issubset(dataset):
        raise ValueError(f"TACO annotations are missing keys: {sorted(required - set(dataset))}")
    return dataset


def assign_batch_splits(images: list[dict[str, Any]], seed: int) -> dict[str, str]:
    batches = sorted({PurePosixPath(image["file_name"]).parts[0] for image in images})
    random.Random(seed).shuffle(batches)
    train_count = max(1, math.ceil(len(batches) * 0.70))
    validation_count = max(1, round(len(batches) * 0.15))
    if train_count + validation_count >= len(batches):
        validation_count = max(1, len(batches) - train_count - 1)
    mapping: dict[str, str] = {}
    for index, batch in enumerate(batches):
        if index < train_count:
            mapping[batch] = "train"
        elif index < train_count + validation_count:
            mapping[batch] = "validation"
        else:
            mapping[batch] = "test"
    return mapping


def coco_bbox_to_yolo(
    bbox: list[float], image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    x, y, width, height = (float(value) for value in bbox)
    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + width))
    y2 = max(0.0, min(float(image_height), y + height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    return (
        ((x1 + x2) / 2) / image_width,
        ((y1 + y2) / 2) / image_height,
        clipped_width / image_width,
        clipped_height / image_height,
    )


def choose_url(image: dict[str, Any], prefer_original: bool) -> str:
    resized = image.get("flickr_640_url")
    original = image.get("flickr_url")
    choices = (original, resized) if prefer_original else (resized, original)
    for value in choices:
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value
    raise ValueError(f"Image {image.get('id')} has no downloadable URL")


def write_jpeg(content: bytes, destination: Path, expected_width: int, expected_height: int) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(content)) as source:
        source = ImageOps.exif_transpose(source)
        actual_width, actual_height = source.size
        expected_ratio = expected_width / expected_height
        actual_ratio = actual_width / actual_height
        if abs(expected_ratio - actual_ratio) / expected_ratio > 0.03:
            raise ValueError(
                f"aspect ratio mismatch: annotations={expected_width}x{expected_height}, "
                f"download={actual_width}x{actual_height}"
            )
        rgb = source.convert("RGB")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".part.jpg")
        rgb.save(temporary, format="JPEG", quality=92, optimize=True)
        temporary.replace(destination)
    return actual_width, actual_height, hashlib.sha256(destination.read_bytes()).hexdigest()


def ensure_no_conflicting_split(output_root: Path, stem: str, expected_split: str) -> None:
    for split in SPLITS:
        if split == expected_split:
            continue
        for relative in (Path("images") / split / f"{stem}.jpg", Path("labels") / split / f"{stem}.txt"):
            if (output_root / relative).exists():
                raise RuntimeError(
                    f"{stem} already exists in split {split}; keep seed=42 or remove the old TACO import first"
                )


def import_image(
    image: dict[str, Any],
    *,
    output_root: Path,
    batch_splits: dict[str, str],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    category_names: dict[int, str],
    timeout: int,
    retries: int,
    prefer_original: bool,
    local_image_root: Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    image_id = int(image["id"])
    source_file = PurePosixPath(image["file_name"])
    batch = source_file.parts[0]
    split = batch_splits[batch]
    stem = f"taco_{image_id:06d}"
    image_path = output_root / "images" / split / f"{stem}.jpg"
    label_path = output_root / "labels" / split / f"{stem}.txt"
    url = ""
    try:
        ensure_no_conflicting_split(output_root, stem, split)
        url = choose_url(image, prefer_original)
        if image_path.is_file():
            with Image.open(image_path) as existing:
                actual_width, actual_height = existing.size
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        else:
            local_source = local_image_root / Path(*source_file.parts) if local_image_root else None
            if local_source is not None and local_source.is_file():
                content = local_source.read_bytes()
            else:
                content = download_bytes(url, timeout=timeout, retries=retries)
            actual_width, actual_height, digest = write_jpeg(
                content,
                image_path,
                int(image["width"]),
                int(image["height"]),
            )

        label_lines: list[str] = []
        original_categories: Counter[str] = Counter()
        for annotation in annotations_by_image[image_id]:
            converted = coco_bbox_to_yolo(
                annotation["bbox"], int(image["width"]), int(image["height"])
            )
            if converted is None:
                continue
            original_categories[category_names[int(annotation["category_id"])]] += 1
            label_lines.append("0 " + " ".join(f"{value:.8f}" for value in converted))
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        return (
            {
                "taco_image_id": image_id,
                "split": split,
                "batch": batch,
                "source_file": image["file_name"],
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "source_url": url,
                "source_license": image.get("license") or "UNSPECIFIED",
                "annotation_width": image["width"],
                "annotation_height": image["height"],
                "download_width": actual_width,
                "download_height": actual_height,
                "box_count": len(label_lines),
                "source_categories": json.dumps(
                    original_categories, ensure_ascii=False, sort_keys=True
                ),
                "sha256": digest,
            },
            None,
        )
    except Exception as exc:
        return None, {"taco_image_id": image_id, "source_url": url, "error": str(exc)}


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    dataset = load_annotations(args.annotations, timeout=args.timeout, retries=args.retries)
    images = sorted(dataset["images"], key=lambda item: int(item["id"]))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        images = images[: args.limit]

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    category_names = {int(item["id"]): item["name"] for item in dataset["categories"]}
    batch_splits = assign_batch_splits(dataset["images"], args.seed)

    args.source_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.annotations, args.source_copy)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def worker(image: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        return import_image(
            image,
            output_root=args.output_root,
            batch_splits=batch_splits,
            annotations_by_image=annotations_by_image,
            category_names=category_names,
            timeout=args.timeout,
            retries=args.retries,
            prefer_original=args.prefer_original,
            local_image_root=args.local_image_root,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for number, (row, failure) in enumerate(executor.map(worker, images), start=1):
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)
            if number % 50 == 0 or number == len(images):
                print(
                    f"Processed {number}/{len(images)}; successful={len(rows)}; "
                    f"failed={len(failures)}",
                    flush=True,
                )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "taco_image_id",
        "split",
        "batch",
        "source_file",
        "image_path",
        "label_path",
        "source_url",
        "source_license",
        "annotation_width",
        "annotation_height",
        "download_width",
        "download_height",
        "box_count",
        "source_categories",
        "sha256",
    ]
    with args.manifest.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failure_path = args.manifest.with_name("taco_import_failures.csv")
    if failures:
        with failure_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["taco_image_id", "source_url", "error"])
            writer.writeheader()
            writer.writerows(failures)
    elif failure_path.exists():
        failure_path.unlink()

    print(f"Manifest: {args.manifest}")
    print(f"Imported images: {len(rows)}")
    print(f"Imported boxes: {sum(int(row['box_count']) for row in rows)}")
    print(f"Splits: {dict(Counter(str(row['split']) for row in rows))}")
    print(f"Licenses: {dict(Counter(str(row['source_license']) for row in rows))}")
    if failures:
        print(f"Failures: {len(failures)} ({failure_path})")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
