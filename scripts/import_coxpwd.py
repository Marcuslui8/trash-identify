from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


DATASET_PAGE = "https://data.mendeley.com/datasets/bdzg4tjy63/3"
DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/bdzg4tjy63/files/"
    "4967ad1f-8844-4a50-a19f-7da25fa1528a/file_downloaded"
)
ARCHIVE_SHA256 = "777c04c6792fcb17ef5da62e9ffced4b0154269a66813818e3e1857bb384ba93"
SOURCE_SPLITS = {"train": "train", "valid": "validation", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import CoxPWD COCO boxes as one-class YOLO waste labels"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/coxpwd/coxpwd_v3.coco.zip"),
        help="CoxPWD v3 ZIP downloaded from Mendeley Data",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the 382 MB official archive when --archive is missing",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/detection"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/annotations/coxpwd_import.csv")
    )
    parser.add_argument("--limit", type=int, default=None, help="Import N images per split")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    downloaded = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "trash-identify/0.6 CoxPWD importer"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
        print(f"Resuming download from {downloaded / 1024 / 1024:.1f} MB ...", flush=True)
    request = urllib.request.Request(
        DOWNLOAD_URL, headers=headers
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        append = downloaded > 0 and response.status == 206
        if downloaded and not append:
            print("Server does not support resume; restarting download.", flush=True)
        with temporary.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def verify_archive(path: Path) -> None:
    actual = sha256_file(path)
    if actual.lower() != ARCHIVE_SHA256:
        raise ValueError(
            f"CoxPWD archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}"
        )


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


def find_annotations(archive: zipfile.ZipFile) -> dict[str, str]:
    found: dict[str, str] = {}
    for member in archive.namelist():
        path = PurePosixPath(member)
        if path.name != "_annotations.coco.json" or len(path.parts) < 2:
            continue
        source_split = path.parts[-2].lower()
        if source_split in SOURCE_SPLITS:
            if source_split in found:
                raise ValueError(f"Multiple annotation files found for split {source_split}")
            found[source_split] = member
    missing = set(SOURCE_SPLITS) - set(found)
    if missing:
        raise ValueError(f"Missing CoxPWD annotation files for: {sorted(missing)}")
    return found


def resolve_image_member(
    members: set[str], annotation_member: str, file_name: str
) -> str:
    relative = PurePosixPath(file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image path in COCO metadata: {file_name}")
    candidate = str(PurePosixPath(annotation_member).parent / relative)
    if candidate in members:
        return candidate
    matches = [name for name in members if PurePosixPath(name).name == relative.name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely locate image in archive: {file_name}")
    return matches[0]


def load_coco(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    dataset = json.loads(archive.read(member).decode("utf-8-sig"))
    required = {"images", "annotations", "categories"}
    if not required.issubset(dataset):
        raise ValueError(f"COCO metadata is missing: {sorted(required - set(dataset))}")
    return dataset


def import_split(
    archive: zipfile.ZipFile,
    annotation_member: str,
    source_split: str,
    output_root: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    target_split = SOURCE_SPLITS[source_split]
    dataset = load_coco(archive, annotation_member)
    images = sorted(dataset["images"], key=lambda item: int(item["id"]))
    if limit is not None:
        images = images[:limit]
    selected_ids = {int(item["id"]) for item in images}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in selected_ids:
            annotations_by_image[image_id].append(annotation)
    categories = {int(item["id"]): str(item["name"]) for item in dataset["categories"]}
    members = set(archive.namelist())
    rows: list[dict[str, Any]] = []

    for number, image in enumerate(images, start=1):
        image_id = int(image["id"])
        member = resolve_image_member(members, annotation_member, str(image["file_name"]))
        content = archive.read(member)
        with Image.open(io.BytesIO(content)) as source:
            actual_width, actual_height = source.size
            rgb = source.convert("RGB")
            stem = f"coxpwd_{source_split}_{image_id:06d}"
            image_path = output_root / "images" / target_split / f"{stem}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = image_path.with_suffix(".part.jpg")
            rgb.save(temporary, format="JPEG", quality=95, optimize=True)
            temporary.replace(image_path)

        annotation_width = int(image.get("width", actual_width))
        annotation_height = int(image.get("height", actual_height))
        if (actual_width, actual_height) != (annotation_width, annotation_height):
            raise ValueError(
                f"Image size mismatch for {member}: COCO={annotation_width}x{annotation_height}, "
                f"actual={actual_width}x{actual_height}"
            )

        label_lines: list[str] = []
        original_categories: Counter[str] = Counter()
        for annotation in annotations_by_image[image_id]:
            converted = coco_bbox_to_yolo(
                annotation["bbox"], annotation_width, annotation_height
            )
            if converted is None:
                continue
            original_categories[categories[int(annotation["category_id"])]] += 1
            label_lines.append("0 " + " ".join(f"{value:.8f}" for value in converted))

        label_path = output_root / "labels" / target_split / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
        )
        rows.append(
            {
                "coxpwd_image_id": image_id,
                "source_split": source_split,
                "split": target_split,
                "source_file": member,
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "box_count": len(label_lines),
                "source_categories": json.dumps(
                    original_categories, ensure_ascii=False, sort_keys=True
                ),
                "sha256": sha256_file(image_path),
                "source_dataset": "CoxPWD 2025 v3",
                "source_url": DATASET_PAGE,
                "source_license": "CC BY 4.0",
            }
        )
        if number % 250 == 0 or number == len(images):
            print(f"{source_split}: imported {number}/{len(images)}", flush=True)
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "coxpwd_image_id",
        "source_split",
        "split",
        "source_file",
        "image_path",
        "label_path",
        "box_count",
        "source_categories",
        "sha256",
        "source_dataset",
        "source_url",
        "source_license",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not args.archive.is_file():
        if not args.download:
            raise FileNotFoundError(
                f"Archive not found: {args.archive}. Download it from {DATASET_PAGE} "
                "or run again with --download."
            )
        print(f"Downloading official CoxPWD archive to {args.archive} ...", flush=True)
        download_archive(args.archive)
    print("Verifying CoxPWD archive SHA-256 ...", flush=True)
    verify_archive(args.archive)

    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(args.archive) as archive:
        annotation_members = find_annotations(archive)
        for source_split in SOURCE_SPLITS:
            rows.extend(
                import_split(
                    archive,
                    annotation_members[source_split],
                    source_split,
                    args.output_root,
                    args.limit,
                )
            )
    write_manifest(args.manifest, rows)
    print(f"Manifest: {args.manifest}")
    print(f"Imported images: {len(rows)}")
    print(f"Imported boxes: {sum(int(row['box_count']) for row in rows)}")
    print(f"Splits: {dict(Counter(str(row['split']) for row in rows))}")


if __name__ == "__main__":
    main()
