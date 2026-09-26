from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


CATEGORY_MAP = {
    "Cardboard": "paper",
    "Food Organics": "unknown",
    "Glass": "glass",
    "Metal": "metal",
    "Miscellaneous Trash": "unknown",
    "Paper": "paper",
    "Plastic": "plastic",
    "Textile Trash": "unknown",
    "Vegetation": "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import RealWaste into the five-class project layout")
    parser.add_argument("archive", type=Path, help="Path to realwaste.zip")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true", help="Replace already extracted images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    project_root = args.project_root.resolve()
    output_root = project_root / "data" / "raw" / "realwaste"
    annotation_path = project_root / "data" / "annotations" / "realwaste.csv"
    if not archive.is_file():
        raise FileNotFoundError(archive)

    records: list[dict[str, str | int]] = []
    invalid: list[str] = []
    hashes: Counter[str] = Counter()

    with zipfile.ZipFile(archive) as dataset_zip:
        for entry in dataset_zip.infolist():
            parts = Path(entry.filename).parts
            if len(parts) != 4 or parts[:2] != ("realwaste-main", "RealWaste"):
                continue
            source_category = parts[2]
            if source_category not in CATEGORY_MAP or not entry.filename.lower().endswith(".jpg"):
                continue

            content = dataset_zip.read(entry)
            digest = hashlib.sha256(content).hexdigest()
            decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                invalid.append(entry.filename)
                continue

            target_class = CATEGORY_MAP[source_category]
            safe_source = source_category.lower().replace(" ", "_")
            target_name = f"{safe_source}__{parts[3]}"
            target = output_root / target_class / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite or not target.exists():
                target.write_bytes(content)

            height, width = decoded.shape[:2]
            hashes[digest] += 1
            records.append(
                {
                    "image_path": target.relative_to(project_root).as_posix(),
                    "source_dataset": "RealWaste",
                    "source_category": source_category,
                    "target_class": target_class,
                    "item_id": target.stem,
                    "width": width,
                    "height": height,
                    "sha256": digest,
                }
            )

    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "source_dataset",
        "source_category",
        "target_class",
        "item_id",
        "width",
        "height",
        "sha256",
    ]
    with annotation_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    target_counts = Counter(str(record["target_class"]) for record in records)
    source_counts = Counter(str(record["source_category"]) for record in records)
    duplicate_groups = sum(1 for count in hashes.values() if count > 1)
    print(f"Imported {len(records)} images to {output_root}")
    print(f"Wrote annotations to {annotation_path}")
    print(f"Target counts: {dict(sorted(target_counts.items()))}")
    print(f"Source counts: {dict(sorted(source_counts.items()))}")
    print(f"Invalid images: {len(invalid)}")
    print(f"Duplicate hash groups: {duplicate_groups}")
    if invalid:
        print("Invalid entries:")
        for value in invalid:
            print(f"- {value}")


if __name__ == "__main__":
    main()
