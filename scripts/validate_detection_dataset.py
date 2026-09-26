from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a one-class YOLO waste detection dataset")
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/detection"))
    return parser.parse_args()


def validate_label(path: Path) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            errors.append(f"{path}:{line_number}: expected 5 values")
            continue
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: values must be numeric")
            continue
        if class_id != 0:
            errors.append(f"{path}:{line_number}: only class 0 (waste) is supported")
        if any(value < 0 or value > 1 for value in coordinates):
            errors.append(f"{path}:{line_number}: normalized coordinates must be within 0..1")
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            errors.append(f"{path}:{line_number}: width and height must be positive")
    return errors


def main() -> None:
    root = parse_args().root.resolve()
    counts: Counter[str] = Counter()
    errors: list[str] = []
    for split in SPLITS:
        image_directory = root / "images" / split
        label_directory = root / "labels" / split
        if not image_directory.is_dir() or not label_directory.is_dir():
            errors.append(f"Missing images/labels directories for split: {split}")
            continue
        images = [path for path in image_directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
        counts[f"{split}_images"] = len(images)
        if split in {"train", "validation"} and not images:
            errors.append(f"No images found for required split: {split}")
        for image in images:
            label = label_directory / f"{image.stem}.txt"
            if not label.is_file():
                errors.append(f"Missing label: {label}")
                continue
            counts[f"{split}_boxes"] += len(label.read_text(encoding="utf-8").splitlines())
            errors.extend(validate_label(label))
        image_stems = {path.stem for path in images}
        for label in label_directory.glob("*.txt"):
            if label.stem not in image_stems:
                errors.append(f"Label has no matching image: {label}")

    print(f"Dataset: {root}")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:100]:
            print(f"- {error}")
        raise SystemExit(1)
    print("Detection dataset is valid")


if __name__ == "__main__":
    main()
