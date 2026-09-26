from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render random YOLO bounding-box previews")
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/detection"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/detection_preview"))
    parser.add_argument(
        "--max-size", type=int, default=1600, help="Maximum preview width or height"
    )
    return parser.parse_args()


def find_image(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    label_directory = args.root / "labels" / args.split
    image_directory = args.root / "images" / args.split
    labels = sorted(label_directory.glob("*.txt"))
    if not labels:
        raise RuntimeError(f"No labels found in {label_directory}")
    selected = random.Random(args.seed).sample(labels, min(args.count, len(labels)))
    output_directory = args.output / args.split
    output_directory.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for label_path in selected:
        image_path = find_image(image_directory, label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"No image for {label_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]
        if max(height, width) > args.max_size:
            scale = args.max_size / max(height, width)
            width = round(width * scale)
            height = round(height * scale)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, center_x, center_y, box_width, box_height = line.split()
            center_x, center_y, box_width, box_height = map(
                float, (center_x, center_y, box_width, box_height)
            )
            x1 = round((center_x - box_width / 2) * width)
            y1 = round((center_y - box_height / 2) * height)
            x2 = round((center_x + box_width / 2) * width)
            y2 = round((center_y + box_height / 2) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 120), 3)
            cv2.putText(
                image,
                f"waste ({class_id})",
                (max(0, x1), max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 120),
                2,
                cv2.LINE_AA,
            )
        destination = output_directory / f"{label_path.stem}.jpg"
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError(f"Failed to write {destination}")
        rendered += 1
    print(f"Rendered {rendered} previews to {output_directory}")


if __name__ == "__main__":
    main()
