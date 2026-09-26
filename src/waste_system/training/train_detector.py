from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the one-class YOLO waste detector")
    parser.add_argument("--data", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--base-model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detection")
    parser.add_argument("--name", default="waste_detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: python -m pip install -e '.[detection]'"
        ) from exc
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    model = YOLO(args.base_model)
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        patience=15,
        seed=42,
    )
    print(result)
    print(
        "After validation, copy runs/detection/waste_detector/weights/best.pt "
        "to models/waste_detector.pt"
    )


if __name__ == "__main__":
    main()
