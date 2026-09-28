from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO detector on an independent split")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("reports/detection"))
    parser.add_argument("--name", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    data = args.data.resolve()
    project = args.project.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not data.is_file():
        raise FileNotFoundError(data)

    # Ultralytics calls the validation directory "val", while this project uses
    # the clearer name "validation" in its CLI and dataset folders.
    split = "val" if args.split == "validation" else args.split
    metrics = YOLO(str(weights)).val(
        data=str(data),
        split=split,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        project=str(project),
        name=args.name,
        plots=True,
    )
    summary = {
        "weights": str(weights),
        "split": args.split,
        "positive_images": int(sum(metrics.nt_per_image)),
        "instances": int(sum(metrics.nt_per_class)),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
    }
    output = Path(metrics.save_dir) / "metrics.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
