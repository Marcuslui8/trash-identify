from __future__ import annotations

import argparse
from pathlib import Path

import torch

from waste_system.training.model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a classifier checkpoint to TorchScript")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/efficientnet_b0_best.pth"))
    parser.add_argument("--output", type=Path, default=Path("models/waste_classifier.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model = load_checkpoint(args.checkpoint, torch.device("cpu"))
    model.eval()
    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    with torch.inference_mode():
        output = traced(example)
    if tuple(output.shape) != (1, 5):
        raise RuntimeError(f"Unexpected exported output shape: {tuple(output.shape)}")
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
