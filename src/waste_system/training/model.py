from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from waste_system.training.dataset import CLASS_NAMES


def build_model(*, pretrained: bool) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    return model


def freeze_features(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def unfreeze_tail(model: nn.Module, blocks: int = 2) -> None:
    for block in model.features[-blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def load_checkpoint(path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes")
    if classes != CLASS_NAMES:
        raise ValueError(f"Checkpoint classes {classes!r} do not match {CLASS_NAMES!r}")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model
