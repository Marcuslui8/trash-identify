from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    configured_path = os.getenv("WASTE_CONFIG_PATH", "configs/settings.json")
    path = Path(configured_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as settings_file:
        settings = json.load(settings_file)
    if model_path := os.getenv("WASTE_MODEL_PATH"):
        settings["model"]["path"] = model_path
    if detector_path := os.getenv("WASTE_DETECTOR_PATH"):
        settings["detector"]["path"] = detector_path
    return settings


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
