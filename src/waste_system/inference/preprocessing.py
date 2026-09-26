from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQuality:
    score: float
    brightness: float
    blur_variance: float
    issues: tuple[str, ...]


def decode_image(content: bytes) -> np.ndarray:
    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image")
    return image


def assess_quality(
    image: np.ndarray,
    *,
    blur_threshold: float,
    minimum_brightness: float,
    maximum_brightness: float,
) -> ImageQuality:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    issues: list[str] = []
    if blur_variance < blur_threshold:
        issues.append("blurred")
    if brightness < minimum_brightness:
        issues.append("too_dark")
    if brightness > maximum_brightness:
        issues.append("overexposed")

    penalties = {
        "blurred": min(0.45, (blur_threshold - blur_variance) / max(blur_threshold, 1) * 0.45),
        "too_dark": min(0.45, (minimum_brightness - brightness) / max(minimum_brightness, 1) * 0.45),
        "overexposed": min(
            0.45,
            (brightness - maximum_brightness) / max(255 - maximum_brightness, 1) * 0.45,
        ),
    }
    score = max(0.0, 1.0 - sum(penalties[issue] for issue in issues))
    return ImageQuality(
        score=round(score, 4),
        brightness=brightness,
        blur_variance=blur_variance,
        issues=tuple(issues),
    )


def prepare_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
