"""Handcrafted features for the course baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from skimage.feature import hog

from src.data import degrade_image, mask_to_labels, read_mask, read_rgb, resize_rgb


def extract_features(image_rgb: np.ndarray, size: int = 64) -> np.ndarray:
    small = resize_rgb(image_rgb, size)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    pixels = small.reshape(-1, 3).astype(np.float32) / 255.0
    color_stats = np.concatenate([pixels.mean(axis=0), pixels.var(axis=0)])
    return np.concatenate([hog_features, color_stats]).astype(np.float32)


def extract_split(
    pairs: Sequence[tuple[Path, Path]],
    image_size: int,
    labels: dict[str, list[int]] | None = None,
    degrade_resolution: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for image_path, mask_path in pairs:
        image = read_rgb(image_path)
        if degrade_resolution is not None:
            image = degrade_image(image, degrade_resolution)
        features.append(extract_features(image, image_size))
        cached = labels.get(str(image_path)) if labels else None
        if cached is None:
            target = mask_to_labels(read_mask(mask_path))
        else:
            target = np.asarray(cached, dtype=np.float32)
        targets.append(target)
    return np.stack(features), np.stack(targets)
