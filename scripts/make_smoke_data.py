"""Create a tiny synthetic dataset in RoofD's real source/mask layout.

This is only for pipeline smoke tests. It is not training data for the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data import CLASS_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="/tmp/roofd_smoke")
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--test", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def make_split(base: Path, split: str, count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    source_dir = base / "dataset" / "augmented_dataset" / split / "source"
    mask_dir = base / "dataset" / "augmented_dataset" / split / "mask"
    source_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = rng.integers(40, 220, (576, 1024, 3), dtype=np.uint8)
        mask = np.zeros_like(image)
        if index % 4 != 0:
            for offset in range(1 + index % 3):
                class_index = 1 + ((index + 2 * offset) % 5)
                x1 = int(rng.integers(0, 800))
                y1 = int(rng.integers(0, 400))
                x2 = x1 + int(rng.integers(50, 180))
                y2 = y1 + int(rng.integers(40, 120))
                color = CLASS_COLORS[class_index].tolist()[::-1]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
        cv2.imwrite(str(source_dir / f"{index:04d}.png"), image)
        cv2.imwrite(str(mask_dir / f"{index:04d}.png"), mask)


def main() -> None:
    args = parse_args()
    root = Path(args.dest)
    make_split(root, "train", args.train, args.seed)
    make_split(root, "test", args.test, args.seed + 1)
    print(f"Synthetic RoofD-format data written to {root}")
    print(
        f"Smoke train: python -m src.train_vit --smoke --limit 32 --data-root {root}"
    )


if __name__ == "__main__":
    main()
