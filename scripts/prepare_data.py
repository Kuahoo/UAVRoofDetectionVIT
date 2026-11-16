"""Milestone 1: verify RoofD pairs, cache class statistics, and draw samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import (
    CLASS_COLORS,
    CLASS_NAMES,
    analyze_pairs,
    load_pairs,
    read_mask,
    read_rgb,
    save_split_manifest,
)
from src.inference import overlay_mask
from src.utils import ensure_dirs, load_config, resolve_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def draw_samples(pairs, path: Path, count: int = 4) -> None:
    step = max(1, len(pairs) // count)
    chosen = pairs[::step][:count]
    fig, axes = plt.subplots(3, len(chosen), figsize=(4 * len(chosen), 10), squeeze=False)
    for col, (image_path, mask_path) in enumerate(chosen):
        image = read_rgb(image_path)
        mask = read_mask(mask_path)
        axes[0, col].imshow(image)
        axes[0, col].set_title(image_path.name, fontsize=8)
        axes[1, col].imshow(CLASS_COLORS[mask])
        axes[2, col].imshow(overlay_mask(image, mask))
        for row in range(3):
            axes[row, col].axis("off")
    axes[0, 0].set_ylabel("Image")
    axes[1, 0].set_ylabel("Mask")
    axes[2, 0].set_ylabel("Overlay")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    ensure_dirs(cfg)

    splits = save_split_manifest(cfg)
    train_full = load_pairs(cfg, "train_full")
    test_pairs = load_pairs(cfg, "test")
    train_stats = analyze_pairs(
        train_full, cfg["paths"]["train_stats"], cfg["data"]["root"], force=args.force
    )
    test_stats = analyze_pairs(
        test_pairs, cfg["paths"]["test_stats"], cfg["data"]["root"], force=args.force
    )

    print("Split sizes:", {name: len(items) for name, items in splits.items()})
    print(f"{'class':12s} {'train images':>14s} {'train pixels':>16s} {'test images':>13s}")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:12s} {train_stats['image_counts'][i]:14d} "
            f"{train_stats['pixel_counts'][i]:16d} {test_stats['image_counts'][i]:13d}"
        )

    if args.verify_images:
        expected = (576, 1024, 3)
        bad = []
        for image_path, _ in train_full + test_pairs:
            image = read_rgb(image_path)
            if image.shape != expected:
                bad.append((str(image_path), image.shape))
        print(f"Image verification: {len(bad)} unexpected shapes")
        for item in bad[:10]:
            print("  ", item)

    draw_samples(
        train_full,
        resolve_path(cfg["paths"]["figures"]) / "dataset_samples.png",
    )
    save_json(
        {
            "splits": {name: len(items) for name, items in splits.items()},
            "class_names": CLASS_NAMES,
            "class_colors": CLASS_COLORS.tolist(),
        },
        resolve_path(cfg["paths"]["metrics"]) / "dataset_summary.json",
    )
    print("Wrote", resolve_path(cfg["paths"]["figures"]) / "dataset_samples.png")


if __name__ == "__main__":
    main()
