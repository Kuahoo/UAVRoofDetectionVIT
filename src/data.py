"""Shared RoofD data pipeline for ViT classification and U-Net segmentation.

The same RGB mask is decoded once into an integer class map.  The classifier
target is derived from the classes present in that map, so both models see
mathematically identical ground truth.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from src.utils import resolve_path, save_json, seed_worker

CLASS_NAMES = ["background", "hollows", "swellings", "folds", "patches", "breaks"]
# Multi-label classifiers (ViT, logistic) predict defect presence only.
# Background is nearly always present and is reserved for segmentation.
DEFECT_CLASS_NAMES = CLASS_NAMES[1:]
CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 255, 0],
        [153, 255, 204],
        [6, 89, 182],
        [0, 255, 255],
        [255, 193, 255],
    ],
    dtype=np.uint8,
)
_CLASS_COLORS_I32 = CLASS_COLORS.astype(np.int32)
# Off-palette RGB values seen in some RoofD masks (compression / edge bleed).
_REMAPPED_MASK_COLORS: set[tuple[int, int, int]] = set()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGE_DIR_NAMES = {"images", "image", "img", "imgs", "jpegimages", "source", "sources"}
MASK_DIR_NAMES = {"masks", "mask", "labels", "label", "gt", "ground_truth", "annotations"}
MASK_SUFFIXES = ("_mask", "_masks", "_label", "_labels", "_gt")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _list_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _canonical_stem(path: Path) -> str:
    stem = path.stem
    lowered = stem.lower()
    for suffix in MASK_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def get_group_id(image_path: str | Path, group_by: str = "tile") -> str:
    """RoofD source names are ``{aug_idx}_{flight_id}_{tile_idx}.png``.

    ``tile`` groups the four pre-augmented copies of one tile; ``flight``
    groups every tile captured for one roof/flight. Grouping prevents the
    augmented copies of a tile from leaking into both train and validation.
    """
    if group_by not in {"tile", "flight"}:
        raise ValueError(f"Unknown group_by: {group_by}")
    no_aug = re.sub(r"^\d{4}_", "", Path(image_path).stem)
    return re.sub(r"_\d{2}$", "", no_aug) if group_by == "flight" else no_aug


def _split_train_val(
    pairs: list[tuple[Path, Path]],
    val_fraction: float,
    seed: int,
    group_by: str,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    groups = [get_group_id(image, group_by) for image, _ in pairs]
    if len(set(groups)) < 2:
        return pairs, []
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(pairs, groups=groups))
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


def _find_named_dir(parent: Path, names: set[str]) -> Path | None:
    for child in sorted(parent.iterdir()):
        if child.is_dir() and child.name.lower() in names:
            return child
    for child in sorted(parent.rglob("*")):
        if child.is_dir() and child.name.lower() in names:
            return child
    return None


def _find_split_dir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_dir():
        return direct
    matches = [p for p in root.rglob(name) if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Split directory '{name}' not found under {root}")
    raise RuntimeError(f"Ambiguous split directory '{name}' under {root}: {matches}")


def discover_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    """Pair image and mask files, accepting the common RoofD layouts."""
    split_dir = resolve_path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    images_dir = _find_named_dir(split_dir, IMAGE_DIR_NAMES)
    masks_dir = _find_named_dir(split_dir, MASK_DIR_NAMES)
    if images_dir is None:
        images_dir = split_dir

    if masks_dir is not None and masks_dir.resolve() != images_dir.resolve():
        image_files = _list_files(images_dir)
        mask_files = _list_files(masks_dir)
    else:
        all_files = _list_files(images_dir)
        image_files = [
            path
            for path in all_files
            if not any(path.stem.lower().endswith(s) for s in MASK_SUFFIXES)
        ]
        mask_files = [
            path
            for path in all_files
            if any(path.stem.lower().endswith(s) for s in MASK_SUFFIXES)
        ]

    if not image_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

    masks_by_stem: dict[str, list[Path]] = {}
    for path in mask_files:
        masks_by_stem.setdefault(_canonical_stem(path), []).append(path)

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for image_path in image_files:
        matches = masks_by_stem.get(_canonical_stem(image_path), [])
        if len(matches) == 1:
            pairs.append((image_path, matches[0]))
        elif not matches:
            missing.append(image_path.name)
        else:
            raise RuntimeError(f"Ambiguous masks for {image_path}: {matches}")

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) have no matching mask in {split_dir}. "
            f"First missing: {missing[:5]}"
        )
    return pairs


def load_pairs(cfg: dict[str, Any], split: str, limit: int | None = None) -> list[tuple[Path, Path]]:
    """Return deterministic pairs for train, val, test, or train_full."""
    if limit is None:
        limit = cfg["data"].get("limit")
    root = resolve_path(cfg["data"]["root"])
    if split in {"train", "val", "train_full"}:
        split_dir = _find_split_dir(root, cfg["data"]["train_dir"])
    elif split == "test":
        split_dir = _find_split_dir(root, cfg["data"]["test_dir"])
    else:
        raise ValueError(f"Unknown split: {split}")

    pairs = discover_pairs(split_dir)
    if split == "test":
        return pairs[:limit] if limit else pairs

    rng = random.Random(cfg["seed"])
    rng.shuffle(pairs)
    if limit:
        pairs = pairs[:limit]
    if split == "train_full":
        return pairs

    train_pairs, val_pairs = _split_train_val(
        pairs,
        float(cfg["data"]["val_fraction"]),
        int(cfg["seed"]),
        str(cfg["data"].get("group_by", "tile")),
    )
    return train_pairs if split == "train" else val_pairs


def save_split_manifest(cfg: dict[str, Any]) -> dict[str, list[str]]:
    root = resolve_path(cfg["data"]["root"])
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = [str(img.relative_to(root)) for img, _ in load_pairs(cfg, split)]
    save_json(
        {
            "seed": cfg["seed"],
            "val_fraction": cfg["data"]["val_fraction"],
            "group_by": cfg["data"].get("group_by", "tile"),
            "splits": splits,
        },
        cfg["paths"]["splits"],
    )
    return splits


# ---------------------------------------------------------------------------
# Mask decoding and targets
# ---------------------------------------------------------------------------


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Could not read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def nearest_class_index(color: tuple[int, int, int]) -> int:
    """Map an RGB triple to the closest RoofD class color (L2)."""
    diff = _CLASS_COLORS_I32 - np.asarray(color, dtype=np.int32)
    return int(np.argmin(np.sum(diff * diff, axis=1)))


def rgb_to_index(mask_rgb: np.ndarray, source: str | Path | None = None) -> np.ndarray:
    """Decode an RGB RoofD mask to class indices 0..5.

    Exact palette hits are preferred. Any off-palette colors (common at
    compressed / anti-aliased mask edges) are remapped to the nearest class
    color instead of failing the whole dataset load.
    """
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] < 3:
        raise ValueError(f"Expected an RGB mask, got shape {mask_rgb.shape}")
    packed = (
        (mask_rgb[:, :, 0].astype(np.uint32) << 16)
        | (mask_rgb[:, :, 1].astype(np.uint32) << 8)
        | mask_rgb[:, :, 2].astype(np.uint32)
    )
    unique, inverse = np.unique(packed, return_inverse=True)
    exact = {tuple(c): i for i, c in enumerate(CLASS_COLORS.tolist())}
    lookup: dict[int, int] = {}
    for value in unique:
        color = ((int(value) >> 16) & 255, (int(value) >> 8) & 255, int(value) & 255)
        if color in exact:
            lookup[int(value)] = exact[color]
            continue
        class_id = nearest_class_index(color)
        lookup[int(value)] = class_id
        if color not in _REMAPPED_MASK_COLORS:
            _REMAPPED_MASK_COLORS.add(color)
            where = f" in {source}" if source is not None else ""
            print(
                f"Warning: off-palette mask color {color}{where} "
                f"→ nearest class '{CLASS_NAMES[class_id]}' {tuple(CLASS_COLORS[class_id])}"
            )
    mapped = np.array([lookup[int(v)] for v in unique], dtype=np.int64)[inverse]
    return mapped.reshape(mask_rgb.shape[:2])


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f"Could not read mask: {path}")
    if mask.ndim == 2:
        if mask.max() >= len(CLASS_NAMES):
            raise ValueError(f"Single-channel mask {path} has values outside 0..5")
        return mask.astype(np.int64)
    if mask.shape[2] == 4:
        mask = mask[:, :, :3]
    return rgb_to_index(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), source=path)


def mask_to_labels(mask: np.ndarray) -> np.ndarray:
    """Length-5 multi-label vector over defect classes (excludes background)."""
    labels = np.zeros(len(DEFECT_CLASS_NAMES), dtype=np.float32)
    present = set(int(v) for v in np.unique(mask))
    for defect_id, _name in enumerate(DEFECT_CLASS_NAMES, start=1):
        if defect_id in present:
            labels[defect_id - 1] = 1.0
    return labels


# ---------------------------------------------------------------------------
# Image transformations
# ---------------------------------------------------------------------------


def resize_rgb(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    height, width = (size, size) if isinstance(size, int) else (size[0], size[1])
    shrinking = image.shape[0] > height or image.shape[1] > width
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_mask(mask: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    height, width = (size, size) if isinstance(size, int) else (size[0], size[1])
    resized = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(np.int64)


def degrade_image(image: np.ndarray, resolution_wh: tuple[int, int]) -> np.ndarray:
    """Simulate a lower-resolution UAV capture, then restore display size."""
    width, height = resolution_wh
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def to_tensor(image: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    array = image.astype(np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _draw_augmentation() -> tuple[bool, bool, int]:
    hflip = bool(np.random.random() < 0.5)
    vflip = bool(np.random.random() < 0.5)
    k = int(np.random.randint(1, 4)) if np.random.random() < 0.5 else 0
    return hflip, vflip, k


def _apply_augmentation(array: np.ndarray, flags: tuple[bool, bool, int]) -> np.ndarray:
    hflip, vflip, k = flags
    if hflip:
        array = array[:, ::-1]
    if vflip:
        array = array[::-1]
    if k:
        array = np.rot90(array, k)
    return np.ascontiguousarray(array)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RoofDDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        task: str,
        image_size: int,
        train: bool,
        mean: Iterable[float],
        std: Iterable[float],
        labels: dict[str, list[int]] | None = None,
        degrade_resolution: tuple[int, int] | None = None,
    ) -> None:
        if task not in {"classification", "segmentation"}:
            raise ValueError(f"Unknown task: {task}")
        self.pairs = pairs
        self.task = task
        self.image_size = image_size
        self.train = train
        self.mean = list(mean)
        self.std = list(std)
        self.labels = labels
        self.degrade_resolution = degrade_resolution

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        image = read_rgb(image_path)

        cached = self.labels.get(str(image_path)) if self.labels else None
        mask = None if cached is not None and self.task == "classification" else read_mask(mask_path)

        if self.train:
            flags = _draw_augmentation()
            image = _apply_augmentation(image, flags)
            if mask is not None:
                mask = _apply_augmentation(mask, flags)

        labels = np.asarray(cached, dtype=np.float32) if cached is not None else mask_to_labels(mask)
        if self.degrade_resolution is not None:
            image = degrade_image(image, self.degrade_resolution)

        image = resize_rgb(image, self.image_size)
        sample: dict[str, Any] = {
            "image": to_tensor(image, self.mean, self.std),
            "labels": torch.from_numpy(labels),
            "path": str(image_path),
        }
        if self.task == "segmentation":
            sample["mask"] = torch.from_numpy(resize_mask(mask, self.image_size)).long()
        return sample


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last and len(dataset) >= batch_size,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Dataset statistics (cached)
# ---------------------------------------------------------------------------


def analyze_pairs(
    pairs: list[tuple[Path, Path]],
    cache_path: str | Path,
    root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Count images/pixels per class and cache per-image presence labels."""
    cache_path = resolve_path(cache_path)
    root = resolve_path(root)
    if cache_path.exists() and not force:
        import json

        with cache_path.open() as f:
            cached = json.load(f)
        sample_labels = next(iter(cached.get("labels", {}).values()), None)
        labels_ok = (
            sample_labels is None or len(sample_labels) == len(DEFECT_CLASS_NAMES)
        )
        if (
            cached.get("root") == str(root)
            and cached.get("num_pairs") == len(pairs)
            and labels_ok
        ):
            return cached

    image_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    pixel_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    labels: dict[str, list[int]] = {}
    for image_path, mask_path in pairs:
        mask = read_mask(mask_path)
        counts = np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))
        pixel_counts += counts
        present = (counts > 0).astype(np.int64)
        image_counts += present
        # Cache classifier targets as defect-only (length 5).
        labels[str(image_path)] = present[1:].tolist()

    stats = {
        "root": str(root),
        "num_pairs": len(pairs),
        "image_counts": image_counts.tolist(),
        "pixel_counts": pixel_counts.tolist(),
        "class_names": CLASS_NAMES,
        "defect_class_names": DEFECT_CLASS_NAMES,
        "labels": labels,
    }
    save_json(stats, cache_path)
    return stats


def compute_pos_weight(stats: dict[str, Any], max_weight: float = 10.0) -> torch.Tensor:
    """Dampened positive weights for the five defect classes.

    Raw ``negatives / positives`` reaches ~50 for ``breaks`` and destabilizes
    fine-tuning; the square root keeps rare classes important without
    producing gradient spikes.
    """
    counts = np.asarray(stats["image_counts"], dtype=np.float64)[1:]
    negatives = stats["num_pairs"] - counts
    raw = np.where(counts > 0, negatives / np.maximum(counts, 1), 1.0)
    weights = np.minimum(np.sqrt(raw), max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights(stats: dict[str, Any]) -> torch.Tensor:
    """Log-smoothed pixel weights that emphasize rare classes."""
    counts = np.asarray(stats["pixel_counts"], dtype=np.float64)
    total = counts.sum()
    frequencies = counts / total if total > 0 else np.ones_like(counts) / len(counts)
    weights = 1.0 / np.log(1.02 + frequencies)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
