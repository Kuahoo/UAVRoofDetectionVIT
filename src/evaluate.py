"""Evaluate trained checkpoints and run the UAV resolution-degradation study."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch

from src.data import CLASS_NAMES, RoofDDataset, load_pairs, make_loader
from src.features import extract_split
from src.inference import (
    build_classifier,
    build_unet,
    fuse_probabilities,
    logistic_positive_proba,
    unet_presence_probs,
)
from src.metrics import (
    classification_metrics_from_probs,
    plot_classification_metrics,
    plot_classification_samples,
    plot_confusion_counts,
    plot_degradation,
    plot_segmentation_metrics,
    plot_segmentation_samples,
    save_degradation_table,
    save_per_class_table,
    segmentation_metrics,
)
from src.utils import (
    amp_settings,
    apply_smoke,
    ensure_dirs,
    get_device,
    load_config,
    resolve_path,
    save_json,
    set_seed,
)

CHECKPOINT_FILES = {
    "vit": "vit_best.pt",
    "unet": "unet_best.pt",
    "logistic": "logistic.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--model",
        choices=["vit", "unet", "logistic", "ensemble", "all"],
        default="all",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint")
    parser.add_argument("--degradation", action="store_true")
    return parser.parse_args()


def collect_vit(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    model.eval()
    start = time.perf_counter()
    probs_list, labels_list = [], []
    sample_images = sample_labels = None
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            labels = batch["labels"].numpy()
            probs_list.append(probs)
            labels_list.append(labels)
            if sample_images is None:
                sample_images = batch["image"].numpy()
                sample_labels = labels
    elapsed = time.perf_counter() - start
    return (
        np.concatenate(probs_list),
        np.concatenate(labels_list),
        elapsed,
        sample_images,
        sample_labels,
    )


def collect_unet(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    start = time.perf_counter()
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    sample_images = sample_masks = sample_preds = None
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy()
            targets = batch["mask"].numpy()
            indices = targets.ravel() * len(CLASS_NAMES) + predictions.ravel()
            confusion += np.bincount(
                indices, minlength=len(CLASS_NAMES) ** 2
            ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
            if sample_images is None:
                sample_images = batch["image"].numpy()
                sample_masks = targets
                sample_preds = predictions
    elapsed = time.perf_counter() - start
    return confusion, elapsed, sample_images, sample_masks, sample_preds


def collect_unet_presence(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    min_pixels: Sequence[int],
) -> tuple[np.ndarray, float]:
    """Pool each softmax mask into length-5 image-level presence probabilities."""
    model.eval()
    start = time.perf_counter()
    presence = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            presence.extend(unet_presence_probs(sample, min_pixels) for sample in probs)
    elapsed = time.perf_counter() - start
    return np.stack(presence), elapsed


def _classification_loader(
    cfg: dict[str, Any],
    pairs: Sequence[tuple[Path, Path]],
    image_size: int,
    degrade_resolution: tuple[int, int] | None,
) -> torch.utils.data.DataLoader:
    dataset = RoofDDataset(
        list(pairs),
        "classification",
        image_size,
        False,
        cfg["data"]["mean"],
        cfg["data"]["std"],
        degrade_resolution=degrade_resolution,
    )
    return make_loader(
        dataset,
        int(cfg["training"]["batch_size_classifier"]),
        False,
        int(cfg["data"]["num_workers"]),
        cfg["seed"],
    )


def _segmentation_loader(
    cfg: dict[str, Any],
    pairs: Sequence[tuple[Path, Path]],
    image_size: int,
    degrade_resolution: tuple[int, int] | None,
) -> torch.utils.data.DataLoader:
    dataset = RoofDDataset(
        list(pairs),
        "segmentation",
        image_size,
        False,
        cfg["data"]["mean"],
        cfg["data"]["std"],
        degrade_resolution=degrade_resolution,
    )
    return make_loader(
        dataset,
        int(cfg["training"]["batch_size_unet"]),
        False,
        int(cfg["data"]["num_workers"]),
        cfg["seed"],
    )


def save_classification_reports(
    name: str,
    split: str,
    metrics: dict[str, Any],
    probs: np.ndarray,
    labels: np.ndarray,
    samples: tuple[np.ndarray | None, np.ndarray | None],
    cfg: dict[str, Any],
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    figures_dir = resolve_path(cfg["paths"]["figures"])
    np.save(metrics_dir / f"{name}_{split}_probs.npy", probs)
    np.save(metrics_dir / f"{name}_{split}_labels.npy", labels)
    save_json(metrics, metrics_dir / f"{name}_{split}.json")
    save_per_class_table(metrics, metrics_dir / f"{name}_{split}_per_class.csv")
    plot_classification_metrics(metrics, figures_dir / f"{name}_{split}_metrics.png")
    plot_confusion_counts(metrics, figures_dir / f"{name}_{split}_confusion.png")
    sample_images, sample_labels = samples
    if sample_images is not None:
        plot_classification_samples(
            sample_images,
            sample_labels,
            probs[: len(sample_labels)],
            figures_dir / f"{name}_{split}_samples.png",
            mean,
            std,
        )


def save_segmentation_reports(
    name: str,
    split: str,
    metrics: dict[str, Any],
    confusion: np.ndarray,
    samples: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    cfg: dict[str, Any],
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    figures_dir = resolve_path(cfg["paths"]["figures"])
    np.save(metrics_dir / f"{name}_{split}_confusion.npy", confusion)
    save_json(metrics, metrics_dir / f"{name}_{split}.json")
    save_per_class_table(metrics, metrics_dir / f"{name}_{split}_per_class.csv")
    plot_segmentation_metrics(metrics, figures_dir / f"{name}_{split}_metrics.png")
    sample_images, sample_masks, sample_preds = samples
    if sample_images is not None:
        plot_segmentation_samples(
            sample_images,
            sample_masks,
            sample_preds,
            figures_dir / f"{name}_{split}_samples.png",
            mean,
            std,
        )


def evaluate_vit(
    cfg: dict[str, Any],
    checkpoint: str,
    split: str,
    device: torch.device,
    degradation: bool,
) -> None:
    model, ckpt = build_classifier(checkpoint, device)
    mean = ckpt.get("mean", cfg["data"]["mean"])
    std = ckpt.get("std", cfg["data"]["std"])
    image_size = int(ckpt.get("image_size", cfg["data"]["image_size_classifier"]))
    thresholds = ckpt.get("thresholds", 0.5)
    pairs = load_pairs(cfg, split)
    use_amp, _ = amp_settings(device, bool(cfg["training"]["amp"]))

    def run(degrade_resolution: tuple[int, int] | None) -> tuple[dict[str, Any], np.ndarray, np.ndarray, tuple[Any, Any]]:
        loader = _classification_loader(cfg, pairs, image_size, degrade_resolution)
        probs, labels, elapsed, images, true_labels = collect_vit(model, loader, device, use_amp)
        metrics = classification_metrics_from_probs(probs, labels, threshold=thresholds)
        metrics["thresholds"] = thresholds
        metrics["inference_ms_per_image"] = elapsed / len(pairs) * 1000
        return metrics, probs, labels, (images, true_labels)

    if not degradation:
        metrics, probs, labels, samples = run(None)
        save_classification_reports("vit", split, metrics, probs, labels, samples, cfg, mean, std)
        print(f"ViT {split}: macro-F1 {metrics['macro_f1']:.4f}")
        return

    results = []
    metrics, _, _, _ = run(None)
    results.append({"label": "native", "resolution": None, "metrics": metrics})
    for resolution in cfg["degradation"]["resolutions"]:
        metrics, _, _, _ = run(tuple(resolution))
        results.append(
            {"label": f"{resolution[0]}x{resolution[1]}", "resolution": resolution, "metrics": metrics}
        )
    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    save_json(results, metrics_dir / f"degradation_vit_{split}.json")
    save_degradation_table(results, metrics_dir / f"degradation_vit_{split}.csv", "classification")
    plot_degradation(
        results,
        resolve_path(cfg["paths"]["figures"]) / f"degradation_vit_{split}.png",
        "classification",
    )
    print("ViT degradation:", {r["label"]: round(r["metrics"]["macro_f1"], 4) for r in results})


def evaluate_unet(
    cfg: dict[str, Any],
    checkpoint: str,
    split: str,
    device: torch.device,
    degradation: bool,
) -> None:
    model, ckpt = build_unet(checkpoint, device)
    mean = ckpt.get("mean", cfg["data"]["mean"])
    std = ckpt.get("std", cfg["data"]["std"])
    image_size = int(ckpt.get("image_size", cfg["data"]["image_size_unet"]))
    pairs = load_pairs(cfg, split)
    use_amp, _ = amp_settings(device, bool(cfg["training"]["amp"]))

    def run(degrade_resolution: tuple[int, int] | None) -> tuple[dict[str, Any], np.ndarray, tuple[Any, Any, Any]]:
        loader = _segmentation_loader(cfg, pairs, image_size, degrade_resolution)
        confusion, elapsed, images, masks, preds = collect_unet(model, loader, device, use_amp)
        metrics = segmentation_metrics(confusion)
        metrics["inference_ms_per_image"] = elapsed / len(pairs) * 1000
        return metrics, confusion, (images, masks, preds)

    if not degradation:
        metrics, confusion, samples = run(None)
        save_segmentation_reports("unet", split, metrics, confusion, samples, cfg, mean, std)
        print(f"U-Net {split}: mIoU {metrics['miou']:.4f}")
        return

    results = []
    metrics, _, _ = run(None)
    results.append({"label": "native", "resolution": None, "metrics": metrics})
    for resolution in cfg["degradation"]["resolutions"]:
        metrics, _, _ = run(tuple(resolution))
        results.append(
            {"label": f"{resolution[0]}x{resolution[1]}", "resolution": resolution, "metrics": metrics}
        )
    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    save_json(results, metrics_dir / f"degradation_unet_{split}.json")
    save_degradation_table(results, metrics_dir / f"degradation_unet_{split}.csv", "segmentation")
    plot_degradation(
        results,
        resolve_path(cfg["paths"]["figures"]) / f"degradation_unet_{split}.png",
        "segmentation",
    )
    print("U-Net degradation:", {r["label"]: round(r["metrics"]["miou"], 4) for r in results})


def evaluate_ensemble(
    cfg: dict[str, Any],
    vit_checkpoint: str,
    unet_checkpoint: str,
    split: str,
    device: torch.device,
    degradation: bool,
) -> None:
    vit, vit_ckpt = build_classifier(vit_checkpoint, device)
    unet, unet_ckpt = build_unet(unet_checkpoint, device)
    vit_size = int(vit_ckpt.get("image_size", cfg["data"]["image_size_classifier"]))
    unet_size = int(unet_ckpt.get("image_size", cfg["data"]["image_size_unet"]))
    mean = vit_ckpt.get("mean", cfg["data"]["mean"])
    std = vit_ckpt.get("std", cfg["data"]["std"])
    thresholds = vit_ckpt.get("thresholds", 0.5)
    alpha = cfg["ensemble"]["alpha"]
    min_pixels = cfg["ensemble"]["min_pixels"]
    pairs = load_pairs(cfg, split)
    use_amp, _ = amp_settings(device, bool(cfg["training"]["amp"]))

    def run(
        degrade_resolution: tuple[int, int] | None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, tuple[Any, Any]]:
        vit_loader = _classification_loader(cfg, pairs, vit_size, degrade_resolution)
        vit_probs, labels, vit_elapsed, images, true_labels = collect_vit(
            vit, vit_loader, device, use_amp
        )
        unet_loader = _segmentation_loader(cfg, pairs, unet_size, degrade_resolution)
        unet_probs, unet_elapsed = collect_unet_presence(
            unet, unet_loader, device, use_amp, min_pixels
        )
        fused = fuse_probabilities(vit_probs, unet_probs, alpha)
        metrics = classification_metrics_from_probs(fused, labels, threshold=thresholds)
        metrics["inference_ms_per_image"] = (vit_elapsed + unet_elapsed) / len(pairs) * 1000
        metrics["thresholds"] = thresholds
        return metrics, fused, labels, (images, true_labels)

    if not degradation:
        metrics, probs, labels, samples = run(None)
        save_classification_reports("ensemble", split, metrics, probs, labels, samples, cfg, mean, std)
        print(f"Ensemble {split}: macro-F1 {metrics['macro_f1']:.4f}")
        return

    results = []
    metrics, _, _, _ = run(None)
    results.append({"label": "native", "resolution": None, "metrics": metrics})
    for resolution in cfg["degradation"]["resolutions"]:
        metrics, _, _, _ = run(tuple(resolution))
        results.append(
            {"label": f"{resolution[0]}x{resolution[1]}", "resolution": resolution, "metrics": metrics}
        )
    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    save_json(results, metrics_dir / f"degradation_ensemble_{split}.json")
    save_degradation_table(results, metrics_dir / f"degradation_ensemble_{split}.csv", "classification")
    plot_degradation(
        results,
        resolve_path(cfg["paths"]["figures"]) / f"degradation_ensemble_{split}.png",
        "classification",
    )
    print("Ensemble degradation:", {r["label"]: round(r["metrics"]["macro_f1"], 4) for r in results})


def evaluate_logistic(
    cfg: dict[str, Any],
    checkpoint: str,
    split: str,
    degradation: bool,
) -> None:
    model = joblib.load(checkpoint)
    image_size = int(cfg["logistic"]["image_size"])
    pairs = load_pairs(cfg, split)

    def run(degrade_resolution: tuple[int, int] | None) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        start = time.perf_counter()
        features, labels = extract_split(
            pairs, image_size, degrade_resolution=degrade_resolution
        )
        probs = logistic_positive_proba(model, features)
        elapsed = time.perf_counter() - start
        metrics = classification_metrics_from_probs(probs, labels)
        metrics["inference_ms_per_image"] = elapsed / len(pairs) * 1000
        return metrics, probs, labels

    if not degradation:
        metrics, probs, labels = run(None)
        save_classification_reports(
            "logistic",
            split,
            metrics,
            probs,
            labels,
            (None, None),
            cfg,
            cfg["data"]["mean"],
            cfg["data"]["std"],
        )
        print(f"Logistic {split}: macro-F1 {metrics['macro_f1']:.4f}")
        return

    results = []
    metrics, _, _ = run(None)
    results.append({"label": "native", "resolution": None, "metrics": metrics})
    for resolution in cfg["degradation"]["resolutions"]:
        metrics, _, _ = run(tuple(resolution))
        results.append(
            {"label": f"{resolution[0]}x{resolution[1]}", "resolution": resolution, "metrics": metrics}
        )
    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    save_json(results, metrics_dir / f"degradation_logistic_{split}.json")
    save_degradation_table(results, metrics_dir / f"degradation_logistic_{split}.csv", "classification")
    plot_degradation(
        results,
        resolve_path(cfg["paths"]["figures"]) / f"degradation_logistic_{split}.png",
        "classification",
    )
    print("Logistic degradation:", {r["label"]: round(r["metrics"]["macro_f1"], 4) for r in results})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.limit is not None:
        cfg["data"]["limit"] = args.limit
    if args.device:
        cfg["training"]["device"] = args.device
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    device = get_device(cfg["training"]["device"])

    models = ["vit", "unet", "logistic", "ensemble"] if args.model == "all" else [args.model]
    if args.checkpoint and len(models) > 1:
        raise ValueError("--checkpoint requires a single --model")
    if args.checkpoint and args.model == "ensemble":
        raise ValueError("--model ensemble needs both vit_best.pt and unet_best.pt")
    checkpoints = {
        name: args.checkpoint
        or str(resolve_path(cfg["paths"]["checkpoints"]) / CHECKPOINT_FILES[name])
        for name in models
        if name != "ensemble"
    }

    for name in models:
        if name == "ensemble":
            ensemble_checkpoints = {
                model: str(resolve_path(cfg["paths"]["checkpoints"]) / CHECKPOINT_FILES[model])
                for model in ("vit", "unet")
            }
            missing = [path for path in ensemble_checkpoints.values() if not Path(path).exists()]
            if missing:
                if args.model == "all":
                    print("Skipping ensemble: missing " + ", ".join(missing))
                    continue
                raise FileNotFoundError(missing[0])
            evaluate_ensemble(
                cfg,
                ensemble_checkpoints["vit"],
                ensemble_checkpoints["unet"],
                args.split,
                device,
                args.degradation,
            )
            continue

        checkpoint = checkpoints[name]
        if not Path(checkpoint).exists():
            if args.model == "all":
                print(f"Skipping {name}: checkpoint not found at {checkpoint}")
                continue
            raise FileNotFoundError(checkpoint)
        if name == "vit":
            evaluate_vit(cfg, checkpoint, args.split, device, args.degradation)
        elif name == "unet":
            evaluate_unet(cfg, checkpoint, args.split, device, args.degradation)
        else:
            evaluate_logistic(cfg, checkpoint, args.split, args.degradation)


if __name__ == "__main__":
    main()
