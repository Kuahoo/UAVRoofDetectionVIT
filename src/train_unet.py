"""Train a segmentation_models_pytorch U-Net for pixel-level roof defects."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data import (
    CLASS_NAMES,
    RoofDDataset,
    analyze_pairs,
    compute_class_weights,
    load_pairs,
    make_loader,
    save_split_manifest,
)
from src.metrics import (
    plot_segmentation_metrics,
    plot_segmentation_samples,
    save_per_class_table,
    segmentation_metrics,
)
from src.utils import (
    amp_settings,
    apply_smoke,
    ensure_dirs,
    get_device,
    load_checkpoint,
    load_config,
    resolve_path,
    save_checkpoint,
    save_json,
    set_seed,
)


class CombinedLoss(nn.Module):
    def __init__(self, cross_entropy: nn.Module, dice: nn.Module) -> None:
        super().__init__()
        self.cross_entropy = cross_entropy
        self.dice = dice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, targets) + self.dice(logits, targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float = 0.0,
) -> tuple[float, dict[str, Any], np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=use_amp
        ):
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)
        predictions = logits.detach().argmax(dim=1).cpu().numpy()
        targets = masks.detach().cpu().numpy()
        indices = targets.ravel() * len(CLASS_NAMES) + predictions.ravel()
        confusion += np.bincount(
            indices, minlength=len(CLASS_NAMES) ** 2
        ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))

    return total_loss / max(total_samples, 1), segmentation_metrics(confusion), confusion


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
    image_size = int(cfg["data"]["image_size_unet"])
    batch_size = int(cfg["training"]["batch_size_unet"])
    num_workers = int(cfg["data"]["num_workers"])
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]

    train_full = load_pairs(cfg, "train_full")
    train_pairs = load_pairs(cfg, "train")
    val_pairs = load_pairs(cfg, "val")
    test_pairs = load_pairs(cfg, "test")
    if not train_pairs or not val_pairs or not test_pairs:
        raise RuntimeError(
            f"Need non-empty train/val/test splits, got {len(train_pairs)}/"
            f"{len(val_pairs)}/{len(test_pairs)}"
        )
    save_split_manifest(cfg)
    stats = analyze_pairs(train_full, cfg["paths"]["train_stats"], cfg["data"]["root"])

    train_ds = RoofDDataset(train_pairs, "segmentation", image_size, True, mean, std)
    val_ds = RoofDDataset(val_pairs, "segmentation", image_size, False, mean, std)
    test_ds = RoofDDataset(test_pairs, "segmentation", image_size, False, mean, std)
    train_loader = make_loader(train_ds, batch_size, True, num_workers, cfg["seed"], drop_last=True)
    val_loader = make_loader(val_ds, batch_size, False, num_workers, cfg["seed"])
    test_loader = make_loader(test_ds, batch_size, False, num_workers, cfg["seed"])

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=cfg["model"]["unet_encoder"],
        encoder_weights=cfg["model"]["unet_encoder_weights"],
        in_channels=3,
        classes=len(CLASS_NAMES),
    ).to(device)
    class_weights = compute_class_weights(stats).to(device)
    criterion = CombinedLoss(
        nn.CrossEntropyLoss(weight=class_weights),
        smp.losses.DiceLoss(mode="multiclass", from_logits=True),
    )
    optimizer = Adam(model.parameters(), lr=float(cfg["training"]["lr_unet"]))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)
    use_amp, scaler = amp_settings(device, bool(cfg["training"]["amp"]))
    grad_clip = float(cfg["training"]["grad_clip"])
    checkpoint_path = resolve_path(cfg["paths"]["checkpoints"]) / "unet_best.pt"
    history: list[dict[str, Any]] = []
    best_metric = float("-inf")

    for epoch in range(1, int(cfg["training"]["epochs_unet"]) + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device, use_amp, optimizer, scaler, grad_clip
        )
        val_loss, val_metrics, _ = run_epoch(
            model, val_loader, criterion, device, use_amp
        )
        metric = val_metrics["miou"]
        improved = metric > best_metric
        if improved:
            best_metric = metric
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "encoder_name": cfg["model"]["unet_encoder"],
                    "image_size": image_size,
                    "mean": mean,
                    "std": std,
                    "num_classes": len(CLASS_NAMES),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "val_miou": metric,
                    "config": cfg,
                },
                checkpoint_path,
            )
        scheduler.step(metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_miou": metric,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"mIoU {metric:.4f}" + (" *" if improved else "")
        )

    save_json(history, resolve_path(cfg["paths"]["metrics"]) / "unet_history.json")
    if not checkpoint_path.exists():
        raise RuntimeError("No U-Net checkpoint was produced; check the configured epochs")

    model.load_state_dict(load_checkpoint(checkpoint_path, device)["model_state"])
    test_loss, test_metrics, test_confusion = run_epoch(
        model, test_loader, criterion, device, use_amp
    )
    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    figures_dir = resolve_path(cfg["paths"]["figures"])
    np.save(metrics_dir / "unet_test_confusion.npy", test_confusion)
    save_json(test_metrics, metrics_dir / "unet_test.json")
    save_per_class_table(test_metrics, metrics_dir / "unet_test_per_class.csv")
    plot_segmentation_metrics(test_metrics, figures_dir / "unet_test_metrics.png")

    sample_batch = next(iter(test_loader))
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, enabled=use_amp
    ):
        sample_predictions = model(sample_batch["image"].to(device)).argmax(dim=1)
    plot_segmentation_samples(
        sample_batch["image"].numpy(),
        sample_batch["mask"].numpy(),
        sample_predictions.cpu().numpy(),
        figures_dir / "unet_test_samples.png",
        mean,
        std,
    )
    print(f"Test loss {test_loss:.4f} | mIoU {test_metrics['miou']:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
