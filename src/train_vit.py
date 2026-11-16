"""Fine-tune a timm Vision Transformer for multi-label roof defect presence."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import (
    DEFECT_CLASS_NAMES,
    RoofDDataset,
    analyze_pairs,
    compute_pos_weight,
    load_pairs,
    make_loader,
    save_split_manifest,
)
from src.metrics import (
    classification_metrics_from_probs,
    plot_classification_metrics,
    plot_classification_samples,
    plot_confusion_counts,
    save_per_class_table,
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
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=use_amp
        ):
            logits = model(images)
            loss = criterion(logits, labels)
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
        all_probs.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_labels)
    metrics = classification_metrics_from_probs(probs, targets)
    return total_loss / max(total_samples, 1), metrics, probs, targets


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
    image_size = int(cfg["data"]["image_size_classifier"])
    batch_size = int(cfg["training"]["batch_size_classifier"])
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

    train_ds = RoofDDataset(
        train_pairs, "classification", image_size, True, mean, std, labels=stats["labels"]
    )
    val_ds = RoofDDataset(
        val_pairs, "classification", image_size, False, mean, std, labels=stats["labels"]
    )
    test_ds = RoofDDataset(test_pairs, "classification", image_size, False, mean, std)
    train_loader = make_loader(train_ds, batch_size, True, num_workers, cfg["seed"], drop_last=True)
    val_loader = make_loader(val_ds, batch_size, False, num_workers, cfg["seed"])
    test_loader = make_loader(test_ds, batch_size, False, num_workers, cfg["seed"])

    import timm

    model = timm.create_model(
        cfg["model"]["vit_name"],
        pretrained=bool(cfg["model"]["vit_pretrained"]),
        num_classes=len(DEFECT_CLASS_NAMES),
        img_size=image_size,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad_(True)

    pos_weight = compute_pos_weight(stats).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    checkpoint_path = resolve_path(cfg["paths"]["checkpoints"]) / "vit_best.pt"
    history: list[dict[str, Any]] = []
    best_metric = float("-inf")

    def validate_and_save(
        phase: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: dict[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        nonlocal best_metric
        metric = val_metrics["macro_f1"]
        improved = metric > best_metric
        if improved:
            best_metric = metric
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "model_name": cfg["model"]["vit_name"],
                    "image_size": image_size,
                    "mean": mean,
                    "std": std,
                    "num_classes": len(DEFECT_CLASS_NAMES),
                    "class_names": DEFECT_CLASS_NAMES,
                    "phase": phase,
                    "epoch": epoch,
                    "val_macro_f1": metric,
                    "config": cfg,
                },
                checkpoint_path,
            )
        history.append(
            {
                "phase": phase,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": metric,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[{phase}] epoch {epoch:02d} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | macro-F1 {metric:.4f}"
            + (" *" if improved else "")
        )

    # Phase 1: linear probe.
    warmup_epochs = int(cfg["training"]["epochs_vit_warmup"])
    finetune_epochs = int(cfg["training"]["epochs_vit_finetune"])
    use_amp, scaler = amp_settings(device, bool(cfg["training"]["amp"]))
    grad_clip = float(cfg["training"]["grad_clip"])
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["lr_vit_warmup"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    for epoch in range(1, warmup_epochs + 1):
        train_loss, _, _, _ = run_epoch(
            model, train_loader, criterion, device, use_amp, optimizer, scaler, grad_clip
        )
        val_loss, val_metrics, _, _ = run_epoch(
            model, val_loader, criterion, device, use_amp
        )
        validate_and_save("warmup", epoch, train_loss, val_loss, val_metrics, optimizer)

    # Phase 2: unfreeze the final two transformer blocks.
    if finetune_epochs > 0:
        if checkpoint_path.exists():
            model.load_state_dict(load_checkpoint(checkpoint_path, device)["model_state"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for block in model.blocks[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in model.get_classifier().parameters():
            parameter.requires_grad_(True)

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg["training"]["lr_vit_finetune"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)
        use_amp, scaler = amp_settings(device, bool(cfg["training"]["amp"]))
        for epoch in range(1, finetune_epochs + 1):
            train_loss, _, _, _ = run_epoch(
                model, train_loader, criterion, device, use_amp, optimizer, scaler, grad_clip
            )
            val_loss, val_metrics, _, _ = run_epoch(
                model, val_loader, criterion, device, use_amp
            )
            validate_and_save("finetune", epoch, train_loss, val_loss, val_metrics, optimizer)
            scheduler.step()

    save_json(history, resolve_path(cfg["paths"]["metrics"]) / "vit_history.json")

    if not checkpoint_path.exists():
        raise RuntimeError("No ViT checkpoint was produced; check the configured epochs")

    model.load_state_dict(load_checkpoint(checkpoint_path, device)["model_state"])
    test_loss, test_metrics, test_probs, test_targets = run_epoch(
        model, test_loader, criterion, device, use_amp
    )
    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    figures_dir = resolve_path(cfg["paths"]["figures"])
    np.save(metrics_dir / "vit_test_probs.npy", test_probs)
    np.save(metrics_dir / "vit_test_labels.npy", test_targets)
    save_json(test_metrics, metrics_dir / "vit_test.json")
    save_per_class_table(test_metrics, metrics_dir / "vit_test_per_class.csv")
    plot_classification_metrics(test_metrics, figures_dir / "vit_test_metrics.png")
    plot_confusion_counts(test_metrics, figures_dir / "vit_test_confusion.png")

    sample_batch = next(iter(test_loader))
    plot_classification_samples(
        sample_batch["image"].numpy(),
        sample_batch["labels"].numpy(),
        test_probs[: len(sample_batch["labels"])],
        figures_dir / "vit_test_samples.png",
        mean,
        std,
    )
    print(f"Test loss {test_loss:.4f} | macro-F1 {test_metrics['macro_f1']:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
