"""Classification and segmentation metrics plus report figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)

from src.data import CLASS_COLORS, CLASS_NAMES, DEFECT_CLASS_NAMES
from src.utils import resolve_path


def classification_metrics(
    y_true: Sequence[Sequence[int]],
    y_pred: Sequence[Sequence[int]],
    class_names: Sequence[str] = DEFECT_CLASS_NAMES,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    confusion = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    confusion_dict = {}
    for i, name in enumerate(class_names):
        tn, fp, fn, tp = confusion[i].ravel()
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        confusion_dict[name] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion": confusion_dict,
    }


def classification_metrics_from_probs(
    probs: np.ndarray,
    y_true: Sequence[Sequence[int]],
    threshold: float | Sequence[float] = 0.5,
    class_names: Sequence[str] = DEFECT_CLASS_NAMES,
) -> dict[str, Any]:
    predictions = np.asarray(probs) >= np.asarray(threshold)
    return classification_metrics(
        np.asarray(y_true) >= 0.5, predictions, class_names
    )


def tune_thresholds(
    probs: np.ndarray,
    y_true: Sequence[Sequence[int]],
    low: float = 0.10,
    high: float = 0.90,
    step: float = 0.05,
    class_names: Sequence[str] = DEFECT_CLASS_NAMES,
) -> list[float]:
    """Per-class F1-maximizing thresholds, tuned only on validation data."""
    probs = np.asarray(probs)
    targets = np.asarray(y_true) >= 0.5
    thresholds = []
    for index in range(len(class_names)):
        column = probs[:, index]
        best_threshold = 0.5
        best_f1 = float(
            f1_score(targets[:, index], column >= best_threshold, zero_division=0)
        )
        for candidate in np.arange(low, high + step / 2, step):
            f1 = float(
                f1_score(targets[:, index], column >= candidate, zero_division=0)
            )
            if f1 > best_f1:
                best_threshold, best_f1 = round(float(candidate), 2), f1
        thresholds.append(best_threshold)
    return thresholds


def _nanmean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else 0.0


def segmentation_metrics(
    confusion: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    confusion = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    iou_den = tp + fp + fn
    dice_den = 2 * tp + fp + fn
    iou = np.divide(tp, iou_den, out=np.full_like(tp, np.nan), where=iou_den > 0)
    dice = np.divide(2 * tp, dice_den, out=np.full_like(tp, np.nan), where=dice_den > 0)
    recall = np.divide(tp, confusion.sum(axis=1), out=np.full_like(tp, np.nan), where=confusion.sum(axis=1) > 0)
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "iou": None if np.isnan(iou[i]) else float(iou[i]),
            "dice": None if np.isnan(dice[i]) else float(dice[i]),
            "recall": None if np.isnan(recall[i]) else float(recall[i]),
            "support": int(confusion[i].sum()),
        }
    total = float(confusion.sum())
    return {
        "miou": _nanmean(iou),
        "mean_dice": _nanmean(dice),
        "pixel_accuracy": float(tp.sum() / total) if total else 0.0,
        "per_class": per_class,
    }


def save_per_class_table(metrics: dict[str, Any], path: str | Path) -> None:
    import pandas as pd

    rows = [{"class": name, **values} for name, values in metrics["per_class"].items()]
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_classification_metrics(
    metrics: dict[str, Any],
    path: str | Path,
    class_names: Sequence[str] = DEFECT_CLASS_NAMES,
) -> None:
    names = list(class_names)
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, key, label in (
        (-width, "precision", "Precision"),
        (0.0, "recall", "Recall"),
        (width, "f1", "F1"),
    ):
        values = [metrics["per_class"][name][key] for name in names]
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Classification metrics (macro F1 = {metrics['macro_f1']:.3f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_confusion_counts(
    metrics: dict[str, Any],
    path: str | Path,
    class_names: Sequence[str] = DEFECT_CLASS_NAMES,
) -> None:
    names = list(class_names)
    rows = 2
    cols = int(np.ceil(len(names) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 6))
    axes_list = list(np.atleast_1d(axes).ravel())
    for ax, name in zip(axes_list, names):
        cell = metrics["confusion"][name]
        matrix = np.array([[cell["tn"], cell["fp"]], [cell["fn"], cell["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", fontsize=10)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(name)
    for ax in axes_list[len(names) :]:
        ax.axis("off")
    _save(fig, path)


def plot_segmentation_metrics(
    metrics: dict[str, Any],
    path: str | Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    names = list(class_names)
    x = np.arange(len(names))
    width = 0.35
    iou = [metrics["per_class"][name]["iou"] for name in names]
    dice = [metrics["per_class"][name]["dice"] for name in names]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, [np.nan if v is None else v for v in iou], width, label="IoU")
    ax.bar(x + width / 2, [np.nan if v is None else v for v in dice], width, label="Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Segmentation metrics (mIoU = {metrics['miou']:.3f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def save_degradation_table(
    results: list[dict[str, Any]],
    path: str | Path,
    task: str,
    class_names: Sequence[str] | None = None,
) -> None:
    import pandas as pd

    if class_names is None:
        class_names = DEFECT_CLASS_NAMES if task == "classification" else CLASS_NAMES
    metric_key = "macro_f1" if task == "classification" else "miou"
    per_class_key = "recall" if task == "classification" else "iou"
    names = list(class_names) if task == "classification" else list(class_names)[1:]
    rows = []
    for result in results:
        row = {"resolution": result["label"], metric_key: result["metrics"][metric_key]}
        for name in names:
            row[f"{name}_{per_class_key}"] = result["metrics"]["per_class"][name][per_class_key]
        rows.append(row)
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_degradation(
    results: list[dict[str, Any]],
    path: str | Path,
    task: str,
    class_names: Sequence[str] | None = None,
) -> None:
    if class_names is None:
        class_names = DEFECT_CLASS_NAMES if task == "classification" else CLASS_NAMES
    metric_key = "macro_f1" if task == "classification" else "miou"
    per_class_key = "recall" if task == "classification" else "iou"
    names = list(class_names) if task == "classification" else list(class_names)[1:]
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, [r["metrics"][metric_key] for r in results], marker="o", linewidth=3, label=metric_key.upper())
    for name in names:
        values = [r["metrics"]["per_class"][name][per_class_key] for r in results]
        ax.plot(x, [np.nan if v is None else v for v in values], marker="o", label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Input resolution before model resize")
    ax.set_ylabel(per_class_key.capitalize())
    ax.set_title("Resolution degradation experiment")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, path)


def _denormalize(image: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    array = image.transpose(1, 2, 0).astype(np.float32)
    array = array * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return np.clip(array, 0, 1)


def _present_names(labels: np.ndarray, class_names: Sequence[str], probabilities: bool = False) -> str:
    names = []
    for i, name in enumerate(class_names):
        if labels[i] >= 0.5:
            names.append(f"{name} ({labels[i]:.0%})" if probabilities else name)
    return ", ".join(names) if names else "none"


def plot_classification_samples(
    images: Sequence[np.ndarray],
    true_labels: Sequence[Sequence[int]],
    probs: np.ndarray,
    path: str | Path,
    mean: Sequence[float],
    std: Sequence[float],
    class_names: Sequence[str] = DEFECT_CLASS_NAMES,
    n: int = 6,
) -> None:
    count = min(n, len(images))
    fig, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.6))
    axes = np.atleast_1d(axes)
    for ax, image, truth, prob in zip(axes, images[:count], true_labels[:count], probs[:count]):
        ax.imshow(_denormalize(np.asarray(image), mean, std))
        ax.set_title(
            f"True: {_present_names(np.asarray(truth), class_names)}\n"
            f"Pred: {_present_names(np.asarray(prob), class_names, probabilities=True)}",
            fontsize=8,
        )
        ax.axis("off")
    _save(fig, path)


def plot_segmentation_samples(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    preds: Sequence[np.ndarray],
    path: str | Path,
    mean: Sequence[float],
    std: Sequence[float],
    n: int = 4,
) -> None:
    count = min(n, len(images))
    fig, axes = plt.subplots(3, count, figsize=(3.2 * count, 9), squeeze=False)
    for col in range(count):
        axes[0, col].imshow(_denormalize(np.asarray(images[col]), mean, std))
        axes[1, col].imshow(CLASS_COLORS[np.asarray(masks[col], dtype=int)])
        axes[2, col].imshow(CLASS_COLORS[np.asarray(preds[col], dtype=int)])
        axes[0, col].set_title(f"Sample {col + 1}", fontsize=9)
        for row in range(3):
            axes[row, col].axis("off")
    axes[0, 0].set_ylabel("Image")
    axes[1, 0].set_ylabel("Ground truth")
    axes[2, 0].set_ylabel("Prediction")
    _save(fig, path)
