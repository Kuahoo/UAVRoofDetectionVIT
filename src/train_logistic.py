"""Course baseline: HOG + color statistics with one-vs-rest logistic regression."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src.data import analyze_pairs, load_pairs, save_split_manifest
from src.features import extract_split
from src.inference import logistic_positive_proba
from src.metrics import (
    classification_metrics_from_probs,
    plot_classification_metrics,
    plot_confusion_counts,
    save_per_class_table,
)
from src.utils import (
    apply_smoke,
    ensure_dirs,
    load_config,
    resolve_path,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.limit is not None:
        cfg["data"]["limit"] = args.limit
    ensure_dirs(cfg)
    save_split_manifest(cfg)

    train_pairs = load_pairs(cfg, "train")
    val_pairs = load_pairs(cfg, "val")
    test_pairs = load_pairs(cfg, "test")
    if not train_pairs or not val_pairs or not test_pairs:
        raise RuntimeError(
            f"Need non-empty train/val/test splits, got {len(train_pairs)}/"
            f"{len(val_pairs)}/{len(test_pairs)}"
        )

    train_full = load_pairs(cfg, "train_full")
    stats = analyze_pairs(train_full, cfg["paths"]["train_stats"], cfg["data"]["root"])
    image_size = int(cfg["logistic"]["image_size"])

    print("Extracting train features...")
    x_train, y_train = extract_split(train_pairs, image_size, stats["labels"])
    print("Extracting validation features...")
    x_val, y_val = extract_split(val_pairs, image_size, stats["labels"])
    print("Extracting test features...")
    x_test, y_test = extract_split(test_pairs, image_size)

    model = OneVsRestClassifier(
        LogisticRegression(
            solver=cfg["logistic"]["solver"],
            max_iter=int(cfg["logistic"]["max_iter"]),
            C=float(cfg["logistic"]["C"]),
        ),
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    checkpoint_path = resolve_path(cfg["paths"]["checkpoints"]) / "logistic.joblib"
    joblib.dump(model, checkpoint_path)

    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    figures_dir = resolve_path(cfg["paths"]["figures"])
    for split, x_data, y_data in (("val", x_val, y_val), ("test", x_test, y_test)):
        probs = logistic_positive_proba(model, x_data)
        metrics = classification_metrics_from_probs(probs, y_data)
        np.save(metrics_dir / f"logistic_{split}_probs.npy", probs)
        np.save(metrics_dir / f"logistic_{split}_labels.npy", y_data)
        save_json(metrics, metrics_dir / f"logistic_{split}.json")
        save_per_class_table(metrics, metrics_dir / f"logistic_{split}_per_class.csv")
        plot_classification_metrics(metrics, figures_dir / f"logistic_{split}_metrics.png")
        plot_confusion_counts(metrics, figures_dir / f"logistic_{split}_confusion.png")
        print(f"{split}: macro-F1 {metrics['macro_f1']:.4f} | exact match {metrics['exact_match']:.4f}")

    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
