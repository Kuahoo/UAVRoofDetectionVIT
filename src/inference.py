"""Checkpoint loading, single-image inference, overlays, and conflict logic."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np
import torch

from src.data import CLASS_COLORS, CLASS_NAMES, DEFECT_CLASS_NAMES, resize_rgb, to_tensor
from src.features import extract_features
from src.utils import load_checkpoint

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _preprocess(
    image_rgb: np.ndarray,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    tensor = to_tensor(resize_rgb(image_rgb, image_size), mean, std)
    return tensor.unsqueeze(0)


def _autocast(device: torch.device):
    return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")


def build_classifier(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    import timm

    checkpoint = load_checkpoint(checkpoint_path, device)
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=int(checkpoint.get("num_classes", len(DEFECT_CLASS_NAMES))),
        img_size=int(checkpoint.get("image_size", 224)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def build_unet(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    import segmentation_models_pytorch as smp

    checkpoint = load_checkpoint(checkpoint_path, device)
    model = smp.Unet(
        encoder_name=checkpoint["encoder_name"],
        encoder_weights=None,
        in_channels=3,
        classes=int(checkpoint.get("num_classes", len(CLASS_NAMES))),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def predict_classification(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    image_size: int,
    device: torch.device,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    inputs = _preprocess(image_rgb, image_size, mean, std).to(device)
    with torch.no_grad(), _autocast(device):
        logits = model(inputs)
        probs = torch.sigmoid(logits.float())[0]
    return probs.cpu().numpy()


def predict_segmentation_probs(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    image_size: int,
    device: torch.device,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    """Return a (6, H, W) float32 softmax map."""
    inputs = _preprocess(image_rgb, image_size, mean, std).to(device)
    with torch.no_grad(), _autocast(device):
        logits = model(inputs)
        probs = torch.softmax(logits.float(), dim=1)[0]
    return probs.cpu().numpy()


def unet_presence_probs(
    class_probs: np.ndarray,
    min_pixels: Sequence[int],
) -> np.ndarray:
    """Image-level presence from the mean softmax of each class's top pixels."""
    class_probs = np.asarray(class_probs)
    flattened = class_probs[1 : 1 + len(min_pixels)].reshape(len(min_pixels), -1)
    presence = []
    for probs, min_px in zip(flattened, min_pixels):
        k = max(1, min(int(min_px), probs.size))
        presence.append(float(np.partition(probs, probs.size - k)[-k:].mean()))
    return np.asarray(presence, dtype=np.float32)


def fuse_probabilities(
    classifier_probs: np.ndarray,
    unet_probs: np.ndarray,
    alpha: Sequence[float],
) -> np.ndarray:
    """Weighted ViT/U-Net image-level fusion; alpha is per defect class."""
    alpha = np.asarray(alpha, dtype=np.float32)
    classifier_probs = np.asarray(classifier_probs, dtype=np.float32)
    return alpha * classifier_probs + (1.0 - alpha) * np.asarray(unet_probs, dtype=np.float32)


def logistic_positive_proba(model: Any, features: np.ndarray) -> np.ndarray:
    """Normalize sklearn multi-label predict_proba output to (N, 5) defect probs."""
    proba = model.predict_proba(features)
    if isinstance(proba, list):
        return np.stack([np.asarray(p)[:, 1] for p in proba], axis=1)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 3:
        return np.stack([p[:, 1] for p in proba], axis=1)
    return proba


def predict_logistic(
    model: Any,
    image_rgb: np.ndarray,
    image_size: int = 64,
) -> np.ndarray:
    features = extract_features(image_rgb, image_size).reshape(1, -1)
    return logistic_positive_proba(model, features)[0]


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    output = image_rgb.astype(np.float32).copy()
    for index in range(1, len(CLASS_COLORS)):
        selected = mask == index
        if selected.any():
            output[selected] = (1 - alpha) * output[selected] + alpha * CLASS_COLORS[index]
    return np.clip(output, 0, 255).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[np.asarray(mask, dtype=np.int64)]


def _threshold_vector(threshold: float | Sequence[float], count: int) -> np.ndarray:
    values = np.asarray(threshold, dtype=np.float32)
    return np.full(count, float(values), dtype=np.float32) if values.ndim == 0 else values


def detected_defects(
    probs: np.ndarray,
    threshold: float | Sequence[float] = 0.5,
    class_names: Sequence[str] = DEFECT_CLASS_NAMES,
) -> list[tuple[str, float]]:
    """`probs` is length-5 over defect classes (no background slot)."""
    probs = np.asarray(probs)
    thresholds = _threshold_vector(threshold, len(probs))
    return [
        (name, float(prob))
        for name, prob, cutoff in zip(class_names, probs, thresholds)
        if prob >= cutoff
    ]


def conflict_messages(
    probs: np.ndarray,
    mask: np.ndarray,
    threshold: float | Sequence[float] = 0.5,
    class_names: Sequence[str] = DEFECT_CLASS_NAMES,
) -> list[str]:
    """Deterministic ViT/U-Net disagreement report; never invents pixels.

    `probs` indexes defect classes 0..4 corresponding to mask IDs 1..5.
    """
    probs = np.asarray(probs)
    thresholds = _threshold_vector(threshold, len(probs))
    spatial = np.bincount(np.asarray(mask).ravel(), minlength=len(CLASS_NAMES)) > 0
    messages = []
    for index, name in enumerate(class_names):
        class_id = index + 1
        confidence = float(probs[index])
        if confidence >= thresholds[index] and not spatial[class_id]:
            messages.append(
                f"Unlocalized Confidence Warning: ViT predicts {name} at "
                f"{confidence:.0%}, but U-Net found no {name} pixels."
            )
        elif confidence < thresholds[index] and spatial[class_id]:
            messages.append(
                f"Sub-threshold Spatial Detection: U-Net found {name} pixels, "
                f"but ViT confidence is {confidence:.0%}."
            )
    return messages
