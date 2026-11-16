"""Streamlit human-in-the-loop demo for the RoofD inspection prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASS_COLORS, CLASS_NAMES, DEFECT_CLASS_NAMES
from src.inference import (
    build_classifier,
    build_unet,
    conflict_messages,
    detected_defects,
    overlay_mask,
    predict_classification,
    predict_segmentation,
)
from src.utils import get_device, load_config, resolve_path

DISCLAIMER = "For research demonstration only; professional inspection is required."


@st.cache_resource
def load_models(config_path: str, device_name: str):
    cfg = load_config(config_path)
    device = get_device(device_name)
    checkpoints = resolve_path(cfg["paths"]["checkpoints"])
    vit, vit_ckpt = build_classifier(str(checkpoints / "vit_best.pt"), device)
    unet, unet_ckpt = build_unet(str(checkpoints / "unet_best.pt"), device)
    return cfg, device, vit, vit_ckpt, unet, unet_ckpt


def decode_upload(uploaded) -> np.ndarray | None:
    buffer = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(page_title="Roof Damage Inspection", layout="wide")
    st.title("Roof Damage Inspection Assistant")
    st.caption("ViT classification + U-Net localization on aerial RGB roof imagery.")

    config_path = st.sidebar.text_input("Config", "configs/default.yaml")
    st.sidebar.markdown(
        "**Defect colors**\n\n"
        + "\n".join(
            f"- <span style='color:rgb({c[0]},{c[1]},{c[2]})'>&#9632;</span> {name}"
            for name, c in zip(DEFECT_CLASS_NAMES, CLASS_COLORS[1:])
        ),
        unsafe_allow_html=True,
    )
    st.sidebar.caption(DISCLAIMER)

    try:
        cfg = load_config(config_path)
    except Exception as exc:  # pragma: no cover - UI path
        st.error(f"Could not load config: {exc}")
        return

    checkpoints = resolve_path(cfg["paths"]["checkpoints"])
    missing = [
        name
        for name in ("vit_best.pt", "unet_best.pt")
        if not (checkpoints / name).exists()
    ]
    if missing:
        st.warning(
            "Missing checkpoint(s): "
            + ", ".join(missing)
            + f". Train them into `{checkpoints}` first."
        )
        st.code(
            "python -m src.train_vit\n"
            "python -m src.train_unet\n"
            "python -m src.train_logistic",
            language="bash",
        )
        st.caption(DISCLAIMER)
        return

    uploaded = st.file_uploader(
        "Upload an aerial roof image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    )
    if uploaded is None:
        st.info("Upload an image to run the dual-model inspection.")
        st.caption(DISCLAIMER)
        return

    image_rgb = decode_upload(uploaded)
    if image_rgb is None:
        st.error("Could not decode the uploaded image.")
        return

    cfg, device, vit, vit_ckpt, unet, unet_ckpt = load_models(
        config_path, cfg["training"]["device"]
    )
    vit_size = int(vit_ckpt.get("image_size", cfg["data"]["image_size_classifier"]))
    unet_size = int(unet_ckpt.get("image_size", cfg["data"]["image_size_unet"]))
    vit_mean = vit_ckpt.get("mean", cfg["data"]["mean"])
    vit_std = vit_ckpt.get("std", cfg["data"]["std"])
    unet_mean = unet_ckpt.get("mean", cfg["data"]["mean"])
    unet_std = unet_ckpt.get("std", cfg["data"]["std"])

    probs = predict_classification(vit, image_rgb, vit_size, device, vit_mean, vit_std)
    mask = predict_segmentation(unet, image_rgb, unet_size, device, unet_mean, unet_std)
    overlay = overlay_mask(image_rgb, mask)
    detected = detected_defects(probs)
    warnings = conflict_messages(probs, mask)

    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(image_rgb, use_container_width=True)
    with right:
        st.subheader("U-Net mask overlay")
        st.image(overlay, use_container_width=True)

    st.subheader("ViT defect probabilities")
    table = pd.DataFrame(
        {
            "Defect": list(DEFECT_CLASS_NAMES),
            "Probability": [f"{float(p):.1%}" for p in probs],
            "Detected (>= 50%)": ["yes" if p >= 0.5 else "no" for p in probs],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    if detected:
        st.success(
            "Detected: "
            + ", ".join(f"{name} ({confidence:.0%})" for name, confidence in detected)
        )
    else:
        st.info("No defect class reached the 50% confidence threshold.")

    spatial_classes = [CLASS_NAMES[i] for i in np.unique(mask) if i > 0]
    st.write(
        "U-Net spatial classes: "
        + (", ".join(spatial_classes) if spatial_classes else "none")
    )

    for message in warnings:
        st.warning(message)

    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
