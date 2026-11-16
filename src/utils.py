"""Small shared helpers: paths, config, seeding, device, JSON, checkpoints."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_path(path or "configs/default.yaml")
    with path.open() as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("outputs", "checkpoints", "metrics", "figures"):
        resolve_path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Make DataLoader workers deterministic relative to the base seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_settings(device: torch.device, enabled: bool) -> tuple[bool, torch.amp.GradScaler | None]:
    use_amp = bool(enabled and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    return use_amp, scaler


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def save_json(value: Any, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(jsonable(value), f, indent=2)


def load_json(path: str | Path) -> Any:
    with resolve_path(path).open() as f:
        return json.load(f)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(resolve_path(path), map_location=map_location, weights_only=False)


def apply_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """Tiny end-to-end run: 1 epoch, 8 samples, no pretrained downloads."""
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["num_workers"] = 0
    cfg["data"]["limit"] = 8
    cfg["model"]["vit_pretrained"] = False
    cfg["model"]["unet_encoder_weights"] = None
    cfg["training"].update(
        {
            "batch_size_classifier": 2,
            "batch_size_unet": 2,
            "epochs_vit_warmup": 1,
            "epochs_vit_finetune": 1,
            "epochs_unet": 1,
        }
    )
    for key in ("outputs", "checkpoints", "metrics", "figures", "splits", "train_stats"):
        value = cfg["paths"][key]
        parts = list(Path(value).parts)
        if parts and parts[0] == "outputs":
            parts[0] = "outputs/smoke"
        else:
            parts = ["outputs", "smoke", *parts]
        cfg["paths"][key] = str(Path(*parts))
    return cfg
