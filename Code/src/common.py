"""Shared config, seeding, and path helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    root = project_root()
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(root)
    for key in (
        "class_map",
        "plantvillage_color",
        "plantdoc_root",
        "mendeley_dir",
        "output_dir",
        "split_dir",
        "dinov2_weights",
        "feature_cache",
    ):
        if key in cfg and cfg[key]:
            candidate = Path(cfg[key])
            if not candidate.is_absolute():
                cfg[key] = str((root / candidate).resolve())
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
