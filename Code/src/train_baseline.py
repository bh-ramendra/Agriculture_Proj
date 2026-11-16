"""Baseline: source-only ResNet-50 or ViT, no counterfactual or DG-PLDR losses."""

from __future__ import annotations

import argparse

from src.common import load_config
from src.engine import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the plain source-domain baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backbone", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["variant"] = "baseline"
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.backbone is not None:
        cfg["backbone"] = args.backbone
    train(cfg)


if __name__ == "__main__":
    main()
