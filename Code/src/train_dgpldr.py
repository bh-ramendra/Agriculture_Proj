"""DG-PLDR reproduction: style-sensitive distillation and logit-prototype matching."""

from __future__ import annotations

import argparse

from src.common import load_config
from src.engine import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DG-PLDR reproduction.")
    parser.add_argument("--config", default="configs/dgpldr.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["variant"] = "dgpldr"
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    train(cfg)


if __name__ == "__main__":
    main()
