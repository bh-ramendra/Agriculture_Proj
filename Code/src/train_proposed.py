"""Proposed model and the ablation variants that add counterfactual learning."""

from __future__ import annotations

import argparse

from src.common import load_config
from src.engine import VARIANTS, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a counterfactual plant-disease variant.")
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument(
        "--variant",
        default="full",
        choices=["cf_aug", "cf_consistency", "proto_stability", "full", "hard_aug", "lesion_dg"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lambda-cf", type=float, default=None)
    parser.add_argument("--lambda-proto", type=float, default=None)
    parser.add_argument("--cf-views", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["variant"] = args.variant
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.lambda_cf is not None:
        cfg["lambda_cf"] = args.lambda_cf
    if args.lambda_proto is not None:
        cfg["lambda_proto"] = args.lambda_proto
    if args.cf_views is not None:
        cfg["cf_views"] = args.cf_views
    if args.variant not in VARIANTS:
        raise SystemExit(f"Unsupported variant {args.variant}")
    train(cfg)


if __name__ == "__main__":
    main()
