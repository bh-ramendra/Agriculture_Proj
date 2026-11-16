"""Save an audit grid of disease-preserving counterfactuals and a metadata table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ensure_dir, load_config
from src.counterfactual import INTERVENTIONS, apply_intervention, make_boundary_counterfactuals
from src.datasets import load_split_rows, materialize_splits
from src.transforms import load_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a counterfactual visual audit.")
    parser.add_argument("--config", default="configs/counterfactual.yaml")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--output", default="results/counterfactuals")
    parser.add_argument("--mode", default="legacy", choices=["legacy", "boundary"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = materialize_splits(cfg)
    rows = load_split_rows(paths["source"], "train")
    if args.mode == "boundary":
        stride = max(1, len(rows) // args.num_images)
        rows = rows[::stride][: args.num_images]
    else:
        rows = rows[: args.num_images]
    output = ensure_dir(ROOT / args.output)
    image_dir = ensure_dir(output / "images")
    metadata = []
    names = ("boundary",) if args.mode == "boundary" else INTERVENTIONS
    figure, axes = plt.subplots(len(rows), 1 + len(names), figsize=(3.2 * (1 + len(names)), 2.4 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, -1)
    for row_index, row in enumerate(rows):
        image = load_rgb(row["image_path"], int(cfg["image_size"]))
        axes[row_index, 0].imshow(image.permute(1, 2, 0).numpy())
        axes[row_index, 0].set_title("original", fontsize=8)
        axes[row_index, 0].axis("off")
        for column, name in enumerate(names, start=1):
            if name == "boundary":
                counterfactual, _, _, _, _ = make_boundary_counterfactuals(image, num_views=1)
                counterfactual = counterfactual[0].clamp(0, 1)
            else:
                counterfactual = apply_intervention(image, name).clamp(0, 1)
            axes[row_index, column].imshow(counterfactual.permute(1, 2, 0).numpy())
            axes[row_index, column].set_title(name, fontsize=8)
            axes[row_index, column].axis("off")
            filename = f"{row_index:02d}_{name}.png"
            plt.imsave(image_dir / filename, counterfactual.permute(1, 2, 0).numpy())
            metadata.append(
                {
                    "original_path": row["image_path"],
                    "cf_path": str(image_dir / filename),
                    "intervention_type": name,
                    "strength": "default",
                    "class_name": row["class_name"],
                }
            )
    figure.tight_layout()
    figure.savefig(output / "audit_grid.png", dpi=120)
    plt.close(figure)
    with (output / "cf_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_path", "cf_path", "intervention_type", "strength", "class_name"],
        )
        writer.writeheader()
        writer.writerows(metadata)
    print(f"wrote {output / 'audit_grid.png'} and {len(metadata)} counterfactual images")


if __name__ == "__main__":
    main()
