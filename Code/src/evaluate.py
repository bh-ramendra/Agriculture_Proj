"""Evaluate a checkpoint on source validation or the unseen target domain."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import ensure_dir, load_config
from src.datasets import PlantDiseaseDataset, collate_batch, load_split_rows, materialize_splits
from src.engine import evaluate_loader
from src.models import PlantDiseaseModel


def save_confusion(matrix: np.ndarray, class_names: list[str], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(8, len(class_names) * 0.35), max(6, len(class_names) * 0.35)))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=90, fontsize=7)
    axis.set_yticklabels(class_names, fontsize=7)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_evaluation(checkpoint: str, cfg: dict, split: str, output_dir: str, limit: int | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device)
    names = payload["class_names"]
    saved = payload.get("config", {})
    model = PlantDiseaseModel(
        num_classes=len(names),
        backbone=saved.get("backbone", cfg.get("backbone", "resnet50")),
        pretrained=False,
        use_ssd=bool(saved.get("use_ssd", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    paths = materialize_splits(cfg)
    if split == "target":
        rows = load_split_rows(paths["target"], "test")
    elif split == "source_val":
        rows = load_split_rows(paths["source"], "val")
    else:
        raise ValueError("split must be 'target' or 'source_val'")
    dataset = PlantDiseaseDataset(rows, image_size=int(cfg["image_size"]), train=False, limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_batch,
    )
    metrics = evaluate_loader(model, loader, names, device)
    out = ensure_dir(output_dir)
    summary_path = out / f"{split}_metrics.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": split,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    per_class_path = out / f"{split}_per_class.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    if metrics["confusion_matrix"] is not None:
        np.save(out / f"{split}_confusion.npy", metrics["confusion_matrix"])
        save_confusion(metrics["confusion_matrix"], names, out / f"{split}_confusion.png")
    print(
        f"{split}: accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
    )
    print(f"wrote {summary_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved plant-disease checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="target", choices=["target", "source_val"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = args.output_dir or str(Path(args.checkpoint).resolve().parent / "eval" / args.split)
    run_evaluation(args.checkpoint, cfg, args.split, output, limit=args.limit)


if __name__ == "__main__":
    main()
