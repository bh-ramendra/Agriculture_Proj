"""Measure prediction consistency under each nuisance intervention.

This script evaluates a finished checkpoint. It does not train, and it builds
counterfactuals from source-validation images so the target set stays untouched.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config
from src.counterfactual import INTERVENTIONS, apply_intervention
from src.datasets import load_split_rows, materialize_splits
from src.models import PlantDiseaseModel
from src.transforms import load_rgb, normalize


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Nuisance-factor robustness table.")
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device)
    model = PlantDiseaseModel(
        num_classes=len(payload["class_names"]),
        backbone=payload.get("config", {}).get("backbone", "resnet50"),
        pretrained=False,
        use_ssd=bool(payload.get("config", {}).get("use_ssd", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    rows = load_split_rows(materialize_splits(cfg)["source"], "val")[: args.limit]
    stats = {
        name: {"same": 0, "count": 0, "conf_delta": 0.0, "feat_dist": 0.0} for name in INTERVENTIONS
    }
    for row in rows:
        image = load_rgb(row["image_path"], int(cfg["image_size"]))
        original = normalize(image).unsqueeze(0).to(device)
        logits, embedding = model(original)
        pred = int(logits.argmax(dim=1))
        confidence = float(torch.softmax(logits, dim=1).max())
        for name in INTERVENTIONS:
            counterfactual = normalize(apply_intervention(image, name)).unsqueeze(0).to(device)
            cf_logits, cf_embedding = model(counterfactual)
            cf_pred = int(cf_logits.argmax(dim=1))
            cf_conf = float(torch.softmax(cf_logits, dim=1).max())
            distance = float(F.mse_loss(F.normalize(embedding, dim=-1), F.normalize(cf_embedding, dim=-1)))
            bucket = stats[name]
            bucket["count"] += 1
            bucket["same"] += int(pred == cf_pred)
            bucket["conf_delta"] += abs(confidence - cf_conf)
            bucket["feat_dist"] += distance
    out = Path(args.output or Path(args.checkpoint).resolve().parent / "nuisance_robustness.csv")
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["intervention", "prediction_consistency", "mean_confidence_change", "mean_feature_distance", "count"],
        )
        writer.writeheader()
        for name, bucket in stats.items():
            count = max(bucket["count"], 1)
            writer.writerow(
                {
                    "intervention": name,
                    "prediction_consistency": bucket["same"] / count,
                    "mean_confidence_change": bucket["conf_delta"] / count,
                    "mean_feature_distance": bucket["feat_dist"] / count,
                    "count": bucket["count"],
                }
            )
            print(
                f"{name}: consistency={bucket['same'] / count:.3f} "
                f"conf_change={bucket['conf_delta'] / count:.3f} "
                f"feat_dist={bucket['feat_dist'] / count:.3f}"
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
