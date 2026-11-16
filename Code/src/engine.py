"""Training loop shared by the baseline, DG-PLDR, and proposed entry points.

Source-validation macro-F1 chooses the checkpoint for the original variants.
The lesion variants choose the checkpoint on a source stress set built from
held-out background textures. Target-domain images are not loaded here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import ensure_dir, set_seed
from src.datasets import (
    PlantDiseaseDataset,
    class_names_from_map,
    collate_batch,
    load_split_rows,
    materialize_splits,
)
from src.losses import (
    classification_loss,
    feature_consistency_loss,
    lesion_asymmetric_loss,
    logit_prototype_contrastive_loss,
    prediction_consistency_loss,
    prototype_stability_loss,
)
from src.metrics import compute_metrics
from src.models import PlantDiseaseModel
from src.prototypes import LogitPrototypeMemory


VARIANTS = {
    "baseline": {
        "use_ssd": False,
        "use_lpms": False,
        "return_counterfactual": False,
        "cls_on_cf": False,
        "lambda_cf": 0.0,
        "lambda_proto": 0.0,
    },
    "dgpldr": {
        "use_ssd": True,
        "use_lpms": True,
        "return_counterfactual": False,
        "cls_on_cf": False,
        "lambda_cf": 0.0,
        "lambda_proto": 0.0,
    },
    "cf_aug": {
        "use_ssd": True,
        "use_lpms": True,
        "return_counterfactual": True,
        "cls_on_cf": True,
        "lambda_cf": 0.0,
        "lambda_proto": 0.0,
    },
    "cf_consistency": {
        "use_ssd": True,
        "use_lpms": True,
        "return_counterfactual": True,
        "cls_on_cf": False,
        "lambda_cf": 1.0,
        "lambda_proto": 0.0,
    },
    "proto_stability": {
        "use_ssd": True,
        "use_lpms": True,
        "return_counterfactual": True,
        "cls_on_cf": False,
        "lambda_cf": 0.0,
        "lambda_proto": 1.0,
    },
    "full": {
        "use_ssd": True,
        "use_lpms": True,
        "return_counterfactual": True,
        "cls_on_cf": False,
        "lambda_cf": 1.0,
        "lambda_proto": 1.0,
    },
    "hard_aug": {
        "use_ssd": True,
        "use_lpms": True,
        "return_counterfactual": True,
        "cls_on_cf": True,
        "lambda_cf": 0.0,
        "lambda_proto": 0.0,
        "lambda_lesion": 0.0,
        "lambda_bg": 0.0,
        "cf_mode": "boundary",
        "select_on_stress": True,
    },
    "lesion_dg": {
        "use_ssd": True,
        "use_lpms": True,
        "return_counterfactual": True,
        "cls_on_cf": False,
        "lambda_cf": 0.0,
        "lambda_proto": 0.0,
        "lambda_lesion": 0.1,
        "lambda_bg": 0.1,
        "cf_mode": "boundary",
        "select_on_stress": True,
    },
}


def apply_variant(cfg: dict) -> dict:
    variant = cfg.get("variant", "baseline")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {sorted(VARIANTS)}.")
    merged = dict(cfg)
    merged.update(VARIANTS[variant])
    # Explicit config values win over the variant preset when the user set them
    # to something other than the preset only for lambda weights already in cfg
    # that were meant as the base weights. Variant lambdas are the experiment.
    for key, value in VARIANTS[variant].items():
        merged[key] = value
    if "lambda_cf" in cfg and variant in {"cf_consistency", "full"}:
        merged["lambda_cf"] = float(cfg["lambda_cf"])
    if "lambda_proto" in cfg and variant in {"proto_stability", "full"}:
        merged["lambda_proto"] = float(cfg["lambda_proto"])
    if "lambda_lpcl" in cfg:
        merged["lambda_lpcl"] = float(cfg["lambda_lpcl"])
    if "lambda_lesion" in cfg and variant == "lesion_dg":
        merged["lambda_lesion"] = float(cfg["lambda_lesion"])
    if "lambda_bg" in cfg and variant == "lesion_dg":
        merged["lambda_bg"] = float(cfg["lambda_bg"])
    merged["variant"] = variant
    return merged


def make_loader(rows, cfg, train: bool, variant_cfg: dict, stress: bool = False) -> DataLoader:
    dataset = PlantDiseaseDataset(
        rows,
        image_size=int(cfg["image_size"]),
        train=train,
        return_counterfactual=bool(variant_cfg["return_counterfactual"]) and train and not stress,
        return_style_view=bool(variant_cfg["use_ssd"]) and train and not stress,
        num_cf_views=int(cfg.get("cf_views", 2)),
        hflip=bool(cfg.get("hflip", True)) and not stress,
        limit=cfg.get("limit"),
        cf_mode=str(variant_cfg.get("cf_mode", "legacy")),
        stress=stress,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=train,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


def batch_loss(model, memory, batch, cfg, device):
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    style = batch.get("style_image")
    if style is not None:
        style = style.to(device, non_blocking=True)
    use_lesion = float(cfg.get("lambda_lesion", 0.0)) > 0 or float(cfg.get("lambda_bg", 0.0)) > 0
    if use_lesion:
        logits, embedding, feat_map = model(images, style, return_map=True)
    else:
        logits, embedding = model(images, style)
        feat_map = None
    loss = classification_loss(logits, labels)
    parts = {"cls": float(loss.detach())}

    if cfg["use_lpms"] and memory is not None:
        memory.update(logits, labels)
        lpcl = logit_prototype_contrastive_loss(
            logits,
            labels,
            memory.active_prototypes(),
            temperature=float(cfg.get("lp_temperature", 0.8)),
        )
        loss = loss + float(cfg.get("lambda_lpcl", 0.1)) * lpcl
        parts["lpcl"] = float(lpcl.detach())

    if "counterfactual" in batch:
        counterfactual = batch["counterfactual"].to(device, non_blocking=True)
        batch_size, views, channels, height, width = counterfactual.shape
        flat = counterfactual.reshape(batch_size * views, channels, height, width)
        if use_lesion:
            cf_logits, cf_embedding, cf_map = model(flat, return_map=True)
        else:
            cf_logits, cf_embedding = model(flat)
            cf_map = None
        cf_logits = cf_logits.reshape(batch_size, views, -1)
        cf_embedding = cf_embedding.reshape(batch_size, views, -1)
        if cfg["cls_on_cf"]:
            cf_labels = labels.unsqueeze(1).expand(-1, views).reshape(-1)
            cf_cls = classification_loss(cf_logits.reshape(batch_size * views, -1), cf_labels)
            loss = loss + cf_cls
            parts["cls_cf"] = float(cf_cls.detach())
        if float(cfg["lambda_cf"]) > 0:
            feat = feature_consistency_loss(embedding, cf_embedding)
            pred = prediction_consistency_loss(logits, cf_logits)
            loss = loss + float(cfg["lambda_cf"]) * (feat + pred)
            parts["cf_feat"] = float(feat.detach())
            parts["cf_pred"] = float(pred.detach())
        if float(cfg["lambda_proto"]) > 0:
            proto = prototype_stability_loss(
                embedding, cf_embedding, labels, num_classes=model.classifier.out_features
            )
            loss = loss + float(cfg["lambda_proto"]) * proto
            parts["proto"] = float(proto.detach())
        if use_lesion and cf_map is not None:
            channels_map = cf_map.shape[1]
            map_h, map_w = cf_map.shape[-2:]
            cf_map = cf_map.reshape(batch_size, views, channels_map, map_h, map_w)
            keep, change = lesion_asymmetric_loss(
                feat_map,
                cf_map,
                batch["preserve_mask"].to(device, non_blocking=True),
                batch["preserve_mask_cf"].to(device, non_blocking=True),
                batch["background_mask"].to(device, non_blocking=True),
                batch["background_mask_cf"].to(device, non_blocking=True),
            )
            loss = loss + float(cfg.get("lambda_lesion", 0.0)) * keep + float(cfg.get("lambda_bg", 0.0)) * change
            parts["lesion"] = float(keep.detach())
            parts["background"] = float(change.detach())
    parts["total"] = float(loss.detach())
    return loss, parts, logits, labels


@torch.no_grad()
def evaluate_loader(model, loader, class_names, device) -> dict:
    model.eval()
    predictions = []
    labels = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits, _ = model(images)
        predictions.append(logits.argmax(dim=1).cpu())
        labels.append(batch["label"])
    if not labels:
        return {"accuracy": 0.0, "macro_f1": 0.0, "confusion_matrix": None, "per_class": []}
    return compute_metrics(torch.cat(labels).numpy(), torch.cat(predictions).numpy(), class_names)


def save_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(cfg: dict) -> Path:
    cfg = apply_variant(cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = materialize_splits(cfg)
    names = class_names_from_map(cfg["class_map"])
    train_rows = load_split_rows(paths["source"], "train")
    val_rows = load_split_rows(paths["source"], "val")
    train_loader = make_loader(train_rows, cfg, train=True, variant_cfg=cfg)
    val_loader = make_loader(val_rows, cfg, train=False, variant_cfg=cfg)
    stress_loader = None
    if cfg.get("select_on_stress"):
        stress_loader = make_loader(val_rows, cfg, train=False, variant_cfg=cfg, stress=True)

    model = PlantDiseaseModel(
        num_classes=len(names),
        backbone=cfg.get("backbone", "resnet50"),
        pretrained=bool(cfg.get("pretrained", True)),
        use_ssd=bool(cfg["use_ssd"]),
    ).to(device)
    memory = LogitPrototypeMemory(len(names), momentum=float(cfg.get("lp_momentum", 0.99))).to(device)
    memory = memory if cfg["use_lpms"] else None

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.get("lr", 0.01)),
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    run_dir = ensure_dir(Path(cfg["output_dir"]) / cfg["variant"] / f"seed_{cfg.get('seed', 42)}")
    history = []
    best_f1 = -1.0
    best_path = run_dir / "best.pth"
    writer = _tensorboard(run_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        totals = []
        progress = tqdm(train_loader, desc=f"{cfg['variant']} epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            loss, parts, _, _ = batch_loss(model, memory, batch, cfg, device)
            loss.backward()
            optimizer.step()
            totals.append(parts["total"])
            progress.set_postfix(loss=f"{parts['total']:.3f}")
        scheduler.step()
        val_metrics = evaluate_loader(model, val_loader, names, device)
        row = {
            "epoch": epoch,
            "train_loss": float(sum(totals) / max(len(totals), 1)),
            "source_val_accuracy": val_metrics["accuracy"],
            "source_val_macro_f1": val_metrics["macro_f1"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        select_f1 = val_metrics["macro_f1"]
        if stress_loader is not None:
            stress_metrics = evaluate_loader(model, stress_loader, names, device)
            row["stress_accuracy"] = stress_metrics["accuracy"]
            row["stress_macro_f1"] = stress_metrics["macro_f1"]
            select_f1 = stress_metrics["macro_f1"]
        history.append(row)
        if writer is not None:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        message = (
            f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"source_acc={row['source_val_accuracy']:.4f} "
            f"source_macro_f1={row['source_val_macro_f1']:.4f}"
        )
        if "stress_macro_f1" in row:
            message += f" stress_acc={row['stress_accuracy']:.4f} stress_macro_f1={row['stress_macro_f1']:.4f}"
        print(message)
        if select_f1 >= best_f1:
            best_f1 = select_f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "memory": None if memory is None else memory.state_dict(),
                    "epoch": epoch,
                    "source_val_macro_f1": val_metrics["macro_f1"],
                    "selection_macro_f1": best_f1,
                    "class_names": names,
                    "config": {key: value for key, value in cfg.items() if not key.startswith("_")},
                },
                best_path,
            )
    save_history(run_dir / "history.csv", history)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in cfg.items() if not key.startswith("_")}, handle, indent=2)
    if writer is not None:
        writer.close()
    label = "stress macro-F1" if stress_loader is not None else "source macro-F1"
    print(f"best {label} {best_f1:.4f} saved to {best_path}")
    return best_path


def _tensorboard(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(run_dir / "tb"))
    except Exception:
        return None
