"""Train classifiers on frozen DINOv2 features.

The backbone is not updated. PlantDoc images are encoded for the final test
only and are not used to choose the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from src.common import ensure_dir, load_config, set_seed
from src.datasets import class_names_from_map, load_class_map, load_split_rows, materialize_splits
from src.dinov2_encoder import load_dinov2, pool_tokens
from src.metrics import compute_metrics
from src.transforms import load_rgb, normalize


class _Rows(Dataset):
    def __init__(self, rows: list[dict], image_size: int):
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = normalize(load_rgb(row["image_path"], self.image_size))
        return image, int(row["class_id"])


def _encode(
    model,
    rows: list[dict],
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    pool_mode: str = "cls",
    topk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        _Rows(rows, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    labels = []
    use_patches = pool_mode != "cls"
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="encode", leave=False):
            tokens = model(images.to(device, non_blocking=True), return_patches=use_patches)
            encoded = pool_tokens(tokens, pool_mode, topk) if use_patches else tokens
            features.append(encoded.cpu())
            labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)


def extract_features(cfg: dict) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(cfg["dinov2_weights"]).to(device)
    paths = materialize_splits(cfg)
    source_train = load_split_rows(paths["source"], "train")
    source_val = load_split_rows(paths["source"], "val")
    target = load_split_rows(paths["target"], "test")
    image_size = int(cfg["image_size"])
    batch_size = int(cfg.get("extract_batch_size", 8))
    workers = int(cfg.get("num_workers", 4))
    pool_mode = str(cfg.get("pool_mode", "cls"))
    topk = int(cfg.get("salient_topk", 8))
    train_x, train_y = _encode(model, source_train, image_size, batch_size, workers, device, pool_mode, topk)
    val_x, val_y = _encode(model, source_val, image_size, batch_size, workers, device, pool_mode, topk)
    target_x, target_y = _encode(model, target, image_size, batch_size, workers, device, pool_mode, topk)
    output = Path(cfg["feature_cache"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "target_x": target_x,
            "target_y": target_y,
            "pool_mode": pool_mode,
        },
        output,
    )
    print(f"cached features train={tuple(train_x.shape)} val={tuple(val_x.shape)} target={tuple(target_x.shape)} -> {output}")
    return output


def _crop_table(cfg: dict) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    class_map = load_class_map(cfg["class_map"])
    ordered = sorted(class_map["classes"], key=lambda entry: int(entry["id"]))
    crops = []
    for entry in ordered:
        crop = entry["crop"]
        if crop not in crops:
            crops.append(crop)
    class_to_crop = torch.tensor([crops.index(entry["crop"]) for entry in ordered], dtype=torch.long)
    mask = torch.zeros(len(crops), len(ordered))
    for class_id, crop_id in enumerate(class_to_crop.tolist()):
        mask[crop_id, class_id] = 1.0
    return crops, class_to_crop, mask


def _class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    return counts.sum() / (num_classes * counts)


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> float:
    predictions = logits.argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(labels.cpu().numpy(), predictions, class_names)
    return float(metrics["macro_f1"])


def _predict_crop_gated(disease_logits: torch.Tensor, crop_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    crop_id = crop_logits.argmax(dim=1)
    allowed = mask[crop_id]
    gated = disease_logits.masked_fill(allowed == 0, -1e4)
    return gated


def _run_probe(cfg: dict, seed: int, linear_name: str = "frozen_linear", crop_name: str = "frozen_crop") -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(cfg["feature_cache"], map_location="cpu")
    names = class_names_from_map(cfg["class_map"])
    num_classes = len(names)
    crops, class_to_crop, crop_mask = _crop_table(cfg)
    train_x = cache["train_x"]
    train_y = cache["train_y"].long()
    val_x = cache["val_x"].to(device)
    val_y = cache["val_y"].long()
    target_x = cache["target_x"].to(device)
    target_y = cache["target_y"].long()
    weights = _class_weights(train_y, num_classes).to(device)
    crop_weights = _class_weights(class_to_crop[train_y], len(crops)).to(device)
    train_crop = class_to_crop[train_y]
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_crop),
        batch_size=int(cfg.get("probe_batch_size", 512)),
        shuffle=True,
        drop_last=False,
    )
    dim = train_x.shape[1]
    disease = torch.nn.Linear(dim, num_classes).to(device)
    crop_head = torch.nn.Linear(dim, len(crops)).to(device)
    optimizer = torch.optim.AdamW(
        list(disease.parameters()) + list(crop_head.parameters()),
        lr=float(cfg.get("probe_lr", 1e-3)),
        weight_decay=float(cfg.get("probe_weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("probe_epochs", 40))
    best = {"linear": -1.0, "crop": -1.0}
    best_state = {}
    crop_mask = crop_mask.to(device)
    for epoch in range(1, epochs + 1):
        disease.train()
        crop_head.train()
        total = 0.0
        for features, labels, crop_labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            crop_labels = crop_labels.to(device)
            disease_logits = disease(features)
            crop_logits = crop_head(features)
            loss = F.cross_entropy(disease_logits, labels, weight=weights)
            loss = loss + float(cfg.get("lambda_crop", 0.5)) * F.cross_entropy(crop_logits, crop_labels, weight=crop_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * labels.shape[0]
        disease.eval()
        crop_head.eval()
        with torch.no_grad():
            val_disease = disease(val_x)
            val_crop = crop_head(val_x)
            linear_f1 = _macro_f1(val_disease, val_y, names)
            gated = _predict_crop_gated(val_disease, val_crop, crop_mask)
            crop_f1 = _macro_f1(gated, val_y, names)
        print(
            f"seed {seed} epoch {epoch}: loss={total / len(train_y):.4f} "
            f"source_linear_f1={linear_f1:.4f} source_crop_f1={crop_f1:.4f}"
        )
        if linear_f1 >= best["linear"]:
            best["linear"] = linear_f1
            best_state["linear"] = {key: value.detach().cpu().clone() for key, value in disease.state_dict().items()}
        if crop_f1 >= best["crop"]:
            best["crop"] = crop_f1
            best_state["crop_disease"] = {key: value.detach().cpu().clone() for key, value in disease.state_dict().items()}
            best_state["crop_head"] = {key: value.detach().cpu().clone() for key, value in crop_head.state_dict().items()}

    disease.load_state_dict(best_state["linear"])
    _save_eval(cfg, linear_name, seed, disease, None, None, target_x, target_y, names, device, best["linear"])
    disease.load_state_dict(best_state["crop_disease"])
    crop_head.load_state_dict(best_state["crop_head"])
    _save_eval(cfg, crop_name, seed, disease, crop_head, crop_mask, target_x, target_y, names, device, best["crop"])


def _save_eval(cfg, variant, seed, disease, crop_head, crop_mask, target_x, target_y, names, device, source_f1):
    disease.eval()
    with torch.no_grad():
        disease_logits = disease(target_x)
        if crop_head is None:
            predictions = disease_logits.argmax(dim=1)
        else:
            crop_head.eval()
            predictions = _predict_crop_gated(disease_logits, crop_head(target_x), crop_mask).argmax(dim=1)
    metrics = compute_metrics(target_y.cpu().numpy(), predictions.cpu().numpy(), names)
    run_dir = ensure_dir(Path(cfg["output_dir"]) / variant / f"seed_{seed}")
    eval_dir = ensure_dir(run_dir / "eval" / "target")
    with (eval_dir / "target_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "target",
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    with (eval_dir / "target_per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    torch.save(
        {
            "disease": disease.state_dict(),
            "crop": None if crop_head is None else crop_head.state_dict(),
            "source_val_macro_f1": source_f1,
            "class_names": names,
        },
        run_dir / "best.pth",
    )
    print(
        f"{variant} seed {seed} target: accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
    )


class HybridHead(torch.nn.Module):
    """Equal mix of a cosine linear score and K visual prototypes per class.

    Features and prototypes are L2-normalized. One temperature, started at 0.07,
    scales both scores so the 0.5/0.5 mix is on the same range.
    """

    def __init__(self, dim: int, num_classes: int, prototypes_per_class: int):
        super().__init__()
        self.linear = torch.nn.Linear(dim, num_classes, bias=False)
        self.prototypes = torch.nn.Parameter(torch.zeros(num_classes, prototypes_per_class, dim))
        self.log_temp = torch.nn.Parameter(torch.tensor(-2.65926))

    def logits(self, features: torch.Tensor, mode: str) -> torch.Tensor:
        normalized = F.normalize(features, dim=-1)
        temperature = self.log_temp.exp().clamp(0.01, 1.0)
        weights = F.normalize(self.linear.weight, dim=-1)
        linear_logits = normalized @ weights.T / temperature
        prototypes = F.normalize(self.prototypes, dim=-1)
        similarity = torch.einsum("bd,ckd->bck", normalized, prototypes) / temperature
        prototype_logits = torch.logsumexp(similarity, dim=-1)
        if mode == "linear":
            return linear_logits
        if mode == "proto":
            return prototype_logits
        return 0.5 * linear_logits + 0.5 * prototype_logits


def _kmeans(features: torch.Tensor, groups: int, iterations: int = 25) -> torch.Tensor:
    count = features.shape[0]
    if count < groups:
        filler = features[torch.arange(groups - count) % count]
        return F.normalize(torch.cat((features, filler), dim=0), dim=-1)
    centers = features[torch.randperm(count)[:groups]].clone()
    for _ in range(iterations):
        assignment = torch.cdist(features, centers).argmin(dim=1)
        for index in range(groups):
            members = features[assignment == index]
            if members.shape[0] > 0:
                centers[index] = members.mean(dim=0)
    return F.normalize(centers, dim=-1)


def _init_hybrid(head: HybridHead, features: torch.Tensor, labels: torch.Tensor) -> None:
    normalized = F.normalize(features, dim=-1)
    groups = head.prototypes.shape[1]
    with torch.no_grad():
        for class_id in range(head.prototypes.shape[0]):
            class_features = normalized[labels == class_id]
            if class_features.shape[0] == 0:
                continue
            mean = F.normalize(class_features.mean(dim=0, keepdim=True), dim=-1)
            head.linear.weight[class_id] = mean[0]
            head.prototypes[class_id] = _kmeans(class_features, groups)


def _fit_scorer(cfg, seed, train_x, train_y, val_x, val_y, names, crop_labels, crop_mask, mode: str):
    device = val_x.device
    num_classes = len(names)
    groups = int(cfg.get("prototypes_per_class", 4))
    head = HybridHead(train_x.shape[1], num_classes, groups)
    _init_hybrid(head, train_x.cpu(), train_y.cpu())
    head = head.to(device)
    crop_head = torch.nn.Linear(train_x.shape[1], int(crop_mask.shape[0])).to(device)
    disease_weights = _class_weights(train_y, num_classes).to(device)
    crop_weights = _class_weights(crop_labels, int(crop_mask.shape[0])).to(device)
    loader = DataLoader(
        TensorDataset(train_x.cpu(), train_y.cpu(), crop_labels.cpu()),
        batch_size=int(cfg.get("probe_batch_size", 512)),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        list(head.parameters()) + list(crop_head.parameters()),
        lr=float(cfg.get("probe_lr", 1e-3)),
        weight_decay=float(cfg.get("probe_weight_decay", 1e-4)),
    )
    crop_lambda = float(cfg.get("lambda_crop", 0.5))
    best = {"open": -1.0, "gated": -1.0}
    best_state = {}
    for epoch in range(1, int(cfg.get("probe_epochs", 40)) + 1):
        head.train()
        crop_head.train()
        total = 0.0
        for features, labels, crops in loader:
            features = features.to(device)
            labels = labels.to(device)
            crops = crops.to(device)
            disease_logits = head.logits(features, mode)
            crop_logits = crop_head(F.normalize(features, dim=-1))
            loss = F.cross_entropy(disease_logits, labels, weight=disease_weights)
            loss = loss + crop_lambda * F.cross_entropy(crop_logits, crops, weight=crop_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * labels.shape[0]
        head.eval()
        crop_head.eval()
        with torch.no_grad():
            val_disease = head.logits(val_x, mode)
            val_crop = crop_head(F.normalize(val_x, dim=-1))
            open_f1 = _macro_f1(val_disease, val_y, names)
            gated_f1 = _macro_f1(_predict_crop_gated(val_disease, val_crop, crop_mask), val_y, names)
        print(
            f"seed {seed} {mode} epoch {epoch}: loss={total / len(train_y):.4f} "
            f"source_open_f1={open_f1:.4f} source_gated_f1={gated_f1:.4f}"
        )
        snapshot = {
            "head": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
            "crop": {key: value.detach().cpu().clone() for key, value in crop_head.state_dict().items()},
        }
        if open_f1 >= best["open"]:
            best["open"] = open_f1
            best_state["open"] = snapshot
        if gated_f1 >= best["gated"]:
            best["gated"] = gated_f1
            best_state["gated"] = snapshot
    return head, crop_head, best, best_state


def _eval_mode(cfg, variant, seed, head, crop_head, state, mode, gate, target_x, target_y, names, crop_mask, source_f1):
    head.load_state_dict(state["head"])
    crop_head.load_state_dict(state["crop"])
    head.eval()
    crop_head.eval()
    with torch.no_grad():
        disease_logits = head.logits(target_x, mode)
        if gate:
            crop_logits = crop_head(F.normalize(target_x, dim=-1))
            predictions = _predict_crop_gated(disease_logits, crop_logits, crop_mask).argmax(dim=1)
        else:
            predictions = disease_logits.argmax(dim=1)
    metrics = compute_metrics(target_y.cpu().numpy(), predictions.cpu().numpy(), names)
    run_dir = ensure_dir(Path(cfg["output_dir"]) / variant / f"seed_{seed}")
    eval_dir = ensure_dir(run_dir / "eval" / "target")
    with (eval_dir / "target_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "target",
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    with (eval_dir / "target_per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    torch.save({"state": state, "source_val_macro_f1": source_f1, "class_names": names}, run_dir / "best.pth")
    print(
        f"{variant} seed {seed} target: accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
    )


def _run_hybrid(cfg: dict, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(cfg["feature_cache"], map_location="cpu")
    names = class_names_from_map(cfg["class_map"])
    _, class_to_crop, crop_mask = _crop_table(cfg)
    train_x = cache["train_x"]
    train_y = cache["train_y"].long()
    val_x = cache["val_x"].to(device)
    val_y = cache["val_y"].long()
    target_x = cache["target_x"].to(device)
    target_y = cache["target_y"].long()
    crop_labels = class_to_crop[train_y]
    crop_mask = crop_mask.to(device)
    tag = str(cfg.get("resolution_tag", cfg.get("image_size", "224")))
    hybrid, hybrid_crop, hybrid_best, hybrid_state = _fit_scorer(
        cfg, seed, train_x, train_y, val_x, val_y, names, crop_labels, crop_mask, "hybrid"
    )
    _eval_mode(
        cfg, f"hybrid_{tag}", seed, hybrid, hybrid_crop, hybrid_state["open"], "hybrid", False,
        target_x, target_y, names, crop_mask, hybrid_best["open"],
    )
    _eval_mode(
        cfg, f"crop_hybrid_{tag}", seed, hybrid, hybrid_crop, hybrid_state["gated"], "hybrid", True,
        target_x, target_y, names, crop_mask, hybrid_best["gated"],
    )
    proto, proto_crop, proto_best, proto_state = _fit_scorer(
        cfg, seed, train_x, train_y, val_x, val_y, names, crop_labels, crop_mask, "proto"
    )
    _eval_mode(
        cfg, f"crop_proto_{tag}", seed, proto, proto_crop, proto_state["gated"], "proto", True,
        target_x, target_y, names, crop_mask, proto_best["gated"],
    )
    if tag != "224":
        _run_probe(cfg, seed, f"linear_{tag}", f"crop_linear_{tag}")


class CropExpertHead(torch.nn.Module):
    """Hard-routed per-crop disease experts on frozen features.

    A crop router picks the crop. Each crop has its own linear disease head over
    only that crop's classes. Experts never see other crops during their loss.
    """

    def __init__(self, dim: int, class_to_crop: torch.Tensor, num_crops: int):
        super().__init__()
        self.register_buffer("class_to_crop", class_to_crop.long())
        self.router = torch.nn.Linear(dim, num_crops)
        self.experts = torch.nn.ModuleList()
        self.local_to_global = []
        for crop_id in range(num_crops):
            members = (class_to_crop == crop_id).nonzero(as_tuple=False).flatten()
            self.local_to_global.append(members)
            self.experts.append(torch.nn.Linear(dim, int(members.numel())))

    def forward_logits(self, features: torch.Tensor):
        crop_logits = self.router(features)
        crop_id = crop_logits.argmax(dim=1)
        global_logits = features.new_full((features.shape[0], int(self.class_to_crop.numel())), -1e4)
        for index, expert in enumerate(self.experts):
            selected = crop_id == index
            if not selected.any():
                continue
            local = expert(features[selected])
            members = self.local_to_global[index].to(features.device)
            rows = selected.nonzero(as_tuple=False).flatten()
            global_logits[rows.unsqueeze(1), members.unsqueeze(0)] = local
        return global_logits, crop_logits

    def expert_losses(self, features: torch.Tensor, labels: torch.Tensor, crop_labels: torch.Tensor) -> torch.Tensor:
        loss = features.new_zeros(())
        count = 0
        for index, expert in enumerate(self.experts):
            selected = crop_labels == index
            if not selected.any():
                continue
            members = self.local_to_global[index].to(features.device)
            remap = {int(global_id): local_id for local_id, global_id in enumerate(members.tolist())}
            local_labels = torch.tensor([remap[int(value)] for value in labels[selected].tolist()], device=features.device)
            loss = loss + F.cross_entropy(expert(features[selected]), local_labels)
            count += 1
        return loss / max(count, 1)


def _run_experts(cfg: dict, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(cfg["feature_cache"], map_location="cpu")
    names = class_names_from_map(cfg["class_map"])
    crops, class_to_crop, crop_mask = _crop_table(cfg)
    train_x = cache["train_x"]
    train_y = cache["train_y"].long()
    val_x = cache["val_x"].to(device)
    val_y = cache["val_y"].long()
    target_x = cache["target_x"].to(device)
    target_y = cache["target_y"].long()
    train_crop = class_to_crop[train_y]
    crop_weights = _class_weights(train_crop, len(crops)).to(device)
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_crop),
        batch_size=int(cfg.get("probe_batch_size", 512)),
        shuffle=True,
    )
    model = CropExpertHead(train_x.shape[1], class_to_crop, len(crops)).to(device)
    # also train a flat crop-gated linear on the same richer features for a fair ablation
    flat = torch.nn.Linear(train_x.shape[1], len(names)).to(device)
    crop_head = torch.nn.Linear(train_x.shape[1], len(crops)).to(device)
    class_weights = _class_weights(train_y, len(names)).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(flat.parameters()) + list(crop_head.parameters()),
        lr=float(cfg.get("probe_lr", 1e-3)),
        weight_decay=float(cfg.get("probe_weight_decay", 1e-4)),
    )
    crop_mask = crop_mask.to(device)
    tag = str(cfg.get("feature_tag", cfg.get("pool_mode", "feat")))
    best = {"experts": -1.0, "crop_linear": -1.0}
    best_state = {}
    for epoch in range(1, int(cfg.get("probe_epochs", 40)) + 1):
        model.train()
        flat.train()
        crop_head.train()
        total = 0.0
        for features, labels, crop_labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            crop_labels = crop_labels.to(device)
            router_loss = F.cross_entropy(model.router(features), crop_labels, weight=crop_weights)
            expert_loss = model.expert_losses(features, labels, crop_labels)
            flat_logits = flat(features)
            crop_logits = crop_head(features)
            flat_loss = F.cross_entropy(flat_logits, labels, weight=class_weights)
            flat_loss = flat_loss + float(cfg.get("lambda_crop", 0.5)) * F.cross_entropy(crop_logits, crop_labels, weight=crop_weights)
            loss = router_loss + expert_loss + flat_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * labels.shape[0]
        model.eval()
        flat.eval()
        crop_head.eval()
        with torch.no_grad():
            expert_logits, _ = model.forward_logits(val_x)
            expert_f1 = _macro_f1(expert_logits, val_y, names)
            gated = _predict_crop_gated(flat(val_x), crop_head(val_x), crop_mask)
            crop_linear_f1 = _macro_f1(gated, val_y, names)
        print(
            f"seed {seed} epoch {epoch}: loss={total / len(train_y):.4f} "
            f"source_experts_f1={expert_f1:.4f} source_crop_linear_f1={crop_linear_f1:.4f}"
        )
        if expert_f1 >= best["experts"]:
            best["experts"] = expert_f1
            best_state["experts"] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if crop_linear_f1 >= best["crop_linear"]:
            best["crop_linear"] = crop_linear_f1
            best_state["flat"] = {key: value.detach().cpu().clone() for key, value in flat.state_dict().items()}
            best_state["crop"] = {key: value.detach().cpu().clone() for key, value in crop_head.state_dict().items()}

    model.load_state_dict(best_state["experts"])
    model.eval()
    with torch.no_grad():
        predictions = model.forward_logits(target_x)[0].argmax(dim=1)
    metrics = compute_metrics(target_y.cpu().numpy(), predictions.cpu().numpy(), names)
    _write_metrics(cfg, f"crop_experts_{tag}", seed, metrics, best["experts"], {"experts": best_state["experts"]}, names)

    flat.load_state_dict(best_state["flat"])
    crop_head.load_state_dict(best_state["crop"])
    flat.eval()
    crop_head.eval()
    with torch.no_grad():
        predictions = _predict_crop_gated(flat(target_x), crop_head(target_x), crop_mask).argmax(dim=1)
    metrics = compute_metrics(target_y.cpu().numpy(), predictions.cpu().numpy(), names)
    _write_metrics(
        cfg,
        f"crop_linear_{tag}",
        seed,
        metrics,
        best["crop_linear"],
        {"flat": best_state["flat"], "crop": best_state["crop"]},
        names,
    )


def _write_metrics(cfg, variant, seed, metrics, source_f1, state, names):
    run_dir = ensure_dir(Path(cfg["output_dir"]) / variant / f"seed_{seed}")
    eval_dir = ensure_dir(run_dir / "eval" / "target")
    with (eval_dir / "target_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "target",
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    with (eval_dir / "target_per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    torch.save({"state": state, "source_val_macro_f1": source_f1, "class_names": names}, run_dir / "best.pth")
    print(
        f"{variant} seed {seed} target: accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen DINOv2 lab-to-field probes.")
    parser.add_argument("--config", default="configs/frozen.yaml")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.extract:
        extract_features(cfg)
    if args.train:
        head = cfg.get("head", "linear")
        if head == "hybrid":
            _run_hybrid(cfg, args.seed)
        elif head == "experts":
            _run_experts(cfg, args.seed)
        else:
            _run_probe(cfg, args.seed)
    if not args.extract and not args.train:
        raise SystemExit("Pass --extract, --train, or both.")


if __name__ == "__main__":
    main()
