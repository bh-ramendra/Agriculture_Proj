"""Source/target datasets for laboratory-to-field plant disease recognition.

Source images are split into train and source-validation. Target images are
test-only. Target labels are never read by the training loop for model selection.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.counterfactual import make_boundary_counterfactuals, make_counterfactuals
from src.transforms import load_rgb, maybe_hflip, normalize, photometric_style_view


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_class_map(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def index_class_folders(root: str | Path) -> dict[str, list[str]]:
    """Map a normalized folder name to image paths.

    A class folder is any directory that directly contains images. This accepts
    both ImageFolder layouts and train/class or test/class layouts.
    """
    root = Path(root)
    if not root.exists():
        return {}
    found: dict[str, list[str]] = defaultdict(list)
    for directory, dirnames, filenames in os_walk_skip_hidden(root):
        if Path(directory).resolve() == root.resolve():
            continue
        images = [
            str(Path(directory) / name)
            for name in filenames
            if Path(name).suffix.lower() in IMAGE_SUFFIXES
        ]
        if not images:
            continue
        found[_norm_name(Path(directory).name)].extend(sorted(images))
    return found


def os_walk_skip_hidden(root: Path):
    import os

    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        yield directory, dirnames, filenames


def _match_folders(index: dict[str, list[str]], aliases: list[str]) -> tuple[str | None, list[str]]:
    for alias in aliases:
        key = _norm_name(alias)
        if key in index and index[key]:
            return alias, index[key]
    return None, []


def build_records(root: str | Path, class_map: dict, side: str) -> list[dict]:
    index = index_class_folders(root)
    alias_key = "source_folders" if side == "source" else "target_folders"
    records = []
    missing = []
    for entry in class_map["classes"]:
        alias, paths = _match_folders(index, entry[alias_key])
        if not paths:
            missing.append(entry["name"])
            continue
        for path in paths:
            records.append(
                {
                    "image_path": path,
                    "class_name": entry["name"],
                    "class_id": int(entry["id"]),
                    "crop": entry["crop"],
                    "matched_folder": alias,
                }
            )
    if missing:
        available = sorted({Path(p).parent.name for paths in index.values() for p in paths[:1]})
        raise FileNotFoundError(
            f"Could not match {side} folders for: {missing}. "
            f"Directories seen under {root}: {available[:40]}"
        )
    return records


def stratified_split(records: list[dict], train_ratio: float, seed: int) -> list[dict]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_class[record["class_id"]].append(record)
    rng = np.random.default_rng(seed)
    split_records = []
    for class_id, items in sorted(by_class.items()):
        items = list(items)
        rng.shuffle(items)
        if len(items) == 1:
            train_count = 1
        else:
            train_count = max(1, min(len(items) - 1, int(round(len(items) * train_ratio))))
        for index, item in enumerate(items):
            copied = dict(item)
            copied["split"] = "train" if index < train_count else "val"
            split_records.append(copied)
    return split_records


def write_table(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_table(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def materialize_splits(cfg: dict) -> dict[str, str]:
    """Create source train/val and target test tables if they are not on disk."""
    class_map = load_class_map(cfg["class_map"])
    split_dir = Path(cfg["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)
    source_csv = split_dir / "source_split.csv"
    target_csv = split_dir / "target_split.csv"
    inventory_csv = split_dir / "data_inventory.csv"
    if source_csv.exists() and target_csv.exists() and inventory_csv.exists():
        return {
            "source": str(source_csv),
            "target": str(target_csv),
            "inventory": str(inventory_csv),
        }

    source_records = stratified_split(
        build_records(cfg["plantvillage_color"], class_map, "source"),
        train_ratio=float(cfg.get("source_train_ratio", 0.8)),
        seed=int(cfg.get("seed", 42)),
    )
    target_records = build_records(cfg["plantdoc_root"], class_map, "target")
    for record in target_records:
        record["split"] = "test"

    fields = ["image_path", "class_name", "class_id", "crop", "matched_folder", "split"]
    write_table(source_csv, source_records, fields)
    write_table(target_csv, target_records, fields)

    inventory = []
    for record, dataset_name, domain in (
        (source_records, "PlantVillage", "source"),
        (target_records, "PlantDoc", "target"),
    ):
        for item in record:
            inventory.append(
                {
                    "image_path": item["image_path"],
                    "class_name": item["class_name"],
                    "crop": item["crop"],
                    "dataset_name": dataset_name,
                    "domain": domain,
                    "split": item["split"],
                }
            )
    write_table(
        inventory_csv,
        inventory,
        ["image_path", "class_name", "crop", "dataset_name", "domain", "split"],
    )
    return {"source": str(source_csv), "target": str(target_csv), "inventory": str(inventory_csv)}


def _limit_rows(rows: list[dict], limit: int | None) -> list[dict]:
    """Keep a class-balanced subset so a smoke test is not a single disease."""
    rows = list(rows)
    if not limit or limit >= len(rows):
        return rows
    by_class: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)
    rng = random.Random(0)
    for items in by_class.values():
        rng.shuffle(items)
    picked: list[dict] = []
    class_ids = list(by_class)
    while len(picked) < limit and any(by_class.values()):
        rng.shuffle(class_ids)
        for class_id in class_ids:
            if by_class[class_id] and len(picked) < limit:
                picked.append(by_class[class_id].pop())
    return picked


class PlantDiseaseDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        image_size: int = 224,
        train: bool = False,
        return_counterfactual: bool = False,
        return_style_view: bool = False,
        num_cf_views: int = 2,
        hflip: bool = True,
        limit: int | None = None,
        cf_mode: str = "legacy",
        stress: bool = False,
    ):
        self.rows = _limit_rows(rows, limit)
        self.image_size = image_size
        self.train = train
        self.return_counterfactual = return_counterfactual
        self.return_style_view = return_style_view
        self.num_cf_views = num_cf_views
        self.hflip = hflip and train and not stress
        self.cf_mode = cf_mode
        self.stress = stress

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        try:
            image = load_rgb(row["image_path"], self.image_size)
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Could not read {row['image_path']}") from exc
        image = maybe_hflip(image, self.hflip)
        if self.stress:
            generator = torch.Generator()
            generator.manual_seed(10_000 + int(index))
            stressed, _, _, _, _ = make_boundary_counterfactuals(
                image, num_views=1, generator=generator, stress=True
            )
            image = stressed[0]
        item = {
            "image": image,
            "label": int(row["class_id"]),
            "path": row["image_path"],
            "class_name": row["class_name"],
        }
        if self.return_style_view:
            item["style_image"] = photometric_style_view(image)
        if self.return_counterfactual:
            if self.cf_mode == "boundary":
                counterfactuals, preserve, preserve_cf, background, background_cf = make_boundary_counterfactuals(
                    image, self.num_cf_views
                )
                item["counterfactual"] = counterfactuals
                item["interventions"] = ["boundary"] * self.num_cf_views
                item["preserve_mask"] = preserve
                item["preserve_mask_cf"] = preserve_cf
                item["background_mask"] = background
                item["background_mask_cf"] = background_cf
            else:
                counterfactuals, names = make_counterfactuals(image, self.num_cf_views)
                item["counterfactual"] = counterfactuals
                item["interventions"] = names
        return item


def collate_batch(batch: list[dict]) -> dict:
    images = normalize(torch.stack([item["image"] for item in batch], dim=0))
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    collated = {
        "image": images,
        "label": labels,
        "path": [item["path"] for item in batch],
        "class_name": [item["class_name"] for item in batch],
    }
    if "style_image" in batch[0]:
        collated["style_image"] = normalize(torch.stack([item["style_image"] for item in batch], dim=0))
    if "counterfactual" in batch[0]:
        counterfactuals = torch.stack([item["counterfactual"] for item in batch], dim=0)
        collated["counterfactual"] = normalize(counterfactuals)
        collated["interventions"] = batch[0]["interventions"]
    if "preserve_mask" in batch[0]:
        collated["preserve_mask"] = torch.stack([item["preserve_mask"] for item in batch], dim=0)
        collated["preserve_mask_cf"] = torch.stack([item["preserve_mask_cf"] for item in batch], dim=0)
        collated["background_mask"] = torch.stack([item["background_mask"] for item in batch], dim=0)
        collated["background_mask_cf"] = torch.stack([item["background_mask_cf"] for item in batch], dim=0)
    return collated


def load_split_rows(csv_path: str, split: str) -> list[dict]:
    rows = [row for row in read_table(csv_path) if row["split"] == split]
    for row in rows:
        row["class_id"] = int(row["class_id"])
    return rows


def class_names_from_map(path: str | Path) -> list[str]:
    class_map = load_class_map(path)
    ordered = sorted(class_map["classes"], key=lambda entry: int(entry["id"]))
    return [entry["name"] for entry in ordered]
