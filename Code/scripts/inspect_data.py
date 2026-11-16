"""Count images, flag unreadable files, and save a sample grid.

Run from the Code directory after the dataset clones have finished:

    python scripts/inspect_data.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ensure_dir, load_config
from src.datasets import materialize_splits, read_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the source and target datasets.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default="results/inspection")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = materialize_splits(cfg)
    rows = read_table(paths["inventory"])
    output = ensure_dir(ROOT / args.output)
    corrupt = []
    sizes = []
    for row in rows:
        try:
            with Image.open(row["image_path"]) as image:
                image.verify()
            with Image.open(row["image_path"]) as image:
                sizes.append(image.size)
        except (UnidentifiedImageError, OSError) as exc:
            corrupt.append({"image_path": row["image_path"], "error": str(exc)})

    counts = Counter((row["domain"], row["split"], row["class_name"]) for row in rows)
    count_rows = [
        {"domain": domain, "split": split, "class_name": name, "count": count}
        for (domain, split, name), count in sorted(counts.items())
    ]
    with (output / "class_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "split", "class_name", "count"])
        writer.writeheader()
        writer.writerows(count_rows)
    with (output / "corrupt_files.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "error"])
        writer.writeheader()
        writer.writerows(corrupt)

    sample = random.Random(0).sample(rows, k=min(8, len(rows)))
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, row in zip(axes.ravel(), sample):
        with Image.open(row["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{row['domain']}:{row['class_name']}"[:32], fontsize=8)
        axis.axis("off")
    for axis in axes.ravel()[len(sample) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "sample_grid.png", dpi=120)
    plt.close(figure)

    widths = [size[0] for size in sizes]
    heights = [size[1] for size in sizes]
    report = output / "inspection_report.txt"
    report.write_text(
        "\n".join(
            [
                f"images {len(rows)}",
                f"corrupt {len(corrupt)}",
                f"source train {sum(1 for row in rows if row['domain']=='source' and row['split']=='train')}",
                f"source val {sum(1 for row in rows if row['domain']=='source' and row['split']=='val')}",
                f"target test {sum(1 for row in rows if row['domain']=='target')}",
                f"width min/max {min(widths) if widths else 0}/{max(widths) if widths else 0}",
                f"height min/max {min(heights) if heights else 0}/{max(heights) if heights else 0}",
                f"inventory {paths['inventory']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
