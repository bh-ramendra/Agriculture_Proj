"""Collect finished target-domain evaluations into one comparison table.

Point this at results/runs after each variant has an eval/target/target_metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ablation / main results table.")
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--output", default="results/tables/main_results.csv")
    args = parser.parse_args()
    rows = []
    for metrics in sorted(Path(args.runs).glob("*/seed_*/eval/target/target_metrics.csv")):
        if metrics.parents[2].name == "seed_0":
            continue
        with metrics.open("r", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                record["variant"] = metrics.parents[3].name
                record["seed"] = metrics.parents[2].name
                rows.append(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("variant,seed,split,accuracy,balanced_accuracy,precision,recall,macro_f1\n", encoding="utf-8")
        print(f"no target metrics yet; wrote header to {output}")
        return
    fieldnames = ["variant", "seed", "split", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
