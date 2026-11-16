"""Verify the hand-checked class map against folders on disk and print exclusions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config
from src.datasets import build_records, index_class_folders, load_class_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Check source and target class folders.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    class_map = load_class_map(cfg["class_map"])
    print(f"mapped classes {len(class_map['classes'])}")
    source = build_records(cfg["plantvillage_color"], class_map, "source")
    target = build_records(cfg["plantdoc_root"], class_map, "target")
    print(f"source images {len(source)}")
    print(f"target images {len(target)}")
    print("excluded:")
    for item in class_map.get("excluded", []):
        print(f"  {item['dataset']} {item['folder']}: {item['reason']}")
    target_index = index_class_folders(cfg["plantdoc_root"])
    used = set()
    for entry in class_map["classes"]:
        for alias in entry["target_folders"]:
            from src.datasets import _norm_name

            used.add(_norm_name(alias))
    unused = sorted(set(target_index) - used)
    if unused:
        print("target folders not used by the map:")
        for name in unused:
            example = Path(target_index[name][0]).parent.name
            print(f"  {example} ({len(target_index[name])} images)")
    report = {
        "source_images": len(source),
        "target_images": len(target),
        "num_classes": len(class_map["classes"]),
    }
    out = Path(cfg["split_dir"]) / "mapping_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
