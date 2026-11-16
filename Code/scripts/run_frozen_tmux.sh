#!/usr/bin/env bash
# Encode every image once with frozen DINOv2, then train the linear probe and
# the crop-gated head. PlantDoc is not used to choose checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/DATA/ramendra_2511ai39/miniconda3/envs/plantdg/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "===== $* $(date) ====="
}

echo "START $(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "extract"
"$PY" -m src.train_frozen --config configs/frozen.yaml --extract
for seed in 42 1 2; do
  log "train seed ${seed}"
  "$PY" -m src.train_frozen --config configs/frozen.yaml --train --seed "${seed}"
done
"$PY" scripts/collect_results.py
echo "ALL_DONE $(date)"
