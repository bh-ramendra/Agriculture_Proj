#!/usr/bin/env bash
# Crop-gated hybrid head on frozen DINOv2.
# First the cached 224 features, then a fresh encode at the native 518 size.
# PlantDoc is not used to choose checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/DATA/ramendra_2511ai39/miniconda3/envs/plantdg/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "===== $* $(date) ====="
}

echo "START $(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
for seed in 42 1 2; do
  log "hybrid 224 seed ${seed}"
  "$PY" -m src.train_frozen --config configs/hybrid224.yaml --train --seed "${seed}"
done
log "extract 518"
"$PY" -m src.train_frozen --config configs/hybrid518.yaml --extract
for seed in 42 1 2; do
  log "hybrid 518 seed ${seed}"
  "$PY" -m src.train_frozen --config configs/hybrid518.yaml --train --seed "${seed}"
done
"$PY" scripts/collect_results.py
echo "ALL_DONE $(date)"
