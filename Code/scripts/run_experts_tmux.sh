#!/usr/bin/env bash
# Re-encode with CLS + patch-mean + salient patches, then train hard-routed
# per-crop experts and a crop-gated linear head on the same features.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/DATA/ramendra_2511ai39/miniconda3/envs/plantdg/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

log() {
  echo "===== $* $(date) ====="
}

echo "START $(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "extract salient features"
"$PY" -m src.train_frozen --config configs/experts.yaml --extract
for seed in 42 1 2; do
  log "experts seed ${seed}"
  "$PY" -m src.train_frozen --config configs/experts.yaml --train --seed "${seed}"
done
"$PY" scripts/collect_results.py
echo "ALL_DONE $(date)"
