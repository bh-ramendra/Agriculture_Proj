#!/usr/bin/env bash
# DG-PLDR is already trained. This run adds the hard-augmentation control and
# the lesion-anchored model, then evaluates both on PlantDoc.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/DATA/ramendra_2511ai39/miniconda3/envs/plantdg/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "===== $* $(date) ====="
}

run() {
  log "START $*"
  "$PY" "$@"
  log "DONE $*"
}

echo "START $(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Target to beat: DG-PLDR PlantDoc accuracy 0.192 macro-F1 0.151"

for seed in 42 1 2; do
  log "seed ${seed}"
  run -m src.train_proposed --config configs/lesion_dg.yaml --variant hard_aug --seed "${seed}"
  run -m src.evaluate --config configs/lesion_dg.yaml \
    --checkpoint "results/runs/hard_aug/seed_${seed}/best.pth" --split target
  run -m src.train_proposed --config configs/lesion_dg.yaml --variant lesion_dg --seed "${seed}"
  run -m src.evaluate --config configs/lesion_dg.yaml \
    --checkpoint "results/runs/lesion_dg/seed_${seed}/best.pth" --split target
  run scripts/collect_results.py
done

echo "ALL_DONE $(date)"
