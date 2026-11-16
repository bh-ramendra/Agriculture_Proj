#!/usr/bin/env bash
# Full public-pair experiment. One method at a time, then its target evaluation.
# Seeds: 42, then 1, then 2. Source validation selects the checkpoint.
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

for seed in 42 1 2; do
  log "seed ${seed}"

  run -m src.train_baseline --config configs/baseline.yaml --seed "${seed}"
  run -m src.evaluate --config configs/baseline.yaml \
    --checkpoint "results/runs/baseline/seed_${seed}/best.pth" --split target

  run -m src.train_dgpldr --config configs/dgpldr.yaml --seed "${seed}"
  run -m src.evaluate --config configs/dgpldr.yaml \
    --checkpoint "results/runs/dgpldr/seed_${seed}/best.pth" --split target

  for variant in cf_aug cf_consistency proto_stability full; do
    run -m src.train_proposed --config configs/proposed.yaml --variant "${variant}" --seed "${seed}"
    run -m src.evaluate --config configs/proposed.yaml \
      --checkpoint "results/runs/${variant}/seed_${seed}/best.pth" --split target
  done

  run scripts/nuisance_test.py --config configs/proposed.yaml \
    --checkpoint "results/runs/full/seed_${seed}/best.pth"
  run scripts/collect_results.py
done

echo "ALL_DONE $(date)"
