#!/usr/bin/env bash
# Reproducible entry point. Full training is opt-in because the shared GPUs
# may already be in use and a 20-epoch run is long.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m unittest tests.test_losses
python scripts/build_class_map.py --config configs/baseline.yaml
python scripts/inspect_data.py --config configs/baseline.yaml
python scripts/generate_cf.py --config configs/counterfactual.yaml --num-images 6

if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m src.train_baseline --config configs/baseline.yaml --epochs 1 --limit 32 --batch-size 8
fi

cat <<'EOF'
Next commands, from the Code directory. Target labels are used only by evaluate.py.

python gpu_check.py
python -m src.train_baseline --config configs/baseline.yaml
python -m src.evaluate --config configs/baseline.yaml --checkpoint results/runs/baseline/seed_42/best.pth --split target

python -m src.train_dgpldr --config configs/dgpldr.yaml
python -m src.train_proposed --config configs/proposed.yaml --variant cf_aug
python -m src.train_proposed --config configs/proposed.yaml --variant cf_consistency
python -m src.train_proposed --config configs/proposed.yaml --variant proto_stability
python -m src.train_proposed --config configs/proposed.yaml --variant full
python scripts/collect_results.py
EOF
