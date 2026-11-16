# Counterfactual domain generalization for plant disease recognition

Laboratory images (PlantVillage) are the source domain. Field images (PlantDoc) are the unseen target. Target images are not used for training or for choosing checkpoints.

The numbered workflow in `Counterfactual_Plant_Disease_Implementation_Guide.pdf` is implemented here.

## Layout

- `src/train_baseline.py` — plain ResNet-50 or ViT, source labels only
- `src/train_dgpldr.py` — style-sensitive distillation and logit-prototype matching
- `src/train_proposed.py` — counterfactual consistency and prototype stability
- `src/counterfactual.py` — background, illumination, and style interventions that keep the leaf
- `src/evaluate.py` — accuracy, precision, recall, macro-F1, balanced accuracy, confusion matrix
- `data/class_maps/plantvillage_plantdoc.json` — the 28 matched classes and the excluded ones

Cloned paper repositories and the Mendeley downloads live in `../Resource`.

## Environment

The driver on this machine is CUDA 11.2, so the environment is Python 3.9 with PyTorch 1.10.2 built for CUDA 11.1. Newer CUDA wheels need a newer driver.

```bash
cd /DATA/ramendra_2511ai39/Agriculture_project/Code
conda activate plantdg
pip install -r requirements.txt
python gpu_check.py
```

## Data

DG-PLDR's own benchmark images are not in the GitHub repo. The README asks for an organization email to `seabearlmx@gmail.com`. Until those arrive, the public pair is:

- source: `Resource/github/PlantVillage-Dataset` (color images)
- target: `Resource/github/PlantDoc-Dataset`
- also downloaded: both Mendeley zips from `https://data.mendeley.com/datasets/tywbtsjrjv/1`

```bash
python scripts/build_class_map.py --config configs/baseline.yaml
python scripts/inspect_data.py --config configs/baseline.yaml
python scripts/generate_cf.py --config configs/counterfactual.yaml
```

Source validation is a stratified 80/20 split of PlantVillage, seed 42. PlantDoc train and test folders are both treated as the unseen target.

## Training

Checkpoints are selected by source-validation macro-F1.

```bash
python -m src.train_baseline --config configs/baseline.yaml
python -m src.train_dgpldr --config configs/dgpldr.yaml
python -m src.train_proposed --config configs/proposed.yaml --variant cf_aug
python -m src.train_proposed --config configs/proposed.yaml --variant cf_consistency
python -m src.train_proposed --config configs/proposed.yaml --variant proto_stability
python -m src.train_proposed --config configs/proposed.yaml --variant full
```

A one-epoch smoke test:

```bash
python -m src.train_baseline --config configs/baseline.yaml --epochs 1 --limit 32 --batch-size 8
```

## Evaluation

Run this only after training. `--split target` is the field test.

```bash
python -m src.evaluate --config configs/baseline.yaml \
  --checkpoint results/runs/baseline/seed_42/best.pth --split target
python scripts/nuisance_test.py --checkpoint results/runs/full/seed_42/best.pth
python scripts/collect_results.py
```

Repeat with `--seed 1` and `--seed 2` when the machine is free, then report mean and standard deviation from `results/tables/main_results.csv`.

## What was adapted from the DG-PLDR paper

The public code is a source-train / target-test mmpretrain setup. SSD, the logit memory bank, and logit-prototype matching are reimplemented from Section 4 of Zhan et al. (2025). SSD runs on the last ResNet map, relations are cosine relations, and the threshold is a steep sigmoid initialized at 0.5. Those changes are commented in `src/dgpldr.py`. Hyperparameters follow the paper where they apply here: lambda 0.1, momentum 0.99, temperature 0.8. The learning rate is 0.01 rather than the paper's 0.045 because this is ImageNet-initialized fine-tuning on the public fallback pair, not their email-gated benchmark.

## Tests

```bash
python -m unittest tests.test_losses
```
