# Agriculture_Proj

Lab-to-field plant leaf disease recognition: PlantVillage (source) → PlantDoc (unseen target), 28 matched classes.

## Layout

- `Code/` — training, evaluation, frozen DINOv2 probes, configs, scripts
- `Resource Papper/` — reference papers
- `Resource/` — local datasets, GitHub clones, and model weights (not in git; see `Resource/SOURCES.txt` when present)

## Environment

Use the `plantdg` conda env (PyTorch 1.10 + CUDA 11.1). Run from `Code/`:

```bash
cd Code
python -m src.train_frozen --config configs/frozen.yaml --extract
python -m src.train_frozen --config configs/frozen.yaml --train --seed 42
```

Long jobs should be started in tmux (see `Code/scripts/`).

## Protocol

- Train and select checkpoints on PlantVillage only
- PlantDoc is final evaluation only (no training / early stopping / HPO on target)
- Report accuracy, precision, recall, F1, macro-F1, and confusion matrix
