# Agriculture_Proj

**Lab-to-field domain generalization for plant leaf disease recognition**

Source domain: PlantVillage color (laboratory images)  
Target domain: PlantDoc (field / in-the-wild images, **unseen**)  
Classes: **28** matched crop–disease labels  
Repository: [https://github.com/bh-ramendra/Agriculture_Proj](https://github.com/bh-ramendra/Agriculture_Proj)

This README documents **everything done so far**: how the project started, what we followed, what we changed, every method we trained, and the final accuracy numbers.

---

## 1. Goal

Train a model **only on clean laboratory leaf images** and test it on **real field photographs**, without using any PlantDoc labels (or images) for training, early stopping, or hyperparameter selection.

We do **not** claim “first-ever” or SOTA. Absolute field accuracy in this protocol is still far from deployment-ready diagnosis. The useful comparison is: **beat a strong source-only baseline under the same split and metrics**.

Chance accuracy on 28 classes ≈ **3.6%** (`1/28`).

---

## 2. How we started

1. Project folder created as `Agriculture_project/` (also tracked as **Agriculture_Proj** on GitHub).
2. Starting materials in the folder:
   - `Counterfactual Plant Disease Recognition Pipeline.png` (flowchart)
   - `Counterfactual_Plant_Disease_Implementation_Guide.pdf` (implementation guide)
   - `Resource Papper/` — reference PDFs (DG-PLDR, DCMF, frozen foundation-model features, PiTLiD, causal proposal, …)
3. From those papers and the guide we:
   - Cloned related GitHub repos into `Resource/github/`
   - Downloaded dataset archives into `Resource/datasets/`
   - Implemented **all runnable code in `Code/`** (not inside the cloned repos)
4. Environment created: conda env **`plantdg`**
   - Python 3.9
   - PyTorch **1.10.2+cu111**, torchvision **0.11.3+cu111**
   - Driver/CUDA limit on this machine: driver 460.x → stay on **cu111** (do not switch to cu113/cu118)
   - Hardware used: shared **4× Tesla V100 32GB**; long jobs always in **tmux**

Data / model blobs stay **local only** (gitignored). Sources are listed in `Resource/SOURCES.txt`.

---

## 3. Protocol we follow (strict)

| Rule | Detail |
|------|--------|
| Source | PlantVillage **color** (`Resource/github/PlantVillage-Dataset/raw/color`) |
| Target | PlantDoc `train/` + `test/` folders treated as **one unseen test set** |
| Classes | Hand-authored map: `Code/data/class_maps/plantvillage_plantdoc.json` (**28** classes, ids 0–27, with `crop` field) |
| Source split | Stratified **80/20** train/val, **seed 42** → CSVs under `Code/data/class_maps/` |
| Counts | Source **38,542** (train 30,835 / val 7,707); target **2,578**; 0 corrupt indexed |
| Selection | Checkpoint by **source-validation** macro-F1 (or a **source stress** set for some lesion variants). **Never** by PlantDoc |
| Metrics | Accuracy, balanced accuracy, precision, recall, macro-F1, per-class F1, confusion matrix |
| Seeds for final tables | **42, 1, 2** (smoke `seed_0` / tiny limits are **not** results) |
| Do not | Change the class map after looking at target results; regenerate splits once written; train on PlantDoc; claim 70–94% numbers from models trained **on** PlantDoc |

Official DG-PLDR Apple benchmark paths are email-gated; until those arrive, the **public PlantVillage → PlantDoc** pair is the experiment.

---

## 4. Project layout

```
Agriculture_project/
├── README.md                          ← this file
├── Counterfactual_*.pdf / *.png       ← original guide + flowchart
├── Resource Papper/                   ← reference papers (PDFs)
├── Resource/
│   ├── SOURCES.txt                    ← download / clone URLs
│   ├── github/                        ← cloned repos + datasets (local, gitignored)
│   ├── datasets/                      ← Mendeley zips extracted (local, gitignored)
│   └── models/                        ← DINOv2 weights (local, gitignored)
└── Code/                              ← all project code
    ├── configs/                       ← YAML for each experiment
    ├── data/class_maps/               ← JSON map + split CSVs
    ├── src/                           ← models, losses, training, frozen DINOv2
    ├── scripts/                       ← tmux runners, eval collectors, audits
    ├── tests/                         ← unit tests
    ├── results/                       ← runs + tables (local, gitignored)
    └── logs/                          ← tmux logs (local, gitignored)
```

**GitHub remote:** `git@github.com:bh-ramendra/Agriculture_Proj.git`  
Tracked: code, configs, class maps, guides, papers, README.  
Not tracked: datasets, cloned full repos, `.pth` weights, `results/`, `logs/`.

---

## 5. What we implemented (timeline of changes)

### Phase A — Data + baselines + DG-PLDR reimplementation

- Built the **28-class map** and materialised `source_split.csv` / `target_split.csv`.
- Fixed indexing so PlantDoc root files (e.g. example PNG) are not treated as a class.
- `--limit` made **class-balanced** (not “first N rows of one class”).
- **Baseline:** ImageNet ResNet-50 + cross-entropy only.
- **DG-PLDR** (Zhan et al., EAAI 2025): official public repo is source-train/target-test only; **SSD / LPMB / LPMS are not in the released code**. We reimplemented them from the paper in `Code/src/dgpldr.py`, `losses.py`, `prototypes.py` with documented, dimensionally valid deviations (last ResNet stage, L2-normalised tokens, row-wise softmax, steep sigmoid gate, eval self-pairing so SSD feature space matches train).
- Hyperparameters for this public pair: λ_lpcl=0.1, momentum 0.99, τ=0.8, SGD lr **0.01** (paper used 0.045 on their gated benchmark).

### Phase B — Counterfactual / disease-preserving methods (failed to beat DG-PLDR)

Idea: edit only background / illumination / style; keep disease; add consistency losses.

Variants trained (seeds 42, 1, 2): `baseline`, `dgpldr`, `cf_aug`, `cf_consistency`, `proto_stability`, `full`.

Findings:

- Lab source macro-F1 ≈ **0.997** — **cannot** select a field-generalising model.
- Nuisance test on lab edits: prediction consistency ~97–100% → consistency loss had almost no gradient.
- **Prototype stability hurt** field scores.
- None of the counterfactual variants beat DG-PLDR on PlantDoc.

### Phase C — Lesion-anchored boundary counterfactuals on DG-PLDR (also failed)

Idea: harder synthetic boundary edits + asymmetric lesion/background feature loss + checkpoint on a **source stress** palette set (not PlantDoc).

Variants: `hard_aug`, `lesion_dg`.

Findings:

- Stress macro-F1 looked high during training but **did not transfer** to PlantDoc.
- Both variants finished **behind** DG-PLDR (and behind baseline on macro-F1 for `lesion_dg`).
- **Do not present** lesion_dg as an improvement.

### Phase D — Frozen DINOv2 (large accuracy jump)

Research match: *“From laboratory to field: Frozen foundation-model features…”* (Smart Agricultural Technology, 2026) — frozen DINOv2 + linear probe trained only on PlantVillage ≈ **51%** on PlantDoc; **fine-tuning the backbone hurts**.

Our machine: `timm` 0.5.4 has **no** dinov2 models → we:

1. Downloaded ViT-L/14 weights → `Resource/models/dinov2_vitl14_pretrain.pth`
2. Wrote a PyTorch 1.10-compatible encoder → `Code/src/dinov2_encoder.py`
3. Feature cache + probe training → `Code/src/train_frozen.py`
4. Added a **crop-gated** head (crop from JSON `crop` field; mask disease logits to that crop) because our ResNet errors were **cross-crop dumping** (e.g. pepper-healthy / tomato late blight absorbing other crops)

Results (PlantDoc, mean of seeds 42 / 1 / 2):

| Method | Accuracy | Macro-F1 |
|--------|----------|----------|
| Frozen linear probe | **53.7%** | 0.490 |
| **Frozen crop-gated linear (best)** | **56.8%** | **0.510** |

This beat DG-PLDR (**19.2%** / 0.151) by a large margin. Crop gate beat the plain linear probe. Lab val macro-F1 remains ~0.99; **field** numbers are what matter.

### Phase E — Hybrid / prototype heads + 518 px (did not beat crop-gated linear)

Combined published head ideas (linear + K visual prototypes, fixed 0.5/0.5 mix) with our crop gate; also re-encoded at DINOv2 native **518** resolution.

None beat **56.8%**. Example means: hybrid_224 56.3%, crop_linear_518 56.4%, crop_hybrid_518 56.1%.

### Phase F — Salient patch pooling + per-crop experts (did not beat crop-gated linear)

Remaining errors are mostly **within-crop** (tomato mean F1 ≈ 0.28 under the best model).

Tried:

- DINOv2-style **CLS + patch mean + top-k salient patches**
- Hard-routed **per-crop expert** heads (router + one linear expert per crop)

Means: crop_experts_salient **54.8%**, crop_linear_salient **54.8%** — **worse** than CLS-only crop-gated linear.

### GitHub load (Nov 2026)

Local project initialised as git and pushed to [bh-ramendra/Agriculture_Proj](https://github.com/bh-ramendra/Agriculture_Proj) with large data/weights/results excluded via `.gitignore`.

---

## 6. Full PlantDoc result table (mean over seeds 42, 1, 2)

Numbers from `Code/results/tables/main_results.csv` (local; not in git).

| Variant | Accuracy | Macro-F1 | Notes |
|---------|----------|----------|--------|
| baseline | 17.0% | 0.136 | ResNet-50 CE |
| dgpldr | 19.2% | 0.151 | Best among fine-tuned ResNets |
| cf_aug | 18.2% | 0.141 | |
| cf_consistency | 18.9% | 0.140 | |
| proto_stability | 16.4% | 0.127 | Worse than baseline |
| full | 17.9% | 0.141 | |
| hard_aug | 18.1% | 0.127 | |
| lesion_dg | 17.2% | 0.120 | Do not present |
| frozen_linear | 53.7% | 0.490 | Frozen DINOv2 CLS |
| **frozen_crop** | **56.8%** | **0.510** | **Current best** |
| hybrid_224 | 56.3% | 0.497 | |
| crop_hybrid_224 | 56.0% | 0.490 | |
| crop_proto_224 | 54.5% | 0.477 | |
| linear_518 | 54.1% | 0.477 | |
| crop_linear_518 | 56.4% | 0.491 | |
| hybrid_518 | 56.4% | 0.489 | |
| crop_hybrid_518 | 56.1% | 0.485 | |
| crop_proto_518 | 56.1% | 0.486 | |
| crop_experts_salient | 54.8% | 0.483 | |
| crop_linear_salient | 54.8% | 0.483 | |

**Best so far:** frozen DINOv2 ViT-L/14 features (224) + crop-gated linear probe → **~56.8%** PlantDoc accuracy.

**Context:** published frozen linear probe ≈ 51%; models reporting 70–94% on PlantDoc typically **train or fine-tune on PlantDoc** (different protocol). ~57% is still **not** deployment-ready.

---

## 7. What worked vs what did not

### Worked

- Strict source-only protocol + fixed 28-class map
- Reimplementing DG-PLDR components missing from the public repo (for a fair ResNet baseline)
- **Freezing** a foundation backbone instead of fine-tuning ResNet / DINOv2
- **Crop gating** on frozen features (addresses cross-crop collapse)
- Running long jobs in **tmux** on a free GPU without killing other users’ jobs

### Did not work (under this protocol)

- Mild or hard **counterfactual** consistency / prototype stability
- Lesion-anchored boundary CF + stress-set selection
- Hybrid / multi-prototype heads on frozen features
- Native **518** encode for the linear/crop heads
- CLS+patch-mean+salient features and **per-crop experts**

### Research constraint we keep

Pieces of later heads (prototypes, experts, multi-backbone fusion) appear in papers that often **train on field data**. Our claim stays: **source-only** PlantVillage → PlantDoc. Present a new head only if it **beats** `frozen_crop` (56.8%) and the relevant ablation on the **same** features.

---

## 8. Environment and how to run

```bash
# Activate env (machine-local path example)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plantdg

cd Agriculture_project/Code

# Unit tests
python -m unittest tests.test_losses

# Frozen DINOv2 (best family): extract once, then train probes
CUDA_VISIBLE_DEVICES=0 python -m src.train_frozen --config configs/frozen.yaml --extract
CUDA_VISIBLE_DEVICES=0 python -m src.train_frozen --config configs/frozen.yaml --train --seed 42

# Collect all finished target metrics into one table
python scripts/collect_results.py
```

**tmux pattern used for full sweeps:**

```bash
tmux new-session -d -s plantdg_frozen -c /path/to/Agriculture_project/Code \
  "bash -lc 'set -o pipefail; CUDA_VISIBLE_DEVICES=0 bash scripts/run_frozen_tmux.sh 2>&1 | tee -a logs/frozen_tmux.log; echo EXIT_STATUS \$?; exec bash'"
```

Scripts of record:

| Script | Purpose |
|--------|---------|
| `scripts/run_end_to_end.sh` | First full ResNet / CF sweep |
| `scripts/run_lesion_tmux.sh` | hard_aug + lesion_dg |
| `scripts/run_frozen_tmux.sh` | Frozen DINOv2 extract + probes |
| `scripts/run_hybrid_tmux.sh` | Hybrid / proto / 518 |
| `scripts/run_experts_tmux.sh` | Salient features + crop experts |

Configs live under `Code/configs/` (`baseline.yaml`, `dgpldr.yaml`, `proposed.yaml`, `lesion_dg.yaml`, `frozen.yaml`, `hybrid224.yaml`, `hybrid518.yaml`, `experts.yaml`, …).

DINOv2 weights (local):

```text
https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
→ Resource/models/dinov2_vitl14_pretrain.pth
```

---

## 9. Datasets and references (followed from papers / flowchart)

Cloned / downloaded (see `Resource/SOURCES.txt`):

- DG-PLDR, PiTLiD, PlantVillage-Dataset, PlantDoc-Dataset, PlantDoc-Object-Detection, AppleLeaf9
- Mendeley plant leaf disease zips (flowchart link; **training source used is spMohanty PlantVillage color**, not the Mendeley CNN pack as the primary source)
- Frozen foundation-model paper: **no public GitHub** — recipe followed from the paper text

Important related results we used for decisions:

- Frozen DINOv2 linear probe ~51% PlantDoc (source-only) — motivates Phase D
- Fine-tuning foundation features collapses field transfer — we keep the backbone **frozen**
- High PlantDoc scores from fusion / ViT+MoE papers often use **PlantDoc training labels** — not our protocol

---

## 10. Current status and next principles

**Status:** Best verified field result is **`frozen_crop` ≈ 56.8% accuracy / 0.510 macro-F1**.  
ResNet/DG-PLDR/CF/lesion lines are finished historical comparisons; do not retrain them unless the protocol changes.

**If improving further, follow:**

1. Keep **source-only** selection (no PlantDoc for train / early stop / HPO).
2. Prefer **frozen** foundation features over full fine-tuning.
3. Target the real failure mode: **within-crop disease confusion** (especially tomato), not crop ID.
4. Report new methods only if they beat **`frozen_crop`** on the same seeds and split.
5. Do not quote in-domain PlantDoc numbers as our transfer accuracy.
6. Long jobs → **tmux**; check GPU free memory on this shared machine first.

---

## 11. Quick command reference

```bash
# Clone this repo (code only)
git clone git@github.com:bh-ramendra/Agriculture_Proj.git
cd Agriculture_Proj

# Local machine already has data under Agriculture_project/Resource/
# Re-download datasets/weights using Resource/SOURCES.txt if needed

cd Code
python scripts/collect_results.py   # needs local results/runs/.../eval/target/
```

Attach to a past tmux session (examples): `plantdg`, `plantdg_lesion`, `plantdg_frozen`, `plantdg_hybrid`, `plantdg_experts`.

```bash
tmux attach -t plantdg_frozen
# detach: Ctrl-b then d
```

---

*Last project update documented in this README: frozen DINOv2 + crop gate as best result; hybrid / 518 / salient-expert ablations completed and did not improve; repository linked to GitHub as Agriculture_Proj.*
