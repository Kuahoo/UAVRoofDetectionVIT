# Roof Damage Inspection Prototype (RoofD)

Academic machine-learning prototype for aerial roof-defect inspection. A Vision
Transformer answers **what** defects are present; a U-Net answers **where** they
are. A logistic-regression baseline provides a course reference point.

> **Disclaimer:** For research demonstration only; professional inspection is required.
> This system is not a certified structural assessment and must not be used for
> safety-critical decisions.

## Components

| Component | Task | Input | Loss | Output | Primary metric |
|---|---|---|---|---|---|
| ViT | Multi-label image classification | 224×224 ImageNet-normalized | BCE with dampened `pos_weight` | 5 sigmoid probabilities (defects only) | Macro F1 (5 defects) |
| U-Net | Semantic pixel segmentation | 256×256 ImageNet-normalized | Focal + Dice | 6-class pixel map (incl. background) | mIoU |
| Baseline | Multi-label image classification | 64×64 grayscale HOG + RGB stats | Log loss (one-vs-rest) | 5 probabilities (defects only) | Macro F1 (5 defects) |

- ViT is the primary implementation focus.
- U-Net is the partner-owned segmentation pipeline; it is included so the demo
  and shared evaluation run end to end.
- The baseline uses the same image-level targets as the ViT.

## Repository layout

```text
configs/default.yaml          shared configuration
scripts/download_roofd.py     RoofD archive download/extract
scripts/prepare_data.py       Milestone 1 verification, stats, sample figure
scripts/make_smoke_data.py    tiny synthetic RoofD-format data for smoke tests
src/data.py                   shared pairs, mask decoding, targets, augmentations
src/features.py               HOG + color statistics
src/train_logistic.py         course baseline
src/train_vit.py              staged ViT fine-tuning + threshold tuning
src/train_unet.py             Focal + Dice U-Net training
src/evaluate.py               test metrics, ensemble, figures, degradation experiment
src/inference.py              checkpoint loading, prediction, fusion, overlays, conflicts
src/metrics.py                Macro F1, mIoU, Dice, figures
src/app.py                    Streamlit human-in-the-loop demo
outputs/                      checkpoints, metrics, figures
data/raw/                     dataset location
```

## Setup

Python 3.10+ is recommended (TIDE Jupyter images ship Python 3.11).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training on TIDE

Full training runs on **SDSU TIDE** (see `docs/context/TIDE.md`). The default
allocation is **8 CPU cores, 32 GB system RAM, and one NVIDIA L40 (48 GB)**.
Prefer the L40 over the scarce A100 node—this workload does not need 80 GB VRAM.

Recommended workflow:

1. Launch JupyterHub with the **pytorch-notebook** image (CUDA PyTorch /
   torchvision already present), or submit a longer run as a **batch container**
   job on the cluster.
2. Clone/sync this repo onto TIDE storage, then
   `pip install -r requirements.txt` so `timm`, SMP, OpenCV, etc. land on top of
   the image’s torch stack (version ranges leave the CUDA build in place).
3. Download RoofD once onto shared storage (`python -m scripts.download_roofd`);
   reuse that tree across sessions.
4. Train from the repo root with `device: auto` and AMP on CUDA
   (`configs/default.yaml`).

`num_workers: 8` matches the default 8-core allocation. Keep ViT batch 16 and
U-Net batch 8 as the reproducibility baseline; the L40 has ample headroom if you
later raise batches for throughput experiments.

## Dataset

### Download

```bash
python -m scripts.download_roofd
```

The script resolves a fresh Yandex Disk link, downloads the roughly 6.2 GB
archive, extracts it into `data/raw/RoofD`, and removes the archive unless
`--keep-archive` is passed.

The real archive layout is:

```text
data/raw/RoofD/
└── dataset/
    └── augmented_dataset/
        ├── train/
        │   ├── source/*.png     # 5,184 aerial RGB images
        │   └── mask/*.png       # 5,184 RGB masks
        └── test/
            ├── source/*.png     # 1,216 aerial RGB images
            └── mask/*.png       # 1,216 RGB masks
```

The loader also accepts `images/` instead of `source/`, and masks colocated with
images using `_mask` suffixes. It auto-discovers the single `train` and `test`
directories below the configured root.

### Class map

| ID | Class | RGB |
|---:|---|---|
| 0 | background | `(0, 0, 0)` |
| 1 | hollows | `(255, 255, 0)` |
| 2 | swellings | `(153, 255, 204)` |
| 3 | folds | `(6, 89, 182)` |
| 4 | patches | `(0, 255, 255)` |
| 5 | breaks | `(255, 193, 255)` |

The classifier target is a length-5 binary presence vector over the five defect
classes only (IDs 1–5). Background (ID 0) is a segmentation-only class: it is
almost always present, so including it in multi-label classification would
inflate Macro F1 and waste capacity on a near-constant label. Both the ViT and
the logistic baseline use this defect-only target derived from the same decoded
mask as the U-Net.

### Verify and cache statistics

```bash
python -m scripts.prepare_data --verify-images
```

This writes:

- `outputs/splits.json` — deterministic group-aware train/val/test split (`seed: 42`)
- `outputs/metrics/train_stats.json` — class image/pixel counts plus cached labels
- `outputs/metrics/test_stats.json` — test counts
- `outputs/figures/dataset_samples.png` — image/mask/overlay examples

The training split holds out 15% of the training set for validation with
`GroupShuffleSplit`. `data.group_by: tile` keeps all four of
`0000_<flight>_<tile>.png` … `0003_<flight>_<tile>.png` on one side of the
split; `group_by: flight` holds out entire roofs instead. Splitting by raw
filename would leak augmented copies of the same tile into validation and
inflate validation F1.

The training scripts reuse the cached statistics, so the 6,400 masks are decoded
once rather than once per model.

## Training

Run from the repository root:

```bash
python -m src.train_logistic
python -m src.train_vit
python -m src.train_unet
```

What each script does:

- **Logistic** — resizes to 64×64, extracts HOG (grayscale) + RGB mean/variance, fits
  `OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0))`
  on the five defect labels.
- **ViT** — `timm` `vit_base_patch16_224` (`drop_path_rate=0.2`), ImageNet
  mean/std, `num_classes=5`. Warm-up: frozen backbone, 5-node head only,
  AdamW (`lr=1e-3`, `weight_decay=0.01`), 5 epochs. Fine-tune: the trailing six
  transformer blocks plus head with layer-wise LRs (blocks 6–8 `1e-5`,
  blocks 9–11 `2e-5`, head `1e-4`), cosine annealing, grad clip `1.0`,
  15 epochs. Loss is `BCEWithLogitsLoss(pos_weight=sqrt(negatives/positives))`
  capped at `10` over the five defect classes (training and validation counts,
  never test).
  After training, per-class thresholds are swept over `0.10–0.90` on validation
  (F1-maximizing, `0.5` breaks ties) and stored in the checkpoint.
- **U-Net** — ResNet34 ImageNet encoder, 256×256 input, Focal loss
  (`gamma=2`, log-smoothed class weights) + Dice loss weighted equally, Adam at
  `1e-4`, 40 epochs, `CosineAnnealingLR` (`eta_min=1e-6`).

Artifacts:

```text
outputs/checkpoints/vit_best.pt
outputs/checkpoints/unet_best.pt
outputs/checkpoints/logistic.joblib
outputs/metrics/*_history.json
outputs/figures/*
```

The ViT and U-Net save the best validation checkpoint and report test metrics
after reloading that checkpoint. `vit_best.pt` also carries the validation-tuned
`thresholds`; `evaluate.py` and the Streamlit demo consume them.

### Smoke test without RoofD

```bash
python -m scripts.make_smoke_data
python -m src.train_vit      --smoke --limit 32 --data-root /tmp/roofd_smoke
python -m src.train_unet     --smoke --limit 32 --data-root /tmp/roofd_smoke
python -m src.train_logistic --smoke --limit 32 --data-root /tmp/roofd_smoke
python -m src.evaluate       --smoke --limit 32 --data-root /tmp/roofd_smoke --model all
```

`--smoke` uses one epoch, batch size 2, no pretrained downloads, and writes to
`outputs/smoke`. It verifies tensor shapes and the end-to-end path; it is not a
performance result.

## Evaluation

```bash
python -m src.evaluate --model all
python -m src.evaluate --model all --degradation
```

- `--model {vit,unet,logistic,ensemble,all}`
- `--split {val,test}` (default `test`)
- `--checkpoint PATH` for a single model checkpoint

`--model ensemble` pools each U-Net softmax mask into five image-level presence
probabilities (mean of the top 100/100/40/30/20 pixels) and fuses them with the
ViT probabilities using per-class weights `alpha=[0.7,0.7,0.7,0.3,0.2]`
(`configs/default.yaml`). `--model all` evaluates ViT, U-Net, logistic, and the
ensemble when both deep checkpoints exist.

Standard evaluation writes per-class tables, JSON metrics, probability arrays,
and figures. The degradation experiment evaluates the native 1024×576 test image
after bilinear down/up sampling to 512×288 and 256×144, then resizing to the model
input. Results are written to `outputs/metrics/degradation_<model>_test.json` and
`outputs/figures/degradation_<model>_test.png`.

## Streamlit demo

```bash
streamlit run src/app.py
```

The sidebar accepts a config path. The app:

1. resizes to 224×224 for the ViT and 256×256 for the U-Net;
2. shows ViT, U-Net presence, and fused probabilities per defect class, with
   the validation-tuned threshold for each class;
3. renders the U-Net mask over the original image;
4. reports deterministic disagreements:
   - **Unlocalized Confidence Warning** — ViT is confident, U-Net found no pixels;
   - **Sub-threshold Spatial Detection** — U-Net found pixels, ViT is below its
     class threshold.
5. never fabricates pixels: the overlay always comes from the U-Net mask.

## Reproducibility and hardware

- One seed (`42`) controls the split, DataLoader workers, and training.
- The validation split is a grouped 15% of the provided training set: all
  augmented copies of a tile stay on one side. The test set stays untouched
  until evaluation.
- `pos_weight`, segmentation class weights, and per-class thresholds are
  computed from training/validation data only and cached or stored in the
  checkpoints.
- **Primary train host:** TIDE L40 (48 GB). Configured batches are ViT 16 /
  U-Net 8 (also safe on ≥8 GB cards for local smoke debugging).
- Smoke tests may run on CPU; full ViT/U-Net training should use a TIDE GPU
  session or batch job, not the A100 advanced node unless L40s are unavailable.

## Limitations

- The 1024×576 images are downscaled to 224/256, which can erase small breaks
  and patches.
- Macro F1 averages the five defect classes equally; per-class rare-defect
  recall (patches, breaks) remains the more informative diagnostic.
- Per-class thresholds are tuned on the same small validation split used for
  checkpoint selection, so they can overfit; test metrics remain the honest
  estimate.
- The dataset is a single public source; results are not evidence of general
  field performance.

## Design note

The detailed ViT-focused plan lives in `docs/plans/roof_damage_VIT.md`.
Classifier targets and Macro F1 cover the five defect classes only; the U-Net
retains all six pixel classes including background.
