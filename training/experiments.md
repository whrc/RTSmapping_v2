
# Experiments

## Overview
This document outlines the tracking configuration, metric definitions, and the logical progression of experiments for the RTS v2 project. Experiments are strictly ordered to answer the most critical "project-breaking" questions (e.g., data scaling and domain shift) before optimizing specific model architectures.

---

## 1. Experiment Tracking

### 1.1 MLflow Configuration

**Tracking URI**: GCS-backed MLflow at `gs://abruptthawmapping/mlflow/`. Configurable via YAML:
```yaml
mlflow:
  tracking_uri: "gs://abruptthawmapping/mlflow/"
  experiment_name: "rts-segmentation-v2"
```
The `MLFLOW_TRACKING_URI` environment variable overrides the YAML value if set (for flexibility in Docker/VM environments).

### 1.2 Required Parameters to Log

| Category | Parameters |
|----------|------------|
| Model | architecture, backbone, pretrained, input_channels (input size = data.tile_size) |
| Loss | loss_function, focal_gamma, focal_alpha, boundary_ignore_width |
| Optimizer | optimizer, learning_rate, weight_decay, gradient_clip_norm |
| Schedule | scheduler, warmup_epochs, base_lr, freeze_backbone_epochs |
| Training | batch_size, max_epochs, early_stopping_patience, ema_decay |
| Data | data_version, train_pos_neg_ratio, curriculum_schedule, subset_pct |
| System | git_commit, pytorch_version, cuda_version, gpu_model, gpu_count |

### 1.3 Metrics & Artifacts

**Metrics to Log Per Epoch**:
- `train_loss`, `train_iou_rts`
- `val_balanced_iou`, `val_balanced_pr_auc`
- For each ratio (200, 500, 1000): `val_{ratio}_pr_auc`, `val_{ratio}_iou_rts`, `val_{ratio}_obj_precision`, `val_{ratio}_obj_recall`

**Artifacts to Save**:
- `best_model.pth` (EMA weights)
- `normalization_stats.json`
- `config.yaml`
- `pr_curves.png`
- `threshold_calibration.json`
- `requirements_frozen.txt`
- `predictions.png` (Fixed validation grid: 3 positive and 3 negative images, 3 cols x 2 rows)

---

## 2. Experiments progression

### Phase 0: Baseline
*Objective: Establish a competent foundation on 100% of the primary training data to act as the reference point for all sanity checks.*
configuration as described in training.md

### Phase 1: Sanity check — temporal domain shift
*Calculate the generalization gap (Delta) between the 2024 Test-Realistic set and the 2025 Test-Realistic set to catch temporal domain shifts.*
Procedure: Run inference using the Phase 0 Baseline on the 2025 micro-set.
The Delta Metric: PR-AUC (2024 Test) - PR-AUC (2025 Test).

**TBD**: The 2025 micro-set (a small labeled validation set from 2025 imagery) has not been defined yet — tile count, region selection, and labeling plan must be decided before this phase can run.

### Phase 2: Data Scaling & Diminishing Returns
*Objective: Empirically determine data and parameter plateau (Neural Scaling Laws).*
Train the Phase Baseline on **25%, 50%, 75%, and 100%** of the available positive RTS tiles and plot data-IoU and data-PRAUC

Deep learning models typically show a power-law relationship between data volume and error reduction. We test this by plotting an empirical learning curve.

**Procedure:**
1. **Control Variables**: Fix the baseline model (`UNet++` + `EfficientNet-B5`), hyperparameters, and keep the validation set strictly constant (`Val-Realistic`).
2. **Data Subsets**: Create four separate training splits using exactly 25%, 50%, 75%, and 100% of the available positive RTS tiles. The full negative pool is kept for all subsets (only positives are subsetted) to isolate the effect of positive data volume — the curriculum sampler draws the appropriate ratio from the available negatives.
3. **Training**: Train the baseline model to convergence on each subset.
4. **Metrics to Track**: Record `PR-AUC` and `IoU_RTS` for each run.

Negative Sample & Imbalance Verification: Because the actual RTS prevalence is extreme (~0.1-0.5%), we do not evaluate negative sample volume by absolute count, but by its effectiveness in suppressing False Positives during the curriculum schedule.

**Procedure & Monitoring:**
1. Train using the standard curriculum learning schedule (progressing from 1:1 up to 1:20 Pos:Neg ratio).
2. At the final 1:20 ratio, thoroughly analyze the False Positives on `Val-Realistic`.

- If the overall False Positive rate drops to acceptable levels, the current negative pool (~20k-25k tiles) provides sufficient background diversity (Degrees of Freedom).

Generalization Gap Monitoring:
To evaluate if the data distribution's degrees of freedom are sufficient to constrain the model's capacity (over-parameterization):
- Track the delta between `Train IoU` and `Val-Realistic IoU`.
- If `Train IoU > 0.9` while `Val-Realistic IoU < 0.5`, the model has too much freedom and the data lacks sufficient variance.
- *Remedy*: Increase the intensity of geometric/color augmentations in Albumentations, increase weight decay

### Phase 3: Test different hyperparameters and training settings (backbone, loss, etc.)
*Objective: Fine-tune the penalty landscape and architectural capacity to suppress False Positives without destroying Recall.*
1. Loss Function: Compare Focal (baseline) vs. Focal+Dice (compound) vs. Tversky (β>α, e.g., α=0.3, β=0.7). Focal+Dice is the priority candidate — if it wins, use it as the foundation for subsequent phases.
2. Boundary Handling: Test `boundary_handling: ignore` with boundary_ignore_width of 1, 2, and 3 pixels. Determine if ignoring boundary ambiguities improves core RTS classification.
3. Backbone Sizing: If Phase 2 indicates a parameter plateau, drop to EfficientNet-B3. If Phase 2 shows continuous scaling, test EfficientNet-B7.
4. Copy-Paste Augmentation: Paste RTS instances from positive tiles onto negative tiles to create synthetic positives. Test whether this improves recall without degrading precision.

### Phase 4: Test different EXTRA channels & Fusion
*Objective: Determine if multi-modal physical context improves the final map.*
**Channel Value**: Compare RGB vs. RGB + NDVI vs. RGB + ArcticDEM derivatives (RE/SR).
**Fusion Strategy** (Only if extra channels shows benefit): Compare Early Fusion (channel stacking) vs. Late Fusion (separate encoders(ask for instructions when implement)).

### Phase 5: Architecture Optimization
*Objective: Test if more advanced (and computationally expensive) feature extractors yield meaningful gains over UNet++.*
**Compare**: UNet++ (Baseline) vs. SegFormer-B5 vs. DINOv3 encoder + dense head.


### optional: try fine-tuning vision FMs
---


## 3. Experiment Execution

A single `scripts/train.py` handles all experiments. Each experiment is defined by its own YAML configuration file in the `configs/` directory.

### Execution
```bash
# Example: Running the diminishing returns test at 50% data
python scripts/train.py --config configs/exp02_scaling_50.yaml
```