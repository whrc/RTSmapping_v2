# RTS Segmentation Model v2: Training Guide

## 1. Training Objective

Train a semantic segmentation model that detects Retrogressive Thaw Slumps (RTS) under extreme class imbalance (~0.1% positive pixels), optimizing for **high precision at acceptable recall** to minimize false positives in the final pan-arctic map.

---

## 2. Model Architecture

### 2.1 Baseline Model

| Component | Choice |
|-----------|--------|
| Architecture | UNet++ |
| Encoder backbone | EfficientNet-B7 |
| Input | RGB, 512×512 |
| Output | Binary segmentation mask |

**Rationale**: Best-performing architecture from Model v1.

### 2.2 Candidate Models for Experimentation

| Category | Models | Notes |
|----------|--------|-------|
| CNN-based | UNet++, DeepLabV3+ | Strong baselines |
| Transformer | SwinTransformer, SegFormer, Mask2Former | May capture long-range context |
| Foundation models | SAM (1/2/3), DINOv2, Prithvi, SATMAE | Require domain adaptation / fine-tuning |

### 2.3 Multi-Modal Fusion (for EXTRA dataset)

| Strategy | Description |
|----------|-------------|
| Early fusion | Channel stack (RGB + aux → single encoder) |
| Late fusion | Separate encoders → feature-level fusion with residual cross-modality attention (Li et al., 2025) |

**Experiment order**: RGB baseline first → early fusion → late fusion.

---

## 3. Loss Functions

### 3.1 Primary: Focal Loss

```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

| Parameter | Baseline | Tuning Range |
|-----------|----------|--------------|
| γ (gamma) | 2 | [1, 2, 3, 5] |
| α (alpha / weight_pos) | 0.25 | [0.1, 0.25, 0.5, 0.75] |

### 3.2 Alternative: Tversky Loss

```python
TL = 1 - (TP + ε) / (TP + α*FN + β*FP + ε)
```

| Parameter | Tuning Range | Notes |
|-----------|--------------|-------|
| α | [0.3, 0.5, 0.7] | Weight on FN |
| β | [0.3, 0.5, 0.7] | Weight on FP |

**For precision-focused training**: Use β > α to penalize false positives more heavily.

### 3.3 Soft Labels

To handle spatial-temporal misalignment between modalities:

| Method | Description |
|--------|-------------|
| Distance-weighted boundaries | Label confidence decays near edges |
| Gaussian blur | Smooth label boundaries |

Both methods are expected to yield similar results; either can be used.

---

## 4. Metrics

### 4.1 Primary Metrics

| Metric | Level | Purpose |
|--------|-------|---------|
| IoU_RTS | Pixel-wise | Primary optimization target (NOT mean IoU) |
| Precision | Object-wise | Map cleanliness |
| Recall | Object-wise | Detection completeness |
| PR-AUC | Summary | Overall performance under imbalance |

### 4.2 Operational Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Precision @ Fixed Recall | P when R ≥ threshold | "Cleaner map at acceptable miss rate" |
| F0.5 Score | (1 + 0.5²) × (P × R) / (0.5² × P + R) | Precision-weighted F-score |

**Goal**: Maximize precision at an acceptable recall threshold. FPs are more costly than FNs because:
- Some missed detections are expected due to image quality
- FPs cause false alarms and erode map credibility

---

## 5. Training Configuration

### 5.1 Baseline Hyperparameters

```yaml
# Model
architecture: unet++
backbone: efficientnet-b7
input_channels: 3  # RGB only for baseline
input_size: [512, 512]

# Loss
loss: focal
focal_gamma: 2
focal_alpha: 0.25  # weight_pos

# Optimizer
optimizer: adamw
learning_rate: 1e-4
weight_decay: 1e-2

# Scheduler
scheduler: cosine_annealing
min_lr: 1e-6

# Training
batch_size: max_that_fits  # A100: likely 8-16
max_epochs: 300
early_stopping_patience: 20
early_stopping_metric: val_balanced_iou_rts

# Data
train_pos_neg_ratio: 1:10  # starting point for f1
val_balanced_ratio: 1:1
val_realistic_ratios: [1:200, 1:500, 1:1000]
test_realistic_ratio: 1:500  # fixed

# Reproducibility
random_seed: 42
```

### 5.2 Augmentation Pipeline (On-the-Fly)

```python
augmentation_pipeline = [
    # Geometric
    RandomRotate90(),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=45,
        p=0.5
    ),
    ElasticTransform(alpha=120, sigma=6, p=0.3),
    
    # Color (for RGB channels only)
    ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        p=0.5
    ),
]
```

**Note**: Color augmentation applies only to RGB channels, not auxiliary bands.

---

## 6. Training Procedure

### 6.1 Training Loop

```
1. Train model
   - Use Val-Balanced (f2a = 1:1) for early stopping
   - Monitor: IoU_RTS, loss

2. Hyperparameter selection
   - Based on Val-Balanced metrics
   - Grid search or Bayesian optimization

3. Train final model
   - Best hyperparameters
   - Full training run

4. Threshold calibration
   - Use Val-Realistic (f2b) at multiple ratios
   - Plot PR curves for each ratio
   - Select threshold where Precision ≥ target
   - Record corresponding Recall
   - Understand sensitivity to prevalence assumptions

5. Final evaluation
   - Report on Test-Realistic (f3 = 1:500)
   - NEVER tune on test set
```

### 6.2 Validation Strategy

| Validation Set | Ratio | Purpose |
|----------------|-------|---------|
| Val-Balanced | 1:1 | Early stopping, hyperparameter tuning |
| Val-Realistic (×3) | 1:200, 1:500, 1:1000 | Threshold calibration across uncertainty range |

---

## 7. Post-Training Calibration

### 7.1 Temperature Scaling

Addresses over/under-confident predictions:

```python
# Calibration procedure
# 1. Freeze all model weights
# 2. Learn temperature T on validation set
# 3. Apply: calibrated_probs = softmax(logits / T)

# Optimization: minimize negative log-likelihood
```

### 7.2 Threshold Selection

```python
# On Val-Realistic, for each prevalence ratio:
for ratio in [1:200, 1:500, 1:1000]:
    pr_curve = compute_pr_curve(model, val_realistic[ratio])
    threshold = find_threshold_for_precision(pr_curve, target_precision=0.8)
    recall_at_threshold = get_recall(pr_curve, threshold)
    log_results(ratio, threshold, recall_at_threshold)
```

---

## 8. Experiment Tracking

### 8.1 MLflow Configuration

```python
mlflow.set_experiment("rts-segmentation-v2")

with mlflow.start_run(run_name="baseline_rgb"):
    # Log parameters
    mlflow.log_params({
        "architecture": "unet++",
        "backbone": "efficientnet-b7",
        "loss": "focal",
        "focal_gamma": 2,
        "focal_alpha": 0.25,
        "learning_rate": 1e-4,
        "batch_size": batch_size,
        "input_channels": 3,
    })
    
    # Log metrics per epoch
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_balanced_iou_rts": val_iou,
        "val_balanced_precision": val_precision,
        "val_balanced_recall": val_recall,
    }, step=epoch)
    
    # Log artifacts
    mlflow.log_artifact("best_model.pth")
    mlflow.log_artifact("pr_curve.png")
```

### 8.2 Experiment Matrix

| Experiment | Variables | Priority |
|------------|-----------|----------|
| Baseline | RGB, UNet++, EfficientNet-B7 | P0 |
| Loss comparison | Focal vs Tversky | P1 |
| Training ratio | 1:10, 1:20, 1:50, dynamic ramp-up | P1 |
| Architecture | CNN vs ViT vs Foundation Models | P2 |
| Auxiliary data | +NDVI, +DEM-derivatives, +all | P2 |
| Fusion method | Early vs late | P2 |

---

## 9. Compute Environment

| Resource | Specification |
|----------|---------------|
| IDE | VSCode + Claude Code |
| VM | Google Colab Pro |
| GPU | A100 or H100 |
| Backup compute | PDG Google funds quota |
| Framework | PyTorch |

### 9.1 Estimated Resource Requirements

| Stage | GPU Hours (estimate) |
|-------|---------------------|
| Baseline training | 10–20 |
| Hyperparameter search | 50–100 |
| Architecture experiments | 50–100 |
| Auxiliary data experiments | 30–50 |
| **Total** | ~150–300 |

---

## 10. Multi-Scale Problem (Open)

**Challenge**: RTS range from tens of meters to several kilometers.
- Small RTS: May be too small in 512×512 @ 3m FOV
- Large RTS: May span multiple tiles

**Potential solutions** (to be decided):
1. Multi-scale inference (image pyramid)
2. Multi-resolution training (mixed tile sizes)
3. Tile stitching for large features
4. Accept detection limitation at extreme scales; focus on clear diagnostic features

**Current approach**: Only label/detect features with clear diagnostic characteristics at PlanetScope resolution.

---

## 11. Fallback: Two-Stage Approach

If precision remains problematic after tuning:

```
Stage 1: RTS Susceptibility Filtering
- Use ML susceptibility model results
- Confine detection to high-likelihood regions
- Eliminate FPs in low-likelihood regions

Stage 2: Segmentation Model
- Run segmentation only on filtered regions
- Reduced false alarm rate
```

This approach is proven effective in fire modeling.

---

## 12. Training Checklist

### Pre-Training
- [ ] Data pipeline validated (correct channels, labels, augmentations)
- [ ] Spatial blocking verified (no train/test geographic overlap)
- [ ] MLflow experiment configured
- [ ] Baseline config committed to version control
- [ ] GPU memory profiled for batch size selection

### During Training
- [ ] Monitor Val-Balanced IoU for early stopping
- [ ] Log learning curves to MLflow
- [ ] Checkpoint best model

### Post-Training
- [ ] Temperature scaling calibration
- [ ] PR curve analysis on Val-Realistic (all ratios)
- [ ] Threshold selection documented
- [ ] Final metrics on Test-Realistic
- [ ] Model artifacts saved to MLflow

---

## 13. Code Organization (Suggested)

```
rts-segmentation-v2/
├── configs/
│   ├── baseline.yaml
│   └── experiments/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── augmentations.py
│   │   └── dataloader.py
│   ├── models/
│   │   ├── unetpp.py
│   │   └── backbones/
│   ├── losses/
│   │   ├── focal.py
│   │   └── tversky.py
│   ├── metrics/
│   │   ├── iou.py
│   │   └── precision_recall.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── calibration/
│       └── temperature_scaling.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── calibrate.py
├── notebooks/
│   └── analysis/
└── mlruns/  # MLflow tracking
```