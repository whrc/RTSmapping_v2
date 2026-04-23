# RTS Segmentation Model v2: Training Guide

## 1. Training Objective

Train a semantic segmentation model that detects Retrogressive Thaw Slumps (RTS) under extreme class imbalance (~0.1% positive pixels), optimising for **high precision at acceptable recall** to minimise false positives in the final pan-arctic map.

---

## 2. Environment Setup

### 2.1 Compute Resources

| Resource | Specification |
|----------|---------------|
| Cloud | Google Cloud Platform |
| GPUs | A100 or H100 VM (multi-GPU spec TBD with PDG team) |
| Budget | $70,000 (training + inference combined) |
| Framework | PyTorch 2.x |
| IDE | VSCode + Remote-SSH (GCP VMs only — no Colab) |
| AI-assist | Claude Code |
| Dev/test | L4 VM (`gpu-vm-l4`) — cheaper, same Docker image |


### 2.2 Reproducibility Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| Random seed | 42 | Fixed for all stochastic processes |
| CUDNN deterministic | True | Reproducible convolution results |
| CUDNN benchmark | False | Disable auto-tuning for reproducibility |
| Python hash seed | 42 | Reproducible dictionary ordering |

**Note**: Deterministic mode may reduce training speed by 10-20%. For hyperparameter search, disable deterministic mode; enable for final runs.

---

## 3. Model Architecture

### 3.1 Baseline Model

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Architecture | UNet++ (via smp library) | Close to UNet3+ (v1 best performer); battle-tested library |
| Encoder backbone | EfficientNet-B5 | Strong ImageNet features, good capacity/memory balance for 512×512 |
| Pretrained weights | ImageNet | Transfer learning for faster convergence |
| Input size | 512×512×3 | RGB channels |
| Output | Binary segmentation mask | Single-class prediction |

EfficientNet variants B3/B7 may be tested for capacity trade-offs. B5 is the baseline default.

### 3.2 Candidate Models for Experimentation

Experiment in priority order (stop when diminishing returns):

| Priority | Model | Notes |
|----------|-------|-------|
| 1 (baseline) | UNet++ + EfficientNet-B5 (smp) | Proven architecture class; strong CNN baseline |
| 2 | SegFormer-B5 | Efficient Vision Transformer; strong on dense prediction tasks |
| 3 | DINOv3 encoder + dense head | Latest DINO self-supervised ViT; confirm model version at time of implementation |

SAM is not a direct fit for pixel-level semantic segmentation (prompt-based mask decoder). Skip unless UNet++ and SegFormer both fail to meet precision targets and a dedicated feasibility study is done.
Skip Prithvi, SATMAE, SwinTransformer, Mask2Former unless experiments clearly plateau.

### 3.3 Multi-Modal Fusion (for EXTRA dataset)

Fusion strategies should be tested in order of complexity:

| Order | Strategy | Description | When to Use |
|-------|----------|-------------|-------------|
| 1 | RGB baseline | No auxiliary data | Establish performance baseline |
| 2 | Individual channels | RGB + one auxiliary channel at a time | Identify which channels help |
| 3 | Early fusion | Channel stack (RGB + helpful auxiliaries → single encoder) | Simple, often sufficient |
| 4 | Late fusion | Separate encoders → feature-level fusion | Only if early fusion underperforms |

Note: late fusion may require architecture redesign
---

## 4 Train-Inference Consistency

The training and inference pipelines share a contract along six axes. Divergence on any of these causes silent degradation that pixel-level tests miss. Each subsection below is normative — inference code must conform.

### 4.1 Normalization

Per-dataset statistics computed on the train split only, saved as `normalization_stats.json` alongside the checkpoint. Inference loads identical stats and applies identical mean-subtract / std-divide. If 2025 imagery has materially different radiometric properties than 2024 training data, this manifests as degraded performance; see `scripts/check_inference_normalization.py` for the pre-deployment drift report.

### 4.2 Model output convention

Models output **logits**, not probabilities. Training losses (focal, BCE, Dice, Tversky, compound) operate on logits directly using `F.logsigmoid` internally for numerical stability. Sigmoid is applied only at two points:
1. Metric computation during validation (IoU, F1, object metrics, PR-AUC).
2. Inference probability-raster generation (§9.1 of `inference.md`).

Rationale: naive `log(sigmoid(x))` underflows at extreme logits; `logsigmoid(x)` is stable across the full float range. Focal's `(1-p)^γ` term is derived from `logsigmoid(-x)`.

### 4.3 Checkpoint convention

Two distinct checkpoint files:

| File | Purpose | Contents |
|------|---------|----------|
| `best_deployment.pth` | Inference artifact, shipped in deployment package | `model_state_dict` = **EMA weights**; `normalization_stats_hash`, `channel_names`, `git_sha`, `epoch`, `best_metric`, `trained_with: {precision, seed, config_sha}` |
| `resume_latest.pth` | Training continuation only; never loaded by inference | `live_state_dict`, `ema_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `epoch`, `rng_states` |

Inference never needs to know about EMA — the deployment checkpoint's `model_state_dict` is already the EMA copy. `resume_latest.pth` rotates last-N.

### 4.4 NoData handling

Satellite basemaps contain NoData pixels (ocean, cloud, non-permafrost masked regions). Model behavior on NoData is undefined unless training explicitly saw it.

| Side | Treatment |
|------|-----------|
| Training — partial NoData tile | NoData pixels receive `label = 255` (ignore index, reuses boundary-ignore machinery). Input NoData pixels substituted with the per-channel training mean before normalization. Loss ignores them; gradient contribution is zero. |
| Training — full NoData tile | Skipped at the dataset level; never enters a batch. |
| Inference — partial NoData tile | Predict normally; post-step set `pred_raster[input_nodata_mask] = -1.0` (the NoData value declared in `inference.md §9.1`). |
| Inference — full NoData tile | Skipped, manifest-log per §8.3 of `inference.md`. |

Phase 0's `data/transforms.py` boundary-ignore logic is reused — a NoData mask becomes an additional source of ignore=255 labels merged with the boundary-dilated mask.

### 4.5 Normalization-stats schema

`normalization_stats.json` carries channel-name bindings (not positional):

```json
{
  "version": "2.0",
  "dataset_hash": "...",
  "channels": ["R", "G", "B"],
  "per_channel": {
    "R": {"mean": ..., "std": ...},
    "G": {"mean": ..., "std": ...},
    "B": {"mean": ..., "std": ...}
  },
  "clip_percentiles": [0.1, 99.9]
}
```

At load time, training and inference both assert `stats.channels == cfg.channels.declared_order`. Prevents silent R-stats-applied-to-G-channel bugs if the 2025 basemap API shifts band ordering. `normalization_stats_hash` (MD5) is persisted in the deployment checkpoint metadata for end-to-end verification.

### 4.6 Calibration-deployment parity

Threshold selection (§12.2) and temperature scaling (§12.1) **must** run with identical precision, TTA config, and `torch.compile` setting as the planned deployment. If deployment uses BF16 + minimal TTA and calibration was done in FP32 + no TTA, the calibrated threshold is systematically wrong — numerical differences in sigmoid outputs shift which pixels cross the threshold.

Implementation: a shared `configs/deployment.yaml` holds `{threshold, temperature, tta, precision, torch_compile, scales, fusion}`. Both `scripts/evaluate_test.py` (post-calibration verification) and the Phase 2 inference pipeline load it. Calibration writes the learned `threshold` and `temperature` back into this file; everything else is set before calibration.

---

## 5. Loss Functions

### 5.1 Focal Loss and Tversky Loss

Focal loss down-weights easy examples, focusing learning on hard cases. Particularly suited for class imbalance.

**Formula**: FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

| Parameter | Baseline | Tuning Range | Effect |
|-----------|----------|--------------|--------|
| γ (gamma) | 2 | [1, 2, 3, 5] | Higher = more focus on hard examples |
| α (alpha) | 0.25 | [0.1, 0.25, 0.5, 0.75] | Weight for positive class |

### 5.2 Tversky Loss

Tversky loss allows explicit control over false positive vs false negative penalty.

**Formula**: TL = 1 - (TP + ε) / (TP + α×FN + β×FP + ε)

| Parameter | Range | Notes |
|-----------|-------|-------|
| α | [0.3, 0.5, 0.7] | Weight on false negatives |
| β | [0.3, 0.5, 0.7] | Weight on false positives |

**For precision-focused training**: Set β > α to penalize false positives more heavily.

### 5.3 Compound Loss (Focal + Dice)

Combine pixel-level and region-level objectives. Focal handles pixel-level calibration; Dice directly optimizes region overlap and is insensitive to the overwhelming number of true-negative pixels. This combination consistently outperforms single-component losses in segmentation benchmarks.

**Formula**: `L = λ₁ × Focal(pred, target) + λ₂ × Dice(pred, target)`

| Parameter | Baseline | Tuning Range |
|-----------|----------|--------------|
| λ₁ (focal weight) | 1.0 | [0.5, 1.0, 2.0] |
| λ₂ (dice weight) | 1.0 | [0.5, 1.0, 2.0] |

### 5.4 Class-Balanced Cross-Entropy

Weight loss inversely proportional to class frequency. Options for computing weights:
- Linear: weight_rts = num_bg_pixels / num_rts_pixels
- Square root: weight_rts = sqrt(num_bg_pixels / num_rts_pixels)
- Log: weight_rts = log(num_bg_pixels / num_rts_pixels)

### 5.5 Boundary Uncertainty Handling

Label boundaries may be uncertain due to resolution mismatch or inherent ambiguity in RTS edges.

Both approaches will be implemented and selected via YAML config for ablation:
```yaml
boundary_handling: none   # options: none | ignore | soft_labels
boundary_ignore_width: 3    # pixels (used when boundary_handling: ignore)
soft_label_value: 0.05      # P(background near boundary) when boundary_handling: soft_labels
```

**Approach 1: Ignore Regions** (`boundary_handling: ignore`)
- Exclude pixels within `boundary_ignore_width` pixels of label boundaries from loss computation (set to ignore index 255)
- Applied on-the-fly in the DataLoader using scipy binary dilation on label mask
- Simple, proven in medical imaging segmentation

**Approach 2: Soft Labels** (`boundary_handling: soft_labels`)
- Near-boundary pixels get softened labels: background → `soft_label_value`, RTS → `1 - soft_label_value`
- Options: constant soft values (0.05/0.95) or distance-based softening
- Requires using BCE with soft targets (not cross-entropy with integer labels)

**Experiment order**: Run baseline with `none` first; then ablate against soft_labels with a narrow band (1–2 pixels) and then a small soft value (~0.1).
---

## 6. Metrics

### 6.1 Pixel-Level Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| IoU_RTS | TP / (TP + FP + FN) | Primary pixel-level metric |
| F1_RTS | 2TP / (2TP + FP + FN) | At operating threshold; for literature comparison |

### 6.2 Object-Level Metrics

Object-level evaluation treats each connected component as a detection instance.

| Metric | Description |
|--------|-------------|
| Object Precision | Fraction of predicted objects that match ground truth |
| Object Recall | Fraction of ground truth objects that are detected |
| Object F1 | 2 × Obj_Precision × Obj_Recall / (Obj_Precision + Obj_Recall) — at operating threshold |

**IoU Threshold for Matching**:

| Threshold | Use Case | Recommendation |
|-----------|----------|----------------|
| 0.5 | Standard (COCO default) | Requires good shape match |
| 0.3 | Relaxed | **Preferred** — approximate detections acceptable |
| 0.1 | Very relaxed | "Did we find something here?" |

**Matching Algorithm**: Greedy 1-to-1 matching:
1. Threshold probability map → binary mask; extract connected components (blobs) for both prediction and ground truth
2. Compute pairwise IoU for all (predicted blob, GT blob) pairs
3. Sort predicted blobs by mean probability (highest first)
4. Match each predicted blob to its highest-IoU GT blob, only if IoU ≥ threshold and that GT blob is unmatched
5. Matched pairs → TP; unmatched predictions → FP; unmatched GT blobs → FN

**Edge cases** (expected to be rare given RTS morphology — noted for awareness, not implemented):
- One large prediction overlapping multiple GT objects → matched to the best-IoU GT; remaining GT blobs count as FN
- Multiple predictions overlapping one GT → only the first (highest confidence) matches; the rest count as FP

**Threshold for in-training reporting**: A fixed reference threshold of 0.5 is applied to extract connected components during training. This threshold is for monitoring trends across epochs, not for deployment — focal-loss outputs are not calibrated, so absolute object precision/recall values at 0.5 should be interpreted as relative to other epochs of the same run. The deployment threshold is selected post-training via the calibration procedure in §6.4 and §12.2.

### 6.3 Summary Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| PR-AUC | Area under precision-recall curve | Overall performance under imbalance |

### 6.4 Threshold Calibration Options

Two approaches for selecting operating threshold:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| Global threshold | Single threshold for all regions | Simple, consistent | May underperform in some regions |
| Region-specific thresholds | Calibrate per Arctic subregion | Adapts to regional characteristics | More complex, requires per-region validation data |

**Recommendation**: Start with global threshold. If post-inference analysis reveals systematic regional performance differences, consider region-specific thresholds.

Threshold calibration is run once, post-training, on the EMA-weight final model. In-training object metrics use the fixed 0.5 reference threshold (§6.2) to avoid a circular dependence between calibration and stopping decisions.

---

## 7. Class Imbalance Strategy

### 7.1 The Problem

Real-world RTS prevalence is ~0.1-0.5%. With naive random sampling:
- Most batches contain zero or near-zero positive pixels
- Gradients dominated by easy negatives
- Model may collapse to "predict all background"

### 7.2 Multi-Pronged Approach

| Technique | Description | Effect |
|-----------|-------------|--------|
| Balanced batch sampling | Each batch has ~50% positive tiles, ~50% negative tiles | Ensures model sees positives every batch |
| Focal loss | Down-weights easy examples | Focuses on hard cases |
| Curriculum learning | Gradually increase negative ratio during training | Prevents early collapse |

### 7.3 Curriculum Learning Schedule

**Concrete Schedule** (based on 300 max epochs):

| Epoch Range | Pos:Neg Ratio | Rationale |
|-------------|---------------|-----------|
| 1–10 | 1:1 | Learn basic RTS features with maximum positive exposure |
| 11–30 | 1:5 | Introduce more negatives, start discriminating |
| 31–50 | 1:10 | Standard training ratio |
| 51–100 | 1:15 | Approaching realistic conditions |
| 101–300 | 1:20 | Near-realistic ratio for final refinement |

**Implementation**: Step-wise ratio changes at epoch boundaries (not interpolated). Ratio changes are applied at the epoch level (batch composition recalculated each epoch).

**Early Stopping Note**: With patience=20 on Val-Realistic, training will likely stop before epoch 300. The curriculum ensures the model has seen realistic ratios before convergence.

---

## 8. Multi-Scale Strategy

### 8.1 The Challenge

RTS range from ~50m to 2+ km. At 512×512 tiles with 3m resolution (~1.5km coverage):
- Small RTS (50-200m): Well captured within single tile
- Medium RTS (200m-1km): Well captured within single tile
- Large RTS (1-2+ km): Span multiple tiles, may never appear complete

### 8.2 Solution: Multi-Resolution Inference

Run inference at multiple effective resolutions to catch different RTS scales. See Inference Guide for detailed procedure.

| Scale | Effective Resolution | Field of View | Target RTS Size |
|-------|---------------------|---------------|-----------------|
| 1.0 | 3m (native) | 1.5 km | Small to medium |
| 0.5 | 6m | 3 km | Medium to large |

### 8.3 Multi-Resolution Training

**Current recommendation**: Train at native resolution only.

**Multi-scale inference without retraining — fractal hypothesis**: EfficientNet-B5 + UNet++ skip connections give multi-scale receptive fields, and RTS features have some self-similarity across 3 m ↔ 6 m views. Scale-0.5 inference on a scale-1.0-trained model may work out-of-the-box. This is tested post-calibration in the inference feasibility step (Phase 1 Step 8.5; see `inference.md §6.4`) before any retraining is considered. Gate: ship multi-scale if large-RTS (bbox > 500 m) PR-AUC gain ≥ 2% and global FP-rate delta ≤ +10%.

**Trigger for multi-resolution training**: If the feasibility test fails AND post-inference analysis shows recall for large RTS (>1 km) is the bottleneck, add context-expanded training samples in Phase 1.5. Context-expansion means fetching 1024×1024 physical pixels (2× field of view) and downsampling to 512×512 — not the current `RandomScale` which blurs within a fixed 1.5 km footprint. These are distributionally different operations.

**Known limitation — EPSG:3857 at high latitudes**: Web Mercator pixels are nominally 3 m but the physical area per pixel varies ~1.7× across 60–74°N (3 m at equator → ~0.9 m at 74°N ground). The same pan-arctic strip spans this range. `RandomScale(0.5, 1.0)` partially absorbs the variation, but latitude-stratified performance analysis belongs in Phase 3 post-inference. Accepting this compromise in exchange for web-map compatibility.

---

## 9. Training Configuration

### 9.1 Baseline Hyperparameters

**Model Configuration**:

| Parameter | Value |
|-----------|-------|
| Architecture | UNet++ (smp) |
| Backbone | EfficientNet-B5 |
| Pretrained weights | ImageNet |
| Input channels | 3 (RGB) |
| Input size | 512×512 |

**Loss Configuration**:

| Parameter | Value |
|-----------|-------|
| Loss function | Focal |
| Gamma (γ) | 2 |
| Alpha (α) | 0.25 |
| boundary_handling | none |

**Optimizer Configuration**:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-2 |
| Gradient clipping | Max norm 1.0 |

**Learning Rate Schedule**:

Phase 1 (frozen backbone) uses constant `frozen_lr`. Phase 2 (after unfreezing) uses cosine annealing with warmup.

| Parameter | Value | Applies to |
|-----------|-------|------------|
| Scheduler | Cosine annealing | Phase 2 only |
| Minimum LR | 1e-6 | Phase 2 only |
| Warmup epochs | 5 | Phase 2 only (epochs 11-15) |
| Warmup start LR | 1e-6 | Phase 2 only |

**Backbone Freeze Strategy**:

| Phase | Epochs | Backbone | Decoder | LR |
|-------|--------|----------|---------|-----|
| Phase 1 | 1–freeze_epochs | Frozen | Training | frozen_lr |
| Phase 2 | freeze_epochs+ | Training | Training | base_lr (backbone: base_lr × backbone_lr_multiplier) |

All LR values configurable in YAML:
```yaml
lr:
  frozen_lr: 1e-3           # decoder-only phase (suggested default)
  base_lr: 1e-4             # full fine-tuning base LR
  backbone_lr_multiplier: 0.1  # backbone LR = base_lr × multiplier
freeze_backbone_epochs: 10  # number of epochs for Phase 1
```
After unfreezing, backbone uses `backbone_lr_multiplier × base_lr` to prevent catastrophic forgetting.

**EMA (Exponential Moving Average)**:

| Parameter | Value |
|-----------|-------|
| Enabled | Yes |
| Decay | 0.999 |
| Used for validation | Yes (swap EMA in for validation, swap live weights back for training) |


EMA maintains a smoothed copy of model weights. Final model uses EMA weights.

**Training Configuration**:

| Parameter | Value |
|-----------|-------|
| Mixed precision | FP16 (enabled) |
| Batch size (per GPU) | 32 |
| Effective batch size | 32 × n_gpus |
| Multi-GPU (DDP) | Not implemented initially; code structured to allow DDP addition later |
| Max epochs | 300 |
| Early stopping patience | 20 epochs |
| Early stopping metric | Val-Realistic PR-AUC at 1:200, moving average over last 3 validations |
| Early stopping min delta | 0.005 (placeholder; calibrate empirically — see §10.4) |
| Early stopping start epoch | 50 | After curriculum reaches 1:15 |
| Validation frequency | Every 5 epochs (configurable: `val_frequency`) |

**Data Loading**:

| Parameter | Value |
|-----------|-------|
| Num workers (per GPU) | 8 |
| Pin memory | True |
| Prefetch factor | 2 |
| Persistent workers | True |
| Drop last batch | True |

**Batch Sampling**:

| Parameter | Value |
|-----------|-------|
| Balanced sampling | Enabled |
| Positive fraction per batch | 0.5 |
| Training ratio (epoch-level) | Curriculum (see Section 7.3) |

**Checkpointing**:

| Parameter | Value |
|-----------|-------|
| Save best metric | Val-Realistic PR-AUC |
| Save every N epochs | 10 |
| Keep last N checkpoints | 3 |

**Reproducibility**:

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Deterministic mode | True |
| Seeds for final model | [42, 43, 44] |

### 9.2 Augmentation Pipeline

Applied on-the-fly during training using Albumentations library.

**Geometric Augmentations**:

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| Random 90° rotation | — | 0.5 |
| Horizontal flip | — | 0.5 |
| Vertical flip | — | 0.5 |
| Shift-scale-rotate | shift=0.1, scale=0.2, rotate=45° | 0.5 |
| Elastic transform | alpha=120, sigma=6 | 0.3 |
| Affine transform | shear=(-10°, 10°) | 0.3 |

**Color/Radiometric Augmentations** (RGB channels only):

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| Brightness | ±0.2 | 0.5 |
| Contrast | ±0.2 | 0.5 |
| Saturation | ±0.2 | 0.5 |
| Gaussian noise | var_limit=(10, 50) | 0.3 |
| CLAHE | clip_limit=4.0, tile_grid_size=8×8 | 0.2 |

**Multi-Scale Augmentation** (applied to all channels):

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| RandomScale + PadIfNeeded/CenterCrop | scale=(0.5, 1.0), pad to 512×512 | 0.3 |

RandomScale simulates the effective resolution variation seen during multi-scale inference (0.5x scale). This reduces the train-inference scale gap by exposing the model to downscaled imagery during training.

**Note**: Color/radiometric augmentations apply only to RGB channels, not auxiliary bands in EXTRA dataset. Geometric and multi-scale augmentations apply to all channels and masks.

---

## 10. Training Procedure

### 10.1 Pre-Training Checklist

**Data Preparation**:
- [ ] Data validation checks pass
- [ ] Normalisation statistics computed and saved
- [ ] If `boundary_handling: ignore`, boundary ignore masks created for all labels
- [ ] Balanced batch sampler configured
- [ ] Spatial blocking verified (no geographic overlap between splits)

Create a standalone script check_data.py that iterates through the DataLoader (not just the files), to ensures that the augmentations, normalization, and tensor collating etc are actually working as expected. This is to prevent running expensive GPUs on bad data.

**Environment**:
- [ ] Docker container built and tested
- [ ] GPU memory profiled, batch size confirmed
- [ ] MLflow tracking server running
- [ ] Library versions pinned in requirements.txt

**Configuration**:
- [ ] Config file committed to version control
- [ ] Git commit hash recorded
- [ ] Baseline config validated

**calibration**
- [ ] Validation noise floor measured: Run validation 5× on same checkpoint with augmentation disabled; compute std of PR-AUC at 1:200; set early_stopping_min_delta = 2 × std in baseline config.

### 10.2 Training Loop

**Phase 1: Backbone Frozen (Epochs 1–10)**
1. Freeze all encoder (backbone) parameters
2. Train decoder with higher learning rate (1e-3)
3. Purpose: Adapt decoder to RTS segmentation task without disturbing pretrained features

**Phase 2: Full Fine-Tuning (Epochs 11+)**
1. Unfreeze backbone with lower learning rate (0.1× base LR)
2. Apply curriculum learning schedule for negative ratio
3. Update EMA weights after each optimizer step
4. Validate on Val-Realistic every val_frequency epochs using EMA weights. Swap EMA weights into the model for the validation pass, then restore live weights before the next training step. All validation metrics, early-stopping decisions, and best-checkpoint comparisons use EMA weights. (configurable in YAML; suggested default 5)
5. Check early stopping criterion on Val-Realistic PR-AUC，Early stopping is gated to begin at epoch 50, when the curriculum reaches near-realistic ratios; before this, validation runs and best-so-far checkpoints are saved but stopping is disabled
6. Save checkpoint if best metric achieved

### 10.3 Validation Strategy

| Dataset | Ratios | Purpose | Tune On? |
|---------|--------|---------|----------|
| Val-Balanced | 1:1 | Quick sanity checks | No |
| Val-Realistic | 1:200, 1:500, 1:1000 | Early stopping, threshold calibration | Yes |
| Test-Realistic | 1:200, 1:500, 1:1000 | Final reporting | **Never** |

**Efficient Multi-Ratio Evaluation**: Run inference once on all validation samples, then subsample negatives to compute metrics at each ratio. No additional GPU time required.

### 10.4 Post-Training Steps

1. Confirm EMA weights for final model: validation already used EMA throughout training, so the final saved model is the EMA copy of the best-validation checkpoint. No metric change is expected at this step.
2. **Temperature scaling calibration**: Learn temperature parameter T on Val-Realistic to calibrate prediction confidence
3. **Threshold selection**: Using Val-Realistic, plot PR curves and select threshold where Precision ≥ target
4. **Test-Time Augmentation evaluation**: Evaluate with and without TTA to quantify benefit
5. **Final evaluation**: Report all metrics on Test-Realistic at all ratios (1:200, 1:500, 1:1000)
6. **Multi-seed runs**: Train final configuration with seeds [42, 43, 44], report mean ± std

### 10.5 Overfitting Indicators

Monitor for these warning signs:

| Indicator | Sign | Remedy |
|-----------|------|--------|
| Train-val divergence | Train loss decreasing while val loss increasing | Increase dropout, stronger augmentation |
| Large IoU gap | Train IoU > 0.9, Val IoU < 0.5 | Reduce model capacity, earlier stopping |
| Balanced vs realistic gap | Val-Balanced >> Val-Realistic | Model overfitting to balanced distribution |

---

## 11. Test-Time Augmentation (TTA)

TTA runs inference multiple times with different augmentations and averages predictions. Typically gives 1-3% IoU improvement at the cost of N× inference time.

### 11.1 TTA Configurations

| Setting | Transforms | Speed | Expected Gain |
|---------|------------|-------|---------------|
| Minimal | Identity, horizontal flip | 2× slower | ~1% |
| Standard | Identity, hflip, vflip, rot180 | 4× slower | ~2% |
| Full | All 8 D4 symmetries | 8× slower | ~2-3% |

**Recommendation**: Use Minimal first, use Standard (4 transforms)only if necessary.

### 11.2 TTA Procedure

For each input image:
1. Apply each transform (e.g., horizontal flip)
2. Run model inference
3. Apply inverse transform to prediction
4. Average all predictions pixel-wise
5. Apply threshold to averaged probabilities

---

## 12. Post-Training Calibration

### 12.1 Temperature Scaling

Neural networks are often overconfident. Temperature scaling learns a single parameter T to calibrate probabilities.

**Procedure**:
1. Freeze all model weights
2. Compute logits on Val-Realistic
3. Find T that minimizes negative log-likelihood
4. Apply calibrated probabilities: P_calibrated = sigmoid(logits / T)

Typical T values range from 1.0 to 3.0.

### 12.2 Threshold Selection

For each prevalence ratio (1:200, 1:500, 1:1000):
1. Compute precision-recall curve on Val-Realistic
2. Find threshold achieving target precision (e.g., Precision ≥ 0.8)
3. Record corresponding recall
4. Document threshold and expected performance

**Calibration-deployment parity (per §4.6)**: the PR curve must be computed with the **exact** precision, TTA config, `torch.compile` setting, and scale/fusion choices that Phase 2 inference will use. Calibration loads `configs/deployment.yaml`, writes the learned `threshold` (and §12.1 `temperature`) back into the same file, and that file then travels with the deployment package. Any mismatch between calibration and deployment silently biases all 40M deployment-time decisions.

---


## 13. Statistical Significance

### 14.1 Multiple Seeds

Single-run results are noisy. For final model and key comparisons:
- Run with seeds [42, 43, 44]
- Report mean ± standard deviation
- Example format: IoU_RTS: 0.723 ± 0.012

### 14.2 Reporting Format

Final results table should include:

| Metric | 1:200 | 1:500 | 1:1000 |
|--------|-------|-------|--------|
| IoU_RTS | X.XX ± X.XX | X.XX ± X.XX | X.XX ± X.XX |
| PR-AUC | X.XX ± X.XX | X.XX ± X.XX | X.XX ± X.XX |

---

