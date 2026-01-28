# RTS Segmentation Model v2: Training Guide

## 1. Training Objective

Train a semantic segmentation model that detects Retrogressive Thaw Slumps (RTS) under extreme class imbalance (~0.1% positive pixels), optimizing for **high precision at acceptable recall** to minimize false positives in the final pan-arctic map.

---

## 2. Model Architecture

### 2.1 Baseline Model

| Component | Choice |
|-----------|--------|
| Architecture | UNet3+ |
| Encoder backbone | EfficientNet-B7 (ImageNet pretrained) |
| Input | RGB, 512×512 |
| Output | Binary segmentation mask |

### 2.2 Candidate Models for Experimentation

| Category | Models | Notes |
|----------|--------|-------|
| non-FM ViTs | SwinTransformer, SegFormer, Mask2Former |
| VFMs | SAM (1/2/3), DINOv2, Prithvi, SATMAE | Require domain adaptation fine-tuning |

### 2.3 Multi-Modal Fusion (for EXTRA dataset)

| Strategy | Description |
|----------|-------------|
| Early fusion | Channel stack (RGB + EXTRA → single encoder) |
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


### 3.3 Alternative: Class-Balanced Cross-Entropy

```python
# Weight inversely proportional to class frequency
weight_bg = 1.0
weight_rts = num_bg_pixels / num_rts_pixels  # or sqrt, or log
CE_balanced = -w_c * y * log(p)
```

### 3.4 Boundary Uncertainty Handling

Label boundaries may be uncertain due to resolution mismatch between RGB and EXTRA data, or inherent ambiguity in RTS edges.

**Option A: Ignore Regions**

Exclude pixels within N pixels of label boundaries from loss computation: Simple to implement, proven effective in medical imaging segmentation.

**Option B: Soft Labels**

For spatial-temporal misalignment between modalities: use gaussian filter or distance-based soft label


## 4. Metrics

### 4.1 Pixel-Level Metrics

| Metric | Formula | 
|--------|---------|
| IoU_RTS | TP / (TP + FP + FN) |

### 4.2 Object-Level Metrics (COCO-Style)

Object-level evaluation treats each connected component as a detection. This measures whether RTS *instances* are found, not just pixels.

| Metric | Formula | 
|--------|---------|
| Instance-level Precision | TP / (TP + FP) **-Primary optimization target**| 
| Instance-level Recall | TP / (TP + FN) | 
**IoU Threshold Selection**:

| Threshold | Use Case |
|-----------|----------|
| 0.5 | Standard (COCO default) — requires good shape match |
| 0.3 | Relaxed — approximate detections **-PREFERRED**|
| 0.1 | Very relaxed — "did we find something here?" |

**Edge Cases**:

| Scenario | Handling |
|----------|----------|
| One prediction covers multiple GT | Count as 1 TP + (N-1) FN |
| Multiple predictions cover one GT | Count as 1 TP + (M-1) FP |
| Fragmented prediction | TBD |

### 4.3 Summary Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| PR-AUC | Area under precision-recall curve | Overall performance under imbalance |
| AP@0.5 | Average precision at IoU=0.5 | Object detection quality |
| F0.5 | (1 + 0.5²) × (P × R) / (0.5² × P + R) | Precision-weighted F-score |

### 4.4 Operational Metric

**Precision @ Fixed Recall**: "What precision can we achieve if we accept missing X% of RTS?"

---

## 5. Class Imbalance Strategy

### 5.1 The Problem

Real-world RTS prevalence is ~0.1-0.5% (1:200 to 1:1000). With random sampling:
- Most batches have zero or near-zero positive pixels
- Gradients are noisy and dominated by easy negatives
- Model may collapse to "predict all background"

Additionally, even within positive tiles, RTS pixels are only 5-30% of the tile (see Data Specification).

### 5.2 Approaches for Extreme Imbalance

| Approach | Description | 
|----------|-------------|
| **Balanced batch sampling** | Each batch has ~50% pos tiles, ~50% neg tiles 
| **Class-weighted loss** | Higher loss weight for positive class 
| **Focal loss** | Down-weights easy examples 
| **Curriculum learning** | Start balanced ratio, increase imbalance over time 

---

## 6. Multi-Scale Strategy

### 6.1 The Challenge

RTS typically range from ~50m to 2+ km. At 512×512 tiles with 3m resolution:
- Tile coverage: ~1.5 km × 1.5 km
- Too small RTS (<50m): ignore
- Small RTS (50-200m): Well captured
- Medium RTS (200m-1km): Well captured  
- Large RTS (1-2+ km): Span multiple tiles, may never appear complete

### 6.2 Approach: Multi-Resolution Inference (Detailed in inference.md)

Run inference at multiple effective resolutions to catch different RTS scales:

**Scale interpretation**:
- `scale=1.0`: Native 3m resolution, good for small-medium RTS
- `scale=0.5`: Effective 6m resolution, 3km FOV, catches larger RTS
- `scale=0.25`: Effective 12m resolution, 6km FOV, catches very large RTS

### 6.3 Do We Need Multi-Resolution Training?
If large RTS recall is poor, consider multi-resolution training

## 7. Training Configuration

### 7.1 Compute Environment

| Resource | Specification |
|----------|---------------|
| Cloud | Google Cloud Platform |
| GPUs | 8× NVIDIA H100 |
| Budget | $70,000 (training + inference) |
| Framework | PyTorch |
| IDE | VSCode + Claude Code |

### 7.2 Baseline Hyperparameters

```yaml
# Model
architecture: unet++
backbone: efficientnet-b7
pretrained: imagenet
input_channels: 3  # RGB only for baseline
input_size: [512, 512]

# Loss
loss: focal
focal_gamma: 2
focal_alpha: 0.25  # weight_pos
boundary_ignore_width: 3  # pixels to ignore at label boundaries

# Optimizer
optimizer: adamw
learning_rate: 1e-4
weight_decay: 1e-2
gradient_clip_norm: 1.0  # prevent exploding gradients

# Learning Rate Schedule
scheduler: cosine_annealing
min_lr: 1e-6
warmup_epochs: 5  # linear warmup from 0 to learning_rate
warmup_start_lr: 1e-6

# Backbone Freeze
freeze_backbone_epochs: 10  # freeze encoder for first N epochs
unfreeze_lr_multiplier: 0.1  # lower LR for pretrained layers after unfreeze

# EMA
use_ema: true
ema_decay: 0.999

# Mixed Precision
mixed_precision: true  # fp16 training on H100

# Training
batch_size: 32  # per GPU; effective batch = 32 × 8 = 256 with DDP
max_epochs: 300
early_stopping_patience: 20
early_stopping_metric: val_realistic_pr_auc  # NOT val_balanced
early_stopping_min_delta: 0.001

# Checkpointing
save_best_metric: val_realistic_pr_auc
save_every_n_epochs: 10
keep_last_n_checkpoints: 3

# Batch Sampling
balanced_batch_sampling: true
pos_fraction_per_batch: 0.5

# Data Ratios
train_pos_neg_ratio: 1:10  # epoch-level (batch-level is 1:1)
val_balanced_ratio: 1:1
val_realistic_ratios: [1:200, 1:500, 1:1000]
test_realistic_ratios: [1:200, 1:500, 1:1000]  # report all

# DataLoader
num_workers: 8  # per GPU
pin_memory: true
prefetch_factor: 2
persistent_workers: true

# Validation Frequency
# TBD: Set based on final dataset size
# Options:
#   val_frequency_epochs: 1  (every epoch)
#   val_frequency_steps: 500  (every N steps)
val_frequency: TBD

# Reproducibility
random_seed: 42
deterministic: true
n_seeds_for_final: 3  # run final model with seeds [42, 43, 44]
```

### 7.3 Mixed Precision Training

~2× speedup on H100 with minimal accuracy loss:

### 7.4 Backbone Freeze/Unfreeze Strategy

Prevent catastrophic forgetting of pretrained features:

```python
def freeze_backbone(model):
    """Freeze encoder (backbone) weights."""
    for name, param in model.named_parameters():
        if 'encoder' in name or 'backbone' in name:
            param.requires_grad = False

def unfreeze_backbone(model, lr_multiplier=0.1):
    """
    Unfreeze encoder with lower learning rate.
    Returns parameter groups for optimizer.
    """
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        param.requires_grad = True
        if 'encoder' in name or 'backbone' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    return [
        {'params': encoder_params, 'lr': base_lr * lr_multiplier},
        {'params': decoder_params, 'lr': base_lr}
    ]

# Training loop
model = build_model()
freeze_backbone(model)

# Phase 1: Train decoder only
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
for epoch in range(freeze_backbone_epochs):
    train_epoch(...)

# Phase 2: Fine-tune all layers
param_groups = unfreeze_backbone(model, lr_multiplier=0.1)
optimizer = AdamW(param_groups, lr=1e-4)
for epoch in range(freeze_backbone_epochs, max_epochs):
    train_epoch(...)
```

### 7.5 Learning Rate Warmup

Gradually increase learning rate from near-zero to target over first N epochs:

```python
def get_lr_with_warmup(epoch, warmup_epochs, base_lr, warmup_start_lr=1e-6):
    """
    Linear warmup followed by cosine annealing.
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)
    else:
        # Cosine annealing (handled by scheduler)
        return None  # Let scheduler take over

# In training loop:
for epoch in range(max_epochs):
    if epoch < warmup_epochs:
        lr = get_lr_with_warmup(epoch, warmup_epochs, base_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()
```

### 7.6 Gradient Clipping

Prevent exploding gradients by capping gradient norm:

```python
# After loss.backward(), before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 7.7 Exponential Moving Average (EMA)

Maintain a smoothed copy of model weights for final evaluation:

```python
class EMA:
    """
    Exponential Moving Average of model weights.
    
    EMA weights are updated each step:
        ema_weights = decay * ema_weights + (1 - decay) * model_weights
    
    Benefits:
    - Reduces noise in final model
    - Often 0.5-1% better than last checkpoint
    - Free improvement at no training cost
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Call after each optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Replace model weights with EMA weights for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

# Usage:
ema = EMA(model, decay=0.999)

for epoch in range(max_epochs):
    for batch in train_loader:
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        ema.update()  # Update EMA after each step
    
    # Evaluate with EMA weights
    ema.apply_shadow()
    val_metrics = evaluate(model, val_loader)
    ema.restore()

# Final model uses EMA weights
ema.apply_shadow()
save_model(model)
```

### 7.8 Checkpointing Strategy

```python
class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, save_dir, save_best_metric='val_realistic_pr_auc', 
                 mode='max', keep_last_n=3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_metric = save_best_metric
        self.mode = mode
        self.keep_last_n = keep_last_n
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.checkpoints = []
    
    def save(self, model, optimizer, epoch, metrics, ema=None):
        """Save checkpoint and manage old ones."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        if ema is not None:
            checkpoint['ema_shadow'] = ema.shadow
        
        # Save latest
        path = self.save_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        torch.save(checkpoint, path)
        self.checkpoints.append(path)
        
        # Save best
        current_value = metrics[self.save_best_metric]
        is_best = (self.mode == 'max' and current_value > self.best_value) or \
                  (self.mode == 'min' and current_value < self.best_value)
        
        if is_best:
            self.best_value = current_value
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model: {self.save_best_metric}={current_value:.4f}")
        
        # Clean old checkpoints
        while len(self.checkpoints) > self.keep_last_n:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
```

### 7.9 DataLoader Optimization

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=BalancedBatchSampler(...),
    num_workers=8,            # CPU cores for data loading
    pin_memory=True,          # Faster GPU transfer
    prefetch_factor=2,        # Batches to prefetch per worker
    persistent_workers=True,  # Don't respawn workers each epoch
    drop_last=True,           # Consistent batch sizes
)
```

### 7.10 Augmentation Pipeline (On-the-Fly)

```python
import albumentations as A

augmentation_pipeline = A.Compose([
    # Geometric
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=45,
        border_mode=cv2.BORDER_REFLECT,
        p=0.5
    ),
    A.ElasticTransform(alpha=120, sigma=6, p=0.3),
    
    # Color (RGB channels only)
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        p=0.5
    ),
])
```
 - add affine transformation
**Note**: Color augmentation applies only to RGB channels, not auxiliary bands.

---

## 8. Training Procedure

### 8.1 Training Loop

```
1. Data Preparation
   - Create ignore masks for boundary pixels
   - Set up balanced batch sampler
   - Verify spatial blocking (no train/test overlap)
   - Compute and save normalization statistics

2. Train Model
   - Phase 1: Freeze backbone, train decoder (10 epochs)
   - Phase 2: Unfreeze all, fine-tune (remaining epochs)
   - Use balanced batch sampling (50% pos, 50% neg per batch)
   - Mixed precision (fp16) training
   - LR warmup for first 5 epochs
   - Gradient clipping (norm=1.0)
   - Update EMA after each step
   - Early stop on Val-Realistic PR-AUC

3. Hyperparameter Selection
   - Based on Val-Realistic metrics
   - Grid search or Bayesian optimization

4. Train Final Model (Multiple Seeds)
   - Best hyperparameters
   - Run with seeds [42, 43, 44]
   - Report mean ± std
   - Save EMA weights as final model

5. Threshold Calibration
   - Use Val-Realistic at multiple ratios (1:200, 1:500, 1:1000)
   - Plot PR curves for each ratio
   - Select threshold where Precision ≥ target
   - Understand sensitivity to prevalence assumptions

6. Final Evaluation
   - Report on Test-Realistic at ALL ratios (1:200, 1:500, 1:1000)
   - Report both pixel-level and object-level metrics
   - NEVER tune on test set
```

### 8.2 Validation and Test Strategy

| Dataset | Ratios | Purpose | Tune On? |
|---------|--------|---------|----------|
| Val-Balanced | 1:1 | Quick sanity checks | No |
| Val-Realistic | 1:200, 1:1000 | Early stopping, threshold calibration | Yes |
| Test-Realistic | 1:200,  1:1000 | Final reporting | **Never** |

**Computing metrics at multiple ratios is cheap**: Run inference once, then subsample negatives differently for each ratio. No additional GPU time.

```python
def evaluate_at_multiple_ratios(model, pos_samples, neg_samples, ratios=[200, 500, 1000]):
    """
    Evaluate model at multiple prevalence ratios.
    Inference runs once; only metric computation varies.
    """
    # Run inference on all samples (once)
    pos_preds = [model.predict(x) for x in pos_samples]
    neg_preds = [model.predict(x) for x in neg_samples]
    
    results = {}
    for ratio in ratios:
        # Subsample negatives to achieve desired ratio
        n_neg = len(pos_samples) * ratio
        neg_subset = random.sample(neg_preds, min(n_neg, len(neg_preds)))
        
        # Compute metrics
        all_preds = pos_preds + neg_subset
        all_labels = [1]*len(pos_preds) + [0]*len(neg_subset)
        
        results[f'1:{ratio}'] = compute_metrics(all_preds, all_labels)
    
    return results
```

### 8.3 Early Stopping Configuration

```python
class EarlyStopping:
    """Early stopping with refinements."""
    
    def __init__(self, patience=20, min_delta=0.001, mode='max', 
                 restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_weights = None
        self.counter = 0
    
    def __call__(self, value, model):
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True  # Stop
            return False

# Usage
early_stopping = EarlyStopping(
    patience=20,
    min_delta=0.001,
    mode='max',
    restore_best_weights=True
)

for epoch in range(max_epochs):
    train_epoch(...)
    val_metrics = validate(...)
    
    if early_stopping(val_metrics['val_realistic_pr_auc'], model):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 8.4 Overfitting Monitoring

Watch for these warning signs:

```python
def check_overfitting(metrics_history):
    """Detect overfitting indicators."""
    warnings = []
    
    # Training loss decreasing while val loss increasing
    if len(metrics_history) >= 10:
        recent_train = metrics_history[-10:]
        if (recent_train[-1]['train_loss'] < recent_train[0]['train_loss'] and
            recent_train[-1]['val_loss'] > recent_train[0]['val_loss']):
            warnings.append("Train loss decreasing while val loss increasing")
    
    # Large gap between train and val IoU
    if metrics_history[-1]['train_iou'] > 0.9 and metrics_history[-1]['val_iou'] < 0.5:
        warnings.append(f"Large train-val IoU gap: {metrics_history[-1]['train_iou']:.2f} vs {metrics_history[-1]['val_iou']:.2f}")
    
    # Val-balanced much better than val-realistic
    if metrics_history[-1].get('val_balanced_iou', 0) > metrics_history[-1].get('val_realistic_iou', 0) + 0.2:
        warnings.append("Val-balanced >> Val-realistic suggests overfitting to balanced distribution")
    
    return warnings

# Remedies if overfitting detected:
# - Increase dropout
# - Stronger augmentation
# - Reduce model capacity
# - Earlier stopping
```

### 8.5 Progressive Resizing (Optional)

Train on smaller tiles first for faster iteration:

```python
# Phase 1: Fast exploration at 256×256
config_phase1 = {
    'input_size': 256,
    'epochs': 50,
    'purpose': 'Find good hyperparameters quickly'
}

# Phase 2: Full resolution at 512×512
config_phase2 = {
    'input_size': 512,
    'epochs': 50,
    'purpose': 'Final training at target resolution',
    'init_from': 'phase1_best_model.pth'
}
```

Benefits:
- Faster hyperparameter search
- May improve generalization
- Lower memory usage in Phase 1

---

## 9. Test-Time Augmentation (TTA)

TTA runs inference multiple times with different augmentations and averages predictions. Typically gives 1-3% IoU improvement for free (at cost of N× inference time).

### 9.1 Implementation

```python
def predict_with_tta(model, image, tta_transforms=None):
    """
    Test-time augmentation: average predictions over augmented views.
    
    Args:
        model: Trained model
        image: Input image (H, W, C)
        tta_transforms: List of (transform, inverse_transform) pairs
    
    Returns:
        Averaged probability map
    """
    if tta_transforms is None:
        tta_transforms = [
            (identity, identity),
            (hflip, hflip),
            (vflip, vflip),
            (rot90, rot270),
            (rot180, rot180),
            (rot270, rot90),
        ]
    
    predictions = []
    
    for transform, inverse in tta_transforms:
        # Apply transform
        aug_image = transform(image)
        
        # Predict
        pred = model.predict(aug_image)
        
        # Inverse transform prediction
        pred_original = inverse(pred)
        
        predictions.append(pred_original)
    
    # Average all predictions
    return np.mean(predictions, axis=0)

# Transform functions
def identity(x): return x
def hflip(x): return np.fliplr(x)
def vflip(x): return np.flipud(x)
def rot90(x): return np.rot90(x, k=1)
def rot180(x): return np.rot90(x, k=2)
def rot270(x): return np.rot90(x, k=3)
```

### 9.2 TTA Configuration

| Setting | Transforms | Speedup vs Accuracy |
|---------|------------|---------------------|
| Minimal | identity, hflip | 2× slower, ~1% gain |
| Standard | identity, hflip, vflip, rot180 | 4× slower, ~2% gain |
| Full | All 8 D4 symmetries | 8× slower, ~2-3% gain |

**Recommendation**: Use Standard (4 transforms) for production — good balance of accuracy and speed.

### 9.3 Combining TTA with Multi-Scale

```python
def predict_full(model, image, scales=[1.0, 0.5], use_tta=True):
    """Combined multi-scale + TTA inference."""
    all_preds = []
    
    for scale in scales:
        scaled = resize(image, scale)
        
        if use_tta:
            pred = predict_with_tta(model, scaled)
        else:
            pred = model.predict(scaled)
        
        pred_native = resize(pred, 1/scale)
        all_preds.append(pred_native)
    
    return np.maximum.reduce(all_preds)  # Max across scales
```

---

## 10. Post-Training Calibration

### 10.1 Temperature Scaling

Addresses over/under-confident predictions:

```python
class TemperatureScaling:
    """
    Learn a single temperature parameter to calibrate probabilities.
    """
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels):
        """
        Find optimal temperature on validation set.
        Minimizes negative log-likelihood.
        """
        from scipy.optimize import minimize_scalar
        
        def nll(T):
            scaled = logits / T
            probs = torch.sigmoid(scaled)
            return F.binary_cross_entropy(probs, labels).item()
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        return self
    
    def calibrate(self, logits):
        """Apply learned temperature."""
        return torch.sigmoid(logits / self.temperature)
```

### 10.2 Threshold Selection

```python
def select_threshold(model, val_data, target_precision=0.8, ratios=[200, 500, 1000]):
    """
    Select operating threshold for each prevalence assumption.
    """
    thresholds = {}
    
    for ratio in ratios:
        # Compute PR curve at this ratio
        preds, labels = get_predictions_at_ratio(model, val_data, ratio)
        precision, recall, thresh = precision_recall_curve(labels, preds)
        
        # Find threshold achieving target precision
        valid = precision >= target_precision
        if valid.any():
            idx = np.where(valid)[0][-1]  # Highest recall at target precision
            thresholds[ratio] = {
                'threshold': thresh[idx],
                'precision': precision[idx],
                'recall': recall[idx]
            }
        else:
            thresholds[ratio] = {'threshold': None, 'note': 'Target precision not achievable'}
    
    return thresholds
```

---

## 11. Experiment Tracking with MLflow

### 11.1 Configuration

```python
import mlflow
import subprocess
import torch

def setup_mlflow_run(config, run_name):
    """Initialize MLflow run with comprehensive logging."""
    
    mlflow.set_experiment("rts-segmentation-v2")
    
    with mlflow.start_run(run_name=run_name):
        # Model hyperparameters
        mlflow.log_params({
            "architecture": config['architecture'],
            "backbone": config['backbone'],
            "loss": config['loss'],
            "focal_gamma": config.get('focal_gamma'),
            "focal_alpha": config.get('focal_alpha'),
            "learning_rate": config['learning_rate'],
            "batch_size": config['batch_size'],
            "warmup_epochs": config['warmup_epochs'],
            "gradient_clip_norm": config['gradient_clip_norm'],
            "ema_decay": config['ema_decay'],
            "boundary_ignore_width": config['boundary_ignore_width'],
            "freeze_backbone_epochs": config['freeze_backbone_epochs'],
            "mixed_precision": config['mixed_precision'],
            "random_seed": config['random_seed'],
        })
        
        # System info (not auto-captured by MLflow)
        mlflow.log_params({
            "git_commit": get_git_commit(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": str(torch.backends.cudnn.version()),
            "pytorch_version": torch.__version__,
            "gpu_model": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "data_version": config['data_version'],
        })
        
        # Save full environment
        save_requirements('requirements_frozen.txt')
        mlflow.log_artifact('requirements_frozen.txt')
        
        return mlflow.active_run()

def get_git_commit():
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode().strip()[:8]
    except:
        return 'unknown'

def save_requirements(path):
    """Save pip freeze output."""
    import pkg_resources
    with open(path, 'w') as f:
        for pkg in pkg_resources.working_set:
            f.write(f"{pkg.key}=={pkg.version}\n")
```

### 11.2 Per-Epoch Logging

```python
def log_epoch_metrics(epoch, train_metrics, val_metrics):
    """Log metrics for one epoch."""
    
    # Training metrics
    mlflow.log_metrics({
        "train_loss": train_metrics['loss'],
        "train_iou_rts": train_metrics['iou_rts'],
    }, step=epoch)
    
    # Validation metrics (all ratios)
    for ratio in [200, 500, 1000]:
        key = f'1:{ratio}'
        if key in val_metrics:
            mlflow.log_metrics({
                f"val_{ratio}_pr_auc": val_metrics[key]['pr_auc'],
                f"val_{ratio}_iou_rts": val_metrics[key]['iou_rts'],
                f"val_{ratio}_obj_precision": val_metrics[key]['obj_precision'],
                f"val_{ratio}_obj_recall": val_metrics[key]['obj_recall'],
            }, step=epoch)
    
    # Primary early stopping metric
    mlflow.log_metric(
        "val_realistic_pr_auc", 
        val_metrics['1:500']['pr_auc'],  # Use middle ratio
        step=epoch
    )
```

### 11.3 Final Artifacts

```python
def log_final_artifacts(model_path, pr_curves, threshold_config):
    """Log final model and analysis artifacts."""
    
    mlflow.log_artifact(model_path)  # best_model_ema.pth
    mlflow.log_artifact(pr_curves)   # pr_curves.png
    
    # Threshold calibration results
    with open('threshold_calibration.json', 'w') as f:
        json.dump(threshold_config, f, indent=2)
    mlflow.log_artifact('threshold_calibration.json')
    
    # Final test metrics
    mlflow.log_artifact('test_results.json')
```

### 11.4 Experiment Matrix

Run experiments in dependency order:

```
Phase 1: Baseline
└── RGB, UNet++, EfficientNet-B7, Focal loss, 1:10 ratio

Phase 2: Loss Ablation (on baseline)
├── Tversky (β > α for precision)
└── Class-balanced CE

Phase 3: Ratio Ablation (on best loss)
├── 1:20 fixed
├── 1:50 fixed
└── Curriculum (1:1 → 1:50)

Phase 4: Architecture (on best loss + ratio)
├── DeepLabV3+
├── SegFormer
└── SAM fine-tuned

Phase 5: Auxiliary Data (on best architecture)
├── RGB + NDVI
├── RGB + DEM derivatives
└── RGB + NDVI + DEM (full EXTRA)

Phase 6: Fusion Method (if EXTRA helps)
├── Early fusion (channel stack)
└── Late fusion (cross-attention)
```

---

## 12. Statistical Significance

### 12.1 Multiple Seeds

Single run results are noisy. For key experiments and final model:

```python
def run_with_multiple_seeds(config, seeds=[42, 43, 44]):
    """Run experiment with multiple seeds, report mean ± std."""
    
    results = []
    for seed in seeds:
        config['random_seed'] = seed
        set_seed(seed)
        
        metrics = train_and_evaluate(config)
        results.append(metrics)
    
    # Aggregate
    summary = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return summary

# Report format:
# IoU_RTS: 0.723 ± 0.012
# PR-AUC: 0.856 ± 0.008
```

### 12.2 Reproducibility Setup

```python
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For full determinism (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
```

---

## 13. Fallback: Two-Stage Approach

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

## 14. Estimated Resource Usage

### 14.1 Training Compute

| Stage | GPU Hours (8×H100) | Estimated Cost |
|-------|-------------------|----------------|
| Baseline + ablations | 50–100 | $5,000–10,000 |
| Architecture experiments | 50–100 | $5,000–10,000 |
| Auxiliary data experiments | 30–50 | $3,000–5,000 |
| Final model (3 seeds) | 30–60 | $3,000–6,000 |
| **Training Total** | ~200–300 | ~$20,000–30,000 |

### 14.2 Inference Compute

See Inference Pipeline document for estimates.

### 14.3 Budget Allocation

| Category | Allocation |
|----------|------------|
| Training experiments | $25,000 |
| Pan-arctic inference | $35,000 |
| Buffer / reruns | $10,000 |
| **Total** | $70,000 |

---

## 15. Training Checklist

### Pre-Training
- [ ] Data pipeline validated (correct channels, labels, augmentations)
- [ ] Boundary ignore masks created
- [ ] Balanced batch sampler configured
- [ ] Spatial blocking verified (no train/test geographic overlap)
- [ ] Normalization statistics computed and saved
- [ ] MLflow experiment configured
- [ ] Baseline config committed to version control
- [ ] GPU memory profiled, batch size fixed
- [ ] Library versions pinned (requirements_frozen.txt)
- [ ] Git commit hash logged

### During Training
- [ ] Mixed precision enabled
- [ ] Backbone frozen for first N epochs
- [ ] LR warmup active for first 5 epochs
- [ ] Gradient clipping enabled
- [ ] EMA updating after each step
- [ ] Monitor Val-Realistic PR-AUC for early stopping
- [ ] Watch for overfitting indicators
- [ ] Log learning curves to MLflow
- [ ] Checkpoint EMA weights as best model

### Post-Training
- [ ] Temperature scaling calibration
- [ ] PR curve analysis on Val-Realistic (all ratios)
- [ ] Threshold selection documented
- [ ] TTA evaluation
- [ ] Multi-scale evaluation (if applicable)
- [ ] Final metrics on Test-Realistic (all ratios)
- [ ] Object-level metrics (IoU=0.5 and IoU=0.3)
- [ ] Run with multiple seeds (42, 43, 44)
- [ ] Report mean ± std
- [ ] Model artifacts saved to MLflow

---

## 16. Code Organization (Suggested)

```
rts-segmentation-v2/
├── configs/
│   ├── baseline.yaml
│   └── experiments/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── sampler.py          # BalancedBatchSampler
│   │   ├── augmentations.py
│   │   ├── normalization.py    # Per-dataset normalization
│   │   └── boundary_mask.py    # Ignore region creation
│   ├── models/
│   │   ├── unetpp.py
│   │   ├── ema.py              # EMA implementation
│   │   └── backbones/
│   ├── losses/
│   │   ├── focal.py
│   │   ├── tversky.py
│   │   └── boundary_weighted.py
│   ├── metrics/
│   │   ├── pixel_metrics.py    # IoU, pixel P/R
│   │   └── object_metrics.py   # COCO-style object P/R
│   ├── training/
│   │   ├── trainer.py
│   │   ├── lr_scheduler.py     # Warmup + cosine
│   │   ├── early_stopping.py
│   │   ├── checkpoint.py
│   │   └── distributed.py      # Multi-GPU setup
│   ├── inference/
│   │   ├── tta.py              # Test-time augmentation
│   │   └── multiscale.py       # Multi-resolution inference
│   └── calibration/
│       └── temperature_scaling.py
├── scripts/
│   ├── train.py
│   ├── train_distributed.py    # Multi-GPU training
│   ├── evaluate.py
│   ├── evaluate_multiratio.py  # Eval at multiple prevalence ratios
│   └── calibrate.py
├── notebooks/
│   └── analysis/
└── mlruns/  # MLflow tracking
```