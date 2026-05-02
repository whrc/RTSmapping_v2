# Experiments

The experimentation plan for RTS Segmentation v2. Five phases of sequential elimination plus a final multi-seed lock. Each phase's winner becomes the next phase's baseline; there are no parallel branches whose winners are stitched together at the end.

## 1. Strategy

Cannot afford joint hyperparameter search. Sequential elimination is the only honest option. 

Phase 0 BS pick and LR range test run on production GPU so the chosen values are directly usable; no re-validation step. 

Test-Realistic touched once, ever (`training.md §10.3`) | All ranking happens on Val-Realistic. Test is for the final lock only. 

Calibration parity (`training.md §4.6`) | Threshold and temperature are calibrated with the deployment-time precision / TTA / scale / `torch.compile` config. Phase 5 architecture changes invalidate calibration; re-run in that case.

### 1.4 What "winning" means — the calibrated gate

Every phase from Phase 1 onward uses one definition of "winner":

> A candidate beats the current baseline if **both** hold on Val-Realistic at the in-training reporting threshold:
> 1. Δ `val_realistic_pr_auc_geomean` ≥ **G**, where **G = max(2σ₀, 0.01)** and σ₀ is the cross-seed std-dev measured in Phase 0.
> 2. Δ Precision @ Recall = 0.5 ≥ 0 (the gain is not bought by giving up precision; matches `training.md §1`'s precision-over-recall priority).

If multiple candidates pass, the winner is the one with the largest Δ PR-AUC. Ties default to the simpler operational config (e.g. focal beats compound beats tversky on operational simplicity; `none` boundary beats `ignore`; smaller backbone beats larger).

The gate floor of 0.01 prevents an unrealistically tight σ from generating spurious "winners" on differences smaller than reasonable run-to-run drift across machines and library versions. The 2σ multiplier prevents declaring winners on noise.

## 2. Tracking infrastructure

This section is the single source of truth for what each training run emits to MLflow. On-disk checkpoints are spec'd in `training.md §4.3`; the post-calibration deployment package in `inference.md §2.2`; the multi-seed reporting format in `training.md §13`.

### 2.1 MLflow configuration

GCS-backed file store. Configurable via YAML (`configs/baseline.yaml:mlflow`):

```yaml
mlflow:
  tracking_uri: "gs://abruptthawmapping/abrupt_thaw/RTS_MODEL_V2/mlflow_tracking"
  experiment_name: "rts-segmentation-v2"
  run_name: "<per-experiment-config-name>"
```

The `MLFLOW_TRACKING_URI` environment variable overrides the YAML when set. No separate tracking-server process; view runs locally with `mlflow ui --backend-store-uri gs://...`.

### 2.2 Required parameters logged

The full config YAML is logged via `training/mlflow_utils.py:_flatten_params` (every dotted key from `configs/<run>.yaml`). The table below highlights categories most relevant for cross-run comparison; it is illustrative, not exhaustive.

| Category | Parameters |
|----------|------------|
| Model | architecture, backbone, pretrained, input_channels (input size = `data.tile_size`) |
| Loss | loss_function, focal_gamma, focal_alpha, lambda_focal, lambda_dice, tversky_alpha, tversky_beta, boundary_handling, boundary_ignore_width |
| Optimizer | optimizer_name, weight_decay, gradient_clip_norm |
| Schedule | scheduler, frozen_lr, base_lr, backbone_lr_multiplier, warmup_epochs, backbone_warmup_epochs, freeze_backbone_epochs |
| Training | batch_size, max_epochs, early_stopping_patience (validation events), early_stopping_metric, ema_decay |
| Data | data_version, positive_fraction, curriculum_schedule, train_positive_subset_pct (Phase 2 only) |
| System | git_commit, pytorch_version, cuda_version, gpu_model, gpu_count |

### 2.3 Metrics and artifacts

**Metrics logged per epoch** (`scripts/train.py` via `mlflow_utils.log_metrics_step`):
- `train_loss`, `train_iou_rts`, `train_nan_steps`
- `scaler_scale`, `scaler_halves_this_epoch` (only when an AMP scaler is active, i.e. fp16)
- `val_balanced_iou`, `val_balanced_pr_auc`, `val_loss`
- For each ratio (200, 500, 1000): `val_{ratio}_pr_auc`, `val_{ratio}_iou_rts`, `val_{ratio}_obj_precision`, `val_{ratio}_obj_recall`
- `val_realistic_pr_auc_geomean` — geomean across the three ratios; the early-stopping metric

**Run-level metrics logged once at end of training** (`scripts/train.py:main` → `training/mlflow_utils.py:log_run_summary`):
- `exposure_max`, `exposure_median`, `exposure_p99`, `exposure_unique_tiles` — per-tile sample-count statistics across the whole run

**Run-level artifacts** (logged via `mlflow.log_artifact`):
- `config.yaml` — full training config, sorted-key dump
- `requirements_frozen.txt` — `pip freeze` at training start
- `run_summary.md` — human-readable summary (final metrics, NaN events, training duration)

**Per-validation figures** (rendered by `training/visualizations.py`, logged each validation epoch, rotated by `scripts/train.py:_rotate_artifacts` to keep the last 10 per pattern):
- `preview_epoch_*.png` — fixed 3-positive + 3-negative tile preview grid (RGB | GT overlay | predicted-prob heatmap)
- `pr_curves_epoch_*.png` — PR curves on Val-Realistic at ratios 1:200 / 1:500 / 1:1000
- `prob_hist_epoch_*.png` — log-scale histogram of predicted probabilities (mode-collapse detector)
- `confusion_epoch_*.png` — pixel-level confusion matrix at the in-training reporting threshold

**Checkpoints** (written to `runs/<name>/checkpoints/`, not MLflow): `best_deployment.pth`, `resume_latest-*.pth`. Payload contracts in `training.md §4.3`. The `best_deployment.pth` is uploaded to MLflow at end of run; resume snapshots stay local.

**Calibration outputs**: post-training, calibration writes `threshold` and `temperature` back into `configs/deployment.yaml` (`training.md §4.6 / §12`). The full deployment package is spec'd in `inference.md §2.2` and assembled by `scripts/package_model.py`.

---

## 3. Phase 0 — Baseline calibration

*Objective: establish a reliable baseline and the noise floor that drives every subsequent winner gate.*

Phase 0 is run on the production GPU (A100 or H100). It has three sub-steps in order: BS pick → LR range test → 3-seed baseline. The order matters because LR scales with the gradient noise scale ∝ LR / BS.

### Phase 0a — RGB normalization arm-out
 
*Objective: lock the RGB input pipeline before measuring the noise floor.*
 
### Rationale
 
The Phase 0 baseline applies per-dataset z-score to RGB. Two inherited assumptions deserve a check rather than a free pass: PlanetScope Visual is already CV-harmonized for downstream analytics (so per-dataset z-score over a harmonized product mostly captures the *content distribution* of training tiles, not sensor variation), and the smp EfficientNet-B5 pretrained weights were trained on `/255 → ImageNet mean/std` inputs (so per-dataset z-score silently shifts inputs away from what pretrained filters were optimised for, most consequentially during the frozen-backbone phase where filters cannot adapt).
 
### Arms
 
EXTRA channels are out of scope; per-channel z-score on physical-meaning bands stays at the `data/data.md §4.2` default.
 
| Arm | RGB preprocessing | Notes |
|---|---|---|
| A | Per-dataset z-score | Current spec default. |
| B | `x / 255` then ImageNet mean/std | Honors pretrained backbone statistics. Use the preprocessing values shipped with the smp encoder weights, not a textbook copy. |
| C | `x / 255` only | Tests whether mean/std subtraction matters at all on a harmonized product. |
 
### Procedure and decision
 
Seed 42 only, all three arms, all other hyperparameters at the `configs/baseline.yaml` defaults. σ₀ does not exist yet, so the §1.4 gate cannot apply its `2σ₀` term — fall back to the gate floor (Δ `val_realistic_pr_auc_geomean` ≥ 0.01).
 
| Outcome | Action |
|---|---|
| No arm beats A by Δ ≥ 0.01 | Lock A. |
| Exactly one of B / C beats A | Lock that arm. |
| Both B and C beat A | Lock the larger Δ. Tie-break: C beats B (no stats file, no recomputation when training data changes). |
 
The locked arm becomes the input pipeline for Phase 0 and every phase after. `data/normalization.py` and the on-disk `normalization_stats.json` are updated (or the stats file is removed, for arm C) before Phase 0 begins.
### 3.1 Batch-size pick

Pick the largest BS that fits memory comfortably (~85% of VRAM) at the locked precision (BF16). Hypothesis under balanced sampling at `positive_fraction = 0.5`: larger BS = more positive instances per gradient step, which is favorable in this small-data regime. This is a defensible default, **not** a universal truth.

The current default (`configs/baseline.yaml:training.batch_size = 32`) was chosen for the L4 dev VM. On A100/H100 the comfortable cap is likely 64 or 128; verify with one short run that profiles peak memory and step time.

Do not add a "BS vs quality" comparison run unless Phase 0 multi-seed produces evidence that the chosen BS is wrong (e.g. high cross-seed variance in early epochs that disappears at smaller BS).

### 3.2 LR range test (Smith 2017)

Run twice on a 30% data subset for ~1 epoch each, on the production GPU at the BS chosen in §3.1.

| Pass | Setting | Output |
|---|---|---|
| Frozen-phase test | Backbone frozen (Phase 1 of `training.md §10.2`); ramp LR 1e-7 → 1e-1 over the epoch | Picks `frozen_lr` |
| Unfrozen-phase test | Backbone unfrozen (Phase 2 of `training.md §10.2`); ramp LR 1e-7 → 1e-1 over the epoch | Picks `base_lr` |

The picked LR is the order of magnitude where the loss curve has the steepest stable descent before divergence. Defaults in `configs/baseline.yaml:lr_schedule` (`frozen_lr = 1e-3`, `base_lr = 1e-4`) are starting points; the range test may revise.

Implemented and active: `training/scheduler.py:_make_lr_range_test_setter` drives the LR ramp; `scripts/train.py:_filter_train_positive_subset` provides the 30 % data subset.

### 3.3 Multi-seed baseline

Run the locked baseline (BS from §3.1, LR from §3.2, all other parameters from `configs/baseline.yaml`) at seeds **42, 43, 44**. Three seeds is the minimum to estimate σ; running fewer would mean Phase 0 cannot calibrate the §1.4 gate.

Each run writes its own MLflow run with `run_name: phase0_seed{seed}`. After all three complete:

- σ₀ = std-dev of `val_realistic_pr_auc_geomean` (the early-stop best, smoothed) across the three seeds.
- μ₀ = mean of the same metric. This is the baseline number every subsequent phase compares against.

### 3.4 σ → protocol decision matrix

The measured σ₀ feeds three decisions: the §1.4 gate, the seed protocol for Phase 3+ comparisons, and the seed count for the Final phase.

| σ₀ band | Designation | §1.4 gate | Phase 3+ comparison protocol | Final-phase seed count |
|---|---|---|---|---|
| σ₀ < 0.005 | **Low-noise** | G = 0.01 | Single seed (42) per candidate is reliable for ranking. | 3 seeds (42, 43, 44) |
| 0.005 ≤ σ₀ < 0.015 | **Medium-noise** | G = 2σ₀ ∈ [0.01, 0.03] | Single seed for first-pass ranking. Top 1–2 candidates per phase that land within 1σ of each other are re-run at seed 43 to break ties. | 3 seeds (42, 43, 44) |
| σ₀ ≥ 0.015 | **High-noise** | G = 2σ₀ > 0.03 | Single-seed comparisons are unreliable. **Either** run all serious candidates with 2 seeds (42, 43) at 2× compute, **or** investigate the noise source (sampler stochasticity, curriculum boundary effects, early-stop thrashing) and re-run Phase 0 after fixing it. | 5 seeds (42, 43, 44, 45, 46) |

The decision is recorded in the Phase 0 results doc. Subsequent phases reference σ₀ by value, not by re-measurement.

---

## 4. Phase 1 — Temporal sanity check (2025 micro-set)

*Objective: detect material domain shift between 2024 training imagery and 2025 deployment imagery before investing in further tuning.*

Phase 1 and Phase 2 can run in parallel because Phase 1 is gated externally (the 2025 micro-set is TBD).

### 4.1 Procedure

Run inference using the Phase 0 baseline checkpoint on the 2025 micro-set. Compute PR-AUC on the micro-set. Define:

> Δ_relative = (PR-AUC_2024 − PR-AUC_2025) / PR-AUC_2024

Relative Δ is used because absolute thresholds (e.g. ≤ 0.05) read very differently at PR-AUC = 0.4 versus PR-AUC = 0.8.

### 4.2 Decision bands

| Δ_relative | Interpretation | Action |
|---|---|---|
| ≤ 10 % | Negligible drift | Proceed to Phase 2/3/4 with 2024-only training. Re-evaluate once Phase 4 winner is locked. |
| 10–20 % | Modest drift | Run `scripts/check_inference_normalization.py` (`inference.md §5.4`) on a larger 2025 sample. If radiometric drift drives most of the gap, consider per-region calibration during deployment (`training.md §6.4`). Continue Phase 2/3 in parallel. |
| > 20 % | Material drift | Halt. Investigate radiometric drift first. If real, the project either retrains with 2025 data included (requires labeling effort) or restricts scope to the 2024 distribution. Phases 2–5 are not invalidated but their winners may not generalise. |

### 4.3 Status

Blocked on the 2025 micro-set definition: tile count, region selection, labeling plan. **TBD** for the user. The Phase 1 inference is one GPU-hour once the micro-set exists.

---

## 5. Phase 2 — Data scaling

*Objective: determine whether more positive labels are likely to help, and whether the model has the capacity to use them.*

Run at the **current ~1900 positives**, marked provisional. Re-run on the full 3500 once labeling completes if and only if a downstream decision (Phase 3 backbone choice, Phase 5 gating) flips on the result.

### 5.1 Procedure

Train the Phase 0 baseline on 25 %, 50 %, 75 %, 100 % of the available positive tiles. The negative pool is held constant — only positives are subsetted, so the curriculum sampler still draws from the full negative set. All other hyperparameters held at the Phase 0 values.

| Subset | Approximate positives at 1900 total | Notes |
|---|---|---|
| 25 % | ≈ 475 | Smallest point on the curve; high variance expected. |
| 50 % | ≈ 950 | |
| 75 % | ≈ 1425 | |
| 100 % | ≈ 1900 | Same as Phase 0 baseline; no separate run if seeds match. |

Plot `val_realistic_pr_auc_geomean` and `val_{ratio}_iou_rts` versus log(n_positives).

### 5.2 Subset mechanism

The config key `splits.train_positive_subset_pct` selects a deterministic seeded subsample of positive tile_ids from `splits.yaml.train` at dataset-construction time. Negatives are not subsetted. Implemented in `scripts/train.py:_filter_train_positive_subset` and active.

### 5.3 Slope decision matrix

Fit a line to `PR-AUC vs log(n_positives)`. Compare the slope between the 75 → 100 % points to the slope between 25 → 50 %.

| Slope ratio (75→100) / (25→50) | Designation | Implications |
|---|---|---|
| < 0.5 | **Plateau before 100 %** | Model has saturated on the available data. Phase 3 should focus on loss / regularisation rather than capacity. Phase 5 is likely a skip. Acquiring more labels above 1900 has weak expected return. |
| 0.5 – 1.0 | **Diminishing but still scaling** | Continued returns. Phase 3 backbone-sizing is worth running if loss-family results suggest underfit. Phase 5 stays in scope. |
| > 1.0 (slope flat / increasing) | **Severely under-scaled** | Even 25 % is enough to start; the model has plenty of capacity left. Phase 5 stays in scope; backbone-up testing in Phase 3 is high-priority. |

### 5.4 Generalisation-gap monitoring

For each subset, track the gap between `train_iou_rts` and `val_realistic_iou_rts` at the final epoch. Indicators:

| Observation | Inference |
|---|---|
| Gap < 0.2 across all subsets | Data variance constrains capacity; weight-decay and augmentation defaults are fine. |
| Gap > 0.4 at 100 % | Severe over-parameterisation. Conditional weight-decay sweep is warranted in Phase 3 (see §6.3). |

The gap-vs-data-size signal also informs whether Phase 5's "did Phase 3+4 close the gap?" gate has a chance.

---

## 6. Phase 3 — Loss family → boundary handling

*Objective: tune the penalty landscape to suppress false positives without sacrificing recall.*

Sequential elimination: pick the loss family first, lock it, then test boundary handling against the locked loss winner. Conditional weight-decay sweep only if Phase 2 shows the over-parameterisation signal.

### 6.1 Loss family

Compared on the Phase 0 baseline (locked LR, BS, augmentation). Each candidate is a single training run unless Phase 0's σ₀ band requires a second seed.

| Candidate | Configuration | Notes |
|---|---|---|
| Focal (baseline) | γ = 2, α = 0.25 (`configs/baseline.yaml` defaults) | Reference from Phase 0. |
| Compound (Focal + Dice) | λ_focal : λ_dice ∈ { 1:1, 1:2, 2:1 } — 3 runs | Priority candidate per `training.md §5.3`. |
| Tversky (precision-focused) | (α, β) ∈ { (0.3, 0.7), (0.2, 0.8) } — 2 runs | β > α only, per `training.md §5.2`. |
| Focal grid (only if all of the above plateau at gate G) | γ ∈ {1, 2, 3} × α ∈ {0.25, 0.5} \ {(2, 0.25)} — 5 cells | Tuning ranges from `training.md §5.1`. |

**Loss winner selection**: §1.4 gate. Tie-breaking by operational simplicity: focal < compound < tversky.

### 6.2 Boundary handling

Run **after** §6.1 locks. The loss-winner config is held constant; only `loss.boundary_handling` and `loss.boundary_ignore_width` change.

| Configuration | Notes |
|---|---|
| `boundary_handling: none` (baseline) | Inherited from §6.1. |
| `boundary_handling: ignore`, width ∈ {1, 2, 3} | 3 runs. Soft-label is deferred to v2.1 per `training.md §5.5`; `data/dataset.py` raises `NotImplementedError` if requested. |

§1.4 gate. Operational tie-break: `none` beats `ignore` (less data prep, no dilation step at load time).

### 6.3 Conditional weight-decay sweep

Run only if Phase 2's §5.4 gap is > 0.4 at 100 % data. Otherwise skip — defaults are fine.

| Candidate | `optimizer.weight_decay` |
|---|---|
| Baseline | 1e-2 |
| Stronger | 5e-2 |

Single-pass against the §6.2 locked config. If 5e-2 passes the §1.4 gate **and** does not destroy precision, lock it. Otherwise revert.

The "raise augmentation probabilities" remedy mentioned in `training.md §10.5` is intentionally not run as a sweep here. Augmentation probabilities sit in `configs/baseline.yaml:augmentation` and remain at their defaults until evidence forces revisiting (§10).

### 6.4 Phase 3 deliverable

A single locked configuration {loss family + parameters, boundary handling, weight decay if changed}. This is the new baseline for Phase 4.

Backbone sizing (B3 / B7 vs B5) is **deferred to Phase 5**, not run inside Phase 3, because the right backbone depends jointly on Phase 2's slope (capacity utilisation) and Phase 4's channel decision (input dimensionality).

---

## 7. Phase 4 — EXTRA channel groups

*Objective: determine whether multi-modal physical context improves the final map, and if so which combination to deploy.*

### 7.1 Group definitions

Group IDs and their band positions are fixed by `data/data.md §9` (single source of truth). Phase 4 ablates the five v2.0 groups:

| Group ID | N bands | Band indices in EXTRA |
|----------|---------|------------------------|
| `NDVI` | 1 | 0 |
| `NBR` | 1 | 1 |
| `SE_PCA` | 3 | 2, 3, 4 |
| `SE_PROTO` | 1 | 5 |
| `TC` | 2 | 6, 7 |

Channel descriptions, sources, and rationale live in `data/data.md §9` — this table is a quick reference for the ablation plan. Each group is selected by listing the corresponding `{name, band}` entries under `channels.extra` in the experiment config (`data/data.md §3.3`).

### 7.2 Single-group ablation

For each declared group EXTRA_i, train one run with `channels.extra = [<EXTRA_i entries>]`. RGB always on; one group on top.

Each run is gated by §1.4 against the Phase 3 baseline (RGB-only, locked loss + boundary). A group **passes individual ablation** if it clears the gate.

### 7.3 Full-stack ceiling

One run with `channels.extra` containing every declared group simultaneously. Establishes the upper bound on what stacking can achieve.

If the full stack fails the §1.4 gate against the Phase 3 baseline, **stop**: no combination is worth deploying. The winner is RGB-only (Phase 3 lock).

### 7.4 Greedy combination

Run only if the full stack passes §1.4 **and** at least one single group passed §7.2.

1. Start with RGB + the single group with the largest Δ in §7.2.
2. Greedily add the next-most-helpful group (by §7.2 ranking).
3. Stop adding when the next group fails the §1.4 gate against the current best combination, or when all groups that passed §7.2 are included.

**Free-rider rule**: each group in the final combination must have individually passed §7.2. A group whose §7.2 single-group result failed the gate cannot be in the deployed stack — even if adding it on top of an existing combination measures positive, that gain is most likely a stochastic ride on the existing stack's signal rather than an independent contribution.

### 7.5 Fusion strategy

Run only if §7.4's combination has ≥ 2 channel groups *and* beats every single-group result by §1.4.

Default is **early fusion** (channel stacking, single encoder). Early fusion is the implementation in `scripts/train.py` today. **Late fusion** (separate encoders → feature-level fusion) requires an architecture change and is not implemented; the user must explicitly authorise late-fusion implementation before this sub-phase starts.

### 7.6 Phase 4 deliverable

A locked `channels.extra` list (possibly empty if §7.3 failed). This is the input to Phase 5 and the Final lock.

---

## 8. Phase 5 — Architecture (gated)

*Objective: test if a more expressive feature extractor yields meaningful gains over UNet++ + EfficientNet-B5.*

### 8.1 Trigger conditions

Phase 5 runs only when **both** are true:

1. Phase 2 §5.3 designated the regime as **diminishing but still scaling** or **severely under-scaled** (slope ratio ≥ 0.5). Architecture matters when data is plentiful enough to feed it.
2. The Phase 3 + Phase 4 winners have not closed the train-val IoU gap below ~0.3 at 100 % data. If the gap is small, the model is already using its capacity; bigger architectures will overfit.

Both conditions failing → **skip Phase 5**. Document the skip with the slope ratio and gap value as evidence in `docs/phase5_skip.md`. The honest expectation in this regime (≈ 1900 positives) is that Phase 5 gets skipped, and that is an acceptable, principled outcome.

### 8.2 Comparison set (only if §8.1 triggers)

Per `training.md §3.2` priority order:

| Candidate | Notes |
|---|---|
| UNet++ + EfficientNet-B5 (Phase 3/4 winner) | Reference. |
| UNet++ + EfficientNet-B3 | Cheaper backbone — runs only if Phase 2 indicates "plateau before 100%" but Phase 3+4 didn't close the gap (capacity maybe overshot). |
| UNet++ + EfficientNet-B7 | Larger backbone — runs only if Phase 2 indicates "severely under-scaled". |
| SegFormer-B5 | Architectural change; needs `models/segmentation.py` extension. |
| DINOv3 encoder + dense head | Architectural change; confirm DINOv3 model version at implementation time. |

### 8.3 Winner criteria

§1.4 gate **plus** the new architecture must pass `scripts/inference_feasibility.py` (re-run with the candidate). A model that wins on PR-AUC but breaks the inference budget for the §3.2 pan-arctic tile count (`inference.md §3.2`; ~7.5M at default stride 344) is not a winner.

Calibration parity (`training.md §4.6`) requires the calibrated threshold + temperature to be re-derived for any architecture change; the locked Phase 4 calibration does not transfer.

---

## 9. Final — multi-seed lock and Test-Realistic report

*Objective: produce the deployment configuration and the single Test-Realistic number that ships.*

After the last winning phase (Phase 3, 4, or 5 depending on what triggered), retrain the locked configuration at the seed count from §3.4's σ → protocol matrix.

| Action | Reference |
|---|---|
| Retrain locked config at k seeds (k ∈ {3, 5}) | `training.md §13.1` |
| For each seed run: post-training calibration sequence | `training.md §10.4` (TTA → multi-scale gate → temperature → threshold) |
| Lock `configs/deployment.yaml` from the seed-42 calibration | `training.md §4.6` |
| Run `scripts/evaluate_test.py` on Test-Realistic with the locked deployment config | `training.md §10.3` |
| Build deployment package via `scripts/package_model.py` | `inference.md §2.2` |
| Report mean ± std on Test-Realistic at all three ratios | `training.md §13.2` |

Test-Realistic is touched **once**, at this step. Re-running Test-Realistic for any reason after this is a project-discipline failure, not a permitted iteration.

---

## 10. What we deliberately don't tune

These knobs sit in `configs/baseline.yaml` and are technically tunable, but tuning them gives near-zero expected information per GPU-hour at this project's regime. They stay at their defaults unless evidence forces revisiting.

| Knob | Default | Why we don't tune | What would force revisiting |
|---|---|---|---|
| `optimizer.name` | `adamw` | AdamW is the strong default for segmentation. Switching to SGD adds a momentum knob without expected gain. | Catastrophic Phase 0 instability that AdamW betas can't fix. |
| AdamW betas | `(0.9, 0.999)` (PyTorch default) | Standard. No reason to expect a 0.95 / 0.99 swap helps. | Same as above. |
| `ema.decay` | 0.999 | The 0.999 vs 0.99 vs 0.9999 spread is generally < 0.01 PR-AUC; below the §1.4 gate. | An exposure-counter pattern showing extreme tile overfitting. |
| `lr_schedule.warmup_epochs`, `backbone_warmup_epochs` | 5, 3 | Defaults are within the literature normal range. Phase 0 LR range test makes them less critical. | Phase 0 multi-seed showing high-σ runs that originate from warmup-period instability. |
| `optimizer.gradient_clip_norm` | 1.0 | Safe default. Loosening or removing risks NaN events under focal loss with extreme imbalance. | Repeated `train_nan_steps` > 0 across seeds. |
| `augmentation.*` probabilities | as in `configs/baseline.yaml` | The aug pipeline was tuned in v1; per-aug ablations are diminishing-returns search. | §5.4 generalisation-gap > 0.4 (then a coarse "all aug p × 1.5" trial, not a per-aug grid). |
| Soft-label boundary handling | not implemented | Deferred to v2.1; `data/dataset.py` raises if requested. | `boundary_handling: ignore` clearly fails to capture annotation noise. |
| Copy-paste augmentation | not implemented | Deferred. Adds implementation surface area for an effect of uncertain magnitude. | Phase 4 reveals positive recall is the bottleneck. |

The trigger for revisiting any of these is **evidence**, not a calendar slot or a feeling that "we should also try X."

---

## 11. Execution

### 11.1 Config naming

Each experiment is one YAML file in `configs/`. The repository commits only the
two configs needed to start any experiment chain:

```
configs/baseline.yaml    ← Phase 0 baseline (and Phase 0a Arm A)
configs/deployment.yaml  ← post-calibration deployment config (per inference.md §2.2)
```

All phase-specific configs are created **on demand**, one per experiment, as the
predecessor phase locks. Naming convention when created:

```
configs/phase0a_arm_b.yaml             ← Phase 0a §Arms — x/255 + ImageNet stats
configs/phase0a_arm_c.yaml             ← Phase 0a §Arms — x/255 only
configs/phase0_lr_test_frozen.yaml     ← Phase 0 §3.2 frozen-phase LR range
configs/phase0_lr_test_unfrozen.yaml   ← Phase 0 §3.2 unfrozen-phase LR range
configs/phase0_seed{42,43,44}.yaml     ← Phase 0 §3.3 multi-seed baseline
configs/phase2_scale_{25,50,75,100}.yaml  ← Phase 2 §5.1 — N% positives
configs/phase3_loss_<family>.yaml      ← Phase 3 §6.1 — per loss-family candidate
configs/phase4_extra_<group_name>.yaml ← Phase 4 §7.1 — per EXTRA group
configs/final_seed{42,43,44}.yaml      ← Final §9 multi-seed lock
```

Each new config copies the prior phase's winner hyperparameters into a fresh
file, then changes only the keys this experiment is testing. This avoids
the drift class that pre-made placeholder configs introduced (audit
2026-05-01: 12 configs deleted; review 2026-05-02: `output_bias_prior` had
reverted in 14 of the remaining configs).

### 11.2 Per-phase results docs

Each phase produces one results doc in `docs/`. Conventions match `CLAUDE.md` ("each major experiment has a single md document").

```
docs/phase0_baseline.md       ← BS, LR, σ₀, μ₀, σ-band designation
docs/phase1_2025_sanity.md    ← Δ_relative, action band
docs/phase2_data_scaling.md   ← curve, slope ratio, regime designation
docs/phase3_loss_boundary.md  ← winner config + per-candidate Δ values
docs/phase4_extra_channels.md ← group ablation + final stack
docs/phase5_arch_or_skip.md   ← either skip evidence or comparison results
docs/final_lock.md            ← Test-Realistic table at k seeds + deployment-config snapshot
```

Each doc records: design decision (what changed vs the previous phase), implementation details (config paths, MLflow run IDs), results (numbers + figures), analysis (why the winner won, residual concerns).

### 11.3 Decisions requiring human input

The following decision points cannot be made autonomously and require explicit user sign-off before the corresponding phase can run:

| Decision | Phase blocked | Owner |
|---|---|---|
| 2025 micro-set scope (tile count, region selection, labeling plan) | Phase 1 | User + Heidi Rodenhizer |
| `splits.train_positive_subset_pct` config-key implementation | Phase 2 (mechanism) | Engineer |
| Late-fusion authorisation if §7.4 calls for it | Phase 4 §7.5 | User |
| Architecture extension to `models/segmentation.py` for SegFormer / DINOv3 | Phase 5 (if triggered) | Engineer |
| Re-running Phase 2 on full 3500 positives | Phase 3+ (if any decision flips on the 1900 result) | User |

Phases run sequentially when not externally blocked. When externally blocked, the next-runnable phase proceeds.
