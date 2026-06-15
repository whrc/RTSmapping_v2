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

### 1.5 Compute posture (8×A100) — saturate, don't stop

This program was originally written under single-GPU scarcity (single-seed first passes, one-point
conditional probes, strict sequential elimination). The node is now **8× A100-80GB** with an abundant
$70k credit (~$5–15k earmarked for ablations; everything proposed ≈ $2–4k). **Compute is not the
constraint — labeled data is.** Two consequences govern execution (full scheduling SSoT in §13):

- **Keep all 8 GPUs busy, within and across phases.** Pad under-full waves and backfill idle slots
  with *independent* runs (the §8 architecture sweep, the §10 curriculum sweep, pre-registered seeds);
  use gated-speculative dispatch for dependency-blocked runs (§13). Saturate utilization *per run* by
  reading data from local disk, not per-epoch from GCS (§13).
- **Do not stop this scarce 8×A100 to save the trivial idle cost** — restart risks GPU stockout
  (`vm_instruction.md` zone-fallback). Stop only for genuinely long idle blocks, after backing up
  keepers to GCS (`infrastructure.md`).

### 1.6 Expectations — incremental vs step-change

Be honest about ceilings. On the **v1.0 data plateau** (§5.3) the model is data-constrained yet
*not* over-parameterised (train/val IoU gap ≈ 0.05 at best epoch, §5.4) — well-matched to its data.
So the cheap hyperparameter levers (loss/boundary §6, regularization §6.3, curriculum §10, decoder
swaps §8) are expected to yield **incremental** gains (~around the gate G = 0.011), and bigger or
more-regularised models are *not* indicated. The **step-changes** live in representation: EXTRA
channels (Phase 4), foundation-encoder architectures (§8), and the user-gated arms (§12). Plan and
report accordingly — do not oversell hyperparameter tuning on a plateaued, well-fit model.

## 2. Tracking infrastructure

This section is the single source of truth for what each training run emits to MLflow. On-disk checkpoints are spec'd in `training.md §4.3`; the post-calibration deployment package in `inference.md §2.2`; the multi-seed reporting format in `training.md §13`.

### 2.1 MLflow configuration

Local file store inside Docker (`/outputs/mlflow` → host `/mnt/outputs/mlflow`). Configurable via YAML
(`configs/baseline.yaml:mlflow`):

```yaml
mlflow:
  tracking_uri: "file:///outputs/mlflow"   # MLflow 2.x+ does NOT support gs:// as a *tracking* backend
  experiment_name: "rts-segmentation-v2"   # (gs:// is valid only as an artifact store). Corrected 2026-06-07.
  run_name: "<per-experiment-config-name>"
```

The `MLFLOW_TRACKING_URI` environment variable overrides the YAML when set. No separate tracking-server
process; view runs with `mlflow ui --backend-store-uri file:///outputs/mlflow`. **The file store is not
concurrency-safe** — for parallel multi-GPU runs switch to a DB backend (e.g. Cloud SQL Postgres) or give
each GPU its own store and merge at report time.

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

**Metrics logged per epoch** (`scripts/train.py` via `mlflow_utils.log_metrics_step`). Names below are the
**actual keys emitted by `training/metrics.py` (the code SSoT)** — reconciled 2026-06-07:
- Train (per epoch): `train_loss`, `train_iou` (pixel IoU over train batches — added 2026-06-07 for §5.4
  gap + §8.1 Phase-5 gate), `train_nan_steps`
- `scaler_scale`, `scaler_halves_this_epoch` (only when an AMP scaler is active, i.e. fp16; bf16 = inactive)
- Val (per validation): `val_loss`; **global** `pixel_iou`, `pixel_f1`, `object_precision`,
  `object_recall`, `object_f1` (at `metrics.reporting_threshold`); `val_n_positive_tiles`,
  `val_n_negative_tiles`
- Per gate ratio r ∈ **`metrics.pr_auc_ratios` (gate = `[5, 10, 20]`)**: `pr_auc_ratio_{r}`
- `val_realistic_pr_auc_geomean` — geomean across the gate ratios; the early-stopping + selection metric

> **Gate ratios are data-limited.** The deployment-prevalence ratios 1:200/500/1000 referenced elsewhere
> in this doc (§5.1, §9) cannot be honestly evaluated at the ~16k negative ceiling, so the **gate** uses
> `[5,10,20]`; 1:200/1000 are deferred to **Test-Realistic** (§9), reported with CIs + the limitation
> stated. Per-ratio IoU/obj-precision and `val_balanced_*` are **not** currently emitted (only per-ratio
> PR-AUC); add them only if a phase needs them.

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

### 4.3 Status — OPTIONAL (not a gate)

**Downgraded to optional (2026-06-15).** A visual review of the 2025 basemap preview showed no
alarming domain shift, and Phase 1 is externally blocked on the micro-set definition anyway (tile
count, region selection, labeling plan — **TBD** for the user). Phases 2–5 proceed regardless; Phase 1
is **not** a gate. Keep it available as a cheap (~1 GPU-hour) sanity check to run opportunistically
once the micro-set exists, and re-confirm at the v2.1 re-baseline. If run and Δ_relative > 20% (§4.2),
re-open it as a blocker at that point.

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

**v1.0 result (2026-06-15) — PLATEAU.** `best_smoothed` at 25/50/75/100% positives (n≈328/656/983/1311)
= 0.7636 / 0.7720 / 0.7916 / 0.7912. Slope ratio (75→100)/(25→50) = **−0.12 < 0.5 → plateau before
100%** (75→100 flat within σ₀). Implications per the matrix: the model has **saturated on available
positives**, Phase-5 architecture is **likely skipped** (its §8.1 slope trigger fails), and acquiring
more RGB labels above ~1300 has **weak expected return** at this architecture. Leverage therefore
pivots to representation (Phase 4 EXTRA, foundation encoders §8). See `docs/phase2_data_scaling.md`.
*(Caveat: contrast with v2.0-alpha's "severely under-scaled, slope 4.4" — the cleaner v1.0 data
changed the regime.)*

### 5.4 Generalisation-gap monitoring

For each subset, track the gap between `train_iou_rts` and `val_realistic_iou_rts` at the final epoch. Indicators:

| Observation | Inference |
|---|---|
| Gap < 0.2 across all subsets | Data variance constrains capacity; weight-decay and augmentation defaults are fine. |
| Gap > 0.4 at 100 % | Severe over-parameterisation. Conditional weight-decay sweep is warranted in Phase 3 (see §6.3). |

The gap-vs-data-size signal also informs whether Phase 5's "did Phase 3+4 close the gap?" gate has a chance.

**v1.0 result (2026-06-15).** At 100% data (baseline `phase0c_seed42`, from MLflow `train_iou` vs val
`pixel_iou`): gap = **0.05 at the best epoch (55)**, **0.17 at the final epoch (105)** — both **< 0.4**.
→ "Gap < 0.2 → data variance constrains capacity; wd/aug defaults are fine." **v1.0 is NOT
over-parameterised** (contrast alpha's 0.43). So the §6.3 conditional weight-decay trigger **does not
fire** for v1.0; the §6.3 regularization grid is therefore run as a **low-expected-value exploratory
wave** (user-elected to confirm, not because the trigger demands it). Combined with the §5.3 plateau,
this says the model is *well-matched* to its data — neither more capacity nor more regularization is
indicated; the binding constraint is the data/representation.

---

## 6. Phase 3 — Loss family → boundary handling

*Objective: tune the penalty landscape to suppress false positives without sacrificing recall.*

**Factorial, not pure sequential (revised 2026-06-15 for 8×A100).** Pure sequential elimination
assumes the best boundary handling is loss-independent; with compute abundant we test the interaction
instead. The loss-family **first pass** (§6.1, Wave-1) ranks the families at `boundary: none`; the
**boundary factorial** (§6.2) then runs the top-2 loss families × boundary settings in parallel, so the
boundary choice is made with any loss×boundary interaction visible. The §6.3 regularization grid is run
as an exploratory wave (the v1.0 over-parameterisation trigger did **not** fire — §5.4).

### 6.1 Loss family — first pass

Compared on the Phase 0 baseline (locked LR, BS, augmentation). Each candidate is a single training run unless Phase 0's σ₀ band requires a second seed.

| Candidate | Configuration | Notes |
|---|---|---|
| Focal (baseline) | γ = 2, α = 0.25 (`configs/baseline.yaml` defaults) | Reference from Phase 0. |
| Compound (Focal + Dice) | λ_focal : λ_dice ∈ { 1:1, 1:2, 2:1 } — 3 runs | Priority candidate per `training.md §5.3`. |
| Tversky (precision-focused) | (α, β) ∈ { (0.3, 0.7), (0.2, 0.8) } — 2 runs | β > α only, per `training.md §5.2`. |
| Focal grid (only if all of the above plateau at gate G) | γ ∈ {1, 2, 3} × α ∈ {0.25, 0.5} \ {(2, 0.25)} — 5 cells | Tuning ranges from `training.md §5.1`. |

**Loss family ranking**: §1.4 gate. Tie-breaking by operational simplicity: focal < compound < tversky.
Take the **top-2** families forward into the §6.2 factorial (in the medium-noise regime, candidates
within 1σ get a seed-43 tie-break run, §3.4). *Wave-1 status (2026-06-15): compound family clearly
leads; tversky weak — so top-2 ≈ {compound-best-ratio, focal-or-compound-runner-up}.*

### 6.2 Boundary handling — loss×boundary factorial

Replaces the old "run after §6.1 locks, boundary-on-winner only." Run the **top-2 loss families ×
boundary settings** in parallel (the `none` cells already exist from §6.1 / Wave-1; run the missing
`ignore` cells), padded to 8 GPUs with seeds (§13):

| Axis | Levels |
|---|---|
| Loss family | top-2 from §6.1 |
| Boundary | `none` (have), `ignore` width ∈ {1, 2, 3} |

= 2 × 4 cells (≈6 new `ignore` runs). Soft-label boundary is deferred (`training.md §5.5`;
`data/dataset.py` raises `NotImplementedError`). Each cell holds its loss config constant and changes
only `loss.boundary_handling` / `loss.boundary_ignore_width`. **Decision**: §1.4 gate on the best cell;
if the boundary effect is consistent across both losses → loss-independent (pick the simpler boundary);
if it flips → report the interaction and pick per the winning loss. Operational tie-break: `none` beats
`ignore`. Across-stage scheduling (gated speculative on the loss leader) per §13.

### 6.3 Regularization grid (wd × aug)

Replaces the old conditional single-point wd probe (proposal #6). **v1.0 trigger status: the §5.4 gap
is 0.05/0.17 < 0.4 → over-parameterisation trigger did NOT fire**, so this grid is an **exploratory,
low-expected-value wave** the user elected to run for completeness (compute is free), not a
trigger-mandated sweep. Run the 2×2 grid in parallel against the §6.2 winner:

| Axis | Levels |
|---|---|
| `optimizer.weight_decay` | 1e-2 (baseline), 5e-2 (stronger) |
| Augmentation strength | base (`configs/baseline.yaml:augmentation`), strong (all aug p × ~1.5, `training.md §10.5`) |

= 4 cells. Lock a cell only if it passes the §1.4 gate **and** does not destroy precision; otherwise
revert to baseline (the expected outcome on a well-fit model). This supersedes the §10 "don't tune aug"
default for this one coarse grid only.

### 6.4 Multi-scale training arm (#3) — POST-INFERENCE GATED

**Do not run now.** Trigger = pan-arctic inference is complete **and** a review of the deployed map
shows a large-RTS / wide-FOV coverage gap (motivated by the 2026-06-12 finding that the single-GSD
model collapses at 2× GSD — 0 vs 9 blobs, `docs/inference_validation.md`). If triggered: add a training
arm on 2×-area-downsampled tiles (`inference.md §6.4` "Phase-1.5" path), ~1 day of data-fetch/downsample
pipeline work, evaluated via the `inference.md §6.4` gate. Listed in §11.3 as a user-gated decision.

### 6.5 Bounded interaction check

Sequential elimination assumes separable knobs. Now affordable: one small factorial on the
most-coupled triple — **loss-winner family × wd {1e-2, 5e-2} × curriculum {base, precision-tilted}**
(≈8 runs, parallel) — to confirm no interaction flips the per-axis winners from §6.2/§6.3/§10. Keep
sequential elimination as the backbone; this is a guard, not a full joint search. If an interaction
flips a winner, re-lock at the joint optimum and note it in `docs/phase3_loss_boundary.md`.

### 6.6 Phase 3 deliverable

A single locked configuration {loss family + parameters, boundary handling, weight decay if changed}. This is the new baseline for Phase 4.

Backbone sizing (B3 / B7 vs B5) is **deferred to Phase 5**, not run inside Phase 3, because the right backbone depends jointly on Phase 2's slope (capacity utilisation) and Phase 4's channel decision (input dimensionality).

---

## 7. Phase 4 — EXTRA channel groups

*Objective: determine whether multi-modal physical context improves the final map, and if so which combination to deploy.*

> **Priority (2026-06-15): this is the primary plateau-breaker.** Given v1.0's data plateau (§5.3) and
> well-fit gap (§5.4), more RGB data / capacity / regularization are low-leverage; **adding physical
> context (EXTRA) is the most direct way to raise the ceiling**. It is blocked only on data generation
> — pushing the data team to produce the EXTRA stack is the highest-value unblock in the program
> (§11.3).

### 7.1 Group definitions

Group IDs and their band positions are fixed by `data/data.md §9` (single source of truth). **Phase 4 is currently blocked**: the v1.0 standard dataset ships **no EXTRA stack** (`EXTRA/` is empty), so this phase cannot run until the data team appends the EXTRA bands. When it does, Phase 4 ablates these groups:

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

## 8. Phase 5 — Architecture (run-now sweep)

*Objective: test if a more expressive feature extractor yields meaningful gains over UNet++ + EfficientNet-B5.*

### 8.1 Gating — skip-trigger overridden as a compute-filler (2026-06-15)

The original trigger (run only if §5.3 slope ratio ≥ 0.5 **and** the gap stayed open) would **skip**
Phase 5 on v1.0 (plateau, ratio −0.12; gap 0.05 < 0.3). **User decision: run the architecture sweep
now anyway**, to use the otherwise-idle 8×A100 — it is the largest *independent* backlog and the prime
backfill for the never-idle scheduler (§13). **Honest expectation:** decoder swaps are **incremental**
on a plateaued, well-fit model; the one candidate with step-change potential is a **foundation encoder**
(external prior knowledge directly attacks the data-limit). Record results in `docs/phase5_arch.md`.

### 8.2 Comparison set

Per `training.md §3.2` priority order, widened (2026-06-15):

| Candidate | Type | Notes |
|---|---|---|
| UNet++ + EfficientNet-B5 (Phase 3/4 winner) | reference | Reference. |
| UNet++ + EfficientNet-B3 | backbone (smaller) | Plateau-appropriate regularizer (down, not up); cheap. |
| UNet++ + EfficientNet-B7 | backbone (larger) | Low priority on a plateau (overfit risk); run only as a bound. |
| **DeepLabV3+ / FPN / PSPNet / MA-Net** (EffB5 encoder) | decoder swap | smp drop-ins (one `build_model` elif each); **DeepLabV3+ leads** (ASPP multi-scale context). Decoder-only → reuse frozen HPs (adjust batch, §8.2a). |
| SegFormer (mit_b5) | architecture | transformer; `models/segmentation.py` already supports it. Needs §8.2a re-tune. |
| **DINOv3 encoder + dense head** | foundation encoder | highest step-change potential; needs encoder integration + §8.2a. Confirm model version at impl. |
| **SAM3 encoder + dense head** | foundation encoder | SAM3 now released (2026); confirm exact model/version at impl. Needs §8.2a. |
| UNet3+ | decoder (conditional) | **not in smp** → custom impl; run only if the smp decoder sweep shows the decoder family moves the gate. |
| ~~YOLO / instance-seg (YOLOv8-seg, Mask R-CNN)~~ | detection | **Rejected (paradigm mismatch):** detection produces coarse, proposal-anchored masks unfit for pixel-accurate geodesic polygons. Reconsider only if RTS is reframed as instance detection or detection-recall becomes the bottleneck. |

### 8.2a Per-architecture-family HP adaptation (fairness requirement)

Comparing every candidate on the **frozen Phase-0 CNN HPs** would bias toward UNet++/EffB5 — candidates
could fail from mis-tuning, not from being worse. So:

- **Decoder-only swaps** (same EffB5 encoder): reuse the frozen LR/schedule; only **adjust batch size**
  to the decoder's memory footprint.
- **Encoder / paradigm changes** (SegFormer, DINOv3, SAM3): run a **quick per-family re-tune before the
  gated run** — a short LR range test (Phase-0 §3.2 style) + per-family **warmup, batch (memory-fit),
  and fine-tuning schedule**: layer-wise LR decay (LLRD), encoder-LR ≪ head-LR, optionally
  linear-probe-then-finetune; wd/betas as the family expects (ViTs often wd≈0.05). BN-comparability
  relaxes when the arch uses LayerNorm. Each candidate is gated at **its own fair HPs**, recorded in
  `docs/phase5_arch.md`, so the comparison measures architecture, not tuning.
- **Code dependency:** this requires `scripts/train.py` to support **LLRD + a configurable
  freeze/linear-probe schedule** (currently only `freeze_backbone_epochs` + a flat
  `backbone_lr_multiplier`) — engineer task, §11.3. These probe runs add to the saturation backlog (§13).

### 8.3 Winner criteria

§1.4 gate **plus** the new architecture must pass `scripts/inference_feasibility.py` (re-run with the candidate). A model that wins on PR-AUC but breaks the inference budget for the §3.2 pan-arctic tile count (`inference.md §3.2`; ~7.5M at default stride 344) is not a winner. Calibration parity (`training.md §4.6`) requires the calibrated threshold + temperature to be re-derived for any architecture change; the locked Phase 4 calibration does not transfer.

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

**Pre-approved Final-phase options (2026-06-15, decide at this step):** (a) **5-seed lock** instead of 3
(proposal #7 — one wallclock unit on 8 GPUs); (b) **ensemble deployment** (proposal #5) — average the
seed members' probabilities at inference for a classic +1–2% PR-AUC via variance reduction, aligned
with the precision-over-recall priority (inference cost ×k members — benchmark before committing).

Test-Realistic is touched **once**, at this step. Re-running Test-Realistic for any reason after this is a project-discipline failure, not a permitted iteration.

---

## 10. Hyperparameter surface — what we tune, freeze, or gate

**Full tunable surface (2026-06-15)**, so nothing important is silently omitted:

| Knob | Status | Where / why |
|---|---|---|
| Loss family + params (focal γ×α, compound λ, tversky α/β) | **Tuned (full grids)** | §6.1 first pass — run full grids, compute is free |
| Boundary handling + width | **Tuned** | §6.2 loss×boundary factorial |
| Weight decay × augmentation strength | **Tuned (exploratory)** | §6.3 grid — v1.0 trigger did not fire (§5.4) |
| Curriculum schedule + `sampling.positive_fraction` | **Tuned (new, §10.1)** | top untested precision lever |
| Backbone size (B3) | **In §8** | B3-as-regularizer (plateau ⇒ smaller, not B7) |
| Architecture / decoder + encoder | **Tuned now (§8)** | run-now sweep (decoders, foundation encoders); YOLO rejected |
| EXTRA channels | **Gated (data)** | Phase 4 — primary plateau-breaker |
| Multi-scale / input context | **Gated (post-inference)** | §6.4 — only after map review |
| Pretraining (ImageNet→MAE) | **User-gated** | §12 |
| LR / schedule / warmup / optimizer / EMA / betas / grad-clip | **Frozen (reference arch); re-opened per architecture family** | §10 table below — Phase-0-locked *within UNet++/EffB5*; encoder/paradigm changes re-tune (§8.2a) |
| Batch size | **Frozen (reference); per-arch by memory** | §3.1; transformers/foundation encoders force smaller (§8.2a) |
| Threshold / temperature / TTA | **Calibration, not training** | `training.md §12 / §10.4` |

### 10.1 Curriculum & sampling-balance sweep (new — precision lever)

The neg:pos curriculum ramp (`sampling.curriculum_schedule`, 1:1→1:20) and `sampling.positive_fraction`
(0.5) directly trade precision↔recall yet were never tuned. For a precision-first project on a
plateaued model this is the **highest-value untested knob**. Small sweep against the §6 winner: final
curriculum ratio {20, 30} × `positive_fraction` {0.5, 0.33} (≈4 cells, parallel), §1.4 gate with the
precision-@-recall guard. Lock only on a precision-positive pass.

### 10.2 What we deliberately don't tune

These knobs sit in `configs/baseline.yaml` and are technically tunable, but tuning them gives near-zero expected information per GPU-hour at this project's regime. They stay at their defaults unless evidence forces revisiting. **Note:** the LR/schedule/optimizer freezes below hold **within the reference UNet++/EffB5 architecture**; architecture/encoder changes re-open them per family (§8.2a).

| Knob | Default | Why we don't tune | What would force revisiting |
|---|---|---|---|
| `optimizer.name` | `adamw` | AdamW is the strong default for segmentation. Switching to SGD adds a momentum knob without expected gain. | Catastrophic Phase 0 instability that AdamW betas can't fix. |
| AdamW betas | `(0.9, 0.999)` (PyTorch default) | Standard. No reason to expect a 0.95 / 0.99 swap helps. | Same as above. |
| `ema.decay` | 0.999 | The 0.999 vs 0.99 vs 0.9999 spread is generally < 0.01 PR-AUC; below the §1.4 gate. | An exposure-counter pattern showing extreme tile overfitting. |
| `lr_schedule.warmup_epochs`, `backbone_warmup_epochs` | 5, 3 | Defaults are within the literature normal range. Phase 0 LR range test makes them less critical. | Phase 0 multi-seed showing high-σ runs that originate from warmup-period instability. |
| `optimizer.gradient_clip_norm` | 1.0 | Safe default. Loosening or removing risks NaN events under focal loss with extreme imbalance. | Repeated `train_nan_steps` > 0 across seeds. |
| `augmentation.*` probabilities | as in `configs/baseline.yaml` | The aug pipeline was tuned in v1; per-aug ablations are diminishing-returns search. | §5.4 generalisation-gap > 0.4 (then a coarse "all aug p × 1.5" trial, not a per-aug grid). |
| Soft-label boundary handling | not implemented | Deferred to a later iteration; `data/dataset.py` raises if requested. | `boundary_handling: ignore` clearly fails to capture annotation noise. |
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
| 2025 micro-set scope (tile count, region selection, labeling plan) | **Phase 1 (OPTIONAL — not a gate, §4.3)** | User + Heidi Rodenhizer |
| `splits.train_positive_subset_pct` config-key implementation | Phase 2 (mechanism) | Engineer |
| EXTRA-stack data generation (highest-value unblock, §7) | Phase 4 | User + data team |
| Late-fusion authorisation if §7.4 calls for it | Phase 4 §7.5 | User |
| Architecture extension to `models/segmentation.py` for SegFormer / DINOv3 / **SAM3**, **plus LLRD + configurable freeze/linear-probe schedule** for foundation-encoder fine-tuning (§8.2a) | Phase 5 (run-now) | Engineer |
| Re-running Phase 2 on full 3500 positives | Phase 3+ (if any decision flips on the 1900 result) | User |
| **Multi-scale / context-expanded training (§6.4)** — single-GSD model does **not** transfer to 2× GSD (0 vs 9 blobs; `docs/inference_validation.md`). **Trigger: pan-arctic inference done AND map review shows a large-RTS/wide-FOV gap** (inference.md §6.4 "Phase-1.5"); ~1 day pipeline work | Post-inference (map review) | User |
| Self-supervised / MAE encoder pretraining (§12.1, proposal #4) | Optional, user-gated | User |
| Hard-negative mining (§12.2) | Optional, user-gated | User |

Phases run sequentially when not externally blocked. When externally blocked, the next-runnable phase proceeds.

---

## 12. Optional high-leverage arms (user-gated)

Documented and ready, but **run only on explicit user go** (not part of the default queue):

### 12.1 Self-supervised / MAE encoder pretraining (proposal #4)
MAE-style pretraining on **unlabeled** Arctic PlanetScope quads (2025 quads already on GCS; no labels
needed), then fine-tune on v1.0. Directly attacks the diagnosed data-limit, and the only arm that can
productively use idle GPUs *before* more labels exist. Cost ≈ 200–550 GPU-h + 1–2 days coding. Pairs
with the §8 foundation-encoder candidates (an alternative source of external prior knowledge).
**Trigger:** explicit user go (deferred 2026-06-12).

### 12.2 Hard-negative mining (precision lever)
Mine false-positives from the negative pool (run the current model over negatives, collect
high-probability tiles), then oversample them in training. Directly serves the precision-over-recall
priority (`training.md §1`) — the most targeted precision lever, currently absent from the program.
**Trigger:** explicit user go, or precision becomes the binding constraint at the §1.4 gate.

---

## 13. Compute saturation (execution-scheduling SSoT)

Keep all 8 GPUs busy, within and across phases (rationale in §1.5).

1. **Within-run — local data staging.** Tiles are read per-epoch from GCS via rasterio `/vsigs/` (no
   gcsfuse mount → the `training.gcsfuse` cache is inert), causing 0%-util troughs. Stage
   `gs://rts-mapping-v2/training/v1.0/{PLANET-RGB,labels}` to local SSD (`/mnt/outputs/v1.0/data_local`,
   ~10–30 GB; tmpfs `/dev/shm` if SSD still bottlenecks) and point `data.data_root` there via a shared
   base-config override. Integrity: local tile count == metadata rows + sample checksums vs GCS.
2. **Within-wave — size every wave to 8.** Pad under-full waves (e.g. reg-grid's 4 cells) with
   independent runs so no GPU idles.
3. **Across-wave — never-idle scheduler.** Drive `scripts/run_gpu_pool.sh` (keeps NGPU in flight) from
   a priority queue. **Independent backlog backfills idle slots anytime:** the §8 architecture sweep
   (largest), the §10.1 curriculum sweep, and pre-registered seeds (43/44). **Dependency-blocked runs**
   (boundary on the loss winner) use **gated-speculative** dispatch — once the loss leader is >1σ ahead,
   launch on it; worst case redo ≤N cells if the leader flips. Density stays **1 run/GPU** (each uses
   ~39/80 GB; 2/GPU OOMs and shrinking batch breaks BN comparability). Multi-scale (§6.4) is **not** in
   the backfill pool — it is post-inference gated.
4. **Idle policy.** Do **not** stop this scarce 8×A100 for short idles — restart risks GPU stockout
   (`vm_instruction.md` zone-fallback) and the ~$30/h saving is trivial against the $70k credit. Stop
   only for genuinely long blocks (days), after backing up keepers to GCS (`/mnt/outputs` is on the
   boot disk; see `infrastructure.md`). Never manufacture runs just to light up GPUs.
