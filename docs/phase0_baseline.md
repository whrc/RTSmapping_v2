# Phase 0 — v1.0 baseline calibration

**Dataset:** v1.0 standard (`gs://rts-mapping-v2/training/v1.0/`, 22,259 tiles: 1,718 pos / 20,541 neg).
**Compute:** 8× A100-80GB (`a100-8x-train`), one run per GPU, BS 32, bf16.
**Finished:** 2026-06-14 (all runs early-stopped cleanly). Raw outputs: `/mnt/outputs/v1.0/runs/`.

Objective: establish the v1.0 baseline performance and the **noise floor** that gates every
subsequent ablation (`training/experiments.md §1.4`). Nothing from the destroyed v2-alpha dataset
carries over — this is a full re-baseline.

---

## Phase 0c — 3-seed baseline → gate G

Identical config (`configs/phase0c_seed{42,43,44}.yaml`), seeds 42/43/44. Metric is the
early-stopping / model-selection value: **`val_realistic_pr_auc_geomean`** (geomean of PR-AUC at
gate ratios **[5, 10, 20]**, 3-validation smoothed).

| Seed | best_smoothed | best_epoch | stopped |
|------|---------------|-----------|---------|
| 42 | 0.789923 | 55 | early-stop (no-improve 10) |
| 43 | 0.786319 | 35 | early-stop (no-improve 14) |
| 44 | 0.797281 | 70 | early-stop (no-improve 8) |

- **μ₀ = 0.791174**
- **σ₀ = 0.005587** (cross-seed sample std, n−1; population std 0.004562)
- **1σ band = [0.7856, 0.7968]**

### Gate (locks the whole program)
> **G = max(2σ₀, 0.01) = 0.011174** (2σ₀ = 0.01117 just clears the 0.01 floor).
>
> A later experiment is a **real winner** only if Δ `val_realistic_pr_auc_geomean` ≥ **G** over the
> baseline — i.e. it must reach **≥ 0.8023**. Smaller deltas are run-to-run noise.

The seed triplet is tight (σ₀ ≈ 0.6% of μ₀), so the baseline is reproducible and the gate is
essentially at its floor — differences below ~1 point are not interpretable.

---

## Phase 0a — RGB normalization arm-out → Lock A

Seed 42, all hyperparameters at baseline except the RGB normalization (`training/experiments.md §0a`).
Decision rule: lock A unless an arm beats it by Δ ≥ 0.01 (σ₀ does not exist at arm-out time, so the
gate falls back to the 0.01 floor).

| Arm | Normalization | best_smoothed | Δ vs A |
|-----|---------------|---------------|--------|
| **A** | **per-dataset z-score** (spec default) | **0.666615** | — |
| B | x/255 → ImageNet mean/std | 0.626122 | −0.0405 |
| C | x/255 only | 0.669727 | +0.0031 |

- C edges A by +0.0031 (< 0.01 floor) → **does not win**. B is clearly worse.
- **→ Lock Arm A (per-dataset z-score).** This is the input pipeline for the baseline and every
  later phase; `normalization_stats.json` in the v1.0 snapshot is the locked A statistics.

**Caveats (do not over-read the arm absolutes):**
- The arms ran on the *pre-0b* LR (`frozen 1e-3 / base 1e-4`) and a short schedule (`max_epochs 100`,
  `start_epoch 50`), so their absolute scores (~0.63–0.67) are **not comparable** to the seed
  baseline (~0.79, which uses the 0b LRs + full schedule). The arm-out is only valid *among A/B/C*.
- Arm C did not early-stop (hit the 100-epoch cap, still improving). Even so it didn't clear the
  floor, and A is the incumbent → the lock holds. The production baseline applies A at the 0b LRs
  (= the Phase-0c seed config).

---

## Phase 0b — LR range test

Single-epoch `lr_range_test` ramping LR 1e-7 → 1e-1 (`configs/phase0b_lr_{frozen,unfrozen}.yaml`).

- **Frozen** (decoder-only) — completed cleanly; the reliable curve. Informs `frozen_lr ≈ 3e-3`.
- **Unfrozen** (full backbone) — diverges to NaN near the top of the ramp; the recorded loss-vs-LR
  curve is **flat/uninformative** (focal loss stays tiny under extreme imbalance even at high LR), so
  it gives *no contraindication* for `base_lr = 1e-3` rather than a positive pick. (The forced final
  validation used to crash on the NaN logits — fixed: range tests now skip validation, commit
  `bd16f63`.)

The chosen LRs (`frozen_lr 3e-3`, `base_lr 1e-3`, backbone ×0.1) are baked into the Phase-0c seed
config; the healthy μ₀ = 0.791 baseline validates the choice empirically.

---

## Locked baseline (inputs to every later phase)

| Knob | Locked value | Source |
|------|-------------|--------|
| RGB normalization | **Arm A — per-dataset z-score** | Phase 0a |
| `frozen_lr` / `base_lr` | **3e-3 / 1e-3** (backbone ×0.1) | Phase 0b |
| Gate ratios | **[5, 10, 20]** | data-limited (§9) |
| Selection metric | `val_realistic_pr_auc_geomean` (smoothed, window 3) | spec |
| **μ₀ / σ₀ / G** | **0.7912 / 0.00559 / 0.0112** | Phase 0c |
| **Winner bar** | **≥ 0.8023** (μ₀ + G) | derived |

Baseline config: `configs/baseline.yaml` (= Phase-0c seed config with the locks above).

---

## Open items (deferred, see `current_working_status.md`)
- 1:200/500/1000 deployment-prevalence PR-AUC → **Test-Realistic** (gate can't honestly evaluate
  them at the ~16–20k negative ceiling).
- Next data re-stage (49 black + 564 degraded negatives out, 28 positives restored) will require a
  re-baseline of μ₀/σ₀/G on the updated dataset.
