> ⚠️ **SUPERSEDED — v2.0 preliminary results (archived 2026-06-13).** These numbers
> (μ₀=0.5683, σ₀=0.0125, gate G=0.025) were measured on the v2.0/v0.2 dataset that was
> **destroyed by an external bucket rewrite on 2026-06-12** and is unrecoverable. They do
> **not** apply to the v1.0 standard dataset. Kept only as a historical record of the
> preliminary calibration. The live re-baseline is in `docs/v1.0_rebaseline.md`; recover the
> full pre-cleanup state from git tag `v2.0-preliminary-archive`.

# Phase 0 — Baseline calibration (results)

Per `training/experiments.md §3` + §11.2. Records the locked baseline, the noise floor, and the gate
that every later phase compares against. **Dataset: v0.2** (frozen `metadata_phase0c.csv` /
`splits_phase0c.yaml`, 15,528 tiles — immature; gate is provisional until v1.0 is anchored).

## Locked configuration
| Item | Value | Source |
|---|---|---|
| RGB normalization | **Arm A** (per-dataset z-score) | Phase 0a: A=0.5525 vs ImageNet B=0.4752 (Δ=0.077 ≫ 0.01) |
| Batch size | 32 (bf16) | §3.1 — fits A100; no OOM over full runs |
| `frozen_lr` | 3.0e-3 | §3.2 LR range test (steepest-descent midpoint) |
| `base_lr` | 1.0e-3 (backbone ×0.1 → 1e-4) | §3.2 unfrozen LR range test |
| Architecture | UNet++ / EfficientNet-B5, ImageNet-pretrained | baseline |

## Gate metric — ratios
The gate metric `val_realistic_pr_auc_geomean` is the geomean of PR-AUC at **honest ratios `[5, 10, 20]`**
(`metrics.pr_auc_ratios`). The originally-specified 1:200/1:1000 are **unsupportable** at the negative
ceiling (~16k tiles need 25.8k/129k val negatives → bootstrap noise) and are **deferred to Test-Realistic**.
See `[[negative-pool-ceiling-and-gate-metric]]` and `docs/baseline_unetpp_effb5.md` "Gate-metric decision".

## Noise floor (3-seed baseline, §3.3)
| Seed | best `val_realistic_pr_auc_geomean` (smoothed) | best epoch |
|---|---|---|
| 42 | 0.5607 | 65 |
| 43 | 0.5828 | 55 |
| 44 | 0.5615 | 35 |

- **μ₀ = 0.5683** (baseline reference for all later Δ comparisons)
- **σ₀ = 0.0125**

## σ-band designation (§3.4) → **Medium-noise** (0.005 ≤ σ₀ < 0.015)
- **Gate `G = max(2σ₀, 0.01) = 0.025`** (`experiments.md §1.4`). A candidate **wins** iff
  Δ(`val_realistic_pr_auc_geomean`) vs μ₀ ≥ G **and** precision @ recall=0.5 does not regress. G is a
  **Δ-over-baseline threshold, not a performance floor**.
- **Comparison protocol:** single-seed (42) for first-pass ranking; top 1–2 candidates within 1σ re-run
  at seed 43 to break ties.
- **Final-phase seed count:** 3 (42, 43, 44).

## Notes
- Report (curves + cards): `docs/report.html` (regenerated from MLflow; not committed).
- Mid-phase data/pipeline fixes (corrupt tiles, transient-read retry, preview-UID, MLflow dedup) are in
  `current_working_status.md` Key Decisions (2026-06-04/05) and git history.
