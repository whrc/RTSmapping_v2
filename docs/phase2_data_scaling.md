# Phase 2 — Data scaling (results)

Per `training/experiments.md §5` + §11.2. Single-seed-42 runs on the frozen **v0.2** snapshot, baseline
config + `splits.train_positive_subset_pct`. Figures + live table: `docs/report.html` (Phase 2).

## Curve — best gate vs train positives
| Subset | ≈ positives | Best gate (smoothed) | train IoU (final) | val IoU (final) | gap |
|---|---|---|---|---|---|
| 25% | 475 | 0.5361 | 0.649 | 0.217 | **0.432** |
| 50% | 950 | 0.5372 | 0.676 | 0.257 | **0.419** |
| 75% | 1425 | 0.5587 | 0.676 | 0.241 | **0.435** |
| 100% | 1900 | 0.5607 (seed42) | — | — | — |

## §5.3 slope → regime: **Severely under-scaled**
Slope of gate vs log(n_pos): (75→100) ≈ 0.0070 vs (25→50) ≈ 0.0016 → **ratio ≈ 4.4 (> 1.0)**.
The curve is **still rising and steepening at the top** → the model is **data-limited, not saturated**.
**Implication:** acquiring more labeled positives (toward ~3500) is expected to pay off; Phase 5 stays
"in scope" by the §5.3 leg (but see the gap below).

## §5.4 generalization gap → **severe over-parameterization**
train IoU ≈ 0.65–0.68 vs val IoU ≈ 0.22–0.26 → **gap ≈ 0.43 across all subsets (≫ 0.4)**. The model fits
train far better than it generalizes — classic small-data overfitting.

## Decisions
1. **§6.3 weight-decay sweep is TRIGGERED** (gap > 0.4) — run `weight_decay 5e-2` vs `1e-2` against the
   Phase-3 loss winner; the spec's "all-aug ×1.5" remedy (§10) is also now on the table.
2. **Phase 5 (architecture) — lean SKIP.** Despite the still-rising curve, the large gap means the model is
   *over*-parameterized for the current data; B7/SegFormer would overfit more. Document the skip with the
   slope (4.4) + gap (0.43) as evidence per §8.1 once Phase 3/4 lock.
   - ⚠ **Spec tension:** §8.1 cond-2 ("gap not closed < 0.3 → run Phase 5") reads backwards vs §5.4 here.
     A large gap = over-parameterized = *skip*. Flagged for a `experiments.md §8.1` wording fix.
3. **Feasibility:** reinforces that **data is the bottleneck** — more positives + regularization, not bigger
   models. Consistent with the QC-assisted-map outlook (`docs/report.html` Findings).

## Caveats
Single-seed points (σ₀=0.0125): 25% vs 50% differ within noise; the real signal is the ~0.022 lift from
25%→100% and the steep 50→75% rise. Re-confirm on the matured (v0.3/v1.0) dataset before acting on the
"more labels" investment (per §5 "provisional; re-run if a downstream decision flips").
