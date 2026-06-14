# v2.0 → v2.1 staleness audit

> Written 2026-06-12, when the v2.1 positive set landed (1,757 quality-filtered positives,
> batch1/2/3) and v2.0 (1,819 pos + ~13.7k neg, frozen `metadata_phase0c.csv`) became
> unrecoverable. Every experimental number so far was measured on v2.0. This doc audits each
> locked decision for sensitivity to the dataset change and defines the **minimal v2.1 re-test**
> per decision — this is the GPU queue plan for when the v2.1 negatives land.
> Program SSoT: `training/experiments.md`. v2.0 results: `current_working_status.md` (Log),
> `docs/phase0_baseline.md`, `docs/phase2_data_scaling.md`, `/mnt/outputs/*/run_summary.md`.

## What changed in the data

| | v2.0 | v2.1 |
|---|---|---|
| Positives | 1,819 (incl. known label-quality issues, e.g. `vs3pfmb0808n` with 0 RTS px) | 1,757, quality-filtered (batch1: 1169, batch2: 261, batch3: 327) |
| Negatives | ~13.7k (after neg-expansion) | regenerating upstream; count/pool TBD |
| EXTRA | present (deleted) | definition being finalized |
| Splits / norm stats | frozen phase0c files (deleted) | to regenerate |

Two distinct shifts: (1) **positive label quality** (fewer, cleaner masks — affects loss-landscape
and boundary-pixel decisions most), (2) **negative pool composition** (unknown until the drop
completes — affects the realistic-ratio gate metric directly).

## Decision-by-decision audit

| # | Locked decision (v2.0 evidence) | Data-sensitivity | Verdict | Minimal v2.1 re-test |
|---|---|---|---|---|
| 1 | **Phase 0a normalization arm: A (per-dataset z-score)**, 0.5525 vs ImageNet/no-norm arms | Low — normalization choice is about input statistics, not label quality; per-dataset z-score also requires recomputing stats on v2.1 anyway | **Likely stands** | None. Recompute `normalization_stats.json` on v2.1 (required regardless); keep Arm A. |
| 2 | **Phase 0b LRs: frozen 3e-3** (+ unfrozen pick) | Low — optimization-side; LR range tests are driven by architecture/loss, weakly by label quality | **Likely stands** | None up front. Re-run LR range test only if the v2.1 baseline shows pathological training curves. |
| 3 | **Phase 0c noise floor: μ₀ = 0.5683, σ₀ = 0.0125 (medium-noise), G = 0.025** | **Definitely stale** — μ₀/σ₀ are absolute levels on v2.0 val splits, which no longer exist | **Stale** | **Re-run baseline at seeds 42/43/44** on frozen v2.1 snapshot → new μ₀/σ₀/G. Prerequisite for everything below. |
| 4 | **Phase 2 §5.3: severely under-scaled (slope ratio ≈ 4.4)**; §5.4 gap ≈ 0.43 → wd sweep triggered, Phase 5 lean-skip | Medium — the *qualitative* regime (data-limited + over-parameterized) is robust to a ~3% positive-count change, but quality filtering could shrink the gap (less label noise to overfit) | **Directionally stands; numbers stale** | Defer. Check gap on the v2.1 baseline (#3) for free; re-run scale_25/50/75 only if the new gap or a downstream gate decision contradicts the v2.0 designation. |
| 5 | **Phase 3 §6.1 loss family — in flight at cutoff**: focal baseline vs compound 1:1 = 0.5568, 1:2 = 0.5460, **2:1 = 0.6035 / 0.5760 (borderline, Δ̄ = +0.021 < G)**, tversky 0.3/0.7 = 0.3486, 2:8 = 0.3282 | **High** — loss-family ranking is exactly the penalty-landscape question; cleaner masks change focal/dice trade-offs. The 2:1 result was unresolved even on v2.0 | **Must re-test** | Re-run **compound 2:1** and **compound 1:1** vs the v2.1 focal baseline (#3), seed 42 first; add seeds only for candidates within G of winning. Drop tversky (lost by 0.2+ — quality filtering won't rescue that margin). |
| 6 | **Phase 3 §6.2 boundary handling — w1 = 0.5376 (no win); w2/w3 never ran** | **High** — boundary-ignore exists to absorb label-edge noise; quality-filtered masks plausibly flip this either way (less noise → even less useful, or cleaner edges → dilation now harmless) | **Must re-test** | Re-run **boundary_ignore_w1 vs none** on v2.1 against the locked §6.1 winner. Sweep w2/w3 only if w1 wins the new gate. |
| 7 | **Phase 3 §6.3 wd 5e-2 sweep — triggered by #4, never ran** | Conditional on #4's gap | **Pending, condition to re-check** | Run iff the v2.1 baseline gap (free from #3) still > 0.4. |
| 8 | **Phase 3 §6.x aug_strong — queued, never ran** | Conditional remedy for the same overfit signal | **Pending, same condition as #7** | Same trigger as #7. |
| 9 | **Phase 5 architecture: lean-skip** (slope 4.4 × gap 0.43; spec-tension in §8.1 cond-2 flagged in `docs/phase2_data_scaling.md`) | Inherits #4 | **Re-decide after Phase 3 locks on v2.1** | No run; recompute the two §8.1 inputs from v2.1 results when available. Resolve the §8.1 wording tension in `training/experiments.md` first. |

## Resulting v2.1 queue (once negatives land and the snapshot is frozen)

> Queue amendments 2026-06-12 (see `docs/experiments_8gpu_proposals.md`): (a) borderline
> candidates get seeds 43/44 launched in parallel immediately (proposal #1); (b) the
> conditional wd_5e2 + aug_strong probes are replaced by the 4-cell wd × aug grid
> (proposal #6) iff the v2.1 gap still exceeds 0.4; (c) the multi-scale-training decision
> (proposal #3, evidence in `docs/inference_validation.md`) must be taken BEFORE Phase-3
> re-runs since it changes the data pipeline.

Priority order, one config per GPU (`scripts/run_ablation_queue.sh`):

1. **Data prep** (no GPU): freeze snapshot → `gs://rts-mapping-v2/training/v2.1/`; `create_splits.py`; recompute norm stats; `check_data.py`.
2. `phase0c_v21_seed42/43/44` — 3 GPUs (new μ₀/σ₀/G; read off train-val gap for #7/#8 trigger).
3. `phase3_v21_compound_2to1`, `phase3_v21_compound_1to1` — 2 GPUs, in parallel with (2) once splits exist (gate applied after (2) finishes).
4. `phase3_v21_boundary_w1` — after §6.1 winner locks.
5. Conditional: `phase3_v21_wd_5e2`, `phase3_v21_aug_strong` (iff gap > 0.4); boundary w2/w3 (iff w1 wins).

All numbers measured against the **new** μ₀/σ₀/G — no comparisons across dataset versions.
