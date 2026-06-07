# Baseline — UNet++ / EfficientNet-B5 / Focal

Living experiment record for the Phase 1 baseline model. Append sub-experiment sections as the model evolves (see `CLAUDE.md` §Documentation).

---

## Objective

Semantic segmentation of Retrogressive Thaw Slumps on 2024 PlanetScope RGB basemap at 3 m / 512 px tiles in EPSG:3857. High precision at acceptable recall; deployment on 2025 imagery after calibration (training.md §12).

---

## Configuration (locked at Step 0.5, 2026-04-23)

- Config file: [configs/baseline.yaml](../configs/baseline.yaml)
- Deployment template: [configs/deployment.yaml](../configs/deployment.yaml)
- Model: UNet++ / EfficientNet-B5, ImageNet pretrained, logits output (training.md §4.2)
- Loss: Focal (γ=2, α=0.25), no boundary handling
- Precision: BF16 on A100/H100; FP16 fallback on L4
- Curriculum: 1:1 → 1:20 over 300 epochs (training.md §7.3)
- Augmentation: geometric + color + multi-scale (training.md §9.2); `worker_init_fn` seeds each DataLoader worker independently
- Early stopping: geomean PR-AUC at 1:200/500/1000, 3-validation moving average, `start_epoch=101` (plan risk #5), `patience=8` validation events (≈ 40 epochs at `val_frequency=5`; matches `configs/baseline.yaml`)
- Checkpointing: best-by-smoothed-metric deployment (EMA), rotating last-3 resumes
- Output-bias init: `-log((1-π)/π)` with π=0.005 (per `configs/baseline.yaml:model.output_bias_prior` — set to the realistic positive-pixel prevalence so the bias init is non-zero)

---

## PR-AUC-at-ratio interpretation

PR-AUC at subsampled ratios 1:200/500/1000 is a **prevalence-conditional deployment estimate**, not a prevalence-free model-quality score. The model's predictions are identical across ratios; only the negative pool changes. Absolute values across ratios are mechanically different (AP scales with prevalence). Only **relative comparisons at the same ratio across epochs or ablations** are meaningful. See training.md §6 + §10.3.

### Gate-metric decision (2026-06-04): honest ratios, not 1:1000

The Phase 0c seed42 calibration run exposed that the **gate metric was starved of negatives**. The full negative inventory is now ~13.7k tiles (ceiling ~16k — the ARTS confirmed-negative source is very unlikely to reach 100k). With ~129 val positives and ~1.4k val negatives, the val set **honestly supports at most ~1:10–1:20**. The configured 1:200/1:1000 ratios need 25.8k / 129k negative tiles respectively — physically impossible from a 16k pool — so they fall back to bootstrap-with-replacement and oscillate (seed42 swung 0.33↔0.62 epoch-to-epoch while pixel_IoU/obj_F1 rose monotonically; the model was fine, the metric was noise).

Augmentation-inflation of negatives was considered and **rejected for evaluation**: augmented copies are highly correlated with their source (a flipped confidently-negative tile is still confidently negative), so they add no independent information — effective sample size stays ~1.4k. It would smooth the number (a "smarter bootstrap") but at ~90× validation forward-pass cost, and it does not make 1:1000 a trustworthy deployment estimate. Augmentation belongs at train time (data.md §7.2), not in the val/test metric.

**Decision:**
- The **gate metric** `val_realistic_pr_auc_geomean` is computed over the **honestly-supported ratios `[5, 10, 20]`** (config: `metrics.pr_auc_ratios`, SSoT in [training/metrics.py](../training/metrics.py)), with **pixel_IoU and obj_F1 as stability anchors**. (The **gate threshold** itself is `G = max(2σ₀, 0.01)` per `training/experiments.md §1.4` — a Δ-over-baseline, not a floor.)
- 1:200/1:1000 are **dropped from the gate** (they were bootstrap noise). Honest deployment-prevalence reporting at high imbalance is deferred to final **Test-Realistic** reporting, where the clean lever is *more real negatives* — not augmentation. If the 16k ceiling holds, the final report uses the highest honestly-supported ratio with CIs and states the limitation transparently.

*(Full Phase 0 results — μ₀, σ₀, σ-band, gate value — live in [docs/phase0_baseline.md](phase0_baseline.md). The experiment program is in [training/experiments.md](../training/experiments.md), the SSoT.)*

---

## Experiment program → see `training/experiments.md` (SSoT)

The ordered, **gated** program — Phase 1 temporal → Phase 2 data-scaling → Phase 3 loss/boundary/WD
→ Phase 4 EXTRA → Phase 5 architecture (gated) → Final — lives in
**[training/experiments.md](../training/experiments.md)**, the single source of truth. Phase 0 results
are in **[docs/phase0_baseline.md](phase0_baseline.md)**. **Do not** keep a second experiment list here:
a drifted copy in this section caused a planning error (2026-06-07) — running a gated phase early and
mis-defining the gate. The program SSoT and the gate definition (§1.4) are authoritative.

---

## Data-refresh procedure (when a new dataset version lands)

The Phase 0 gate is **provisional on the current immature snapshot** and must be re-measured whenever the
dataset changes (e.g. the v0.3 selection upgrade). Sequence:

1. `validate_training_data.py` (full, not sampled) — CRS/bands/dims/labels + the per-band degradation
   WARN; exclude flagged ambiguous/degraded tiles.
2. `create_splits.py --out-dir /outputs/new_splits` → inspect realized train/val/test pos·neg counts →
   upload `splits.yaml`/`splits_summary.json` (back up old first).
3. `compute_normalization_stats.py` over the **new** train split → upload `normalization_stats.json`.
4. Freeze the snapshot (`metadata_vX.csv`/`splits_vX.yaml`); pin configs; bump `data/version.json`.
5. Re-run the 3-seed baseline → recompute μ₀, σ₀, and the gate **G = max(2σ₀, 0.01)** via
   `report_phase0.py` → update `docs/phase0_baseline.md`.
6. Resume the experiment program (`training/experiments.md`) against the new baseline.

Deferred infra (not blocking): multi-GPU orchestrator (`scripts/run_experiments.py`) + a
concurrency-safe MLflow backend (Cloud SQL Postgres or per-GPU file-stores; the file-store is **not**
concurrency-safe); `torch.compile` + `channels_last` (needs state_dict/EMA/deploy testing).

---

## Multi-seed finalization

Per training.md §13.1, the *final* chosen configuration runs with seeds [42, 43, 44] (sequentially, per plan risk #14 — GCS-backed MLflow isn't concurrency-safe). Report mean ± std of every Test-Realistic metric in the table below.

Test-Realistic is touched **exactly once** per seed, after calibration (threshold + temperature) is frozen into `configs/deployment.yaml` and `scripts/package_model.py` has produced the per-seed deployment package. `scripts/evaluate_test.py` writes `test_metrics.json` into the package; aggregate across seeds for the final row.

---

## Final results (Test-Realistic, 1:200 / 1:500 / 1:1000)

To be filled after Phase 1 completes. Format per training.md §13.2:

| Metric | 1:200 | 1:500 | 1:1000 |
|--------|-------|-------|--------|
| IoU_RTS (pixel) | — ± — | — ± — | — ± — |
| F1_RTS (pixel) | — ± — | — ± — | — ± — |
| Object precision | — ± — | — ± — | — ± — |
| Object recall | — ± — | — ± — | — ± — |
| Object F1 | — ± — | — ± — | — ± — |
| PR-AUC | — ± — | — ± — | — ± — |

Deployment package paths:
- `gs://abruptthawmapping/models/rts-v2-seed42/` (TBD)
- `gs://abruptthawmapping/models/rts-v2-seed43/` (TBD)
- `gs://abruptthawmapping/models/rts-v2-seed44/` (TBD)

Feasibility gates (inference.md §6.4 + §7.4) report outcomes per seed (copied from `feasibility_report.md` inside each package) — or a single outcome if all three seeds agree, which is the common case.

---

## Sub-experiment template (copy when iterating)

```
### <minor version> — <short title>  (<date>)

Config diff vs baseline: <keys changed>
Motivation: <why this experiment>

Results vs baseline (val-realistic, calibrated threshold):
| Metric | baseline | this run | delta |
| ... | ... | ... | ... |

Analysis: <what we learned; ship / kill decision>
MLflow run: <run_id>
```
