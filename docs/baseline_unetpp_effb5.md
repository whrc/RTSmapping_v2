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

---

## Experiment queue (priority order)

Ablations to run after the baseline completes. Each gets its own sub-section below; copy the template.

1. **Boundary handling** (likely biggest lever for object recall on small RTS):
    - `boundary_handling: ignore` with `boundary_ignore_width: 2`
    - `boundary_handling: soft_labels` with `soft_label_value: 0.05`
2. **Loss function**:
    - Compound (Focal + Dice)
    - Tversky (α=0.3, β=0.7) for precision-focused training
3. **Focal hyperparameters**: γ ∈ {1, 2, 3, 5}, α ∈ {0.1, 0.25, 0.5}
4. **Encoder size**: EfficientNet-B3 (capacity-down) / B7 (capacity-up)
5. **EXTRA channels**: NDVI, NIR, Red Edge, SAR — individually and stacked
6. **Architecture family**: SegFormer-B5 (training.md §3.2 priority 2)

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
