# Hypothesis-testing experiments — RTS detection, capacity vs representation vs labels

**How to use this document.** Paste it as the opening prompt for Claude Code **plan mode inside the code repo
`RTSmapping_v2`**. It is self-contained: it assumes no prior conversation. Your job is to run a small, controlled
experiment battery, then export **one results CSV**. A separate manuscript (which you do **not** have access to and
do **not** need) consumes that CSV. Do not edit the manuscript; do not assume anything about it beyond the CSV
schema in §6.

---

## 1. Context and hypothesis

We are stress-testing the central claim of a methodology paper built on this repo's RTS (retrogressive thaw slump)
mapping study. The claim, in its **scoped, honestly-testable** form, is:

> *In the scarce-label regime, at a good cheap representation, RTS detection is representation- and label-limited,
> not capacity-limited.*

The **unqualified** "not a model problem" is deliberately **not** the claim: labels are scarce by premise, so the
"large model + abundant labels" cell can never be tested. Capacity can only be ruled out **within the attainable
data budget**. State this scope explicitly in your report.

**This is an *interaction* claim.** "Capacity doesn't help" is only meaningful if measured *across* representation
(RGB vs RGB+NDVI) **and** *across* data budget (25→100%). The existing evidence is suggestive but incomplete:
adding NDVI raises the small CNN by +0.069 PR-AUC but the large ViT-L by only ~+0.006 — i.e. representation and
capacity may be *substitutable*, and the clean capacity comparison currently exists **only** at the saturated
RGB+NDVI operating point.

### Outcome → conclusion map (all outcomes are usable — there is NO "failure" branch)

- **A — capacity flat across representation and data budget:** scoped thesis holds.
- **B — capacity closes the RGB gap but ties at RGB+NDVI, and the capacity gap is flat across data budgets
  (most likely):** "representation and capacity are interchangeable routes to a label-limited ceiling;
  representation is the cost-effective one."
- **C — capacity pulls ahead as data grows, or the same-family sweep rises:** the conclusion *inverts* into a
  scaling result — "capacity becomes the lever once labels pass a threshold; below it, representation dominates."

Report whichever the data supports. **Do not tune experiments to rescue any outcome.**

---

## 2. Ground truth already in this repo (reuse it; do not reinvent)

- **Selection metric:** `val_realistic_pr_auc_geomean` — geometric-mean PR-AUC at prevalence ratios 1:5/1:10/1:20.
  Early stopping and best-checkpoint selection both use it (`training/early_stopping.py`,
  `training/checkpoint.py`), on a 3-validation smoothed value.
- **All metrics per validation step** are computed by `training/metrics.py::ValidationAccumulator.compute()`:
  `pixel_iou`, `pixel_f1`, `object_precision/recall/f1`, per-ratio `pr_auc_ratio_{r}`, and the geomean. F1/IoU use
  `reporting_threshold` (0.5 in training). Object-F1 uses object-IoU 0.3 + `min_blob_size_px`.
- **Metrics history** is logged every step by `training/mlflow_utils.py::log_metrics_step()` into the MLflow file
  store (`configs/baseline.yaml: mlflow.tracking_uri: file:///outputs/mlflow`, experiment `rts-segmentation-v2`),
  synced to GCS `gs://rts-mapping-v2/...`. **The secondary metrics for existing runs are therefore already stored
  — you can read them without re-training.**
- **Run ledger / single source of truth:** `docs/experiment_ledger.md` (+ `docs/experiment_ledger_v21.md`).
- **Corrected leakage-free split:** `docs/v1.0_region_leakage.md`; data root `/outputs/v1.0/data_local`.
- **Config pattern** (inherit a base, override a one-line delta):
  - locked v1.0 baseline recipe: `base: phase0c_seed42.yaml`
  - locked v2 deploy recipe (boundary ignore, multi_scale off): `base: base_v2_fast.yaml`
  - representation delta: `channels.extra: []` (RGB) vs `channels.extra: [{name: ndvi, band: 0}]` (RGB+NDVI); NDVI
    runs use `normalization_stats_ndvi.json`, RGB runs use `normalization_stats.json`.
  - data-budget delta: `splits.train_positive_subset_pct: {25|50|75}` (negatives are not subsetted).
  - capacity delta: `model.backbone: efficientnet-b{0|3|5|7}`.
- **Reference runs and configs** (existing): `phase4_extra_rgb_baseline` (EffB5 RGB, 0.830, **single-seed**),
  `phase4_extra_ndvi(+seed43/44)` (EffB5+NDVI, mean 0.8985), `aug_trivialaugment_deploy(+seed43/44)` (EffB5+NDVI
  deploy recipe, mean **0.9218** = the 100% data point), `effb3_deploy` (0.905), `fm_dinov3sat_l_ndvi_locked
  (+seed43/44)` (ViT-L+NDVI, mean **0.9191**), `fm_dinov3sat_l_rgb(+seed43/44)` (ViT-L RGB, mean 0.913 —
  **confounded**, see §5), `scale_ndvi_25/50/75` (data-scaling on corrected split).

**First, in plan mode: read `docs/experiment_ledger.md` and the configs named above, and confirm every fact in this
section against the repo before running anything.** If any diverge, report the discrepancy rather than proceeding.

---

## 3. The experiment battery

Hold everything fixed except the one factor each experiment varies. Use each architecture's **own** locked recipe
(do not sandbag transformers). 3 seeds (42/43/44) for every new cell. Corrected split only.

### C1 — representation × capacity (2×2), fixed data (100%)

Complete the 2×2 so the capacity effect is measured at *both* representation levels.

| | RGB | RGB+NDVI |
|---|---|---|
| EffB5 (small) | **RUN: EffB5-RGB, 3 seeds** (existing is single-seed + confounded norm — re-run clean) | exists: `aug_trivialaugment_deploy*` 0.9218 |
| ViT-L (large) | exists: `fm_dinov3sat_l_rgb*` 0.913 (**verify comparability**, §5) | exists: `fm_dinov3sat_l_ndvi_locked*` 0.9191 |

*Isolates:* capacity effect at RGB and at RGB+NDVI; the representation×capacity interaction.
*Falsifies the dichotomy if:* ViT-L-RGB ≈ EffB5-RGB+NDVI (capacity substitutes for representation).
*New runs:* EffB5-RGB × 3 seeds (base `phase0c_seed42.yaml`, `channels.extra: []`; match the recipe of the
existing EffB5+NDVI comparator so only channels differ).

### C2 — same-family capacity sweep at fixed input (RGB+NDVI, deploy recipe)

The clean capacity axis — no architecture/pretraining confound (unlike the CNN→ViT jump).

- Backbones: `efficientnet-b0`, **b3 (exists)**, **b5 (exists)**, `efficientnet-b7`.
- Recipe: identical to `effb3_deploy.yaml` (base `base_v2_fast.yaml`, NDVI, boundary ignore w2, multi_scale off),
  varying **only** `model.backbone`. 3 seeds each for B0 and B7.

*Isolates:* capacity alone. *Confirms if:* PR-AUC (and mask metrics) flat B0→B7. *Falsifies if:* monotone rise.
*New runs:* B0 ×3, B7 ×3.

### C3 — capacity × data-budget interaction (most decision-relevant)

Does capacity pay off with more data? Overlay a large-model scaling curve on the existing small-model one.

- Existing (EffB5+NDVI): `scale_ndvi_25/50/75` + 100% = `aug_trivialaugment_deploy*` → 0.79/0.86/0.87/0.92.
- **RUN (ViT-L+NDVI):** the same 25/50/100% budgets (`splits.train_positive_subset_pct`), ViT-L locked recipe,
  3 seeds each. (75% optional if 25/50/100 already show the trend.)

*Confirms if:* the two curves are parallel (capacity gap flat at every attainable budget). *Falsifies if:* the
ViT-L curve pulls ahead as data grows. *New runs:* ViT-L+NDVI at 25/50/100% × 3 seeds ≈ 9.

### C4 — seed-completion + statistics (RSE requires "statistically sound validation")

- Seed-complete the load-bearing single-seed points so every headline delta is 3-seed vs 3-seed:
  **EffB5-RGB** (covered by C1), **data-scaling 50%** (`scale_ndvi_50`), **data-scaling 75%** (`scale_ndvi_75`).
- Compute **confidence intervals / effect sizes** (e.g. bootstrap or seed-spread CIs) for: the NDVI effect, the
  data-volume effect, and each capacity comparison. These CIs are a **required** part of the export.

### Scope boundaries — state in the report, do NOT run

- Capacity at >100% labels: unattainable (labels scarce by premise) — name it as the scope limit of the claim.
- No label-noise / inter-annotator study (excluded by the authors).
- No decoder-capacity redo.

---

## 4. Parity gates (run BEFORE trusting any new number)

Reproduce these **known** values within tolerance from the MLflow store / a re-run smoke check; if any fails,
**STOP and report** — it signals recipe or environment drift, not a result:

- EffB5+NDVI (`aug_trivialaugment_deploy*` 3-seed) → **0.9218**
- ViT-L+NDVI (`fm_dinov3sat_l_ndvi_locked*` 3-seed) → **0.9191**
- EffB3+NDVI (`effb3_deploy`) → **0.905**
- EffB5+NDVI single (`phase4_extra_ndvi`) → **0.8879** (3-seed mean 0.8985)

Also re-extract each existing run's `pixel_iou`/`pixel_f1`/`object_f1` at its best-checkpoint step and sanity-check
against `docs/experiment_ledger.md` where the ledger quotes them (e.g. EffB5+NDVI 0.612/0.438, ViT-L+NDVI
0.612/0.437).

---

## 5. Confound to resolve (do not repeat the mistake)

The existing `phase4_extra_rgb_baseline` (EffB5-RGB, 0.830) vs `fm_dinov3sat_l_rgb` (ViT-L-RGB, 0.913) pairing is
**confounded** on ≥4 axes — different normalisation (z-score vs satellite-native; the `fm_dinov3sat_l_rgb.yaml`
header flags this), single-seed vs 3-seed, different optimiser/LR/batch/freeze, different stop schedule — and
ViT-L-RGB ≈ EffB5+NDVI, not EffB5-RGB. **Do not report this pairing as a capacity effect.** For C1, produce a
clean EffB5-RGB (3-seed) that differs from its EffB5+NDVI comparator **only** in channels, and confirm the ViT-L
RGB vs +NDVI pair share one recipe/norm regime (they should — same family of configs). If a fully matched
cross-architecture RGB comparison is not achievable, report the capacity-at-RGB effect as *directional only* and
say so.

---

## 6. Output contract — one CSV the manuscript consumes

Emit `outputs/metric_robustness.csv` (and copy to the repo's results/ledger area). **Schema:**

`run_name, family, backbone, representation, train_positive_subset_pct, seed, split, best_epoch, threshold,
pr_auc_geomean, pixel_iou, pixel_f1, object_precision, object_recall, object_f1, git_sha, config_path`

Rules:
- One row per (run, seed); plus one **aggregate** row per condition with 3-seed **mean and CI/spread** for every
  metric (mark it, e.g. `seed=mean`).
- `threshold` = the fixed reporting threshold the F1/IoU are read at (state it).
- Metrics read at the **best-checkpoint step** (peak smoothed `val_realistic_pr_auc_geomean`), matching how PR-AUC
  is reported.
- Include **existing** runs in scope as well as the new ones, so the CSV is a complete, standalone artefact.

### Coverage the CSV must support (specified by content, not by any manuscript figure names)

A separate downstream step must be able to build all of these **without re-running anything**:
1. **Capacity axis at fixed input (RGB+NDVI):** EffNet B0/B3/B5/B7 (C2) + every foundation-model point, on all four
   metrics — to show capacity flat-or-not.
2. **Representation × capacity 2×2:** {EffB5, ViT-L} × {RGB, RGB+NDVI} (C1) — NDVI effect at each capacity + the
   interaction, on all four metrics.
3. **Data × capacity interaction:** EffB5+NDVI and ViT-L+NDVI at 25/50/100% (C3) — two scaling curves, on all four
   metrics.
4. **Lever deltas with CIs:** ΔPR-AUC / Δpixel-IoU / Δpixel-F1 / Δobject-F1 for +NDVI, for data 25→100%, and for
   capacity at fixed RGB+NDVI.

---

## 7. Reporting

Produce a short summary that: (a) names the outcome (A/B/C from §1) the data supports, with the CIs; (b) states the
capacity gap at each representation level and each data budget; (c) flags any parity-gate or confound issue; (d)
points to `metric_robustness.csv`. Do not pick the paper's title — just report which conclusion the data supports.

---

## 8. Guardrails (non-negotiable)

- Do **not** modify or overwrite deployed configs or checkpoints; add new configs under clear names
  (`c1_effb5_rgb_seed42.yaml`, `c2_effb0_ndvi_seed42.yaml`, `c3_vitl_ndvi_scale25_seed42.yaml`, …).
- Do **not** change the corrected split, the selection metric, or the reporting threshold.
- Do **not** invent or retune recipes toward a target; use each architecture's existing locked recipe.
- **Plan first:** present the full run matrix (every new config + its base + one-line delta + seeds) and get user
  approval before executing. Do not proceed past a failed parity gate.
- Report negative/failing outcomes plainly; never adjust an experiment to make the hypothesis hold.
