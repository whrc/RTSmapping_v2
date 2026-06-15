# Master Working Document

Living doc maintained by YYang and Claude Code. Track development progress, record critical decisions, log current status and the next steps. This is the diary and roadmap of this project. Stale decision/information that has been replaced should be deleted (pre-v1.0 history lives in `docs/archive/v2-alpha/`).

---

## Project Summary

Semantic segmentation of **Retrogressive Thaw Slumps (RTS)** in Arctic satellite imagery (60–74°N). Train on 2024 PlanetScope Quarterly Basemap (RGB, ~3m), deploy inference on 2025 imagery for a pan-arctic RTS survey map. Solo research project — flat code structure, minimal abstraction.

**Core constraints** (non-negotiable, see `CLAUDE.md`):
- CRS: EPSG:3857 everywhere
- Tile size: 512×512 px
- Labels: 0=bg, 1=RTS, 255=ignore
- Normalization: per-dataset stats, saved as `normalization_stats.json` alongside model
- Seed 42, deterministic CUDNN

**Stack**: PyTorch 2.x + `segmentation_models_pytorch` (UNet++/EfficientNet-B5 baseline), albumentations, rasterio, geopandas. MLflow file store per run (`MLFLOW_TRACKING_URI` override). Compute: **`a100-8x-train`** (8× A100-80GB) via Docker `rts-train:v2`.

**Imbalance strategy** (real prevalence ~0.1–0.5% positive pixels): balanced batch sampling (50/50 tile-level) + focal loss + curriculum schedule (1:1 → 1:20 pos:neg over 300 epochs). Optimize for high precision at acceptable recall.

Specs are the source of truth. Always read the relevant md before implementing (see `CLAUDE.md` §Rule 1).

---

## Status — 2026-06-13 (runs updated 2026-06-14)

**Dataset: v1.0 is the STANDARD training dataset** (declared 2026-06-13 after a full fresh-from-rasters QC — `docs/v1.0_qc.md`). The previous v2-alpha/v0.x dataset was destroyed 2026-06-12 and is gone; **all v2-alpha numbers are invalid and archived** (`docs/archive/v2-alpha/`).

- **Snapshot**: `gs://rts-mapping-v2/training/v1.0/` (our bucket — survives external rewrites). Cleaned metadata 22,259 tiles (**1,718 pos / 20,541 neg**), `splits.yaml`, `normalization_stats.json`, `TESTING/`, QC artifacts under `qc/`. Provenance + version semantics in `data/version.json`.
- **QC verdict**: labels + structure flawless (0 unreadable, 0 invalid labels, 0 empty, **0 full-frame** — batch2 rasterization bug fixed). Known issues (next re-stage, negatives only): **49 black + 564 degraded** negatives to drop; also **restore 28 positives** wrongly dropped by the stale-QC staging (target clean pos = 1,746). List: `…/v1.0/qc/known_issues_v1.0.json`.
- **EXTRA absent** → **Phase 4 blocked** until the data team appends the stack. **TESTING/** set (3,000 tiles, spatially entangled with train) — final-lock decision still open (`docs/v1.0_rebaseline.md`).

**Experiment program: Phase-0 re-baseline COMPLETE** on v1.0 (all 6 runs early-stopped cleanly 2026-06-14; full writeup `docs/phase0_baseline.md`). **Paused here for review before the next phase.**
- `phase0c_seed{42,43,44}` (3-seed baseline) → best smoothed PR-AUC **0.7899 / 0.7863 / 0.7973** → **μ₀=0.7912, σ₀=0.00559, gate G=max(2σ₀,0.01)=0.0112**. Winner bar = **≥0.8023**.
- `phase0a_arm_{a,b,c}` (RGB norm arm-out, seed42, old-LR/short schedule — comparable only among A/B/C): A per-dataset-z **0.6666**, B x255+ImageNet 0.6261, C x255-only 0.6697. C beats A by only +0.003 (<0.01) → **Lock Arm A** (per-dataset z-score). NOTE: arm absolutes are NOT comparable to the seed baseline (arms used the pre-0b LR 1e-3/1e-4 + max_epochs 100).
- `phase0b_lr_{frozen,unfrozen}` → frozen curve clean (informs frozen_lr≈3e-3); unfrozen curve flat/uninformative (no contraindication for base_lr=1e-3). LRs (3e-3/1e-3) baked into the seed config; healthy μ₀ validates them.
- **All locks confirmed on v1.0**: Arm A norm, frozen_lr=3e-3/base_lr=1e-3, gate ratios [5,10,20], metric `val_realistic_pr_auc_geomean`.

**Repo cleanup (2026-06-13)**: tag `v2-alpha-archive` preserves everything; archived the 2 stale results docs + condensed dev-log to `docs/archive/v2-alpha/`; removed 18 dead pre-made configs (recreated on-demand per `experiments.md §11.1`); re-pointed `baseline`/`smoke` to v1.0; fixed stale EXTRA/bucket refs. `configs/` is now 8 active phase0 + `baseline`/`deployment`/`smoke`/`preview_tiles`.

**Branch state** (consolidation pending, after the runs finish — merging now would disturb the mounted repo): `v21-rebaseline` = active mainline (v1.0 work). Unmerged unique work to fold in: `inference-pipeline` (Phase-2 inference code) and `docs-eval-interim` (QC code `validate_v21_positives.py`, `base:` config inheritance, staleness audit).

**Compute**: $70k GCP credit (compute-only), **expires Sep 2026** — must be substantially spent. Stop VMs when idle.

**Next steps** (⏸ PAUSED FOR REVIEW — Phase 0 done):
1. **Review the v1.0 baseline** (`docs/phase0_baseline.md`): μ₀=0.7912, gate G=0.0112, Arm A locked. ← you are here.
2. After review → resume the ablation program (Phase 2 data-scaling, Phase 3 loss/boundary/wd) against the gate — configs recreated on-demand; runs now land in `/mnt/outputs/v1.0/runs/` (versioned convention).
3. Branch consolidation (fold inference-pipeline + docs-eval-interim into the mainline; retire feature branches); regenerate `preview_tiles.yaml` from the v1.0 val split.
4. Idle 8×A100 box — stop or repurpose between phases (credit burn).

---

## Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 0** | Data pipeline (`data/`, `scripts/create_splits.py`, `compute_normalization_stats.py`, `check_data*.py`, tests, `configs/baseline.yaml`) | **complete** |
| **Phase 1** | Training loop (`models/`, `losses/`, `training/`, `scripts/train.py`, `evaluate_test.py`, `package_model.py`, MLflow, Docker) | **complete** |
| **Exp Phase 0** | Baseline calibration on the **v1.0 standard dataset**: normalization arm-out (0a), LR range test (0b), 3-seed baseline (0c) → new gate G. | **re-baselining** (running on v1.0; v2-alpha results archived) |
| Exp Phases 2–5 | Data-scaling, loss/boundary, EXTRA channels (blocked), architecture (gated) — `training/experiments.md` | pending (after Phase 0 lock + review) |
| Phase 2 (build) | Inference pipeline (`scripts/inference.py`, tiling, merge, vectorize) | drafted on `inference-pipeline` branch |
| Phase 3 (build) | Post-inference spec + implementation | pending |

Build order is strict (`CLAUDE.md` §Rule 2).

---

## Key Decisions Log

- **2026-06-15** — **Experiment program widened + saturated for 8×A100 (`training/experiments.md` revised).** Compute is abundant ($70k credit, ablations ≈$2–4k); the binding constraint is data. v1.0 Phase-2 is a **plateau** (slope ratio −0.12 <0.5) and **not over-parameterised** (train/val IoU gap 0.05 best / 0.17 final, <0.4) → the model is *well-matched* to its data: more capacity (Phase 5 bigger backbone) and more regularization are low-leverage; **leverage pivots to representation** (Phase 4 EXTRA = primary plateau-breaker; foundation encoders). Committed: loss×boundary **factorial** (§6.2), reg grid (§6.3, exploratory — trigger didn't fire), curriculum/sampling sweep (§10.1), **run-now architecture sweep** (§8: DeepLabV3+/DINOv3/**SAM3**/UNet3+-conditional; YOLO rejected — paradigm) with **per-family HP re-tuning** (§8.2a, needs LLRD+freeze-schedule in `train.py`), bounded interaction check (§6.5). **Gated:** multi-scale (§6.4, post-inference map-review), MAE + hard-neg (§12, user-go). **Phase 1 → optional** (§4.3). **Saturation policy (§13):** local data staging + never-idle scheduler + gated-speculative; **don't stop the scarce 8×A100** (stockout risk > trivial idle cost). Gate math (μ₀=0.7912, G=0.0112) unchanged.
- **2026-06-13** — **Output dirs organized by dataset version.** `/mnt/outputs` is now `<version>/{runs,mlflow,logs,qc,staging}` + `_archive/<dataset>/` (map: `/mnt/outputs/README.md`); `run_ablation_queue.sh` writes future runs to `<VERSION>/runs/<name>` (default `v1.0`). All v2-alpha raw outputs archived to `_archive/v2-alpha/` (superseded — trained on the destroyed dataset; conclusions live in `docs/archive/v2-alpha/`). The 8 active Phase-0 runs stay flat until they finish, then move to `v1.0/runs/`.
- **2026-06-13** — **v1.0 = standard dataset.** Declared after a full fresh-from-rasters QC (`scripts/qc_full_dataset.py`; `docs/v1.0_qc.md`): labels + structure flawless, positives pristine. Stable ground for the program, minor fixes expected (49 black + 564 degraded negatives to drop, 28 positives to restore — next re-stage). **Always QC fresh from rasters** — a stale mid-drop QC csv twice produced false alarms (phantom "231 batch2-full-frame" and 28 "empty" positives that were actually fine).
- **2026-06-13** — **Batch size stays 32.** Profiled UNet++/EffB5 @ 512² bf16 on A100-80GB: BS32 = 37 GB (46%), BS64 = 74 GB (92%, too tight for a 13 h run), BS96 = OOM. Model is memory-heavy (dense skips); 32 is not an L4 leftover. A 1.5× bump to ~48 would force an LR re-calibration + gate reset for marginal gain → keep 32.
- **2026-06-13** — **No DDP for the experiment program.** Embarrassingly parallel (one run per GPU) already saturates 8 GPUs; DDP wouldn't raise experiments/hour and would invalidate the LR calibration + need a sampler rewrite. Revisit only at final-lock / Phase-5 big-backbone / <8-candidate critical paths.
- **2026-06-04** — **Gate metric = `val_realistic_pr_auc_geomean` over honest ratios `[5,10,20]`** + pixel_IoU/obj_F1 anchors. 1:200/1:1000 need 25.8k/129k val negatives (unsupportable at the ~16–20k pool) → noisy; deferred to Test-Realistic. `metrics.pr_auc_ratios` config-driven. (Being re-verified on v1.0.)
- **2026-04-22** — **EXTRA channels are config-driven**, not hardcoded: `configs/*.yaml` declares which bands to stack; `data/dataset.py` reads count/names from config.

(Pre-v1.0 decisions and the full dev-log are in `docs/archive/v2-alpha/`.)

---

## Dev Log Convention

Append entries below with date prefix `YYYY-MM-DD — <summary>`. When a decision changes a spec, also edit the relevant md and note the path. On first load, read this doc + the relevant spec md(s) — this living doc is the launchpad. Pre-2026-06-13 entries: `docs/archive/v2-alpha/devlog_v2-alpha.md`.

### Log

- **2026-06-14** — **Baseline locked for the project; re-baseline triggers defined.** μ₀=0.7912 / σ₀=0.00559 / **G=0.0112**, the normalization stats, and the **val/test split are frozen** as the project-wide yardstick — every ablation compares to them ("freeze eval, vary one thing"). Default data work = add tiles to **train only** and keep training (no recompute). Recompute "key things" only for: large train growth / new-domain batch (→re-anchor μ₀ + re-baseline), label-semantics change (→re-baseline), val/test split change (→re-score baseline), radiometric-domain change (→recompute norm + re-baseline), architecture/EXTRA-channel change (→re-baseline). The pending re-stage (keep 564 degraded negs via §4.4, restore 28 pos, drop 49 black) is small + train-only → baseline & norm stay locked. Full table + lessons in `docs/phase0_baseline.md`.
- **2026-06-14** — **Phase-0 v1.0 re-baseline complete → gate locked.** All 6 runs early-stopped cleanly. 3-seed baseline → **μ₀=0.7912, σ₀=0.00559, G=max(2σ₀,0.01)=0.0112** (winner bar ≥0.8023). RGB norm arm-out → **Lock Arm A** (per-dataset z-score; B/C don't clear the 0.01 floor). LRs 3e-3/1e-3 carried from 0b. Full writeup `docs/phase0_baseline.md`. **Correction to prior status notes:** the phase0a arms ran on the old LR (1e-3/1e-4) + max_epochs 100, so their ~0.65 scores are an internal A/B/C comparison only — NOT a 0.13 deficit vs the seed baseline (that earlier framing was an apples-to-oranges error). Outputs moved to `/mnt/outputs/v1.0/runs/` (Wave 2). **Paused for review** before the next ablation phase.
- **2026-06-13** — **`lr_range_test` no longer crashes on divergence.** The unfrozen LR-range test (`phase0b_lr_unfrozen`) ramps LR to 1e-1 over one epoch; the full backbone diverges to NaN weights, and the **forced final-epoch validation** then fed NaN logits to `average_precision_score` / figure rendering → crash. Fixed at the source in `scripts/train.py`: gate the validation block on the existing `is_range_test` flag so range tests skip validation entirely (deliverable is the per-step `lr_range_test.csv`, dumped in `finally`, not val metrics on a blown-up model). First attempt — a NaN guard in `metrics.py` — was whack-a-mole (NaN just resurfaced downstream) and was reverted. Rerun now exits 0; all 8 Phase-0 runs accounted for. The unfrozen curve is flat/uninformative (focal loss stays small even at high LR) → treat as no-contraindication for `base_lr=1e-4`; the frozen test is the reliable one.
- **2026-06-13** — **v1.0 re-baseline + standard-dataset declaration + repo cleanup.** Staged the regenerated data into `gs://rts-mapping-v2/training/v1.0/` (`scripts/stage_v1_snapshot.py`), regenerated splits + normalization stats, re-pointed the 8 phase0 configs, fixed an MLflow env-override bug + a `--privileged`/`--gpus` GPU-pinning bug (`run_ablation_queue.sh`), and launched the parallel Phase-0 calibration on 8 A100s at BS 32. Ran a full fresh QC (`scripts/qc_full_dataset.py`) → declared **v1.0 the standard dataset** (`docs/v1.0_qc.md`). Cleanup: tagged `v2-alpha-archive`, archived stale v2-alpha results + dev-log, removed 18 dead configs, fixed stale refs. Decisions: BS=32, no-DDP (above). Pending: compute the new gate when runs finish (pause for review), then branch consolidation.
