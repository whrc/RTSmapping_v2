# Master Working Document

Living doc maintained by YYang and Claude Code. Track development progress, record critical decisions, log current status and the next two steps. This is the diary and roadmap of this project. Stale decision/information that has been replaced should be deleted.

---

## Project Summary

Semantic segmentation of **Retrogressive Thaw Slumps (RTS)** in Arctic satellite imagery (60–74°N). Train on 2024 PlanetScope Quarterly Basemap (RGB, ~3m), deploy inference on 2025 imagery for a pan-arctic RTS survey map. Solo research project — flat code structure, minimal abstraction.

**Core constraints** (non-negotiable, see `CLAUDE.md`):
- CRS: EPSG:3857 everywhere
- Tile size: 512×512 px
- Labels: 0=bg, 1=RTS, 255=ignore
- Normalization: per-dataset stats, saved as `normalization_stats.json` alongside model
- Seed 42, deterministic CUDNN

**Stack**: PyTorch 2.x + `segmentation_models_pytorch` (UNet++/EfficientNet-B5 baseline), albumentations, rasterio, geopandas. MLflow tracking URI is configured in `configs/baseline.yaml:mlflow.tracking_uri` (single source). Compute: L4 VM (dev) → A100/H100 VM (prod training) via Docker.

**Imbalance strategy** (real prevalence ~0.1–0.5% positive pixels): balanced batch sampling (50/50 tile-level) + focal loss + curriculum schedule (1:1 → 1:20 pos:neg over 300 epochs). Optimize for high precision at acceptable recall.

Specs are the source of truth. Always read the relevant md before implementing (see `CLAUDE.md` §Rule 1).

---

## Status — 2026-05-29

- **Spec phase**: complete except `post-inference/post-inference.md`.
- **Phase 0** (data pipeline): complete, merged as PR #8.
- **Phase 1** (training loop): complete — Docker image `rts-train:v2` built + smoke passed (PR #9–11).
  - `rts-train:v2` at `us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2` (9.6 GB, sha256:9e298061).
- **Exp Phase 0** (baseline calibration): **in progress** (branch `phase_0`, started 2026-05-29).
  - Experiment configs created: `phase0a_arm_{a,b,c}.yaml`, `phase0b_lr_{frozen,unfrozen}.yaml`, `phase0c_seed{42,43,44}.yaml`.
  - Artifact summary rule added to `scripts/train.py` (`_print_artifact_summary`).
  - HTML report generator at `scripts/report_phase0.py` → `docs/phase0_report.html`.
  - Pending: run 0a arms on A100, pick normalization winner; run 0b LR tests; update 0c TODO fields; run 3-seed baseline.
- **A100 VM (`ml-training-vm`)**: running, Docker not yet installed (plain Ubuntu 22.04). Install Docker + NVIDIA stack before first training run.
- **Dataset v2.0**: bucket validated, normalization stats at `gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA/normalization_stats.json`.
- **Next step**: install Docker + NVIDIA on A100 VM → pull `rts-train:v2` → run `phase0a_arm_a`, `phase0a_arm_b`, `phase0a_arm_c` → update 0b/0c configs with winner.

---

## Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 0** | Data pipeline (`data/`, `utils/`, `scripts/create_splits.py`, `scripts/compute_normalization_stats.py`, `scripts/check_data_content.py`, `scripts/check_data.py`, tests, `configs/baseline.yaml`) | **complete** (PR #8 merged 2026-04-23) |
| **Phase 1** | Training loop (`models/`, `losses/`, `training/`, `scripts/train.py`, `scripts/evaluate_test.py`, `scripts/package_model.py`, `scripts/check_inference_normalization.py`, `scripts/inference_feasibility.py`, `configs/deployment.yaml`, MLflow, visualizations, Dockerfile build) | **complete** (Docker image + smoke PASSED 2026-05-28) |
| **Exp Phase 0** | Baseline calibration: normalization arm-out (0a), LR range test (0b), multi-seed baseline (0c). Configs + report script in `phase_0` branch. | **in progress** (2026-05-29) |
| Phase 2 | Inference (`scripts/inference.py`: overlap-aggregated tiling per inference.md §4, optional multi-scale / TTA per §6.4/§7.4, COG output, vectorization) | pending |
| Phase 3 | Post-inference spec finalization + implementation (`scripts/post_inference.py`) | pending |

Build order is strict (`CLAUDE.md` §Rule 2): complete and test each phase before moving on.

---

## Key Decisions Log

- **2026-04-22** — EXTRA channels (NDVI / NIR / RE / SR) made **config-driven**, not hardcoded. `configs/*.yaml` declares which bands to stack; `data/dataset.py` reads count and names from config. Spec mds updated to treat NDVI/NIR/RE/SR as *examples*, not a fixed registry. Reason: flexibility for future auxiliary channels (Sentinel-2 other bands, SAR, GEE satellite embeddings, etc.) without code changes.
- **2026-04-22** — Flat layout confirmed: code lives in `data/`, `utils/`, `scripts/` at repo root, beside its spec md. `src/__init__.py` stays empty. Per `CLAUDE.md` §Project Structure.
- **2026-04-22** — Phase 0 verification split into two tiers: Tier 1 (pytest on synthetic fixtures, must pass to call Phase 0 done) and Tier 2 (real-bucket runs, executed as v2.0 data finalizes). Reason: dataset partially ready, don't block on bucket completion.

---

## Dev Log Convention

Append entries below with date prefix `YYYY-MM-DD — <summary>`. When a decision changes a spec, also edit the relevant md in `data/`, `training/`, `inference/`, or `post-inference/`, then note the md path in the log entry.

For the coding agent: on first load, read this doc and the relevant spec md(s) for the current task. Skip the full re-read of every doc — this living doc is the launchpad.

### Log

- 2026-04-22 — Living doc seeded. Phase 0 data pipeline build started on L4 VM.
- 2026-04-23 — Phase 0 PR #8 merged to `main`; `phase1-training-loop` rebased. Phase 1 Step 0.5 methodology lock-in committed (train-inference consistency contract in training.md §4.1–§4.6; overlap math + NoData + deployment-package layout in inference.md). Phase 1 code shipped in 7 logical commits: Step 0.5 (methodology), Steps 1–2 (models + losses), Step 3 (training utilities), Step 5 (MLflow + visualizations), Steps 4 + 7a (train.py + synthetic end-to-end smoke), Steps 6a + 8 + 8.5 (deployment package + test eval + feasibility gates), and docs updates. 113 tests green (105 fast ~12 s + 8 end-to-end train-smoke ~130 s). Deferred: Step 6b Dockerfile.train (after real-data smoke), Step 7b real-data smoke on L4, the actual A100/H100 300-epoch production run, and the Step 8/8.5 gates against that run's deployment package.
- 2026-05-01 — Pre-real-data audit + fix pass. Three parallel code reviewers (ML core, scripts/tests, specs/configs) surfaced 10 Critical + ~20 Important issues; user-approved decisions:
  - Config matrix kept self-contained; placeholder configs deleted (12 files: phase3_boundary_*, phase4_extra_*, final_seed*) and will be recreated per-phase as winners lock.
  - Phase 0 §3.2 LR range test implemented end-to-end: `_make_lr_range_test_setter` in `training/scheduler.py` (logarithmic per-step ramp), `_filter_train_positive_subset` in `scripts/train.py` (deterministic seed=42 positive subsample, negatives untouched). Phase 2 §5.1 also unblocked (same `train_positive_subset_pct` mechanism).
  - `output_bias_prior` set to `0.005` in `configs/baseline.yaml` (was 0.5, a no-op for class-imbalance init).
  - `deterministic` flag stays configurable: `false` for exploration, `true` for `final_seed*` runs; train.py logs a warning if `run_name` starts with `final` and deterministic is false.
  - Multi-scale evaluation declared optional and post-1×: `scripts/evaluate_test.py` refuses multi-scale inputs; multi-scale eval moves to Phase 2 inference. Spec language in training.md §4.6 + inference.md §6.4 updated.
  - SSoT sweep across data.md / training.md / experiments.md / inference.md: removed yaml fenced blocks duplicating values in `configs/*.yaml`, replaced with one-line config-key references. MLflow URI canonicalised — single source is `configs/baseline.yaml:mlflow.tracking_uri`. Stale `gs://abruptthawmapping/mlflow/` references eliminated from CLAUDE.md, computing/docker_training.md, training/mlflow_utils.py, and current_working_status.md.
  - Critical script fixes: `check_inference_normalization.py` reads correct `rgb`/`extra` schema; `evaluate_test.py` is 1×-only; `inference_feasibility.py` 8.5b runs real-TTA forwards (was mathematically broken output-flip averaging) and `--update-config` is opt-in until expanded-tile half-scale path lands; `_resume_from` restores EMA shadow weights so post-resume validation stays on EMA (was silently falling back to live weights — direct §10.2 violation); narrow `FileNotFoundError` exception so corrupt normalization JSON surfaces.
  - Other Important fixes: Phase-2 first-epoch decoder LR off-by-one corrected (warmup now starts AT `warmup_start_lr`, ends AT `base_lr`); visualization ignore overlay rendered grey instead of transparent red; `_denormalize_rgb` accepts explicit `max_value`; DataLoader gets a seeded `generator`; dead `import pandas` removed from visualizations; `_resolve_path` extracted into `utils/config.py` for reuse; `apt-key` snippet in docker_training.md replaced with a pointer to the modern-keyring `Dockerfile.train`.
  - Tests added (9 new, 122 total green): lr_range_test endpoints + log midpoint + bounds validation + uniform per-group LR + unknown-scheduler error path; `_filter_train_positive_subset` keeps-negatives + determinism + 100%-no-op invariants. `np.random` seeded in `test_visualizations.py` randomized cases.
- 2026-05-02 — Phase 1 code-review pass + pre-smoke prep. Code-reviewer surfaced 3 Critical (C1 channel-name binding never asserted at training load; C2 `output_bias_prior: 0.5` reverted in 14 configs; C3 color/radiometric augmentations applied to EXTRA channels) plus 9 Important. Plan `docs/superpowers/plans/2026-05-02-pre-smoke-fixes.md` lands C1, C3, I1 (document `clip_percentiles` as unimplemented), I5 (resume regression test), and dissolves C2 by deleting the 15 pre-made phase configs. Per-group EXTRA normalization design intent (per-band z-score + [0.1, 99.9] clip for NDVI/NBR/SE_PCA/TC; SE_PROTO bypasses z-score) captured in `data/data.md §9`; clipping + per-channel-mode dispatch deferred to v2.1. Other Important items (I2, I4, I6, I7, I8, I9 + Minor) deferred to post-smoke housekeeping plan.
- 2026-05-28 — **Phase 1 Step 7b complete: real-data smoke passed on L4 VM.** Full Tier 2 validation sequence run against `gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA` (v2.0, 4572 tiles: 1819 pos / 2753 neg across 50 ecoregions / 37 train regions). Summary of work:
  - **Schema migration landed**: `Tile_id` → `Tile_ID` (capital D) and `TrainClass` values `"positive"`/`"negative"` (lowercase) across all production files (`data/`, `training/`, `scripts/`) and test fixtures. PR branch: `l4-test-real-data-and-dockerize`.
  - **GCS auth wiring**: All GCS-accessing scripts auto-set `GOOGLE_APPLICATION_CREDENTIALS` from ADC file on startup. `data/splits.py` and `data/normalization.py` gained `gcsfs`-based `_open_text()` helpers for `gs://` URIs.
  - **Bucket validation** (`check_data_bucket.py`): 5/6 checks passed. One data quality note: tile `vs3pfmb0808n` (positive) has 0 RTS pixels in label — non-blocking (1/1819 = 0.05%). Split design confirmed: `val_realistic` and `val_balanced` intentionally share the same 9 regions (different sampling ratios, same geography).
  - **Normalization stats**: Welford pass over 3638 train tiles. RGB mean=[48.3, 57.7, 47.2], std=[33.7, 29.2, 37.6] (typical PlanetScope summer Arctic). Uploaded to `gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA/normalization_stats.json`.
  - **DataLoader gate** (`check_data.py`): 20 batches, 32 tiles/batch, 0 errors. Negative tiles correctly return synthetic all-zero labels (no label file in GCS by design).
  - **Smoke training** (`configs/smoke.yaml`, 3 epochs, CPU, 10% positive subset): clean exit. Loss 0.035→0.016→0.013, PR-AUC geomean 0.002→0.019→0.026, obj F1 0.023→0.015→0.075. 3 resume checkpoints written.
  - **Bug fixes**:
    - `training/metrics.py` + `scripts/train.py`: PR-AUC bootstrap OOM — proportional per-tile pixel subsampling (cap 10M px total) applied *before* `np.concatenate` to prevent 25GB+ allocation when bootstrap resamples 25K+ negative tile copies for ratio 1:200 with only 354 val negatives.
    - `data/dataset.py` `_read_label`: return zeros for negative tiles instead of attempting to open a non-existent GCS label file.
- 2026-05-29 — Exp Phase 0 experiment infrastructure created (branch `phase_0`). Phase 0a normalization arm configs (`configs/phase0a_arm_{a,b,c}.yaml`) + stats files (`data/normalization_stats_arm_{b,c}.json`). Phase 0b LR range test configs (`configs/phase0b_lr_{frozen,unfrozen}.yaml`). Phase 0c multi-seed templates (`configs/phase0c_seed{42,43,44}.yaml` — TODOs for Phase 0a/0b winner values). Artifact summary rule added to `scripts/train.py:_print_artifact_summary` (runs on every training exit). HTML report script at `scripts/report_phase0.py` → `docs/phase0_report.html`. A100 VM needs Docker + NVIDIA install before training.
- 2026-05-28 — **Step 6b complete: Docker image `rts-train:v2` built and pushed to Artifact Registry.**
  - Image: `us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2` (sha256:9e298061, 9.6 GB).
  - Base: `nvcr.io/nvidia/pytorch:24.05-py3` (Python 3.10, CUDA 12.4). Dockerfile at `computing/Dockerfile.train`.
  - Docker built locally on L4 VM (Cloud Build SA lacked Artifact Registry push permission — organizational project `pdg-project-406720` IAM is not user-editable). Auth via ADC (`docker login us-west1-docker.pkg.dev` with `oauth2accesstoken`).
  - `requirements.txt` used in Docker (version ranges, compatible with Python 3.10); `requirements_frozen.txt` stays as L4/Python 3.12 dev reference. Each training run logs an in-container freeze as an MLflow artifact.
  - `computing/docker_training.md` updated: correct project (`pdg-project-406720`), Artifact Registry path (replacing old `gcr.io/abruptthawmapping`), modern-keyring gcsfuse install (`[signed-by=...]`).
  - **Ready for production run on A100.** See `computing/vm_instruction.md` for how to start `ml-training-vm`.
