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

## Status — 2026-06-12

- **Spec phase**: complete except `post-inference/post-inference.md`.
- **Phase 0 / Phase 1**: complete. Docker image `rts-train:v2` at `us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2`.
- **Exp Phase 0** (baseline calibration, branch `phase_0`): **COMPLETE — 0a + 0b + 0c all done.**
  - **Phase 0a winner: Arm A** — per-dataset z-score (0.5525 vs ImageNet 0.4752, Δ=0.077). Locked.
  - **Phase 0b** (LR range tests): `frozen_lr = 3e-3`, `base_lr = 1e-3` (backbone ×0.1 → 1e-4). Locked.
  - **Gate metric**: `val_realistic_pr_auc_geomean` over **honest ratios `[5,10,20]`** + pixel_IoU/obj_F1 anchors (negatives cap ~16k → 1:200/1:1000 unsupportable; deferred to Test-Realistic). `metrics.pr_auc_ratios` config-driven ([training/metrics.py](training/metrics.py)).
  - **Phase 0c 3-seed baseline (frozen 15,528-tile snapshot, `metadata_phase0c.csv` / `splits_phase0c.yaml`):**
    - seed42 = **0.5607** (e65) · seed43 = **0.5828** (e55) · seed44 = **0.5615** (e35), ~13.7h each.
    - **μ₀ = 0.5683 (baseline ref), σ₀ = 0.0125 → medium-noise band.**
    - **Gate G = max(2σ₀, 0.01) = 0.025** (`training/experiments.md §1.4`, the SSoT). A candidate **wins** iff Δ(`val_realistic_pr_auc_geomean`) vs μ₀ ≥ G **and** precision @ recall=0.5 does not regress. G is a Δ-over-baseline threshold, **not** a performance floor. (Earlier "G=μ₀−2σ₀=0.5433 floor" was an error — corrected 2026-06-07.)
    - σ-band protocol (§3.4 medium): single-seed first-pass ranking; top ties (within 1σ) re-run at seed 43; final lock at 3 seeds. Recorded in `docs/phase0_baseline.md`.
    - Report: `docs/report.html` (per-seed train/val-loss + gate/IoU/F1 curves; μ₀/σ₀/G card).
  - **Mid-phase fixes** (see Key Decisions 2026-06-04/05): 9 corrupt tiles dropped; transient-GCS-read retry in `data/dataset.py`; preview tiles pinned by UID (`configs/preview_tiles.yaml`); MLflow dedupe + 2 stale seed42 runs deleted.
- **Known data-quality debt (v2.1):** ~2.2% degraded tiles (209 missing BLUE band) — `/mnt/outputs/degraded_tiles.json`; 14 valid negatives unregistered in metadata. Left as-is (frozen baseline).
- **Git**: `phase_0` → PR #14 (open, → main). Active branch **`phase1-prep`** pushed to origin (8 commits ahead of phase_0: v0.2, per-band validator, SegFormer, SSoT repair, gate fix, train_iou, re-queue).
- **Compute**: $70k GCP credit (compute-only), **expires September 2026** — must be substantially spent. Plan: thorough ablation program on a multi-GPU node (8× A100 now / 8× H100 when quota lands) run in *short bursts* (~$5–15k), bulk of budget → pan-arctic inference + EXTRA generation + multi-year/ensemble (~$40–55k). Stop VMs when idle.
- **Status**: Phase 0 PR #14 open (phase_0→main). Working on branch `phase1-prep`. **`training/experiments.md` is the SSoT for the experiment program** — all other docs defer to it (SSoT-drift repair, 2026-06-07).
- **Next two steps**:
  1. ✅ **SSoT repair done (2026-06-07)**: gate → `G=max(2σ₀,0.01)` everywhere; `training/experiments.md` is the program SSoT (§2.1/§2.3 fixed); baseline doc deduped → pointer; `docs/phase0_baseline.md` created; `train_iou` logged. 135 tests pass; committed + pushed.
  2. ✅ **Migration to `a100-8x-train` complete** (8× A100-80GB, us-central1-a; landed 03:22 UTC after 336 spin-retry attempts) — outputs (28.6 GB incl. MLflow), repo (`phase1-prep`), docker image, ADC all moved; environment validated (122 fast tests + cuda test green in-container). H100 spin-retry killed; `ml-training-vm` STOPPED (not deleted). Runbook/checklist: `computing/migrate_vm.md`. **All future sessions run on `a100-8x-train`.**
  3. 🚨 **BLOCKER — v2.0 training data deleted from `gs://abrupt_thaw/.../TRAINING_DATA/`** (in-place rewrite ~04:30–05:45 UTC 2026-06-12 by the data-production side; bucket versioning Suspended → unrecoverable from GCS). New `metadata.csv` = 1,757 tiles, `Version=batch1` (in-progress drop, not a full dataset). The 04:10 UTC run crashes were this rewrite, not transient 404s. **Entire experiment backlog blocked** (w2/w3, wd_5e2, aug_strong reruns + seed44 tiebreak — configs point at the deleted frozen snapshot). Next: contact data team (v2.0 restore? new-drop ETA?), decide re-baselining; stage future frozen snapshots in `gs://rts-mapping-v2/training/<version>/` so this cannot recur. §5.3 slope fit + §8.1 gate eval need only existing outputs → still doable now. 8× A100 box idles ~$30/hr meanwhile — stopping it risks losing stockout-won capacity (user call).
  - **Phase 3 §6.1 compound 2:1 seed pair — borderline:** seed42 = 0.6035, seed43 = **0.5760** (best e60, early-stop e105) → 2-seed mean 0.5897, Δ=+0.021 vs μ₀ — just under G=0.025 → **seed-44 tiebreak queued**.
  - **Phase 3 §6.2 boundary:** ignore w1 = **0.5376** (no win) · w2/w3 pending rerun. `none` keeps the operational tie-break so far.
  - **Phase 2 data-scaling (§5) — complete:** 25% (~475 pos) = **0.5361** · 50% (~950) = **0.5372** · 75% (~1425) = **0.5587** · 100% (~1900) = 0.5607 (seed42). Curve **still rising** (~0.536 → ~0.561 over 4× data) → **data-limited, not saturated** — more positives likely help. §5.3 slope fit + §8.1 Phase-5 gate + §6.3 WD trigger (`train_iou` gap) still to be computed.
  - **Phase 3 §6.1 loss family — results so far:** focal-only baseline = 0.5607; compound 1:1 = **0.5568** (≈baseline); compound 1:2 = **0.5460** (no win); compound 2:1 = **0.6035** (Δ=+0.035 vs μ₀ ≥ G=0.025 → **candidate win**, seed-43 confirmation running); tversky 0.3/0.7 = **0.3486**, tversky 0.2/0.8 = **0.3282** (tversky family out).

---

## Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 0** | Data pipeline (`data/`, `utils/`, `scripts/create_splits.py`, `scripts/compute_normalization_stats.py`, `scripts/check_data_content.py`, `scripts/check_data.py`, tests, `configs/baseline.yaml`) | **complete** (PR #8 merged 2026-04-23) |
| **Phase 1** | Training loop (`models/`, `losses/`, `training/`, `scripts/train.py`, `scripts/evaluate_test.py`, `scripts/package_model.py`, `scripts/check_inference_normalization.py`, `scripts/inference_feasibility.py`, `configs/deployment.yaml`, MLflow, visualizations, Dockerfile build) | **complete** (Docker image + smoke PASSED 2026-05-28) |
| **Exp Phase 0** | Baseline calibration: normalization arm-out (0a), LR range test (0b), multi-seed baseline (0c) → gate G=0.025. | **complete** (2026-06-06). Experiment program (`training/experiments.md` Phase 2 data-scaling + Phase 3 loss) now running on the A100. |
| Phase 2 | Inference (`scripts/inference.py`: overlap-aggregated tiling per inference.md §4, optional multi-scale / TTA per §6.4/§7.4, COG output, vectorization) | pending |
| Phase 3 | Post-inference spec finalization + implementation (`scripts/post_inference.py`) | pending |

Build order is strict (`CLAUDE.md` §Rule 2): complete and test each phase before moving on.

---

## Key Decisions Log

- **2026-06-04** — **Phase 0c gate metric = honest ratios `[5,10,20]` + IoU/F1 anchors; 1:200/1:1000 dropped from the gate.** The negative pool ceiling is ~16k (ARTS confirmed-negative source; 100k very unlikely), so val honestly supports at most ~1:10–20. The configured 1:200/1:1000 need 25.8k/129k val negatives → structural bootstrap → noisy gate (seed42 oscillated 0.33↔0.62). **Augmentation-inflation of eval negatives was rejected**: correlated copies add no independent information (effective N stays ~1.4k), cost ~90× validation compute, and don't make 1:1000 trustworthy; augmentation belongs at train time (data.md §7.2). Honest high-imbalance reporting deferred to final Test-Realistic, where the clean lever is *more real negatives*. Reason: a gate only needs to be stable + discriminative for model selection; the true 1:1000 deployment number is a test-time concern, not a gate concern. Impl: `metrics.pr_auc_ratios` config-driven SSoT.
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

- 2026-05-28 — **Step 6b complete: Docker image `rts-train:v2` built and pushed to Artifact Registry.**
  - Image: `us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2` (sha256:9e298061, 9.6 GB).
  - Base: `nvcr.io/nvidia/pytorch:24.05-py3` (Python 3.10, CUDA 12.4). Dockerfile at `computing/Dockerfile.train`.
  - Docker built locally on L4 VM (Cloud Build SA lacked Artifact Registry push permission — organizational project `pdg-project-406720` IAM is not user-editable). Auth via ADC (`docker login us-west1-docker.pkg.dev` with `oauth2accesstoken`).
  - `requirements.txt` used in Docker (version ranges, compatible with Python 3.10); `requirements_frozen.txt` stays as L4/Python 3.12 dev reference. Each training run logs an in-container freeze as an MLflow artifact.
  - `computing/docker_training.md` updated: correct project (`pdg-project-406720`), Artifact Registry path (replacing old `gcr.io/abruptthawmapping`), modern-keyring gcsfuse install (`[signed-by=...]`).
  - **Ready for production run on A100.** See `computing/vm_instruction.md` for how to start `ml-training-vm`.

- 2026-05-29 — Exp Phase 0 experiment infrastructure created (branch `phase_0`). Phase 0a normalization arm configs (`configs/phase0a_arm_{a,b,c}.yaml`) + stats files (`data/normalization_stats_arm_{b,c}.json`). Phase 0b LR range test configs (`configs/phase0b_lr_{frozen,unfrozen}.yaml`). Phase 0c multi-seed templates (`configs/phase0c_seed{42,43,44}.yaml`). Artifact summary rule in `scripts/train.py`. HTML report at `scripts/report_phase0.py`. Created `data/version.json` (dataset v2.0 anchor — 4572 tiles, 1819 pos / 2753 neg, 50 ecoregions, per `data/data.md §4`). Docker 29.5.2 + NVIDIA driver 595 + container toolkit installed on A100 VM; `rts-train:v2` pulled and GPU verified; cv2/albumentations DictValue conflict patched in `Dockerfile.train` + `requirements.txt`. Docker smoke tests (phase0a, phase0b) passed on A100 GPU.

- 2026-05-30 — **Phase 0a Arm A complete.** Best smoothed PR-AUC geomean = 0.5525 at epoch 60; early-stopped at epoch 90 (patience=6 validations). GCS ADC auth configured on A100 (user ADC mounted into Docker). MLflow tracking at `/mnt/outputs/mlflow`. Phase 0a Arm B (ImageNet norm) launched.

- 2026-06-06 — **Computing infrastructure SSoT created.** New `computing/infrastructure.md` consolidates all infra facts: the two GCP projects (PDG `pdg-project-406720` = VMs + $70k compute-only credit expiring Sep 2026 + Artifact Registry + bucket `rts-mapping-v2`; non-PDG `abruptthawmapping` proj# 801926669176 = bucket `abrupt_thaw` with current training data), VM inventory (existing L4/A100 + planned multi-GPU node, spec TBD; inference-compute decision open), regions, on-VM storage tiers, a **data storage map** (current → recommended location for training images/metadata, training outputs, artifacts, deployment packages, inference I/O, scratch), MLflow infra, and org-project permission gotchas. **Clarified a long-standing naming error**: `abruptthawmapping` is a *project ID*, not a bucket — `gs://abruptthawmapping/` 404s; the real data bucket is `gs://abrupt_thaw/`. Fixed the bucket references in `CLAUDE.md`, `computing/vm_instruction.md`, and `computing/docker_training.md` (+ pointers to the new doc). Reworked `README.md` into a full repo doc-map + a "source of truth" table. Remaining stale `gs://abruptthawmapping/...` example/default paths in `inference/inference.md`, `training/experiments.md`, `configs/baseline.yaml`, and `scripts/{package_model,evaluate_test,inference_feasibility,check_data_content}.py` are flagged in `infrastructure.md §10` as a separate cleanup (not changed here).

- 2026-06-12 (later) — **Migration executed + v2.0 dataset loss discovered.** Full migration `ml-training-vm` → `a100-8x-train` per `computing/migrate_vm.md`: outputs archived to `gs://rts-mapping-v2/runs/ml-training-vm-outputs/` (28.6 GB, size-verified) and tar-over-ssh'd to the new box; docker + nvidia-container-toolkit installed (8 GPUs verified in-container); `rts-train:v2` pulled; repo cloned; tests green. Launched the 5-run backlog as per-GPU queues → **all crashed on missing `metadata_phase0c.csv`** → discovered the **entire v2.0 TRAINING_DATA prefix was rewritten in place this morning** (new 1,757-tile `metadata.csv`, `Version=batch1`; frozen snapshot + EXTRA/ + norm stats deleted; versioning Suspended → unrecoverable). Backlog blocked pending data-team contact; re-baselining likely if v2.0 can't be restored. H100 spin-retry killed; old VM stopped. Lesson recorded in migrate_vm.md: frozen training snapshots must live in our own bucket (`gs://rts-mapping-v2/training/<version>/`).

- 2026-06-12 — **8× A100-80GB VM landed + overnight results + migration started.** `a100-8x-train` (a2-ultragpu-8g, us-central1-a) created 03:22 UTC by the spin-retry (336 attempts) under the newly approved 8-GPU quota; H100 hunt to be stopped. Overnight: seed43 compound 2:1 finished at **0.5760** (best e60) → 2-seed mean Δ=+0.021 < G → seed-44 tiebreak (`configs/phase3_loss_compound_2to1_seed44.yaml`); boundary w1 = 0.5376 (no win); **w2/w3, wd_5e2, aug_strong all crashed ~60 s in** on a transient GCS 404 burst (04:10 UTC, tile `ym82530p0p05` — file fine now) and wrote `best_epoch=-1` summaries that the queue then treated as complete. Queue script hardened: container `set -o pipefail` (crashes no longer masked by `| tee` exit 0), crash-artifact summaries rerun, `GPU=N` env for per-GPU parallel queues on the 8-GPU node. Migration runbook + checklist: `computing/migrate_vm.md`.

- 2026-06-11 — **VM interruption recovery + EMA-resume bugfix.** The A100 `ml-training-vm` was interrupted 2026-06-10 ~16:45 UTC (rebooted 2026-06-11 10:51), killing the running `phase3_loss_compound_2to1_seed43` (died at epoch 23) and the H100 spin-retry script. Recovery: (1) seed43 resumed from `resume_latest-0020.pth` in a detached container; (2) remaining queue re-chained behind it (`phase3_boundary_ignore_w1/w2/w3` → `phase3_wd_5e2` → `phase3_aug_strong`); (3) H100 spin-retry (`~/create_h100_vm.sh`, 2× H100 a3-highgpu-2g, us-west1-a/b) restarted. **Bug found by the resume**: `_resume_from` in `scripts/train.py` restored the EMA shadow from the CPU-loaded checkpoint without moving it to the model's device → `EMA.update` crashed with cpu/cuda mismatch. Fixed (shadow `.to(model device)`), cuda-gated regression test `test_resume_ema_shadow_on_model_device` added (passes in `rts-train:v2`). Results recorded meanwhile: scale_75 = 0.5587 (Phase 2 complete); compound 1:2 = 0.5460, **compound 2:1 = 0.6035 (candidate win)**, tversky 0.2/0.8 = 0.3282.

- 2026-06-07 — **Exp Phase 0 complete + SSoT-drift repair + corrected re-queue.** Phase 0c 3-seed baseline done: seeds 42/43/44 = 0.5607/0.5828/0.5615 → **μ₀=0.5683, σ₀=0.0125 (medium-noise), gate G=max(2σ₀,0.01)=0.025** (per `training/experiments.md §1.4`). Recorded in new `docs/phase0_baseline.md`. **Caught a planning error**: an ablation queue was built from the short list in `docs/baseline_unetpp_effb5.md` instead of the authoritative `training/experiments.md`, which led to a mis-ordered queue, a **gated phase (Phase 5) run early** (encoder/SegFormer), and a **mis-defined gate** (μ₀−2σ₀ floor instead of the §1.4 Δ-threshold). Root cause = SSoT drift (decisions recorded only in derived docs). **Repair**: `experiments.md` is now THE program SSoT (§2.1 file-store URI, §2.3 real metric names + [5,10,20] gate ratios + `train_iou`); gate corrected to `G=max(2σ₀,0.01)` in report/status/baseline doc; baseline doc deduped → pointer; `docs/*.md` un-gitignored. Added `train_iou` logging (train.py) for the §5.4 data-scaling gap + §8.1 Phase-5 gate (135 tests pass). **Corrected A100 queue (plan order):** Phase 2 data-scaling (`phase2_scale_25/50/75`) → complete Phase 3 §6.1 loss family (`phase3_loss_compound_1to2/2to1`, `tversky_2to8`); gated Phase-5 configs moved to `phase5_*` (kept, not queued). SegFormer support added to `build_model` (smp.Segformer/mit_b5) as the §11.3 prerequisite. **Loss-family results so far:** compound 1:1 = 0.5568 (≈baseline), tversky 0.3/0.7 = 0.3486 (clear loss) → focal-only winning. Committed on `phase1-prep` (pushed).
