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

## Status — 2026-06-15

**Dataset: v1.0 is the STANDARD training dataset** (declared 2026-06-13 after a full fresh-from-rasters QC — `docs/v1.0_qc.md`). The previous v2-alpha/v0.x dataset was destroyed 2026-06-12 and is gone; **all v2-alpha numbers are invalid and archived** (`docs/archive/v2-alpha/`).

- **Snapshot**: `gs://rts-mapping-v2/training/v1.0/` (our bucket — survives external rewrites). Cleaned metadata 22,259 tiles (**1,718 pos / 20,541 neg**), `splits.yaml`, `normalization_stats.json`, `TESTING/`, QC artifacts under `qc/`. Provenance + version semantics in `data/version.json`.
- **QC verdict**: labels + structure flawless (0 unreadable, 0 invalid labels, 0 empty, **0 full-frame** — batch2 rasterization bug fixed). Known issues (next re-stage, negatives only): **49 black + 564 degraded** negatives to drop; also **restore 28 positives** wrongly dropped by the stale-QC staging (target clean pos = 1,746). List: `…/v1.0/qc/known_issues_v1.0.json`.
- **EXTRA absent** → **Phase 4 blocked** until the data team appends the stack. **TESTING/** set (3,000 tiles, spatially entangled with train) — final-lock decision still open (`docs/v1.0_rebaseline.md`).

**Experiment program: Phase-0 re-baseline COMPLETE + program RESUMED** (review done 2026-06-14; ablation waves running on 8× A100).
- `phase0c_seed{42,43,44}` (3-seed baseline) → best smoothed PR-AUC **0.7899 / 0.7863 / 0.7973** → **μ₀=0.7912, σ₀=0.00559, gate G=max(2σ₀,0.01)=0.0112**. Winner bar = **≥0.8023**.
- `phase0a_arm_{a,b,c}` (RGB norm arm-out, seed42, old-LR/short schedule — comparable only among A/B/C): A per-dataset-z **0.6666**, B x255+ImageNet 0.6261, C x255-only 0.6697. C beats A by only +0.003 (<0.01) → **Lock Arm A** (per-dataset z-score). NOTE: arm absolutes are NOT comparable to the seed baseline (arms used the pre-0b LR 1e-3/1e-4 + max_epochs 100).
- `phase0b_lr_{frozen,unfrozen}` → frozen curve clean (informs frozen_lr≈3e-3); unfrozen curve flat/uninformative (no contraindication for base_lr=1e-3). LRs (3e-3/1e-3) baked into the seed config; healthy μ₀ validates them.
- **All locks confirmed on v1.0**: Arm A norm, frozen_lr=3e-3/base_lr=1e-3, gate ratios [5,10,20], metric `val_realistic_pr_auc_geomean`.

**Phase-3 loss family (DONE on v1.0)** — winner bar ≥0.8023 (μ₀+G):
- compound 1:2 = **0.7998** (Δ=+0.0086, just under G) · compound 2:1 = **0.7933** · compound 1:1 = **0.7878** · tversky 3:7 = **0.5902** · tversky 2:8 = **0.0729** (collapse).
- **No loss clears the gate → focal stays the baseline winner.** compound 1:2 is the near-miss → carried into the boundary factorial as the second loss arm.

**Running now (8× A100, ~Up 3 h, no summaries yet):** Phase-3 **boundary factorial** — `phase3_bd_{focal,compound_1to2}_ignore_w{1,2,3}` (6 runs) + `phase3_loss_compound_1to2_seed43` (seed confirm); Phase-5 **architecture sweep** — `phase5_arch_deeplabv3plus` (first of the §8 smp-decoder drop-ins). Results pending.

**Inference compute/region DECIDED (2026-06-15, PR #19):** pan-arctic inference runs on **us-west1, 2× `g2-standard-96` = 16× L4** (forward-only bf16, GCS-I/O-bound), co-located with `pdg-planet-data` (us-west1) → egress-free. Outputs + 2025 EXTRA → new bucket `gs://woodwell-rts-inference-arts-south`. A full ~7.5M-tile pass ≈ 3–9 h ≈ $30–90. Spec in `inference.md §2.1/§2.2` + `infrastructure.md`. Quota ask + bucket create are user/PDG-admin action items.

**EXTRA pipeline (Phase 4 prep, branch `phase4-extra`):** shared derivation SSoT `data/extra_channels.py` + bulk generator `scripts/generate_extra_tiles.py`. **S2 bands (NDVI/NBR/TC) bulk-generated for all 22,259 tiles (2026-06-15, exit 0).** Remaining: deferred norm features ([0.1,99.9] clip + per-channel mode/SE_PROTO bypass), SE path (global PCA(3) + contrastive prototype → bands 2–5), then the Phase-4 ablation waves (S2 groups, then SE + full stack). The same generator (`--year 2025`) is the data-team handoff for inference EXTRA.

**Repo cleanup (2026-06-13)**: tag `v2-alpha-archive` preserves everything; archived the 2 stale results docs + condensed dev-log to `docs/archive/v2-alpha/`; removed 18 dead pre-made configs (recreated on-demand per `experiments.md §11.1`); re-pointed `baseline`/`smoke` to v1.0; fixed stale EXTRA/bucket refs. `configs/` is now 8 active phase0 + `baseline`/`deployment`/`smoke`/`preview_tiles`.

**Repo renamed** `RTSmappingDL` → **`RTSmapping_v2`** (GitHub; old name redirects). Local remote re-pointed 2026-06-15; the local working dir is still `RTSmappingDL` (cosmetic, left as-is).

**Branch state** (PR #18 v1.0-lock is now merged to `main`; consolidation continuing):
- `main` = v1.0 baseline locked (PR #18 merged).
- `inference-pipeline` (PR #19, **MERGEABLE/CLEAN** after merging main 2026-06-15) — Phase-2 inference pipeline + the us-west1 inference-infra decision. 3 of 4 pre-merge checklist items done: rebase-onto-main, the shared NoData helper (cherry-picked `f26dca5` + `tiles.py` swap `38f7909`), and the post-inference merge-ownership doc fix (`ae1dd7a`). Only the deployed-checkpoint norm-stats check remains, blocked on a winner lock.
- `phase4-extra` = EXTRA generation pipeline (S2 bands done; SE path pending).
- Also unmerged: `v1.0-restage`, `phase3-ablations`, `docs-eval-interim` — fold the shared helper + post-inference doc fix into `main` during consolidation.

**Compute**: $70k GCP credit (compute-only), **expires Sep 2026** — must be substantially spent. Stop VMs when idle.

**Next steps:**
1. **Finish the running wave** → read Phase-3 boundary factorial (6 `phase3_bd_*` + `compound_1to2_seed43`) + Phase-5 `arch_deeplabv3plus` against the gate (≥0.8023); pick the boundary winner and queue the rest of the §8 architecture sweep on the freed GPUs.
2. **EXTRA (`phase4-extra`)**: implement the deferred norm features ([0.1,99.9] clip + per-channel mode, SE_PROTO bypass) → recompute S2-inclusive norm stats → build the SE path (global PCA(3) + contrastive prototype, fill bands 2–5) → run the Phase-4 ablation waves (S2 groups, then SE + full stack).
3. **EXTRA handoff doc** (`docs/extra_channels_handoff.md`): final channel list + `generate_extra_tiles.py --year 2025` for the data team → writes 2025 EXTRA to `gs://woodwell-rts-inference-arts-south` (us-west1).
4. **Branch consolidation**: merge PR #19 (only the winner-blocked norm-stats check remains); fold `v1.0-restage`/`phase3-ablations`/`phase4-extra`/`docs-eval-interim` into `main` (the NoData helper + post-inference fix are now also on `inference-pipeline`, so those merges are clean); regenerate `preview_tiles.yaml` from the v1.0 val split.
5. **Inference provisioning** (user/PDG admin): request `NVIDIA_L4_GPUS=16` in us-west1, create `gs://woodwell-rts-inference-arts-south`, benchmark one subregion to pin co-located tiles/s.
6. Stop the 8× A100 box when the wave drains and no follow-on is queued (credit burn; stockout risk on restart — user call).

---

## Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 0** | Data pipeline (`data/`, `scripts/create_splits.py`, `compute_normalization_stats.py`, `check_data*.py`, tests, `configs/baseline.yaml`) | **complete** |
| **Phase 1** | Training loop (`models/`, `losses/`, `training/`, `scripts/train.py`, `evaluate_test.py`, `package_model.py`, MLflow, Docker) | **complete** |
| **Exp Phase 0** | Baseline calibration on the **v1.0 standard dataset**: normalization arm-out (0a), LR range test (0b), 3-seed baseline (0c) → gate G=0.0112. | **complete + locked** (μ₀=0.7912, Arm A) |
| Exp Phases 2–5 | Data-scaling, loss/boundary, EXTRA channels, architecture — `training/experiments.md` | **in progress** (Phase-3 loss done → focal winner; boundary factorial + Phase-5 arch running; Phase-4 EXTRA S2 bands generated) |
| Phase 2 (build) | Inference pipeline (`scripts/inference.py`, tiling, merge, vectorize) + compute/region decided (us-west1 16× L4) | drafted on `inference-pipeline` (PR #19, mergeable) |
| Phase 3 (build) | Post-inference spec + implementation | pending |

Build order is strict (`CLAUDE.md` §Rule 2).

---

## Key Decisions Log

- **2026-06-15** — **Inference runs in us-west1 on a 16× L4 fleet.** Anchored on verified bucket regions (`pdg-planet-data` = us-west1 single-region, holds the TB-scale 2025 Planet quads). Co-locate compute (2× `g2-standard-96` = 16 L4) + a new us-west1 output bucket `gs://woodwell-rts-inference-arts-south` with the input → egress-free. Forward-only bf16 is GCS-I/O-bound, so no A100 (cheap L4, low stockout); Spot for the bulk pass (resume via `inference_log.json`), stop when idle. SSoT: `inference.md §2.1/§2.2`, `infrastructure.md`.
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

- **2026-06-16** — **Region-label leakage found & fixed (split regenerated).** The v1.0 `RegionName` (Dinerstein ecoregion) was wrong for **~59% of tiles** (cropped `circumpolar_subregions.geojson` + centroid CRS issue), violating the spatial-block split: **3,494 tiles (303 pos) leaked across splits, incl. 1,894 (68 pos) contaminating test**. Applied the data team's `metadata_region_hotfix.csv` (Robb/Heidi) via `scripts/apply_region_hotfix.py`, regenerated splits (`create_splits.py`) → **leakage-free** (0 ecoregions span >1 split; 49 ecos → train 40 / val 5 / test 4; test 107 pos / 2,050 neg). Corrected metadata/splits uploaded to GCS (pre-hotfix kept as `*.pre_hotfix`). **Per user policy: experiments NOT re-run** (Phase-0/3/5 numbers are leaky-split → absolute optimistic, relative ~preserved); the corrected split is used for the **honest final test** + Phase-4 EXTRA ablation. Branch `fix-region-split-leakage` (PR). Full writeup `docs/v1.0_region_leakage.md`.
- **2026-06-15** — **Program resumed + inference infra decided + repo rename + EXTRA S2 done.** (1) **Phase-3 loss family complete on v1.0** (winner bar ≥0.8023): compound 1:2 = 0.7998 (near-miss, Δ<G), 2:1 = 0.7933, 1:1 = 0.7878, tversky 3:7 = 0.5902, 2:8 = 0.0729 → **no loss clears the gate, focal stays winner**; compound 1:2 carried into the boundary factorial. Running now on 8× A100: boundary factorial (`phase3_bd_{focal,compound_1to2}_ignore_w{1,2,3}` + `compound_1to2_seed43`) + `phase5_arch_deeplabv3plus`. (2) **Inference compute/region decided** — us-west1, 2× `g2-standard-96` = 16× L4, outputs → `gs://woodwell-rts-inference-arts-south`; co-located with `pdg-planet-data` (egress-free); spec in `inference.md §2.1/§2.2` + `infrastructure.md` (commit `2aff5bf`). PR #19 merged `main` (PR #18 v1.0 lock) → **MERGEABLE/CLEAN** (`1adbb71`); 2/4 reconciliation items still open (NoData helper + post-inference disambiguation live on `v1.0-restage`/`phase4-extra`, not yet on `main`; norm-stats check blocked on winner). (3) **Repo renamed `RTSmappingDL` → `RTSmapping_v2`** on GitHub (old name redirects); local remote re-pointed. (4) **EXTRA S2 bands (NDVI/NBR/TC) bulk-generated for all 22,259 tiles** (`scripts/generate_extra_tiles.py`, exit 0) on `phase4-extra`; SE path + deferred norm features next.
- **2026-06-14** — **Baseline locked for the project; re-baseline triggers defined.** μ₀=0.7912 / σ₀=0.00559 / **G=0.0112**, the normalization stats, and the **val/test split are frozen** as the project-wide yardstick — every ablation compares to them ("freeze eval, vary one thing"). Default data work = add tiles to **train only** and keep training (no recompute). Recompute "key things" only for: large train growth / new-domain batch (→re-anchor μ₀ + re-baseline), label-semantics change (→re-baseline), val/test split change (→re-score baseline), radiometric-domain change (→recompute norm + re-baseline), architecture/EXTRA-channel change (→re-baseline). The pending re-stage (keep 564 degraded negs via §4.4, restore 28 pos, drop 49 black) is small + train-only → baseline & norm stay locked. Full table + lessons in `docs/phase0_baseline.md`.
- **2026-06-14** — **Phase-0 v1.0 re-baseline complete → gate locked.** All 6 runs early-stopped cleanly. 3-seed baseline → **μ₀=0.7912, σ₀=0.00559, G=max(2σ₀,0.01)=0.0112** (winner bar ≥0.8023). RGB norm arm-out → **Lock Arm A** (per-dataset z-score; B/C don't clear the 0.01 floor). LRs 3e-3/1e-3 carried from 0b. Full writeup `docs/phase0_baseline.md`. **Correction to prior status notes:** the phase0a arms ran on the old LR (1e-3/1e-4) + max_epochs 100, so their ~0.65 scores are an internal A/B/C comparison only — NOT a 0.13 deficit vs the seed baseline (that earlier framing was an apples-to-oranges error). Outputs moved to `/mnt/outputs/v1.0/runs/` (Wave 2). **Paused for review** before the next ablation phase.
- **2026-06-13** — **`lr_range_test` no longer crashes on divergence.** The unfrozen LR-range test (`phase0b_lr_unfrozen`) ramps LR to 1e-1 over one epoch; the full backbone diverges to NaN weights, and the **forced final-epoch validation** then fed NaN logits to `average_precision_score` / figure rendering → crash. Fixed at the source in `scripts/train.py`: gate the validation block on the existing `is_range_test` flag so range tests skip validation entirely (deliverable is the per-step `lr_range_test.csv`, dumped in `finally`, not val metrics on a blown-up model). First attempt — a NaN guard in `metrics.py` — was whack-a-mole (NaN just resurfaced downstream) and was reverted. Rerun now exits 0; all 8 Phase-0 runs accounted for. The unfrozen curve is flat/uninformative (focal loss stays small even at high LR) → treat as no-contraindication for `base_lr=1e-4`; the frozen test is the reliable one.
- **2026-06-13** — **v1.0 re-baseline + standard-dataset declaration + repo cleanup.** Staged the regenerated data into `gs://rts-mapping-v2/training/v1.0/` (`scripts/stage_v1_snapshot.py`), regenerated splits + normalization stats, re-pointed the 8 phase0 configs, fixed an MLflow env-override bug + a `--privileged`/`--gpus` GPU-pinning bug (`run_ablation_queue.sh`), and launched the parallel Phase-0 calibration on 8 A100s at BS 32. Ran a full fresh QC (`scripts/qc_full_dataset.py`) → declared **v1.0 the standard dataset** (`docs/v1.0_qc.md`). Cleanup: tagged `v2-alpha-archive`, archived stale v2-alpha results + dev-log, removed 18 dead configs, fixed stale refs. Decisions: BS=32, no-DDP (above). Pending: compute the new gate when runs finish (pause for review), then branch consolidation.
