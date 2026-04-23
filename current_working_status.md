# Master Working Document

Living doc maintained by YYang and Claude Code. Track development progress, record decisions, log status.

---

## Project Summary

Semantic segmentation of **Retrogressive Thaw Slumps (RTS)** in Arctic satellite imagery (60–74°N). Train on 2024 PlanetScope Quarterly Basemap (RGB, ~3m), deploy inference on 2025 imagery for a pan-arctic RTS survey map. Solo research project — flat code structure, minimal abstraction.

**Core constraints** (non-negotiable, see `CLAUDE.md`):
- CRS: EPSG:3857 everywhere
- Tile size: 512×512 px
- Labels: 0=bg, 1=RTS, 255=ignore
- Normalization: per-dataset stats, saved as `normalization_stats.json` alongside model
- Seed 42, deterministic CUDNN

**Stack**: PyTorch 2.x + `segmentation_models_pytorch` (UNet++/EfficientNet-B5 baseline), albumentations, rasterio, geopandas. MLflow on GCS (`gs://abruptthawmapping/mlflow/`). Compute: L4 VM (dev) → A100/H100 VM (prod training) via Docker.

**Imbalance strategy** (real prevalence ~0.1–0.5% positive pixels): balanced batch sampling (50/50 tile-level) + focal loss + curriculum schedule (1:1 → 1:20 pos:neg over 300 epochs). Optimize for high precision at acceptable recall.

Specs are the source of truth. Always read the relevant md before implementing (see `CLAUDE.md` §Rule 1).

---

## Status — 2026-04-23

- **Spec phase**: complete except `post-inference/post-inference.md`; `training/training.md` and `inference/inference.md` expanded with the §4 train-inference consistency contract, overlap math, multi-scale / TTA gates, NoData handling, deployment-package layout (Phase 1 Step 0.5).
- **Phase 0** (data pipeline): complete, merged as PR #8.
- **Phase 1** (training loop): code-complete on synthetic fixtures; real-data smoke pending.
  - Models, losses, training utilities, scripts/train.py, MLflow wiring, visualizations, packaging & evaluation scripts, deployment-config template — all landed. 113 tests green (105 fast + 8 end-to-end training smoke).
  - Pending: real-data smoke on L4 VM (`scripts/train.py --config configs/smoke.yaml`) → then Dockerfile materialization → then production run on A100/H100. Phase 1 Step 8.5 (inference feasibility gates) and Step 8 (one-shot test eval) run after the production baseline completes.
- **Dataset v2.0**: see status below — Phase 0 runs on synthetic fixtures for tests; real-data validation is the next gate.
- **Next step**: Phase 1 Step 7b — real-data smoke on L4 VM once v2.0 bucket is finalized enough to have sample tiles for at least 2 regions.

---

## Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 0** | Data pipeline (`data/`, `utils/`, `scripts/create_splits.py`, `scripts/compute_normalization_stats.py`, `scripts/check_data_content.py`, `scripts/check_data.py`, tests, `configs/baseline.yaml`) | **complete** (PR #8 merged 2026-04-23) |
| **Phase 1** | Training loop (`models/`, `losses/`, `training/`, `scripts/train.py`, `scripts/evaluate_test.py`, `scripts/package_model.py`, `scripts/check_inference_normalization.py`, `scripts/inference_feasibility.py`, `configs/deployment.yaml`, MLflow, visualizations, Dockerfile build) | **code-complete on synthetic** (2026-04-23); pending real-data smoke on L4 and Dockerfile build |
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
