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

## Status — 2026-04-22

- **Spec phase**: complete except `post-inference/post-inference.md` (deferred until after inference is built).
- **SE-channel investigation**: landed (`scripts/channel_correlation.py`, `scripts/se_variants.py`, plots). Done.
- **Production code**: none yet. Only `scripts/check_data_format.py` exists from the data-ops side.
- **Dataset v2.0**: partially ready in `gs://abruptthawmapping/training/v2.0/` — some tiles/metadata/region boundaries exist, not all. Phase 0 runs on synthetic fixtures for tests; real-data validation happens as the bucket finalizes.
- **Next step**: Phase 0 — build the shared data pipeline on the L4 VM.

---

## Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 0** | Data pipeline (`data/`, `utils/`, `scripts/create_splits.py`, `scripts/compute_normalization_stats.py`, `scripts/check_data_content.py`, `scripts/check_data.py`, tests, `configs/baseline.yaml`) | **in progress** (2026-04-22 start) |
| Phase 1 | Training loop (`models/`, `losses/`, `scripts/train.py`, MLflow wiring, Dockerfile build) | pending |
| Phase 2 | Inference (`scripts/inference.py`: tiling, multi-scale, TTA, COG output, vectorization) | pending |
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
