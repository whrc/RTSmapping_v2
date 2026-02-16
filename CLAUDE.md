# RTSmappingDL

## Project Overview

Semantic segmentation of Retrogressive Thaw Slumps (RTS) in Arctic satellite imagery for pan-arctic mapping (60–74°N). Trains on 2024 PlanetScope basemap imagery, deploys inference on 2025 data.

This is a **solo research project**. Prioritise simplicity, maintainability, and reproducibility. Do not over-engineer with unnecessary abstractions, factory patterns, or deep class hierarchies. Flat is better than nested. A solo researcher needs to read and debug this code quickly.

## Project Structure
```
RTSmappingDL/
├── CLAUDE.md                  ← you are here
├── .claude/skills/            ← implementation recipes (loaded on demand)
├── data/
│   ├── data.md                ← data pipeline spec
│   └── data_format.md         ← format standards for all data
├── training/
│   └── training.md            ← training loop, model, experiments
├── inference/
│   └── inference.md           ← inference pipeline and deployment
├── post-inference/
│   └── post-inference.md      ← post-processing, evaluation
├── computing/
│   ├── docker_training.md     ← Docker environment
│   └── vm_instruction.md      ← GCP VM setup
├── src/                       ← all source code
│   ├── data/                  ← data loading, transforms, normalization
│   ├── models/                ← model definitions
│   ├── losses/                ← loss functions
│   └── ...
├── tests/                     ← unit and integration tests
├── configs/                   ← experiment YAML configs
├── scripts/                   ← entry-point scripts (train.py, inference.py)
├── notebooks/                 ← exploration only, not production code
└── docs/                      ← living documentation of results and decisions
```

## How to Work in This Repo

### Rule 1: Spec First, Code Second

Every component has a detailed markdown spec. **Always read the relevant spec before implementing anything.** The spec is the source of truth.

| Task | Read first |
|------|-----------|
| Data loading, labels, splits, normalization | `data/data.md` and `data/data_format.md` |
| Model, loss, training loop, experiments | `training/training.md` |
| Tiling, inference pipeline, merging | `inference/inference.md` |
| Vectorization, QC, evaluation | `post-inference/post-inference.md` |
| Docker setup | `computing/docker_training.md` |
| VM provisioning | `computing/vm_instruction.md` |

If a spec is unclear or incomplete, **ask — do not assume**.

### Rule 2: One Component at a Time

Build in this order: **data → training → inference → post-inference**. Do not implement ahead of the current component unless explicitly asked. Each component should be code-complete and tested before moving on. However, when building early component, their impact on the late components should also be considered.

### Rule 3: Shared Preprocessing

Data normalization and transforms **must** be implemented as shared modules in `src/data/` that both training and inference import from. Never duplicate preprocessing logic. Training–inference consistency is critical (see `training/training.md` §4).

### Rule 4: Test Before Moving On

Write tests in `tests/` for each module. Tests should be runnable without GPU where possible. For the data pipeline, create a standalone `scripts/check_data.py` that iterates through the DataLoader to verify augmentations, normalization, and tensor collation work end-to-end (see `training/training.md` §10.1).

## Technical Constraints

- **CRS**: EPSG:3413 everywhere. No exceptions.
- **Tile size**: 512×512 pixels.
- **Label values**: 0 = background, 1 = RTS, 255 = ignore.
- **Normalization**: Per-dataset statistics (not per-image). Saved as `normalization_stats.json` alongside model checkpoints.
- **Data formats**: Raw values stored on disk. Normalization applied at load time only.
- **Reproducibility**: Seed 42, deterministic CUDNN, pinned library versions.

## Technical Stack

- **Framework**: PyTorch 2.x
- **Geospatial**: rasterio, geopandas
- **Augmentation**: albumentations
- **Experiment tracking**: MLflow
- **Environment**: Docker (see `computing/docker_training.md`)
- **Compute**: Test/sanity check with GPU provided by Google colab Pro+ (H100), and full scale training and inference with GCP VMs.

## Code Style

- Type hints on all function signatures.
- Docstrings on all public functions (Google style).
- Config via YAML files in `configs/`, not hardcoded values.
- Log with Python `logging` module, not print statements.
- No wildcard imports.

## Documentation

Process and results go in `docs/` as living markdowns. Each major experiment has a single md document (major model version). For each sub-experiment (minor model version) iteration records: design decision, implementation details, results, and analysis on the same md document.