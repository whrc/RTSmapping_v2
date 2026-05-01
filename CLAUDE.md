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
│   └── post-inference.md      ← post-processing, evaluation (spec TBD)
├── computing/
│   ├── docker_training.md     ← Docker environment
│   └── vm_instruction.md      ← GCP VM setup
├── models/                    ← model definitions
├── losses/                    ← loss functions
├── utils/                     ← shared utilities
├── src/                       ← package init only
├── tests/                     ← unit and integration tests
├── configs/                   ← experiment YAML configs (one per experiment)
├── scripts/                   ← entry-point scripts
│   ├── train.py               ← single training entry point (config-driven)
│   ├── inference.py           ← inference entry point
│   ├── check_data.py          ← standalone data validation script
│   └── create_splits.py       ← generate splits.yaml from metadata + regions
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

Data normalization and transforms **must** be implemented as shared modules in `data/` that both training and inference import from. Never duplicate preprocessing logic. Training–inference consistency is critical (see `training/training.md` §4).

### Rule 4: Test Before Moving On

Write tests in `tests/` for each module. Tests should be runnable without GPU where possible. For the data pipeline, create a standalone `scripts/check_data.py` that iterates through the DataLoader to verify augmentations, normalization, and tensor collation work end-to-end (see `training/training.md` §10.1).

See `tests/tests.md` for the test-suite living doc — per-test inventory, strictness ratings, known coverage gaps, and conventions. **When you add, remove, or meaningfully change a test, update `tests/tests.md` in the same change.**

### Rule 5: Single Source of Truth

This repo follows SSoT standard, if one variable is mentioned multiple places, reference to the place where the SSoT exist.

## Technical Constraints

- **CRS**: EPSG:3857 everywhere. No exceptions.
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
- **Compute**: GCP VMs only. Dev/test on L4 VM via VSCode Remote-SSH; production training on A100/H100 VM. No Colab. See `computing/vm_instruction.md`.
- **MLflow**: GCS-backed; tracking URI is configured in `configs/baseline.yaml:mlflow.tracking_uri` (single source).
- **Data storage**: GCS bucket `gs://abruptthawmapping/`, mounted via gcsfuse in Docker. All paths configured in YAML — no hardcoded GCS paths.

## Code Style

- Type hints on all function signatures.
- Docstrings on all public functions (Google style).
- Config via YAML files in `configs/`, not hardcoded values.
- Log with Python `logging` module, not print statements.
- No wildcard imports.
- Install packages via requirements.txt, not ad-hoc pip install

## Documentation

Process and results go in `docs/` as living markdowns. Each major experiment has a single md document (major model version). For each sub-experiment (minor model version) iteration records: design decision, implementation details, results, and analysis on the same md document.

All critical milestone, decision, progress, current status should go to the living document current_working_status.md, this is the dairy and roadmap of this project.

## andrej-karpathy rules

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.