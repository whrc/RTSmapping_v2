# RTS Segmentation Model v2

Semantic segmentation of **Retrogressive Thaw Slumps (RTS)** in Arctic satellite imagery for pan-arctic mapping.

## Overview

This project trains a deep learning model to detect RTS from PlanetScope basemap imagery (up to 74N) and deploys it for pan-arctic inference to produce an RTS survey map.

This README is the **map of the repo** — every canonical document is linked below, and the
[Source of truth](#source-of-truth) table says where each kind of fact lives. Specs are the source
of truth: always read the relevant doc before implementing (see [CLAUDE.md](CLAUDE.md) §Rule 1).

## Data

- **Training**: 2024 PlanetScope Quarterly Basemap (RGB 3m)
- **Inference**: 2025 PlanetScope Quarterly Basemap
- **Labels**: Refined from ARTS dataset on 2024 imagery (~2–3k positive, ~20–25k negative tiles)
- **Auxiliary** (optional): Sentinel-2 NDVI/NIR, ArcticDEM derivatives

## Document map

### Project & process
| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | How to work in this repo: rules, structure, technical constraints, code style |
| [current_working_status.md](current_working_status.md) | Diary & roadmap — live status, key decisions, budget plan, next steps |

### Data
| Document | Purpose |
|----------|---------|
| [data/data.md](data/data.md) | Data pipeline spec — sources, labels, splits, normalization, disk layout (§9 = EXTRA bands) |
| [data/data_format.md](data/data_format.md) | Format standards for all data (CRS, tile size, label values, dtypes) |
| [data/datacheck.md](data/datacheck.md) | Data-validation checks at each lifecycle stage |

### Training
| Document | Purpose |
|----------|---------|
| [training/training.md](training/training.md) | Model, loss, metrics, training loop, train–inference consistency contract |
| [training/experiments.md](training/experiments.md) | The phased experimentation plan (sequential elimination + multi-seed lock) |
| [docs/baseline_unetpp_effb5.md](docs/baseline_unetpp_effb5.md) | Living experiment record for the UNet++/EfficientNet-B5 baseline |

### Inference & post-inference
| Document | Purpose |
|----------|---------|
| [inference/inference.md](inference/inference.md) | Deployment workflow — tiling, overlap aggregation, merging, vectorization |
| [post-inference/post-inference.md](post-inference/post-inference.md) | Post-processing, QC, evaluation, threshold tuning *(spec in progress)* |

### Computing
| Document | Purpose |
|----------|---------|
| [computing/infrastructure.md](computing/infrastructure.md) | **Infra SSoT** — GCP projects, buckets, VM inventory, regions, compute budget, data storage map |
| [computing/vm_instruction.md](computing/vm_instruction.md) | Daily VM/SSH how-to — start/stop, config, Python env, file transfer |
| [computing/docker_training.md](computing/docker_training.md) | Docker build/run how-to — image, mounts, GCS auth |

### Domain
| Document | Purpose |
|----------|---------|
| [domain/inference_domain.md](domain/inference_domain.md) | Inference domain and circumpolar subregions (H. Rodenhizer) |
| [domain/training_data_distribution.md](domain/training_data_distribution.md) | Geographic/ecological distribution of the training data (H. Rodenhizer) |

### Tests
| Document | Purpose |
|----------|---------|
| [tests/tests.md](tests/tests.md) | Test-suite living doc — per-test inventory, strictness, coverage gaps |

## Source of truth

This repo follows a single-source-of-truth standard. Where each kind of fact lives:

| Concern | Source of truth |
|---------|-----------------|
| Config values — hyperparameters, paths, thresholds | `configs/*.yaml` (`configs/baseline.yaml` is primary) |
| MLflow tracking URI | `configs/baseline.yaml:mlflow.tracking_uri` |
| Core constants — CRS, tile size, label values, seed | [CLAUDE.md](CLAUDE.md) §Technical Constraints |
| Data disk layout & EXTRA bands | [data/data.md](data/data.md) (§9 for bands) |
| Status, roadmap, decisions | [current_working_status.md](current_working_status.md) |
| Test inventory | [tests/tests.md](tests/tests.md) |
| Infra facts — projects, buckets, VMs, regions, budget | [computing/infrastructure.md](computing/infrastructure.md) |

## Todos
1. training in multi-scale
2. explore GEE satellite embedding as input feature
3. 2025 micro set to test temporal domain shift
