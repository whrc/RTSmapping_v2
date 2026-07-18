# RTS Segmentation Model v2

Semantic segmentation of **Retrogressive Thaw Slumps (RTS)** in Arctic satellite imagery for pan-arctic mapping.

## Overview

This project trains a deep learning model to detect RTS from PlanetScope basemap imagery (up to 74N) and deploys it for pan-arctic inference to produce an RTS survey map.

**Status (2026-07):** model v2 (3-seed UNet++/EfficientNet-B5 ensemble, RGB+NDVI, thr 0.65) is
**deployed** — the pan-Arctic South run (2025 Q3 imagery, ≈50–76°N, 41.57M tiles) is complete and its
products are shipped (see [Deployed products](#deployed-products)). Model v2.1 (DINOv3-L MAE
self-supervised pretraining) is in progress on branch `v2.1-pretraining`.

This README is the **map of the repo** — every canonical document is linked below, and the
[Source of truth](#source-of-truth) table says where each kind of fact lives. Specs are the source
of truth: always read the relevant doc before implementing (see [CLAUDE.md](CLAUDE.md) §Rule 1).

## Data

- **Training**: 2024 PlanetScope Quarterly Basemap (RGB 3m)
- **Inference**: 2025 PlanetScope Quarterly Basemap
- **Labels**: Refined from ARTS dataset on 2024 imagery (~2–3k positive, ~20–25k negative tiles)
- **Auxiliary** (optional): Sentinel-2 NDVI/NIR, ArcticDEM derivatives

## Deployed products

The 2025 Q3 pan-Arctic South run produced a tiered product family in
`gs://rts-mapping-v2-usw1/inference/2025q3_south/products/` — catalog SSoT:
[post-inference/south_products.md](post-inference/south_products.md).

- **High-confidence map**: `south_rts_high_confidence.gpkg` — 19,068 polygons / 529.7 km², QC-calibrated
  `rts_class` from a measured precision grid (279 human ratings).
- **Full candidate inventory** (MMU≈0): 60,167 polygons / 688.2 km² across confidence tiers;
  original delivered `south_rts.gpkg` (10,984 / 238.1 km² at thr 0.65) kept for provenance.
- **Rasters**: probability + mask mosaics (1,633 COG shards), 95 m browse likelihood surface,
  10 km / 0.5° hotspot density grids (threshold-free expected RTS area 1,037.4 km²).
- **Open in ArcGIS Pro**: [post-inference/arcgis_south_products.md](post-inference/arcgis_south_products.md).

## Document map

### Project & process
| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | How to work in this repo: rules, structure, technical constraints, code style, the 3-doc update ritual |
| [docs/experiment_ledger.md](docs/experiment_ledger.md) | **Experiments SSoT** — every run, the locked recipe, per-family findings, dropped ideas (scores auto-harvested) |
| [current_working_status.md](current_working_status.md) | Project diary — rolling progress (just-completed · now · future); links to the ledger for numbers |
| [docs/report.html](docs/report.html) | Generated analytical + visual report (build with `scripts/build_report.py`) |

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
| [docs/optimization_roadmap.md](docs/optimization_roadmap.md) | Cross-aspect optimization roadmap (training/inference/infra) + experiment-fairness & validity audit |

### Inference & post-inference
| Document | Purpose |
|----------|---------|
| [inference/inference.md](inference/inference.md) | Deployment workflow — tiling, overlap aggregation, merging, vectorization |
| [post-inference/post-inference.md](post-inference/post-inference.md) | Post-processing, QC, evaluation, threshold tuning *(spec complete; multi-scale fusion deferred)* |
| [post-inference/south_products.md](post-inference/south_products.md) | **Product-catalog SSoT** — every shipped South product: provenance, decode, tier table, caveats |
| [post-inference/arcgis_south_products.md](post-inference/arcgis_south_products.md) | How to download and open the South products in ArcGIS Pro |
| [deliverables/README.md](deliverables/README.md) | ADC/PDG handover doc — submission manifest, WMTS tiling convention, methods, attribute dictionary |

### Computing
| Document | Purpose |
|----------|---------|
| [computing/infrastructure.md](computing/infrastructure.md) | **Infra SSoT** — GCP projects, buckets, VM inventory, regions, compute budget, data storage map |
| [computing/vm_instruction.md](computing/vm_instruction.md) | Daily VM/SSH how-to — start/stop, config, Python env, file transfer |
| [computing/docker_training.md](computing/docker_training.md) | Docker build/run how-to — image, mounts, GCS auth |
| [computing/artifact_inventory.md](computing/artifact_inventory.md) | Artifact → bucket/path map — where every durable artifact lives, SSoT vs backup |

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
| Config values — hyperparameters, paths, thresholds | `configs/*.yaml` (`configs/base_v2_fast.yaml` is the canonical base for current experiments; `configs/baseline.yaml` records the original Phase-0/1 baseline + shared infra keys) |
| MLflow tracking URI | `configs/baseline.yaml:mlflow.tracking_uri` |
| Core constants — CRS, tile size, label values, seed | [CLAUDE.md](CLAUDE.md) §Technical Constraints |
| Data disk layout & EXTRA bands | [data/data.md](data/data.md) (§9 for bands) |
| Status, roadmap, project decisions | [current_working_status.md](current_working_status.md) |
| Experiments, scores, recipe, findings | [docs/experiment_ledger.md](docs/experiment_ledger.md) |
| Test inventory | [tests/tests.md](tests/tests.md) |
| Infra facts — projects, buckets, VMs, regions, budget | [computing/infrastructure.md](computing/infrastructure.md) |
| Shipped South products — files, numbers, caveats | [post-inference/south_products.md](post-inference/south_products.md) |
| Artifact locations — what lives in which bucket | [computing/artifact_inventory.md](computing/artifact_inventory.md) |

## Todos
1. ~~training in multi-scale~~ **DONE** (ledger family M, 2026-07-02): 0.5× re-stage + joint dual-scale training, 3 seeds — gates 1+2 pass, gate 3 (fusion recall) fail; inference multiscale path implemented, deploy stays `scales:[1.0]`.
2. ~~pan-Arctic South inference + products~~ **DONE** (2026-07): full run + tiered QC-calibrated products shipped (see [Deployed products](#deployed-products)).
3. **v2.1** — DINOv3-L MAE self-supervised pretraining on the 295k-tile South corpus (in progress, branch `v2.1-pretraining`); v3 hard-negative mining seeded from `qc_false_hard_negatives.gpkg` (152 QC-verified FPs).
4. explore GEE satellite embedding as input feature
5. 2025 micro set to test temporal domain shift
