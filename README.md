# RTS Segmentation Model v2

Semantic segmentation of **Retrogressive Thaw Slumps (RTS)** in Arctic satellite imagery for pan-arctic mapping.

## Overview

This project trains a deep learning model to detect RTS from PlanetScope basemap imagery (up to 74N) and deploys it for pan-arctic inference to produce an RTS survey map.

## Quick Links

| Document | Description |
|----------|-------------|
| [Data Specification](data/data.md) | Data sources, labeling rules, split strategy |
| [Training Guide](training/training.md) | Model architecture, loss, metrics, hyperparameters |
| [Inference Pipeline](inference/inference.md) | Deployment workflow, tiling, post-processing |
| [Post-Inference](post-inference/post-inference.md) | Post-processing, map-making, visualisation,  Quality control, failure mode analysis, threshold tuning |

## Data

- **Training**: 2024 PlanetScope Quarterly Basemap (RGB 3m)
- **Inference**: 2025 PlanetScope Quarterly Basemap
- **Labels**: Refined from ARTS dataset on 2024 imagery(~2–3k positive, ~20–25k negative tiles)
- **Auxiliary** (optional): Sentinel-2 NDVI/NIR, ArcticDEM derivatives

## Training


## Inference


## Post-inference


## Computation

- Google Cloud Platform VM (L4, A100/H100) via PDG
- Dockerization for environment control on different VMs for training/inferencing
