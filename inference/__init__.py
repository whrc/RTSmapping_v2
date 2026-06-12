"""Inference pipeline package (inference/inference.md).

Modules:
    quad_index — index of 2025 PlanetScope basemap quads on GCS
    tiles      — windowed 512x512 tile reads from quads + normalization
    predictor  — deployment-package loading, TTA, temperature, probabilities
    writer     — COG outputs + inference_log.json manifest (resumability)
"""
