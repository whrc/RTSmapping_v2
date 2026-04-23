"""Training utilities for the RTS pipeline.

Modules:
    ema              — EMAModel + swap_in context manager
    scheduler        — two-phase LR schedule (frozen -> cosine with warmup)
    metrics          — pixel IoU/F1, object P/R/F1, PR-AUC at ratios
    checkpoint       — deployment + resume checkpoint managers
    freeze           — backbone freeze/unfreeze + param group builder
    early_stopping   — 3-validation moving average, start-epoch gated
    mlflow_utils     — run setup, param/metric/artifact logging (Step 5)
    visualizations   — prediction-preview figures, PR curves, etc. (Step 5)
"""

from __future__ import annotations
