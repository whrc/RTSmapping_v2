"""Inference entry point (inference.md §8, §12.2).

Reads a pre-filtered tile list, runs the deployment-package model over batched
512x512 tiles windowed from the 2025 basemap quads, and writes per-tile
probability COGs + a resumable inference_log.json manifest. Merging and
thresholding are a separate pass (scripts/merge_predictions.py).

The setup + per-tile loop live in `inference/runner.py` so this CLI and the
queue worker (`scripts/run_inference_worker.py`) share one code path (Rule 3).

Usage:
    python scripts/inference.py \
        --config configs/deployment.yaml \
        --tile-list tiles.csv \
        --quad-index quad_index_2025q3.csv \
        --package gs://rts-arctic-us/models/rts-v2-seed42 \
        --output gs://rts-arctic-us/inference/2025-Q3/tiles
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# GDAL /vsigs/ + google-cloud auth via ADC (same bootstrap as train.py).
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if _adc.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_adc)

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.runner import build_context, run_inference, run_metadata  # noqa: E402
from inference.writer import Manifest  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/deployment.yaml")
    p.add_argument("--tile-list", required=True)
    p.add_argument("--quad-index", required=True)
    p.add_argument("--s2-index", default=None,
                   help="S2 composite index CSV (scripts/build_s2_index.py); "
                        "required iff the package declares EXTRA=NDVI")
    p.add_argument("--package", required=True, action="append",
                   help="deployment package dir (local or gs://); repeat for the "
                        "N-seed mean-prob ensemble (configs/deployment.yaml `ensemble`)")
    p.add_argument("--output", required=True,
                   help="output dir for probability tiles + inference_log.json")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--scale", type=float, default=1.0,
                   help="inference scale (inference.md §6.2); 0.5 = 2x GSD / "
                        "4x FOV decimated reads. Experimental — production "
                        "multi-scale is gated by §6.4; tile list must be "
                        "generated with the matching scale.")
    args = p.parse_args()
    setup_logging()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_cfg = load_config(args.config)

    ctx = build_context(run_cfg, args.package, args.quad_index, args.s2_index, device)
    tiles = pd.read_csv(args.tile_list)
    out = args.output.rstrip("/")

    manifest = Manifest(f"{out}/inference_log.json", run_metadata(ctx, device))
    run_inference(ctx, tiles, out, manifest, device,
                  num_workers=args.num_workers, scale=args.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
