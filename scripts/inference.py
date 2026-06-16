"""Inference entry point (inference.md §8, §12.2).

Reads a pre-filtered tile list, runs the deployment-package model over batched
512x512 tiles windowed from the 2025 basemap quads, and writes per-tile
probability COGs + a resumable inference_log.json manifest. Merging and
thresholding are a separate pass (scripts/merge_predictions.py).

Usage:
    python scripts/inference.py \
        --config configs/deployment.yaml \
        --tile-list tiles.csv \
        --quad-index quad_index_2025q3.csv \
        --package gs://rts-mapping-v2/models/rts-v2-seed42 \
        --output gs://rts-mapping-v2/inference/2025-Q3/tiles
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# GDAL /vsigs/ + google-cloud auth via ADC (same bootstrap as train.py).
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if _adc.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_adc)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.predictor import (  # noqa: E402
    assert_runtime_matches_package, load_deployment_package, predict_probs,
)
from inference.quad_index import load_quad_index  # noqa: E402
from inference.tiles import InferenceTileDataset  # noqa: E402
from inference.writer import NODATA_PROB, Manifest, write_probability_tile  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _collate(items: list[dict]) -> dict:
    """Stack a batch, keeping per-tile metadata as lists."""
    return {
        "tile_id": [it["tile_id"] for it in items],
        "image": torch.from_numpy(np.stack([it["image"] for it in items])),
        "nodata_mask": np.stack([it["nodata_mask"] for it in items]),
        "all_nodata": [it["all_nodata"] for it in items],
        "bounds": [tuple(it["bounds"]) for it in items],
    }


def _weights_sha256(package: str) -> str:
    path = f"{package.rstrip('/')}/weights.pth"
    h = hashlib.sha256()
    if path.startswith("gs://"):
        import gcsfs
        f = gcsfs.GCSFileSystem(token="google_default").open(path[5:], "rb")
    else:
        f = open(path, "rb")
    with f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/deployment.yaml")
    p.add_argument("--tile-list", required=True)
    p.add_argument("--quad-index", required=True)
    p.add_argument("--package", required=True,
                   help="deployment package dir (local or gs://)")
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

    pkg = load_deployment_package(args.package, device)
    dep_cfg = pkg["dep_cfg"]
    assert_runtime_matches_package(run_cfg, dep_cfg)
    if pkg["n_channels"] != 3:
        raise NotImplementedError("EXTRA channels at inference await the final "
                                  "EXTRA definition (RGB-only for now)")

    quad_index = load_quad_index(args.quad_index)
    tiles = pd.read_csv(args.tile_list)
    out = args.output.rstrip("/")

    manifest = Manifest(f"{out}/inference_log.json", run_metadata={
        "model_version": Path(str(args.package).rstrip("/")).name,
        "deployment_package_path": str(args.package),
        "model_checkpoint_sha": _weights_sha256(str(args.package)),
        "inference_date": datetime.now(timezone.utc).isoformat(),
        "scales_used": dep_cfg.get("scales", [1.0]),
        "tta_config": dep_cfg.get("tta", "none"),
        "precision": dep_cfg.get("precision"),
        "torch_compile": bool(dep_cfg.get("torch_compile", False)),
        "threshold": dep_cfg["threshold"],
        "temperature": dep_cfg["temperature"],
        "stride_px": run_cfg["inference"]["stride_px"],
        "overlap_aggregation": "gaussian_weighted_mean",
        "fusion_sigma_px": run_cfg["inference"]["fusion_sigma_px"],
        "gpu_type": (torch.cuda.get_device_name(device)
                     if device.type == "cuda" else "cpu"),
    })

    todo = tiles[~tiles["tile_id"].astype(str).isin(manifest.completed)]
    logger.info("%d tiles total, %d already done, %d to process",
                len(tiles), len(tiles) - len(todo), len(todo))
    if todo.empty:
        manifest.save()
        return 0

    dataset = InferenceTileDataset(todo, quad_index, pkg["mean"], pkg["std"],
                                   scale=args.scale)
    loader = DataLoader(dataset, batch_size=run_cfg["inference"]["batch_size"],
                        num_workers=args.num_workers, collate_fn=_collate)

    t0, n_done = time.time(), 0
    for batch in loader:
        keep = [i for i, all_nd in enumerate(batch["all_nodata"]) if not all_nd]
        for i, all_nd in enumerate(batch["all_nodata"]):
            if all_nd:
                manifest.mark(batch["tile_id"][i], "all_nodata")
        if keep:
            images = batch["image"][keep].to(device)
            probs = predict_probs(pkg["model"], images,
                                  temperature=dep_cfg["temperature"],
                                  tta=dep_cfg.get("tta", "none"),
                                  precision=dep_cfg.get("precision", "fp32"))
            probs = probs.clamp_(0.0, 1.0).cpu().numpy()  # §10.1 range guard
            for j, i in enumerate(keep):
                prob = probs[j]
                prob[batch["nodata_mask"][i]] = NODATA_PROB  # §5.3 output mask
                tile_id = batch["tile_id"][i]
                write_probability_tile(f"{out}/{tile_id}.tif", prob,
                                       batch["bounds"][i])
                manifest.mark(tile_id, "done")
        n_done += len(batch["tile_id"])
        rate = n_done / (time.time() - t0)
        if n_done % 512 < len(batch["tile_id"]):
            logger.info("%d/%d tiles (%.1f tiles/s, ETA %.1f h)",
                        n_done, len(todo), rate, (len(todo) - n_done) / rate / 3600)

    manifest.save()
    logger.info("Done: %s", manifest.counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
