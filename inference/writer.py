"""Probability-tile COG writing + inference_log.json manifest (inference.md §8.3, §9).

Outputs follow §9.1: Float32 COG, NoData -1.0, EPSG:3857, deflate. GCS writes
go through a local temp file then a single upload (GCS object creation is
atomic, so a crashed upload never leaves a half-written object). The manifest
records completed/skipped tiles so a restarted job skips finished work.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

logger = logging.getLogger(__name__)

NODATA_PROB = -1.0  # §9.1 sentinel
NODATA_MASK = 255   # §9.2

_COG_PROFILE = dict(
    driver="GTiff", compress="deflate", tiled=True,
    blockxsize=256, blockysize=256, crs="EPSG:3857",
)


def _write_raster(path: str, array: np.ndarray, bounds: tuple, dtype: str,
                  nodata: float | int) -> None:
    """Write a single-band georeferenced raster to local path or gs:// URI."""
    h, w = array.shape
    transform = transform_from_bounds(*bounds, w, h)
    profile = dict(_COG_PROFILE, height=h, width=w, count=1, dtype=dtype,
                   nodata=nodata, transform=transform)
    if str(path).startswith("gs://"):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(array.astype(dtype), 1)
            from google.cloud import storage
            bucket_name, blob_name = str(path)[5:].split("/", 1)
            storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)
        os.replace(tmp_path, path)  # atomic on the same filesystem


def write_probability_tile(path: str, probs: np.ndarray, bounds: tuple) -> None:
    """Write a §9.1 probability tile (float32, NoData -1.0)."""
    _write_raster(path, probs, bounds, "float32", NODATA_PROB)


def write_binary_mask(path: str, mask: np.ndarray, bounds: tuple) -> None:
    """Write a §9.2 binary mask (uint8 0/1, NoData 255)."""
    _write_raster(path, mask, bounds, "uint8", NODATA_MASK)


class Manifest:
    """inference_log.json — progress manifest + run metadata (§8.3, §9.4)."""

    def __init__(self, path: str, run_metadata: dict, checkpoint_every: int = 100):
        self.path = path
        self.checkpoint_every = checkpoint_every
        self.completed: dict[str, str] = {}   # tile_id -> "done" | skip reason
        self.metadata = dict(run_metadata)
        self._since_save = 0
        self._t0 = time.time()
        existing = self._load_existing()
        if existing:
            self.completed = existing.get("tiles", {})
            logger.info("Manifest resume: %d tiles already recorded in %s",
                        len(self.completed), path)

    def _load_existing(self) -> dict | None:
        try:
            if str(self.path).startswith("gs://"):
                import gcsfs
                fs = gcsfs.GCSFileSystem(token="google_default")
                if not fs.exists(self.path[5:]):
                    return None
                with fs.open(self.path[5:], "r") as f:
                    return json.load(f)
            p = Path(self.path)
            return json.loads(p.read_text()) if p.exists() else None
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load existing manifest %s: %s", self.path, exc)
            return None

    def is_done(self, tile_id: str) -> bool:
        return tile_id in self.completed

    def mark(self, tile_id: str, status: str = "done") -> None:
        self.completed[tile_id] = status
        self._since_save += 1
        if self._since_save >= self.checkpoint_every:
            self.save()

    def counts(self) -> dict[str, int]:
        n_done = sum(1 for v in self.completed.values() if v == "done")
        return {
            "n_tiles_processed": n_done,
            "n_tiles_skipped_nodata": sum(1 for v in self.completed.values()
                                          if v == "all_nodata"),
        }

    def save(self) -> None:
        payload = {
            **self.metadata,
            **self.counts(),
            "processing_time_hours": round((time.time() - self._t0) / 3600, 4),
            "tiles": self.completed,
        }
        text = json.dumps(payload, indent=1)
        if str(self.path).startswith("gs://"):
            import gcsfs
            fs = gcsfs.GCSFileSystem(token="google_default")
            with fs.open(self.path[5:], "w") as f:
                f.write(text)
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            tmp = f"{self.path}.tmp"
            Path(tmp).write_text(text)
            os.replace(tmp, self.path)
        self._since_save = 0
