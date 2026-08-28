"""Index of the bulk Sentinel-2 composite COGs in a GCS bucket (inference.md §5; data/s2_rgb_data.md).

The NDVI EXTRA channel at inference is derived on the fly by windowing these
composites to each Planet tile bbox (the per-tile-materialization pivot, diary
2026-06-24) — mirroring how RGB is mosaicked from Planet quads. This index is the
S2-side analogue of ``inference.quad_index``: one row per composite cell with its
EPSG:3857 bounds + GCS path.

Unlike the Planet quad grid, the composite cells are 1°×3° lat/lon boxes clipped
to the domain then reprojected to EPSG:3857 (scripts/export_s2_composites.py), so
their projected bounds are not derivable from the cell id — we read each COG's
bounds once (the expensive listing/opens run here; everything downstream reads the
CSV).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_s2_index(
    bucket: str = "rts-arctic-usw1",
    prefix: str = "S2_RGB/2025_south",
) -> pd.DataFrame:
    """List the bucket and build the S2 composite index (one row per COG).

    Returns a DataFrame with columns: cell_id, gcs_path, minx, miny, maxx, maxy
    (EPSG:3857). Each COG's bounds are read from its georeferencing via rasterio
    (/vsigs/), so this needs GCS read access + GOOGLE_CLOUD_PROJECT on bare ADC.
    """
    from google.cloud import storage  # deferred: not needed for local index use
    import rasterio

    client = storage.Client()
    rows: list[dict] = []
    n_objects = 0
    for blob in client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"):
        n_objects += 1
        if not blob.name.endswith(".tif"):
            continue
        gcs_path = f"gs://{bucket}/{blob.name}"
        with rasterio.open(f"/vsigs/{bucket}/{blob.name}") as src:
            b = src.bounds
            if src.crs is None or src.crs.to_epsg() != 3857:
                raise ValueError(f"{gcs_path}: CRS {src.crs} is not EPSG:3857")
        rows.append({
            "cell_id": Path(blob.name).stem,
            "gcs_path": gcs_path,
            "minx": b.left, "miny": b.bottom, "maxx": b.right, "maxy": b.top,
        })

    index = pd.DataFrame(sorted(rows, key=lambda r: r["cell_id"]))
    if index.empty:
        raise RuntimeError(f"No S2 composite COGs found under gs://{bucket}/{prefix}")
    logger.info("S2 composite index: %d cells from %d objects under gs://%s/%s",
                len(index), n_objects, bucket, prefix)
    return index


def load_s2_index(path: str | Path) -> pd.DataFrame:
    """Load an S2 composite index CSV written by scripts/build_s2_index.py."""
    index = pd.read_csv(path)
    required = {"cell_id", "gcs_path", "minx", "miny", "maxx", "maxy"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"{path}: S2 index missing columns {sorted(missing)}")
    return index
