"""Index of PlanetScope basemap quads in a GCS bucket (inference.md §3.1).

The 2025 Global Quarterly basemap is delivered as 4096x4096 uint8 RGBA quads on
the zoom-15 Web Mercator mosaic grid, laid out as:

    gs://<bucket>/global_quarterly/<year>/<quarter>/<x>/<y>/<order_uuid>/
        global_quarterly_<year><quarter>_mosaic/<x>-<y>_quad_file_format.tif

A quad can appear under several order UUIDs (overlapping delivery orders); the
index keeps one path per quad id, preferring the most recently updated object.

Quad bounds are derived purely from the (x, y) grid indices — the mosaic grid is
2048x2048 quads covering the full EPSG:3857 extent, x from the west edge, y from
the south edge. No raster opens are needed to build the index.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# EPSG:3857 world extent and the Planet zoom-15 mosaic grid (quad_size=4096 px,
# resolution 4.777314267823516 m → 19567.88 m per quad; 2048 quads per axis).
WORLD_MIN = -20037508.34
WORLD_MAX = 20037508.34
GRID_N = 2048
QUAD_SIZE_M = (WORLD_MAX - WORLD_MIN) / GRID_N
QUAD_SIZE_PX = 4096
RESOLUTION_M = QUAD_SIZE_M / QUAD_SIZE_PX  # 4.7773142...

QUAD_SUFFIX = "_quad_file_format.tif"
UDM2_SUFFIX = "_ortho_udm2_file_format.tif"


def quad_bounds(x: int, y: int) -> tuple[float, float, float, float]:
    """Projected (minx, miny, maxx, maxy) of mosaic quad (x, y) in EPSG:3857."""
    minx = WORLD_MIN + x * QUAD_SIZE_M
    miny = WORLD_MIN + y * QUAD_SIZE_M
    return minx, miny, minx + QUAD_SIZE_M, miny + QUAD_SIZE_M


def build_quad_index(
    bucket: str = "pdg-planet-data",
    prefix: str = "global_quarterly/2025/q3/",
) -> pd.DataFrame:
    """List the bucket and build the quad index (one row per quad id).

    Returns a DataFrame with columns:
        quad_id, x, y, gcs_path, udm2_path, minx, miny, maxx, maxy

    Duplicate quad ids (several order UUIDs) keep the most recently updated
    quad object.
    """
    from google.cloud import storage  # deferred: not needed for local index use

    client = storage.Client()
    rows: dict[str, dict] = {}
    n_objects = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
        n_objects += 1
        name = blob.name
        if not name.endswith(QUAD_SUFFIX):
            continue
        quad_id = name.rsplit("/", 1)[-1][: -len(QUAD_SUFFIX)]
        prev = rows.get(quad_id)
        if prev is None or blob.updated > prev["_updated"]:
            x_str, y_str = quad_id.split("-")
            minx, miny, maxx, maxy = quad_bounds(int(x_str), int(y_str))
            rows[quad_id] = {
                "quad_id": quad_id,
                "x": int(x_str),
                "y": int(y_str),
                "gcs_path": f"gs://{bucket}/{name}",
                # UDM2 sits in the same order dir with the same stem.
                "udm2_path": f"gs://{bucket}/{name[: -len(QUAD_SUFFIX)]}{UDM2_SUFFIX}",
                "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
                "_updated": blob.updated,
            }

    index = pd.DataFrame(sorted(rows.values(), key=lambda r: (r["x"], r["y"])))
    if index.empty:
        raise RuntimeError(f"No quads found under gs://{bucket}/{prefix}")
    index = index.drop(columns=["_updated"])
    logger.info("Quad index: %d quads from %d objects under gs://%s/%s",
                len(index), n_objects, bucket, prefix)
    return index


def load_quad_index(path: str | Path) -> pd.DataFrame:
    """Load a quad index CSV written by scripts/build_quad_index.py."""
    index = pd.read_csv(path)
    required = {"quad_id", "x", "y", "gcs_path", "minx", "miny", "maxx", "maxy"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"{path}: quad index missing columns {sorted(missing)}")
    return index
