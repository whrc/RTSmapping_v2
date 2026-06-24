"""Bulk Sentinel-2 RGB+NIR composite export (doc s2_extra_data_prep.md §3).

Exports the cloud-masked Jul–Sep median S2 surface-reflectance composite — the
SSoT recipe in ``data/extra_channels.s2_sr_composite`` — over an inference/training
domain, as deflate-compressed GeoTIFFs in **EPSG:3857**, one per land grid cell,
via Earth Engine ``Export.image.toCloudStorage``. This is the imagery for the
pure-S2 RGB RTS model (2024 training / 2025 inference, North + South).

Grid: a clean ``dlat`` × ``dlon`` (default 1°×3°) lat/lon grid, kept only where it
intersects the domain polygon (the domain — permafrost ∩ Arctic-boreal ∩ ArcticDEM,
optionally ∩ Planet coverage — is already land-only, subsuming an LSIB land filter).
Each cell's composite is clipped to cell ∩ domain so ocean / out-of-domain pixels are
masked. CRS is EPSG:3857 everywhere (project standard; for the 74–84°N North this is a
deliberate ~5–6× Web-Mercator distortion sign-off, doc §6.6).

Resumable: skips cells whose output object already exists under the GCS prefix and
backs off when the GEE task queue is full ("Too many tasks").

Usage (inside rts-train Docker, ADC mounted):
  python scripts/export_s2_composites.py --year 2025 \
     --domain domain/circumpolar_south_domain.geojson \
     --bucket woodwell-rts-inference-arts-south --prefix S2_RGB/2025_south \
     [--bands B4,B3,B2,B8] [--dlat 1 --dlon 3] [--scale 10] [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `data.*`

from data.extra_channels import init_ee, s2_sr_composite  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_s2")

DEFAULT_BANDS = ["B4", "B3", "B2", "B8"]   # RGB + NIR (NIR enables NDVI)
MAX_ACTIVE_TASKS = 2000                    # GEE per-user running+pending task ceiling


def cell_id(lon0: float, lat0: float) -> str:
    """Deterministic cell name from its SW corner (0.1° precision, sign-safe)."""
    def tag(v: float, pos: str, neg: str) -> str:
        return f"{pos if v >= 0 else neg}{abs(round(v * 10)):04d}"
    return f"{tag(lon0, 'E', 'W')}_{tag(lat0, 'N', 'S')}"


def latlon_grid(bbox: tuple[float, float, float, float], dlat: float, dlon: float):
    """SW-anchored lat/lon cells (lon0, lat0, lon1, lat1) covering `bbox` (lon/lat deg).

    Cells are aligned to multiples of (dlon, dlat) from the origin so the same patch
    of ground always gets the same cell id across runs/years.
    """
    minx, miny, maxx, maxy = bbox
    import math
    c0 = math.floor(minx / dlon); c1 = math.ceil(maxx / dlon)
    r0 = math.floor(miny / dlat); r1 = math.ceil(maxy / dlat)
    cells = []
    for c in range(c0, c1):
        for r in range(r0, r1):
            cells.append((c * dlon, r * dlat, (c + 1) * dlon, (r + 1) * dlat))
    return cells


def domain_cells(domain: gpd.GeoDataFrame, dlat: float, dlon: float):
    """Cells intersecting the domain → list of (cell_id, (lon0,lat0,lon1,lat1), clip_geom).

    `domain` is reprojected to EPSG:4326; `clip_geom` is the cell ∩ domain polygon
    (WGS84), kept small so it converts cheaply to an ee.Geometry per cell.
    """
    dom = domain.to_crs("EPSG:4326")
    union = dom.geometry.union_all() if hasattr(dom.geometry, "union_all") else dom.geometry.unary_union
    out = []
    for lon0, lat0, lon1, lat1 in latlon_grid(union.bounds, dlat, dlon):
        cell = box(lon0, lat0, lon1, lat1)
        if not union.intersects(cell):
            continue
        clip = union.intersection(cell)
        if clip.is_empty:
            continue
        out.append((cell_id(lon0, lat0), (lon0, lat0, lon1, lat1), clip))
    return out


def _existing_cell_ids(bucket: str, prefix: str) -> set[str]:
    """Cell ids already exported under gs://bucket/prefix (resume support)."""
    from google.cloud import storage
    client = storage.Client()
    ids = set()
    for blob in client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"):
        name = Path(blob.name).stem
        if name.endswith(".tif"):
            name = name[:-4]
        ids.add(name)
    return ids


def _wait_for_task_slot(ee, poll: int = 30) -> None:
    """Block while the GEE task queue is at the per-user ceiling (avoids 'Too many tasks')."""
    while True:
        active = [t for t in ee.data.listOperations()
                  if t.get("metadata", {}).get("state") in ("PENDING", "RUNNING")]
        if len(active) < MAX_ACTIVE_TASKS:
            return
        logger.info("  GEE task queue full (%d active); backing off %ds", len(active), poll)
        time.sleep(poll)


def export_cell(ee, cid, clip_geom, year, bands, bucket, prefix, scale):
    """Launch one Export.image.toCloudStorage task for a cell; returns the task."""
    from shapely.geometry import mapping
    region = ee.Geometry(mapping(clip_geom), proj="EPSG:4326", geodesic=False)
    img = s2_sr_composite(region, year).select(bands).clip(region).toFloat()
    task = ee.batch.Export.image.toCloudStorage(
        image=img, description=f"s2_{prefix.replace('/', '_')}_{cid}"[:100],
        bucket=bucket, fileNamePrefix=f"{prefix.rstrip('/')}/{cid}",
        region=region, scale=scale, crs="EPSG:3857", maxPixels=int(1e13),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--domain", required=True, type=Path, help="domain geojson (any CRS)")
    ap.add_argument("--bucket", required=True, help="output GCS bucket (no gs://)")
    ap.add_argument("--prefix", required=True, help="output key prefix, e.g. S2_RGB/2025_south")
    ap.add_argument("--bands", default=",".join(DEFAULT_BANDS),
                    help="comma list of S2 SR bands (default RGB+NIR: B4,B3,B2,B8)")
    ap.add_argument("--dlat", type=float, default=1.0)
    ap.add_argument("--dlon", type=float, default=3.0)
    ap.add_argument("--scale", type=int, default=10, help="export resolution m (S2 native 10)")
    ap.add_argument("--project", default="pdg-project-406720")
    ap.add_argument("--limit", type=int, default=0, help="cap cells (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="list cells, launch nothing")
    args = ap.parse_args()

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    cells = domain_cells(gpd.read_file(args.domain), args.dlat, args.dlon)
    logger.info("domain %s -> %d intersecting %g°x%g° cells", args.domain.name,
                len(cells), args.dlat, args.dlon)
    if args.limit:
        cells = cells[: args.limit]

    if args.dry_run:
        for cid, bbox, _ in cells:
            logger.info("  cell %s bbox=%s", cid, tuple(round(v, 3) for v in bbox))
        logger.info("DRY RUN: %d cells, bands=%s scale=%dm crs=EPSG:3857", len(cells), bands, args.scale)
        return 0

    import ee
    init_ee(args.project)
    done = _existing_cell_ids(args.bucket, args.prefix)
    todo = [c for c in cells if c[0] not in done]
    logger.info("year=%d bands=%s: %d/%d cells to export (%d already in gs://%s/%s)",
                args.year, bands, len(todo), len(cells), len(done), args.bucket, args.prefix)

    launched = 0
    for cid, _bbox, clip in todo:
        _wait_for_task_slot(ee)
        export_cell(ee, cid, clip, args.year, bands, args.bucket, args.prefix, args.scale)
        launched += 1
        if launched % 50 == 0:
            logger.info("  launched %d/%d", launched, len(todo))
    logger.info("DONE: launched %d export tasks -> gs://%s/%s (monitor with `earthengine task list`)",
                launched, args.bucket, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
