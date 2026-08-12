"""Parallel region vectorization (post-inference.md §9.3 at region scale).

`vectorize_predictions.py` polygonizes one merged mask that must fit in RAM (the
GDAL polygonize scan is single-threaded); a full region's mask does not fit
(South ≈ 7.7 TB). This vectorizes the per-block mask COGs written by
`assemble_region.py` **in parallel**, then stitches them:

  - a polygon fully interior to its block is final (min_blob filter applied in
    the worker — cheap, and it can never grow by a seam merge);
  - a polygon touching a block edge is deferred; all edge polygons are then
    dissolved (`unary_union`) so a slump split across a block seam becomes one
    polygon, and min_blob is applied **after** the dissolve (a slump split across
    a seam can be sub-min_blob in each half but ≥ min_blob combined).

min_blob uses the exact pixel count `area_3857 / pixel_area` (polygonized edges
are pixel-aligned, and the projection distortion cancels since both areas are in
the same EPSG:3857 grid). Per-polygon stats match `vectorize_predictions`.

Usage:
    python scripts/vectorize_region.py \
        --blocks-dir /local/banks/out/blocks \
        --prob /local/banks/out/probability.tif \
        --tile-list banks_tiles.csv --package pkg_dir \
        --output banks_rts.gpkg --workers 32
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil, floor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features, windows
from shapely import wkb as shp_wkb
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.quad_index import WORLD_MIN  # noqa: E402
from inference.writer import NODATA_SCALED_U8, SCALE_U8  # noqa: E402
from scripts.vectorize_predictions import _GEOD, _TO_WGS84  # noqa: E402
from utils.config import load_config, vectorize_min_blob_px  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

R_MERC = 6378137.0

# fork-inherited state for the parallel _record pass (COW — set in the parent
# right before the pool is created; pickling the 41.5M-tile code array per
# task would dwarf the work itself)
_REC: dict = {}


def tile_codes_from_list(tile_ids: pd.Series) -> np.ndarray:
    """Sorted int64 codes (col<<32 | row) of the t{col}_{row} tile ids."""
    cr = tile_ids.str.slice(1).str.split("_", expand=True).astype(np.int64)
    return np.sort((cr[0].values << 32) | cr[1].values)


def _tile_join_state(tiles: pd.DataFrame) -> dict:
    """Arithmetic-join state from a conforming t{col}_{row} stride-grid list;
    non-conforming lists (tests, ad-hoc AOIs) fall back to the bbox scan."""
    ids = tiles["tile_id"].astype(str)
    conforming = ids.str.match(r"^t\d+_\d+$").all()
    nz = None
    if conforming:
        cols = ids.str.slice(1).str.split("_").str[0].astype(np.int64)
        nzi = np.nonzero(cols.values)[0]
        nz = nzi[0] if len(nzi) else None
    if not conforming or nz is None:
        return {"tiles_df": tiles}
    t = tiles.iloc[nz]
    return {"codes": tile_codes_from_list(ids),
            "stride_m": (t["minx"] - WORLD_MIN) / int(cols.values[nz]),
            "tile_m": float(t["maxx"] - t["minx"])}


def tiles_for_bounds(bounds: tuple, codes: np.ndarray, stride_m: float,
                     tile_m: float) -> list[str]:
    """Tile ids intersecting ``bounds`` — arithmetic on the stride grid
    (generate_tile_grid convention), then membership against ``codes``.
    Replaces the per-polygon scan of the 41.5M-row tile list."""
    bminx, bminy, bmaxx, bmaxy = bounds
    c0 = floor((bminx - WORLD_MIN - tile_m) / stride_m) + 1
    c1 = ceil((bmaxx - WORLD_MIN) / stride_m) - 1
    r0 = floor((bminy - WORLD_MIN - tile_m) / stride_m) + 1
    r1 = ceil((bmaxy - WORLD_MIN) / stride_m) - 1
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1, dtype=np.int64),
                         np.arange(r0, r1 + 1, dtype=np.int64))
    cand = (cc.ravel() << 32) | rr.ravel()
    idx = np.searchsorted(codes, cand)
    idx = np.clip(idx, 0, len(codes) - 1)
    hit = cand[codes[idx] == cand] if len(codes) else cand[:0]
    return [f"t{c >> 32}_{c & 0xFFFFFFFF}" for c in hit]


def _polygonize_block(spec: dict):
    """Polygonize one block/window → (interior WKB list, edge-touching WKB list).

    Two modes: mask (``threshold`` absent — block is a 0/1/255 mask, take ==1)
    and threshold (``threshold`` set — block is a probability window; binarize
    at the decoded threshold, scaled_uint8 NoData 255 excluded explicitly since
    it is numerically above any scaled threshold). Interior polygons are
    min_blob-filtered here; edge-touching ones are deferred to the cross-block
    dissolve. Returned as WKB so they pickle across the pool.
    """
    min_blob_px = spec["min_blob_px"]
    with rasterio.open(spec["path"]) as src:
        if "window" in spec:
            win = windows.Window(*spec["window"])
            m = src.read(1, window=win)
            transform = windows.transform(win, src.transform)
            left, bottom, right, top = windows.bounds(win, src.transform)
        else:
            m = src.read(1)
            transform = src.transform
            left, bottom, right, top = src.bounds
        xres, yres = src.res
        dtype = src.dtypes[0]
    thr = spec.get("threshold")
    if thr is None:
        binm = m == 1
    elif dtype == "uint8":  # scaled_uint8: prob×250, NoData 255
        binm = (m != NODATA_SCALED_U8) & (m >= int(round(thr * SCALE_U8)))
    else:  # float32: NoData -1 is below any positive threshold
        binm = m >= thr
    if not binm.any():
        return [], []
    px_area = xres * yres
    tol = xres  # ~1 px: a polygon reaching the block boundary may continue next door
    interior, edge = [], []
    for gj, _ in features.shapes(binm.astype("uint8"), mask=binm, transform=transform):
        geom = shape(gj)
        b = geom.bounds
        on_edge = (b[0] <= left + tol or b[1] <= bottom + tol
                   or b[2] >= right - tol or b[3] >= top - tol)
        if on_edge:
            edge.append(geom.wkb)
        elif geom.area >= min_blob_px * px_area:
            interior.append(geom.wkb)
    return interior, edge


def _record(rts_id: int, geom, prb, scales: list[float]) -> dict:
    """Per-polygon §9.3 attributes (mirrors vectorize_predictions.vectorize).

    tile_ids come from the arithmetic stride-grid join against the
    fork-inherited ``_REC`` state (codes/stride_m/tile_m)."""
    transform = prb.transform
    fwin = windows.from_bounds(*geom.bounds, transform=transform)
    c0, r0 = floor(fwin.col_off), floor(fwin.row_off)
    c1, r1 = ceil(fwin.col_off + fwin.width), ceil(fwin.row_off + fwin.height)
    win = windows.Window(c0, r0, c1 - c0, r1 - r0)
    local = features.rasterize([mapping(geom)], out_shape=(r1 - r0, c1 - c0),
                               transform=windows.transform(win, transform))
    pvals = prb.read(1, window=win)[local == 1]
    if prb.dtypes[0] == "uint8":  # scaled_uint8 product: prob×250, NoData 255
        pvals = pvals[pvals != NODATA_SCALED_U8].astype(np.float32) / SCALE_U8
    else:  # float32 product: prob in [0,1], NoData -1
        pvals = pvals[pvals >= 0]
    geom_wgs = gpd.GeoSeries([geom], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
    area, perim = _GEOD.geometry_area_perimeter(geom_wgs)
    lon, lat = _TO_WGS84.transform(geom.centroid.x, geom.centroid.y)
    if "codes" in _REC:
        tids = tiles_for_bounds(geom.bounds, _REC["codes"], _REC["stride_m"],
                                _REC["tile_m"])
    else:  # non-conforming tile list: bbox scan (small lists only)
        t, b = _REC["tiles_df"], geom.bounds
        tids = list(t[(t["minx"] < b[2]) & (t["maxx"] > b[0])
                      & (t["miny"] < b[3]) & (t["maxy"] > b[1])]["tile_id"]
                    .astype(str))

    def _area_frac(t: float) -> float:
        # geodesic area × in-polygon fraction of pixels ≥ t (keeps the multi-
        # threshold areas geodesically consistent with area_m2)
        return abs(area) * float((pvals >= t).mean()) if pvals.size else float("nan")

    return {
        "rts_id": rts_id,
        "area_m2": abs(area),
        "perimeter_m": perim,
        "centroid_lat": lat,
        "centroid_lon": lon,
        "mean_prob": float(pvals.mean()) if pvals.size else float("nan"),
        "max_prob": float(pvals.max()) if pvals.size else float("nan"),
        "area_m2_t45": _area_frac(0.45),
        "area_m2_t65": _area_frac(0.65),
        "area_m2_t80": _area_frac(0.80),
        "detection_scale": ",".join(str(s) for s in scales),
        "tile_ids": ",".join(tids),
    }


def _record_batch(spec: tuple[str, list[float], int, list[bytes]]) -> list[dict]:
    """Worker: stats for a batch of polygons (opens its own prob handle)."""
    prob_path, scales, start_id, wkbs = spec
    with rasterio.open(prob_path) as prb:
        return [_record(start_id + i, shp_wkb.loads(w), prb, scales)
                for i, w in enumerate(wkbs)]


def dissolve_edges(edge_wkb: list[bytes], min_blob_px: int,
                   px_area: float) -> list:
    """Union edge-touching polygons across seams, then min_blob-filter."""
    if not edge_wkb:
        return []
    merged = unary_union([shp_wkb.loads(w) for w in edge_wkb])
    polys = list(merged.geoms) if merged.geom_type.startswith("Multi") else [merged]
    return [g for g in polys if g.area >= min_blob_px * px_area]


def vectorize_region(blocks_dir: str, prob_path: str, tile_list: str,
                     scales: list[float], min_blob_px: int,
                     workers: int, threshold: float | None = None,
                     window_px: int = 8192,
                     min_area_m2: float | None = None) -> gpd.GeoDataFrame:
    """Parallel polygonize of block masks → dissolved, size-filtered polygons.

    Two mutually exclusive size floors, and they are not interchangeable:

    * ``min_area_m2`` (not None) — the **shipped** rule: a geodesic MMU in m²,
      constant on the ground at every latitude. ``0`` means no MMU, leaving only
      the 2-px technical floor. ``min_blob_px`` is *overwritten* in this mode.
    * ``min_blob_px`` (when ``min_area_m2`` is None) — the **superseded** pixel
      floor from the package (``vectorize_min_blob_px``). A pixel count in 3857,
      so its ground area slides ~7x from 50°N to 76°N.

    With ``threshold`` set, polygonizes the probability super-tile COGs
    (``probability_*.tif``, the delivered product shards) at that decoded
    probability instead of the pre-thresholded ``mask_*.tif`` blocks — each COG
    is processed in ``window_px``² windows so a full super-tile never has to fit
    in RAM; the existing edge-dissolve stitches polygons across window and COG
    seams alike.
    """
    if min_area_m2 is not None:
        # geodesic MMU: constant ground-area floor. Cheap pixel prefilter at
        # the count a min_area_m2 object would have at the canvas's most
        # equatorward row (3857 px ground area = res²·cos²lat is largest
        # there, so this never drops anything the exact filter would keep);
        # exact geodesic filter applied on area_m2 after _record. Technical
        # floor 2 px kills single-pixel noise even at min_area_m2=0.
        with rasterio.open(prob_path) as s:
            b = s.bounds
            pa3857 = abs(s.res[0] * s.res[1])
        lat_min = min(abs(np.degrees(np.arctan(np.sinh(b.bottom / R_MERC)))),
                      abs(np.degrees(np.arctan(np.sinh(b.top / R_MERC)))))
        max_geo_px = pa3857 * np.cos(np.radians(lat_min)) ** 2
        min_blob_px = max(2, int(min_area_m2 / max_geo_px))
        logger.info("geodesic MMU %.0f m² → pixel prefilter %d px "
                    "(most-equatorward lat %.2f°)", min_area_m2, min_blob_px,
                    lat_min)

    if threshold is None:
        mask_blocks = sorted(glob.glob(f"{blocks_dir.rstrip('/')}/mask_*.tif"))
        if not mask_blocks:
            raise RuntimeError(f"no mask_*.tif under {blocks_dir}")
        specs = [dict(path=mb, min_blob_px=min_blob_px) for mb in mask_blocks]
    else:
        prob_blocks = sorted(glob.glob(f"{blocks_dir.rstrip('/')}/probability_*.tif"))
        if not prob_blocks:
            raise RuntimeError(f"no probability_*.tif under {blocks_dir}")
        specs = []
        for pb in prob_blocks:
            with rasterio.open(pb) as src:
                w, h = src.width, src.height
            for r0 in range(0, h, window_px):
                for c0 in range(0, w, window_px):
                    specs.append(dict(
                        path=pb, min_blob_px=min_blob_px, threshold=threshold,
                        window=(c0, r0, min(window_px, w - c0),
                                min(window_px, h - r0))))

    interior_wkb, edge_wkb = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for inter, edge in ex.map(_polygonize_block, specs):
            interior_wkb += inter
            edge_wkb += edge
    logger.info("polygonized %d blocks: %d interior + %d edge-touching",
                len(specs), len(interior_wkb), len(edge_wkb))

    with rasterio.open(prob_path) as prb:
        px_area = abs(prb.res[0] * prb.res[1])
    dissolved = dissolve_edges(edge_wkb, min_blob_px, px_area)
    final = [shp_wkb.loads(w) for w in interior_wkb] + dissolved
    logger.info("%d final polygons (%d interior + %d dissolved-edge)",
                len(final), len(interior_wkb), len(dissolved))

    # arithmetic tile join state — derived from the tile list itself (SSoT for
    # what exists), grid geometry from any row's id + bounds; inherited by the
    # _record workers via fork (never pickled per task)
    _REC.clear()
    _REC.update(_tile_join_state(pd.read_csv(tile_list)))

    batches = [(prob_path, scales, i + 1, [g.wkb for g in final[i:i + 200]])
               for i in range(0, len(final), 200)]
    if workers > 1 and len(batches) > 1:
        records: list[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for recs in ex.map(_record_batch, batches):
                records += recs
    else:
        records = [r for b in batches for r in _record_batch(b)]

    gdf = gpd.GeoDataFrame(records, geometry=final, crs="EPSG:3857")
    if min_area_m2 is not None:
        n0 = len(gdf)
        gdf = gdf[gdf["area_m2"] >= min_area_m2].reset_index(drop=True)
        gdf["rts_id"] = range(1, len(gdf) + 1)
        logger.info("exact geodesic MMU filter: %d → %d polygons", n0, len(gdf))
    logger.info("Vectorized %d polygons, total %.2f km2", len(gdf),
                gdf["area_m2"].sum() / 1e6 if len(gdf) else 0.0)
    return gdf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks-dir", required=True,
                   help="dir of assemble_region.py mask_*.tif block COGs "
                        "(or probability_*.tif shards with --threshold)")
    p.add_argument("--prob", required=True, help="region probability COG")
    p.add_argument("--tile-list", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--threshold", type=float, default=None,
                   help="polygonize probability_*.tif at this decoded prob "
                        "instead of the pre-thresholded mask blocks")
    p.add_argument("--window-px", type=int, default=8192,
                   help="processing window size for --threshold mode")
    p.add_argument("--min-area-m2", type=float, default=0.0,
                   help="geodesic MMU in m², latitude-constant. DEFAULT 0 = the "
                        "SHIPPED rule: keep everything above the 2-px technical "
                        "floor (no minimum mapping unit)")
    p.add_argument("--legacy-min-blob-px", action="store_true",
                   help="opt in to the SUPERSEDED pixel floor from the package's "
                        "vectorize_min_blob_px (2000 px) instead of the geodesic "
                        "MMU. Reproduces the legacy south_rts.gpkg; a pixel floor "
                        "slides ~7x with latitude. Ignores --min-area-m2.")
    args = p.parse_args()
    setup_logging()

    dep_cfg = load_config(f"{str(args.package).rstrip('/')}/deployment_config.yaml")
    if args.legacy_min_blob_px:
        logger.warning("--legacy-min-blob-px: using the SUPERSEDED pixel floor; this "
                       "reproduces south_rts.gpkg, NOT the shipped inventory")
    gdf = vectorize_region(args.blocks_dir, args.prob, args.tile_list,
                           scales=dep_cfg.get("scales", [1.0]),
                           min_blob_px=vectorize_min_blob_px(dep_cfg),
                           workers=args.workers, threshold=args.threshold,
                           window_px=args.window_px,
                           min_area_m2=None if args.legacy_min_blob_px else args.min_area_m2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.output, driver="GPKG")
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
