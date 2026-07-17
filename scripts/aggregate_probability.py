"""Streaming probability-canvas aggregation → hotspot density grids (D2).

One pass over the probability_*.tif super-tile shards accumulates, per grid
cell, the **threshold-free expected RTS area** Σ decoded P × geodesic pixel
area (probabilities are temperature-calibrated, training.md §12, so the sum is
an expectation no threshold choice can bias) and the geodesic valid-pixel area,
on two grids at once:

  - a metric EPSG:3857 grid (default 10 km) — regional planning / field targeting
  - a 0.5° WGS84 grid — climate-model-friendly

EPSG:3857 pixel ground area is res² × cos²(lat) (Mercator scale factor 1/cos
per axis); each window row has constant latitude, so the correction is a
row vector. Windows bin into cells with monotonic row/col bins →
np.add.reduceat block sums (no per-pixel index arrays).

Usage:
    python scripts/aggregate_probability.py \
        --shards-dir /outputs/.../probability_cog_shards \
        --out-dir /outputs/.../products_local \
        --candidates /outputs/.../south_rts_candidates.gpkg --workers 32
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil, floor
from pathlib import Path

import numpy as np
import rasterio
from rasterio import windows

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.writer import NODATA_SCALED_U8, SCALE_U8  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

R_MERC = 6378137.0  # EPSG:3857 sphere radius


def _lat_of_y(y: np.ndarray) -> np.ndarray:
    """Latitude (deg) of 3857 northings."""
    return np.degrees(np.arctan(np.sinh(np.asarray(y) / R_MERC)))


def _bin_sums(w: np.ndarray, row_bin: np.ndarray, col_bin: np.ndarray):
    """Block-sum ``w`` into (row_bin, col_bin) cells.

    Bins must be monotonic non-decreasing along each axis (true for any
    north-up window). Returns (urows, ucols, block) with
    block[i, j] = Σ w[row_bin==urows[i], col_bin==ucols[j]].
    """
    cs = np.flatnonzero(np.r_[True, col_bin[1:] != col_bin[:-1]])
    rs = np.flatnonzero(np.r_[True, row_bin[1:] != row_bin[:-1]])
    block = np.add.reduceat(np.add.reduceat(w, cs, axis=1), rs, axis=0)
    return row_bin[rs], col_bin[cs], block


def _shard_contrib(spec: dict):
    """Accumulate one shard into dense (ny, nx) arrays for both grids."""
    g3, gd = spec["grid_3857"], spec["grid_deg"]
    out = [np.zeros((g3["ny"], g3["nx"])), np.zeros((g3["ny"], g3["nx"])),
           np.zeros((gd["ny"], gd["nx"])), np.zeros((gd["ny"], gd["nx"]))]
    with rasterio.open(spec["path"]) as src:
        px_area_3857 = abs(src.res[0] * src.res[1])
        wp = spec["window_px"]
        for r0 in range(0, src.height, wp):
            for c0 in range(0, src.width, wp):
                win = windows.Window(c0, r0, min(wp, src.width - c0),
                                     min(wp, src.height - r0))
                v = src.read(1, window=win)
                valid = v != NODATA_SCALED_U8
                if not valid.any():
                    continue
                t = windows.transform(win, src.transform)
                xs = t.c + (np.arange(v.shape[1]) + 0.5) * t.a
                ys = t.f + (np.arange(v.shape[0]) + 0.5) * t.e
                lat = _lat_of_y(ys)
                pxa = (px_area_3857 * np.cos(np.radians(lat)) ** 2)[:, None]
                p_area = np.where(valid, v / SCALE_U8, 0.0) * pxa
                v_area = valid * pxa
                rb3 = np.clip(((g3["y_top"] - ys) // g3["cell_m"]).astype(int),
                              0, g3["ny"] - 1)
                cb3 = np.clip(((xs - g3["x0"]) // g3["cell_m"]).astype(int),
                              0, g3["nx"] - 1)
                rbd = np.clip(((gd["lat_top"] - lat) // gd["deg"]).astype(int),
                              0, gd["ny"] - 1)
                cbd = np.clip(((np.degrees(xs / R_MERC) - gd["lon0"])
                               // gd["deg"]).astype(int), 0, gd["nx"] - 1)
                for i, (warr, rb, cb) in enumerate(
                        ((p_area, rb3, cb3), (v_area, rb3, cb3),
                         (p_area, rbd, cbd), (v_area, rbd, cbd))):
                    ur, uc, blk = _bin_sums(warr, rb, cb)
                    out[i][np.ix_(ur, uc)] += blk
    return out


def aggregate_shards(shard_paths: list[str], cell_m: float = 10000.0,
                     deg: float = 0.5, window_px: int = 4096,
                     workers: int = 16) -> dict:
    """One streaming pass → per-cell expected/valid geodesic area, both grids.

    Returns ``{"grid_3857": {x0, y_top, cell_m, nx, ny, expected_m2,
    valid_m2}, "grid_deg": {lon0, lat_top, deg, nx, ny, ...}}`` with
    ``expected_m2``/``valid_m2`` as dense (ny, nx) float64 arrays.
    """
    bx0 = by0 = np.inf
    bx1 = by1 = -np.inf
    for p in shard_paths:
        with rasterio.open(p) as src:
            b = src.bounds
        bx0, by0 = min(bx0, b.left), min(by0, b.bottom)
        bx1, by1 = max(bx1, b.right), max(by1, b.top)
    x0 = floor(bx0 / cell_m) * cell_m
    y_top = ceil(by1 / cell_m) * cell_m
    g3 = dict(x0=x0, y_top=y_top, cell_m=cell_m,
              nx=ceil((bx1 - x0) / cell_m), ny=ceil((y_top - by0) / cell_m))
    lon0 = floor(np.degrees(bx0 / R_MERC) / deg) * deg
    lat_top = ceil(_lat_of_y(by1) / deg) * deg
    gd = dict(lon0=lon0, lat_top=lat_top, deg=deg,
              nx=ceil((np.degrees(bx1 / R_MERC) - lon0) / deg),
              ny=ceil((lat_top - _lat_of_y(by0)) / deg))

    sums = [np.zeros((g3["ny"], g3["nx"])), np.zeros((g3["ny"], g3["nx"])),
            np.zeros((gd["ny"], gd["nx"])), np.zeros((gd["ny"], gd["nx"]))]
    specs = [dict(path=p, grid_3857=g3, grid_deg=gd, window_px=window_px)
             for p in shard_paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for contrib in ex.map(_shard_contrib, specs, chunksize=4):
            for s, c in zip(sums, contrib):
                s += c
    logger.info("aggregated %d shards: expected RTS area %.2f km² over "
                "%.0f km² valid", len(shard_paths), sums[0].sum() / 1e6,
                sums[1].sum() / 1e6)
    return {"grid_3857": {**g3, "expected_m2": sums[0], "valid_m2": sums[1]},
            "grid_deg": {**gd, "expected_m2": sums[2], "valid_m2": sums[3]}}


def _cell_bins_3857(g: dict, xs: np.ndarray, ys: np.ndarray):
    """(row, col) cell bins of 3857 points, clipped to the grid."""
    r = np.clip(((g["y_top"] - ys) // g["cell_m"]).astype(int), 0, g["ny"] - 1)
    c = np.clip(((xs - g["x0"]) // g["cell_m"]).astype(int), 0, g["nx"] - 1)
    return r, c


def _cell_bins_deg(g: dict, xs: np.ndarray, ys: np.ndarray):
    r = np.clip(((g["lat_top"] - _lat_of_y(ys)) // g["deg"]).astype(int),
                0, g["ny"] - 1)
    c = np.clip(((np.degrees(xs / R_MERC) - g["lon0"]) // g["deg"]).astype(int),
                0, g["nx"] - 1)
    return r, c


def write_grids(res: dict, out_dir: str | Path, candidates: str | None = None
                ) -> None:
    """Write both grids as cell-polygon GPKGs + expected-area GeoTIFFs.

    GPKG rows are cells with valid coverage: ``expected_rts_m2``,
    ``valid_km2``, and — when ``candidates`` (a GPKG with ``conf_class`` and
    ``area_m2``) is given — per-class ``n_<class>`` / ``rts_m2_<class>``
    from centroid-in-cell assignment. GeoTIFFs carry expected_rts_m2 over the
    full grid (NoData -1 outside coverage).
    """
    import geopandas as gpd
    from shapely.geometry import box

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand = gpd.read_file(candidates) if candidates else None
    if cand is not None:
        cx, cy = cand.geometry.centroid.x.values, cand.geometry.centroid.y.values

    grids = (("grid_3857", f"density_{res['grid_3857']['cell_m'] / 1000:g}km",
              "EPSG:3857", _cell_bins_3857),
             ("grid_deg", f"density_{res['grid_deg']['deg']:g}deg",
              "EPSG:4326", _cell_bins_deg))
    for key, tag, crs, bins_of in grids:
        g = res[key]
        rows, cols = np.nonzero(g["valid_m2"] > 0)
        data = {"expected_rts_m2": g["expected_m2"][rows, cols],
                "valid_km2": g["valid_m2"][rows, cols] / 1e6}
        if cand is not None:
            cr, cc = bins_of(g, cx, cy)
            flat_cells = rows * g["nx"] + cols
            for cls, idx in cand.groupby("conf_class").groups.items():
                flat = cr[idx] * g["nx"] + cc[idx]
                n = np.bincount(flat, minlength=g["ny"] * g["nx"])
                a = np.bincount(flat, weights=cand["area_m2"].values[idx],
                                minlength=g["ny"] * g["nx"])
                data[f"n_{cls}"] = n[flat_cells]
                data[f"rts_m2_{cls}"] = a[flat_cells]
        if key == "grid_3857":
            x = g["x0"] + cols * g["cell_m"]
            y = g["y_top"] - rows * g["cell_m"]
            geoms = [box(xi, yi - g["cell_m"], xi + g["cell_m"], yi)
                     for xi, yi in zip(x, y)]
            origin, step = (g["x0"], g["y_top"]), g["cell_m"]
        else:
            x = g["lon0"] + cols * g["deg"]
            y = g["lat_top"] - rows * g["deg"]
            geoms = [box(xi, yi - g["deg"], xi + g["deg"], yi)
                     for xi, yi in zip(x, y)]
            origin, step = (g["lon0"], g["lat_top"]), g["deg"]
        gpd.GeoDataFrame(data, geometry=geoms, crs=crs).to_file(
            out_dir / f"{tag}.gpkg", driver="GPKG")

        arr = np.where(g["valid_m2"] > 0, g["expected_m2"], -1.0).astype("float32")
        transform = rasterio.transform.from_origin(*origin, step, step)
        with rasterio.open(
                out_dir / f"{tag}_expected_m2.tif", "w", driver="GTiff",
                width=g["nx"], height=g["ny"], count=1, dtype="float32",
                crs=crs, nodata=-1.0, transform=transform,
                compress="deflate") as dst:
            dst.write(arr, 1)
            dst.update_tags(1, STATISTICS_MINIMUM="0",
                            STATISTICS_MAXIMUM=str(float(arr.max())))

        # browse companion: RGBA color-relief on log-percentile breaks — the
        # float tif spans ~7 decades and default-stretches to black; this one
        # is informative with zero styling (hotspots colored, voids transparent)
        pos = g["expected_m2"][g["expected_m2"] > 0]
        rgba = np.zeros((4, g["ny"], g["nx"]), dtype=np.uint8)
        if pos.size:
            breaks = np.percentile(pos, [50, 75, 90, 97, 99.5])
            #            ≥p50      ≥p75      ≥p90      ≥p97     ≥p99.5
            ramp = [(254, 232, 200), (253, 187, 132), (252, 141, 89),
                    (227, 74, 51), (127, 0, 0)]
            lvl = np.digitize(g["expected_m2"], breaks)  # 0..5
            show = (g["valid_m2"] > 0) & (lvl > 0)
            for i, (r, gg, bb) in enumerate(ramp, start=1):
                m = show & (lvl == i)
                rgba[0][m], rgba[1][m], rgba[2][m], rgba[3][m] = r, gg, bb, 255
        with rasterio.open(
                out_dir / f"{tag}_browse.tif", "w", driver="GTiff",
                width=g["nx"], height=g["ny"], count=4, dtype="uint8",
                crs=crs, transform=transform, compress="deflate") as dst:
            dst.write(rgba)
        logger.info("wrote %s (+_expected_m2.tif, +_browse.tif): %d cells, "
                    "Σ expected %.2f km²", out_dir / f"{tag}.gpkg", len(rows),
                    g["expected_m2"].sum() / 1e6)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--candidates", default=None,
                   help="tiered candidates GPKG for per-conf_class cell stats")
    p.add_argument("--cell-m", type=float, default=10000.0)
    p.add_argument("--deg", type=float, default=0.5)
    p.add_argument("--window-px", type=int, default=4096)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    setup_logging()

    shards = sorted(glob.glob(f"{args.shards_dir.rstrip('/')}/probability_*.tif"))
    if not shards:
        raise RuntimeError(f"no probability_*.tif under {args.shards_dir}")
    res = aggregate_shards(shards, cell_m=args.cell_m, deg=args.deg,
                           window_px=args.window_px, workers=args.workers)
    write_grids(res, args.out_dir, candidates=args.candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
