"""Generate the canonical 8-band EXTRA stack per tile (data/data.md §9).

The data-team handoff for inference EXTRA: same script, same `data/extra_channels`
module, parameterized by --year + footprint source + output, so 2024 training and
2025 inference tiles are produced identically (CLAUDE Rule 3).

For each tile it obtains the EPSG:3857 bounds (from the metadata CSV's bbox columns
if present, else from the matching PLANET-RGB GeoTIFF), queries Earth Engine
(Sentinel-2 / Satellite Embedding) on a co-registered grid, and writes/updates
EXTRA/<tile_id>.tif (8-band float32, raw values, NaN for not-yet-generated bands).
Phased + resumable via --groups:
  --groups s2  -> fill bands 0,1,6,7 (NDVI,NBR,TCB,TCW)
  --groups se  -> fill bands 2,3,4,5 (SE_PCA x3, SE_PROTO)   [SE path, later]
  --groups all -> both
A tile is skipped if the requested group's bands are already non-NaN.

Footprint source (doc s2_extra_data_prep.md §6.5): the 2025 inference tile-grid CSV
(`tile_id,minx,miny,maxx,maxy`) already carries per-tile bounds, so for inference we
feed bounds straight from the CSV and `--rgb-dir` is not needed; for the 2024 training
set (a `Tile_ID` CSV without bbox cols) pass `--rgb-dir` to read bounds from the tiles.

Run inside rts-train Docker with earthengine-api installed + ADC mounted.
Usage:
  # 2024 training (bounds from RGB tiles):
  python scripts/generate_extra_tiles.py --groups s2 --year 2024 \
     --metadata /outputs/v1.0/data_local/metadata.csv \
     --rgb-dir  /outputs/v1.0/data_local/PLANET-RGB \
     --out-dir  /outputs/v1.0/data_local/EXTRA [--workers 16] [--limit N]
  # 2025 inference (bounds from the tile-grid CSV, no --rgb-dir):
  python scripts/generate_extra_tiles.py --groups s2 --year 2025 \
     --metadata /outputs/inference/tiles_2025q3_domain_full.csv \
     --out-dir  gs_or_local/EXTRA_2025 [--workers 16] [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds as _affine_from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `data.*`

from data.extra_channels import (  # noqa: E402
    N_EXTRA_BANDS, S2_BAND_IDX, SE_BAND_IDX, init_ee, load_se_artifacts,
    s2_bands, se_bands, tile_grid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gen_extra")

GROUP_IDX = {"s2": S2_BAND_IDX, "se": SE_BAND_IDX, "all": S2_BAND_IDX + SE_BAND_IDX}
_BBOX_COLS = ("minx", "miny", "maxx", "maxy")


def _load_ids_and_bounds(
    metadata: Path,
) -> tuple[list[str], dict[str, tuple[float, float, float, float]] | None]:
    """Tile ids from `metadata`, plus a {id: bounds} map iff it carries bbox cols.

    Accepts the training `Tile_ID` schema or the inference tile-grid
    `tile_id,minx,miny,maxx,maxy` schema (data-team handoff, doc §6.5). When bbox
    columns are present, bounds come from the CSV and no RGB tiles are needed.
    """
    df = pd.read_csv(metadata)
    id_col = "tile_id" if "tile_id" in df.columns else "Tile_ID"
    ids = df[id_col].astype(str).tolist()
    bounds = None
    if all(c in df.columns for c in _BBOX_COLS):
        arr = df[list(_BBOX_COLS)].to_numpy(dtype="float64")
        bounds = {t: (float(a), float(b), float(c), float(d))
                  for t, (a, b, c, d) in zip(ids, arr)}
    return ids, bounds


def _profile_from_bounds(bounds: tuple[float, float, float, float],
                         size_px: int = 512) -> dict:
    """Rasterio profile for an 8-band EPSG:3857 EXTRA tile co-registered to `bounds`."""
    minx, miny, maxx, maxy = bounds
    return {
        "driver": "GTiff", "dtype": "float32", "count": N_EXTRA_BANDS,
        "width": size_px, "height": size_px, "crs": "EPSG:3857",
        "transform": _affine_from_bounds(minx, miny, maxx, maxy, size_px, size_px),
        "nodata": float("nan"), "compress": "deflate",
    }


def _needs_work(path: Path, band_idx: list[int]) -> bool:
    """True if any requested band is missing/all-NaN (resumable)."""
    if not path.exists():
        return True
    with rasterio.open(path) as ds:
        if ds.count != N_EXTRA_BANDS:
            return True
        for b in band_idx:
            if np.isnan(ds.read(b + 1)).all():
                return True
    return False


def _write_bands(path: Path, profile: dict, bands: dict[int, np.ndarray]) -> None:
    """Create (8-band NaN, using `profile`) if absent, then write band_index→array."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = profile["height"], profile["width"]
        with rasterio.open(path, "w", **profile) as dst:
            nan = np.full((h, w), np.nan, dtype="float32")
            for b in range(1, N_EXTRA_BANDS + 1):
                dst.write(nan, b)
    with rasterio.open(path, "r+") as dst:
        for idx, arr in bands.items():
            dst.write(arr.astype("float32"), idx + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", choices=["s2", "se", "all"], required=True)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--rgb-dir", type=Path, default=None,
                    help="dir of PLANET-RGB tiles; only needed when --metadata has no bbox cols")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--project", default="pdg-project-406720")
    ap.add_argument("--se-artifacts", type=Path, default=None,
                    help="se_artifacts.npz (required for --groups se/all); from build_se_artifacts.py")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap tiles (smoke)")
    args = ap.parse_args()

    se_art = None
    if args.groups in ("se", "all"):
        if not args.se_artifacts or not args.se_artifacts.exists():
            raise SystemExit("--se-artifacts <se_artifacts.npz> is required for groups se/all "
                             "(build it with scripts/build_se_artifacts.py)")
        se_art = load_se_artifacts(args.se_artifacts)

    ids, bounds_map = _load_ids_and_bounds(args.metadata)
    if bounds_map is None and args.rgb_dir is None:
        raise SystemExit("--metadata has no bbox columns (minx,miny,maxx,maxy); "
                         "pass --rgb-dir to read per-tile bounds from RGB GeoTIFFs")
    if args.limit:
        ids = ids[: args.limit]
    init_ee(args.project)
    band_idx = GROUP_IDX[args.groups]
    todo = [t for t in ids if _needs_work(args.out_dir / f"{t}.tif", band_idx)]
    logger.info("groups=%s year=%d source=%s: %d/%d tiles to generate", args.groups,
                args.year, "csv-bbox" if bounds_map else "rgb-dir", len(todo), len(ids))

    def one(tid: str) -> str:
        if bounds_map is not None:
            bounds = bounds_map[tid]
            profile = _profile_from_bounds(bounds)
        else:
            rgb = args.rgb_dir / f"{tid}.tif"
            if not rgb.exists():
                return "norgb"
            with rasterio.open(rgb) as src:
                bounds = tuple(src.bounds)
                profile = dict(src.profile)
            profile.update(count=N_EXTRA_BANDS, dtype="float32",
                           nodata=float("nan"), compress="deflate")
        grid = tile_grid(bounds)
        bands: dict[int, np.ndarray] = {}
        if args.groups in ("s2", "all"):
            bands.update(s2_bands(bounds, grid, args.year))         # {0,1,6,7}
        if args.groups in ("se", "all"):
            bands.update(se_bands(bounds, grid, args.year, se_art))  # {2,3,4,5}
        _write_bands(args.out_dir / f"{tid}.tif", profile, bands)
        return "ok"

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result(); ok += r == "ok"; skip += r == "norgb"
            except Exception as e:  # noqa: BLE001
                fail += 1
                if fail <= 10:
                    logger.error("FAIL %s: %r", futs[fut], repr(e)[:200])
            if i % 500 == 0:
                logger.info("  %d/%d (ok=%d norgb=%d fail=%d)", i, len(todo), ok, skip, fail)
    logger.info("DONE ok=%d norgb=%d fail=%d of %d", ok, skip, fail, len(todo))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
