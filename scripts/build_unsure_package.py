"""Build the ArcGIS Pro package for adjudicating the campaign's `unsure` polygons.

The review app's reviewers rated every polygon `rts` / `false` / `unsure`. The
`unsure` ones go to a second, hand pass in ArcGIS Pro with the Esri World
Imagery basemap and the Planet 2025 Q3 imagery side by side. That pass runs on
a collaborator's machine, so the output is one **self-contained folder** that
needs nothing but ArcGIS Pro to open:

    unsure_pkg_<date>/
      README.md                how to load the imagery and polygons, nothing more
      unsure_polygons.shp      one row per polygon: rts_id, rev_order, area_m2,
                               no_planet
      issued_ids.csv           pass to --exclude on the next build
      planet_2025q3.vrt        one raster layer for all the imagery
      planet_2025q3/
        r_<gx>_<gy>.vrt        one small VRT per ~200 km region
        chips/t<x>_<y>.tif     the RGB chips the polygons touch, as built by
                               scripts/build_rgb_chips.py

The imagery is a two-level VRT on purpose. One VRT over ~17k chips makes GDAL
walk every source on every read (~2 s per window, `review_campaign.md` §4.1),
which is unusable for panning. Regional VRTs keep each walk to a few dozen
sources.

**Blinded:** `max_prob`, `conf_class`, batch id and the first-pass reviewer are
deliberately not in the package. A visible probability primes the rater, and
the campaign exists to measure the model (`review_campaign.md` §6).

The chips are PlanetScope-derived and **not for redistribution**: keep the
package inside the team and never put it under `products/` or in the repo.

Re-runnable: pass `--exclude` with the ids already issued and a later package
holds only polygons that have become `unsure` since.

Needs GDAL (`gdalbuildvrt`) on PATH — the VM that built the chips has it.

Spec: `post-inference/review_campaign.md` §7.1.

Usage:
    python scripts/build_unsure_package.py \
        --verdicts gs://rts-arctic-usw1/inference/2025q3_south/review/verdicts \
        --candidates /outputs/.../south_rts_candidates.gpkg \
        --chips gs://rts-arctic-usw1/inference/2025q3_south/internal/rgb_chips \
        --no-imagery /outputs/.../no_imagery.csv \
        --out-dir /outputs/.../adjudication [--exclude issued_ids.csv] [--limit 50]
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.merge_review_verdicts import read_verdicts  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

SEED = 42
EARTH_R = 6378137.0       # EPSG:3857 sphere radius, m
ORDER_CELL_M = 25_000.0   # review order walks a 25 km grid so neighbours share imagery
REGION_M = 200_000.0      # one regional VRT per 200 km cell
DOWNLOAD_WORKERS = 32
LAYER_COLUMNS = ["rts_id", "rev_order", "area_m2", "no_planet", "geometry"]

README = """\
# Unsure polygons ({date})

{n} polygons, {n_no_planet} of them with no Planet imagery (`no_planet` = 1).

## Loading

Keep the folder structure exactly as unzipped: `planet_2025q3.vrt` finds the
imagery by relative path.

- **Planet imagery** (2025 Q3, EPSG:3857): add `planet_2025q3.vrt` as a raster
  layer.
- **Polygons** (EPSG:3857): add `unsure_polygons.shp`.

Polygon fields: `rts_id`, `rev_order` (neighbouring polygons are consecutive),
`area_m2`, `no_planet`.
"""


# --------------------------------------------------------------------------
# selection and ordering (pandas only)
# --------------------------------------------------------------------------
def select_unsure(verdicts: pd.DataFrame, exclude: set[int] | None = None
                  ) -> list[int]:
    """Ids whose coverage verdict is `unsure`, minus any already issued.

    Replicates (``injected``) are ignored and a repeated coverage verdict keeps
    the latest, exactly as `scripts/merge_review_verdicts.py` pools them, so the
    package agrees with the merged inventory about what is unsure.

    Args:
        verdicts: raw verdict records (``read_verdicts`` output).
        exclude: ids to leave out, e.g. those in an earlier package.

    Returns:
        Sorted polygon ids.
    """
    if verdicts.empty:
        return []
    cov = (verdicts[~verdicts["injected"]].sort_values("reviewed_at")
           .drop_duplicates("rts_id", keep="last"))
    ids = set(cov.loc[cov["verdict"] == "unsure", "rts_id"].astype(int))
    return sorted(ids - (exclude or set()))


def web_mercator(lon: pd.Series, lat: pd.Series) -> tuple[pd.Series, pd.Series]:
    """WGS84 degrees -> EPSG:3857 metres."""
    x = EARTH_R * np.radians(lon)
    y = EARTH_R * np.log(np.tan(math.pi / 4 + np.radians(lat) / 2))
    return x, y


def order_polygons(df: pd.DataFrame, cell_m: float = ORDER_CELL_M) -> pd.DataFrame:
    """Add `rev_order` (1..N): a serpentine walk over a ``cell_m`` grid.

    Named `rev_order`, not `order`: ORDER is a reserved SQL word and trips
    ArcGIS field expressions and sorting.

    Consecutive polygons are neighbours, so the imagery stays cached while the
    reviewer works through an area. Deterministic: ties break on ``rts_id``.

    Args:
        df: needs ``rts_id``, ``centroid_lon``, ``centroid_lat``.
        cell_m: grid cell edge in EPSG:3857 metres.

    Returns:
        A copy sorted by ``order``.
    """
    x, y = web_mercator(df["centroid_lon"], df["centroid_lat"])
    out = df.assign(_gx=np.floor(x / cell_m).astype(int),
                    _gy=np.floor(y / cell_m).astype(int))
    # Alternate the x direction on odd rows so the walk never jumps back.
    out["_key"] = np.where(out["_gy"] % 2 == 0, out["_gx"], -out["_gx"])
    out = out.sort_values(["_gy", "_key", "rts_id"], ignore_index=True)
    out["rev_order"] = np.arange(1, len(out) + 1)
    return out.drop(columns=["_gx", "_gy", "_key"])


def chip_regions(df: pd.DataFrame, region_m: float = REGION_M) -> pd.DataFrame:
    """Assign every chip the polygons touch to exactly one regional VRT.

    A chip goes to the region of the first (lowest `order`) polygon that
    references it, so a chip shared by two polygons is stored and listed once.

    Args:
        df: ordered polygons with ``tile_ids`` (comma-separated), and the
            columns :func:`order_polygons` needs.
        region_m: region cell edge in EPSG:3857 metres.

    Returns:
        Columns ``chip`` (``t{x}_{y}``) and ``region`` (``r_{gx}_{gy}``).
    """
    d = df.sort_values("rev_order")
    x, y = web_mercator(d["centroid_lon"], d["centroid_lat"])
    region = ("r_" + np.floor(x / region_m).astype(int).astype(str) + "_"
              + np.floor(y / region_m).astype(int).astype(str))
    rows = []
    for tiles, reg in zip(d["tile_ids"], region):
        rows += [(t, reg) for t in str(tiles).split(",") if t]
    return pd.DataFrame(rows, columns=["chip", "region"]).drop_duplicates(
        "chip", keep="first", ignore_index=True)


def layer_table(df: pd.DataFrame, no_imagery: set[int]) -> pd.DataFrame:
    """The columns the reviewer's layer carries — and only those.

    ``max_prob``, ``conf_class``, batch id and reviewer are dropped on purpose
    (blinding, see the module docstring). There is no verdict field: how the
    adjudicator records decisions is their choice, and is parsed on return.

    Args:
        df: ordered polygons (:func:`order_polygons`) with a geometry column.
        no_imagery: ids with no Planet coverage (`review/no_imagery.csv`).
    """
    out = df[["rts_id", "rev_order", "area_m2", "geometry"]].copy()
    out["no_planet"] = out["rts_id"].isin(no_imagery).astype(int)
    return out[LAYER_COLUMNS]


# --------------------------------------------------------------------------
# chips and VRTs
# --------------------------------------------------------------------------
def fetch_chips(names: list[str], src: str, dst_dir: Path,
                workers: int = DOWNLOAD_WORKERS) -> list[str]:
    """Copy chips from a ``gs://`` prefix or a local directory, skipping any
    already present. Returns the names that could not be found.

    Writes to ``.partial`` then renames, so an interrupted run never leaves a
    truncated chip that the skip-existing check would accept.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    bucket = prefix = None
    if src.startswith("gs://"):
        from google.cloud import storage
        from requests.adapters import HTTPAdapter
        bucket_name, _, prefix = src[5:].partition("/")
        client = storage.Client()
        # The default pool holds 10 connections; one per worker thread stops
        # urllib3 discarding connections (and warning) once workers > 10.
        client._http.mount("https://", HTTPAdapter(pool_connections=workers,
                                                   pool_maxsize=workers))
        bucket = client.bucket(bucket_name)
        prefix = prefix.strip("/")

    def one(name: str) -> str | None:
        dst = dst_dir / f"{name}.tif"
        if dst.exists():
            return None
        tmp = dst.with_suffix(".tif.partial")
        try:
            if bucket is not None:
                bucket.blob(f"{prefix}/{name}.tif").download_to_filename(str(tmp))
            else:
                shutil.copyfile(Path(src) / f"{name}.tif", tmp)
        except Exception as exc:  # noqa: BLE001 - one missing chip must not kill the run
            logger.warning("chip %s: %s", name, exc)
            tmp.unlink(missing_ok=True)
            return name
        tmp.replace(dst)
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        missing = [m for m in pool.map(one, names) if m]
    return missing


def _build_vrt(vrt_name: str, members: list[str], cwd: Path) -> None:
    """``gdalbuildvrt`` with a file list (a region can exceed the Windows/POSIX
    argv limit) and 0 as nodata, so empty chip edges stay see-through over the
    Esri basemap. Members are relative to ``cwd`` and so are stored relative."""
    listing = cwd / f"{vrt_name}.list.txt"
    listing.write_text("\n".join(members))
    try:
        subprocess.run(["gdalbuildvrt", "-q", "-srcnodata", "0",
                        "-vrtnodata", "0", "-input_file_list", listing.name,
                        vrt_name], check=True, cwd=str(cwd),
                       stdout=subprocess.DEVNULL)
    finally:
        listing.unlink(missing_ok=True)


def build_imagery(assign: pd.DataFrame, pkg: Path) -> None:
    """Regional VRTs over ``planet_2025q3/chips``, then the top-level VRT."""
    planet = pkg / "planet_2025q3"
    regions = sorted(assign["region"].unique())
    for reg in regions:
        chips = sorted(assign.loc[assign["region"] == reg, "chip"])
        _build_vrt(f"{reg}.vrt", [f"chips/{c}.tif" for c in chips], planet)
    _build_vrt("planet_2025q3.vrt", [f"planet_2025q3/{r}.vrt" for r in regions],
               pkg)
    logger.info("built %d regional VRTs over %d chips", len(regions), len(assign))


def zip_package(pkg: Path) -> Path:
    """Zip the folder. Stored, not deflated: the chips are already compressed,
    so deflating costs minutes and saves nothing."""
    out = pkg.with_suffix(".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(pkg.parent))
    return out


# --------------------------------------------------------------------------
def _read_ids(path: str | None) -> set[int]:
    return set() if not path else set(
        pd.read_csv(path)["rts_id"].astype(int))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verdicts", required=True,
                   help="local dir or gs:// prefix holding the batch JSONLs")
    p.add_argument("--candidates", required=True,
                   help="local south_rts_candidates.gpkg")
    p.add_argument("--chips", required=True,
                   help="gs:// prefix or local dir holding t<x>_<y>.tif chips")
    p.add_argument("--no-imagery", help="review/no_imagery.csv (rts_id column)")
    p.add_argument("--exclude", help="CSV with rts_id: polygons already issued")
    p.add_argument("--limit", type=int,
                   help="random sample of N polygons — a small package to test "
                        "the Esri imagery and the README before the full run")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    args = p.parse_args()
    setup_logging()

    import geopandas as gpd

    ids = select_unsure(read_verdicts(args.verdicts), _read_ids(args.exclude))
    if args.limit and args.limit < len(ids):
        ids = sorted(np.random.default_rng(SEED).choice(
            ids, args.limit, replace=False).tolist())
    if not ids:
        logger.warning("no unsure polygons to issue")
        return 0

    gdf = gpd.read_file(args.candidates)
    gdf["rts_id"] = gdf["rts_id"].astype(int)
    gdf = gdf[gdf["rts_id"].isin(ids)]
    if len(gdf) != len(ids):
        raise ValueError(f"{len(ids) - len(gdf)} unsure ids are not in "
                         f"{args.candidates}")
    gdf = order_polygons(gdf)

    pkg = args.out_dir / f"unsure_pkg_{datetime.date.today():%Y-%m-%d}"
    assign = chip_regions(gdf)
    missing = fetch_chips(assign["chip"].tolist(), args.chips,
                          pkg / "planet_2025q3" / "chips", args.workers)
    if missing:
        raise RuntimeError(f"{len(missing)} chips not found, e.g. {missing[:5]}")
    build_imagery(assign, pkg)

    out = layer_table(gdf, _read_ids(args.no_imagery))
    out.to_file(pkg / "unsure_polygons.shp", driver="ESRI Shapefile")
    (pkg / "README.md").write_text(README.format(
        date=f"{datetime.date.today():%Y-%m-%d}", n=len(out),
        n_no_planet=int(out["no_planet"].sum())))
    out[["rts_id"]].to_csv(pkg / "issued_ids.csv", index=False)

    z = zip_package(pkg)
    logger.info("wrote %s — %d polygons, %d chips, %.1f GB", z, len(out),
                len(assign), z.stat().st_size / 1e9)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
