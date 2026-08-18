"""Step 2 — clip the year's quad list to the circumpolar-south inference domain.

Port of `2_circumpolar_south_basemap_grids.qmd` (R/sf) to geopandas. Porting it
off R means the VM needs one language and one environment rather than two; the
operations are a 1:1 translation of the original:

    st_transform(3857) -> .to_crs(3857)
    st_make_valid()    -> .make_valid()
    st_filter(domain)  -> intersects-any against the unioned domain
    arrange(col, row)  -> .sort_values(["grid_column", "grid_row"])

The row count this prints is the number step 5 reconciles the quad index
against (`build_quad_index.py --expect-quads`), so it is worth recording.

Usage:
    python planetscope-download/filter_to_domain.py --year 2022 \
        --grids planetscope-download/data/circumpolar_basemap_grids_2022.geojson \
        --output planetscope-download/data/circumpolar_south_planet_basemap_grids_2022.geojson
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOMAIN = REPO_ROOT / "domain" / "circumpolar_south_domain.geojson"
WORKING_CRS = 3857  # EPSG:3857 everywhere, per CLAUDE.md


def filter_to_domain(grids: gpd.GeoDataFrame, domain: gpd.GeoDataFrame,
                     year: int) -> gpd.GeoDataFrame:
    """Clip `grids` to `domain` and derive the ordering columns.

    Args:
        grids: Quad list from step 1 (any CRS; reprojected here).
        domain: Inference domain polygon(s).
        year: Calendar year, used to build the mosaic name.

    Returns:
        GeoDataFrame sorted by (grid_column, grid_row) with the columns the
        order step needs: year, id, grid_column, grid_row, basemap_name,
        delivery_location, link.
    """
    grids = grids.to_crs(WORKING_CRS)
    domain = domain.to_crs(WORKING_CRS)
    domain.geometry = domain.geometry.make_valid()

    keep = grids.geometry.intersects(domain.geometry.union_all())
    out = grids.loc[keep].copy()
    logger.info("Domain clip: %d of %d quads intersect", len(out), len(grids))
    if out.empty:
        return out.reindex(columns=["year", "id", "grid_column", "grid_row",
                                    "basemap_name", "delivery_location", "link",
                                    "geometry"])

    ids = out["id"].str.split("-", n=1, expand=True)
    out["year"] = year
    out["grid_column"] = ids[0].astype(int)
    out["grid_row"] = ids[1].astype(int)
    out["basemap_name"] = f"global_quarterly_{year}q3_mosaic"
    out["delivery_location"] = (
        "global_quarterly/" + str(year) + "/q3/"
        + out["grid_column"].astype(str) + "/" + out["grid_row"].astype(str) + "/"
    )
    cols = ["year", "id", "grid_column", "grid_row", "basemap_name",
            "delivery_location", "link", "geometry"]
    return out[cols].sort_values(["grid_column", "grid_row"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--grids", type=Path, required=True, help="step 1 output")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--domain", type=Path, default=DEFAULT_DOMAIN,
                   help="inference domain (default: the repo's circumpolar_south_domain)")
    args = p.parse_args()
    setup_logging()

    out = filter_to_domain(gpd.read_file(args.grids), gpd.read_file(args.domain), args.year)
    if out.empty:
        logger.error("No quads intersect the domain — refusing to write an empty grid file.")
        return 1

    per_col = out.groupby("grid_column").size()
    logger.info("Per-column quad count: min %d, median %d, max %d, across %d columns",
                per_col.min(), per_col.median(), per_col.max(), len(per_col))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(args.output, driver="GeoJSON")
    logger.info("Wrote %d quads to %s", len(out), args.output)
    logger.info("RECORD THIS: %d quads ordered for %d "
                "(pass to build_quad_index.py --expect-quads)", len(out), args.year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
