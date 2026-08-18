"""Step 1 — search Planet's Basemaps API for every q3 quad in a year.

Port of `1_basemap_grid_search.qmd` from HRodenhizer/circumpolar_planet_basemaps
@ initial-download, with the year as an argument instead of a hardcoded literal.

Writes a GeoJSON of every quad in the Global Quarterly q3 mosaic for `--year`
over the Arctic/boreal bbox, which step 2 then clips to the inference domain.

Usage:
    python planetscope-download/search_basemap_grids.py --year 2022 \
        --output planetscope-download/data/circumpolar_basemap_grids_2022.geojson
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import shapely as shp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

API_URL = "https://api.planet.com/basemaps/v1/mosaics"
SERIES_URL = "https://api.planet.com/basemaps/v1/series"
SERIES_NAME = "Global Quarterly"
# Arctic/boreal superset — step 2 does the real domain clip.
SEARCH_BBOX = "-180,40,180,85"
# (connect, read). Every call gets one: a hung socket with no timeout is the
# silent-stall failure mode utils/watchdog.py exists to catch downstream.
TIMEOUT = (10, 120)


def find_mosaic(session: requests.Session, year: int) -> str:
    """Return the q3 mosaic name for `year` from the Global Quarterly series.

    Args:
        session: Authenticated requests session.
        year: Calendar year, e.g. 2022.

    Returns:
        The mosaic name, e.g. "global_quarterly_2022q3_mosaic".

    Raises:
        RuntimeError: If the series or a unique q3 mosaic for `year` is absent.
    """
    res = session.get(SERIES_URL, timeout=TIMEOUT)
    res.raise_for_status()
    series = res.json()["series"]
    links = [s["_links"]["mosaics"] for s in series if s["name"] == SERIES_NAME]
    if not links:
        raise RuntimeError(f"No {SERIES_NAME!r} series found; got {[s['name'] for s in series]}")

    mosaics = session.get(links[0], timeout=TIMEOUT).json()["mosaics"]
    names = [m["name"] for m in mosaics
             if re.search(r"q3", m["name"]) and str(year) in m["name"]]
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly one {year} q3 mosaic, got {names}")
    logger.info("Mosaic for %d: %s", year, names[0])
    return names[0]


def fetch_quads(session: requests.Session, mosaic_name: str) -> gpd.GeoDataFrame:
    """Page the mosaic's quad list over SEARCH_BBOX into a GeoDataFrame.

    Args:
        session: Authenticated requests session.
        mosaic_name: e.g. "global_quarterly_2022q3_mosaic".

    Returns:
        GeoDataFrame (EPSG:4326) with columns name, id, link, percent_covered.
    """
    res = session.get(API_URL, params={"name__is": mosaic_name}, timeout=TIMEOUT)
    res.raise_for_status()
    mosaic = res.json()["mosaics"][0]
    logger.info("Mosaic id %s, bbox %s", mosaic["id"], mosaic["bbox"])

    url = f"{API_URL}/{mosaic['id']}/quads"
    params = {"bbox": SEARCH_BBOX, "minimal": True}
    rows, geoms = [], []
    while url:
        res = session.get(url, params=params, timeout=TIMEOUT)
        res.raise_for_status()
        page = res.json()
        for item in page["items"]:
            rows.append({"name": mosaic_name, "id": item["id"],
                         "link": item["_links"]["download"],
                         "percent_covered": item["percent_covered"]})
            geoms.append(shp.geometry.box(*item["bbox"]))
        if len(rows) % 20000 < len(page["items"]):
            logger.info("  ... %d quads", len(rows))
        url = page.get("_links", {}).get("_next")
        params = None  # the _next link already carries the query

    logger.info("Found %d quads in %s", len(rows), mosaic_name)
    return gpd.GeoDataFrame(pd.DataFrame(rows),
                            geometry=gpd.GeoSeries(geoms, crs="EPSG:4326"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    setup_logging()

    api_key = os.environ.get("PL_BM_API_KEY")
    if not api_key:
        logger.error("PL_BM_API_KEY is not set — run through run_year.sh, which prompts for it.")
        return 2

    session = requests.Session()
    session.auth = (api_key, "")

    try:
        grids = fetch_quads(session, find_mosaic(session, args.year))
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        if code in (401, 403):
            logger.error("Planet rejected the API key (HTTP %s). Check PL_BM_API_KEY "
                         "and start the run again.", code)
            return 2
        logger.error("Planet API error (HTTP %s) while searching for %d q3: %s",
                     code, args.year, e)
        return 1
    except requests.RequestException as e:
        logger.error("Could not reach the Planet API: %s", e)
        return 1
    except RuntimeError as e:      # no series / no unique q3 mosaic for the year
        logger.error("%s", e)
        return 1
    if grids.empty:
        logger.error("No quads returned for %d — refusing to write an empty grid file.", args.year)
        return 1

    empty = int((grids["percent_covered"] == 0).sum())
    if empty:
        logger.warning("%d quads report percent_covered == 0", empty)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grids.to_file(args.output, driver="GeoJSON")
    logger.info("Wrote %d quads to %s", len(grids), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
