"""ArcticDEM availability diagnostic — coverage and acquisition dates.

Facts only: this script computes numbers, it does not recommend anything.
Feeds `docs/arcticdem_diagnostic.md`.

Two independent questions, run separately because they need different deps:

  --part coverage   what fraction of the deployed domain and of the training
                    tiles does ArcticDEM cover?  (geopandas only)
  --part dates      when was the terrain under each positive tile actually
                    observed, relative to the 2024 labels?  (needs earthengine-api)

Coverage uses the domain pair already in `domain/`: `*_ArcticDEM.geojson` is the
domain intersected with ArcticDEM coverage (domain/inference_domain.md), so the
fraction is a straight area ratio. Areas are measured in EPSG:6931 (NSIDC
EASE-Grid 2.0 North, Lambert azimuthal equal-area) — EPSG:3413 is conformal, not
equal-area, so it would bias an area ratio spanning 60-84 deg N.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Equal-area CRS for every area measurement (the domain files are EPSG:3413).
EQUAL_AREA_CRS = "EPSG:6931"

REPO = Path(__file__).resolve().parent.parent
DOMAIN_DIR = REPO / "domain"
DEFAULT_DATA_ROOT = Path("/mnt/outputs/v1.0/data_local")

# ArcticDEM assets. The V4 2m mosaic is the surface the EXTRA channels sample and
# it carries its own date bands, so no strip-collection lookup is needed.
ARCTICDEM_MOSAIC = "UMN/PGC/ArcticDEM/V4/2m_mosaic"

# `mindate` / `maxdate` are integer days since 2000-01-01 (verified against a
# sample tile: maxdate 7463 -> 2020.4, consistent with ArcticDEM V4's 2010-2022
# strip window).
DATE_EPOCH = "2000-01-01"
DAYS_PER_YEAR = 365.2425

# Half a v1.0 tile on the *ground*. Tiles are 512 px x ~4.77 EPSG:3857 map units
# = 2442 map units, and Web Mercator's scale factor is 1/cos(lat), so the ground
# width is ~915 m at 68 deg N. 460 m is that half-width; exactness does not matter
# for a date statistic, but using the projected 1221 m would oversample
# neighbouring terrain.
TILE_HALFWIDTH_GROUND_M = 460.0


def _days_to_year(days: float | None) -> float | None:
    """ArcticDEM date band (days since DATE_EPOCH) as a fractional year."""
    if days is None:
        return None
    return 2000.0 + float(days) / DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Part 1a — coverage
# ---------------------------------------------------------------------------

def domain_coverage() -> dict[str, dict[str, float]]:
    """Area fraction of each inference domain that ArcticDEM covers."""
    import geopandas as gpd

    out: dict[str, dict[str, float]] = {}
    for name in ("circumpolar_south_domain", "circumpolar_domain"):
        full = gpd.read_file(DOMAIN_DIR / f"{name}.geojson").to_crs(EQUAL_AREA_CRS)
        dem = gpd.read_file(DOMAIN_DIR / f"{name}_ArcticDEM.geojson").to_crs(EQUAL_AREA_CRS)
        a_full = float(full.area.sum()) / 1e6      # km^2
        a_dem = float(dem.area.sum()) / 1e6
        out[name] = {
            "area_km2": a_full,
            "arcticdem_area_km2": a_dem,
            "arcticdem_fraction": a_dem / a_full,
        }
        logger.info("%s: %.0f km2 total, %.0f km2 with ArcticDEM (%.4f)",
                    name, a_full, a_dem, a_dem / a_full)
    return out


def _tile_frame(data_root: Path):
    """metadata.csv as a GeoDataFrame of centroids, plus the split->regions map.

    Returns (frame, splits). A region can belong to more than one split —
    val_realistic and val_balanced share the same 5 regions — so splits stay a
    map instead of collapsing into one per-tile column.
    """
    import geopandas as gpd
    import pandas as pd
    import yaml

    meta = pd.read_csv(data_root / "metadata.csv")
    splits = {k: v for k, v in
              yaml.safe_load((data_root / "splits.yaml").read_text()).items()
              if isinstance(v, list)}

    frame = gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta["centroid_lon"], meta["centroid_lat"]),
        crs="EPSG:4326",
    )
    return frame, splits


def tile_coverage(data_root: Path) -> dict[str, Any]:
    """Fraction of training tiles whose centroid falls inside ArcticDEM coverage."""
    import geopandas as gpd

    tiles, splits = _tile_frame(data_root)
    dem = gpd.read_file(DOMAIN_DIR / "circumpolar_south_domain_ArcticDEM.geojson")
    dem_union = dem.to_crs(tiles.crs).union_all()

    tiles["in_arcticdem"] = tiles.geometry.within(dem_union)

    def summarize(frame) -> dict[str, float]:
        n = int(len(frame))
        n_in = int(frame["in_arcticdem"].sum())
        return {"n": n, "n_in_arcticdem": n_in,
                "fraction": (n_in / n) if n else float("nan")}

    out: dict[str, Any] = {
        "all_tiles": summarize(tiles),
        "by_train_class": {str(k): summarize(g)
                           for k, g in tiles.groupby("TrainClass")},
        "by_split": {}, "by_split_positive": {},
    }
    for split, regions in splits.items():
        rows = tiles[tiles["RegionName"].isin(regions)]
        out["by_split"][split] = summarize(rows)
        out["by_split_positive"][split] = summarize(
            rows[rows["TrainClass"] == "positive"])
    logger.info("tiles inside ArcticDEM: %d/%d (%.4f)",
                out["all_tiles"]["n_in_arcticdem"], out["all_tiles"]["n"],
                out["all_tiles"]["fraction"])
    return out


# ---------------------------------------------------------------------------
# Part 1b — acquisition dates
# ---------------------------------------------------------------------------

def probe_date_sources() -> dict[str, Any]:
    """What date information do the ArcticDEM assets actually expose?

    Recorded so the report can state which source the dates came from. The V4 2m
    mosaic carries `mindate` / `maxdate` / `datamask` bands, so the plan's first
    ladder rung succeeds and the strip collection is never needed.
    """
    import ee

    info = ee.Image(ARCTICDEM_MOSAIC).getInfo()
    return {
        "asset": ARCTICDEM_MOSAIC,
        "bands": [b["id"] for b in info.get("bands", [])],
        "date_band_epoch": DATE_EPOCH,
        "date_band_units": "days since the epoch",
    }


def mosaic_dates_for_tiles(data_root: Path, train_class: str | None = None,
                           batch: int = 100, scale_m: float = 30.0) -> dict[str, Any]:
    """Per-tile ArcticDEM acquisition dates from the mosaic's `maxdate` band.

    `maxdate` is the most recent contributing strip per pixel — the date that
    decides whether a 2024-labelled slump could already be carved into the DEM.
    Reduced over each tile footprint with mean/min/max, batched through
    `reduceRegions` to keep the request count (and the GEE quota) low. `datamask`
    gives the authoritative per-tile coverage fraction as a by-product.
    """
    import ee
    import numpy as np

    tiles, splits = _tile_frame(data_root)
    subset = (tiles if train_class is None
              else tiles[tiles["TrainClass"] == train_class]).reset_index(drop=True)
    image = ee.Image(ARCTICDEM_MOSAIC).select(["maxdate", "mindate", "datamask"])

    rows: list[dict[str, Any]] = []
    for start in range(0, len(subset), batch):
        chunk = subset.iloc[start:start + batch]
        feats = [
            ee.Feature(
                ee.Geometry.Point([float(r["centroid_lon"]),
                                   float(r["centroid_lat"])])
                .buffer(TILE_HALFWIDTH_GROUND_M).bounds(),
                {"tile_id": r["Tile_ID"]},
            )
            for _, r in chunk.iterrows()
        ]
        # reduceRegions drops masked pixels, so a tile with no ArcticDEM at all
        # comes back with null reducer outputs rather than datamask 0.
        reduced = image.reduceRegions(
            collection=ee.FeatureCollection(feats),
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
            scale=scale_m,
        ).getInfo()
        for feat in reduced["features"]:
            p = feat["properties"]
            rows.append({
                "tile_id": p["tile_id"],
                "maxdate_year_mean": _days_to_year(p.get("maxdate_mean")),
                "maxdate_year_max": _days_to_year(p.get("maxdate_max")),
                "mindate_year_mean": _days_to_year(p.get("mindate_mean")),
                "datamask_fraction": p.get("datamask_mean"),
            })
        logger.info("  %d/%d tiles reduced", min(start + batch, len(subset)),
                    len(subset))

    by_id = {r["tile_id"]: r for r in rows}
    subset["has_dem"] = subset["Tile_ID"].map(
        lambda t: by_id.get(t, {}).get("maxdate_year_mean") is not None)

    def summarize(frame) -> dict[str, Any]:
        n = int(len(frame))
        n_in = int(frame["has_dem"].sum())
        years = np.array([by_id[t]["maxdate_year_mean"]
                          for t in frame.loc[frame["has_dem"], "Tile_ID"]])
        return {
            "n": n,
            "n_with_dem": n_in,
            "fraction_with_dem": (n_in / n) if n else float("nan"),
            "median_maxdate_year": float(np.median(years)) if len(years) else None,
            "fraction_maxdate_ge_2024": (float((years >= 2024).mean())
                                         if len(years) else None),
        }

    all_years = np.array([r["maxdate_year_mean"] for r in rows
                          if r["maxdate_year_mean"] is not None])
    out: dict[str, Any] = {
        "train_class": train_class or "all",
        "all_tiles": summarize(subset),
        "by_train_class": {str(k): summarize(g)
                           for k, g in subset.groupby("TrainClass")},
        "by_split": {}, "by_split_positive": {},
        "maxdate_year_histogram": {
            str(y): int(((all_years >= y) & (all_years < y + 1)).sum())
            for y in range(2007, 2026)
        },
        "mindate_year_range": [
            float(min(r["mindate_year_mean"] for r in rows
                      if r["mindate_year_mean"] is not None)),
            float(max(r["mindate_year_mean"] for r in rows
                      if r["mindate_year_mean"] is not None)),
        ],
        "per_tile": rows,
    }
    for split, regions in splits.items():
        frame = subset[subset["RegionName"].isin(regions)]
        out["by_split"][split] = summarize(frame)
        out["by_split_positive"][split] = summarize(
            frame[frame["TrainClass"] == "positive"])

    logger.info("tiles with ArcticDEM: %d/%d (%.4f); median maxdate year %.1f; "
                "fraction 2024+: %.4f",
                out["all_tiles"]["n_with_dem"], out["all_tiles"]["n"],
                out["all_tiles"]["fraction_with_dem"],
                out["all_tiles"]["median_maxdate_year"],
                out["all_tiles"]["fraction_maxdate_ge_2024"])
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["coverage", "dates"], required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, required=True,
                        help="JSON output path")
    parser.add_argument("--allowlist-out", type=Path,
                        help="dates only: write the DEM-covered tile IDs here, one "
                             "per line, for splits.tile_allowlist")
    parser.add_argument("--ee-project", default="pdg-project-406720")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.part == "coverage":
        result = {"domain": domain_coverage(), "tiles": tile_coverage(args.data_root)}
    else:
        import ee
        ee.Initialize(project=args.ee_project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
        dates = mosaic_dates_for_tiles(args.data_root)
        result = {"sources": probe_date_sources(), "tiles": dates}
        if args.allowlist_out:
            covered = sorted(r["tile_id"] for r in dates["per_tile"]
                             if r["maxdate_year_mean"] is not None)
            args.allowlist_out.parent.mkdir(parents=True, exist_ok=True)
            args.allowlist_out.write_text("\n".join(covered) + "\n")
            logger.info("wrote %d DEM-covered tile ids to %s",
                        len(covered), args.allowlist_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
