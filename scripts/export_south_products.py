"""Package the tiered RTS inventory into its four access forms (D1 products).

From the raw thr-0.30 candidates GPKG (scripts/vectorize_region.py
--threshold 0.30) produce:

  south_rts_candidates.gpkg  — flagship: + conf_class (high ≥0.65 /
                               medium ≥0.45 / low ≥0.30 by max_prob) and
                               rts_class (QC-calibrated adaptive-MMU rule)
  south_rts_high_confidence.gpkg   — rts_class == high_confidence (the "fact map")
  south_rts_centroids.gpkg   — representative_point() per polygon (guaranteed
                               inside), same attributes; pan-Arctic-zoom layer
  south_rts_attributes.csv/.parquet — attribute table, no GIS needed

Usage:
    python scripts/export_south_products.py \
        --candidates /outputs/.../south_rts_candidates_t30.gpkg \
        --out-dir /outputs/.../products_local
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

TIER_BOUNDS = (0.45, 0.65)  # medium ≥ .45, high ≥ .65 (SSoT for conf_class)
# rts_class rule locked from the 2026-07 South QC precision grid (279 ratings,
# scripts/score_qc_ratings.py, floor 0.5): all high-tier cells cleared the
# floor (0.54–0.90; <500 m² unmeasured, accepted by monotone tier extension);
# the only other clearing cell is medium <500 m² (0.53) → candidate.
CANDIDATE_MAX_AREA_M2 = 500.0


def assign_rts_class(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``rts_class`` (high_confidence/candidate/marginal) from the QC-calibrated
    adaptive-MMU rule: conf_class high → high_confidence; medium under
    ``CANDIDATE_MAX_AREA_M2`` → candidate; everything else → marginal."""
    out = gdf.copy()
    out["rts_class"] = "marginal"
    out.loc[(gdf["conf_class"] == "medium")
            & (gdf["area_m2"] < CANDIDATE_MAX_AREA_M2),
            "rts_class"] = "candidate"
    out.loc[gdf["conf_class"] == "high", "rts_class"] = "high_confidence"
    return out


def assign_conf_class(gdf: gpd.GeoDataFrame,
                      bounds: tuple[float, float] = TIER_BOUNDS
                      ) -> gpd.GeoDataFrame:
    """Add ``conf_class`` (low/medium/high) from ``max_prob``, inclusive bounds."""
    med, high = bounds
    cls = gdf["max_prob"].map(
        lambda p: "high" if p >= high else "medium" if p >= med else "low")
    out = gdf.copy()
    out["conf_class"] = cls
    return out


def add_nodata_frac(gdf: gpd.GeoDataFrame, prob_raster: str,
                    pad_frac: float = 0.5) -> gpd.GeoDataFrame:
    """Add ``nodata_frac``: fraction of NoData (255) pixels in each polygon's
    bbox padded by ``pad_frac`` × its extent, read from the probability
    raster. Soft triage attribute only — the QC found FPs concentrate on
    high-NoData context, but real RTS can also contain NoData, so this must
    never be a hard veto."""
    import numpy as np
    import rasterio
    from rasterio import windows

    fracs = []
    with rasterio.open(prob_raster) as src:
        for geom in gdf.geometry:
            b = geom.bounds
            pad = pad_frac * max(b[2] - b[0], b[3] - b[1])
            win = windows.from_bounds(b[0] - pad, b[1] - pad, b[2] + pad,
                                      b[3] + pad, transform=src.transform)
            block = src.read(1, window=win, boundless=True, fill_value=255)
            fracs.append(float((block == 255).mean()) if block.size else 1.0)
    out = gdf.copy()
    out["nodata_frac"] = np.round(fracs, 4)
    return out


def export_products(candidates_gpkg: str, out_dir: str | Path,
                    bounds: tuple[float, float] = TIER_BOUNDS,
                    prob_raster: str | None = None) -> None:
    """Write the four D1 access forms from the raw candidates GPKG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf = assign_rts_class(assign_conf_class(gpd.read_file(candidates_gpkg),
                                             bounds))
    if prob_raster:
        gdf = add_nodata_frac(gdf, prob_raster)

    gdf.to_file(out_dir / "south_rts_candidates.gpkg", driver="GPKG")
    gdf[gdf["rts_class"] == "high_confidence"].to_file(
        out_dir / "south_rts_high_confidence.gpkg", driver="GPKG")
    pts = gdf.copy()
    pts["geometry"] = gdf.geometry.representative_point()
    pts.to_file(out_dir / "south_rts_centroids.gpkg", driver="GPKG")

    attrs = gdf.drop(columns="geometry")
    attrs.to_csv(out_dir / "south_rts_attributes.csv", index=False)
    attrs.to_parquet(out_dir / "south_rts_attributes.parquet", index=False)
    counts = gdf["conf_class"].value_counts()
    logger.info("exported %d candidates (%s) → %s", len(gdf),
                ", ".join(f"{k}={v}" for k, v in counts.items()), out_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True,
                   help="raw thr-0.30 GPKG from vectorize_region --threshold")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--tier-bounds", type=float, nargs=2,
                   default=list(TIER_BOUNDS), metavar=("MED", "HIGH"))
    p.add_argument("--prob-raster", default=None,
                   help="probability VRT/tif — adds the nodata_frac soft "
                        "triage attribute")
    args = p.parse_args()
    setup_logging()
    export_products(args.candidates, args.out_dir, tuple(args.tier_bounds),
                    args.prob_raster)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
