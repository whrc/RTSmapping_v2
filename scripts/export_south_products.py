"""Package the tiered RTS inventory into its four access forms (D1 products).

From the raw thr-0.30 candidates GPKG (scripts/vectorize_region.py
--threshold 0.30) produce:

  south_rts_candidates.gpkg  — flagship: + conf_class (high ≥0.65 /
                               medium ≥0.45 / low ≥0.30 by max_prob;
                               boundaries locked by the Phase-B South QC)
  south_rts_high.gpkg        — high tier standalone (the "fact map")
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


def export_products(candidates_gpkg: str, out_dir: str | Path,
                    bounds: tuple[float, float] = TIER_BOUNDS) -> None:
    """Write the four D1 access forms from the raw candidates GPKG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf = assign_conf_class(gpd.read_file(candidates_gpkg), bounds)

    gdf.to_file(out_dir / "south_rts_candidates.gpkg", driver="GPKG")
    gdf[gdf["conf_class"] == "high"].to_file(out_dir / "south_rts_high.gpkg",
                                             driver="GPKG")
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
    args = p.parse_args()
    setup_logging()
    export_products(args.candidates, args.out_dir, tuple(args.tier_bounds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
