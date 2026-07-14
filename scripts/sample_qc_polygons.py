"""Stratified QC sample for the South tier-precision rating (Phase B).

Samples n polygons per conf_class band from the tiered candidates GPKG,
stratified longitude × area (the val sweep is only 2 regions; cross-region
variation is the known weak spot, so the sample must span the domain). Adds an
empty ``qc_verdict`` column — rate each polygon ``rts`` / ``false`` /
``unsure`` in ArcGIS Pro, save, and feed the rated file back to compute
per-band precision (the numbers published in south_products.md).

Usage:
    python scripts/sample_qc_polygons.py \
        --candidates /outputs/.../south_rts_candidates.gpkg \
        --output /outputs/.../qc_sample.gpkg --n-per-band 60
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

N_LON_BINS = 6
N_AREA_BINS = 3


def sample_qc(gdf: gpd.GeoDataFrame, n_per_band: int = 60,
              seed: int = 42) -> gpd.GeoDataFrame:
    """n per conf_class band, spread over longitude × area strata (fixed seed).

    Each band's quota is split evenly across the its occupied
    longitude-bin × area-tercile strata (shortfall refilled band-wide at
    random); bands smaller than the quota are returned whole.
    """
    rng = np.random.default_rng(seed)
    out = []
    for cls, band in gdf.groupby("conf_class"):
        if len(band) <= n_per_band:
            out.append(band)
            continue
        lon_bin = pd.cut(band["centroid_lon"], N_LON_BINS, labels=False)
        area_bin = pd.qcut(band["area_m2"], N_AREA_BINS, labels=False,
                           duplicates="drop")
        strata = list(band.groupby([lon_bin, area_bin]).groups.values())
        quota = max(1, n_per_band // len(strata))
        picked: list = []
        for idx in strata:
            k = min(quota, len(idx))
            picked += list(rng.choice(idx, size=k, replace=False))
        rest = band.index.difference(picked)
        short = n_per_band - len(picked)
        if short > 0:
            picked += list(rng.choice(rest, size=min(short, len(rest)),
                                      replace=False))
        out.append(band.loc[picked[:n_per_band]])
    s = gpd.GeoDataFrame(pd.concat(out), crs=gdf.crs).sort_values("rts_id")
    s["qc_verdict"] = ""
    logger.info("QC sample: %s", s["conf_class"].value_counts().to_dict())
    return s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n-per-band", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    setup_logging()
    s = sample_qc(gpd.read_file(args.candidates), args.n_per_band, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    s.to_file(args.output, driver="GPKG")
    logger.info("wrote %s (%d polygons)", args.output, len(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
