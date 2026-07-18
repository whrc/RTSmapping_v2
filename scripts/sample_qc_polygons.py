"""Stratified QC sample for the South tier × size precision rating (Phase 3).

Samples n polygons per (conf_class tier × geodesic size band) cell — the grid
whose measured precision defines the adaptive-MMU acceptance rule A(p) — each
cell spread across longitude bins (the val sweep is only 2 regions;
cross-region variation is the known weak spot). Adds an empty ``qc_verdict``
column: rate each polygon ``rts`` / ``false`` / ``unsure`` (GEE rater
`post-inference/ee_qc_rater.js` or ArcGIS), then feed the ratings to
`scripts/score_qc_ratings.py`.

Size bands: SSoT in `scripts/score_qc_ratings.py:SIZE_BANDS`.

Usage:
    python scripts/sample_qc_polygons.py \
        --candidates /outputs/.../south_rts_candidates.gpkg \
        --output /outputs/.../qc_sample.gpkg --n-per-cell 20
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

from scripts.score_qc_ratings import SIZE_BANDS  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

N_LON_BINS = 6


def sample_qc(gdf: gpd.GeoDataFrame, n_per_cell: int = 20,
              seed: int = 42) -> gpd.GeoDataFrame:
    """n per (tier × size band) cell, spread over longitude bins (fixed seed).

    Each cell's quota is split evenly across its occupied longitude bins
    (shortfall refilled cell-wide at random); cells smaller than the quota are
    returned whole.
    """
    rng = np.random.default_rng(seed)
    out = []
    for (cls, band_label), lo, hi in [((c, b), lo, hi)
                                      for c in gdf["conf_class"].unique()
                                      for b, lo, hi in SIZE_BANDS]:
        cell = gdf[(gdf["conf_class"] == cls) & (gdf["area_m2"] >= lo)
                   & (gdf["area_m2"] < hi)]
        if len(cell) <= n_per_cell:
            out.append(cell)
            continue
        lon_bin = pd.cut(cell["centroid_lon"], N_LON_BINS, labels=False)
        strata = list(cell.groupby(lon_bin).groups.values())
        quota = max(1, n_per_cell // len(strata))
        picked: list = []
        for idx in strata:
            k = min(quota, len(idx))
            picked += list(rng.choice(idx, size=k, replace=False))
        rest = cell.index.difference(picked)
        short = n_per_cell - len(picked)
        if short > 0:
            picked += list(rng.choice(rest, size=min(short, len(rest)),
                                      replace=False))
        out.append(cell.loc[picked[:n_per_cell]])
    s = gpd.GeoDataFrame(pd.concat(out), crs=gdf.crs).sort_values("rts_id")
    s["qc_verdict"] = ""
    logger.info("QC sample %d polygons: %s", len(s),
                s["conf_class"].value_counts().to_dict())
    return s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n-per-cell", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    setup_logging()
    s = sample_qc(gpd.read_file(args.candidates), args.n_per_cell, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    s.to_file(args.output, driver="GPKG")
    logger.info("wrote %s (%d polygons)", args.output, len(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
