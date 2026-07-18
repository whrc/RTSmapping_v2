"""Score the rated QC sample → precision per (tier × size band) → A(p) grid.

Input: the rated sample (GPKG, or the CSV exported by
`post-inference/ee_qc_rater.js`) carrying `conf_class`, `area_m2`,
`qc_verdict` ∈ {rts, false, unsure}. Output: the full tier × size-band grid —
n rated, precision, 95% Wilson interval, and the acceptance decision at the
precision floor (`accept` drives the adaptive-MMU `rts_class` rule; unmeasured
cells are never accepted). `unsure` verdicts are excluded from precision and
reported separately.

Usage:
    python scripts/score_qc_ratings.py --ratings qc_rated.csv \
        --sample qc_sample.gpkg --floor 0.5 --out qc_precision_grid.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from math import inf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# (label, lo, hi) in geodesic m² — SSoT for the QC size bands
SIZE_BANDS = [("<500", 0.0, 500.0), ("500-2k", 500.0, 2000.0),
              ("2k-5k", 2000.0, 5000.0), ("5k-20k", 5000.0, 20000.0),
              (">20k", 20000.0, inf)]
TIERS = ("high", "medium", "low")
Z95 = 1.959963984540054


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    z2 = Z95 * Z95
    den = 1 + z2 / n
    centre = p + z2 / (2 * n)
    half = Z95 * sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (centre - half) / den, (centre + half) / den


def precision_grid(df: pd.DataFrame, floor: float = 0.5) -> pd.DataFrame:
    """Full tier × size-band grid; unmeasured cells present with n=0/NaN."""
    rows = []
    for tier in TIERS:
        for label, lo, hi in SIZE_BANDS:
            cell = df[(df["conf_class"] == tier) & (df["area_m2"] >= lo)
                      & (df["area_m2"] < hi)]
            rated = cell[cell["qc_verdict"].isin(["rts", "false"])]
            n, k = len(rated), int((rated["qc_verdict"] == "rts").sum())
            wl, wh = _wilson(k, n)
            p = k / n if n else float("nan")
            rows.append(dict(conf_class=tier, band=label, n_rated=n, n_rts=k,
                             n_unsure=int((cell["qc_verdict"] == "unsure").sum()),
                             precision=p, wilson_lo=wl, wilson_hi=wh,
                             accept=bool(n and p >= floor)))
    return pd.DataFrame(rows)


def export_false_polygons(ratings_csv: str, sample_gpkg: str,
                          out_gpkg: str) -> None:
    """Export the rated-``false`` polygons with geometry — the v3
    hard-negative seed set (user-verified false positives)."""
    import geopandas as gpd
    ratings = pd.read_csv(ratings_csv)
    sample = gpd.read_file(sample_gpkg).drop(columns=["qc_verdict"],
                                             errors="ignore")
    merged = sample.merge(ratings, on="rts_id", validate="1:1")
    false = merged[merged["qc_verdict"] == "false"]
    false.to_file(out_gpkg, driver="GPKG")
    logger.info("hard-negative seed: %d rated-false polygons → %s",
                len(false), out_gpkg)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratings", required=True,
                   help="rated CSV (rts_id, qc_verdict) from the GEE rater, "
                        "or a rated GPKG with qc_verdict filled")
    p.add_argument("--sample", default=None,
                   help="qc_sample.gpkg — joins conf_class/area_m2 onto a "
                        "verdict-only CSV")
    p.add_argument("--floor", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--false-out", default=None,
                   help="also export rated-false polygons (v3 hard-negative "
                        "seed GPKG); needs --sample")
    args = p.parse_args()
    setup_logging()

    if str(args.ratings).endswith(".csv"):
        df = pd.read_csv(args.ratings)
        if "conf_class" not in df.columns:
            import geopandas as gpd
            s = gpd.read_file(args.sample)[["rts_id", "conf_class", "area_m2"]]
            df = df.merge(s, on="rts_id", validate="1:1")
    else:
        import geopandas as gpd
        df = gpd.read_file(args.ratings)
    grid = precision_grid(df, args.floor)
    grid.to_csv(args.out, index=False)
    logger.info("precision grid → %s\n%s", args.out, grid.to_string(index=False))
    if args.false_out:
        export_false_polygons(str(args.ratings), args.sample, args.false_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
