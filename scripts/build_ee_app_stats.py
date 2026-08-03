"""Precompute the GEE app's MMU retention ladders + min_blob reference values.

The app's size-filter readout must update with no perceptible latency, so no
server-side aggregation may run while the user drags the slider (an
`ee.FeatureCollection.aggregate_sum` over the filtered 60k inventory costs
seconds per tick). Every count and area the panel reports is therefore computed
here, once, and pasted into `post-inference/ee_south_app.js` as a JSON literal.

`conf_class` is disjoint and exhaustive (`export_south_products.assign_conf_class`),
so per-tier ladders sum client-side to the exact number for any tier combination
— three series cover all eight subsets. The 0.65 core inventory needs one
series: every polygon in it has `max_prob >= 0.65` by construction, so it is
all-`high` and the tier control does not apply to it.

`min_blob_size_px` is a *pixel* count, so its ground floor scales with
`res² · cos²(lat)` — the latitude bias that motivated the move to a geodesic
MMU. The reference values below are emitted per latitude, plus one
representative value at the inventory's median latitude for the app's preset.

Usage:
    python scripts/build_ee_app_stats.py \
        --candidates /outputs/.../south_rts_attributes.parquet \
        --t65        /outputs/.../south_rts_t65.gpkg \
        --region-log /outputs/.../region_log.json \
        --out        /outputs/.../app_stats.js
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Non-uniform so the bottom decade — where nearly all polygons sit — keeps
# resolution a uniform slider step would lose. The app's slider indexes this
# list rather than sweeping m² directly. 79 m² is ARTS P1 (the published
# inventory's minimum mapping unit), carried so the preset lands exactly on it.
LADDER_M2 = [0, 10, 20, 30, 50, 79, 100, 150, 200, 300, 500, 750, 1000, 1500,
             2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000]

# Latitudes the min_blob reference table is reported at (the domain spans
# ~50-76°N of real data on a canvas reaching 45.57°N).
REF_LATS = [50, 60, 70, 76]

TIERS = ("high", "medium", "low")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read an attribute table from parquet/csv, or a GPKG's attributes."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    import geopandas as gpd  # only needed for the geometry-carrying forms
    return pd.DataFrame(gpd.read_file(path).drop(columns="geometry"))


def min_blob_m2(min_blob_px: int, resolution_m: float, lat_deg: float) -> float:
    """Ground area (m²) of a ``min_blob_px`` EPSG:3857 pixel count at ``lat_deg``.

    3857 pixel ground area is ``res² · cos²(lat)`` — a pixel MMU is therefore a
    latitude-varying area floor, not a constant one.
    """
    return min_blob_px * resolution_m ** 2 * np.cos(np.radians(lat_deg)) ** 2


def retention(areas: np.ndarray, ladder: list[int]) -> dict:
    """Polygons and km² retained at each ladder value (keep ``area_m2 >= mmu``).

    Matches the app's ``ee.Filter.gte('area_m2', mmu)`` and the exact geodesic
    filter in ``vectorize_region.vectorize_region``.
    """
    srt = np.sort(np.asarray(areas, dtype=np.float64))
    cum = np.concatenate([[0.0], np.cumsum(srt)])
    idx = np.searchsorted(srt, ladder, side="left")
    return {"n": (len(srt) - idx).tolist(),
            # 6 dp = 1 m², exact for any area the app displays and free of the
            # float noise a raw repr would put in the pasted literal
            "km2": [round(v, 6) for v in (cum[-1] - cum[idx]) / 1e6]}


def build_stats(candidates: pd.DataFrame, t65: pd.DataFrame,
                min_blob_px: int, resolution_m: float) -> dict:
    """Assemble the JSON payload the app embeds."""
    med_lat = float(np.median(np.abs(candidates["centroid_lat"])))
    repr_m2 = min_blob_m2(min_blob_px, resolution_m, med_lat)
    ladder = sorted(set(LADDER_M2) | {int(round(repr_m2))})

    series = {t: retention(candidates.loc[candidates["conf_class"] == t,
                                          "area_m2"].values, ladder)
              for t in TIERS}
    series["t65"] = retention(t65["area_m2"].values, ladder)

    missing = len(candidates) - sum(s["n"][0] for t, s in series.items()
                                    if t in TIERS)
    if missing:
        raise ValueError(f"conf_class is not exhaustive: {missing} polygons "
                         "outside high/medium/low")

    return {
        "ladder_m2": ladder,
        "series": series,
        "min_blob": {
            "px": min_blob_px,
            "resolution_m": resolution_m,
            "by_lat": {str(la): round(min_blob_m2(min_blob_px, resolution_m, la))
                       for la in REF_LATS},
            "median_lat": round(med_lat, 2),
            "representative_m2": int(round(repr_m2)),
        },
        "totals": {
            "candidates_n": int(len(candidates)),
            "candidates_km2": round(float(candidates["area_m2"].sum()) / 1e6, 2),
            "t65_n": int(len(t65)),
            "t65_km2": round(float(t65["area_m2"].sum()) / 1e6, 2),
        },
    }


def _compact_arrays(js: str) -> str:
    """Collapse numeric arrays onto one line — a 25-value ladder reads far better
    as one line in the app source than as 25 lines of one number each."""
    return re.sub(r"\[\s+([-\d.,\s]+?)\s+\]",
                  lambda m: "[" + " ".join(m.group(1).split()) + "]", js)


def write_js(stats: dict, out: str | Path, sources: list[str]) -> None:
    """Write the paste-into-the-app JS literal block."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = "\n".join(f"//   {s}" for s in sources)
    out.write_text(
        "// Generated by scripts/build_ee_app_stats.py — do not hand-edit.\n"
        "// Paste this block into post-inference/ee_south_app.js.\n"
        "// Sources:\n" + src + "\n"
        "var APP_STATS = " + _compact_arrays(json.dumps(stats, indent=2)) + ";\n")
    logger.info("Wrote %s (%d ladder steps)", out, len(stats["ladder_m2"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True,
                   help="thr-0.30 inventory attributes (parquet/csv/gpkg)")
    p.add_argument("--t65", required=True,
                   help="0.65 core outlines (parquet/csv/gpkg)")
    p.add_argument("--region-log", required=True,
                   help="region_log.json — SSoT for resolution_m")
    p.add_argument("--deployment-config", default="configs/deployment.yaml",
                   help="SSoT for min_blob_size_px")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    setup_logging()

    candidates = read_table(args.candidates)
    t65 = read_table(args.t65)
    resolution_m = float(json.loads(Path(args.region_log).read_text())["resolution_m"])
    min_blob_px = int(load_config(args.deployment_config)["min_blob_size_px"])

    stats = build_stats(candidates, t65, min_blob_px, resolution_m)
    logger.info("candidates %d (%.1f km²) | t65 %d (%.1f km²) | min_blob %d px "
                "= %d m² at median lat %.2f°",
                stats["totals"]["candidates_n"], stats["totals"]["candidates_km2"],
                stats["totals"]["t65_n"], stats["totals"]["t65_km2"],
                min_blob_px, stats["min_blob"]["representative_m2"],
                stats["min_blob"]["median_lat"])
    write_js(stats, args.out, [str(args.candidates), str(args.t65),
                               str(args.region_log), str(args.deployment_config)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
