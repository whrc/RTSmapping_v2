"""v3-alpha Phase 0.1 / 0.2: where do the census false positives sit?

Reads the review-status snapshot (one row per candidate polygon, from
``export_candidates_geojson.py``) and the candidate attribute table, then reports:

  0.1a  false share by max_prob / area / nodata_frac / latitude / region
  0.1b  removable bound: how many false polygons have no verified RTS in their own
        tile. A tile-level existence gate (``cgm``) can only remove these.
  0.2   decided verdicts per split (train / val_realistic / test_realistic / unassigned)

The tile is the model input: 512 px at 4.7773 m (EPSG:3857, Web Mercator z15), so a
tile centred on a polygon covers +-1,223 m *projected* in each axis. Neighbours are
therefore found in EPSG:3857 with a Chebyshev (square-window) test, not in ground
metres: at 70N a projected 1,223 m is only ~420 m on the ground.

Two bounds are reported for the removable share:
  * lower: the tile holds no other candidate that is rts / unsure / unreviewed
           (an unreviewed neighbour might be a real slump, so this is certain);
  * upper: the tile holds no *verified* rts (unreviewed neighbours count as not-rts).

Usage:
    python scripts/diag_fp_census.py \
        --snapshot rts_candidates_review_status_<stamp>.geojson.gz \
        --attrs south_rts_attributes.parquet \
        --regions circumpolar_subregions.geojson --splits splits.yaml \
        --out-dir out/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Web Mercator zoom-15 pixel (constant in projected space) and the model input tile.
PIXEL_M = 156543.03392804097 / 2**15
TILE_PX = 512
TILE_HALF_M = TILE_PX / 2 * PIXEL_M

DECIDED = ("rts", "false_positive")
OPEN = ("rts", "unsure", "unverified")  # statuses that may hide a real slump

BANDS: dict[str, list[float]] = {
    "max_prob": [0.30, 0.45, 0.65, 0.80, 0.90, 0.95, 1.0001],
    "area_m2": [0.0, 500.0, 2000.0, 10000.0, 50000.0, float("inf")],
    "nodata_frac": [0.0, 0.01, 0.05, 0.20, 1.0001],
    "abs_lat": [0.0, 55.0, 60.0, 65.0, 70.0, 90.0],
}

# Counts documented in post-inference/south_products.md for the 2026-11-17 snapshot.
DOCUMENTED = {"rts": 12372, "false_positive": 9636, "unsure": 5792, "unverified": 32367}


def read_snapshot(path: str | Path) -> pd.DataFrame:
    """Read the property table (no geometry) of the review-status GeoJSON.

    Args:
        path: ``.geojson`` or ``.geojson.gz`` written by ``export_candidates_geojson.py``.

    Returns:
        One row per candidate polygon, with an integer ``rts_id``.
    """
    import pyogrio

    p = str(Path(path).resolve()).replace("\\", "/")
    src = f"/vsigzip/{p}" if p.endswith(".gz") else p
    df = pyogrio.read_dataframe(src, read_geometry=False)
    df["rts_id"] = df["rts_id"].astype(int)
    return pd.DataFrame(df)


def attach_attrs(snap: pd.DataFrame, attrs_path: str | Path) -> pd.DataFrame:
    """Join ``nodata_frac`` (not in the snapshot) from the attribute table by ``rts_id``."""
    attrs = pd.read_parquet(attrs_path, columns=["rts_id", "nodata_frac"])
    attrs["rts_id"] = attrs["rts_id"].astype(int)
    out = snap.merge(attrs, on="rts_id", how="left", validate="one_to_one")
    n_missing = int(out["nodata_frac"].isna().sum())
    if n_missing:
        logger.warning("%d polygons have no nodata_frac in the attribute table", n_missing)
    out["abs_lat"] = out["centroid_lat"].abs()
    return out


def check_documented_counts(df: pd.DataFrame) -> dict[str, int]:
    """Compare status counts with the documented snapshot; warn on any difference."""
    got = {k: int((df["status"] == k).sum()) for k in DOCUMENTED}
    for k, want in DOCUMENTED.items():
        if got[k] != want:
            logger.warning("status %-14s %6d (documented %6d) - snapshot differs", k, got[k], want)
    return got


def to_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat (degrees) to EPSG:3857 metres."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = t.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def removable_bounds(df: pd.DataFrame, half_m: float = TILE_HALF_M) -> pd.DataFrame:
    """Flag, for every polygon, whether its centred tile holds other candidates.

    Adds boolean columns ``tile_has_rts`` (another verified rts in the tile) and
    ``tile_has_open`` (another rts / unsure / unreviewed polygon in the tile). The
    tile is a square window of +-``half_m`` projected metres around the centroid.

    Args:
        df: needs ``centroid_lon``, ``centroid_lat`` and ``status``.
        half_m: half the tile side in EPSG:3857 metres.
    """
    x, y = to_mercator(df["centroid_lon"].to_numpy(), df["centroid_lat"].to_numpy())
    tree = cKDTree(np.column_stack([x, y]))
    status = df["status"].to_numpy()
    is_rts = status == "rts"
    is_open = np.isin(status, OPEN)
    neigh = tree.query_ball_point(np.column_stack([x, y]), r=half_m, p=np.inf)
    has_rts = np.zeros(len(df), dtype=bool)
    has_open = np.zeros(len(df), dtype=bool)
    for i, idx in enumerate(neigh):
        others = [j for j in idx if j != i]
        if others:
            has_rts[i] = bool(is_rts[others].any())
            has_open[i] = bool(is_open[others].any())
    out = df.copy()
    out["tile_has_rts"] = has_rts
    out["tile_has_open"] = has_open
    return out


def removable_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Lower and upper bound on the share of false polygons a tile-level gate could remove."""
    fp = df[df["status"] == "false_positive"]
    n = len(fp)
    return {
        "n_false": n,
        "removable_lower": float((~fp["tile_has_open"]).mean()) if n else float("nan"),
        "removable_upper": float((~fp["tile_has_rts"]).mean()) if n else float("nan"),
    }


def rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group verdict counts, false rate among decided, and share of all false polygons."""
    counts = (
        df.groupby(by, observed=True)["status"].value_counts().unstack(fill_value=0)
    )
    for col in ("rts", "false_positive", "unsure", "unverified"):
        if col not in counts.columns:
            counts[col] = 0
    counts["n_decided"] = counts["rts"] + counts["false_positive"]
    counts["false_rate"] = counts["false_positive"] / counts["n_decided"].where(counts["n_decided"] > 0)
    # Denominator is every false polygon, not just those in a group: a NaN group key (e.g. a centroid
    # outside every region) is dropped by groupby and would otherwise inflate every share.
    total_fp = max(int((df["status"] == "false_positive").sum()), 1)
    counts["share_of_all_false"] = counts["false_positive"] / total_fp
    return counts[["rts", "false_positive", "unsure", "unverified",
                   "n_decided", "false_rate", "share_of_all_false"]]


def confidence_profile(df: pd.DataFrame,
                       cuts: tuple[float, ...] = (0.45, 0.65, 0.80, 0.90, 0.95)) -> dict:
    """How well does ``max_prob`` already rank verified rts above false polygons?

    This is the baseline any tile-level gate has to beat. Needs ``status``,
    ``max_prob``, ``conf_class``, ``region`` and ``split``; unsure/unreviewed are ignored.

    Returns:
        precision (and FPs kept) when keeping ``max_prob >= cut``; AUROC overall, in the
        high tier, and per validation region; and how concentrated the FPs are by region.
    """
    from sklearn.metrics import roc_auc_score

    dec = df[df["status"].isin(DECIDED)].copy()
    dec["y_rts"] = (dec["status"] == "rts").astype(int)

    def auc(d: pd.DataFrame) -> float:
        if d["y_rts"].nunique() < 2:
            return float("nan")
        return float(roc_auc_score(d["y_rts"], d["max_prob"]))

    out: dict = {"precision_at_max_prob": {}, "auroc_by_val_region": {}}
    for c in cuts:
        k = dec[dec["max_prob"] >= c]
        out["precision_at_max_prob"][f"{c:.2f}"] = {
            "n": int(len(k)), "precision": float(k["y_rts"].mean()),
            "false_kept": int((1 - k["y_rts"]).sum()),
        }
    out["auroc_all"] = auc(dec)
    out["auroc_high_tier"] = auc(dec[dec["conf_class"] == "high"])
    val = dec[dec["split"] == "val_realistic"]
    out["auroc_val_pooled"] = auc(val)
    for name, g in val.groupby("region"):
        out["auroc_by_val_region"][name] = {
            "n_rts": int(g["y_rts"].sum()), "n_false": int((1 - g["y_rts"]).sum()), "auroc": auc(g)}

    fp = dec[dec["y_rts"] == 0]
    out["share_of_false_below_0.9"] = float((fp["max_prob"] < 0.9).mean())
    out["share_of_false_below_0.8"] = float((fp["max_prob"] < 0.8).mean())
    reg = dec.groupby("region").agg(n=("y_rts", "size"), n_false=("y_rts", lambda s: int((1 - s).sum())),
                                    n_rts=("y_rts", "sum"))
    reg["false_rate"] = reg["n_false"] / reg["n"]
    total_fp = max(int((dec["y_rts"] == 0).sum()), 1)   # all false polygons, incl. those with no region
    for thr in (0.8, 0.9):
        sel = reg[(reg["false_rate"] >= thr) & (reg["n"] >= 20)]
        out[f"false_regions_ge_{thr}"] = {
            "n_regions": int(len(sel)), "n_false": int(sel["n_false"].sum()),
            "share_of_all_false": float(sel["n_false"].sum() / total_fp),
            "n_rts_there": int(sel["n_rts"].sum())}
    top5 = reg.sort_values("n_false", ascending=False).head(5)
    out["top5_false_regions"] = {r: int(v) for r, v in top5["n_false"].items()}
    out["top5_share_of_all_false"] = float(top5["n_false"].sum() / total_fp)
    return out


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<col>_band`` columns using the fixed edges in ``BANDS``."""
    out = df.copy()
    for col, edges in BANDS.items():
        out[f"{col}_band"] = pd.cut(out[col], bins=edges, right=False)
    return out


def assign_split(df: pd.DataFrame, regions_path: str | Path,
                 splits_path: str | Path) -> pd.DataFrame:
    """Add ``region`` (ECO_NAME) and ``split`` by locating each centroid in the region polygons."""
    import geopandas as gpd

    regions = gpd.read_file(regions_path)[["ECO_NAME", "geometry"]]
    pts = gpd.GeoDataFrame(
        df[["rts_id"]].copy(),
        geometry=gpd.points_from_xy(df["centroid_lon"], df["centroid_lat"]),
        crs="EPSG:4326",
    ).to_crs(regions.crs)
    joined = gpd.sjoin(pts, regions, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    out = df.copy()
    out["region"] = joined["ECO_NAME"].reindex(df.index).to_numpy()
    with Path(splits_path).open(encoding="utf-8") as f:
        split_lists = yaml.safe_load(f)
    lookup = {r: s for s in ("train", "val_realistic", "test_realistic")
              for r in split_lists.get(s, [])}
    out["split"] = out["region"].map(lookup).fillna("unassigned")
    return out


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--attrs", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    setup_logging()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = attach_attrs(read_snapshot(args.snapshot), args.attrs)
    logger.info("%d polygons; status counts %s", len(df), check_documented_counts(df))
    df = add_bands(removable_bounds(assign_split(df, args.regions, args.splits)))

    summary: dict = {
        "n_polygons": len(df),
        "status_counts": df["status"].value_counts().to_dict(),
        "tile_half_m_projected": TILE_HALF_M,
        "removable_all": removable_summary(df),
        "confidence_profile": confidence_profile(df),
        "removable_by_max_prob_band": {
            str(k): removable_summary(g) for k, g in df.groupby("max_prob_band", observed=True)
        },
    }
    for by in ("max_prob_band", "area_m2_band", "nodata_frac_band", "abs_lat_band",
               "conf_class", "region", "split"):
        tab = rate_table(df, by)
        tab.to_csv(out_dir / f"false_rate_by_{by}.csv")
        logger.info("false rate by %s:\n%s", by, tab.round(3).to_string())

    with (out_dir / "fp_census_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("removable (all tiers): %s", summary["removable_all"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
