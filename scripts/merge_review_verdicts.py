"""Merge the campaign's verdicts into the verified RTS inventory.

Reads every submitted batch (`review/verdicts/*.jsonl`), pools them into one
verdict per polygon, and writes the verified product family plus the agreement
report. Runs happily on a partial campaign: unreviewed polygons keep a null
verdict and are excluded from the verified cut, so the product is honest at any
stopping point.

Verdict precedence: an item's **coverage** verdict is authoritative; an
injected replicate contributes only to the inter-rater agreement sample.

Outputs (into --out-dir):
    review_verdicts.csv          rts_id, qc_verdict — shaped like qc_ratings.csv
                                 so scripts/score_qc_ratings.py consumes it
    south_rts_verified.gpkg      candidates + verdict columns
    south_rts_verified_true.gpkg qc_verdict == 'rts' — the verified inventory
    qc_false_hard_negatives.gpkg qc_verdict == 'false' — the v3 hard negatives
    review_agreement.json        coverage, per-reviewer counts, Cohen's kappa,
                                 and the confusion matrix vs the 2026-07 pass

Spec: `post-inference/review_campaign.md` §7–§8.

Usage:
    python scripts/merge_review_verdicts.py \
        --verdicts gs://.../2025q3_south/review/verdicts \
        --manifest /outputs/.../manifest.parquet \
        --candidates /outputs/.../south_rts_candidates.gpkg \
        --out-dir /outputs/.../verified \
        [--prior-ratings post-inference/qc_ratings.csv]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

VERDICT_COLS = ["rts_id", "verdict", "reviewer", "batch_id", "injected",
                "reviewed_at"]


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def read_verdicts(location: str) -> pd.DataFrame:
    """Read every `*.jsonl` under a local directory or a ``gs://`` prefix."""
    rows: list[dict] = []
    if location.startswith("gs://"):
        from google.cloud import storage
        bucket_name, _, prefix = location[5:].partition("/")
        client = storage.Client()
        for blob in client.bucket(bucket_name).list_blobs(prefix=prefix):
            if blob.name.endswith(".jsonl"):
                rows += [json.loads(ln) for ln in
                         blob.download_as_text().splitlines() if ln.strip()]
    else:
        for path in sorted(Path(location).glob("*.jsonl")):
            rows += [json.loads(ln) for ln in
                     path.read_text().splitlines() if ln.strip()]
    if not rows:
        return pd.DataFrame(columns=VERDICT_COLS)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# agreement
# --------------------------------------------------------------------------
def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two paired verdict series over a shared vocabulary."""
    if len(a) == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = float((a.to_numpy() == b.to_numpy()).mean())
    pe = sum((a == k).mean() * (b == k).mean() for k in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def replicate_pairs(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Pair each injected replicate with its coverage verdict for the same id."""
    # Deduplicate first: joining on a non-unique index would multiply rows and
    # silently inflate the agreement sample.
    cov = (verdicts[~verdicts["injected"]].sort_values("reviewed_at")
           .drop_duplicates("rts_id", keep="last").set_index("rts_id"))
    rep = verdicts[verdicts["injected"]]
    joined = rep.join(cov[["verdict", "reviewer"]], on="rts_id",
                      rsuffix="_coverage", how="inner")
    return joined.rename(columns={"verdict": "verdict_replicate",
                                  "reviewer": "reviewer_replicate"})


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------
def merge_verdicts(verdicts: pd.DataFrame,
                   manifest: pd.DataFrame) -> pd.DataFrame:
    """Pool verdicts into one authoritative record per reviewed polygon.

    Args:
        verdicts: the raw JSONL records.
        manifest: the campaign manifest, used to reject stray ids.

    Returns:
        One row per polygon that has a coverage verdict, with `qc_verdict`,
        `n_reviews`, `reviewers`, `agreement` and `reviewed_at`.

    Raises:
        ValueError: if a verdict names an `rts_id` the manifest does not carry.
    """
    if verdicts.empty:
        return pd.DataFrame(columns=["rts_id", "qc_verdict", "n_reviews",
                                     "reviewers", "agreement", "reviewed_at"])
    known = set(manifest["rts_id"].astype(int))
    stray = set(verdicts["rts_id"].astype(int)) - known
    if stray:
        raise ValueError(f"verdicts for ids not in the manifest: "
                         f"{sorted(stray)[:5]} ({len(stray)} total)")

    cov = verdicts[~verdicts["injected"]].sort_values("reviewed_at")
    dupes = cov["rts_id"].duplicated().sum()
    if dupes:
        logger.warning("%d polygons carry more than one coverage verdict; "
                       "keeping the latest", dupes)
    cov = cov.drop_duplicates("rts_id", keep="last")

    pairs = replicate_pairs(verdicts).set_index("rts_id")
    agreement = (pairs["verdict_replicate"]
                 == pairs["verdict_coverage"]) if len(pairs) else pd.Series(
                     dtype=bool)

    out = pd.DataFrame({
        "rts_id": cov["rts_id"].astype(int).to_numpy(),
        "qc_verdict": cov["verdict"].to_numpy(),
        "reviewed_at": cov["reviewed_at"].to_numpy(),
    })
    counts = verdicts.groupby("rts_id").size()
    reviewers = verdicts.groupby("rts_id")["reviewer"].apply(
        lambda s: ",".join(sorted(set(s))))
    out["n_reviews"] = out["rts_id"].map(counts).astype(int)
    out["reviewers"] = out["rts_id"].map(reviewers)
    out["agreement"] = out["rts_id"].map(agreement)
    return out[["rts_id", "qc_verdict", "n_reviews", "reviewers", "agreement",
                "reviewed_at"]]


def agreement_report(verdicts: pd.DataFrame, merged: pd.DataFrame,
                     manifest: pd.DataFrame,
                     prior: pd.DataFrame | None = None) -> dict:
    """Coverage, per-reviewer counts, kappa, and drift vs the 2026-07 pass."""
    n_total = int((~manifest["injected"]).sum())
    pairs = replicate_pairs(verdicts)
    cross = pairs[pairs["reviewer_replicate"] != pairs["reviewer_coverage"]] \
        if len(pairs) else pairs

    report = {
        "polygons_total": n_total,
        "polygons_reviewed": int(len(merged)),
        "fraction_reviewed": round(len(merged) / n_total, 4) if n_total else 0.0,
        "verdict_counts": (merged["qc_verdict"].value_counts().to_dict()
                           if len(merged) else {}),
        "per_reviewer": (verdicts.groupby("reviewer").size().to_dict()
                         if len(verdicts) else {}),
        "replicate_pairs": int(len(pairs)),
        "replicate_pairs_cross_reviewer": int(len(cross)),
    }
    if len(cross):
        report["kappa_cross_reviewer"] = round(cohens_kappa(
            cross["verdict_replicate"], cross["verdict_coverage"]), 4)
        report["raw_agreement_cross_reviewer"] = round(float(
            (cross["verdict_replicate"].to_numpy()
             == cross["verdict_coverage"].to_numpy()).mean()), 4)

    if prior is not None and len(merged):
        joined = prior.merge(merged[["rts_id", "qc_verdict"]], on="rts_id",
                             suffixes=("_2026_07", "_campaign"))
        if len(joined):
            report["vs_2026_07"] = {
                "n": int(len(joined)),
                "raw_agreement": round(float(
                    (joined["qc_verdict_2026_07"]
                     == joined["qc_verdict_campaign"]).mean()), 4),
                "kappa": round(cohens_kappa(joined["qc_verdict_2026_07"],
                                            joined["qc_verdict_campaign"]), 4),
                "confusion": (joined.groupby(
                    ["qc_verdict_2026_07", "qc_verdict_campaign"]).size()
                    .unstack(fill_value=0).to_dict()),
            }
    return report


# --------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------
def write_products(candidates: str, merged: pd.DataFrame, out_dir: str) -> None:
    """Join verdicts onto the candidate polygons and cut the verified layers."""
    import geopandas as gpd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(candidates)
    gdf["rts_id"] = gdf["rts_id"].astype(int)
    verified = gdf.merge(merged, on="rts_id", how="left")

    verified.to_file(out / "south_rts_verified.gpkg", driver="GPKG")
    for name, mask in (("south_rts_verified_true", "rts"),
                       ("qc_false_hard_negatives", "false")):
        cut = verified[verified["qc_verdict"] == mask]
        cut.to_file(out / f"{name}.gpkg", driver="GPKG")
        logger.info("%s: %d polygons / %.2f km²", name, len(cut),
                    cut["area_m2"].sum() / 1e6)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verdicts", required=True,
                   help="local dir or gs:// prefix holding the batch JSONLs")
    p.add_argument("--manifest", required=True)
    p.add_argument("--candidates", help="omit to skip the GPKG products")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prior-ratings",
                   help="the 2026-07 qc_ratings.csv, for the drift check")
    args = p.parse_args()
    setup_logging()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.manifest)
    verdicts = read_verdicts(args.verdicts)
    logger.info("read %d verdict records", len(verdicts))

    merged = merge_verdicts(verdicts, manifest)
    merged[["rts_id", "qc_verdict"]].to_csv(out / "review_verdicts.csv",
                                            index=False)

    prior = None
    if args.prior_ratings:
        prior = pd.read_csv(args.prior_ratings)
        prior["rts_id"] = prior["rts_id"].astype(int)
    report = agreement_report(verdicts, merged, manifest, prior)
    (out / "review_agreement.json").write_text(json.dumps(report, indent=2,
                                                          default=str))
    logger.info("reviewed %d/%d polygons (%.1f%%) — %s",
                report["polygons_reviewed"], report["polygons_total"],
                100 * report["fraction_reviewed"], report["verdict_counts"])

    if args.candidates:
        write_products(args.candidates, merged, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
