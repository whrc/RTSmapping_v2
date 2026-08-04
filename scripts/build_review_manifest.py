"""Build the review campaign queue: manifest.parquet.

Orders every candidate polygon by descending `max_prob` and cuts the sequence
into fixed-size batches, so at any point in the campaign the claim
"every polygon with max_prob >= p is human-reviewed" holds at batch
granularity. Item order *within* a batch is shuffled, which keeps that property
while denying a reviewer a run of 200 consecutive identical calls.

A small number of already-queued items are additionally **injected** into later
batches as replicates; those rows carry `injected = True`, are excluded from
coverage accounting, and give the merge step its inter-rater agreement sample.

Deterministic: same inventory in, byte-identical manifest out.

Spec: `post-inference/review_campaign.md` §3, §5.

Usage:
    python scripts/build_review_manifest.py \
        --attributes /outputs/.../south_rts_attributes.parquet \
        --out /outputs/.../manifest.parquet [--batch-size 200]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

SEED = 42
REPLICATE_OFFSET = 3  # inject a replicate this many batches after its source

# Context columns carried into the manifest so the app needs no second lookup.
CONTEXT_COLS = ["max_prob", "conf_class", "area_m2", "centroid_lat",
                "centroid_lon"]


def _batch_id(i: int) -> str:
    return f"b{i:05d}"


def build_manifest(attributes: str, batch_size: int = 200,
                   n_replicates: int = 300) -> pd.DataFrame:
    """Order, batch, inject replicates, and shuffle within batches.

    Args:
        attributes: path to `south_rts_attributes.parquet`.
        batch_size: polygons per batch (coverage rows only; injected rows ride
            on top, so a batch may hold slightly more items).
        n_replicates: how many items to re-serve to a second reviewer.

    Returns:
        The manifest, one row per queue item, sorted by (batch_id, seq).
    """
    df = pd.read_parquet(attributes, columns=["rts_id"] + CONTEXT_COLS)
    if df["rts_id"].duplicated().any():
        raise ValueError("duplicate rts_id in the attribute table")

    # max_prob descending; rts_id ascending breaks ties so the order is total.
    df = df.sort_values(["max_prob", "rts_id"], ascending=[False, True],
                        ignore_index=True)
    df["batch_id"] = [_batch_id(i // batch_size) for i in range(len(df))]
    df["injected"] = False
    n_batches = (len(df) + batch_size - 1) // batch_size

    # Replicates: uniformly spaced through the queue, re-served REPLICATE_OFFSET
    # batches later. The offset makes a different claimer likely; the merge only
    # counts pairs whose reviewers actually differ, so it never has to be sure.
    #
    # Sources are drawn only from batches that can take the offset without
    # running off the end. Clamping instead would land a replicate in its own
    # batch, putting one rts_id in a batch twice — which a verdict map keyed by
    # rts_id cannot represent.
    if n_replicates > 0:
        last_eligible = (n_batches - 1 - REPLICATE_OFFSET) * batch_size
        eligible = df.iloc[:max(0, last_eligible)]
        if len(eligible) < n_replicates:
            logger.warning("only %d of %d requested replicates fit before the "
                           "queue tail", len(eligible), n_replicates)
            n_replicates = len(eligible)
        if n_replicates:
            stride = max(1, len(eligible) // n_replicates)
            src = eligible.iloc[::stride].head(n_replicates).copy()
            src["injected"] = True
            src["batch_id"] = [_batch_id(int(b[1:]) + REPLICATE_OFFSET)
                               for b in src["batch_id"]]
            df = pd.concat([df, src], ignore_index=True)

    # Shuffle within each batch. The seed is derived from the batch index, so
    # the result never depends on group iteration order.
    order = []
    for bid, grp in df.groupby("batch_id", sort=True):
        rng = np.random.default_rng([SEED, int(bid[1:])])
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        order.append(pd.DataFrame({"index": idx, "batch_id": bid,
                                   "seq": np.arange(len(idx))}))
    seq = pd.concat(order, ignore_index=True).set_index("index")["seq"]
    df["seq"] = seq
    df = df.sort_values(["batch_id", "seq"], ignore_index=True)

    coverage = df.loc[~df["injected"], "rts_id"]
    if coverage.duplicated().any() or len(coverage) != len(
            pd.read_parquet(attributes, columns=["rts_id"])):
        raise ValueError("coverage invariant broken: every rts_id must appear "
                         "exactly once with injected=False")
    return df[["rts_id", "batch_id", "seq", "injected"] + CONTEXT_COLS]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attributes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--n-replicates", type=int, default=300)
    args = p.parse_args()
    setup_logging()

    man = build_manifest(args.attributes, args.batch_size, args.n_replicates)
    man.to_parquet(args.out, index=False)

    n_cov = int((~man["injected"]).sum())
    summary = {
        "items": len(man),
        "coverage_items": n_cov,
        "injected_items": int(man["injected"].sum()),
        "batches": man["batch_id"].nunique(),
        "batch_size": args.batch_size,
        "max_prob_first": float(man["max_prob"].max()),
        "max_prob_last": float(man["max_prob"].min()),
        "area_km2": float(man.loc[~man["injected"], "area_m2"].sum() / 1e6),
    }
    Path(args.out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2))
    logger.info("wrote %s — %d items (%d coverage + %d injected) in %d batches",
                args.out, summary["items"], n_cov, summary["injected_items"],
                summary["batches"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
