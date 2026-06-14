"""Re-stage the v1.0 snapshot: restore wrongly-dropped positives and drop the
fully-black negatives, WITHOUT disturbing the frozen baseline.

Background (docs/phase0_baseline.md "Baseline & re-baseline policy"): μ₀, the gate,
the norm stats and the val/test split are frozen for the project. A fresh QC
(docs/v1.0_qc.md) showed the *current* v1.0 already contains 555/564 of the
"degraded" negatives (so they're already in the locked baseline — only 8
train-region ones are missing) and all 49 fully-black negatives (never dropped),
while the 28 valid positives were wrongly dropped (a stale-QC bug). So the real
delta is small:

  ADD (train-region, genuinely missing): 27 positives + 8 degraded negatives.
  REMOVE: the 49 fully-black negatives (all-zero, useless).
  Excluded: any add that falls in a val/test region (keep the gate's val_realistic
            untouched). Black removal DOES touch val_balanced/test_realistic
            (13+5 tiles) — chosen deliberately (Option 1): all-zero tiles have no
            place in any eval set, and the gate set val_realistic has zero black,
            so μ₀ comparability holds.

Writes metadata_restage.csv (= frozen metadata.csv − black + new rows); the
original metadata.csv stays frozen. splits.yaml is unchanged (adds are train-region;
removals only prune tile_ids the splits resolve from metadata). A config opts in via
data.metadata_csv: metadata_restage.csv + data.nodata_handling: true (§4.4 neutralizes
the ~563 already-present degraded negs). NO re-baseline, NO norm recompute (locked).

Run inside rts-train:v2 with the user ADC mounted and /mnt/outputs at /outputs.
Usage: python scripts/restage_v1_additive.py [--workers 16] [--no-copy-tiles]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from google.cloud import storage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make top-level pkgs importable
from data.splits import load_splits_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("restage_v1")

PROJECT = "pdg-project-406720"
SRC_BUCKET = "abrupt_thaw"
SRC_PREFIX = "RTS_MODEL_V2/DATA/TRAINING_DATA/"
DST_BUCKET = "rts-mapping-v2"
DST_PREFIX = "training/v1.0/"
KNOWN_ISSUES = "/outputs/v1.0/qc/known_issues_v1.0.json"


def build_restage(client: storage.Client):
    """Return (restaged_metadata_df, manifest, add_neg, add_pos, remove_ids)."""
    tmp = tempfile.mkdtemp()
    with open(KNOWN_ISSUES) as f:
        ki = json.load(f)["next_restage_actions"]
    keep_neg = ki["drop_negatives_degraded_band_gt50pct"]
    restore_pos = ki["restore_positives_wrongly_dropped"]
    black = set(ki["drop_negatives_black"])

    client.bucket(SRC_BUCKET).blob(SRC_PREFIX + "metadata.csv").download_to_filename(tmp + "/src.csv")
    client.bucket(DST_BUCKET).blob(DST_PREFIX + "metadata.csv").download_to_filename(tmp + "/v1.csv")
    client.bucket(DST_BUCKET).blob(DST_PREFIX + "splits.yaml").download_to_filename(tmp + "/splits.yaml")
    src = pd.read_csv(tmp + "/src.csv", dtype={"Tile_ID": str, "UIDs": str}).set_index("Tile_ID")
    v1 = pd.read_csv(tmp + "/v1.csv", dtype={"Tile_ID": str, "UIDs": str})
    v1_ids = set(v1["Tile_ID"])
    region_split = {r: s for s, rs in load_splits_yaml(tmp + "/splits.yaml").items() for r in rs}

    # Candidates to add: train-region tiles from either QC list that are missing
    # from v1.0. Classify by the tile's TRUE source class (the "degraded-neg" list
    # turns out to contain positives) so labels get copied for every positive.
    cand = [t for t in dict.fromkeys(list(keep_neg) + list(restore_pos))
            if t not in v1_ids
            and t in src.index
            and region_split.get(src.loc[t, "RegionName"]) == "train"]
    add_ids = cand
    add_label_ids = [t for t in cand if src.loc[t, "TrainClass"] == "positive"]
    remove_ids = sorted(black & v1_ids)

    kept = v1[~v1["Tile_ID"].isin(remove_ids)]
    new_rows = src.loc[add_ids].reset_index()[v1.columns.tolist()]
    full = pd.concat([kept, new_rows], ignore_index=True)
    assert full["Tile_ID"].duplicated().sum() == 0, "duplicate Tile_IDs in restaged metadata"

    n_pos = len(add_label_ids); n_neg = len(add_ids) - n_pos
    manifest = {
        "added_positives": n_pos,
        "added_negatives": n_neg,
        "removed_black": len(remove_ids),
        "v1_total": len(v1),
        "restage_total": len(full),
        "added_ids": add_ids,
        "removed_ids": remove_ids,
        "note": "use metadata_restage.csv + data.nodata_handling=true; splits.yaml unchanged; gate val_realistic untouched",
    }
    logger.info("restage: +%d pos +%d neg (train-region), -%d black | v1 %d -> restage %d",
                n_pos, n_neg, len(remove_ids), len(v1), len(full))
    return full, manifest, add_ids, add_label_ids, remove_ids


def copy_tiles(client: storage.Client, add_ids: list[str], add_label_ids: list[str], workers: int) -> None:
    src, dst = client.bucket(SRC_BUCKET), client.bucket(DST_BUCKET)
    jobs = [(SRC_PREFIX + f"PLANET-RGB/{t}.tif", DST_PREFIX + f"PLANET-RGB/{t}.tif") for t in add_ids]
    jobs += [(SRC_PREFIX + f"labels/{t}.tif", DST_PREFIX + f"labels/{t}.tif") for t in add_label_ids]

    def one(job):
        s, d = job
        sb = src.blob(s); sb.reload()
        ex = dst.get_blob(d)
        if ex is not None and ex.size == sb.size:
            return "skip"
        src.copy_blob(sb, dst, d)
        return "copy"

    copied = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                r = fut.result(); copied += r == "copy"; skipped += r == "skip"
            except Exception as e:  # noqa: BLE001
                failed += 1; logger.error("FAILED %s: %r", futs[fut][1], e)
    logger.info("tiles: copied=%d skipped=%d failed=%d (of %d)", copied, skipped, failed, len(jobs))
    if failed:
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-copy-tiles", dest="copy_tiles", action="store_false", default=True)
    args = ap.parse_args()

    client = storage.Client(project=PROJECT)
    full, manifest, add_ids, add_label_ids, _ = build_restage(client)
    if args.copy_tiles:
        copy_tiles(client, add_ids, add_label_ids, args.workers)
    bkt = client.bucket(DST_BUCKET)
    bkt.blob(DST_PREFIX + "metadata_restage.csv").upload_from_string(
        full.to_csv(index=False), content_type="text/csv")
    bkt.blob(DST_PREFIX + "restage_manifest.json").upload_from_string(
        json.dumps(manifest, indent=2), content_type="application/json")
    logger.info("wrote metadata_restage.csv (%d rows) + restage_manifest.json", len(full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
