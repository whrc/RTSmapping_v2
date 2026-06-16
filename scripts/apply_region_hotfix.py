"""Apply the data team's RegionName hotfix to the v1.0 metadata (leakage fix).

The original v1.0 `metadata.csv` mis-assigned the ecoregion (`RegionName`) of ~59%
of tiles — the `circumpolar_subregions.geojson` was cropped to the inference domain
and a centroid CRS step was off, so tiles got the wrong Dinerstein ecoregion. Because
`scripts/create_splits.py` blocks the train/val/test split *by RegionName*, the wrong
labels leaked tiles across splits (notably into the held-out test set). The data team's
`metadata_region_hotfix.csv` (Robb Young; confirmed by Heidi Rodenhizer) corrects
`RegionName`.

This applies the corrected `RegionName` to OUR tile set (left-join on `Tile_ID`, so our
QC drops and all other columns are preserved; tiles absent from the hotfix keep their old
region). Imagery/labels are unchanged — only the region label moves. Re-run
`scripts/create_splits.py` afterwards to regenerate a leakage-free split.

Usage:
  python scripts/apply_region_hotfix.py \
     --metadata /outputs/v1.0/data_local/metadata.csv \
     --hotfix   /tmp/metadata_region_hotfix.csv \
     --backup   /outputs/v1.0/data_local/metadata_pre_hotfix.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True, type=Path, help="current metadata.csv (overwritten in place)")
    ap.add_argument("--hotfix", required=True, type=Path, help="data-team metadata_region_hotfix.csv")
    ap.add_argument("--backup", required=True, type=Path, help="where to copy the pre-hotfix metadata")
    args = ap.parse_args()

    cur = pd.read_csv(args.metadata, dtype={"Tile_ID": str})
    hot = pd.read_csv(args.hotfix, dtype={"Tile_ID": str})[["Tile_ID", "RegionName"]]
    hot = hot.rename(columns={"RegionName": "_RegionName_hotfix"})

    m = cur.merge(hot, on="Tile_ID", how="left")
    n_missing = int(m["_RegionName_hotfix"].isna().sum())
    n_changed = int(((m["RegionName"] != m["_RegionName_hotfix"]) & m["_RegionName_hotfix"].notna()).sum())
    m["RegionName"] = m["_RegionName_hotfix"].fillna(m["RegionName"])
    m = m.drop(columns=["_RegionName_hotfix"])
    assert list(m["Tile_ID"]) == list(cur["Tile_ID"]), "tile set/order must be preserved"

    if not args.backup.exists():
        shutil.copy2(args.metadata, args.backup)
        print(f"backed up original -> {args.backup}")
    m.to_csv(args.metadata, index=False)

    print(f"tiles: {len(m)}  region changed: {n_changed} ({100*n_changed/len(m):.1f}%)  "
          f"not in hotfix (kept old): {n_missing}")
    print(f"distinct RegionName: before={cur['RegionName'].nunique()} after={m['RegionName'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
