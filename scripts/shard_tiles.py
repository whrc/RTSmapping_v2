"""Split the pan-Arctic tile list into spatially-contiguous shards (plan Phase 1).

The dual-fleet run is driven by a GCS shard-claim queue (`inference/claim.py`):
workers atomically claim one shard at a time, so the work auto-balances across
the heterogeneous A100 + L4 GPUs. This script produces the shard universe.

Tiles are spatially sorted (reusing `inference.tiles._spatial_sort`, the same
order the per-worker quad cache relies on) and cut into contiguous chunks of
``--shard-size`` tiles. Each shard is one CSV under ``<output>/shards/`` and an
``index.json`` lists every shard id + tile count (the workers' shard universe,
the monitor's totals).

Spatial contiguity is a *cache/egress* optimization only — correctness does not
depend on it. Each 512x512 tile is an independent forward pass writing its own
probability COG; the stride-344 overlap is reconciled only at merge time
(inference.md §4.3). So shard boundaries are invisible downstream; the one
invariant is that every tile lands in exactly one shard.

Usage:
    python scripts/shard_tiles.py \
        --tile-list /mnt/outputs/inference/tiles_2025q3_domain_full.csv \
        --output gs://rts-arctic-usw1/inference/2025q3_south \
        --shard-size 20000
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.tiles import _spatial_sort  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

SHARD_ID_FMT = "shard_{:06d}"


def make_shards(tiles: pd.DataFrame, shard_size: int) -> list[tuple[str, pd.DataFrame]]:
    """Spatially sort and split into contiguous shards of <= shard_size tiles.

    Args:
        tiles: tile list with columns tile_id, minx, miny, maxx, maxy.
        shard_size: max tiles per shard (last shard may be smaller).

    Returns:
        List of (shard_id, shard_df) in spatial order. Every input tile appears
        in exactly one shard; shard order is the spatial sort order.
    """
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    ordered = _spatial_sort(tiles).reset_index(drop=True)
    shards: list[tuple[str, pd.DataFrame]] = []
    for i, start in enumerate(range(0, len(ordered), shard_size)):
        chunk = ordered.iloc[start:start + shard_size].reset_index(drop=True)
        shards.append((SHARD_ID_FMT.format(i), chunk))
    return shards


def _write_text(path: str, text: str) -> None:
    """Write text to a local path or gs:// URI."""
    if str(path).startswith("gs://"):
        import gcsfs

        with gcsfs.GCSFileSystem(token="google_default").open(path[5:], "w") as f:
            f.write(text)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tile-list", required=True, help="tile CSV (tile_id,minx,miny,maxx,maxy)")
    p.add_argument("--output", required=True,
                   help="base prefix (local or gs://); shards go under <output>/shards/")
    p.add_argument("--shard-size", type=int, default=20000,
                   help="tiles per shard (default 20000 ~ the Phase-3 benchmark starting point)")
    args = p.parse_args()
    setup_logging()

    tiles = pd.read_csv(args.tile_list)
    shards = make_shards(tiles, args.shard_size)
    base = args.output.rstrip("/")
    logger.info("%d tiles -> %d shards of <= %d (output %s/shards/)",
                len(tiles), len(shards), args.shard_size, base)

    index = {"n_tiles": int(len(tiles)), "n_shards": len(shards),
             "shard_size": args.shard_size, "shards": []}
    for shard_id, chunk in shards:
        buf = io.StringIO()
        chunk.to_csv(buf, index=False)
        _write_text(f"{base}/shards/{shard_id}.csv", buf.getvalue())
        index["shards"].append({"shard_id": shard_id, "n_tiles": int(len(chunk))})

    _write_text(f"{base}/shards/index.json", json.dumps(index, indent=1))
    logger.info("wrote shards/index.json (%d shards, %d tiles)",
                len(shards), len(tiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
