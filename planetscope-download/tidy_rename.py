"""Optional: flatten Planet's order-UUID directories out of a delivered year.

**Not required.** `inference/quad_index.py` lists recursively and matches on the
basename, so it indexes the raw delivery and the flattened layout identically.
This exists only so the bucket reads tidily by hand.

Heidi's original version tracked a `renaming_guide` DataFrame and a batch delete
across cells, which lost its place whenever a batch failed. This derives the
work from bucket state on every run instead, so it needs no saved plan and no
checkpoint: re-running after any failure simply recomputes what is left. Two
consequences worth knowing:

* copy is skipped when the destination already exists, so a half-finished run
  costs only the listing to resume;
* deletes are per-object with 404 tolerated, rather than
  `storage_client.batch()`, which aborts the whole batch on one missing object —
  the failure mode that made re-runs unreliable.

    gs://<bucket>/global_quarterly/2022/q3/338/1474/<uuid>/<mosaic>/338-1474_quad.tif
 -> gs://<bucket>/global_quarterly/2022/q3/338/1474/<mosaic>_338-1474_quad.tif

Usage:
    python planetscope-download/tidy_rename.py --year 2022 --dry-run
    python planetscope-download/tidy_rename.py --year 2022
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

UUID_DIR_RE = re.compile(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/",
                         re.IGNORECASE)


def flatten_name(name: str) -> str:
    """Strip the order-UUID directory and fold the mosaic directory into the stem.

    Args:
        name: Blob name as delivered.

    Returns:
        The flattened blob name (unchanged if there is no UUID directory).
    """
    out = UUID_DIR_RE.sub("/", name)
    # ".../<mosaic_name>/<col>-<row>_quad.tif" -> ".../<mosaic_name>_<col>-<row>_quad.tif"
    return re.sub(r"/(global_quarterly_\d{4}q\d_mosaic)/", r"/\1_", out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--bucket", default="pdg-planet-data")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    setup_logging()

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    prefix = f"global_quarterly/{args.year}/q3/"

    all_names = {b.name for b in
                 client.list_blobs(args.bucket, prefix=prefix,
                                   fields="items(name),nextPageToken")}
    todo = [(n, flatten_name(n)) for n in sorted(all_names) if UUID_DIR_RE.search(n)]
    todo = [(src, dst) for src, dst in todo if src != dst]

    if not todo:
        logger.info("Nothing to flatten under gs://%s/%s", args.bucket, prefix)
        return 0
    logger.info("%d objects to flatten under gs://%s/%s", len(todo), args.bucket, prefix)
    if args.dry_run:
        for src, dst in todo[:10]:
            logger.info("  %s\n    -> %s", src, dst)
        logger.info("Dry run — nothing written.")
        return 0

    def move(pair: tuple[str, str]) -> None:
        src, dst = pair
        try:
            if dst not in all_names:      # resume: skip what a prior run copied
                bucket.copy_blob(bucket.blob(src), bucket, dst)
            bucket.blob(src).delete()
        except Exception as e:  # noqa: BLE001
            from google.api_core.exceptions import NotFound
            if isinstance(e, NotFound):
                return                     # already moved by a prior run
            logger.warning("failed on %s: %s", src, e)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(move, todo))

    left = sum(1 for b in client.list_blobs(args.bucket, prefix=prefix,
                                            fields="items(name),nextPageToken")
               if UUID_DIR_RE.search(b.name))
    logger.info("Done. %d objects still under a UUID directory "
                "(re-run to finish; it is idempotent).", left)
    return 0 if left == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
