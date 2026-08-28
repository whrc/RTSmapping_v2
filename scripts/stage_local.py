"""Stage v1.0 training tiles from GCS to local disk (experiments.md §13).

Per-epoch reads via rasterio /vsigs/ stall the GPU (no gcsfuse mount → the
training.gcsfuse cache is inert). Copy the per-tile rasters to local SSD once and
point `data.data_root` at the local path so reads are local and the GPU saturates.

Resumable (skips files already present with matching size). Run inside rts-train:v2
with the user ADC mounted (the VM's default compute SA cannot read rts-mapping-v2).

Usage: python scripts/stage_local.py [--workers 64] [--dst /outputs/v1.0/data_local]
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_local")

PROJECT = "abruptthawmapping"
BUCKET = "rts-arctic-us"
PREFIX = "training/v1.0/"
SUBDIRS = ("PLANET-RGB", "labels")


def stage(dst_root: Path, workers: int) -> None:
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    jobs = []
    for sub in SUBDIRS:
        for blob in client.list_blobs(bucket, prefix=PREFIX + sub + "/"):
            if blob.name.endswith("/"):
                continue
            local = dst_root / blob.name[len(PREFIX):]
            jobs.append((blob, local))
    logger.info("staging %d files -> %s", len(jobs), dst_root)

    def one(job):
        blob, local = job
        if local.exists() and local.stat().st_size == (blob.size or -1):
            return "skip"
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))
        return "copy"

    copied = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result(); copied += r == "copy"; skipped += r == "skip"
            except Exception as e:  # noqa: BLE001
                failed += 1; logger.error("FAILED %s: %r", futs[fut][1], e)
            if i % 2000 == 0:
                logger.info("  %d/%d (copied=%d skipped=%d failed=%d)", i, len(jobs), copied, skipped, failed)
    logger.info("DONE copied=%d skipped=%d failed=%d of %d", copied, skipped, failed, len(jobs))
    if failed:
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dst", default="/outputs/v1.0/data_local", type=Path)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    stage(args.dst, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
