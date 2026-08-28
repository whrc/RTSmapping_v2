"""Stage a frozen v1.0 training snapshot into our own bucket.

One-off operational script (re-baseline after the v2-alpha loss, 2026-06-13). Mirrors the
regenerated training data out of the data-production bucket (which was rewritten in place
once already) into gs://rts-mapping-v2/training/v1.0/ so reproducibility cannot be destroyed
by another external rewrite. See docs/v1.0_rebaseline.md and computing/migrate_vm.md.

Actions:
  1. Read source metadata.csv; drop QC-flagged tiles (empty-mask positives + >50%-zero-band
     tiles from /mnt/outputs/inference/v21_qc); write cleaned metadata.csv + a relabel-candidate
     list (the empty-mask positives are a label bug, flagged back to the data team).
  2. Server-side copy PLANET-RGB/, labels/, and the data-team TESTING/ subtree (idempotent:
     skip dest blobs whose size already matches), plus the regions GeoJSON.

Auth: google-cloud-storage via the mounted user ADC (the gsutil/gcloud CLI auth is broken on
this VM, but the SDK works — verified). Run inside the rts-train:v2 container.

Usage (in Docker, ADC mounted, GOOGLE_CLOUD_PROJECT set):
  python scripts/stage_v1_snapshot.py [--workers 32] [--copy-tiles/--no-copy-tiles]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_v1")

PROJECT = "abruptthawmapping"
SRC_BUCKET = "abrupt_thaw"
SRC_PREFIX = "RTS_MODEL_V2/DATA/TRAINING_DATA/"
SRC_GEOJSON = "RTS_MODEL_V2/DATA/circumpolar_subregions.geojson"
DST_BUCKET = "rts-arctic-us"
DST_PREFIX = "training/v1.0/"
QC_DIR = Path("/outputs/inference/v21_qc")
SUBTREES = ["PLANET-RGB/", "labels/", "TESTING/"]


def build_cleaned_metadata(client: storage.Client) -> tuple[pd.DataFrame, list[str]]:
    """Load source metadata, drop QC-flagged tiles, return (cleaned_df, dropped_ids)."""
    src = f"gs://{SRC_BUCKET}/{SRC_PREFIX}metadata.csv"
    meta = pd.read_csv(src)
    ids = set(meta["Tile_ID"])

    empty = pd.read_csv(QC_DIR / "empty_mask_positives.csv")
    empty_ids = set(empty["Tile_ID"]) & ids

    import ast
    qt = pd.read_csv(QC_DIR / "qc_per_tile.csv")
    qt["maxz"] = qt["band_zero_frac"].apply(lambda s: max(ast.literal_eval(s)))
    degraded_ids = set(qt.loc[qt["maxz"] > 0.5, "tile_id"]) & ids

    drop = empty_ids | degraded_ids
    cleaned = meta[~meta["Tile_ID"].isin(drop)].reset_index(drop=True)
    logger.info(
        "metadata: %d total (%d pos / %d neg) → drop %d (empty-mask %d, degraded %d) → cleaned %d",
        len(meta), (meta.TrainClass == "positive").sum(), (meta.TrainClass == "negative").sum(),
        len(drop), len(empty_ids), len(degraded_ids), len(cleaned),
    )

    # Persist cleaned metadata + provenance to the destination snapshot.
    dst = client.bucket(DST_BUCKET)
    dst.blob(DST_PREFIX + "metadata.csv").upload_from_string(
        cleaned.to_csv(index=False), content_type="text/csv")
    dst.blob(DST_PREFIX + "dropped_tiles.json").upload_from_string(
        json.dumps({"empty_mask_positives": sorted(empty_ids),
                    "degraded_band_gt50pct": sorted(degraded_ids)}, indent=2),
        content_type="application/json")
    # Relabel candidates: empty-mask positives are a label bug for the data team.
    relabel = empty[empty["Tile_ID"].isin(empty_ids)]
    dst.blob(DST_PREFIX + "relabel_candidates.csv").upload_from_string(
        relabel.to_csv(index=False), content_type="text/csv")
    return cleaned, sorted(drop)


def copy_subtrees(client: storage.Client, workers: int) -> None:
    """Idempotent server-side copy of the tile subtrees + GeoJSON src→dst."""
    src = client.bucket(SRC_BUCKET)
    dst = client.bucket(DST_BUCKET)

    jobs: list[tuple[str, str, int]] = []  # (src_name, dst_name, src_size)
    for sub in SUBTREES:
        for b in client.list_blobs(src, prefix=SRC_PREFIX + sub):
            if b.name.endswith("/"):
                continue
            jobs.append((b.name, DST_PREFIX + b.name[len(SRC_PREFIX):], b.size))
    jobs.append((SRC_GEOJSON, DST_PREFIX + "circumpolar_subregions.geojson", None))
    logger.info("copy plan: %d objects across %s + geojson", len(jobs), SUBTREES)

    def one(job: tuple[str, str, int]) -> str:
        src_name, dst_name, src_size = job
        existing = dst.get_blob(dst_name)
        if existing is not None and src_size is not None and existing.size == src_size:
            return "skip"
        src_blob = src.blob(src_name)
        src.copy_blob(src_blob, dst, dst_name)
        return "copy"

    copied = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                copied += r == "copy"
                skipped += r == "skip"
            except Exception as e:  # noqa: BLE001
                failed += 1
                logger.error("FAILED %s: %r", futs[f][1], e)
            if i % 2000 == 0:
                logger.info("progress %d/%d (copied=%d skip=%d fail=%d)",
                            i, len(jobs), copied, skipped, failed)
    logger.info("DONE copy: copied=%d skipped=%d failed=%d", copied, skipped, failed)
    if failed:
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--copy-tiles", action="store_true", default=True)
    ap.add_argument("--no-copy-tiles", dest="copy_tiles", action="store_false")
    args = ap.parse_args()

    client = storage.Client(project=PROJECT)
    build_cleaned_metadata(client)
    if args.copy_tiles:
        copy_subtrees(client, args.workers)
    logger.info("staged → gs://%s/%s", DST_BUCKET, DST_PREFIX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
