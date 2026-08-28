"""Verify the migrated deployment packages are the weights that made the 2025 map.

Gate row 3 of `computing/pdg_migration.md` §5.

`gcs_parity.py` proves the copy is faithful to what sat in PDG: same names, sizes and
MD5s on both sides. That is a statement about the *copy*, not about the *model* -- it
would hold just as well if the PDG package had drifted from what production actually ran.

There is a better anchor, and the production run wrote it itself. Every shard manifest
under `inference/2025q3_south/logs/` records `model_checkpoint_sha`: the SHA256 of each
seed's `weights.pth` as loaded on 2026-07-07, at the moment the tiles were predicted.
Checking the migrated packages against *that* ties them to the delivered product through
a hash neither copy could have influenced.

The manifests are also cross-checked against each other: if the shards do not all name the
same three hashes, the ensemble changed mid-run and the delivered map is not the product
of one model. That would be a far larger problem than the migration.

Usage:
    python scripts/verify_frozen_model.py
    python scripts/verify_frozen_model.py --manifests 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.cloud import storage  # noqa: E402

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

BUCKET = "rts-arctic-usw1"
BASE = "inference/2025q3_south"
SEEDS = ("seed42", "seed43", "seed44")


def recorded_anchor(bucket: storage.Bucket, n: int) -> tuple[list[str], dict]:
    """Return the ensemble SHA triple every sampled shard manifest agrees on.

    Args:
        bucket: The migrated inference bucket.
        n: How many manifests to read, spread evenly across the run.

    Returns:
        ``(sha_triple, one_manifest)`` -- the triple all sampled manifests carry, and one
        manifest whole, for the calibration cross-check.

    Raises:
        SystemExit: If the sampled manifests do not agree on a single triple.
    """
    names = [b.name for b in bucket.list_blobs(prefix=f"{BASE}/logs/", fields="items(name),nextPageToken")]
    step = max(1, len(names) // n)
    picked = names[::step][:n]
    logger.info("%d shard manifests; reading %d spread across the run", len(names), len(picked))

    seen: dict[tuple[str, ...], list[str]] = {}
    one = None
    for name in picked:
        m = json.loads(bucket.blob(name).download_as_bytes())
        one = one or m
        seen.setdefault(tuple(m["model_checkpoint_sha"]), []).append(name)

    if len(seen) != 1:
        for triple, where in seen.items():
            logger.error("  %d manifest(s) name %s ... e.g. %s", len(where), triple[0][:16], where[0])
        raise SystemExit("shard manifests disagree on the ensemble — the delivered map is not one model")
    return list(next(iter(seen))), one


def sha256_of(blob: storage.Blob) -> str:
    """Stream one blob and return its SHA256.

    GCS stores MD5, not SHA256, so this has to read the bytes -- which is the point: the
    anchor being matched was itself computed over the bytes at inference time.
    """
    h = hashlib.sha256()
    with blob.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    """Check migrated weights and calibration against what production recorded."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests", type=int, default=25,
                   help="manifests to cross-check (default 25)")
    args = p.parse_args()
    setup_logging()

    bucket = storage.Client().bucket(BUCKET)
    anchor, manifest = recorded_anchor(bucket, args.manifests)
    logger.info("all sampled manifests agree on one ensemble")

    bad = 0
    for seed, want in zip(SEEDS, anchor):
        got = sha256_of(bucket.blob(f"{BASE}/packages/{seed}/weights.pth"))
        ok = got == want
        bad += not ok
        logger.info("%-8s weights.pth recorded=%s migrated=%s %s",
                    seed, want[:16], got[:16], "MATCH" if ok else "MISMATCH")

    # The package must still describe the product that was delivered, not merely weigh the
    # same: threshold and temperature are baked into the map and recorded in the manifest.
    for seed in SEEDS:
        dep = yaml.safe_load(bucket.blob(f"{BASE}/packages/{seed}/deployment_config.yaml").download_as_bytes())
        for key in ("threshold", "temperature"):
            ok = dep.get(key) == manifest[key]
            bad += not ok
            logger.info("%-8s %-12s package=%-10s manifest=%-10s %s", seed, key,
                        dep.get(key), manifest[key], "MATCH" if ok else "MISMATCH")

    if bad:
        logger.error("FROZEN MODEL FAIL — %d check(s) failed; do NOT delete the PDG packages", bad)
    else:
        logger.info("FROZEN MODEL PASS — migrated packages are the model that produced the 2025 map")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
