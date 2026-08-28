"""Build the 2025 basemap quad index CSV (inference.md §3; inference/quad_index.py).

The expensive GCS listing runs once; everything downstream (tile grid,
inference) reads the CSV.

Usage:
    python scripts/build_quad_index.py \
        --bucket pdg-planet-data --prefix global_quarterly/2025/q3/ \
        --output /mnt/outputs/inference/quad_index_2025q3.csv \
        --expect-quads 309100

Needs GOOGLE_CLOUD_PROJECT set when running on bare ADC (the storage client
requires a quota project, e.g. abruptthawmapping).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.quad_index import build_quad_index  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="rts-arctic-usw1")
    p.add_argument("--prefix", default="global_quarterly/2025/q3/")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expect-quads", type=int, default=None,
                   help="Grid count from the acquisition step 2 geojson. If given, the "
                        "index must land within --tolerance of it or the build fails: a "
                        "short index is the signature of a filename regime the matcher "
                        "does not cover (see inference/quad_index._QUAD_NAME_RE).")
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="Allowed fractional deviation from --expect-quads (default 1%%).")
    args = p.parse_args()
    setup_logging()

    index = build_quad_index(args.bucket, args.prefix)

    if args.expect_quads is not None:
        drift = abs(len(index) - args.expect_quads) / args.expect_quads
        if drift > args.tolerance:
            logger.error(
                "Quad index has %d quads but %d were ordered (%.1f%% off, tolerance "
                "%.1f%%). A short index usually means the quad filenames in this "
                "delivery do not match inference/quad_index._QUAD_NAME_RE; check one "
                "object under gs://%s/%s before trusting this index.",
                len(index), args.expect_quads, 100 * drift, 100 * args.tolerance,
                args.bucket, args.prefix)
            return 1
        logger.info("Reconciled: %d quads vs %d ordered (%.2f%% off)",
                    len(index), args.expect_quads, 100 * drift)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output, index=False)
    logger.info("Wrote %d quads to %s", len(index), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
