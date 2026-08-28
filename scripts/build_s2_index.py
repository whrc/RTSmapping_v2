"""Build the S2 composite index CSV (inference.md §5; inference/s2_index.py).

The expensive GCS listing + per-COG bounds reads run once; inference reads the
CSV to window NDVI from the composites on the fly.

Usage:
    python scripts/build_s2_index.py \
        --bucket rts-mapping-v2-usw1 --prefix S2_RGB/2025_south \
        --output /mnt/outputs/inference/s2_index_2025_south.csv

Needs GOOGLE_CLOUD_PROJECT set when running on bare ADC (the storage client
requires a quota project).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.s2_index import build_s2_index  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="rts-arctic-usw1")
    p.add_argument("--prefix", default="S2_RGB/2025_south")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    setup_logging()

    index = build_s2_index(args.bucket, args.prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output, index=False)
    logger.info("Wrote %d S2 composite cells to %s", len(index), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
