"""Build the 2025 basemap quad index CSV (inference.md §3; inference/quad_index.py).

The expensive GCS listing runs once; everything downstream (tile grid,
inference) reads the CSV.

Usage:
    python scripts/build_quad_index.py \
        --bucket pdg-planet-data --prefix global_quarterly/2025/q3/ \
        --output /mnt/outputs/inference/quad_index_2025q3.csv

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
    p.add_argument("--bucket", default="pdg-planet-data")
    p.add_argument("--prefix", default="global_quarterly/2025/q3/")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    setup_logging()

    index = build_quad_index(args.bucket, args.prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output, index=False)
    logger.info("Wrote %d quads to %s", len(index), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
