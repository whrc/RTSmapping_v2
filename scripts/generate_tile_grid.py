"""Generate the inference tile grid CSV (inference.md §4.4).

Tiles are 512x512 px windows on a global EPSG:3857 stride grid (stride from
configs/deployment.yaml inference.stride_px — SSoT). A tile enters the grid if
it intersects at least one indexed quad (and the optional AOI bbox). External
land/permafrost filtering happens on this CSV outside the pipeline (§4.4); the
inference script consumes the (filtered) CSV as-is.

Tile ids are deterministic grid coordinates: t{col}_{row} in stride units from
the EPSG:3857 origin, so the same location always gets the same id.

Usage:
    python scripts/generate_tile_grid.py \
        --quad-index /mnt/outputs/inference/quad_index_2025q3.csv \
        --config configs/deployment.yaml \
        --output tiles_2025q3.csv \
        [--aoi minx,miny,maxx,maxy]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.quad_index import RESOLUTION_M, WORLD_MIN, load_quad_index  # noqa: E402
from inference.tiles import TILE_SIZE_PX  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def generate_tile_grid(
    quad_index: pd.DataFrame,
    stride_px: int,
    aoi: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Enumerate stride-grid tiles intersecting the indexed quads (and AOI)."""
    stride_m = stride_px * RESOLUTION_M
    tile_m = TILE_SIZE_PX * RESOLUTION_M
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()

    for _, quad in quad_index.iterrows():
        qminx, qminy = quad["minx"], quad["miny"]
        qmaxx, qmaxy = quad["maxx"], quad["maxy"]
        if aoi is not None:
            qminx, qminy = max(qminx, aoi[0]), max(qminy, aoi[1])
            qmaxx, qmaxy = min(qmaxx, aoi[2]), min(qmaxy, aoi[3])
            if qminx >= qmaxx or qminy >= qmaxy:
                continue
        # Grid cols/rows whose tile [c*stride, c*stride + tile_m) overlaps the quad.
        c0 = math.floor((qminx - WORLD_MIN - tile_m) / stride_m) + 1
        c1 = math.floor((qmaxx - WORLD_MIN) / stride_m)
        r0 = math.floor((qminy - WORLD_MIN - tile_m) / stride_m) + 1
        r1 = math.floor((qmaxy - WORLD_MIN) / stride_m)
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                if (c, r) in seen:
                    continue
                seen.add((c, r))
                minx = WORLD_MIN + c * stride_m
                miny = WORLD_MIN + r * stride_m
                rows.append({"tile_id": f"t{c}_{r}",
                             "minx": minx, "miny": miny,
                             "maxx": minx + tile_m, "maxy": miny + tile_m})

    grid = pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)
    logger.info("Tile grid: %d tiles (stride %d px, %d quads%s)",
                len(grid), stride_px, len(quad_index),
                f", AOI {aoi}" if aoi else "")
    return grid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quad-index", required=True)
    p.add_argument("--config", default="configs/deployment.yaml")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--aoi", default=None,
                   help="minx,miny,maxx,maxy in EPSG:3857 (optional)")
    args = p.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    stride_px = cfg["inference"]["stride_px"]
    aoi = tuple(float(v) for v in args.aoi.split(",")) if args.aoi else None
    grid = generate_tile_grid(load_quad_index(args.quad_index), stride_px, aoi)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output, index=False)
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
