"""Slice a normalization_stats.json into a per-arm variant by EXTRA channel name.

Why slice instead of recompute: EXTRA stats come from a NaN-safe random subsample
(compute_normalization_stats.py), so recomputing for a different channel subset
perturbs the constants of the channels the arms have in common. An ablation whose
arms disagree on NDVI's mean is measuring two things at once. Compute the superset
once, then slice — every arm then shares bit-identical constants for every channel
it shares, which is the discipline family D used for the SE_PCA decomposition.

The `rgb` block is copied verbatim. The `extra` block is reordered/subset to the
requested names; each of `mean`/`std`/`mode`/`clip`/`scale` is a list parallel to
`channel_names` (data/normalization.py:build_stats_dict), so all of them are
subset by the same index map.

Usage:
  python scripts/slice_normalization_stats.py \
      --in  /outputs/v1.0/staging/v1_splits/normalization_stats_arcticdem_super.json \
      --out /outputs/v1.0/staging/v1_splits/normalization_stats_arcticdem_terrain.json \
      --channels dem_relev dem_slope dem_tpi dem_curv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Parallel-to-channel_names keys in the `extra` block. Optional ones are absent in
# older stats files (mode/clip/scale postdate the per-channel norm work).
_PARALLEL_KEYS = ("mean", "std", "mode", "clip", "scale")


def slice_stats(stats: dict, channels: list[str]) -> dict:
    """Return a copy of `stats` whose EXTRA block is exactly `channels`, in order."""
    if "extra" not in stats:
        raise ValueError("source stats has no 'extra' block to slice")
    src = stats["extra"]
    names = list(src["channel_names"])
    missing = [c for c in channels if c not in names]
    if missing:
        raise ValueError(f"channels not in source stats: {missing} (have {names})")
    idx = [names.index(c) for c in channels]

    out_extra: dict = {"channel_names": list(channels)}
    for key in _PARALLEL_KEYS:
        if key not in src:
            continue
        values = src[key]
        if len(values) != len(names):
            raise ValueError(
                f"stats['extra']['{key}'] has {len(values)} entries but there are "
                f"{len(names)} channel_names — the lists must stay parallel")
        out_extra[key] = [values[i] for i in idx]

    out = {k: v for k, v in stats.items() if k != "extra"}
    out["extra"] = out_extra
    out["sliced_from_channels"] = names
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--channels", nargs="+", required=True,
                    help="EXTRA channel names to keep, in config order")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stats = json.loads(args.src.read_text())
    out = slice_stats(stats, args.channels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s with extra channels %s", args.out, args.channels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
