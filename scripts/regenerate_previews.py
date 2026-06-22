"""Regenerate the 3-pos/3-neg preview panel for completed runs.

Runs trained before the 2026-06-21 preview-tiles fix rendered a degenerate 1-tile
preview (stale `configs/preview_tiles.yaml` UIDs fell out of val after the 2026-06-16
split regeneration). This reruns each run's **best_deployment.pth** (EMA weights) over
the corrected 6 preview tiles and writes `figures/preview_regenerated.png` — without
touching the original (buggy) `preview_epoch_*.png` files (provenance preserved).

Per run it reuses the recorded resolved `config.yaml` to rebuild the exact model
(channels/backbone/fusion) + normalization, so RGB / RGB+NDVI / full-stack / foundation
runs all regenerate correctly.

Usage:
    python scripts/regenerate_previews.py [--runs-dir /outputs/v1.0/runs]
                                          [--preview-cfg configs/preview_tiles.yaml]
                                          [--only name1,name2]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import RTSDataset, parse_extra_spec  # noqa: E402
from data.splits import load_metadata  # noqa: E402
from data.transforms import build_eval_transforms  # noqa: E402
from models.segmentation import build_model  # noqa: E402
from training import visualizations as viz  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("regen_previews")

# Legacy runs (phase0/2/3/5) recorded a now-dead `gs://rts-mapping-v2/...` norm path.
# Their data_root is already local; only the norm json is unreachable. They are all
# RGB-only, so the local superset stats (RGB + all bands) reproduce the RGB mean/std the
# model was trained with (per-dataset z-score; corrected-split RGB stats differ negligibly).
LOCAL_NORM_FALLBACK = "/outputs/v1.0/staging/v1_splits/normalization_stats.json"


def _resolve_norm_path(norm_path: str) -> str:
    """Remap a dead gs:// (or missing) norm path to the local superset stats json."""
    if norm_path.startswith("gs://") or not Path(norm_path).exists():
        return LOCAL_NORM_FALLBACK
    return norm_path


def _regen_one(run_dir: Path, preview_ids: list[str]) -> str:
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "checkpoints" / "best_deployment.pth"
    if not cfg_path.exists() or not ckpt_path.exists():
        return "skip (no config/ckpt)"
    cfg = yaml.safe_load(cfg_path.read_text())
    data = cfg["data"]
    root = data["data_root"].rstrip("/")
    extra_channels = parse_extra_spec(cfg.get("channels", {}).get("extra", []) or [])
    metadata = load_metadata(f"{root}/{data['metadata_csv']}")

    # Only preview tiles that exist for this run's data; clean GT (no boundary dilation).
    ds = RTSDataset(
        tile_ids=preview_ids, metadata=metadata, data_root=root,
        rgb_dir=data["rgb_dir"], extra_dir=data["extra_dir"], labels_dir=data["labels_dir"],
        extra_channels=extra_channels,
        norm_stats_path=_resolve_norm_path(data["normalization_stats_path"]),
        transform=build_eval_transforms(),
        tile_size=int(data["tile_size"]), label_ignore_index=int(data["label_ignore_index"]),
        boundary_handling="none",
    )
    # The checkpoint supplies all weights → skip the pretrained download (faster, no HF rate-limit).
    cfg.setdefault("model", {})["pretrained"] = False
    model = build_model(cfg).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    records = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            x = item["image"].unsqueeze(0)
            prob = torch.sigmoid(model(x).float()).squeeze().cpu().numpy()
            records.append({
                "tile_id": ds.tile_ids[i],
                "image": item["image"].cpu().numpy(),
                "label": item["label"].cpu().numpy().astype(np.int64),
                "prob": prob,
            })
    out = run_dir / "figures" / "preview_regenerated.png"
    out.parent.mkdir(exist_ok=True)
    viz.prediction_preview_grid(records, ds.mean[:3], ds.std[:3], out)
    return f"OK ({len(records)} tiles) -> {out.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="/outputs/v1.0/runs", type=Path)
    ap.add_argument("--preview-cfg", default="configs/preview_tiles.yaml")
    ap.add_argument("--only", default=None, help="comma-separated run names (default: all)")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.preview_cfg).read_text())
    preview_ids = [str(t) for t in (spec.get("positive", []) + spec.get("negative", []))]
    logger.info("Preview tiles (%d): %s", len(preview_ids), preview_ids)

    run_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())
    if args.only:
        want = set(args.only.split(","))
        run_dirs = [d for d in run_dirs if d.name in want]

    ok = fail = 0
    for d in run_dirs:
        try:
            msg = _regen_one(d, preview_ids)
        except Exception as e:  # keep going; report per-run
            msg = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
        if msg.startswith("OK"):
            ok += 1
        elif msg.startswith("FAIL"):
            fail += 1
        logger.info("%-34s %s", d.name, msg)
    logger.info("DONE: %d regenerated, %d failed", ok, fail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
