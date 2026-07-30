"""Frozen-feature linear probe — how much RTS signal is in an encoder before any training.

Built for the SAM2 investigation. `fm_sam2_rgb` scored 0.5558, but its deficit is already
present in the linear-probe phase (val PR-AUC 0.486 at ep20 vs DINOv3-B's 0.682 on the
identical decoder, recipe and normalization), so the question is whether the *features* are
weak — and if so, whether that is caused by encoder size, the normalization mismatch (SAM2
expects ImageNet stats; the run used per-dataset z-score), or the resolution mismatch (SAM2
is pretrained at 896, we train at 512).

A full LP-FT run is ~12 GPU-h per cell. This answers the same question per cell in minutes
by freezing the encoder, pooling its native pyramid, and fitting a closed-form logistic
probe on pixel features — so the expensive runs can be aimed rather than swept.

Reuses the training data path verbatim (RTSDataset + build_eval_transforms + apply_norm via
the config's stats file), so a probe number is comparable to what training would see.

Usage:
  python scripts/probe_frozen_features.py --config configs/fm_sam2_rgb.yaml \
      --backbones sam2_hiera_small,sam2_hiera_base_plus,vit_base_patch16_dinov3 \
      --stats normalization_stats.json,normalization_stats_imagenet.json \
      --input-sizes 512,896 --n-tiles 300 --out /outputs/v1.0/qc/frozen_probe.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if _adc.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_adc)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import RTSDataset, parse_extra_spec  # noqa: E402
from data.splits import get_tile_ids, load_metadata, load_splits_yaml  # noqa: E402
from data.transforms import build_eval_transforms  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)

# Label stride for the probe. The FPN decoder fuses at /4, so scoring at /4 matches what
# the segmentation head actually consumes.
LABEL_STRIDE = 4
IGNORE = 255


def build_encoder(backbone: str, device: str) -> tuple[torch.nn.Module, list[int]]:
    """Frozen timm encoder emitting a 4-level pyramid, mirroring models/foundation.py.

    Hierarchical backbones (SAM2/Hiera) expose a native pyramid via features_only; plain
    ViTs are tapped at 4 evenly-spaced blocks via forward_intermediates. Same split as
    FoundationSegmenter.__init__, so probe features are the ones the real model would get.
    """
    if backbone.startswith(("sam2", "hiera")):
        enc = timm.create_model(backbone, pretrained=True, features_only=True,
                                out_indices=(0, 1, 2, 3))
        chans = enc.feature_info.channels()
    else:
        enc = timm.create_model(backbone, pretrained=True, num_classes=0)
        chans = [enc.embed_dim] * 4
    return enc.eval().to(device), chans


def forward_pyramid(enc: torch.nn.Module, backbone: str, x: torch.Tensor) -> list[torch.Tensor]:
    """Encoder → list of 4 feature maps (NCHW)."""
    if backbone.startswith(("sam2", "hiera")):
        return list(enc(x))
    depth = len(enc.blocks)
    taps = [max(0, round((j + 1) * depth / 4) - 1) for j in range(4)]
    return list(enc.forward_intermediates(x, indices=taps, norm=True, output_fmt="NCHW",
                                          intermediates_only=True))


def collect(cfg: dict, stats_path: str, backbone: str, input_size: int, n_tiles: int,
            px_per_tile: int, device: str, seed: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Extract per-pixel frozen features + labels over `n_tiles` val_realistic tiles.

    Returns (features (N, sum(chans)), labels (N,), tile index (N,), per-stage channel
    counts). Pixels are sampled at LABEL_STRIDE with ignore(255) dropped; each stage is
    bilinearly resampled to the /4 grid before concatenation, matching the FPN decoder's
    fusion resolution. The tile index is what lets the probe split train/test by TILE —
    splitting by pixel leaks, because neighbouring pixels of one tile are near-identical
    (a pixel-split smoke test read AP 0.99 for an encoder whose real LP score is ~0.49).
    """
    data_root = cfg["data"]["data_root"]
    metadata = load_metadata(f"{data_root.rstrip('/')}/{cfg['data']['metadata_csv']}")
    splits = load_splits_yaml(f"{data_root.rstrip('/')}/{cfg['data']['splits_yaml']}")
    val_ids = get_tile_ids("val_realistic", metadata, splits)

    # Deterministic tile subset, and positives first so the probe sees RTS pixels at all
    # (val_realistic is ~1:20 imbalanced; a uniform draw would be nearly all background).
    rng = np.random.default_rng(seed)
    pos = [t for t in val_ids if metadata.set_index("Tile_ID").loc[t, "TrainClass"] == "positive"]
    neg = [t for t in val_ids if t not in set(pos)]
    n_pos = min(len(pos), n_tiles // 2)
    n_neg = min(len(neg), n_tiles - n_pos)
    chosen = list(rng.choice(pos, n_pos, replace=False)) + list(rng.choice(neg, n_neg, replace=False))

    ds = RTSDataset(
        tile_ids=chosen, metadata=metadata, data_root=data_root,
        rgb_dir=cfg["data"]["rgb_dir"], extra_dir=cfg["data"]["extra_dir"],
        labels_dir=cfg["data"]["labels_dir"],
        extra_channels=parse_extra_spec(cfg["channels"].get("extra", [])),
        norm_stats_path=stats_path, transform=build_eval_transforms(),
        tile_size=int(cfg["data"]["tile_size"]),
        label_ignore_index=IGNORE,
        boundary_handling=cfg["loss"]["boundary_handling"],
        boundary_ignore_width=int(cfg["loss"].get("boundary_ignore_width", 3)),
        seed=seed,
    )

    enc, chans = build_encoder(backbone, device)
    feats: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for i in range(len(ds)):
        sample = ds[i]
        img = torch.as_tensor(sample["image"])[None].to(device)
        lab = np.asarray(sample["label"])
        if input_size != img.shape[-1]:
            img = F.interpolate(img, size=(input_size, input_size), mode="bilinear",
                                align_corners=False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            maps = forward_pyramid(enc, backbone, img)
        gh = gw = lab.shape[0] // LABEL_STRIDE
        maps = [F.interpolate(m.float(), size=(gh, gw), mode="bilinear", align_corners=False)
                for m in maps]
        feat = torch.cat(maps, dim=1)[0].permute(1, 2, 0).reshape(-1, sum(chans)).cpu().numpy()

        # Label at /4: max-pool so a cell is positive if it contains any RTS pixel, and
        # ignore(255) dominates so boundary/ignore cells never enter the probe.
        blocks = lab.reshape(gh, LABEL_STRIDE, gw, LABEL_STRIDE)
        cell_ign = (blocks == IGNORE).any(axis=(1, 3)).reshape(-1)
        cell_pos = (blocks == 1).any(axis=(1, 3)).reshape(-1)
        keep = ~cell_ign
        idx = np.flatnonzero(keep)
        if idx.size > px_per_tile:
            idx = rng.choice(idx, px_per_tile, replace=False)
        feats.append(feat[idx])
        labs.append(cell_pos[idx].astype(np.int8))
        groups.append(np.full(idx.size, i, dtype=np.int32))
        if (i + 1) % 50 == 0:
            logger.info("  %s @%d: %d/%d tiles", backbone, input_size, i + 1, len(ds))

    return np.concatenate(feats), np.concatenate(labs), np.concatenate(groups), chans


def probe(X: np.ndarray, y: np.ndarray, g: np.ndarray, seed: int) -> float:
    """Closed-form logistic probe → pixel average-precision on held-out TILES.

    Split is by tile (`g`), never by pixel: pixels within a tile are spatially
    autocorrelated, so a pixel-wise split scores memorisation, not transfer.
    """
    rng = np.random.default_rng(seed)
    tiles = np.unique(g)
    held = set(rng.choice(tiles, max(1, len(tiles) // 2), replace=False).tolist())
    te_mask = np.isin(g, list(held))
    tr, te = np.flatnonzero(~te_mask), np.flatnonzero(te_mask)
    if y[tr].sum() == 0 or y[te].sum() == 0:
        return float("nan")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit((X[tr] - mu) / sd, y[tr])
    return float(average_precision_score(y[te], clf.decision_function((X[te] - mu) / sd)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--backbones", required=True, help="comma-separated timm names")
    p.add_argument("--stats", required=True,
                   help="comma-separated normalization stats FILENAMES (resolved next to "
                        "the config's normalization_stats_path)")
    p.add_argument("--input-sizes", default="512")
    p.add_argument("--n-tiles", type=int, default=300)
    p.add_argument("--px-per-tile", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    setup_logging()
    seed_everything(args.seed, deterministic=False)
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats_dir = Path(cfg["data"]["normalization_stats_path"]).parent

    rows = []
    cells = itertools.product(args.backbones.split(","), args.stats.split(","),
                              [int(s) for s in args.input_sizes.split(",")])
    for backbone, stats_name, size in cells:
        stats_path = str(stats_dir / stats_name.strip())
        logger.info("CELL backbone=%s stats=%s size=%d", backbone, stats_name, size)
        try:
            X, y, g, chans = collect(cfg, stats_path, backbone, size, args.n_tiles,
                                     args.px_per_tile, device, args.seed)
        except Exception as exc:  # a bad cell must not abort the sweep
            logger.error("  FAILED: %s", exc)
            rows.append({"backbone": backbone, "stats": stats_name, "input_size": size,
                         "error": str(exc)})
            continue

        row = {"backbone": backbone, "stats": stats_name, "input_size": size,
               "n_px": int(len(y)), "pos_frac": float(y.mean()),
               "ap_fused": probe(X, y, g, args.seed)}
        # Per-stage, to localise a broken level rather than only scoring the fusion.
        off = 0
        for si, c in enumerate(chans):
            row[f"ap_stage{si}"] = probe(X[:, off:off + c], y, g, args.seed)
            off += c
        rows.append(row)
        logger.info("  fused AP=%.4f  stages=%s", row["ap_fused"],
                    [round(row[f"ap_stage{i}"], 4) for i in range(len(chans))])

    rows.sort(key=lambda r: r.get("ap_fused", -1), reverse=True)
    print(f"\n{'backbone':34s} {'stats':38s} {'size':>5s} {'fused':>7s}  per-stage")
    for r in rows:
        if "error" in r:
            print(f"{r['backbone']:34s} {r['stats']:38s} {r['input_size']:5d}   ERROR {r['error'][:40]}")
            continue
        stages = " ".join(f"{r[k]:.3f}" for k in sorted(r) if k.startswith("ap_stage"))
        print(f"{r['backbone']:34s} {r['stats']:38s} {r['input_size']:5d} {r['ap_fused']:7.4f}  {stages}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
