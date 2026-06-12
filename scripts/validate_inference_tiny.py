"""Tiny-area inference validation: overlap, stitching, and critical ops.

Tier-2 harness (real 2025 quads, GPU): runs the production pipeline over a
small AOI snapped to a zoom-15 quad 4-corner and turns each critical operation
into a PASS/FAIL check. See docs/inference_validation.md for the results.

Checks:
  1  coverage accounting (stride/overlap geometry)
  2  stitching continuity (offset-grid reference + seam gradients)
  3  fusion correctness vs brute-force NumPy
  4  quad-straddle read correctness
  5  NoData propagation (edge AOI)
  6  determinism (two runs, identical tile rasters)
  7  resume equivalence (partial run + resumed run == single run)
  8  TTA sanity (minimal vs none; flip symmetry)
  9  geo-alignment + overlay PNG
  10 detection plausibility (top blobs vs v2.1 positive centroids)

Usage (inside rts-train:v2, GPU 0):
    python scripts/validate_inference_tiny.py \
        --quad-index /outputs/inference/validation/quad_index.csv \
        --package /outputs/inference/dev_package_seed42 \
        --out-root /outputs/inference/validation \
        [--corner-x 338 --corner-y 1622] [--metadata gs://.../metadata.csv]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if _adc.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_adc)

import numpy as np
import pandas as pd
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.predictor import load_deployment_package, predict_probs  # noqa: E402
from inference.quad_index import (  # noqa: E402
    QUAD_SIZE_M, RESOLUTION_M, WORLD_MIN, load_quad_index,
)
from inference.tiles import TILE_SIZE_PX, InferenceTileDataset, read_tile  # noqa: E402
from inference.writer import NODATA_MASK, NODATA_PROB  # noqa: E402
from scripts.generate_tile_grid import generate_tile_grid  # noqa: E402
from scripts.merge_predictions import gaussian_center_weights, merge_tiles  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS: list[tuple[str, str, str]] = []  # (check, PASS/FAIL/INFO, detail)


def record(check: str, ok: bool | None, detail: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((check, status, detail))
    logger.info("[%s] %s — %s", status, check, detail)


def run_inference_cli(tile_list: Path, quad_index: Path, package: str,
                      out_dir: Path) -> None:
    """Run the real entry point (subprocess) so the CLI path is what's tested."""
    cmd = [sys.executable, str(Path(__file__).parent / "inference.py"),
           "--config", "configs/deployment.yaml",
           "--tile-list", str(tile_list), "--quad-index", str(quad_index),
           "--package", package, "--output", str(out_dir), "--num-workers", "4"]
    subprocess.run(cmd, check=True)


def sha_of_dir_tifs(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.glob("*.tif")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_coverage(tiles: pd.DataFrame, bounds: tuple) -> None:
    minx, miny, maxx, maxy = bounds
    w = int(round((maxx - minx) / RESOLUTION_M))
    h = int(round((maxy - miny) / RESOLUTION_M))
    cover = np.zeros((h, w), dtype=np.int16)
    for _, t in tiles.iterrows():
        c0 = int(round((t.minx - minx) / RESOLUTION_M))
        r0 = int(round((maxy - t.maxy) / RESOLUTION_M))
        cover[max(r0, 0):r0 + TILE_SIZE_PX, max(c0, 0):c0 + TILE_SIZE_PX] += 1
    interior = cover[TILE_SIZE_PX:-TILE_SIZE_PX, TILE_SIZE_PX:-TILE_SIZE_PX]
    # At 33% overlap (stride 344 of 512), tile-center pixels are covered once
    # by design; seams 2x; tile corners 4x. Gaps are the failure mode.
    counts = dict(zip(*np.unique(interior, return_counts=True)))
    ok = (cover.min() >= 1 and interior.max() <= 4
          and {1, 2, 4} <= set(counts))
    record("1 coverage", ok,
           f"no gaps={cover.min() >= 1}; interior coverage histogram "
           f"{ {int(k): int(v) for k, v in counts.items()} } (design: 1/2/4 at stride 344)")


def check_stitching(merged: np.ndarray, mbounds: tuple, tiles: pd.DataFrame,
                    quad_index: pd.DataFrame, pkg: dict, stride_px: int,
                    device: torch.device) -> None:
    # Reference: tiles offset by half a stride — their centers sit on the
    # production grid's seams. Compare central 256px vs the merged raster.
    minx, miny, maxx, maxy = mbounds
    off = (stride_px // 2) * RESOLUTION_M
    ref = tiles.copy()
    ref["minx"] += off; ref["maxx"] += off
    ref["miny"] += off; ref["maxy"] += off
    ref = ref[(ref.maxx <= maxx) & (ref.maxy <= maxy)].head(20)
    ds = InferenceTileDataset(ref, quad_index, pkg["mean"], pkg["std"])
    dep = pkg["dep_cfg"]
    deltas = []
    for i in range(len(ds)):
        item = ds[i]
        if item["all_nodata"]:
            continue
        img = torch.from_numpy(item["image"][None]).to(device)
        prob = predict_probs(pkg["model"], img, dep["temperature"],
                             dep.get("tta", "none"), dep.get("precision", "fp32"))
        prob = prob[0].cpu().numpy()
        b = item["bounds"]
        c0 = int(round((b[0] - minx) / RESOLUTION_M))
        r0 = int(round((maxy - b[3]) / RESOLUTION_M))
        sl = (slice(r0 + 128, r0 + 384), slice(c0 + 128, c0 + 384))
        m = merged[sl]
        p = prob[128:384, 128:384]
        valid = (m != NODATA_PROB) & ~item["nodata_mask"][128:384, 128:384]
        if valid.any():
            deltas.append(np.abs(m[valid] - p[valid]))
    d = np.concatenate(deltas)
    mean_d, p99_d = float(d.mean()), float(np.percentile(d, 99))
    record("2a stitch vs offset-grid reference", p99_d <= 0.05,
           f"mean|Δ|={mean_d:.5f} p99|Δ|={p99_d:.5f} over {d.size} px "
           "(merged value at seams vs direct center prediction)")

    # Seam gradients: pixel-difference across production-tile boundary
    # columns/rows vs everywhere else.
    valid = merged != NODATA_PROB
    gx = np.abs(np.diff(merged, axis=1)); gvx = valid[:, 1:] & valid[:, :-1]
    seam_cols = sorted({int(round((t - minx) / RESOLUTION_M)) - 1
                        for t in tiles["minx"].unique() if minx < t < maxx})
    seam_mask = np.zeros_like(gx, dtype=bool)
    for c in seam_cols:
        if 0 <= c < gx.shape[1]:
            seam_mask[:, c] = True
    on = gx[seam_mask & gvx]; off_ = gx[~seam_mask & gvx]
    p99_on, p99_off = np.percentile(on, 99), np.percentile(off_, 99)
    record("2b seam gradients", p99_on <= max(2 * p99_off, 1e-4),
           f"p99 |∇| on seam cols={p99_on:.5f} vs off-seam={p99_off:.5f}")


def check_fusion_bruteforce(merged: np.ndarray, mbounds: tuple,
                            tiles: pd.DataFrame, tiles_dir: Path,
                            sigma_px: float) -> None:
    minx, miny, maxx, maxy = mbounds
    weights = gaussian_center_weights(TILE_SIZE_PX, sigma_px)
    rasters = {}
    for _, t in tiles.iterrows():
        p = tiles_dir / f"{t.tile_id}.tif"
        if p.exists():
            with rasterio.open(p) as src:
                rasters[t.tile_id] = (src.read(1), t)
    rng = np.random.default_rng(42)
    h, w = merged.shape
    n_checked, worst = 0, 0.0
    while n_checked < 20:
        r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
        px = minx + (c + 0.5) * RESOLUTION_M
        py = maxy - (r + 0.5) * RESOLUTION_M
        num = den = 0.0
        for arr, t in rasters.values():
            if not (t.minx <= px < t.maxx and t.miny <= py < t.maxy):
                continue
            tc = int((px - t.minx) / RESOLUTION_M)
            tr = int((t.maxy - py) / RESOLUTION_M)
            v = arr[tr, tc]
            if v != NODATA_PROB:
                num += v * weights[tr, tc]
                den += weights[tr, tc]
        expect = num / den if den > 0 else NODATA_PROB
        got = merged[r, c]
        worst = max(worst, abs(got - expect))
        n_checked += 1
    record("3 fusion brute-force", worst <= 1e-5,
           f"max|merged - bruteforce| = {worst:.2e} over {n_checked} random px")


def check_quad_straddle(tiles: pd.DataFrame, quad_index: pd.DataFrame,
                        corner_xy: tuple[float, float]) -> None:
    cx, cy = corner_xy
    picks = []
    for _, t in tiles.iterrows():
        sx = t.minx < cx < t.maxx
        sy = t.miny < cy < t.maxy
        if sx and sy:
            picks.append(("4-corner", t))
        elif sx or sy:
            picks.append(("edge", t))
    picks = ([p for p in picks if p[0] == "4-corner"][:1]
             + [p for p in picks if p[0] == "edge"][:2])
    ok_all, details = True, []
    for kind, t in picks:
        bbox = (t.minx, t.miny, t.maxx, t.maxy)
        rgb, nodata = read_tile(bbox, quad_index)
        # Reference: independent boundless reads of every intersecting quad,
        # composited in plain numpy.
        ref = np.zeros_like(rgb); refv = np.zeros(rgb.shape[1:], bool)
        hits = quad_index[(quad_index.minx < t.maxx) & (quad_index.maxx > t.minx)
                          & (quad_index.miny < t.maxy) & (quad_index.maxy > t.miny)]
        for _, q in hits.iterrows():
            with rasterio.open(q.gcs_path) as src:
                win = rasterio.windows.from_bounds(*bbox, transform=src.transform)
                data = src.read(window=win, boundless=True, fill_value=0)
            a = data[3] > 0
            inq = np.zeros_like(a)
            c0 = int(round((q.minx - t.minx) / RESOLUTION_M))
            r0 = int(round((t.maxy - q.maxy) / RESOLUTION_M))
            r1 = r0 + int(round((q.maxy - q.miny) / RESOLUTION_M))
            c1 = c0 + int(round((q.maxx - q.minx) / RESOLUTION_M))
            inq[max(r0, 0):r1, max(c0, 0):c1] = True
            sel = a & inq & ~refv
            ref[:, sel] = data[:3, sel]
            refv |= sel
        same = np.array_equal(rgb[:, refv], ref[:, refv]) and np.array_equal(~refv, nodata)
        seam_ok = not nodata[1:-1, 1:-1].all()
        ok_all &= same
        details.append(f"{kind}:{t.tile_id} identical={same} "
                       f"n_quads={len(hits)} seam_has_data={seam_ok}")
    record("4 quad-straddle reads", ok_all, "; ".join(details))


def check_nodata_edge(quad_index: pd.DataFrame, package: str, stride_px: int,
                      sigma_px: float, out_root: Path, quad_index_path: Path,
                      threshold: float) -> None:
    # AOI straddling the TOP of the topmost available quad column — beyond it
    # there is no quad => guaranteed NoData region in the merge canvas.
    col = quad_index[quad_index.x == quad_index.x.iloc[0]]
    top = col.loc[col.y.idxmax()]
    aoi = (top.minx + QUAD_SIZE_M * 0.4, top.maxy - QUAD_SIZE_M * 0.1,
           top.minx + QUAD_SIZE_M * 0.6, top.maxy + QUAD_SIZE_M * 0.1)
    tiles = generate_tile_grid(quad_index, stride_px, aoi=aoi)
    tl = out_root / "edge_tiles.csv"; tiles.to_csv(tl, index=False)
    out = out_root / "edge_tiles_out"
    run_inference_cli(tl, quad_index_path, package, out)
    merged, mb = merge_tiles(tiles, str(out), sigma_px=sigma_px)
    n_nodata = (merged == NODATA_PROB).sum()
    # Top rows of the canvas extend past quad coverage -> must be NoData.
    rows_above = int((mb[3] - top.maxy) / RESOLUTION_M) - 1
    top_nodata = (merged[:rows_above] == NODATA_PROB).all() if rows_above > 0 else True
    in_range = merged[merged != NODATA_PROB]
    vals_ok = bool(((in_range >= 0) & (in_range <= 1)).all())
    mask = np.where(merged == NODATA_PROB, NODATA_MASK,
                    (merged >= threshold).astype(np.uint8))
    mask_ok = bool(((mask == NODATA_MASK) == (merged == NODATA_PROB)).all())
    record("5 NoData propagation", bool(top_nodata and vals_ok and mask_ok
                                        and n_nodata > 0),
           f"nodata_px={n_nodata} beyond-coverage all NoData={top_nodata}, "
           f"valid in [0,1]={vals_ok}, mask 255 ⇔ prob -1: {mask_ok}")


def check_tta(pkg: dict, tiles: pd.DataFrame, quad_index: pd.DataFrame,
              device: torch.device) -> None:
    ds = InferenceTileDataset(tiles.head(4), quad_index, pkg["mean"], pkg["std"])
    imgs = torch.from_numpy(np.stack([ds[i]["image"] for i in range(len(ds))])).to(device)
    dep = pkg["dep_cfg"]
    p_none = predict_probs(pkg["model"], imgs, dep["temperature"], "none",
                           dep.get("precision", "fp32"))
    p_min = predict_probs(pkg["model"], imgs, dep["temperature"], "minimal",
                          dep.get("precision", "fp32"))
    d = (p_none - p_min).abs()
    # Flip symmetry in probability space.
    p_flip = predict_probs(pkg["model"], torch.flip(imgs, dims=(-1,)),
                           dep["temperature"], "none", dep.get("precision", "fp32"))
    sym = (torch.flip(p_flip, dims=(-1,)) - p_none).abs()
    consistent = bool(torch.allclose(2 * p_min - p_none,
                                     torch.flip(p_flip, dims=(-1,)), atol=1e-3))
    record("8 TTA sanity", bool(d.max() > 0 and consistent),
           f"mean|none-minimal|={d.mean():.5f} max={d.max():.5f}; "
           f"minimal == (identity+hflip)/2 algebra holds={consistent}; "
           f"hflip asymmetry mean={sym.mean():.5f} (model is not flip-invariant — expected)")


def _read_rgb_canvas(bounds: tuple, quad_index: pd.DataFrame,
                     shape: tuple[int, int]) -> np.ndarray:
    """Rectangular RGB mosaic of the AOI (validation-only; read_tile is
    square-by-contract for the 512px pipeline)."""
    minx, miny, maxx, maxy = bounds
    h, w = shape
    rgb = np.zeros((3, h, w), dtype=np.uint8)
    hits = quad_index[(quad_index.minx < maxx) & (quad_index.maxx > minx)
                      & (quad_index.miny < maxy) & (quad_index.maxy > miny)]
    for _, q in hits.iterrows():
        with rasterio.open(q.gcs_path) as src:
            win = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            data = src.read(window=win, boundless=True, fill_value=0,
                            out_shape=(src.count, h, w))
        a = data[3] > 0
        rgb[:, a] = data[:3, a]
    return rgb


def check_geo_overlay(merged: np.ndarray, mbounds: tuple, merged_path: Path,
                      tiles: pd.DataFrame, quad_index: pd.DataFrame,
                      out_root: Path, threshold: float) -> None:
    with rasterio.open(merged_path) as src:
        res_ok = (abs(src.res[0] - RESOLUTION_M) < 1e-6
                  and abs(src.res[1] - RESOLUTION_M) < 1e-6)
        b = src.bounds
        bounds_ok = all(abs(g - e) < RESOLUTION_M for g, e in
                        zip((b.left, b.bottom, b.right, b.top), mbounds))
        crs_ok = src.crs.to_string() == "EPSG:3857"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = _read_rgb_canvas(mbounds, quad_index, merged.shape)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.moveaxis(rgb, 0, -1)); axes[0].set_title("2025 RGB")
    pm = np.ma.masked_where(merged == NODATA_PROB, merged)
    im = axes[1].imshow(pm, vmin=0, vmax=max(0.2, float(pm.max())), cmap="inferno")
    axes[1].set_title("merged probability"); fig.colorbar(im, ax=axes[1], shrink=0.7)
    axes[2].imshow(np.moveaxis(rgb, 0, -1))
    axes[2].contour(np.where(merged == NODATA_PROB, 0.0, merged),
                    levels=[threshold], colors="red", linewidths=0.8)
    axes[2].set_title(f"contours @ {threshold}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    png = out_root / "validation_overlay.png"
    fig.savefig(png, dpi=110); plt.close(fig)
    record("9 geo-alignment", bool(res_ok and bounds_ok and crs_ok),
           f"res==4.7773m:{res_ok} bounds==AOI:{bounds_ok} crs==3857:{crs_ok}; "
           f"overlay: {png}")


def check_blobs(merged: np.ndarray, mbounds: tuple, metadata: str | None) -> None:
    from scipy import ndimage
    from pyproj import Transformer
    minx, _, _, maxy = mbounds
    hot = (merged >= 0.1)
    lab, n = ndimage.label(hot)
    sizes = ndimage.sum(hot, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1][:5]
    inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lines = []
    cents = None
    if metadata:
        df = pd.read_csv(metadata)
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        mx, my = t.transform(df.centroid_lon.values, df.centroid_lat.values)
        cents = np.stack([mx, my], 1)
    for i in order:
        r, c = ndimage.center_of_mass(hot, lab, i + 1)
        px, py = minx + c * RESOLUTION_M, maxy - r * RESOLUTION_M
        lon, lat = inv.transform(px, py)
        peak = float(ndimage.maximum(merged, lab, i + 1))
        s = f"blob sz={int(sizes[i])}px peak={peak:.3f} @ {lat:.4f},{lon:.4f}"
        if cents is not None:
            dist = float(np.sqrt(((cents - [px, py]) ** 2).sum(1)).min())
            s += f" nearest-v2.1-positive={dist/1000:.2f}km"
        lines.append(s)
    record("10 detection plausibility", None,
           f"{n} blobs ≥0.1 | top: " + " | ".join(lines) if n else "no blobs ≥ 0.1")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quad-index", required=True, type=Path)
    p.add_argument("--package", required=True)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--corner-x", type=int, default=338)
    p.add_argument("--corner-y", type=int, default=1622)
    p.add_argument("--aoi-half-km", type=float, default=5.0)
    p.add_argument("--metadata", default=None,
                   help="v2.1 metadata.csv (local copy) for check 10")
    args = p.parse_args()
    setup_logging()
    out_root = args.out_root; out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config("configs/deployment.yaml")
    stride_px = cfg["inference"]["stride_px"]
    sigma_px = cfg["inference"]["fusion_sigma_px"]
    quad_index = load_quad_index(args.quad_index)
    pkg = load_deployment_package(args.package, device)
    threshold = pkg["dep_cfg"]["threshold"]

    # Corner AOI around the quad 4-corner point.
    cx = WORLD_MIN + args.corner_x * QUAD_SIZE_M
    cy = WORLD_MIN + args.corner_y * QUAD_SIZE_M
    half = args.aoi_half_km * 1000
    aoi = (cx - half, cy - half, cx + half, cy + half)
    tiles = generate_tile_grid(quad_index, stride_px, aoi=aoi)
    tl = out_root / "corner_tiles.csv"; tiles.to_csv(tl, index=False)
    logger.info("Corner AOI at (%.0f, %.0f): %d tiles", cx, cy, len(tiles))

    # Main run (CLI) + merge.
    out1 = out_root / "run1"
    run_inference_cli(tl, args.quad_index, args.package, out1)
    merged, mbounds = merge_tiles(tiles, str(out1), sigma_px=sigma_px)
    from inference.writer import write_probability_tile
    merged_path = out_root / "merged_prob.tif"
    write_probability_tile(str(merged_path), merged, mbounds)

    check_coverage(tiles, mbounds)
    check_stitching(merged, mbounds, tiles, quad_index, pkg, stride_px, device)
    check_fusion_bruteforce(merged, mbounds, tiles, out1, sigma_px)
    check_quad_straddle(tiles, quad_index, (cx, cy))
    check_nodata_edge(quad_index, args.package, stride_px, sigma_px,
                      out_root, args.quad_index, threshold)

    # 6 determinism: second full run, compare tile rasters byte-wise.
    out2 = out_root / "run2"
    run_inference_cli(tl, args.quad_index, args.package, out2)
    s1, s2 = sha_of_dir_tifs(out1), sha_of_dir_tifs(out2)
    record("6 determinism", s1 == s2, f"run1 sha={s1[:12]} run2 sha={s2[:12]}")

    # 7 resume equivalence: first 20 tiles, then resume the rest; compare.
    out3 = out_root / "run3"
    head = out_root / "corner_tiles_head.csv"; tiles.head(20).to_csv(head, index=False)
    run_inference_cli(head, args.quad_index, args.package, out3)
    n_after_partial = len(list(out3.glob("*.tif")))
    run_inference_cli(tl, args.quad_index, args.package, out3)
    s3 = sha_of_dir_tifs(out3)
    # Byte-identity is not guaranteed under bf16: different batch shapes pick
    # different kernels (measured ~6e-3 max prob jitter). Resume must be
    # *value*-equivalent within that tolerance.
    worst = 0.0
    for p in sorted(out1.glob("*.tif")):
        with rasterio.open(p) as a, rasterio.open(out3 / p.name) as b:
            worst = max(worst, float(np.abs(a.read(1) - b.read(1)).max()))
    record("7 resume equivalence", n_after_partial <= 20 and worst <= 0.01,
           f"partial run wrote {n_after_partial} tiles; byte-identical={s3 == s1}; "
           f"max|Δprob| vs run1={worst:.5f} (bf16 batch-shape jitter; ≤0.01 tolerated)")

    check_tta(pkg, tiles, quad_index, device)
    check_geo_overlay(merged, mbounds, merged_path, tiles, quad_index,
                      out_root, threshold)
    check_blobs(merged, mbounds, args.metadata)

    print("\n===== VALIDATION SUMMARY =====")
    n_fail = 0
    for check, status, detail in RESULTS:
        print(f"  {status:4s}  {check}: {detail}")
        n_fail += status == "FAIL"
    print(f"===== {len(RESULTS)} checks, {n_fail} FAIL =====")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
