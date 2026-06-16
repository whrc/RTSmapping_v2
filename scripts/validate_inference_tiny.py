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
                      out_dir: Path, scale: float = 1.0) -> None:
    """Run the real entry point (subprocess) so the CLI path is what's tested."""
    cmd = [sys.executable, str(Path(__file__).parent / "inference.py"),
           "--config", "configs/deployment.yaml",
           "--tile-list", str(tile_list), "--quad-index", str(quad_index),
           "--package", package, "--output", str(out_dir), "--num-workers", "4",
           "--scale", str(scale)]
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


def _seam_positions(tiles: pd.DataFrame, mbounds: tuple) -> tuple[list[int], list[int]]:
    """Pixel columns/rows of production-tile boundaries (stitch lines) on the
    merge canvas — every interior tile edge in x and y."""
    minx, miny, maxx, maxy = mbounds
    edges_x = set(tiles["minx"]).union(tiles["maxx"])
    edges_y = set(tiles["miny"]).union(tiles["maxy"])
    cols = sorted({int(round((e - minx) / RESOLUTION_M)) - 1
                   for e in edges_x if minx < e < maxx})
    rows = sorted({int(round((maxy - e) / RESOLUTION_M)) - 1
                   for e in edges_y if miny < e < maxy})
    return cols, rows


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
    seam_cols, _ = _seam_positions(tiles, mbounds)
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
                      out_root: Path, threshold: float,
                      corner_xy: tuple[float, float] | None = None) -> None:
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
    from matplotlib.patches import Rectangle

    minx, miny, maxx, maxy = mbounds
    h, w = merged.shape
    rgb = _read_rgb_canvas(mbounds, quad_index, merged.shape)
    seam_cols, seam_rows = _seam_positions(tiles, mbounds)
    pm = np.ma.masked_where(merged == NODATA_PROB, merged)

    fig, axes = plt.subplots(2, 2, figsize=(16, 17))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    title = (f"merge canvas {w}×{h} px @ {RESOLUTION_M:.4f} m "
             f"({w * RESOLUTION_M / 1000:.1f} × {h * RESOLUTION_M / 1000:.1f} km), "
             f"{len(tiles)} tiles, stride 344 px")
    fig.suptitle(title, fontsize=13)

    # A: RGB + tile outlines + quad boundaries.
    ax_a.imshow(np.moveaxis(rgb, 0, -1))
    for _, t in tiles.iterrows():
        c0 = (t.minx - minx) / RESOLUTION_M
        r0 = (maxy - t.maxy) / RESOLUTION_M
        ax_a.add_patch(Rectangle((c0, r0), TILE_SIZE_PX, TILE_SIZE_PX,
                                 fill=False, edgecolor="cyan", linewidth=0.4))
    if corner_xy is not None:
        cx, cy = corner_xy
        ax_a.axvline((cx - minx) / RESOLUTION_M, color="yellow", ls="--", lw=1.2)
        ax_a.axhline((maxy - cy) / RESOLUTION_M, color="yellow", ls="--", lw=1.2)
    ax_a.set_title("2025 RGB — tile outlines (cyan, 512 px) + quad boundaries (yellow)")

    # B: probability + stitch lines.
    im = ax_b.imshow(pm, vmin=0, vmax=max(0.2, float(pm.max())), cmap="inferno")
    for c in seam_cols:
        ax_b.axvline(c, color="lime", lw=0.4, alpha=0.7)
    for r in seam_rows:
        ax_b.axhline(r, color="lime", lw=0.4, alpha=0.7)
    ax_b.set_title("merged probability + stitch lines (green = tile edges)")
    fig.colorbar(im, ax=ax_b, shrink=0.7)

    # C: RGB + threshold contours.
    ax_c.imshow(np.moveaxis(rgb, 0, -1))
    ax_c.contour(np.where(merged == NODATA_PROB, 0.0, merged),
                 levels=[threshold], colors="red", linewidths=0.8)
    ax_c.set_title(f"RGB + contours @ threshold {threshold}")

    # D: 400px seam zoom centered on the seam intersection nearest the hottest pixel.
    hot_r, hot_c = np.unravel_index(np.argmax(np.where(pm.mask, -1, pm.data)), merged.shape)
    zc = min(seam_cols, key=lambda c: abs(c - hot_c)) if seam_cols else w // 2
    zr = min(seam_rows, key=lambda r: abs(r - hot_r)) if seam_rows else h // 2
    r0, c0 = max(zr - 200, 0), max(zc - 200, 0)
    crop = pm[r0:r0 + 400, c0:c0 + 400]
    imd = ax_d.imshow(crop, vmin=0, vmax=max(0.2, float(pm.max())), cmap="inferno")
    for c in seam_cols:
        if c0 <= c < c0 + 400:
            ax_d.axvline(c - c0, color="lime", lw=0.8, alpha=0.8)
    for r in seam_rows:
        if r0 <= r < r0 + 400:
            ax_d.axhline(r - r0, color="lime", lw=0.8, alpha=0.8)
    ax_d.set_title(f"seam zoom 400×400 px @ ({zc},{zr}) near hottest blob — "
                   "continuity across stitch lines")
    fig.colorbar(imd, ax=ax_d, shrink=0.7)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = out_root / "validation_overlay.png"
    fig.savefig(png, dpi=150); plt.close(fig)
    record("9 geo-alignment", bool(res_ok and bounds_ok and crs_ok),
           f"res==4.7773m:{res_ok} bounds==AOI:{bounds_ok} crs==3857:{crs_ok}; "
           f"canvas {w}x{h}px; overlay: {png}")


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

def run_scale05_experiment(args, quad_index: pd.DataFrame, stride_px: int,
                           sigma_px: float) -> None:
    """Scale-0.5 (2x GSD, 4x FOV) run over the same AOI + comparison figure.

    Qualitative evidence for/against multi-scale training (inference.md §6.2;
    the §6.4 quantitative gate runs on the val set, not here).
    """
    from scipy import ndimage
    from inference.writer import write_probability_tile

    out_root = args.out_root
    cx = WORLD_MIN + args.corner_x * QUAD_SIZE_M
    cy = WORLD_MIN + args.corner_y * QUAD_SIZE_M
    half = args.aoi_half_km * 1000
    aoi = (cx - half, cy - half, cx + half, cy + half)
    res05 = RESOLUTION_M / 0.5  # 9.5546 m

    tiles05 = generate_tile_grid(quad_index, stride_px, aoi=aoi, scale=0.5)
    tl = out_root / "scale05_tiles.csv"; tiles05.to_csv(tl, index=False)
    logger.info("Scale-0.5 grid: %d tiles (each %.1f km FOV)",
                len(tiles05), TILE_SIZE_PX * res05 / 1000)
    out = out_root / "scale05_out"
    run_inference_cli(tl, args.quad_index, args.package, out, scale=0.5)
    merged05, mb05 = merge_tiles(tiles05, str(out), sigma_px=sigma_px,
                                 resolution_m=res05)
    write_probability_tile(str(out_root / "merged_prob_scale05.tif"),
                           merged05, mb05)

    with rasterio.open(out_root / "merged_prob.tif") as src:
        m10 = src.read(1)
        b10 = src.bounds
    mb10 = (b10.left, b10.bottom, b10.right, b10.top)

    # Common window, pixel-aligned to the scale-1.0 grid; upsample 0.5 by 2x.
    ix0, iy0 = max(mb10[0], mb05[0]), max(mb10[1], mb05[1])
    ix1, iy1 = min(mb10[2], mb05[2]), min(mb10[3], mb05[3])
    def crop(arr, b, res):
        c0 = int(round((ix0 - b[0]) / res)); c1 = int(round((ix1 - b[0]) / res))
        r0 = int(round((b[3] - iy1) / res)); r1 = int(round((b[3] - iy0) / res))
        return arr[r0:r1, c0:c1]
    c10 = crop(m10, mb10, RESOLUTION_M)
    c05 = crop(merged05, mb05, res05)
    up05 = np.repeat(np.repeat(c05, 2, axis=0), 2, axis=1)[:c10.shape[0], :c10.shape[1]]
    if up05.shape != c10.shape:  # pad if rounding left a 1px shortfall
        pad = np.full(c10.shape, NODATA_PROB, dtype=up05.dtype)
        pad[:up05.shape[0], :up05.shape[1]] = up05
        up05 = pad

    # Stats: distributions, blobs, agreement.
    v10, v05 = c10[c10 != NODATA_PROB], up05[up05 != NODATA_PROB]
    both = (c10 != NODATA_PROB) & (up05 != NODATA_PROB)
    hot10, hot05 = (c10 >= 0.1) & both, (up05 >= 0.1) & both
    inter, union = (hot10 & hot05).sum(), (hot10 | hot05).sum()
    lab10, n10 = ndimage.label(hot10)
    lab05, n05 = ndimage.label(hot05)
    only05 = hot05 & ~ndimage.binary_dilation(hot10, iterations=10)
    _, n_only05 = ndimage.label(only05)
    fused = np.where(both, (c10 + up05) / 2.0,
                     np.where(c10 != NODATA_PROB, c10,
                              np.where(up05 != NODATA_PROB, up05, NODATA_PROB)))
    record("S scale-0.5 vs 1.0", None,
           f"P>=0.1 blobs: scale1.0={n10}, scale0.5={n05}, only-at-0.5 (>=100m "
           f"from any 1.0 blob)={n_only05}; IoU(hot)={inter / union if union else 0:.3f}; "
           f"max prob 1.0={v10.max():.3f} vs 0.5={v05.max():.3f}; "
           f"mean 1.0={v10.mean():.4f} vs 0.5={v05.mean():.4f}")

    # Figure: 2x3 — RGB | P1.0 | P0.5 ; fused | zoom@hottest-1.0 | zoom@max(P0.5-P1.0).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = _read_rgb_canvas((ix0, iy0, ix1, iy1), quad_index, c10.shape)
    vmax = max(0.2, float(max(v10.max(), v05.max())))
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.suptitle(f"scale 1.0 (4.78 m/px, 2.4 km FOV) vs scale 0.5 (9.55 m/px, 4.9 km FOV) — "
                 f"common window {c10.shape[1]}×{c10.shape[0]} px", fontsize=13)
    def show(ax, arr, title, prob=True):
        if prob:
            im = ax.imshow(np.ma.masked_where(arr == NODATA_PROB, arr),
                           vmin=0, vmax=vmax, cmap="inferno")
            fig.colorbar(im, ax=ax, shrink=0.6)
        else:
            ax.imshow(arr)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    show(axes[0][0], np.moveaxis(rgb, 0, -1), "2025 RGB", prob=False)
    show(axes[0][1], c10, "scale 1.0 probability")
    show(axes[0][2], up05, "scale 0.5 probability (upsampled 2x)")
    show(axes[1][0], fused, "fused mean (§7.3 over valid scales)")
    hr, hc = np.unravel_index(np.argmax(np.where(both, c10, -1)), c10.shape)
    diff = np.where(both, up05 - c10, -1)
    dr, dc = np.unravel_index(np.argmax(diff), diff.shape)
    for ax, (r, c), name in ((axes[1][1], (hr, hc), "hottest scale-1.0 blob"),
                             (axes[1][2], (dr, dc), "max(P0.5 − P1.0)")):
        r0, c0 = max(r - 200, 0), max(c - 200, 0)
        a = c10[r0:r0 + 400, c0:c0 + 400]
        b = up05[r0:r0 + 400, c0:c0 + 400]
        sep = np.full((a.shape[0], 6), vmax, dtype=a.dtype)  # bright divider
        show(ax, np.concatenate([a, sep, b], axis=1), f"zoom {name}: 1.0 | 0.5")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = out_root / "scale_comparison.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    logger.info("Wrote %s", png)


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
    p.add_argument("--overlay-only", action="store_true",
                   help="regenerate validation_overlay.png from the existing "
                        "merged_prob.tif + corner_tiles.csv (no GPU run)")
    p.add_argument("--scale05", action="store_true",
                   help="scale-0.5 experiment (2x GSD, 4x FOV) over the same "
                        "AOI; compares against the existing scale-1.0 "
                        "merged_prob.tif -> scale_comparison.png")
    args = p.parse_args()
    setup_logging()
    out_root = args.out_root; out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/deployment.yaml")
    stride_px = cfg["inference"]["stride_px"]
    sigma_px = cfg["inference"]["fusion_sigma_px"]
    quad_index = load_quad_index(args.quad_index)

    if args.overlay_only:
        dep_cfg = load_config(f"{str(args.package).rstrip('/')}/deployment_config.yaml")
        cx = WORLD_MIN + args.corner_x * QUAD_SIZE_M
        cy = WORLD_MIN + args.corner_y * QUAD_SIZE_M
        tiles = pd.read_csv(out_root / "corner_tiles.csv")
        merged_path = out_root / "merged_prob.tif"
        with rasterio.open(merged_path) as src:
            merged = src.read(1)
            b = src.bounds
        check_geo_overlay(merged, (b.left, b.bottom, b.right, b.top),
                          merged_path, tiles, quad_index, out_root,
                          dep_cfg["threshold"], corner_xy=(cx, cy))
        return 0

    if args.scale05:
        run_scale05_experiment(args, quad_index, stride_px, sigma_px)
        for check, status, detail in RESULTS:
            print(f"  {status:4s}  {check}: {detail}")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                      out_root, threshold, corner_xy=(cx, cy))
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
