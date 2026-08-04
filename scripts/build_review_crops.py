"""Render the review campaign's crop archive: 2 JPEGs per candidate polygon.

Every polygon in `south_rts_candidates.gpkg` gets a tight crop (~3× the
feature) and a wide crop (~1.5 km context), red outline burned in — the same
geometry the offline pack builder uses (`review/crops.py`), so both review
surfaces show the identical view.

Resumable: a polygon whose two crops already exist on disk is skipped, so an
interrupted run continues where it stopped. Blank crops (chips missing → the
render is all fill) are reported to `blank_crops.csv` rather than silently
shipped to a reviewer.

Output goes to a **local** directory; upload it to `internal/review_crops/`
with `gsutil -m rsync`. The archive is PlanetScope-derived and must never land
under `products/` (see `post-inference/south_products.md`, 2026-07-18 audit).

Usage:
    python scripts/build_review_crops.py \
        --candidates /outputs/.../south_rts_candidates.gpkg \
        --chips-vrt /outputs/.../rgb_chips.vrt \
        --out-dir /outputs/.../review_crops [--workers 90]
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely import wkb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from review.crops import crop_bounds, has_imagery, render_crop  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

CHUNK = 500  # polygons per worker task

_INDEX = None  # per-process chip index, set in the initializer


def chip_index(chips_vrt: str) -> pd.DataFrame:
    """Index the mosaic's chips: absolute path + EPSG:3857 bounds.

    Reading a crop straight from the 29,850-source mosaic VRT costs ~2 s
    because GDAL scans the whole source list per window. Indexing the sources
    once lets each crop be read from a micro-VRT of the two or three chips it
    actually touches, which is ~100× faster.

    Args:
        chips_vrt: path to the mosaic VRT.

    Returns:
        Columns ``path, minx, miny, maxx, maxy`` — one row per chip.
    """
    vrt = Path(chips_vrt)
    gt, rows, band = None, [], 0
    for event, el in ET.iterparse(str(vrt), events=("start", "end")):
        if event == "end" and el.tag == "GeoTransform":
            gt = [float(x) for x in el.text.split(",")]
        elif event == "start" and el.tag == "VRTRasterBand":
            band += 1
            if band > 1:
                break  # every band lists the same sources
        elif event == "end" and el.tag == "ComplexSource" and band == 1:
            dst = el.find("DstRect")
            rows.append((el.find("SourceFilename").text,
                         float(dst.get("xOff")), float(dst.get("yOff")),
                         float(dst.get("xSize")), float(dst.get("ySize"))))
            el.clear()

    df = pd.DataFrame(rows, columns=["name", "xoff", "yoff", "xsize", "ysize"])
    ox, px, _, oy, _, py = gt  # py is negative (north-up)
    df["minx"] = ox + df["xoff"] * px
    df["maxx"] = ox + (df["xoff"] + df["xsize"]) * px
    df["maxy"] = oy + df["yoff"] * py
    df["miny"] = oy + (df["yoff"] + df["ysize"]) * py
    df["path"] = [str((vrt.parent / n).resolve()) for n in df["name"]]
    return df[["path", "minx", "miny", "maxx", "maxy"]]


def _init(index: pd.DataFrame) -> None:
    global _INDEX
    _INDEX = {c: index[c].to_numpy() for c in index.columns}


def _chips_for(bounds: tuple) -> list[str]:
    """Chip paths whose extent intersects ``bounds``."""
    minx, miny, maxx, maxy = bounds
    hit = ((_INDEX["minx"] < maxx) & (_INDEX["maxx"] > minx)
           & (_INDEX["miny"] < maxy) & (_INDEX["maxy"] > miny))
    return list(_INDEX["path"][hit])


def _render_one(rts_id: int, geom, out_dir: Path, png_px: int) -> bool:
    """Write all four crops for one polygon. True if it has no imagery.

    Four, not two: tight and wide, each with and without the red outline. The
    outline is drawn into the pixels, so the app's toggle needs a second copy
    of the imagery rather than a client-side switch.

    All four are read through one micro-VRT built over the *wide* extent,
    which contains the tight extent by construction.
    """
    from osgeo import gdal

    tight, wide = crop_bounds(geom.bounds)
    chips = _chips_for(wide)
    if not chips:
        return True  # no chip covers this polygon at all

    with tempfile.NamedTemporaryFile(suffix=".vrt") as tmp:
        gdal.BuildVRT(tmp.name, chips).FlushCache()
        with rasterio.open(tmp.name) as src:
            empty = not has_imagery(src, tight)
            rendered = {
                f"{rts_id}_t.jpg": render_crop(src, [geom], tight, png_px),
                f"{rts_id}_w.jpg": render_crop(src, [geom], wide, png_px),
                f"{rts_id}_t_plain.jpg": render_crop(src, [geom], tight, png_px,
                                                     outline=False),
                f"{rts_id}_w_plain.jpg": render_crop(src, [geom], wide, png_px,
                                                     outline=False),
            }
    for name, jpg in rendered.items():
        (out_dir / name).write_bytes(jpg)
    return empty


def _render_chunk(items: list[tuple[int, bytes]], out_dir: str,
                  png_px: int) -> tuple[int, list[int]]:
    """Render a chunk of (rts_id, geometry-WKB). Returns (n_done, blank_ids)."""
    out = Path(out_dir)
    done, blank = 0, []
    for rts_id, geom_wkb in items:
        try:
            if _render_one(rts_id, wkb.loads(geom_wkb), out, png_px):
                blank.append(rts_id)
            done += 1
        except Exception:  # noqa: BLE001 - one bad polygon must not kill a chunk
            logger.exception("failed to render rts_id=%s", rts_id)
    return done, blank


def build_crops(candidates: str, chips_vrt: str, out_dir: str,
                workers: int = 90, png_px: int = 560,
                overwrite: bool = False) -> int:
    """Render the crop archive. Returns the number of polygons rendered.

    Args:
        candidates: path to `south_rts_candidates.gpkg`.
        chips_vrt: path to the RGB chip mosaic VRT.
        out_dir: local output directory for the JPEGs.
        workers: process-pool size.
        png_px: crop edge length in pixels.
        overwrite: re-render polygons whose crops already exist.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(candidates)
    logger.info("read %d candidate polygons from %s", len(gdf), candidates)

    pending = []
    for rts_id, geom in zip(gdf["rts_id"].astype(int), gdf.geometry):
        if not overwrite and all((out / f"{rts_id}_{s}.jpg").exists()
                                 for s in ("t", "w", "t_plain", "w_plain")):
            continue
        pending.append((int(rts_id), geom.wkb))
    logger.info("%d to render, %d already present", len(pending),
                len(gdf) - len(pending))
    if not pending:
        return 0

    index = chip_index(chips_vrt)
    logger.info("indexed %d chips from %s", len(index), chips_vrt)

    chunks = [pending[i:i + CHUNK] for i in range(0, len(pending), CHUNK)]
    done, blanks = 0, []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(index,)) as pool:
        futures = [pool.submit(_render_chunk, c, str(out), png_px)
                   for c in chunks]
        for i, fut in enumerate(as_completed(futures), 1):
            n, blank = fut.result()
            done += n
            blanks.extend(blank)
            if i % 20 == 0 or i == len(futures):
                logger.info("chunk %d/%d — %d rendered, %d blank",
                            i, len(futures), done, len(blanks))

    # Sweep the whole inventory, not just this run's slice, so the report is
    # correct after a resume: a polygon with no chips writes no files at all.
    all_ids = gdf["rts_id"].astype(int)
    missing = [i for i in all_ids if not (out / f"{i}_t.jpg").exists()]
    blank_csv = out.parent / "no_imagery.csv"
    pd.DataFrame({"rts_id": sorted(set(blanks) | set(missing))}).to_csv(
        blank_csv, index=False)
    n_blank = len(set(blanks) | set(missing))
    logger.info("rendered %d polygons; %d of %d have no imagery (listed in %s)",
                done, n_blank, len(all_ids), blank_csv)
    if n_blank:
        logger.warning("%d polygons (%.2f%%) have no imagery — they are still "
                       "served; reviewers should rate them 'unsure'",
                       n_blank, 100 * n_blank / len(all_ids))
    return done


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True)
    p.add_argument("--chips-vrt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=90)
    p.add_argument("--png-px", type=int, default=560)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    setup_logging()
    build_crops(args.candidates, args.chips_vrt, args.out_dir, args.workers,
                args.png_px, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
