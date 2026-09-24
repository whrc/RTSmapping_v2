"""Render the review campaign's crop archive: 2 JPEGs per candidate polygon.

Every polygon in `south_rts_candidates.gpkg` gets a tight crop (~3× the
feature) and a wide crop (~1.5 km context), red outline burned in — the same
geometry the offline pack builder uses (`review/crops.py`), so both review
surfaces show the identical view.

Resumable: a polygon whose four crops already exist on disk is skipped, so an
interrupted run continues where it stopped. **Pass `--overwrite` after the chip
archive changes** — otherwise the resume silently keeps crops rendered against
the older, sparser mosaic, which is how the 2026-08 archive shipped 44% of its
wide views part-black (`review_campaign.md` §4.1).

Crops that could not be filled are reported rather than silently shipped: no
imagery at all to `no_imagery.csv`, and an incomplete 1.5 km context to
`partial_context.csv`.

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

from review.crops import (crop_bounds, has_imagery, imagery_fraction,  # noqa: E402
                          render_crop)
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

CHUNK = 500  # polygons per worker task

# A wide crop below this much real imagery is reported to partial_context.csv.
# Not an error: at the domain edge there is genuinely nothing to show. It is a
# tripwire — the 2026-08 archive shipped 44% of wide views partly blank and
# nothing noticed, because the only coverage probe looked at the tight window.
MIN_WIDE_COVERAGE = 0.90

CROP_SUFFIXES = ("t", "w", "t_plain", "w_plain")

_INDEX = None  # per-process chip index, set in the initializer
_POLYS = None  # per-process candidate index, for the neighbour outlines


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


def _already_rendered(out: Path, rts_id: int) -> bool:
    """True if all four of this polygon's crops are already on disk.

    The resume check, pulled out so it can be tested: when it says True and
    ``--overwrite`` was not passed, the polygon keeps whatever pixels it has —
    including pixels rendered against a mosaic that has since been extended.
    """
    return all((out / f"{rts_id}_{s}.jpg").exists() for s in CROP_SUFFIXES)


def _polygon_index(gdf) -> pd.DataFrame:
    """Candidate bboxes + WKB, the neighbour-outline counterpart of `chip_index`.

    Built once in the parent and handed to every worker through the pool
    initializer, because a worker is given only the polygon it is rendering and
    cannot otherwise know what sits next to it.

    Args:
        gdf: the candidate GeoDataFrame, EPSG:3857.

    Returns:
        Columns ``rts_id, minx, miny, maxx, maxy, wkb`` — one row per polygon.
    """
    b = gdf.geometry.bounds
    return pd.DataFrame({
        "rts_id": gdf["rts_id"].astype(int).to_numpy(),
        "minx": b["minx"].to_numpy(), "miny": b["miny"].to_numpy(),
        "maxx": b["maxx"].to_numpy(), "maxy": b["maxy"].to_numpy(),
        "wkb": [g.wkb for g in gdf.geometry],
    })


def _init(index: pd.DataFrame, polys: pd.DataFrame) -> None:
    global _INDEX, _POLYS
    _INDEX = {c: index[c].to_numpy() for c in index.columns}
    _POLYS = {c: polys[c].to_numpy() for c in polys.columns}


def _chips_for(bounds: tuple) -> list[str]:
    """Chip paths whose extent intersects ``bounds``."""
    minx, miny, maxx, maxy = bounds
    hit = ((_INDEX["minx"] < maxx) & (_INDEX["maxx"] > minx)
           & (_INDEX["miny"] < maxy) & (_INDEX["maxy"] > miny))
    return list(_INDEX["path"][hit])


def _neighbours_for(bounds: tuple, self_id: int) -> list:
    """Other candidate polygons whose bbox overlaps ``bounds``.

    Same vectorized bbox filter as :func:`_chips_for` over the per-process
    candidate index, so no spatial tree is needed: 60k strict-inequality
    comparisons cost far less than the render that follows. Only the hits are
    deserialized from WKB.
    """
    minx, miny, maxx, maxy = bounds
    hit = ((_POLYS["minx"] < maxx) & (_POLYS["maxx"] > minx)
           & (_POLYS["miny"] < maxy) & (_POLYS["maxy"] > miny)
           & (_POLYS["rts_id"] != self_id))
    return [wkb.loads(b) for b in _POLYS["wkb"][hit]]


def _render_one(rts_id: int, geom, out_dir: Path,
                png_px: int) -> tuple[bool, float]:
    """Write all four crops for one polygon.

    Four, not two: tight and wide, each with and without the red outline. The
    outline is drawn into the pixels, so the app's toggle needs a second copy
    of the imagery rather than a client-side switch.

    All four are read through one micro-VRT built over the *wide* extent,
    which contains the tight extent by construction.

    Returns:
        ``(empty, wide_coverage)`` — whether the tight view has no imagery at
        all, and what fraction of the wide view the mosaic actually carries.
    """
    from osgeo import gdal

    tight, wide = crop_bounds(geom.bounds)
    chips = _chips_for(wide)
    if not chips:
        return True, 0.0  # no chip covers this polygon at all

    others = _neighbours_for(wide, rts_id)
    tight_others = _neighbours_for(tight, rts_id)
    with tempfile.NamedTemporaryFile(suffix=".vrt") as tmp:
        gdal.BuildVRT(tmp.name, chips).FlushCache()
        with rasterio.open(tmp.name) as src:
            empty = not has_imagery(src, tight)
            wide_cov = imagery_fraction(src, wide)
            rendered = {
                f"{rts_id}_t.jpg": render_crop(src, [geom], tight, png_px,
                                               neighbours=tight_others),
                f"{rts_id}_w.jpg": render_crop(src, [geom], wide, png_px,
                                               neighbours=others),
                f"{rts_id}_t_plain.jpg": render_crop(src, [geom], tight, png_px,
                                                     outline=False),
                f"{rts_id}_w_plain.jpg": render_crop(src, [geom], wide, png_px,
                                                     outline=False),
            }
    for name, jpg in rendered.items():
        (out_dir / name).write_bytes(jpg)
    return empty, wide_cov


def _render_chunk(items: list[tuple[int, bytes]], out_dir: str,
                  png_px: int) -> tuple[int, list[int], list[tuple[int, float]]]:
    """Render a chunk of (rts_id, geometry-WKB).

    Returns:
        ``(n_done, blank_ids, partial)`` where ``partial`` is the
        ``(rts_id, wide_coverage)`` pairs below :data:`MIN_WIDE_COVERAGE`.
    """
    out = Path(out_dir)
    done, blank, partial = 0, [], []
    for rts_id, geom_wkb in items:
        try:
            empty, wide_cov = _render_one(rts_id, wkb.loads(geom_wkb), out,
                                          png_px)
            if empty:
                blank.append(rts_id)
            if wide_cov < MIN_WIDE_COVERAGE:
                partial.append((rts_id, wide_cov))
            done += 1
        except Exception:  # noqa: BLE001 - one bad polygon must not kill a chunk
            logger.exception("failed to render rts_id=%s", rts_id)
    return done, blank, partial


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
        overwrite: re-render polygons whose crops already exist. Required
            whenever the chip archive has changed under an existing archive.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(candidates)
    logger.info("read %d candidate polygons from %s", len(gdf), candidates)

    pending = []
    for rts_id, geom in zip(gdf["rts_id"].astype(int), gdf.geometry):
        if not overwrite and _already_rendered(out, rts_id):
            continue
        pending.append((int(rts_id), geom.wkb))
    logger.info("%d to render, %d already present", len(pending),
                len(gdf) - len(pending))
    if not pending:
        return 0

    index = chip_index(chips_vrt)
    logger.info("indexed %d chips from %s", len(index), chips_vrt)
    polys = _polygon_index(gdf)

    chunks = [pending[i:i + CHUNK] for i in range(0, len(pending), CHUNK)]
    done, blanks, partials = 0, [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(index, polys)) as pool:
        futures = [pool.submit(_render_chunk, c, str(out), png_px)
                   for c in chunks]
        for i, fut in enumerate(as_completed(futures), 1):
            n, blank, partial = fut.result()
            done += n
            blanks.extend(blank)
            partials.extend(partial)
            if i % 20 == 0 or i == len(futures):
                logger.info("chunk %d/%d — %d rendered, %d blank, "
                            "%d partial context",
                            i, len(futures), done, len(blanks), len(partials))

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

    # Partial *context* is the failure the tight-window probe above cannot see:
    # the polygon is visible but the 1.5 km view around it is part black.
    partial_csv = out.parent / "partial_context.csv"
    pd.DataFrame(sorted(partials), columns=["rts_id", "wide_coverage"]).to_csv(
        partial_csv, index=False)
    if partials:
        logger.warning("%d polygons (%.2f%% of those rendered) have under %.0f%% "
                       "imagery in the wide view (listed in %s)",
                       len(partials), 100 * len(partials) / max(done, 1),
                       100 * MIN_WIDE_COVERAGE, partial_csv)
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
