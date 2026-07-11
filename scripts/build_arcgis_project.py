"""Load a region's QC package into ArcGIS Pro (post-inference.md's review
step): the merged probability/mask rasters, the RTS polygon layer, and the RGB
"underlying tile" context chips built by scripts/build_rgb_chips.py.

Region-generic: probability/mask are read as `.tif` (single-COG regions like
Banks) or `.vrt` (sharded regions like the full South run — continental-scale
rasters are assembled as many COG shards tied together by a VRT, see
`scripts/assemble_region.py --cog-tile-px`), whichever is present. The RTS
polygon layer is whatever single `*.gpkg` is in the products dir (`banks_rts.
gpkg`, `south_rts.gpkg`, ...) — auto-discovered, not hardcoded.

**Windows-only, run inside ArcGIS Pro's own Python environment** (arcpy is not
importable anywhere else — there is no Linux/CI path for this script). Two
ways to run it:

  1. Open ArcGIS Pro, open (or start a new blank) project, then in the
     Python window (a plain Python console — no IPython, so no `%run` magic):
        ```
        import sys
        sys.path.insert(0, r"E:\path\to\this\script's\folder")
        import build_arcgis_project as bap
        sys.argv = ["build_arcgis_project.py",
                    "--products-dir", r"D:\rts_qc\banks",
                    "--products-uri", "gs://rts-mapping-v2-usw1/inference/banks/products/"]
        bap.main()
        ```
     — this adds layers to the *currently open* project/map.
  2. From the "Python Command Prompt" ArcGIS Pro installs (Start Menu ->
     ArcGIS -> Python Command Prompt; activates arcgispro-py3), pointing at
     an existing .aprx (no open Pro session needed, so `--project` is
     required — "CURRENT" only resolves inside a live Pro session):
     `python build_arcgis_project.py --products-dir D:\rts_qc\banks --project D:\rts_qc\banks.aprx`

Either way, pass `--products-uri` to have it pull the whole products/ prefix
(probability.{tif,vrt}, mask.{tif,vrt} [+ shard dirs for the .vrt case],
<region>_rts.gpkg, region_log.json, rgb_chips/, rgb_chips.vrt) down to
`--products-dir` first via `gcloud storage rsync` (requires the Google Cloud
SDK on the Windows machine — the same one you already use to reach this
project's buckets).

Usage:
    python build_arcgis_project.py --products-dir D:\\rts_qc\\banks ^
        --products-uri gs://rts-mapping-v2-usw1/inference/banks/products/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import arcpy
except ImportError:
    sys.exit(
        "arcpy is not importable. This script must be run inside ArcGIS "
        "Pro's own Python environment (arcgispro-py3), not a plain Windows "
        "Python install — see the module docstring for the two supported "
        "invocation modes."
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def download_products(products_uri: str, products_dir: Path) -> None:
    """One-shot pull of the whole products/ prefix via the GCS CLI.

    Resolves the executable via ``shutil.which`` rather than passing the bare
    "gcloud" to subprocess: on Windows gcloud is installed as ``gcloud.cmd``,
    and unlike a real shell, Python's subprocess doesn't consult PATHEXT for a
    bare name — it needs the resolved path (WinError 2 otherwise).
    """
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise RuntimeError("gcloud not found on PATH — install the Google Cloud SDK")
    products_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [gcloud, "storage", "rsync", "-r", products_uri, str(products_dir)],
        check=True,
    )
    logger.info("Synced %s -> %s", products_uri, products_dir)


def find_rts_layer_source(products_dir: Path) -> str:
    """Discover the region's RTS gpkg + feature class (filename varies by
    region: banks_rts.gpkg, south_rts.gpkg, ...) — there must be exactly one
    .gpkg in the products dir; more than one is an ambiguity worth surfacing
    rather than guessing."""
    gpkgs = sorted(products_dir.glob("*.gpkg"))
    if not gpkgs:
        raise RuntimeError(f"No .gpkg found in {products_dir}")
    if len(gpkgs) > 1:
        raise RuntimeError(f"Multiple .gpkg files in {products_dir}: "
                           f"{[g.name for g in gpkgs]} — expected exactly one")
    gpkg_path = gpkgs[0]
    arcpy.env.workspace = str(gpkg_path)
    fcs = arcpy.ListFeatureClasses()
    if not fcs:
        raise RuntimeError(f"No feature classes found in {gpkg_path}")
    return f"{gpkg_path}\\{fcs[0]}"


def find_raster(products_dir: Path, stem: str) -> str:
    """Resolve `{stem}.tif` (single-COG regions, e.g. Banks) or `{stem}.vrt`
    (sharded regions, e.g. South's cog_tile_px shards) — whichever exists."""
    for ext in (".tif", ".vrt"):
        candidate = products_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(f"Neither {stem}.tif nor {stem}.vrt found in {products_dir}")


def add_layers(m, products_dir: Path):
    """Add the 4 QC layers bottom -> top; returns the RTS polygon layer."""
    rgb_lyr = m.addDataFromPath(str(products_dir / "rgb_chips.vrt"))
    prob_lyr = m.addDataFromPath(find_raster(products_dir, "probability"))
    mask_lyr = m.addDataFromPath(find_raster(products_dir, "mask"))
    rts_source = find_rts_layer_source(products_dir)
    rts_lyr = m.addDataFromPath(rts_source)

    try:
        prob_sym = prob_lyr.symbology
        prob_sym.colorizer.stretchType = "StandardDeviation"
        ramps = m.listColorRamps("Yellow-Orange-Red (Continuous)")
        if ramps:
            prob_sym.colorizer.colorRamp = ramps[0]
        prob_lyr.symbology = prob_sym
        prob_lyr.transparency = 40
    except Exception:
        logger.warning("Could not set the probability layer's symbology "
                        "(Pro-version API drift) — layer added with default "
                        "symbology.", exc_info=True)

    mask_lyr.visible = False  # redundant with the vector layer; kept for QC

    try:
        rts_sym = rts_lyr.symbology
        rts_sym.renderer.symbol.applySymbolFromGallery("Black Outline (1pt)")
        rts_sym.renderer.symbol.color = {"RGB": [255, 0, 0, 100]}
        rts_sym.renderer.symbol.outlineColor = {"RGB": [255, 0, 0, 100]}
        rts_sym.renderer.symbol.size = 0
        rts_lyr.symbology = rts_sym
    except Exception:
        logger.warning("Could not set the RTS polygon layer's symbology "
                        "(Pro-version API drift) — layer added with default "
                        "symbology.", exc_info=True)

    logger.info("Added layers: %s, %s, %s, %s", rgb_lyr.name, prob_lyr.name,
                mask_lyr.name, rts_lyr.name)
    return rts_lyr


def zoom_to_layer(aprx, layer) -> None:
    """Best-effort zoom to the RTS layer's extent (needs an active map view)."""
    try:
        view = aprx.activeView
        extent = view.getLayerExtent(layer, False, True)
        view.camera.setExtent(extent)
    except Exception:
        logger.warning("Could not zoom to the RTS layer extent — no active "
                        "map view, or a Pro-version API mismatch; navigate "
                        "manually.", exc_info=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products-dir", required=True, type=Path,
                   help="local dir to hold/read probability.{tif,vrt}, "
                        "mask.{tif,vrt}, <region>_rts.gpkg, rgb_chips/, "
                        "rgb_chips.vrt")
    p.add_argument("--products-uri", default=None,
                   help="gs:// products prefix to sync down first; omit to "
                        "use --products-dir as already-downloaded")
    p.add_argument("--project", default=None,
                   help="path to an existing .aprx to open; omit to use the "
                        "currently-open ArcGIS Pro project (CURRENT)")
    p.add_argument("--map-name", default=None,
                   help="map to add layers to; omit to use the first map")
    args = p.parse_args()

    if args.products_uri:
        download_products(args.products_uri, args.products_dir)

    aprx = arcpy.mp.ArcGISProject(args.project or "CURRENT")
    maps = aprx.listMaps(args.map_name) if args.map_name else aprx.listMaps()
    m = maps[0] if maps else aprx.createMap("RTS QC")
    # A newly-created map (or one that just isn't the active view) exists in
    # the project but isn't necessarily visible — openView() activates it so
    # the layers we're about to add actually show up on screen.
    m.openView()

    rts_lyr = add_layers(m, args.products_dir)
    zoom_to_layer(aprx, rts_lyr)
    aprx.save()
    logger.info("Saved %s", aprx.filePath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
