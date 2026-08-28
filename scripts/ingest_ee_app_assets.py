"""Ingest the public GEE app's EE assets (post-inference/ee_south_app.js).

Tables go in as shapefiles, which is where the sharp edge is: the `.dbf` format
truncates field names to 10 characters, and the inventory schema has two fields
(`centroid_lat`/`centroid_lon`) that COLLIDE under that rule — ogr2ogr would
silently rename one to `centroid_1`. `tile_ids` is dropped outright: it is a
comma-joined list that can exceed the 254-character `.dbf` text cap, and the app
has no use for it. FIELD_MAP below is therefore the single source of truth for
the property names the app reads; change it here and in the app together.

Usage:
    python scripts/ingest_ee_app_assets.py --kind table \
        --source /outputs/.../south_rts_candidates.gpkg \
        --asset  projects/abruptthawmapping/assets/south_rts_candidates \
        --staging gs://rts-mapping-v2-usw1/inference/2025q3_south/ee_staging

    python scripts/ingest_ee_app_assets.py --kind image \
        --source gs://.../density_10km_expected_m2.tif \
        --asset  projects/abruptthawmapping/assets/south_density_10km \
        --pyramiding MEAN --nodata -1
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Long names → the <=10-char names the app reads. SSoT: keep in step with the
# property names in post-inference/ee_south_app.js.
FIELD_MAP = {
    "centroid_lat": "clat",     # collides with centroid_lon at 10 chars
    "centroid_lon": "clon",     # "
    "perimeter_m": "perim_m",
    "nodata_frac": "nodata_f",
    "area_m2_t45": "a_t45",
    "area_m2_t65": "a_t65",
    "area_m2_t80": "a_t80",
}
DROP_FIELDS = ["tile_ids", "detection_scale"]
SHP_PARTS = (".shp", ".dbf", ".shx", ".prj", ".cpg")


def prepare_shapefile(gpkg: str | Path, out_dir: Path, stem: str) -> Path:
    """Rewrite ``gpkg`` as a shapefile with app-ready field names."""
    import geopandas as gpd

    gdf = gpd.read_file(gpkg)
    gdf = gdf.drop(columns=[c for c in DROP_FIELDS if c in gdf.columns])
    gdf = gdf.rename(columns={k: v for k, v in FIELD_MAP.items()
                              if k in gdf.columns})
    too_long = [c for c in gdf.columns if c != "geometry" and len(c) > 10]
    if too_long:
        raise ValueError(f"fields exceed the .dbf 10-char limit: {too_long} — "
                         "add them to FIELD_MAP")
    out_dir.mkdir(parents=True, exist_ok=True)
    shp = out_dir / f"{stem}.shp"
    gdf.to_file(shp, driver="ESRI Shapefile")
    logger.info("%s → %s (%d features, fields %s)", gpkg, shp, len(gdf),
                [c for c in gdf.columns if c != "geometry"])
    return shp


def stage_to_gcs(paths: list[Path], staging: str, project: str) -> None:
    """Copy the shapefile parts to the GCS prefix EE ingests from.

    Uses google-cloud-storage rather than shelling out: neither the dataprep nor
    the training image ships gsutil.
    """
    from google.cloud import storage

    bucket_name, _, prefix = staging.removeprefix("gs://").partition("/")
    bucket = storage.Client(project=project).bucket(bucket_name)
    for p in paths:
        bucket.blob(f"{prefix.rstrip('/')}/{p.name}").upload_from_filename(str(p))
        logger.info("staged %s", f"gs://{bucket_name}/{prefix}/{p.name}")


def _ee(project: str, *args: str) -> str:
    """Run the earthengine CLI and return stdout (warnings stripped)."""
    res = subprocess.run(["earthengine", "--project", project, *args],
                         capture_output=True, text=True, check=True)
    return "\n".join(ln for ln in res.stdout.splitlines()
                     if "warn" not in ln.lower())


# Upload tasks report SUCCEEDED; Export tasks report COMPLETED. Only these two
# sets are terminal — anything else (PENDING, READY, RUNNING, a state EE adds
# later) means keep waiting. Enumerating the *pending* states instead would let
# an unrecognised one abort a perfectly healthy task.
DONE_STATES = ("SUCCEEDED", "COMPLETED")
FAILED_STATES = ("FAILED", "CANCELLED", "CANCEL_REQUESTED")


def wait_for_task(task_id: str, project: str, poll_s: int = 20) -> None:
    """Block until an ingestion task reaches a terminal state."""
    while True:
        info = _ee(project, "task", "info", task_id)
        match = re.search(r"State:\s*(\w+)", info)
        state = match.group(1) if match else "UNKNOWN"
        if state in DONE_STATES:
            logger.info("task %s %s", task_id, state)
            return
        if state in FAILED_STATES:
            raise RuntimeError(f"task {task_id} ended {state}:\n{info}")
        logger.info("task %s %s …", task_id, state)
        time.sleep(poll_s)


def _start_upload(kind: str, source: str, asset: str, project: str,
                  pyramiding: str | None, nodata: str | None) -> str:
    args = ["upload", kind, f"--asset_id={asset}"]
    if kind == "image":
        if pyramiding:
            args.append(f"--pyramiding_policy={pyramiding}")
        if nodata is not None:
            args.append(f"--nodata_value={nodata}")
    args.append(source)
    out = _ee(project, *args)
    task_id = re.search(r"ID:\s*(\S+)", out)
    if not task_id:
        raise RuntimeError(f"no task id in upload output:\n{out}")
    return task_id.group(1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True, choices=("table", "image"))
    p.add_argument("--source", required=True,
                   help="table: local GPKG. image: gs:// URI of the GeoTIFF")
    p.add_argument("--asset", required=True, help="destination EE asset id")
    p.add_argument("--staging", help="gs:// prefix for the shapefile (tables)")
    p.add_argument("--project", default="abruptthawmapping")
    p.add_argument("--pyramiding", default=None, help="image only, e.g. MEAN")
    p.add_argument("--nodata", default=None, help="image only, e.g. -1")
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/ee_ingest"))
    args = p.parse_args()
    setup_logging()

    if args.kind == "table":
        if not args.staging:
            raise SystemExit("--staging is required for table ingests")
        stem = args.asset.rsplit("/", 1)[-1]
        shp = prepare_shapefile(args.source, args.work_dir, stem)
        staging = args.staging.rstrip("/")
        stage_to_gcs([shp.with_suffix(e) for e in SHP_PARTS
                      if shp.with_suffix(e).exists()], staging, args.project)
        source = f"{staging}/{stem}.shp"
    else:
        source = args.source

    task_id = _start_upload(args.kind, source, args.asset, args.project,
                            args.pyramiding, args.nodata)
    logger.info("upload task %s → %s", task_id, args.asset)
    wait_for_task(task_id, args.project)

    # The app is public; every asset it reads must be too, or anonymous viewers
    # get an empty map with no error.
    _ee(args.project, "acl", "set", "public", args.asset)
    logger.info("%s is public", args.asset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
