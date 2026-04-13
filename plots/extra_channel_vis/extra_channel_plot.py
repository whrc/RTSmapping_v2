"""
Extra channel visualization for RTS segmentation.

Produces two figures per polygon comparing auxiliary channels with RTS overlay:
  1. ArcticDEM derivatives (3x3 grid)
  2. Sentinel-derived channels (2x5 grid)

Reads polygon-tile pairs from selected_7_polygons_for_vis.geojson and
polygon_image_mapping.csv.

Requires: earthengine-api, numpy, matplotlib, Pillow
Environment: conda env rts_dataset
GEE auth: must run `earthengine authenticate` first
"""

import csv
import json
import logging
import time
from pathlib import Path

import ee
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
TILE_SIZE = 512

# Tasseled Cap coefficients for Sentinel-2 (Shi & Xu 2019)
# Bands: [B2, B3, B4, B8, B11, B12]
TC_BRIGHTNESS = [0.2381, 0.2569, 0.2934, 0.3020, 0.1863, 0.0818]
TC_WETNESS = [0.1825, 0.1763, 0.1615, 0.0486, -0.7020, -0.6424]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mapping() -> list[dict]:
    """Load polygon-tile mapping from CSV and GeoJSON.

    Returns list of dicts with keys: object_id, tile_name, tile_path,
    polygon_coords, origin_x, origin_y, pixel_scale.
    """
    # Read CSV mapping
    csv_path = SCRIPT_DIR / "polygon_image_mapping.csv"
    oid_to_tile = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
            if row[0].strip() == "ObjectID":
                continue
            oid = int(row[0].strip())
            tile_name = row[1].strip()
            oid_to_tile[oid] = tile_name

    # Read GeoJSON polygons
    geojson_path = SCRIPT_DIR / "selected_7_polygons_for_vis.geojson"
    with open(geojson_path) as f:
        gj = json.load(f)

    entries = []
    for feature in gj["features"]:
        oid = feature["properties"]["OBJECTID"]
        if oid not in oid_to_tile:
            continue
        tile_name = oid_to_tile[oid]
        tile_path = SCRIPT_DIR / f"{tile_name}.tif"
        coords = [(c[0], c[1]) for c in feature["geometry"]["coordinates"][0]]

        # Read tile metadata from GeoTIFF tags
        img = Image.open(tile_path)
        tiepoint = img.tag_v2[33922]
        pixel_scale_tag = img.tag_v2[33550]
        origin_x, origin_y = tiepoint[3], tiepoint[4]
        pixel_scale = pixel_scale_tag[0]

        entries.append({
            "object_id": oid,
            "tile_name": tile_name,
            "tile_path": tile_path,
            "polygon_coords": coords,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "pixel_scale": pixel_scale,
        })

    return entries


def make_gee_grid(origin_x: float, origin_y: float,
                  pixel_scale: float) -> dict:
    """Build GEE grid spec for computePixels."""
    return {
        "dimensions": {"width": TILE_SIZE, "height": TILE_SIZE},
        "affineTransform": {
            "scaleX": pixel_scale, "shearX": 0, "translateX": origin_x,
            "shearY": 0, "scaleY": -pixel_scale, "translateY": origin_y,
        },
        "crsCode": "EPSG:3857",
    }


def polygon_to_pixel_coords(
    coords: list[tuple[float, float]],
    origin_x: float, origin_y: float, pixel_scale: float,
) -> list[tuple[float, float]]:
    """Convert EPSG:3857 polygon coordinates to pixel (col, row)."""
    return [
        ((x - origin_x) / pixel_scale, (origin_y - y) / pixel_scale)
        for x, y in coords
    ]


def load_planetscope_rgb(path: Path) -> np.ndarray:
    """Load PlanetScope tile as float RGB array in [0, 1]."""
    img = Image.open(path)
    arr = np.array(img)[:, :, :3]
    return arr.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------

def initialize_gee() -> None:
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
        logger.info("GEE initialized")
    except Exception:
        ee.Authenticate()
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")


def make_tile_bbox(origin_x: float, origin_y: float,
                   pixel_scale: float) -> ee.Geometry:
    """Create GEE rectangle for the tile extent."""
    xmin = origin_x
    xmax = origin_x + TILE_SIZE * pixel_scale
    ymax = origin_y
    ymin = origin_y - TILE_SIZE * pixel_scale
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj="EPSG:3857")


def fetch_pixels(image: ee.Image, grid: dict,
                 label: str = "") -> dict[str, np.ndarray]:
    """Fetch pixels via computePixels with timing."""
    band_names = image.bandNames().getInfo()
    t0 = time.time()
    result = ee.data.computePixels({
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": grid,
    })
    dt = time.time() - t0
    logger.info("  [%.1fs] %s", dt, label or ", ".join(band_names))
    return {name: result[name] for name in band_names}


def add_polygon_overlay(
    ax: plt.Axes,
    pixel_coords: list[tuple[float, float]],
    color: str = "red",
    linewidth: float = 1.5,
) -> None:
    """Draw RTS polygon outline on a matplotlib axes."""
    polygon = mpatches.Polygon(
        pixel_coords, closed=True,
        facecolor="none", edgecolor=color, linewidth=linewidth,
    )
    ax.add_patch(polygon)


# ---------------------------------------------------------------------------
# GEE data fetching
# ---------------------------------------------------------------------------

def fetch_arcticdem_derivatives(grid: dict,
                                pixel_scale: float) -> dict[str, np.ndarray]:
    """Fetch ArcticDEM and compute terrain derivatives.

    Fast GEE terrain ops (slope, aspect, hillshade, curvature) are computed
    server-side. Large-radius focal means (relative elevation, TPI) are
    computed client-side with scipy to avoid GEE timeouts at high latitudes.
    """
    logger.info("Fetching ArcticDEM derivatives...")

    dem = ee.Image("UMN/PGC/ArcticDEM/V4/2m_mosaic").select("elevation")

    # Terrain derivatives at native 2m (small kernels, fast)
    terrain = (
        dem.rename("elevation")
        .addBands(ee.Terrain.slope(dem).rename("slope"))
        .addBands(ee.Terrain.aspect(dem).rename("aspect"))
        .addBands(ee.Terrain.hillshade(dem).rename("hillshade"))
        .addBands(dem.convolve(ee.Kernel.laplacian8()).rename("curvature"))
    ).toFloat()
    data = fetch_pixels(terrain, grid, "DEM terrain derivatives")

    # Focal operations client-side (GEE focalMean too slow at high latitudes)
    t0 = time.time()
    elev = data["elevation"]
    re_radius_px = max(1, round(500.0 / pixel_scale))
    tpi_radius_px = max(1, round(300.0 / pixel_scale))
    # uniform_filter size must be odd for symmetric window
    re_size = 2 * re_radius_px + 1
    tpi_size = 2 * tpi_radius_px + 1
    data["relative_elevation"] = elev - uniform_filter(elev, size=re_size,
                                                       mode="nearest")
    data["tpi"] = elev - uniform_filter(elev, size=tpi_size, mode="nearest")
    logger.info("  [%.1fs] RE + TPI (client-side, r=%d/%d px)",
                time.time() - t0, re_radius_px, tpi_radius_px)

    # Shaded relief (client-side)
    hs_norm = data["hillshade"] / 255.0
    curv = data["curvature"]
    curv_clipped = np.clip(curv, np.nanpercentile(curv, 2),
                           np.nanpercentile(curv, 98))
    curv_range = curv_clipped.max() - curv_clipped.min()
    if curv_range > 0:
        curv_norm = (curv_clipped - curv_clipped.min()) / curv_range
    else:
        curv_norm = np.zeros_like(curv_clipped)
    data["shaded_relief"] = 0.7 * hs_norm + 0.3 * curv_norm

    return data


def fetch_sentinel2_channels(bbox: ee.Geometry,
                             grid: dict) -> dict[str, np.ndarray]:
    """Fetch Sentinel-2 2024 summer composite and spectral indices."""
    logger.info("Fetching Sentinel-2 channels...")

    def mask_clouds(image: ee.Image) -> ee.Image:
        qa = image.select("QA60")
        mask = (qa.bitwiseAnd(1 << 10).eq(0)
                .And(qa.bitwiseAnd(1 << 11).eq(0)))
        return image.updateMask(mask)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox)
        .filterDate("2024-07-01", "2024-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_clouds)
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
        .median()
        .divide(10000)
    )

    b2, b3, b4 = s2.select("B2"), s2.select("B3"), s2.select("B4")
    b8, b11, b12 = s2.select("B8"), s2.select("B11"), s2.select("B12")

    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("ndvi")
    ndwi = b3.subtract(b8).divide(b3.add(b8)).rename("ndwi")
    nbr = b8.subtract(b12).divide(b8.add(b12)).rename("nbr")
    ndmi = b8.subtract(b11).divide(b8.add(b11)).rename("ndmi")

    bands = ee.Image.cat([b2, b3, b4, b8, b11, b12])
    tcb = bands.expression(
        "+".join(f"b({i}) * {c}" for i, c in enumerate(TC_BRIGHTNESS))
    ).rename("tcb")
    tcw = bands.expression(
        "+".join(f"b({i}) * {c}" for i, c in enumerate(TC_WETNESS))
    ).rename("tcw")

    stacked = (
        s2.select("B8").rename("nir")
        .addBands(s2.select("B11").rename("swir"))
        .addBands(ndvi).addBands(ndwi).addBands(nbr).addBands(ndmi)
        .addBands(tcb).addBands(tcw)
    ).toFloat()

    return fetch_pixels(stacked, grid, "S2 channels")


def fetch_sentinel1_sar(bbox: ee.Geometry,
                        grid: dict) -> dict[str, np.ndarray] | None:
    """Fetch Sentinel-1 SAR VV composite. Returns None if no coverage."""
    logger.info("Fetching Sentinel-1 SAR...")

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(bbox)
        .filterDate("2023-01-01", "2024-12-31")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select("VV")
        .mean()
        .rename("sar_vv")
    ).toFloat()

    try:
        return fetch_pixels(s1, grid, "S1 SAR VV")
    except ee.ee_exception.EEException as e:
        if "no bands" in str(e).lower():
            logger.warning("  No Sentinel-1 coverage at this location")
            return None
        raise


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dem_figure(
    rgb: np.ndarray, dem_data: dict[str, np.ndarray],
    pixel_coords: list[tuple[float, float]],
    title_prefix: str, output_path: Path,
) -> None:
    """Create the ArcticDEM derivatives figure (3x3 grid)."""
    panels = [
        ("RGB (PlanetScope)", rgb, None, {}),
        ("Elevation (m)", dem_data["elevation"], "terrain", {}),
        ("Relative Elevation", dem_data["relative_elevation"], "RdBu_r", {}),
        ("Hillshade", dem_data["hillshade"], "gray", {}),
        ("Curvature", dem_data["curvature"], "RdBu_r", {}),
        ("Shaded Relief", dem_data["shaded_relief"], "gray", {}),
        ("Slope (\u00b0)", dem_data["slope"], "YlOrRd", {}),
        ("Aspect (\u00b0)", dem_data["aspect"], "hsv", {"vmin": 0, "vmax": 360}),
        ("TPI", dem_data["tpi"], "RdBu_r", {}),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle(f"ArcticDEM Derivatives — {title_prefix}",
                 fontsize=16, fontweight="bold", y=0.98)

    for ax, (title, data, cmap, kwargs) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(data)
        else:
            if "vmin" not in kwargs:
                valid = data[~np.isnan(data)] if np.any(np.isnan(data)) else data
                kwargs["vmin"] = np.percentile(valid, 2)
                kwargs["vmax"] = np.percentile(valid, 98)
            im = ax.imshow(data, cmap=cmap, **kwargs)
            fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        add_polygon_overlay(ax, pixel_coords)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_sentinel_figure(
    rgb: np.ndarray, s2_data: dict[str, np.ndarray],
    s1_data: dict[str, np.ndarray] | None,
    pixel_coords: list[tuple[float, float]],
    title_prefix: str, output_path: Path,
) -> None:
    """Create the Sentinel-derived channels figure (2x5 grid)."""
    sar_data = s1_data["sar_vv"] if s1_data is not None else None

    panels = [
        ("RGB (PlanetScope)", rgb, None, {}),
        ("NIR (B8)", s2_data["nir"], "gray", {}),
        ("SWIR (B11)", s2_data["swir"], "gray", {}),
        ("NBR", s2_data["nbr"], "RdYlGn", {}),
        ("NDVI", s2_data["ndvi"], "RdYlGn", {}),
        ("NDWI", s2_data["ndwi"], "RdYlGn", {}),
        ("NDMI", s2_data["ndmi"], "RdYlGn", {}),
        ("TC Brightness", s2_data["tcb"], "YlOrBr", {}),
        ("TC Wetness", s2_data["tcw"], "BrBG", {}),
        ("SAR VV (dB)", sar_data, "gray", {}),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"Sentinel-Derived Channels — {title_prefix}",
                 fontsize=16, fontweight="bold", y=0.98)

    for ax, (title, data, cmap, kwargs) in zip(axes.flat, panels):
        if data is None:
            ax.text(0.5, 0.5, "No data\navailable",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="gray")
            ax.set_facecolor("#f0f0f0")
        elif cmap is None:
            ax.imshow(data)
        else:
            valid = data[~np.isnan(data)] if np.any(np.isnan(data)) else data
            if "vmin" not in kwargs:
                kwargs["vmin"] = np.percentile(valid, 2)
                kwargs["vmax"] = np.percentile(valid, 98)
            im = ax.imshow(data, cmap=cmap, **kwargs)
            fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if data is not None:
            add_polygon_overlay(ax, pixel_coords)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    entries = load_mapping()
    logger.info("Loaded %d polygon-tile pairs", len(entries))

    initialize_gee()

    for i, entry in enumerate(entries, 1):
        oid = entry["object_id"]
        tile_name = entry["tile_name"]
        label = f"OID {oid} ({tile_name})"
        logger.info("=== [%d/%d] %s ===", i, len(entries), label)

        t_start = time.time()

        rgb = load_planetscope_rgb(entry["tile_path"])
        pixel_coords = polygon_to_pixel_coords(
            entry["polygon_coords"],
            entry["origin_x"], entry["origin_y"], entry["pixel_scale"],
        )
        grid = make_gee_grid(
            entry["origin_x"], entry["origin_y"], entry["pixel_scale"])
        bbox = make_tile_bbox(
            entry["origin_x"], entry["origin_y"], entry["pixel_scale"])

        dem_data = fetch_arcticdem_derivatives(grid, entry["pixel_scale"])
        s2_data = fetch_sentinel2_channels(bbox, grid)
        s1_data = fetch_sentinel1_sar(bbox, grid)

        out_prefix = f"oid{oid}"
        plot_dem_figure(
            rgb, dem_data, pixel_coords, label,
            SCRIPT_DIR / f"{out_prefix}_arcticdem.png")
        plot_sentinel_figure(
            rgb, s2_data, s1_data, pixel_coords, label,
            SCRIPT_DIR / f"{out_prefix}_sentinel.png")

        logger.info("  Total for %s: %.1fs", label, time.time() - t_start)

    logger.info("Done — all %d polygons processed.", len(entries))


if __name__ == "__main__":
    main()
