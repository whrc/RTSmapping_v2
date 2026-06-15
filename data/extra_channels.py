"""Shared EXTRA-channel derivation (data/data.md §9) — the single source of truth
used by BOTH training-tile generation (2024) and the data team's inference-tile
generation (2025), so the two produce identical bands (CLAUDE Rule 3).

EXTRA is the canonical 8-band stack, fixed order (data.md §9):
    0 NDVI, 1 NBR, 2-4 SE_PCA(3), 5 SE_PROTO, 6 TCB, 7 TCW
Sources: Sentinel-2 (NDVI/NBR/TC) + Google Satellite Embedding (SE_PCA/SE_PROTO),
both Earth Engine. Group → band-index map below is referenced by the generator
and by configs (channels.extra).

Earth Engine is imported lazily so importing this module does not require
earthengine-api in the plain training image (only the generator needs it).
Consolidated from plots/extra_channel_vis/{extra_channel_plot,se_sar_plot}.py.
"""

from __future__ import annotations

import numpy as np

# Canonical band layout (data.md §9). One source of truth for the generator + configs.
N_EXTRA_BANDS = 8
GROUP_BANDS: dict[str, list[int]] = {
    "NDVI": [0],
    "NBR": [1],
    "SE_PCA": [2, 3, 4],
    "SE_PROTO": [5],
    "TC": [6, 7],
}
S2_GROUPS = ("NDVI", "NBR", "TC")        # Sentinel-2 derived
SE_GROUPS = ("SE_PCA", "SE_PROTO")       # Satellite-Embedding derived
S2_BAND_IDX = [0, 1, 6, 7]               # ndvi, nbr, tcb, tcw
SE_BAND_IDX = [2, 3, 4, 5]               # se_pca1..3, se_proto

# Sentinel-2 acquisition (matches plots/extra_channel_vis/extra_channel_plot.py).
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_WINDOW = ("-07-01", "-09-30")         # growing season (suffix to year)
S2_CLOUD_PCT = 20
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
# Tasseled Cap for Sentinel-2 (Shi & Xu 2019), over S2_BANDS order.
TC_BRIGHTNESS = [0.2381, 0.2569, 0.2934, 0.3020, 0.1863, 0.0818]
TC_WETNESS = [0.1825, 0.1763, 0.1615, 0.0486, -0.7020, -0.6424]

# Satellite Embedding (matches plots/extra_channel_vis/se_sar_plot.py).
SE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def init_ee(project: str = "pdg-project-406720") -> None:
    """Initialize Earth Engine via ADC against the high-volume endpoint."""
    import ee
    ee.Initialize(project=project,
                  opt_url="https://earthengine-highvolume.googleapis.com")


def tile_grid(bounds: tuple[float, float, float, float], size_px: int = 512) -> dict:
    """computePixels grid co-registered to an RGB tile's EPSG:3857 GeoTIFF bbox
    (data.md §3.4). `bounds` = (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bounds
    sx = (maxx - minx) / size_px
    sy = (maxy - miny) / size_px
    return {
        "dimensions": {"width": size_px, "height": size_px},
        "affineTransform": {"scaleX": sx, "shearX": 0, "translateX": minx,
                            "shearY": 0, "scaleY": -sy, "translateY": maxy},
        "crsCode": "EPSG:3857",
    }


def _bbox(bounds: tuple[float, float, float, float]):
    import ee
    minx, miny, maxx, maxy = bounds
    return ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:3857",
                                 evenOdd=False)


def _fetch(image, grid: dict, band_names: list[str]) -> dict[str, np.ndarray]:
    import ee
    arr = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY",
                                 "grid": grid})
    return {b: np.asarray(arr[b], dtype="float32") for b in band_names}


def s2_image(bounds, year: int):
    """ee.Image with bands ndvi, nbr, tcb, tcw — Jul–Sep `year` median composite."""
    import ee
    bbox = _bbox(bounds)

    def mask_clouds(img):
        qa = img.select("QA60")
        m = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(m)

    s2 = (ee.ImageCollection(S2_COLLECTION)
          .filterBounds(bbox)
          .filterDate(f"{year}{S2_WINDOW[0]}", f"{year}{S2_WINDOW[1]}")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_PCT))
          .map(mask_clouds)
          .select(S2_BANDS)
          .median()
          .divide(10000))
    b4, b8, b12 = s2.select("B4"), s2.select("B8"), s2.select("B12")
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("ndvi")
    nbr = b8.subtract(b12).divide(b8.add(b12)).rename("nbr")
    tcb = s2.expression("+".join(f"b({i})*{c}" for i, c in enumerate(TC_BRIGHTNESS))).rename("tcb")
    tcw = s2.expression("+".join(f"b({i})*{c}" for i, c in enumerate(TC_WETNESS))).rename("tcw")
    return ee.Image.cat([ndvi, nbr, tcb, tcw]).toFloat()


def s2_bands(bounds, grid: dict, year: int) -> dict[int, np.ndarray]:
    """Return {band_index: array} for the Sentinel-2 groups (0,1,6,7)."""
    px = _fetch(s2_image(bounds, year), grid, ["ndvi", "nbr", "tcb", "tcw"])
    return {0: px["ndvi"], 1: px["nbr"], 6: px["tcb"], 7: px["tcw"]}
