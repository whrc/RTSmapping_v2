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
SE_N_BANDS = 64
# A full 64-band 512² tile (~67 MB) exceeds GEE computePixels' ~50 MB limit, so
# fetch_se_raw splits into <=32-band requests (matches se_sar_plot.fetch_se_tile).
SE_MAX_BANDS_PER_FETCH = 32

# Normalization treatment per group (data/data.md §9), the SSoT consumed by
# scripts/compute_normalization_stats.py and data/dataset.py:
#   zscore       -> [clip to CLIP_PERCENTILES] then per-band (x-μ)/σ
#   fixed_scale  -> x / SE_PROTO_SCALE (no z-score; keeps the meaningful zero)
GROUP_NORM_MODE: dict[str, str] = {
    "NDVI": "zscore", "NBR": "zscore", "SE_PCA": "zscore", "TC": "zscore",
    "SE_PROTO": "fixed_scale",
}
SE_PROTO_SCALE = 0.5
CLIP_PERCENTILES = (0.1, 99.9)


def band_norm_mode(band: int) -> str:
    """Normalization mode for an EXTRA band index (data.md §9), via GROUP_BANDS."""
    for group, bands in GROUP_BANDS.items():
        if band in bands:
            return GROUP_NORM_MODE[group]
    raise ValueError(f"band {band} is not in any EXTRA group (0-{N_EXTRA_BANDS - 1})")


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


def s2_sr_composite(geom, year: int):
    """Cloud-masked Jul–Sep `year` median S2 surface-reflectance composite.

    The single source of truth for the Sentinel-2 recipe (collection, window,
    cloud filter, QA60 mask, median, /10000 → reflectance) over `S2_BANDS`.
    `geom` is an ``ee.Geometry`` (e.g. a tile bbox or a grid cell). Both the EXTRA
    index derivation (``s2_image``) and the bulk RGB+NIR export
    (``scripts/export_s2_composites.py``) build on this so the two products match
    (CLAUDE Rule 3/5)."""
    import ee

    def mask_clouds(img):
        qa = img.select("QA60")
        m = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(m)

    return (ee.ImageCollection(S2_COLLECTION)
            .filterBounds(geom)
            .filterDate(f"{year}{S2_WINDOW[0]}", f"{year}{S2_WINDOW[1]}")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_PCT))
            .map(mask_clouds)
            .select(S2_BANDS)
            .median()
            .divide(10000))


def s2_image(bounds, year: int):
    """ee.Image with bands ndvi, nbr, tcb, tcw — Jul–Sep `year` median composite."""
    import ee
    s2 = s2_sr_composite(_bbox(bounds), year)
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


# --- Satellite Embedding (SE) → SE_PCA (2,3,4) + SE_PROTO (5) -----------------
# Self-contained (numpy + ee only) so the data-team inference handoff needs no
# plots/ deps. The global-PCA basis + RTS prototype are built once by
# scripts/build_se_artifacts.py and passed in via load_se_artifacts().

def se_image(bounds, year: int):
    """ee.Image: 64-band Satellite-Embedding annual mosaic over the tile bbox."""
    import ee
    bbox = _bbox(bounds)
    return (ee.ImageCollection(SE_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(bbox)
            .mosaic()
            .toFloat())


def fetch_se_raw(bounds, grid: dict, year: int) -> np.ndarray:
    """Full 64-band SE stack on the co-registered grid → (64, H, W) float32.

    Bands in sorted name order (A00..A63); fetched in <=SE_MAX_BANDS_PER_FETCH
    requests to stay under the computePixels payload limit.
    """
    img = se_image(bounds, year)
    names = sorted(img.bandNames().getInfo())
    out: list[np.ndarray] = []
    for i in range(0, len(names), SE_MAX_BANDS_PER_FETCH):
        chunk = names[i:i + SE_MAX_BANDS_PER_FETCH]
        px = _fetch(img.select(chunk), grid, chunk)
        out.extend(px[b] for b in chunk)
    return np.stack(out, axis=0).astype("float32")


def load_se_artifacts(path) -> dict:
    """Load global-PCA basis + RTS prototype (scripts/build_se_artifacts.py)."""
    d = np.load(path)
    return {"pca_components": d["pca_components"],   # (3, 64)
            "pca_mean": d["pca_mean"],               # (64,)
            "prototype": d["prototype"]}             # (64,) unit


def se_bands(bounds, grid: dict, year: int, artifacts: dict) -> dict[int, np.ndarray]:
    """Return {2,3,4,5}: SE_PCA(3) via global-PCA projection + SE_PROTO cosine.

    SE_PCA = (se - pca_mean) @ pca_components.T  (first 3 global axes).
    SE_PROTO = cosine similarity of each pixel's unit SE vector to the unit RTS
    prototype, ∈ [-1, 1]. NaN where SE has no coverage (propagates like S2).
    """
    se = fetch_se_raw(bounds, grid, year)        # (64, H, W)
    n, h, w = se.shape
    flat = se.reshape(n, -1).T                    # (H*W, 64)
    comps = np.asarray(artifacts["pca_components"], dtype="float32")  # (3, 64)
    mean = np.asarray(artifacts["pca_mean"], dtype="float32")        # (64,)
    proto = np.asarray(artifacts["prototype"], dtype="float32")      # (64,)
    proto = proto / (np.linalg.norm(proto) + 1e-12)                  # ensure unit

    pca = ((flat - mean) @ comps.T).T.reshape(3, h, w).astype("float32")
    norm = np.linalg.norm(flat, axis=1, keepdims=True)               # (H*W, 1)
    unit = flat / np.where(norm < 1e-12, np.nan, norm)
    cos = (unit @ proto).reshape(h, w).astype("float32")             # SE_PROTO
    return {2: pca[0], 3: pca[1], 4: pca[2], 5: cos}
