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
    "DEM": [8, 9, 10, 11],
}
S2_GROUPS = ("NDVI", "NBR", "TC")        # Sentinel-2 derived
SE_GROUPS = ("SE_PCA", "SE_PROTO")       # Satellite-Embedding derived
DEM_GROUPS = ("DEM",)                    # ArcticDEM derived
S2_BAND_IDX = [0, 1, 6, 7]               # ndvi, nbr, tcb, tcw
SE_BAND_IDX = [2, 3, 4, 5]               # se_pca1..3, se_proto
DEM_BAND_IDX = [8, 9, 10, 11]            # dem_relev, dem_slope, dem_tpi, dem_curv

# DEM bands live at 8-11 rather than 0-3 so a band index means the same thing in
# every config and in band_norm_mode(), but they are NOT in the canonical 8-band
# EXTRA/ tiles. They are written to a 12-band sidecar directory (EXTRA_DEM/) whose
# bands 1-7 stay NaN — see data/data.md §9. N_EXTRA_BANDS stays 8 because it
# describes EXTRA/; the sidecar width is N_EXTRA_BANDS_DEM.
N_EXTRA_BANDS_DEM = 12

# Sentinel-2 acquisition (matches plots/extra_channel_vis/extra_channel_plot.py).
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_WINDOW = ("-07-01", "-09-30")         # growing season (suffix to year)
S2_CLOUD_PCT = 20
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
# Tasseled Cap for Sentinel-2 (Shi & Xu 2019), over S2_BANDS order.
TC_BRIGHTNESS = [0.2381, 0.2569, 0.2934, 0.3020, 0.1863, 0.0818]
TC_WETNESS = [0.1825, 0.1763, 0.1615, 0.0486, -0.7020, -0.6424]

# No-coverage sentinel for S2 indices. computePixels fills masked pixels (no valid
# in-season S2 observation) with 0 — a *valid* NDVI/NBR/TC value — so a cloud/edge gap
# would silently leak in as a zero and bias the per-channel stats. We unmask to this
# sentinel in s2_bands and convert it back to NaN so the no-coverage contract (NaN,
# dropped by compute_normalization_stats + neutralized by apply_norm) holds end-to-end.
# -9999 is far outside every S2 index range and exactly representable in float32.
S2_NODATA_SENTINEL = -9999.0

# ArcticDEM (terrain). The V4 2m mosaic carries elevation + a datamask; its
# coverage is partial over the Planet domain (docs/arcticdem_diagnostic.md).
ARCTICDEM_MOSAIC = "UMN/PGC/ArcticDEM/V4/2m_mosaic"
# Focal radii in GROUND metres for the two scale-relative bands.
DEM_RELEV_RADIUS_M = 500.0     # relative elevation: hillslope context
DEM_TPI_RADIUS_M = 300.0       # topographic position index
# The focal means are computed on an elevation array decimated by this factor over
# a footprint padded by DEM_RELEV_RADIUS_M. A 500 m ground radius is ~307 px at
# 70 deg N, so a full-resolution padded fetch would be ~1126**2 px/tile; the focal
# mean of a >=300 m neighbourhood carries no information at 1.6 m sampling, and
# decimating first cuts the fetch ~60x.
DEM_COARSE_FACTOR = 8
# Elevation halo (px, tile grid) for the 3x3 gradient/Laplacian stencils.
DEM_HALO_PX = 4
# No-coverage sentinel, same contract as S2_NODATA_SENTINEL: ArcticDEM is masked
# over water and data voids, and computePixels would fill those with 0 — a valid
# elevation. Unmask to the sentinel, restore NaN client-side.
DEM_NODATA_SENTINEL = -9999.0

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
    "SE_PROTO": "fixed_scale", "DEM": "zscore",
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
    """Return {band_index: array} for the Sentinel-2 groups (0,1,6,7).

    No-coverage pixels (no valid in-season S2 observation) are masked in Earth Engine;
    computePixels would fill them with 0, indistinguishable from a real index value. We
    unmask to ``S2_NODATA_SENTINEL`` and convert it back to NaN here so cloud/edge gaps
    are honoured as NoData (NaN) throughout the pipeline rather than leaking in as zeros.
    """
    img = s2_image(bounds, year).unmask(S2_NODATA_SENTINEL)
    px = _fetch(img, grid, ["ndvi", "nbr", "tcb", "tcw"])
    out = {0: px["ndvi"], 1: px["nbr"], 6: px["tcb"], 7: px["tcw"]}
    for arr in out.values():
        arr[arr == S2_NODATA_SENTINEL] = np.nan
    return out


# --- ArcticDEM → DEM (8,9,10,11) ---------------------------------------------
# Every derivative is scale-relative: no absolute elevation band, because raw
# elevation is a geographic fingerprint that a segmentation encoder can memorise
# per region rather than a property of the landform.
#
# Ground scale, not map scale. Tiles are EPSG:3857, whose scale factor is 1/cos(lat),
# so a v1.0 tile's ~4.77 map-units/px is ~2.4 ground m/px at 60 deg N and ~1.3 at 74.
# Mercator is conformal, so that single factor applies to both axes and one scalar
# `ground_scale` converts every derivative correctly. Deriving slope or a metre-radius
# focal window on the map grid instead (as plots/extra_channel_vis does) understates
# slope by 1/cos(lat) — a latitude-dependent 2.0-3.6x error across 60-74 deg N.

def ground_scale_m(bounds: tuple[float, float, float, float],
                   size_px: int = 512) -> float:
    """Ground metres per pixel for an EPSG:3857 tile bbox at `size_px`.

    `bounds` = (minx, miny, maxx, maxy) in EPSG:3857 metres.
    """
    minx, miny, maxx, maxy = bounds
    map_scale = (maxx - minx) / size_px
    lat = _lat_of_mercator_y(0.5 * (miny + maxy))
    return map_scale * np.cos(np.radians(lat))


def _lat_of_mercator_y(y: float) -> float:
    """Inverse Web-Mercator northing → latitude in degrees."""
    r = 6378137.0
    return np.degrees(2.0 * np.arctan(np.exp(y / r)) - 0.5 * np.pi)


def _nan_uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """NaN-aware box mean: a void shrinks the window instead of poisoning it."""
    from scipy.ndimage import uniform_filter

    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    total = uniform_filter(filled, size=size, mode="nearest")
    weight = uniform_filter(valid.astype("float32"), size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / weight
    out[weight == 0] = np.nan
    return out


def dem_derivatives(elev: np.ndarray, coarse: np.ndarray,
                    ground_scale: float, coarse_scale: float,
                    coarse_pad_px: float) -> dict[int, np.ndarray]:
    """Terrain derivatives from an elevation tile, keyed by EXTRA band index.

    Args:
        elev: (H+2*halo, W+2*halo) elevation in metres on the tile grid, NaN where
            ArcticDEM has no data. The halo feeds the 3x3 stencils and is cropped off.
        coarse: elevation over the tile padded by ``DEM_RELEV_RADIUS_M`` and
            decimated by ``DEM_COARSE_FACTOR``, for the large-radius focal means.
        ground_scale: ground metres per pixel of `elev`.
        coarse_scale: ground metres per pixel of `coarse`.
        coarse_pad_px: padding per side in `coarse` pixels, i.e. how much of it to
            crop off before resampling onto the tile grid.

    Returns:
        {8: relative elevation (m), 9: slope (deg), 10: TPI (m), 11: curvature (1/m)}
        each (H, W) float32.
    """
    from scipy.ndimage import zoom

    halo = DEM_HALO_PX
    core = elev[halo:-halo, halo:-halo]
    h, w = core.shape

    # Slope: |grad| on the metre grid → degrees. np.gradient is central-difference
    # in the interior, which is why elev carries a halo.
    dy, dx = np.gradient(elev, ground_scale)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))[halo:-halo, halo:-halo]

    # Curvature: Laplacian, 1/m. Concave (slump floor) positive, convex negative.
    lap = (elev[halo - 1:-halo - 1, halo:-halo] + elev[halo + 1:-halo + 1, halo:-halo]
           + elev[halo:-halo, halo - 1:-halo - 1] + elev[halo:-halo, halo + 1:-halo + 1]
           - 4.0 * core) / (ground_scale ** 2)

    # Relative elevation / TPI: elevation minus a focal mean at a fixed GROUND
    # radius. The mean is taken on the padded coarse array so the window sees real
    # terrain, then cropped to the tile footprint and resampled up.
    pad = int(round(coarse_pad_px))
    rel = {}
    for band, radius_m in ((8, DEM_RELEV_RADIUS_M), (10, DEM_TPI_RADIUS_M)):
        size = 2 * max(1, int(round(radius_m / coarse_scale))) + 1
        smooth = _nan_uniform_filter(coarse, size)
        inner = smooth[pad:smooth.shape[0] - pad, pad:smooth.shape[1] - pad]
        upscaled = zoom(inner, (h / inner.shape[0], w / inner.shape[1]),
                        order=1, mode="nearest")
        rel[band] = core - upscaled

    out = {8: rel[8], 9: slope, 10: rel[10], 11: lap}
    # A void is a void in every band. np.gradient's central difference reads the
    # neighbours, not the centre, so without this a NaN pixel would come back with
    # an interpolated slope while its neighbours went NaN — the hole displaced by
    # one pixel instead of where ArcticDEM actually has no data.
    void = ~np.isfinite(core)
    for arr in out.values():
        arr[void] = np.nan
    return {b: arr.astype("float32") for b, arr in out.items()}


def dem_bands(bounds: tuple[float, float, float, float],
              grid: dict) -> dict[int, np.ndarray]:
    """Return {band_index: array} for the ArcticDEM group (8,9,10,11).

    Two computePixels calls per tile: the tile grid plus a small halo at full
    resolution, and a padded decimated grid for the focal means. Pixels where
    ArcticDEM has no data come back NaN (``DEM_NODATA_SENTINEL`` round-trip), which
    compute_normalization_stats drops and apply_norm neutralises.
    """
    import ee

    dem = (ee.Image(ARCTICDEM_MOSAIC).select("elevation")
           .unmask(DEM_NODATA_SENTINEL))

    size_px = int(grid["dimensions"]["width"])
    gs = ground_scale_m(bounds, size_px)

    fine = _fetch(dem, _halo_grid(grid, DEM_HALO_PX), ["elevation"])["elevation"]
    coarse_grid, coarse_scale, coarse_pad_px = _coarse_grid(bounds, grid, gs)
    coarse = _fetch(dem, coarse_grid, ["elevation"])["elevation"]

    for arr in (fine, coarse):
        arr[arr == DEM_NODATA_SENTINEL] = np.nan

    return dem_derivatives(fine, coarse, gs, coarse_scale, coarse_pad_px)


def _halo_grid(grid: dict, halo: int) -> dict:
    """`grid` grown by `halo` px on every side, same pixel size and CRS."""
    t = grid["affineTransform"]
    return {
        "dimensions": {"width": grid["dimensions"]["width"] + 2 * halo,
                       "height": grid["dimensions"]["height"] + 2 * halo},
        "affineTransform": {**t,
                            "translateX": t["translateX"] - halo * t["scaleX"],
                            "translateY": t["translateY"] - halo * t["scaleY"]},
        "crsCode": grid["crsCode"],
    }


def _coarse_grid(bounds: tuple[float, float, float, float], grid: dict,
                 ground_scale: float) -> tuple[dict, float, float]:
    """Decimated grid over the tile padded by DEM_RELEV_RADIUS_M of ground.

    Returns (grid, ground metres per coarse pixel, padding per side in coarse px).
    """
    minx, miny, maxx, maxy = bounds
    size_px = int(grid["dimensions"]["width"])
    map_scale = (maxx - minx) / size_px
    # Pad in map units: ground metres / (ground m per map unit).
    pad_map = DEM_RELEV_RADIUS_M * map_scale / ground_scale
    coarse_map_scale = map_scale * DEM_COARSE_FACTOR
    # Pin the pad to a whole number of coarse pixels so the crop in
    # dem_derivatives lands exactly on the tile footprint.
    pad_px = int(np.ceil(pad_map / coarse_map_scale))
    pad_map = pad_px * coarse_map_scale
    n = int(np.ceil(size_px * map_scale / coarse_map_scale)) + 2 * pad_px
    return ({
        "dimensions": {"width": n, "height": n},
        "affineTransform": {"scaleX": coarse_map_scale, "shearX": 0,
                            "translateX": minx - pad_map,
                            "shearY": 0, "scaleY": -coarse_map_scale,
                            "translateY": maxy + pad_map},
        "crsCode": grid["crsCode"],
    }, ground_scale * DEM_COARSE_FACTOR, float(pad_px))


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
    prototype, ∈ [-1, 1]. No-coverage pixels (the SE mosaic returns an all-zero
    vector there) yield NaN for both SE_PCA and SE_PROTO — without the explicit
    guard, ``(0 - pca_mean) @ comps.T`` would emit a spurious nonzero SE_PCA. NaN is
    the NoData contract (dropped by stats, neutralized by apply_norm), matching S2.
    """
    se = fetch_se_raw(bounds, grid, year)        # (64, H, W)
    n, h, w = se.shape
    flat = se.reshape(n, -1).T                    # (H*W, 64)
    comps = np.asarray(artifacts["pca_components"], dtype="float32")  # (3, 64)
    mean = np.asarray(artifacts["pca_mean"], dtype="float32")        # (64,)
    proto = np.asarray(artifacts["prototype"], dtype="float32")      # (64,)
    proto = proto / (np.linalg.norm(proto) + 1e-12)                  # ensure unit

    norm = np.linalg.norm(flat, axis=1, keepdims=True)               # (H*W, 1)
    no_cov = (norm < 1e-12).reshape(h, w)                            # all-zero SE = no coverage
    pca = ((flat - mean) @ comps.T).T.reshape(3, h, w).astype("float32")
    pca[:, no_cov] = np.nan
    unit = flat / np.where(norm < 1e-12, np.nan, norm)
    cos = (unit @ proto).reshape(h, w).astype("float32")             # SE_PROTO
    return {2: pca[0], 3: pca[1], 4: pca[2], 5: cos}
