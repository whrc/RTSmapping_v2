# Extra Channel Visualization

## Purpose

Visually compare auxiliary channels beyond RGB for RTS segmentation input. Two multi-subplot figures overlay the RTS polygon on each channel so spatial relationships between terrain/spectral features and slump morphology are immediately apparent.

## Environment

- **Conda env**: `rts_dataset`
- **Required packages**: earthengine-api, numpy, matplotlib, Pillow (all installed)
- **GEE auth**: must be authenticated (`earthengine authenticate`) before running

## Usage

```bash
conda run -n rts_dataset python plots/extra_channel_vis/extra_channel_plot.py
```

## Input Data

### PlanetScope tile

`plots/extra_channel_vis/tile_1603_1706_c5_r1.tif` — RGB basemap (512x512, EPSG:3857, ~4.77 m/px)

### RTS polygon

Single polygon in EPSG:3857 (embedded in script). Source: ARTS dataset, OBJECTID 187 from RTS final revision batch 2 (Yili). Located in the Taymyr Peninsula region.

## Plot 1: ArcticDEM Derivatives

GEE source: ArcticDEM v4.1 (`UMN/PGC/ArcticDEM/V4/2m_mosaic`)

3x3 subplot grid:

| | | |
|---|---|---|
| RGB (PlanetScope) | DEM (raw elevation) | Relative Elevation |
| Hillshade | Curvature | Shaded Relief |
| Slope | Aspect | TPI |

### Channel definitions

| Channel | Computation | Notes |
|---------|-------------|-------|
| DEM (raw) | ArcticDEM elevation band | meters |
| Relative Elevation | `DEM - focalMean(radius=500m)` | Pixel vs. local average; positive = locally high |
| Hillshade | `ee.Terrain.hillshade(dem)` | Simulated illumination, 0-255 |
| Curvature | `dem.convolve(Laplacian8 kernel)` | Concavity (+) vs convexity (-) |
| Shaded Relief | `0.7 * hillshade_norm + 0.3 * curvature_norm` | Blended terrain visualization |
| Slope | `ee.Terrain.slope(dem)` | Degrees |
| Aspect | `ee.Terrain.aspect(dem)` | Degrees, 0-360 |
| TPI | `DEM - focalMean(radius=300m)` | Topographic Position Index; ridge (+) vs valley (-) |

Output: `plots/extra_channel_vis/arcticdem_derivatives.png`

## Plot 2: Sentinel-Derived Channels

GEE sources: Sentinel-2 SR Harmonized (2024 Jul-Sep median), Sentinel-1 GRD (2024 Jul-Sep mean)

2x5 subplot grid:

| | | | | |
|---|---|---|---|---|
| RGB (PlanetScope) | NIR | SWIR | NBR | NDVI |
| NDWI | NDMI | TCB | TCW | SAR |

### Channel definitions

| Channel | Formula / Source | Notes |
|---------|-----------------|-------|
| NIR | Sentinel-2 B8 | Near-infrared, vegetation reflectance |
| SWIR | Sentinel-2 B11 | Short-wave infrared, moisture/minerals |
| NBR | (B8 - B12) / (B8 + B12) | Normalized Burn Ratio, bare soil |
| NDVI | (B8 - B4) / (B8 + B4) | Vegetation greenness |
| NDWI | (B3 - B8) / (B3 + B8) | Water content |
| NDMI | (B8 - B11) / (B8 + B11) | Moisture index |
| TCB | Tasseled Cap Brightness (Shi & Xu 2019 coefficients) | Bare soil exposure indicator |
| TCW | Tasseled Cap Wetness (Shi & Xu 2019 coefficients) | Soil moisture differences |
| SAR | Sentinel-1 VV backscatter (IW mode) | Surface roughness/moisture |

Output: `plots/extra_channel_vis/sentinel_channels.png`
