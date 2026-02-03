# RTS Segmentation Model v2: Data Specification

## Project Context

**Objective**: Semantic segmentation of Retrogressive Thaw Slumps (RTS) in Arctic satellite imagery for pan-arctic mapping (60-74°N).

---

## 1. Data Sources

### 1.1 Primary Training Data: PlanetScope Basemap

| Attribute | Value |
|-----------|-------|
| Product | Global Quarterly PlanetScope Basemap |
| Temporal window | July–September (growing season) |
| Training year | 2024 composites |
| Inference year | 2025 composites |
| Bands | RGB (3 channels) |
| Resolution | 1.3–3.0 m (varies by latitude; ~3 m in study region) |
| Effective GSD | ~10 m (due to mosaic processing) |
| Coverage | Below 74°N only |
| Notes | Proprietary color-correction optimized for CV analytics |


### 1.2 Auxiliary Data Sources

| Source | Resolution | Channels/Derivatives | Purpose |
|--------|------------|---------------------|---------|
| Sentinel-2 | 10 m | NDVI, NIR | Differentiate vegetation from background|
| ArcticDEM | 2 m | Relative Elevation (RE), Shaded Relief (SR) | Terrain context |

### 1.3 Secondary Training Data (Optional)

| Source | Resolution | Volume | Use Case |
|--------|------------|--------|----------|
| Maxar (Yang et al. 2023) | 4 m | 900+ images | Cross-sensor generalisation experiments |

**Note**: Sentinel-2, Maxar, and other sensors exhibit domain shift from PlanetScope. Cross-sensor models require separate experimentation.

---


## 2. Label Source and Refinement

### 2.1 Source Dataset
- **ARTS** (Arctic RTS compilation dataset) provides initial polygon locations
- Polygons were manually refined on 2024 PlanetScope imagery

| Category | Count | Notes |
|----------|-------|-------|
| Positive tiles  | TBD | From ARTS polygon refinement |
| Negative tiles  | 20,000–25,000 | From ARTS confirmed negatives |
| Hard negatives | TBD | From Lingcao Huang's model false positives |

### 2.2 Labeling Criteria
**Include in label** (AND):
- Visible headwall with cast shadow
- Barren slump floor (indicates active RTS)
- Clear morphological distinction from surrounding terrain

**Exclude from label** (OR):
- Features too small to show clear diagnostic characteristics at PlanetScope resolution
- Ambiguous features lacking obvious headwall shadow
- Inactive/stabilized RTS without barren floor

### 2.3 Partial Target Handling

This is critical for training data quality:

| Scenario | Action |
|----------|--------|
| Complete RTS fully within tile | Label as RTS ||
 Partial RTS with **both** headwall and floor visible | Label as RTS |
| Partial RTS with **only** floor visible (no headwall in tile) | Ignore Index：255 |
| Partial RTS with **only** headwall visible (no floor in tile) | Ignore Index：255 |


**Rationale**: The model learns that "only barren floor associated with a headwall with shadow is RTS." Overlapping inference tiles ensure partial targets are detected where both features are visible. Use an Ignore Index （255） for pixels that are part of an RTS but lack the diagnostic headwall in that specific tile. This prevents the model from learning conflicting information while maintaining your strict detection criteria.

### 2.4 Label Values

| Value | Meaning |
|-------|---------|
| 0 | Background (no RTS) |
| 1 | RTS (positive class) |
| 255 | Ignore |
---

## 3. Training Image Specification

### 3.1 Tile Configuration

| Parameter | Value |
|-----------|-------|
| Tile size | 512 × 512 pixels |
| Spatial coverage | ~1.5 km × 1.5 km (at 3 m resolution) |
| CRS | EPSG:3413 (NSIDC Sea Ice Polar Stereographic North) |
| Format | GeoTIFF |
| Grid alignment | Planet tile grid (same grid used for polygon refinement) |

### 3.2 File Naming Convention

```
{tile_id}.tif
```
### 3.3 File Structure
```
---
data/
├── PLANET-RGB/
│   ├── 000001.tif
│   ├── 000002.tif
│   └── ...
├── EXTRA/
│   ├── 000001.tif
│   ├── 000002.tif
│   └── ...
├── labels/
│   ├── 000001.tif
│   └── ...
├── metadata.csv     
└── splits.yaml
```
metadata.csv:

| Tile_id |centroid_lat|centroid_lon| TrainClass | RegionName | UIDs |
|---------|------------|---------|-----------|--------|------|
0001| xx|xx |HardNegative | yakutia| xxx, xxx|

Note:UIDs are RTS UIDs that contained within the tile (used for tracking individual RTS)
RegionName is Arctic Subregion divided based on ecology/permafrost extent (Heidi's working on it)
split.yaml (e.g.):
```
train:
  - yakutia
  - alaska_northslope
  - taymyr
val:
  - yamal
  - nunavut
test:
  - svalbard
  - greenland
```

**PLANET-RGB: derived from PlanetScope Basemap**
```
Image: (512, 512, 3) — RGB
Label: (512, 512, 1) — uint8, values {0, 1， 255}
```

**EXTRA: derived from other sources, resolution resampled to match the RGB**
```
Image: (512, 512, 4) — NDVI + NIR + RE + SR
Label: (512, 512, 1) — uint8, values {0, 1，255}
```
NDVI,NIR are derived from Sentinel2
RE is relative elevation, SR is Shaded relief, both derived from ArcticDEM

**Build order**: Generate planet_rgb first for positive and negative samples, then derive EXTRA by adding auxiliary channels via script.

### 3.4 EXTRA Channel Processing

All auxiliary data must be:
1. Reprojected to EPSG:3413
2. Resampled to match PlanetScope resolution (~3 m)
3. Co-registered with RGB using GeoTIFF bounding box information
4. Stacked as channels

---

## 4. Data Values

**Both PLANET-RGB and EXTRA should store raw values**

Both PLANET-RGB and EXTRA store **raw values** (no normalization applied to stored files).

**Normalisation** Should be calculated per-dataset, rather than per-image, to:
- Preserves absolute radiometric information
- Consistent inference behavior regardless of batch composition
- Satellite imagery has consistent acquisition conditions within a sensor

Normalisation for EXTRA should be done channel-specific to respect the physical signal meanings


## 5. Imbalance and Split

| Estimation | Value |
|-----------|-------|
| Within Positive tiles | 5–30% of tile area |
| Real Arctic prevalance | 0.1-0.5% |

### 5.1 Split Ratios

| Split | Positive | Negative | Purpose |
|-------|----------|----------|---------|
| Train | 80% of N | 80% * M / f1 | Model training by curriculum learning |
| Val-Balanced | 10% of N | 10% * M / f2a | quick val during training |
| Val-Realistic | 10% of N | 10% * M / f2b | Early stopping, full val during training  |
| Test-Realistic | 10% of N | 10% * M / f3 | Final test score to report |

| Factor | Value | 
|--------|-------|
|N | Number of positive tiles|
|M | Number of negative tiles|
|f1 (training) | Start at 1:1, warm up to 1:20 |
|f2a (val-balanced) | 1:20 |
|f2b (val-realistic) | 1:200, 1:1000 | 
|f3 (test-realistic) | 1:200, 1:1000 |

## 6 Spatial Blocking

TBD, need Heidi's help

## 7. Negative Data Strategy

### 7.1 Sources

1. **ARTS confirmed negatives**: Known non-RTS locations
2. **Hard negatives**: False positive locations from Lingcao Huang's model 

### 7.2 Augmentation

Negative samples can be inflated on-the-fly through augmentation to achieve desired imbalance ratios. See Training Guide for augmentation strategy.

---

## 8. Data Check
Run before training:

- Positive labels contain RTS pixels 
- Negative labels contain no RTS pixels 
- Range of value is correct

---

## 9. Channel Index Reference

### RGB
| Index | Channel | 
|-------|---------|
| 0 | Red | 
| 1 | Green | 
| 2 | Blue | 

### EXTRA
| Index | Channel | 
|-------|---------|
| 0 | NDVI |
| 1 | NIR | 
| 2 | Relative Elevation (RE) | 
| 3 | Shaded Relief (SR) |

### Label File
| Value | Meaning |
|-------|---------|
| 0 | Background |
| 1 | RTS |
|255 | ignore |