# RTS Segmentation Model v2: Data Specification

## Project Context

**Objective**: Semantic segmentation of Retrogressive Thaw Slumps (RTS) in Arctic satellite imagery for pan-arctic mapping (60-74°N).

**Core Detection Principle**: An RTS is only labeled/detected when **both** the shadowed headwall **and** barren slump floor are visible in the same image. This dual-feature requirement minimizes false alarms.

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
| Sentinel-2 | 10 m | NDVI, NIR | Vegetation index, spectral extension |
| ArcticDEM | ~2–10 m | Relative Elevation (RE), Shaded Relief (SR) | Terrain context |

### 1.3 Secondary Training Data (Optional)

| Source | Resolution | Volume | Use Case |
|--------|------------|--------|----------|
| Maxar (Yang et al. 2023) | 4 m | 900+ images | Cross-sensor generalization experiments |

**Note**: Sentinel-2, Maxar, and other sensors exhibit domain shift from PlanetScope. Cross-sensor models require separate experimentation.

---

## 2. Label Source and Refinement

### 2.1 Source Dataset
- **ARTS** (Arctic RTS compilation dataset) provides initial polygon locations
- Polygons are being refined on 2024 PlanetScope imagery

### 2.2 Labeling Criteria

**Include in label** (positive):
- Visible headwall with cast shadow
- Barren slump floor (indicates active RTS)
- Clear morphological distinction from surrounding terrain

**Exclude from label**:
- Features too small to show clear diagnostic characteristics at PlanetScope resolution
- Ambiguous features lacking obvious headwall shadow
- Inactive/stabilized RTS without barren floor

### 2.3 Partial Target Handling

This is critical for training data quality:

| Scenario | Action |
|----------|--------|
| Partial RTS with **both** headwall and floor visible | ✅ Label as RTS |
| Partial RTS with **only** floor visible (no headwall in tile) | ❌ Do not label in this tile |
| Partial RTS with **only** headwall visible (no floor in tile) | ❌ Do not label in this tile |
| Complete RTS fully within tile | ✅ Label as RTS |

**Rationale**: The model learns that "only barren floor associated with a headwall with shadow is RTS." Overlapping inference tiles ensure partial targets are detected where both features are visible.

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

### 3.2 Dataset Versions

**Version 1: ARTS-PLANET-BASE**
```
Channels: RGB + Label
Shape: (512, 512, 4)
```

**Version 2: ARTS-PLANET-EXTRA**
```
Channels: RGB + Label + NDVI + NIR + RE + SR
Shape: (512, 512, 8)
```

**Build order**: Generate BASE first for positive and negative samples, then derive EXTRA by adding auxiliary channels via script.

### 3.3 Auxiliary Channel Processing

All auxiliary data must be:
1. Reprojected to EPSG:3413
2. Resampled to match PlanetScope resolution (~3 m)
3. Co-registered using GeoTIFF bounding box information
4. Stacked as additional channels

---

## 4. Sample Volumes (Estimates)

| Category | Count | Notes |
|----------|-------|-------|
| Positive tiles (RTS present) | 2,000–3,000 | From ARTS polygon refinement |
| Negative tiles (no RTS) | 20,000–25,000 | From ARTS confirmed negatives |
| Hard negatives | TBD | From Lingcao Huang's model false positives |

---

## 5. Data Split Strategy

### 5.1 Split Ratios

| Split | Positive | Negative | Purpose |
|-------|----------|----------|---------|
| Train | 80% of N | 80% of M × f1 | Model training |
| Val-Balanced | 10% of N | 10% of M × f2a | Hyperparameter tuning, early stopping |
| Val-Realistic | 10% of N | 10% of M × f2b | Threshold calibration |
| Test-Realistic | 10% of N | 10% of M × f3 | Final evaluation (never tune on this) |

### 5.2 Imbalance Factors

| Factor | Value | Rationale |
|--------|-------|-----------|
| f1 (training) | Start at 1:10, experiment up to 1:50 | Can be dynamic during training |
| f2a (val-balanced) | 1:1 | Clean signal for early stopping |
| f2b (val-realistic) | 1:200, 1:500, 1:1000 | Multiple ratios for threshold calibration |
| f3 (test-realistic) | 1:500 (fixed) | Conservative estimate of real-world prevalence |

**Real-world prevalence estimate**: 0.1%–0.6% (Nitze estimation), corresponding to 1:1000 to 1:167.

### 5.3 Spatial Blocking

**Critical requirement**: Train and test regions must not overlap.

- Group RTS by geographic region
- Assign entire regions to train/val/test (not individual samples)
- Ensures model generalizes to unseen geography

---

## 6. Negative Data Strategy

### 6.1 Sources

1. **ARTS confirmed negatives**: Known non-RTS locations
2. **Hard negatives**: False positive locations from Lingcao Huang's model (to be provided)

### 6.2 Augmentation

Negative samples can be inflated through augmentation to achieve desired imbalance ratios. See Training Guide for augmentation strategy.

---

## 7. File Naming Convention

```
{dataset}_{split}_{polarity}_{region}_{tile_id}.tif

Examples:
arts_planet_base_train_pos_yakutia_00001.tif
arts_planet_extra_val_neg_alaska_00042.tif
```

---

## 8. Quality Control Checklist

- [ ] All positive tiles contain both headwall shadow AND barren floor
- [ ] Partial targets without both features are unlabeled
- [ ] Tiles align with Planet grid
- [ ] Auxiliary channels are co-registered and resolution-matched
- [ ] Spatial blocking ensures no train/test geographic overlap
- [ ] Hard negatives integrated from false positive mining

---

## 9. Open Items / Decisions Needed

| Item | Status | Owner |
|------|--------|-------|
| Final positive sample count | In progress | Yili Yang |
| Hard negative delivery | Pending | Lingcao Huang |
| ArcticDEM version/year selection | TBD | — |
| Exact Planet grid specification | Needs documentation | — |

---

## Appendix: Channel Index Reference

### ARTS-PLANET-BASE
| Index | Channel |
|-------|---------|
| 0 | Red |
| 1 | Green |
| 2 | Blue |
| 3 | Label |

### ARTS-PLANET-EXTRA
| Index | Channel |
|-------|---------|
| 0 | Red |
| 1 | Green |
| 2 | Blue |
| 3 | Label |
| 4 | NDVI |
| 5 | NIR |
| 6 | Relative Elevation (RE) |
| 7 | Shaded Relief (SR) |