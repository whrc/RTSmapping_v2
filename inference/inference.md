# RTS Segmentation Model v2: Inference Pipeline

## 1. Inference Objective

Deploy trained segmentation model for pan-arctic inference (60-74°N) on 2025 PlanetScope basemap imagery to produce a comprehensive RTS survey map.

---

## 2. Inference Region

| Parameter | Value |
|-----------|-------|
| Latitude range | 60°N – 74°N |
| Imagery source | PlanetScope Quarterly Basemap (2025) |
| CRS | EPSG:3413 |
| Upper limit constraint | 74°N (basemap availability limit) |

**Note**: Regions above 74°N would require Sentinel-2 (10m) and a separate model due to domain shift.

---

## 3. Tiling Strategy

### 3.1 Tile Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Tile size | 512 × 512 pixels | Matches training |
| Resolution | ~3 m | PlanetScope native |
| Spatial coverage | ~1.5 km × 1.5 km per tile | — |
| Overlap | 25% (128 pixels) | Balance compute vs edge effects |
| Stride | 384 pixels | 512 - 128 |

### 3.2 Overlap Rationale

**Why overlap is necessary**:
- Detection requires **both** headwall shadow AND barren floor visible
- A partial RTS at tile edge may only show one feature
- Overlapping tiles ensure most partial targets appear complete in at least one tile

**Example scenario**:
```
RTS "A" spans tiles 1, 2, and 3:
- Tile 1: Only barren floor visible → No detection
- Tile 2: Only headwall visible → No detection  
- Tile 3: Both headwall AND floor visible → Detection ✓
```

## 4. Inference Workflow

### 4.1 Pipeline Overview

### 4.2 Collaboration with PDG Team
PDG workflow optimization team (Luigi/Todd) handles:
- Parallelized inference infrastructure
- Leverages existing DARTS inference pipeline
- Provide VM through GCP, H100 GPUs x8
- Budget $70k covers training and inference
