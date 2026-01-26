# RTS Segmentation Model v2: Inference Pipeline

## 1. Inference Objective

Deploy trained segmentation model for pan-arctic inference (60-74°N) on 2025 PlanetScope basemap imagery to produce a comprehensive RTS survey map.

**Target timeline**: Start inference within 2 months.

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

### 3.3 Tile Generation

```python
def generate_inference_tiles(region_bounds, tile_size=512, stride=384):
    """
    Generate overlapping tile coordinates for inference.
    
    Args:
        region_bounds: (xmin, ymin, xmax, ymax) in EPSG:3413
        tile_size: Tile dimension in pixels
        stride: Step size in pixels (tile_size - overlap)
    
    Returns:
        List of tile bounds [(xmin, ymin, xmax, ymax), ...]
    """
    tiles = []
    pixel_size = 3.0  # meters
    
    x = region_bounds[0]
    while x < region_bounds[2]:
        y = region_bounds[1]
        while y < region_bounds[3]:
            tile_bounds = (
                x,
                y,
                x + tile_size * pixel_size,
                y + tile_size * pixel_size
            )
            tiles.append(tile_bounds)
            y += stride * pixel_size
        x += stride * pixel_size
    
    return tiles
```

---

## 4. Inference Workflow

### 4.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PAN-ARCTIC INFERENCE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Tile Generation                                         │
│     └── Generate overlapping 512×512 tiles @ 25% overlap    │
│                                                             │
│  2. Image Retrieval                                         │
│     └── Fetch PlanetScope basemap for each tile             │
│                                                             │
│  3. Preprocessing                                           │
│     └── Normalize, add auxiliary channels (if used)         │
│                                                             │
│  4. Model Inference                                         │
│     └── Batch inference on GPU                              │
│     └── Output: probability map per tile                    │
│                                                             │
│  5. Overlap Resolution                                      │
│     └── Merge overlapping predictions                       │
│                                                             │
│  6. Thresholding                                            │
│     └── Apply calibrated threshold → binary mask            │
│                                                             │
│  7. Polygon Extraction                                      │
│     └── Vectorize binary mask → GeoPackage polygons         │
│                                                             │
│  8. Output Generation                                       │
│     └── Probability raster + polygon layer                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Collaboration with PDG Team

**Integration point**: PDG workflow optimization team (Luigi/Todd) handles:
- Parallelized inference infrastructure
- Leverages existing DARTS inference pipeline

**Model team provides**:
- Trained model weights
- Training code / fine-tuning scripts
- Preprocessing specifications
- Threshold and calibration parameters

**PDG constraints**:
- Inference-only workflow (no distributed training needed)
- Single VM with GPUs for fine-tuning
- Open-ended on model architecture

---

## 5. Overlap Resolution

### 5.1 Strategies for Merging Overlapping Predictions

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Max probability** | Take maximum probability in overlap zones | Preserves high-confidence detections | May amplify noise |
| **Average** | Mean probability in overlap zones | Smooths predictions | May suppress weak detections |
| **Weighted average** | Higher weight toward tile center | Reduces edge artifacts | More complex |

**Recommended**: Start with **max probability** — aligns with detection goal (don't miss RTS).

### 5.2 Implementation

```python
def merge_overlapping_predictions(tiles, predictions, tile_size=512, stride=384):
    """
    Merge overlapping tile predictions using max probability.
    
    Args:
        tiles: List of tile bounds
        predictions: List of probability arrays (H, W)
        tile_size: Tile dimension
        stride: Step size
    
    Returns:
        Merged probability raster
    """
    overlap = tile_size - stride
    
    # Initialize output raster (full region size)
    output = np.zeros(region_shape, dtype=np.float32)
    count = np.zeros(region_shape, dtype=np.uint8)
    
    for tile_bounds, pred in zip(tiles, predictions):
        # Map tile to output coordinates
        x_start, y_start = tile_to_output_coords(tile_bounds)
        
        # Max merge
        output[y_start:y_start+tile_size, x_start:x_start+tile_size] = np.maximum(
            output[y_start:y_start+tile_size, x_start:x_start+tile_size],
            pred
        )
    
    return output
```

---

## 6. Post-Processing

### 6.1 Thresholding

Apply calibrated threshold from training (determined on Val-Realistic):

```python
binary_mask = probability_raster > calibrated_threshold
```

**Threshold selection** (from training):
- Target: High precision at acceptable recall
- Calibrated across multiple prevalence assumptions (1:200, 1:500, 1:1000)
- Final threshold documented in model artifacts

### 6.2 Morphological Cleaning (Optional)

```python
from scipy import ndimage

# Remove small isolated predictions (likely noise)
min_area_pixels = 50  # ~450 m² at 3m resolution
labeled, num_features = ndimage.label(binary_mask)
for i in range(1, num_features + 1):
    if (labeled == i).sum() < min_area_pixels:
        binary_mask[labeled == i] = 0

# Fill small holes
binary_mask = ndimage.binary_fill_holes(binary_mask)
```

### 6.3 Polygon Extraction

```python
import rasterio
from rasterio.features import shapes
import geopandas as gpd

def extract_polygons(binary_mask, transform, crs):
    """
    Convert binary mask to vector polygons.
    
    Args:
        binary_mask: Binary numpy array
        transform: Affine transform from raster
        crs: Coordinate reference system
    
    Returns:
        GeoDataFrame with polygon geometries
    """
    # Extract shapes
    mask_shapes = shapes(
        binary_mask.astype(np.uint8),
        mask=binary_mask,
        transform=transform
    )
    
    # Convert to GeoDataFrame
    polygons = []
    for geom, value in mask_shapes:
        if value == 1:
            polygons.append(shape(geom))
    
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    
    # Add attributes
    gdf['area_m2'] = gdf.geometry.area
    gdf['centroid_x'] = gdf.geometry.centroid.x
    gdf['centroid_y'] = gdf.geometry.centroid.y
    
    return gdf
```

---

## 7. Output Products

### 7.1 Primary Outputs

| Product | Format | Description |
|---------|--------|-------------|
| **RTS Polygons** | GeoPackage (.gpkg) | Vector layer with RTS boundaries |
| **Probability Raster** | Cloud-Optimized GeoTIFF | Full pan-arctic probability map |

### 7.2 Polygon Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `geometry` | Polygon | RTS boundary |
| `area_m2` | Float | Area in square meters |
| `centroid_x` | Float | Centroid X coordinate (EPSG:3413) |
| `centroid_y` | Float | Centroid Y coordinate (EPSG:3413) |
| `confidence` | Float | Mean probability within polygon |
| `tile_id` | String | Source tile identifier |

### 7.3 Probability Raster Specification

| Property | Value |
|----------|-------|
| Format | Cloud-Optimized GeoTIFF (COG) |
| Data type | Float32 |
| Value range | [0, 1] |
| NoData | -1 or NaN |
| CRS | EPSG:3413 |
| Resolution | 3 m |
| Compression | LZW or DEFLATE |
| Tiled | Yes (256×256 internal tiles) |
| Overviews | Yes (for visualization) |

---

## 8. Quality Assurance

### 8.1 Visual Spot-Checks

- Sample random tiles across geographic regions
- Verify detections against basemap imagery
- Check for systematic false positive patterns

### 8.2 Quantitative Validation

- Compare against held-out test regions (not used in training)
- Compute precision/recall on validation polygons
- Generate confusion matrix

### 8.3 Hard Negative Mining

**Iterative improvement loop**:

```
1. Run inference
2. Identify false positives (manual review or automated)
3. Add FPs to hard negative training set
4. Retrain model
5. Repeat
```

---

## 9. Compute Estimation

### 9.1 Scale Estimation

| Parameter | Estimate |
|-----------|----------|
| Total area (60-74°N) | ~30 million km² |
| Tile coverage | ~1.5 km × 1.5 km = 2.25 km² |
| Tiles needed (no overlap) | ~13.3 million |
| Tiles with 25% overlap | ~17.8 million |
| Inference time per tile | ~50 ms (A100) |
| **Total GPU hours** | ~250 hours |

### 9.2 Parallelization

PDG workflow handles parallelization across multiple GPUs/nodes.

---

## 10. Error Handling

### 10.1 Missing/Corrupt Tiles

```python
def safe_inference(tile_data, model):
    """Inference with error handling."""
    try:
        if tile_data is None or np.isnan(tile_data).all():
            return None  # Mark as no-data
        
        prediction = model.predict(tile_data)
        return prediction
    
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return None  # Mark for re-processing
```

### 10.2 Edge Cases

| Scenario | Handling |
|----------|----------|
| Cloud cover | No masking in basemap (pre-processed) |
| Water bodies | Model should learn to ignore |
| Missing basemap tiles | Mark as no-data |
| Memory overflow | Reduce batch size, stream tiles |

---

## 11. Deliverables Checklist

### Model Artifacts (to PDG team)
- [ ] Trained model weights (.pth)
- [ ] Model architecture definition
- [ ] Preprocessing pipeline code
- [ ] Calibrated threshold value
- [ ] Inference script

### Output Products
- [ ] Pan-arctic probability raster (COG)
- [ ] RTS polygon layer (GeoPackage)
- [ ] Metadata and documentation
- [ ] QA report

### Documentation
- [ ] Inference configuration (tile size, overlap, threshold)
- [ ] Processing log with any failed tiles
- [ ] Validation metrics on test set

---

## 12. Open Items / Decisions Needed

| Item | Status | Notes |
|------|--------|-------|
| Final overlap percentage | Recommended 25% | To be confirmed after test inference |
| Overlap merge strategy | Recommended max | To be confirmed after test inference |
| Minimum polygon area filter | TBD | Depends on expected RTS size distribution |
| Probability raster compression | TBD | LZW vs DEFLATE vs ZSTD |
| Hard negative mining iteration count | TBD | Based on FP rate in initial results |

---

## Appendix: PDG Integration Checklist

**Questions answered by PDG team**:
- [x] Is the workflow inference only, or training and inference? → **Inference only**
- [x] Special requirements for model training? → **Open-ended, no restrictions**
- [x] Training code required? → **Yes**
- [x] Fine-tuning setup? → **1 VM with GPUs, no parallelization needed**

**Remaining coordination**:
- [ ] Data handoff format (model weights, configs)
- [ ] Inference output location / storage
- [ ] Progress monitoring / logging integration
- [ ] Error handling / retry policy