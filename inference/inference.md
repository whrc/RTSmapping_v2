# RTS Segmentation Model v2: Inference Pipeline

## 1. Inference Objective

Deploy the trained segmentation model for pan-arctic inference (60-74°N) on 2025 PlanetScope basemap imagery to produce an RTS survey map. The pipeline prioritizes **precision over recall** to minimize false alarms in the final product.

---

## 2. Infrastructure

### 2.1 Compute Environment

| Resource | Specification |
|----------|---------------|
| Cloud | Google Cloud Platform |
| VM Type | GPU-enabled VM (specific type TBD with PDG team) |
| Storage | Google Cloud Storage bucket: `abruptthawmapping` |
| Collaboration | PDG workflow optimization team (Luigi/Todd) |

### 2.2 Storage Structure

```
gs://abruptthawmapping/
├── models/
│   └── rts-v2/
│       ├── best_model.pth
│       ├── normalization_stats.json
│       └── config.yaml
├── inference/
│   ├── 2025-Q3/
│   │   ├── tiles/                    # Raw prediction tiles
│   │   │   ├── tile_0001.tif
│   │   │   └── ...
│   │   ├── merged/                   # Merged prediction rasters
│   │   │   ├── region_yakutia.tif
│   │   │   └── ...
│   │   ├── vectors/                  # Vectorized polygons
│   │   │   ├── rts_predictions.gpkg
│   │   │   └── ...
│   │   └── logs/
│   │       └── inference_log.json
│   └── ...
└── basemaps/
    └── 2025-Q3/
        └── ... (input imagery)
```

### 2.3 Docker Environment

**Base Image**: Same as training—`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`

**Additional Inference Requirements**:

| Package | Purpose |
|---------|---------|
| google-cloud-storage | GCS bucket access |
| geopandas | Vector operations |
| shapely | Geometry handling |
| pyproj | Coordinate transformations |

**Docker Configuration for Inference**:

| Flag | Purpose |
|------|---------|
| `--gpus all` | Enable GPU access |
| `-v /path/to/cache:/cache` | Local cache for tiles |
| `--env GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` | GCS authentication |

**GCS Authentication**:
1. Create service account with Storage Object Viewer and Storage Object Creator roles
2. Download JSON key file
3. Mount key file into container and set environment variable

---

## 3. Input Data

### 3.1 Source Imagery

| Attribute | Value |
|-----------|-------|
| Product | Global Quarterly PlanetScope Basemap |
| Year | 2025 |
| Quarter | Q3 (July-September) |
| Bands | RGB |
| Resolution | ~3 m |
| Coverage | 60-74°N (pan-arctic) |
| CRS | EPSG:3413 |

### 3.2 Coverage Estimation

| Parameter | Estimate |
|-----------|----------|
| Total area | ~20 million km² |
| Tile size | 512×512 @ 3m = ~2.36 km² per tile |
| Estimated tiles | ~8-10 million tiles (without overlap) |
| With 50% overlap | ~32-40 million tile inferences |

---

## 4. Tiling Strategy

### 4.1 Tile Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Tile size | 512×512 pixels | Matches training tile size |
| Spatial coverage | ~1.5 km × 1.5 km | At 3m resolution |
| CRS | EPSG:3413 | Consistent with training |
| Format | GeoTIFF | Preserves georeferencing |

### 4.2 Overlap Configuration

Overlapping tiles ensure RTS at tile boundaries are detected where both headwall and floor are visible.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Overlap (stride) | 256 pixels (50%) | Ensures most partial RTS captured in adjacent tile |
| Step size | 256 pixels | tile_size - overlap |

**Overlap rationale**: An RTS split at a tile boundary may show only floor in tile A and only headwall in tile B. With 50% overlap, an intermediate tile C will likely contain both features.

### 4.3 Tile Grid Generation

The inference tile grid is **pre-filtered externally** (land-only, permafrost zones) before the inference pipeline runs. The inference code receives a pre-filtered tile list and processes it as-is — no filtering logic inside the inference container.

1. Define bounding box for inference region (or per-region bounding boxes)
2. Generate tile grid with specified overlap
3. Apply land/permafrost filtering externally (outside this pipeline)
4. Save filtered tile grid as CSV with tile IDs and bounding boxes → this is the `--tile-list` input to the inference script

---

## 5. Normalization

### 5.1 Loading Statistics

**Critical**: Use the exact normalization statistics from training.

1. Load `normalization_stats.json` from model directory
2. Verify dataset version matches expected training data version
3. Apply mean subtraction and std division to each input tile

### 5.2 Application

For each input tile:
1. Load RGB values as float32
2. Subtract channel means: `x = x - mean[channel]`
3. Divide by channel stds: `x = x / std[channel]`
4. Feed normalized tensor to model

**Warning**: If normalization statistics are not applied identically to training, model predictions will be unreliable.

---

## 6. Multi-Resolution Inference

### 6.1 Rationale

RTS range from ~50m to 2+ km. A single resolution cannot optimally detect all sizes:
- Native 3m: Good for small-medium RTS, may miss context for large RTS
- Downscaled: Larger effective field of view captures large RTS

### 6.2 Scale Configuration

| Scale | Effective Resolution | Field of View | Target RTS |
|-------|---------------------|---------------|------------|
| 1.0 | 3m (native) | 1.5 km | Small-medium (50m-500m) |
| 0.5 | 6m | 3 km | Medium-large (200m-1km) |
| 0.25 | 12m | 6 km | Very large (1km+) |

**Recommended configuration**: Start with scales [1.0, 0.5]. Add 0.25 only if large RTS recall is problematic.

### 6.3 Multi-Scale Procedure

For each tile location:

**Scale 1.0 (native)**:
1. Load 512×512 tile at native resolution
2. Normalize using training statistics
3. Run inference → probability map P_1.0

**Scale 0.5**:
1. Load 1024×1024 region centered on tile location
2. Downsample to 512×512 (bilinear interpolation)
3. Normalize using training statistics
4. Run inference → probability map at 512×512
5. Upsample prediction back to 1024×1024
6. Crop center 512×512 → P_0.5

**Scale 0.25**:
1. Load 2048×2048 region centered on tile location
2. Downsample to 512×512
3. Normalize, run inference
4. Upsample to 2048×2048, crop center 512×512 → P_0.25

### 6.4 Scale Fusion

Combine predictions across scales using **pixel-wise maximum**:

```
P_final = max(P_1.0, P_0.5, P_0.25)
```

**Rationale**: If any scale confidently detects RTS, include it. Maximum operation is conservative toward detection while individual scale thresholds control precision.

---

## 7. Test-Time Augmentation (TTA)

### 7.1 Configuration

| Setting | Transforms | Speed Multiplier |
|---------|------------|------------------|
| Disabled | None | 1× |
| Minimal | Identity, hflip | 2× |
| Standard | Identity, hflip, vflip, rot180 | 4× |

**Recommendation**: For pan-arctic inference, use **Minimal TTA** (2×) as balance between accuracy and compute cost. Full TTA on 40M+ tiles is expensive.

### 7.2 TTA Procedure

For each input tile:
1. Original → predict → P_orig
2. Horizontal flip → predict → flip back → P_hflip
3. Average: P_tta = (P_orig + P_hflip) / 2

### 7.3 Combining TTA with Multi-Scale

Order of operations:
1. For each scale:
   a. Apply TTA transforms
   b. Average TTA predictions at this scale
2. Take maximum across scales

Total inference passes per tile location: n_scales × n_tta_transforms

| Configuration | Passes per Location |
|---------------|---------------------|
| 2 scales, no TTA | 2 |
| 2 scales, minimal TTA | 4 |
| 3 scales, minimal TTA | 6 |
| 2 scales, standard TTA | 8 |

---

## 8. Batch Inference

### 8.1 Batching Strategy

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 64-128 | Tune based on GPU memory |
| Tile loading | Async prefetch | Overlap I/O with compute |
| GPU utilization target | >90% | Monitor with nvidia-smi |

### 8.2 Inference Loop

1. **Initialize**: Load model, load normalization stats, set model to eval mode
2. **Tile iteration**:
   - Load batch of tiles from GCS (with prefetching)
   - Normalize batch
   - For each scale: run inference (with TTA if enabled)
   - Fuse scales
   - Save predictions to GCS
3. **Progress tracking**: Log completed tiles, estimated time remaining
4. **Checkpointing**: Save progress every N tiles for resumability

### 8.3 Resumability

The inference job must be resumable after interruption:
1. Maintain manifest of completed tiles in `inference_log.json`
2. On restart, load manifest and skip completed tiles
3. Use atomic writes to GCS (write to temp, then rename)

---

## 9. Prediction Merging

### 9.1 Overlap Handling

Adjacent tiles overlap by 50%. The overlapping regions have multiple predictions that must be merged.

**Merging strategy: Maximum**

For pixels covered by multiple tiles, take the maximum probability:
```
P_merged(x, y) = max(P_tile1(x, y), P_tile2(x, y), ...)
```

**Rationale**: Consistent with the detection philosophy—if any tile view detects RTS, include it.

### 9.2 Merging Procedure

**Option A: On-the-fly merging (memory-efficient)**
1. Create output raster for region with NoData fill
2. For each tile prediction:
   - Read overlapping region from output
   - Compute pixel-wise maximum with new prediction
   - Write merged result back
3. Advantage: Low memory; Disadvantage: Many I/O operations

**Option B: Batch merging (faster)**
1. Accumulate all tile predictions in memory (or memory-mapped file)
2. Apply reduction (maximum) across overlapping tiles
3. Write final merged raster
4. Advantage: Faster; Disadvantage: High memory for large regions

**Recommendation**: Use Option B for manageable regions (e.g., per Arctic subregion), Option A for full pan-arctic if memory-constrained.

### 9.3 Output Chunking

For pan-arctic scale, produce merged outputs per region rather than single global raster:
- Easier to manage and distribute
- Enables parallel processing
- Allows region-specific quality control

---

## 10. Output Specification

### 10.1 Probability Raster

| Attribute | Value |
|-----------|-------|
| Format | Cloud-Optimized GeoTIFF (COG) |
| Data type | Float32 |
| Range | [0.0, 1.0] |
| NoData value | -1.0 |
| CRS | EPSG:3413 |
| Resolution | 3m (native) |
| Compression | LZW |

### 10.2 Binary Mask

| Attribute | Value |
|-----------|-------|
| Format | Cloud-Optimized GeoTIFF (COG) |
| Data type | UInt8 |
| Values | 0 (background), 1 (RTS) |
| NoData value | 255 |
| CRS | EPSG:3413 |
| Resolution | 3m |
| Compression | LZW |

Threshold applied: Use calibrated threshold from training (documented in model config).

### 10.3 Vector Output

| Attribute | Value |
|-----------|-------|
| Format | GeoPackage (.gpkg) |
| Geometry | Polygon (MultiPolygon for fragmented) |
| CRS | EPSG:3413 |

**Attributes per polygon**:

| Field | Type | Description |
|-------|------|-------------|
| rts_id | Integer | Unique identifier |
| area_m2 | Float | Polygon area in square meters |
| perimeter_m | Float | Polygon perimeter in meters |
| centroid_lat | Float | Centroid latitude (WGS84) |
| centroid_lon | Float | Centroid longitude (WGS84) |
| mean_prob | Float | Mean probability within polygon |
| max_prob | Float | Maximum probability within polygon |
| detection_scale | String | Scale(s) that detected this RTS |
| tile_ids | String | Comma-separated tile IDs containing this RTS |

### 10.4 Inference Metadata

Save with each inference run:

**inference_log.json**:

| Field | Description |
|-------|-------------|
| model_version | Model identifier |
| model_checkpoint | Path to model weights |
| normalization_stats_hash | MD5 hash of normalization file |
| inference_date | ISO timestamp |
| basemap_version | 2025-Q3 |
| scales_used | [1.0, 0.5] |
| tta_config | "minimal" |
| threshold | 0.XX (from calibration) |
| n_tiles_processed | Total tiles |
| n_tiles_with_detection | Tiles with any RTS prediction |
| total_rts_area_km2 | Sum of predicted RTS area |
| processing_time_hours | Wall clock time |
| gpu_type | e.g., "NVIDIA H100" |

---

## 11. Quality Control

### 11.1 Sanity Checks During Inference

| Check | Action if Failed |
|-------|------------------|
| Tile has valid data (not all NoData) | Skip tile, log warning |
| Prediction values in [0, 1] | Clip and log error |
| Tile georeferencing valid | Stop and investigate |
| GPU memory stable | Reduce batch size |

### 11.2 Post-Inference Validation

Performed before releasing results (detailed in post-inference.md):
- Visual inspection of sample predictions
- Comparison with known RTS locations
- False positive analysis
- Regional performance assessment

---

## 12. Performance Optimization

### 12.1 I/O Optimization

| Technique | Description |
|-----------|-------------|
| Tile caching | Cache frequently accessed tiles locally |
| Prefetching | Load next batch while current batch processes |
| COG format | Cloud-Optimized GeoTIFF enables efficient partial reads |
| Batch GCS operations | Upload predictions in batches, not per-tile |

### 12.2 GPU Optimization

| Technique | Description |
|-----------|-------------|
| Mixed precision (FP16) | 2× throughput on tensor cores |
| Batch size tuning | Maximize GPU utilization |
| Multiple streams | Overlap data transfer and compute |
| Model compilation | torch.compile() for additional speedup |

### 12.3 Estimated Throughput

| Configuration | Tiles/Second (est.) | Time for 40M tiles |
|---------------|---------------------|-------------------|
| 1 scale, no TTA, batch=64 | ~100-200 | 2-4 days |
| 2 scales, minimal TTA, batch=64 | ~50-100 | 4-8 days |
| 2 scales, standard TTA, batch=64 | ~25-50 | 8-16 days |

**Note**: Estimates are rough; actual performance depends on I/O bandwidth, tile complexity, and GCS latency.

---

## 13. Workflow Integration

### 13.1 PDG Workflow

The inference pipeline integrates with the existing PDG (Permafrost Discovery Gateway) workflow infrastructure developed for DARTS inference.

**Integration points**:
- Input: Basemap tiles from GCS
- Output: Prediction tiles and vectors to GCS
- Logging: Compatible format for PDG monitoring
- Parallelization: Workflow handles VM orchestration

### 13.2 Docker Entry Point

The inference container exposes a CLI interface for PDG workflow integration:

```bash
python scripts/inference.py --config configs/inference.yaml --tile-list tiles.csv
```

- `--config`: YAML file specifying model path, GCS paths, scales, TTA config, threshold
- `--tile-list`: CSV file with tile IDs and bounding boxes to process (pre-filtered by PDG/RTS team)
- Output: Prediction tiles written to GCS path defined in config; `inference_log.json` updated on completion

### 13.3 Parallelization Strategy

**Tile-level parallelism** (managed by PDG workflow):
1. RTS team generates the full filtered tile grid (CSV)
2. PDG team (Luigi/Todd) partitions the CSV into chunks and spawns VMs
3. Each VM runs the inference container with its assigned tile list chunk
4. RTS team merges outputs after all chunks complete

**Within-VM parallelism**:
- Single GPU processes tiles in batches
- Multiple CPU workers handle I/O prefetching
- No multi-GPU within single VM (simplifies code)

### 13.4 Coordination

| Responsibility | Owner |
|----------------|-------|
| Tile grid generation (filtered CSV) | RTS team |
| VM orchestration + tile partitioning | PDG team (Luigi/Todd) |
| Inference Docker container | RTS team |
| Output merging | RTS team |
| Quality control | RTS team |

**Interface contract** (to finalize with PDG team):
- Input: `configs/inference.yaml` + `tiles.csv` (tile_id, bbox columns)
- Output: Prediction tiles at `{config.output_path}/{tile_id}.tif`; log at `{config.output_path}/inference_log.json`

---

## 14. Inference Checklist

### Pre-Inference
- [ ] Model artifacts uploaded to GCS (model, normalization stats, config)
- [ ] Docker image built and pushed to container registry
- [ ] Tile grid generated and validated
- [ ] GCS permissions configured (service account)
- [ ] Test inference on small region successful
- [ ] Throughput estimate matches budget

### During Inference
- [ ] Progress monitoring active
- [ ] GPU utilization >90%
- [ ] No error accumulation in logs
- [ ] Checkpoint saves working

### Post-Inference
- [ ] All tiles processed (compare manifest to grid)
- [ ] Merged rasters generated
- [ ] Vectorization complete
- [ ] Metadata logged
- [ ] Sanity checks passed
- [ ] Ready for quality control (post-inference.md)

---

## 15. Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| OOM errors | Batch size too large | Reduce batch size |
| Slow inference | I/O bottleneck | Enable prefetching, use local cache |
| Inconsistent predictions | Wrong normalization | Verify normalization_stats.json hash |
| Missing tiles in output | Job interrupted | Check manifest, restart from checkpoint |
| High false positive rate | Threshold too low | Re-calibrate threshold on validation set |
| Predictions all zero | Model loading error | Verify model checkpoint, test on known positive |