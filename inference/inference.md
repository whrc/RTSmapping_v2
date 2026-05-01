# RTS Segmentation Model v2: Inference Pipeline

## 1. Inference Objective

Deploy the trained segmentation model for pan-arctic inference (60-74°N) on 2025 PlanetScope basemap imagery to produce an RTS survey map. The pipeline prioritizes **precision over recall** to minimize false alarms in the final product.

The data and model operation in inference should exactly match those in training. The best 'recipe' will be provided once the training and experiments are done.
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
│   └── rts-v2-seed42/                   # one deployment package per seed
│       ├── weights.pth                  # EMA weights only (see training.md §4.3)
│       ├── normalization_stats.json     # channel-name bindings (training.md §4.5)
│       ├── model_config.yaml            # architecture, backbone, channels, data.tile_size (input size derives from it)
│       ├── deployment_config.yaml       # threshold, temperature, tta, precision, torch_compile, scales, fusion
│       ├── run_metadata.json            # git_sha, mlflow_run_id, training_date, seed
│       └── requirements_frozen.txt      # exact env for reproducibility
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

This section owns the post-calibration deployment-package layout. MLflow-side artifacts produced during training (per-epoch metrics, figures, `run_summary.md`, etc.) are spec'd in `training/experiments.md §1.3`; on-disk checkpoint payloads (`best_deployment.pth`, `resume_latest-*.pth`) in `training.md §4.3`.

Note: `scripts/package_model.py` renames the training-time `best_deployment.pth` to `weights.pth` when assembling this deployment package — same EMA state dict, new filename.

### 2.3 Docker Environment

**Base Image**: Same as training — see `computing/docker_training.md` for the authoritative Dockerfile and base image.

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
| Resolution | 4.77 m projected (EPSG:3857; Web Mercator zoom 15; constant in projected space). Ground sample varies with latitude (see `training.md §8.3`). |
| Coverage | 60-74°N (pan-arctic) |
| CRS | EPSG:3857 |

### 3.2 Coverage Estimation

| Parameter | Estimate |
|-----------|----------|
| Total area | ~20 million km² |
| Tile size | 512×512 @ 4.77 m projected = 2442 m projected ≈ 5.81 km² per tile (projected; ground area shrinks at high latitude) |
| Estimated tiles | ~3.4 million tiles (without overlap, projected) |
| With overlap (stride per `configs/deployment.yaml.inference.stride_px`, see §4.2) | tile inferences = base_tiles × `(tile_size / stride_px)²`; at default stride 344 → ~7.5M (≈2.22× compute multiplier) |

---

## 4. Tiling Strategy

### 4.1 Tile Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Tile size | 512×512 pixels | Matches training tile size |
| Spatial coverage | ~2.4 km × 2.4 km projected (512 × 4.77 m) | Web Mercator zoom 15; constant in projected space, shrinks with latitude on the ground |
| CRS | EPSG:3857 | Consistent with training |
| Format | GeoTIFF | Preserves georeferencing |

### 4.2 Overlap Configuration (math-derived)

Overlap exists so that an RTS straddling a tile boundary is captured fully in *some* tile. At tile size T = 2442 m projected (512 px × 4.77 m), an RTS of length L (in projected meters) fits entirely in at least one tile iff stride **S ≤ T − L**.

Known RTS-size distribution (source: training label statistics):

| RTS population | Max bbox edge L | Required stride S = T−L | Stride in pixels (S/4.77) | Overlap p = 1 − S/T | Compute multiplier (1/(1−p))² |
|----------------|-----------------|-------------------------|---------------------------|---------------------|-------------------------------|
| 99.5% | ≤ 800 m | ≤ 1642 m | ≤ 344 px | ≥ 33% | 2.22× |
| 99.9% | ≤ 1300 m | ≤ 1142 m | ≤ 239 px | ≥ 53% | 4.59× |

**Default: stride 344 px (~33% overlap), persisted as `configs/deployment.yaml.inference.stride_px`** — chosen for the 99.5% RTS-size row above. Change the config to retune.

**Overlap rationale**: An RTS split at a tile boundary may show only floor in tile A and only headwall in tile B. At the default stride, intermediate tiles will contain both features for >99.5% of the RTS size distribution.

Flow:
1. Single pass at the configured `inference.stride_px` across all filtered tiles.
2. Merge per §4.4 → regional probability rasters.
3. Threshold candidate polygons.

### 4.3 Overlap Aggregation

Fusion method: **distance-from-tile-center weighted average**, Gaussian weighting with σ = 128 px in tile coordinates, normalized per pixel.

Rationale: edge-of-tile predictions come from locations where the model has seen fewer surrounding pixels within *this* tile. Center-of-tile predictions are more trustworthy. Max fusion (taking the highest probability across tiles) is recall-biased and contradicts §1's precision-over-recall goal; averaging preserves calibration.

Implementation: per-tile probability rasters persist to GCS first, then a separate regional merge pass computes the weighted average per output pixel.

**Note on training loss**: this edge-down-weighting is an **inference-only fusion decision** across multiple predictions of the same physical pixel. Training loss weights all pixels uniformly — weighting by tile position during training would teach the model to ignore edges, which would hurt exactly the inference scenario where edge predictions are being averaged in. Orthogonal decisions.

### 4.4 Tile Grid Generation

The inference tile grid is **pre-filtered externally** (land-only, permafrost zones) before the inference pipeline runs. The inference code receives a pre-filtered tile list and processes it as-is — no filtering logic inside the inference container.

1. Define bounding box for inference region (or per-region bounding boxes)
2. Apply land/permafrost filtering externally (outside this pipeline)
3. Generate tile grid using `configs/deployment.yaml.inference.stride_px`
4. Save filtered tile grid as CSV with tile IDs and bounding boxes → this is the `--tile-list` input to the inference script

---

## 5. Normalization

### 5.1 Loading Statistics

**Critical**: Use the exact normalization statistics from training. Training-inference consistency on normalization is codified in `training.md §4.1` and §4.5.

1. Load `normalization_stats.json` from the deployment package.
2. **Assert channel-name binding** (`training.md §4.5`): `stats["rgb"]["channel_names"] == ["R", "G", "B"]` and, if EXTRA channels are declared, `stats["extra"]["channel_names"] == [c.name for c in model_config["channels"]["extra"]]`. Prevents silent position-vs-name mismatches if the 2025 basemap API changes band ordering. Abort inference on mismatch.
3. Apply mean subtraction and std division per channel using the name-bound stats (not positional).

### 5.2 Application

Use the exact normalization methods and statistics identically to training.

### 5.3 NoData Handling

Per `training.md §4.4`, the training side labels NoData pixels as ignore=255 so the model never receives gradient signal from them. The inference side mirrors this:

| Case | Treatment |
|------|-----------|
| Full-NoData tile | Skipped at the tile-list stage. Manifest-logged with reason `"all_nodata"`. |
| Partial-NoData tile | Predict normally (substitute per-channel training mean for NoData pixels before normalization, matching training). After prediction, **mask the output**: `pred_raster[input_nodata_mask] = -1.0` (the NoData value declared in §9.1). |

Rationale: the model output on NoData input is undefined. Propagating NoData through to the probability raster ensures downstream overlap aggregation (§4.4) and vectorization (§9.3) treat those pixels correctly.

### 5.4 Pre-deployment drift check

Before running full inference on a new region, run `scripts/check_inference_normalization.py` (owned by the training team) against a sample of 2025 tiles from that region:
- Computes per-channel mean/std on the 2025 sample.
- Compares to `normalization_stats.json`.
- Reports drift as `|Δmean| / σ_training` and `|σ_sample / σ_training − 1|` per channel.
- **Concern thresholds**: |Δmean| > 0.5σ_training OR |σ_sample / σ_training − 1| > 0.25. If tripped, pause deployment and investigate — likely distribution shift from 2024 to 2025 imagery, a region-specific radiometric issue, or a basemap-API change.

---

## 6. Multi-Resolution Inference

### 6.1 Rationale

RTS range from ~50m to 2+ km. A single resolution cannot optimally detect all sizes:
- Native 4.77 m projected: Good for small-medium RTS, may miss context for large RTS
- Downscaled: Larger effective field of view captures large RTS

### 6.2 Scale Configuration

| Scale | Effective Resolution (projected) | Field of View (projected) | Target RTS |
|-------|----------------------------------|---------------------------|------------|
| 1.0 | 4.77 m (native) | 2.4 km | Small-medium (50m-500m) |
| 0.5 | 9.55 m | 4.9 km | Medium-large (200m-1km) |

**Phase 1 default: scale 1.0 only.** Multi-scale deployment is gated by a feasibility test (§6.4). Training is at scale 1.0 only per `training.md §8.3`, but the fractal nature of earth features plus the encoder's multi-scale receptive fields suggest scale-0.5 inference *may* work without retraining. Test before assuming.

### 6.3 Multi-Scale Procedure

For each tile location:

**Scale 1.0 (native)**:
1. Load 512×512 tile at native resolution
2. Normalize using training statistics
3. Run inference → probability map P_1.0

**Scale 0.5** (only if §6.4 gate passes):
1. Load 1024×1024 region centered on tile location
2. Downsample to 512×512 (bilinear interpolation)
3. Normalize using training statistics
4. Run inference → probability map at 512×512
5. Upsample prediction back to 1024×1024
6. Crop center 512×512 → P_0.5

**Edge case — basemap boundary.** A scale-0.5 fetch needs 1024 × 1024 projected pixels centered on the tile. If any side of that window falls outside the basemap coverage of the input region (geographic edge or NoData border wider than 256 px), **skip scale 0.5 for that tile** and treat its scale-0.5 prediction as NoData in the §7.3 fusion. Do not pad with reflection or zeros — the model has not seen such patterns. The §7.3 valid-scales rule degrades gracefully to scale-1.0-only for these edge tiles.

### 6.4 Multi-Scale Feasibility Gate

Multi-scale evaluation is **optional and deferred**: the canonical Test-Realistic result comes from `scripts/evaluate_test.py` at `scales: [1.0]` (see `training.md §4.6`). Multi-scale runs **after** the 1× number is locked, lives in this Phase 2 inference pipeline, and is gated as below — it never runs inside `evaluate_test.py`.

Run once per trained model, post-calibration, pre-deployment. Owned by `scripts/inference_feasibility.py` (Phase 1 Step 8.5). Procedure:

1. Run scale-0.5 inference on the val set using the baseline (scale-1.0-trained) model.
2. Average-fuse with cached scale-1.0 val predictions (per §7.3).
3. Compute three measurements at the calibrated threshold:
   - PR-AUC on the **large-RTS subset** (bbox > 500 m).
   - PR-AUC on the full val set.
   - Global false-positive-rate delta vs scale-1.0-only.

**Decision gate** — ship multi-scale if **both**:
- Large-RTS PR-AUC gain ≥ +2%
- Global FP-rate delta ≤ +10%

Otherwise keep `scales: [1.0]`. Context-expanded training (fetch 2× physical area, downsample to 512) is a Phase-1.5 consideration triggered only if the gate fails *and* post-inference analysis identifies large-RTS recall as the primary precision bottleneck.

The gate's outcome is written into `deployment_config.yaml.scales` and the feasibility report is attached to the MLflow run.

---

## 7. Test-Time Augmentation (TTA)

### 7.1 Configuration

| Setting | Transforms | Speed Multiplier |
|---------|------------|------------------|
| Disabled | None | 1× |
| Minimal | Identity, hflip | 2× |
| Standard | Identity, hflip, vflip, rot180 | 4× |

**Recommendation**: For pan-arctic inference, use **Minimal TTA** (2×) as balance between accuracy and compute cost. Full TTA on the §3.2 tile count is expensive.

### 7.2 TTA Procedure

For each input tile:
1. Original → predict → P_orig
2. Horizontal flip → predict → flip back → P_hflip
3. Average: P_tta = (P_orig + P_hflip) / 2

### 7.3 Combining TTA with Multi-Scale

Order of operations (matches §8.2 step 2):
1. For each scale:
   a. For each TTA transform:
      - Run model → raw **logits**.
      - Apply temperature scaling: `scaled_logits = logits / temperature` (per `training.md §12.1`).
      - Apply sigmoid: `probs = sigmoid(scaled_logits)`.
      - Apply the inverse TTA transform to the probability map.
   b. Average TTA probability maps within this scale (arithmetic mean).
2. **Average probability maps across scales** (arithmetic mean over **valid** scales — see NoData rule below), then apply the calibrated threshold for the binary mask.

Temperature scaling **must be applied to logits before sigmoid**, not to probabilities. Folding temperature into the per-pass sigmoid keeps the math consistent with the calibration definition in `training.md §12.1`.

Max fusion was the original spec but biases toward recall (any scale says "positive" → positive), directly contradicting §1's precision-over-recall priority. Arithmetic averaging preserves probability calibration and lets the threshold do its job.

**NoData handling during scale fusion**: a per-pixel scale prediction is treated as NoData when it equals `−1.0` (the §5.3 sentinel) **or** when it falls inside the input NoData mask of that scale's tile fetch. Per-pixel fusion rule: arithmetic mean over the valid scales for that pixel. If all scales are NoData at a pixel, the fused output is `−1.0`.

Total inference passes per tile location: n_scales × n_tta_transforms

| Configuration | Passes per Location |
|---------------|---------------------|
| 1 scale, no TTA (Phase 1 default) | 1 |
| 1 scale, minimal TTA | 2 |
| 2 scales, no TTA | 2 |
| 2 scales, minimal TTA | 4 |
| 2 scales, standard TTA | 8 |

### 7.4 TTA Cost–Benefit

Pan-arctic cost analysis for the §3.2 tile count (~7.5M at default stride 344) on A100 (~$3.67/hr on-demand):

| Config | Passes/tile | Throughput (tiles/s) | Wallclock @ 7.5M | GPU-hrs | Cost |
|--------|-------------|----------------------|------------------|---------|------|
| No TTA | 1 | ~150 | 14 hr | 14 | ~$50 |
| Minimal (identity, hflip) | 2 | ~75 | 28 hr | 28 | ~$100 |
| Standard (identity, hflip, vflip, rot180) | 4 | ~37 | 56 hr | 56 | ~$210 |
| Full D4 (8 symmetries) | 8 | ~19 | 110 hr | 110 | ~$400 |

Against the $70K training+inference budget, all four configs are affordable — the choice is driven by **precision preservation at the calibrated threshold**, not cost. TTA averaging can either improve calibration (good) or pull confident positives below the threshold (bad for precision-over-recall).

**TTA is validated before deployment, not assumed**: Step 8.5b of Phase 1 measures val PR-AUC and precision@threshold under each TTA config using the cached val predictions. Ship the cheapest config that (a) gains ≥ 1% PR-AUC *and* (b) drops precision by ≤ 0.5% at the calibrated threshold. Default in `configs/deployment.yaml`: `tta: none`.

---

## 8. Batch Inference

### 8.1 Batching Strategy

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 64-128 | Tune based on GPU memory |
| Tile loading | Async prefetch | Overlap I/O with compute |
| GPU utilization target | >90% | Monitor with nvidia-smi |

### 8.2 Inference Loop

1. **Initialize**:
   - Load deployment package directory (see §2.2). Required files: `weights.pth`, `normalization_stats.json`, `model_config.yaml`, `deployment_config.yaml`.
   - Build model per `model_config.yaml`; load `weights.pth` into the model state dict (already EMA — see `training.md §4.3`).
   - Load `normalization_stats.json`; assert channel-name binding per §5.1.
   - Load `deployment_config.yaml`: `threshold`, `temperature`, `tta`, `precision`, `torch_compile`, `scales`, `fusion`. These must match the values used during calibration (`training.md §4.6`).
   - `model.eval()`; if `torch_compile: true`, run `torch.compile(model)` here.
2. **Tile iteration** (sequence per §7.3):
   - Load batch of tiles from GCS (with prefetching).
   - Handle NoData per §5.3 (skip full-NoData tiles; mean-substitute partial NoData before normalization).
   - Normalize batch per §5.2.
   - For each scale in `scales`:
     - For each TTA transform (per `tta`):
       - Forward pass → raw logits.
       - Apply temperature: `scaled_logits = logits / temperature`.
       - Apply sigmoid: `probs = sigmoid(scaled_logits)`.
       - Apply inverse TTA transform to the probability map.
     - Average TTA probability maps within this scale.
   - Fuse across scales per §7.3 (arithmetic mean over valid scales).
   - The **probability raster is written pre-threshold**; the calibrated threshold is applied separately to produce the binary mask (§9.2).
   - Mask NoData in output raster per §5.3 (`pred[nodata_mask] = -1.0`).
   - Save probability tile to GCS.
3. **Progress tracking**: Log completed tiles, estimated time remaining.
4. **Checkpointing**: Save progress every N tiles for resumability.

### 8.3 Resumability

The inference job must be resumable after interruption:
1. Maintain manifest of completed tiles in `inference_log.json`
2. On restart, load manifest and skip completed tiles
3. Use atomic writes to GCS (write to temp, then rename)
---

## 9. Output Specification

### 9.1 Probability Raster

| Attribute | Value |
|-----------|-------|
| Format | Cloud-Optimized GeoTIFF (COG) |
| Data type | Float32 |
| Valid range | [0.0, 1.0] |
| NoData sentinel | -1.0 (out-of-range; uniquely identifies NoData) |
| CRS | EPSG:3857 |
| Resolution | 4.77 m projected (native; Web Mercator zoom 15) |
| Compression | Deflate |

### 9.2 Binary Mask

| Attribute | Value |
|-----------|-------|
| Format | Cloud-Optimized GeoTIFF (COG) |
| Data type | UInt8 |
| Values | 0 (background), 1 (RTS) |
| NoData value | 255 |
| CRS | EPSG:3857 |
| Resolution | 4.77 m projected (native; Web Mercator zoom 15) |
| Compression | Deflate |

Threshold applied: Use calibrated threshold from training (documented in model config).

### 9.3 Vector Output

| Attribute | Value |
|-----------|-------|
| Format | GeoPackage (.gpkg) |
| Geometry | Polygon (MultiPolygon for fragmented) |
| CRS | EPSG:3857 |

**Attributes per polygon**:

| Field | Type | Description |
|-------|------|-------------|
| rts_id | Integer | Unique identifier |
| area_m2 | Float | Polygon area in square meters (geodesic) |
| perimeter_m | Float | Polygon perimeter in meters (geodesic) |
| centroid_lat | Float | Centroid latitude (WGS84) |
| centroid_lon | Float | Centroid longitude (WGS84) |
| mean_prob | Float | Mean probability within polygon |
| max_prob | Float | Maximum probability within polygon |
| detection_scale | String | Scale(s) that detected this RTS |
| tile_ids | String | Comma-separated tile IDs containing this RTS |

### 9.4 Inference Metadata

Save with each inference run:

**inference_log.json**:

| Field | Description |
|-------|-------------|
| model_version | Model identifier (e.g., `rts-v2-seed42`) |
| deployment_package_path | `gs://` URI of the deployment package directory |
| model_checkpoint_sha | SHA256 of `weights.pth` |
| inference_date | ISO timestamp |
| basemap_version | 2025-Q3 |
| scales_used | e.g., `[1.0]` or `[1.0, 0.5]` (per §6.4 gate) |
| fusion_method | `weighted_mean` \| `max` \| `consensus` (default `weighted_mean`) |
| tta_config | `none` \| `minimal` \| `standard` \| `full` |
| precision | `bf16` \| `fp16` \| `fp32` (must match calibration) |
| torch_compile | boolean (must match calibration) |
| threshold | Calibrated threshold from `deployment_config.yaml` |
| temperature | Calibrated temperature (§12.1 of training.md) |
| stride_px | value used at run time, mirrors `configs/deployment.yaml.inference.stride_px` |
| overlap_aggregation | `gaussian_weighted_mean`, σ from `configs/deployment.yaml.inference.fusion_sigma_px` |
| n_tiles_processed | Total tiles |
| n_tiles_skipped_nodata | Tiles skipped per §5.3 |
| n_tiles_with_detection | Tiles with any RTS prediction |
| total_rts_area_km2 | Sum of predicted RTS area |
| processing_time_hours | Wall clock time |
| gpu_type | e.g., "NVIDIA H100" |

---

## 10. Quality Control

### 10.1 Sanity Checks During Inference

| Check | Action if Failed |
|-------|------------------|
| Tile has valid data (not all NoData) | Skip tile, log warning |
| Prediction values in [0, 1] | Clip and log error |
| Tile georeferencing valid | Stop and investigate |
| GPU memory stable | Reduce batch size |

### 10.2 Post-Inference Validation

Performed before releasing results (detailed in post-inference.md):
- Visual inspection of sample predictions
- Comparison with known RTS locations
- False positive analysis
- Regional performance assessment

---

## 11. Performance Optimization

### 11.1 I/O Optimization

| Technique | Description |
|-----------|-------------|
| Tile caching | Cache frequently accessed tiles locally |
| Prefetching | Load next batch while current batch processes |
| COG format | Cloud-Optimized GeoTIFF enables efficient partial reads |
| Batch GCS operations | Upload predictions in batches, not per-tile |

### 11.2 GPU Optimization

| Technique | Description |
|-----------|-------------|
| Mixed precision | BF16 on A100/H100 (preferred — no dynamic loss scaling); FP16 fallback on older GPUs. Must match `training.md §4.6` calibration precision. The operative source of truth is `configs/deployment.yaml.precision`; both training-time AMP and inference read from there. |
| Batch size tuning | Maximize GPU utilization |
| Multiple streams | Overlap data transfer and compute |
| Model compilation | **Opt-in only.** `torch.compile()` changes numerics slightly; if enabled at deployment but disabled during calibration (or vice versa), the calibrated threshold is systematically wrong. Phase 1 baseline: `torch_compile: false`. Enable only when a benchmark demonstrates > 15% throughput gain *and* calibration is re-run with compile enabled. |

### 11.3 Estimated Throughput

| Configuration | Tiles/Second (est.) | Wallclock for the §3.2 tile count |
|---------------|---------------------|-----------------------------------|
| 1 scale, no TTA, batch=64 | ~100-200 | ≈ tiles / throughput |
| 2 scales, minimal TTA, batch=64 | ~50-100 | ≈ tiles / throughput |
| 2 scales, standard TTA, batch=64 | ~25-50 | ≈ tiles / throughput |

**Note**: Estimates are rough pre-Phase-1 numbers; actual performance depends on I/O bandwidth, tile complexity, and GCS latency. Replace with the measured A100/H100 throughput from `scripts/inference_feasibility.py` (Phase 1 Step 8.5) before publishing the deployment plan.

---

## 12. Workflow Integration

### 12.1 PDG Workflow

The inference pipeline integrates with the existing PDG (Permafrost Discovery Gateway) workflow infrastructure developed for DARTS inference.

**Integration points**:
- Input: Basemap tiles from GCS
- Output: Prediction tiles and vectors to GCS
- Logging: Compatible format for PDG monitoring
- Parallelization: Workflow handles VM orchestration

### 12.2 Docker Entry Point

The inference container exposes a CLI interface for PDG workflow integration:

```bash
python scripts/inference.py --config configs/deployment.yaml --tile-list tiles.csv
```

- `--config`: `configs/deployment.yaml` — single source for threshold, temperature, scales, tta, precision, torch_compile, fusion, stride_px, fusion_sigma_px (see §2.2 deployment package)
- `--tile-list`: CSV file with tile IDs and bounding boxes to process (pre-filtered by PDG/RTS team)
- Output: Prediction tiles written to GCS path defined in config; `inference_log.json` updated on completion

### 12.3 Parallelization Strategy

**Tile-level parallelism** (managed by PDG workflow):
1. RTS team generates the full filtered tile grid (CSV)
2. PDG team (Luigi/Todd) partitions the CSV into chunks and spawns VMs
3. Each VM runs the inference container with its assigned tile list chunk
4. RTS team merges outputs after all chunks complete

**Within-VM parallelism**:
- Single GPU processes tiles in batches
- Multiple CPU workers handle I/O prefetching
- No multi-GPU within single VM (simplifies code)

### 12.4 Coordination

| Responsibility | Owner |
|----------------|-------|
| Tile grid generation (filtered CSV) | RTS team |
| VM orchestration + tile partitioning | PDG team (Luigi/Todd) |
| Inference Docker container | RTS team |
| Output merging | RTS team |
| Quality control | RTS team |

**Interface contract** (to finalize with PDG team):
- Input: `configs/deployment.yaml` + `tiles.csv` (tile_id, bbox columns)
- Output: Prediction tiles at `{config.output_path}/{tile_id}.tif`; log at `{config.output_path}/inference_log.json`

---

## 13. Inference Checklist

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

## 14. Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| OOM errors | Batch size too large | Reduce batch size |
| Slow inference | I/O bottleneck | Enable prefetching, use local cache |
| Inconsistent predictions | Wrong normalization | Verify normalization_stats.json hash |
| Missing tiles in output | Job interrupted | Check manifest, restart from checkpoint |
| Global FP rate ≫ val reported | Train-inference distribution shift | Run `scripts/check_inference_normalization.py` on a 2025 sample (§5.4) and compare per-channel drift vs `normalization_stats.json`. If drift is real, consider histogram matching or retraining with 2025 data included. |
| Regional FP rate ≫ val reported | Region has characteristics under-represented in val | Collect 50–100 hand-labelled negatives from that region; calibrate a region-specific threshold per `training.md §6.4`. Do not re-run the global calibration — re-running on the same val set cannot fix a regional bias. |
| Calibration-deployment mismatch | Precision / TTA / compile differ between calibration and run | Verify `inference_log.json.precision`, `.tta_config`, `.torch_compile` match the deployment package's `deployment_config.yaml`. Inference aborts at startup on mismatch. **TODO(impl):** the abort behavior is a spec promise — `scripts/inference.py` must implement the precision/tta/compile assertion at startup. Currently no code enforces it. |
| Predictions all zero | Model loading error | Verify `weights.pth` SHA256 in run log matches deployment package; confirm EMA weights loaded (not random-init). Test on a known-positive val tile first. |