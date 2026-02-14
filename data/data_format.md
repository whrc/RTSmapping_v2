## Data Format Standards

All project outputs must follow these format conventions:

### Raster Formats

| Data Type | Format | Extension | Compression | Notes |
|-----------|--------|-----------|-------------|-------|
| Training images (RGB) | GeoTIFF | `.tif` | LZW | 512×512, EPSG:3413 |
| Training images (EXTRA) | GeoTIFF | `.tif` | LZW | 512×512, 4 channels |
| Training labels | GeoTIFF | `.tif` | LZW | uint8, values {0, 1, 255} |
| Prediction tiles | COG | `.tif` | Deflate | float32 probability |
| Regional mosaics | COG | `.tif` | Deflate | With overviews |
| Binary masks | COG | `.tif` | Deflate | uint8, nodata=255 |
| Pan-arctic mosaic | VRT | `.vrt` | N/A | Virtual mosaic over COGs |

### Vector Formats

| Data Type | Format | Extension | CRS | Notes |
|-----------|--------|-----------|-----|-------|
| RTS polygons (working) | GeoPackage | `.gpkg` | EPSG:3413 | Primary working format, indexed |
| RTS polygons (archive) | GeoParquet | `.parquet` | EPSG:3413 | Long-term storage, analytics |
| RTS polygons (sharing) | GeoJSON | `.geojson` | EPSG:4326 | Small subsets only (<1000 features) |
| Training AOIs | GeoPackage | `.gpkg` | EPSG:3413 | Region boundaries |
| Spatial blocks | GeoPackage | `.gpkg` | EPSG:3413 | Train/val/test splits |

### Metadata & Configuration

| Data Type | Format | Extension | Notes |
|-----------|--------|-----------|-------|
| Training config | YAML | `.yaml` | Hyperparameters, paths |
| Inference config | YAML | `.yaml` | Must reference training config |
| Normalization stats | JSON | `.json` | Per-dataset mean/std, saved with model |
| Tile metadata | JSON | `.json` | tile_id, bounds, paths |
| Experiment tracking | MLflow | N/A | Metrics, artifacts, parameters |
| Statistics/results | JSON | `.json` | Counts, areas, metrics |
| Logs | JSON | `.json` | Structured logging |

### Model Artifacts

| Data Type | Format | Extension | Notes |
|-----------|--------|-----------|-------|
| Model checkpoint | PyTorch | `.pth` | Includes EMA weights, optimizer state |
| Final model (deploy) | PyTorch | `.pth` | EMA weights only, with norm_stats |

### Format Selection Principles

| Principle | Guideline |
|-----------|-----------|
| Rasters | Always COG for outputs; GeoTIFF acceptable for intermediate/training data |
| Vectors (large) | GeoPackage for working, GeoParquet for archive |
| Vectors (small/web) | GeoJSON only for sharing/visualization, must reproject to EPSG:4326 |
| CRS | EPSG:3413 throughout pipeline; convert to EPSG:4326 only for GeoJSON export |
| Compression | Deflate for COGs, LZW for training GeoTIFFs |
| Metadata | YAML for config (human-editable), JSON for computed outputs |

### COG Creation Standard
```yaml
profile:
  driver: GTiff
  compress: deflate
  predictor: 2
  tiled: true
  blockxsize: 512
  blockysize: 512
  BIGTIFF: IF_SAFER
overviews: [2, 4, 8, 16, 32]
resampling: average
```