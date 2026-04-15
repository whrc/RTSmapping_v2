# Data Check

Two separate checks at different lifecycle stages. Source of truth is `data/data.md`.

---

## 1. Format Check

**When**: after a dataset is crafted in the bucket, before any data loading.

**What**: verifies all required files are present with correct names and folder structure per `data.md` §3. Does **not** open or read raster contents.

**Script**: `scripts/check_data_format.py`

```bash
python scripts/check_data_format.py --bucket gs://abrupt_thaw/RTS_MODEL_V2/DATA
```

Output: printed report. Exit 0 on pass, 1 on fail.

### Checks

1. **Folder structure** — bucket root contains required directories (`PLANET-RGB/`, `labels/`) and files (`metadata.csv`, `splits.yaml`). Optional: `EXTRA/`, `version.json`.
2. **File presence** — the directory for image (RGB or EXTRA) contains `.tif` files only; no unexpected nesting or non-`.tif` files.
3. **Naming convention** — all `.tif` filenames follow `{tile_id}.tif` pattern (numeric IDs, consistent zero-padding).
4. **Tile ID correspondence** — tile IDs identical across `PLANET-RGB/`, `labels/`, `EXTRA/` (if present), and `metadata.csv` `Tile_id` column. Lists files and reads CSV column values only — never opens rasters.
5. **metadata.csv schema** — has all required columns in correct order: `Tile_id`, `centroid_lat`, `centroid_lon`, `TrainClass`, `RegionName`, `UIDs`.
6. **splits.yaml structure** — valid YAML; each split key maps to a list of region name strings.

Report: Checks that passed/failed. Number of positive/negative tiles, number of tiles in train/val/test, number of regions in train/val/test.
---

## 2. Content Check

**When**: at data loading time, as a sanity check before training or inference.

**What**: validates actual data values, shapes, dtypes, geospatial properties, and semantic correctness by reading file contents. Check random 5% of the data.

**Script**: `scripts/check_data_content.py`

```bash
python scripts/check_data_content.py --bucket gs://abrupt_thaw/RTS_MODEL_V2/DATA
```

Output: printed report. Exit 0 on pass, 1 on fail.

### Content checks

1. **Metadata values** — `Tile_id` unique; `TrainClass` ∈ {Positive, Negative}; `UIDs` non-empty if Positive; `RegionName` non-empty for all rows.
2. **Splits consistency** — every region in `splits.yaml` exists in metadata; no region in multiple splits; flag unassigned regions.
3. **Raster shape & dtype** — RGB: `(3, 512, 512)` uint8, no NaN. EXTRA: `(4, 512, 512)`, all finite. Label: `(512, 512)` uint8, values ⊂ {0, 1, 255}.
4. **CRS & geospatial** — CRS = EPSG:3857; bounds and transform identical across RGB, label, and EXTRA for each tile.
5. **Label semantics** — Positive tiles have ≥1 pixel equal to 1; Negative tiles have 0 pixels equal to 1.
6. 

Report：checks that passed/failed.