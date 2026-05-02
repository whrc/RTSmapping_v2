# RTS Segmentation Model v2: Data Specification

## Project Context

**Objective**: Semantic segmentation of Retrogressive Thaw Slumps (RTS) in Arctic satellite imagery for pan-arctic mapping (60-74°N).

---
**Data versioning**: Use semantic versioning (major.minor)
- Major: Added new training data or significant changes to processing
- Minor: No new training data but changes in existing labels

## 1. Data Sources

### 1.1 Primary Training Data: PlanetScope Basemap

| Attribute | Value |
|-----------|-------|
| Product | Global Quarterly PlanetScope Basemap |
| Temporal window | July–September (growing season) |
| Training year | 2024 composites |
| Inference year | 2025 composites |
| Bands | RGB (3 channels) |
| Resolution | 4.77 m projected (EPSG:3857; Web Mercator zoom 15; constant in projected space). Ground sample = 4.77 × cos(latitude) m: ~1.63 m at 70°N, ~1.32 m at 74°N. |
| Effective GSD | ~10 m (due to mosaic processing) |
| Coverage | Below 74°N only |
| Notes | Proprietary color-correction optimized for CV analytics |


### 1.2 Auxiliary Data Sources

| Source | Resolution | Purpose |
|--------|------------|---------|
| Sentinel-2 | 10 m | Vegetation / burn / spectral indices (NDVI, NBR, Tasselled Cap pair) |
| Google Satellite Embedding | varies | Pretrained feature representation (PCA axes + RTS-prototype cosine similarity) |

The full v2.0 EXTRA channel layout — group IDs, band indices, total band count — lives in §9 (single source of truth). Other auxiliary sources (ArcticDEM derivatives, NDMI, SAR backscatter, etc.) were considered but not included in v2.0; see §9 for the deferred-to-v2.1 list.

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
| Negative tiles  | TBD | From ARTS confirmed negatives + hard negatives |

### 2.2 Labeling Criteria
**Include in label** (AND):
- Visible headwall with cast shadow (visually concaved landscape)
- Barren slump floor (indicates active RTS)
- Clear morphological distinction from surrounding terrain

**Exclude from label** (OR):
- Features too small to show clear diagnostic characteristics at PlanetScope resolution 
- Ambiguous features lacking obvious headwall shadow
- Inactive/stabilized RTS without barren floor
- Long debris tongue or mudflow that is distance from the slump floor (no thawing)

### 2.3 Consistency

HOW TO DEFINE BOUNDARY: enforcing perfect consistency would be exhausting and arguably impossible for gradational geomorphic boundaries. The label boundary always try to follow the visible colour/texture contrast between disturbed and undisturbed ground; where no clear contrast exists, connect the endpoints of the headwall arc with a natural curve. 

PATTERN: made sure the overall morphology read as a collapsed landscape (concaved).

### 2.3 Partial Target Handling

This is critical for training data quality:

| Scenario | Action |
|----------|--------|
| Complete RTS fully within tile | Label as RTS |
| Partial RTS with **both** headwall and floor visible | Label as RTS |
| Partial RTS with **only** floor visible (no headwall in tile) | Ignore Index：255 |
| Partial RTS with **only** headwall visible (no floor in tile) | Ignore Index：255 |


**Rationale**: The model learns that "only barren floor associated with a headwall with shadow is RTS." Overlapping inference tiles ensure partial targets are detected where both features are visible. Use an Ignore Index （255） for pixels that are part of an RTS but lack the diagnostic headwall in that specific tile. This prevents the model from learning conflicting information while maintaining strict detection criteria. This is feature engineering with domain knowledge which especially important when training data is limited.

How to make the decision of whether a partial object should be trained: put the partial object to isolation (masking the adjacent tiles when labelling), if the partial object can be confirmed without neighbour tiles context, train it.


### 2.4 Label Values

| Value | Meaning |
|-------|---------|
| 0 | Background (no RTS) |
| 1 | RTS (positive class) |
| 255 | Ignore |
---
The ignore values could be applied to several conditions, for example:
- partial object that can't be confirmed without neighbouring tile context, even if it's obvious if with context
- RTS-like features that can't be confirmed under the Planet image quality/resolution, even if it's clear in Esri basemap



## 3. Training Image Specification

### 3.1 Tile Configuration

| Parameter | Value |
|-----------|-------|
| Tile size | 512 × 512 pixels |
| Spatial coverage | ~2.4 km × 2.4 km projected (512 × 4.77 m); ground coverage shrinks with latitude. |
| CRS | EPSG:3857 (Pseudo-Mercator -- Spherical Mercator, Google Maps, OpenStreetMap, Bing, ArcGIS, ESRI) |
| Format | GeoTIFF |
| Grid alignment | Planet tile grid (same grid used for polygon refinement) |

### 3.2 File Naming Convention

```
{tile_id}.tif
```
### 3.3 File Structure
Data lives in the GCS bucket, mounted via gcsfuse at training time. All paths are configured in the YAML config — no hardcoded paths in code (see `configs/baseline.yaml:data`).

GCS directory layout:
```
{data.root}/
├── PLANET-RGB/
│   ├── 000001.tif
│   ├── 000002.tif
│   └── ...
├── EXTRA/
│   ├── 000001.tif          ← multi-band, channel order per §9
│   ├── 000002.tif
│   └── ...
├── labels/
│   ├── 000001.tif
│   └── ...
├── metadata.csv
├── splits.yaml             ← lists region names per split (see below)
└── splits_summary.json     ← generated by scripts/create_splits.py; reports tile counts
```

**splits.yaml example format** — lists Arctic region names per split (not individual tile IDs):
more details see domain/inference_domain.md
```yaml
train:
  - elias range tundra
val:
  - arctic foothills tundra
test:
  - beringia lowland tundra
```
The DataLoader cross-references `metadata.csv` (which has `RegionName` per tile) to resolve tile IDs for each split. `scripts/create_splits.py` generates both `splits.yaml` and `splits_summary.json` (with per-split tile counts); both are committed to the repo for reproducibility.
metadata.csv:

| Tile_id |centroid_lat|centroid_lon| TrainClass | RegionName | UIDs |
|---------|------------|---------|-----------|--------|------|
0001| xx|xx |Negative | yakutia| |
0002| xx|xx |Positive | yakutia| xxx, xxx|

Note: TrainClass values are `Positive` or `Negative` only. Hard negatives, if exist (e.g. from Lingcao Huang's model false positives), are stored as `Negative` — no separate class needed.
UIDs are RTS UIDs contained within the tile (used for tracking individual RTS); empty for Negative tiles.
RegionName is Arctic subregion defined by ecology/permafrost extent (boundaries provided by Heidi Rodenhizer, see files in '/domain').

**PLANET-RGB: derived from PlanetScope Basemap**
```
Image: (512, 512, 3) — RGB
Label: (512, 512, 1) — uint8, values {0, 1， 255}
```

**EXTRA: derived from other sources, resolution resampled to match the RGB**
```
Image: (512, 512, 8) — multi-band GeoTIFF; v2.0 layout fixed by §9
Label: (512, 512, 1) — uint8, values {0, 1, 255}
```
The v2.0 EXTRA disk layout and its 8 channels (NDVI, NBR, SE_PCA×3, SE_PROTO, TC×2) are fixed by §9; experiment configs select which subset to load.

**Channel selection at training time**: Specified in the YAML config (see `configs/baseline.yaml:channels.extra`) as a list of `{name, band}` entries. `name` is an arbitrary label used in `normalization_stats.json` and logs; `band` is the 0-indexed position inside the EXTRA GeoTIFF (band positions per §9). Example loading the NDVI group only:
```yaml
channels:
  extra:
    - {name: ndvi, band: 0}
```
Changing which channels load at training time = edit the experiment YAML. The disk layout itself does not change; that's a v2.1 concern.

**Build order**: Generate planet_rgb first for positive and negative samples, then derive EXTRA by extracting auxiliary channels with the planet_rgb extent (footprint).

### 3.4 EXTRA Channel Processing

All auxiliary data must be:
1. Reprojected to EPSG:3857
2. Resampled to match PlanetScope nominal resolution (~4.77 m projected) using **bilinear interpolation** for all channels
3. Co-registered with RGB using GeoTIFF bounding box information
4. Stacked as channels in an order you keep stable across the dataset (that same order is what you reference by `band` index in the YAML config). §9 shows one example layout.

---

## 4. Data Values

**Both PLANET-RGB and EXTRA should store raw values**

Both PLANET-RGB and EXTRA store **raw values** (no normalization applied to stored files).

### 4.1 Normalization
Before computing statistics, apply percentage clipping to remove outliers. This is a **one-off step** during statistics computation (not applied per-image at load time):
- Run `scripts/check_data.py` first to visualise per-channel histograms and choose appropriate percentile bounds
- PlanetScope pre-processing may have already handled normalisation and outliers, the normalisation in this step is mostly for value alignment with the pretrained backbone, rather than improving image quality.
- Pass clipping percentiles as arguments to `scripts/compute_normalization_stats.py`
- Clipping percentile decision should be decided by looking at the histogram. to use histogram to decide the clipping optimal value: calculate a histogram with all available postive and negative tiles and save the figure and raw data for manual assessment. 
- The computed mean/std (on clipped data) are saved in `normalization_stats.json` — the DataLoader uses only those stored values

**Normalisation** Should be calculated per-dataset, rather than per-image, to:
- Consistent inference behavior regardless of batch composition
- Satellite imagery has consistent acquisition conditions within a sensor

Normalisation for EXTRA should be done channel-specific to respect the physical signal meanings

Use **per-dataset statistics** computed once over the entire training set. This preserves absolute radiometric information critical for distinguishing RTS features.

### 4.2 Statistics Computation

Compute mean and standard deviation for each channel across all training tiles:
- For RGB: compute over all training images (both positive and negative)
- For EXTRA: compute separately for each channel respecting physical meaning

### 4.3 Storage Specification

Store normalization statistics in a JSON file that travels with the model:

```
models/
├── experiment_name/
│   ├── normalization_stats.json
```

**normalization_stats.json structure**:

| Field | Description |
|-------|-------------|
| dataset_version | Version string from `data/version.json` (e.g. "2.0"). This file is created as part of the data pipeline and committed to the repo. |
| computed_date | ISO timestamp of computation |
| n_tiles_used | Number of tiles used in computation |
| rgb.channel_names | Fixed: `["R", "G", "B"]` |
| rgb.mean | List of 3 values, order matches `rgb.channel_names` |
| rgb.std | List of 3 values, order matches `rgb.channel_names` |
| extra.channel_names | List of N names declared in the experiment config's `channels.extra` (e.g. `["ndvi", "nbr", "se_pca_1", "se_pca_2", "se_pca_3", "se_proto", "tc_1", "tc_2"]` for the full v2.0 stack per §9). Omit the whole `extra` block when training RGB-only. |
| extra.mean | List of N values, order matches `extra.channel_names` |
| extra.std | List of N values, order matches `extra.channel_names` |

Note: the above extracts mean and std for z-score standardisation, can also get mins and maxs for 0-1 normalisation.

**compute**: Loading terabytes of GeoTIFFs to calculate mean/std can be challenging. Suggestion: Use Welford’s Online Algorithm to compute mean/variance in a single pass without loading all data

## 5. Imbalance and Split

| Estimation | Value |
|-----------|-------|
| Within Positive tiles | 5–70% of tile area |
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
|f2a (val-balanced) | 1:20 (physical pool size; subsampled to 1:1 at evaluation time) |
|f2b (val-realistic) | 1:200, 1:1000 | 
|f3 (test-realistic) | 1:200, 1:1000 |

## 6 Spatial Blocking
### 6.1 Purpose
Prevent geographic data leakage between train/val/test splits. RTS in nearby tiles may share visual characteristics due to:
- Similar geology and permafrost conditions
- Similar vegetation patterns
- Correlated image acquisition conditions

### 6.2 Blocking Strategy

Group tiles by Arctic subregion based on ecology/permafrost extent. Entire regions are assigned to train, val, or test — no region spans multiple splits.

**Region definitions**: Provided by Heidi Rodenhizer (boundaries complete). Storage format is geojson. see files in '/domain'.

**Implementation**: `scripts/create_splits.py` reads `metadata.csv` (which has `RegionName` per tile) and assigns each region to train/val/test according to the target ratios. The output `splits.yaml` is committed to the repo for reproducibility.

### 6.3 Tie-Break Rules for Region Assignment 

- Whole-region assignment and 80/10/10 tile-count targets cannot both be satisfied exactly when regions vary in size and RTS density. ‘scripts/create_splits.py’ resolves conflicts using the following priority order. Constraints earlier in the list take precedence; when one fails, the script exits with an error rather than silently producing a degenerate split.

1. Test set minimum positives — Test set must contain at least 100 positive tiles to enable statistically meaningful PR-AUC reporting at 1:1000 prevalence. If no whole-region assignment achieves this, fail loudly.
2. Validation set ecoregion diversity — Val set must span at least 2 distinct ecoregions, so that early-stopping decisions are not tied to a single regional artifact. If only one region can be assigned to val without breaking constraint 1, fail loudly.
3. Train set positive coverage — Train set should hold at least 70% of total positive tiles. Below this, emit a warning; do not fail.
4. Tile-count ratio drift tolerance — Once constraints 1–3 are satisfied, accept up to ±10% drift from the 80/10/10 target (train: 70–90% of total tiles). Beyond this, fail loudly.

- Geographic priority: assign the largest, most RTS-dense regions to train, the most diverse subset to val, and morphologically representative regions to test. The script logs its assignment reasoning.

Outputs:

- splits.yaml — region assignments per split (committed)
splits_summary.json — per-split tile counts, positive counts, region list, observed vs. target drift, and the constraint-resolution log (committed)

If the available regions cannot satisfy constraints 1–2, the project needs more labelled regions before splitting — not a softer split rule.

## 7. Negative Data Strategy

### 7.1 Sources

1. **ARTS confirmed negatives**: Known non-RTS locations
2. **Hard negatives**: False positive locations from Lingcao Huang's model (could be a source, but no plan for implement now)

Both are stored with `TrainClass = Negative` in `metadata.csv`. No separate class distinction is needed in code — treated identically during sampling.

### 7.2 Augmentation

Negative samples can be inflated on-the-fly through augmentation to achieve desired imbalance ratios. See Training Guide for augmentation strategy.

---

## 8. Data Check
Run before training:

- [ ] All positive tiles contain RTS pixels (label sum > 0)
- [ ] All negative tiles contain no RTS pixels (label sum = 0)
- [ ] RGB values are in valid range (0-255 for uint8)
- [ ] All tiles have matching image and label dimensions
- [ ] No NaN or infinite values in EXTRA channels
- [ ] GeoTIFF metadata (CRS, bounds) is consistent across tiles
- [ ] metadata.csv has entries for all tiles
- [ ] Spatial blocking is respected (no region spans train/val/test)

---

## 9. Channel Index Reference

RGB band order is fixed. EXTRA is the **canonical 8-band stack for v2.0** — one multi-band GeoTIFF in `EXTRA/{tile_id}.tif` with the band order below. Per-experiment configs select which subset to load via `configs/*.yaml:channels.extra` (a list of `{name, band}` entries; see §3.3). Adding a new group in v2.1 = appending bands to the EXTRA GeoTIFF and adding a row here. This section is the single source of truth for EXTRA channel composition; other docs cross-reference rather than enumerate.

### RGB (PLANET-RGB GeoTIFF, fixed)

| Band index | Channel |
|------------|---------|
| 0 | Red |
| 1 | Green |
| 2 | Blue |

### EXTRA (v2.0 stack — 8 bands; selection rationale and per-channel diagnostics in `./plots/extra_channel_vis/`)

Each row is one **group** — a semantic unit treated together by the Phase 4 ablation (see `training/experiments.md §7`). Single-band groups occupy one index; multi-band groups span a contiguous range. Group IDs are the names referenced from experiment configs and from Phase 4 results docs.

| Group ID | Band indices | N bands | Source | Description | Normalization treatment (intended; see note below) |
|----------|--------------|---------|--------|-------------|------|
| `NDVI` | 0 | 1 | Sentinel-2 | Normalized Difference Vegetation Index — vegetation density signal | Per-dataset z-score with `[0.1, 99.9]` percentile clip. Bounded but no load-bearing zero for a randomly-initialised first conv. Standard treatment. |
| `NBR` | 1 | 1 | Sentinel-2 | Normalized Burn Ratio — sensitive to bare/burned ground vs. healthy vegetation | Per-dataset z-score with `[0.1, 99.9]` percentile clip. Same family as NDVI; same logic. |
| `SE_PCA` | 2, 3, 4 | 3 | Google Satellite Embedding | First 3 Global PCA axes of the SE feature vector — generic pretrained representation | Per-dataset z-score, **per band independently**, with `[0.1, 99.9]` percentile clip. Global-PCA orthogonality isn't load-bearing for this task; the network re-learns its own use of PC1/2/3 from the Arctic subset distribution. |
| `SE_PROTO` | 5 | 1 | Google Satellite Embedding | Cosine similarity (contrast) to a hand-curated RTS prototype embedding | **No z-score.** Already bounded (cosine similarity ∈ [-1, 1]). Either pass raw, or apply fixed `(x − 0) / 0.5` to give a unit-scale signal without erasing the meaningful zero. Per-channel mode required (see note). |
| `TC` | 6, 7 | 2 | Sentinel-2 (Tasselled Cap transform) | TCB, TCW | Per-dataset z-score, **per band independently**, with `[0.1, 99.9]` percentile clip. Tasselled-cap components are radiometric quantities with no semantic zero; standard treatment. Two bands → two independent (μ, σ). |

Total EXTRA band count: **8**.

> **Implementation status (2026-05-02):** the per-band z-score column reflects what `data/normalization.py:WelfordStats` already computes (each EXTRA channel gets its own μ, σ — see [data/normalization.py:74-75](../data/normalization.py#L74-L75)). The `[0.1, 99.9]` percentile clipping and the `SE_PROTO` per-channel-mode bypass are **not yet implemented**; `scripts/compute_normalization_stats.py` runs Welford on raw values, and `RTSDataset.__getitem__` applies z-score uniformly to every channel. The treatments above are the **design intent** for when Phase 4 EXTRA experiments fire — see `training/experiments.md §7`. Implementing them requires (a) clipping in `compute_normalization_stats.py` and (b) a per-channel `normalization_mode` schema entry in `normalization_stats.json` plus a dispatch in the loader. Both are deferred to v2.1.

Channels considered but **not** included in v2.0: NIR (Sentinel-2), Relative Elevation / Shaded Relief / slope / aspect (ArcticDEM), NDMI, SAR backscatter. These remain available for v2.1 if Phase 4 results or post-inference analysis flags a gap that the included groups cannot fill.

### Label File (labels GeoTIFF, fixed)

| Value | Meaning |
|-------|---------|
| 0 | Background |
| 1 | RTS |
| 255 | Ignore |