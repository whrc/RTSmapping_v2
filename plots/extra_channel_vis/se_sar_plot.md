# SE + Extended SAR Channel Visualization

## Purpose

Two additional multi-subplot figures, extending the existing channel visualization work:

1. **Satellite Embeddings (SE) feasibility** — evaluate whether AlphaEarth Foundations embeddings can serve as a coarse sifter to eliminate impossible regions for RTS. Three complementary diagnostics in one figure: per-tile PCA (does SE locally represent RTS?), global PCA (could an unsupervised PCA-based sifter work?), and prototype cosine similarity (could a minimal supervised sifter work?).

2. **Extended SAR** — plot every SAR-derived channel that might carry RTS signal, using 2024 data only (no cross-year dependency).

Both plots follow the conventions of `extra_channel_plot.py`: PlanetScope RGB reference, RTS polygon overlaid on every spatial subplot, consistent styling.

## Environment

- **Conda env**: `rts_dataset`
- **Required packages**: earthengine-api, numpy, matplotlib, Pillow, scikit-learn, rasterio, google-cloud-storage
- **GEE auth**: must be authenticated (`earthengine authenticate`) before running
- **GCS auth**: service account or ADC configured for reading from `gs://abrupt_thaw/`

## Usage

```bash
conda run -n rts_dataset python plots/extra_channel_vis/se_sar_plot.py
```

## Input Data

- **PlanetScope tile**: seven tifs in \plots
- **RTS polygon**: the geotiff in \plots
- **correspondence**: the csv in \plots maps polygon to the tile
- **Training labels**: `gs://abrupt_thaw/RTS_MODEL_V2/DATA/labels/{tile_id}.tif` — uint8, values {0, 1, 255}

Reuse polygon-loading and overlay helpers from `extra_channel_plot.py` verbatim.

---

## Plot 1: SE Feasibility — PCA (per-tile and global) + Prototype Cosine Similarity

### Why three diagnostics, not one

Each answers a different question. Running them together gives a decision matrix rather than a single ambiguous signal.

| Diagnostic | Question answered | Method |
|---|---|---|
| Per-tile PCA | Does SE locally represent RTS as a distinct feature within this tile? | Unsupervised, variance-driven |
| Global PCA | Would an unsupervised PCA-based sifter carry RTS signal in its top components? | Unsupervised, variance-driven |
| Prototype cosine | Would the intended sifter mechanism (cosine to a labeled-RTS prototype) discriminate RTS from background? | Supervised, mean-based |

Per-tile PCA is the cheapest failure detector — if even within-tile variance doesn't surface RTS, skip the rest. Global PCA tests unsupervised viability. Prototype cosine tests the actual deployment mechanism.

### GEE source (common to all three)

- Collection: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- 64 bands, annual composite, 10 m native resolution, model v2.1
- **Embeddings are unit-length** (64-D unit sphere) — cosine similarity = dot product
- Year: 2024
- Resampling to PlanetScope grid: **nearest-neighbour** (bilinear smears 10 m semantics)

### Diagnostic A: Per-tile PCA

1. Download SE for the tile region: shape `(64, 512, 512)`.
2. Reshape to `(n_pixels, 64)`, fit `sklearn.PCA(n_components=3)`.
3. Project and reshape to `(3, 512, 512)`. Per-component 2–98% stretch to `[0, 1]` → PCA-RGB.
4. Record explained variance ratios.

### Diagnostic B: Global PCA

1. AOI: 60–74°N land-only bounding box, EPSG:4326. Apply a land mask (LSIB, MODIS land cover, or SRTM extent — document the choice). **Do not skip masking** — ocean embeddings dominate the fit.
2. Sample: `ee.Image.sample(region=aoi, numPixels=10000, scale=10, seed=42)` → `(10000, 64)` numpy array.
3. Fit `sklearn.PCA(n_components=3)` on the sample.
4. Project the tile's pixels via `global_pca.transform()`. Reshape to `(3, 512, 512)`, per-component 2–98% stretch → PCA-RGB.

Caveat for caption: the 60–74°N bounding box approximates the true inference domain (Arctic boreal ∩ permafrost ∩ Planet coverage per `domain/inference_domain.md`). Adequate for feasibility, not production calibration.

### Diagnostic C: Prototype Cosine Similarity

**What this does:** builds a single "RTS prototype vector" by averaging the SE embeddings of all labeled RTS pixels across the training dataset, then scores every pixel in the target tile by cosine similarity to this prototype. High similarity = "this pixel lives in a similar part of embedding space as known RTS." Low similarity = "this pixel is unlike any known RTS."

**Construction steps:**

1. Load tiffs from the GCS. **Exclude** the current 7 test tiles so the evaluation isn't circular (find spatial overlap).
2. For each tile:
   - Load its label raster from `gs://abrupt_thaw/RTS_MODEL_V2/DATA/labels/{tile_id}.tif`.
   - Extract coordinates of all pixels where label == 1 (RTS pixels). Convert pixel coordinates to geographic coordinates (EPSG:3857 → EPSG:4326) using the tile's geotransform.
3. Merge all RTS pixel coordinates across all tiles into a single GEE FeatureCollection.
   - **If total RTS pixels exceed ~2 million**: randomly subsample to 2M points (seeded, seed=42). GEE's `sampleRegions` has practical limits on FeatureCollection size; 2M is conservative. Log the subsample fraction.
   - **If total RTS pixels are under 2 million**: use all of them.
4. Call `ee.Image.sampleRegions(collection=rts_points, scale=10)` on the 2024 SE image. **One GEE call**, not per-tile calls.
5. Export the resulting `(n_sampled_pixels, 64)` array.
6. Compute prototype: `mean(axis=0)` → **re-normalize to unit length**. Re-normalization is critical — averaging moves the vector off the unit sphere; without it, cosine similarity scale becomes meaningless.
7. Cache the prototype vector and metadata (n_tiles_used, n_pixels_sampled, n_pixels_total, region distribution, date) to `plots/extra_channel_vis/.cache/prototype.npz`.

**Scoring the target tile:**

1. Load the tile's SE embeddings (already downloaded for Diagnostic A): `(n_pixels, 64)`.
2. Dot product with prototype vector → `(n_pixels,)` cosine similarity scores ∈ [−1, 1].
3. Reshape to `(512, 512)`.

**Threshold selection:**

From the prototype construction step, you also have the cosine similarity of every sampled RTS pixel to the prototype (each pixel's dot product with the final prototype). Take the 5th percentile of those values as the threshold T — this means "at least 95% of known RTS pixels pass this cutoff." Note T in the panel title.

### On the geometry

SE embeddings are constrained to the unit sphere. All semantic information lives in direction, not magnitude. This is why cosine similarity (= dot product on unit vectors) is the designed metric.

- **For prototype cosine:** geometrically perfect. No compromise.
- **For PCA:** imperfect. PCA treats data as Euclidean; projecting unit-sphere data into 3-D breaks the sphere structure. Acceptable for visualization, not for quantitative downstream use. Spherical PCA or tangent-space projection would be more principled for serious work.

### Subplot layout

4x3 grid:

| | | |
|---|---|---|
| RGB (PlanetScope) | Per-tile PCA-RGB | Global PCA-RGB |
| Per-tile PC1 | Per-tile PC2 | Per-tile PC3 |
| Global PC1 | Global PC2 | Global PC3 |
| Prototype cosine similarity | Prototype at threshold T | Similarity histogram |

Styling:

- PC panels: `gray` or `viridis`, per-panel 2–98% stretch, colorbar.
- PCA-RGB panels: explained variance ratios in title, e.g. `"Per-tile PCA-RGB (0.52, 0.18, 0.09)"`.
- Prototype similarity map: `RdBu_r` diverging, centered at 0, vmin=−1, vmax=1, colorbar labeled "cos similarity".
- Prototype at threshold T: binary mask. Threshold T = 5th percentile of labeled-RTS cosine similarities (noted in panel title). This approximates "pixels at least as RTS-like as 95% of known RTS pixels."
- Similarity histogram: log-scale y-axis; vertical line at threshold T; annotate fraction of tile pixels above T and the mean cosine similarity of pixels within the polygon.
- Polygon overlay on all spatial panels. No overlay on histogram.
- Top of figure: text noting global sample size (10K), prototype stats (n_tiles, n_pixels, region count), year 2024.

### What to look for

**PCA panels (rows 1–3):**

- Per-tile PCA-RGB shows polygon as distinct → SE has locally usable RTS information.
- Global PCA-RGB shows polygon as distinct → unsupervised PCA sifter feasible (rare, best case).
- Per-tile succeeds, global fails → expected outcome. PCA is the wrong global projection; supervised alternative indicated.

**Prototype panels (row 4):**

- Similarity map shows polygon as a coherent bright region relative to dark surroundings → prototype sifter works on this tile.
- Thresholded mask roughly overlaps the polygon, with tolerable false positives. Sifter overcalls are cheap; missed calls are expensive.
- Histogram: clear separation between main mass and a tail containing polygon pixels → threshold choice is meaningful. If the polygon's mean similarity sits in the middle of the main distribution, the prototype cannot separate RTS from background here.

**Decision matrix:**

| Per-tile PCA | Global PCA | Prototype | Interpretation |
|---|---|---|---|
| Fail | Fail | Fail | SE doesn't represent RTS. Abandon. |
| Pass | Fail | Fail | SE has local structure but it doesn't align with RTS semantics. Abandon. |
| Pass | Fail | Pass | **Expected case.** SE works as a supervised sifter; PCA is the wrong projection. Proceed to build the prototype sifter at scale. |
| Pass | Pass | Pass | Best case. Even unsupervised PCA could serve. |

### Known limitations of the prototype test

- The prototype is a **mean of positives** — it doesn't know about negatives. If RTS embeddings cluster near other bare-ground classes (gravel bars, fire scars, exposed lakeshores), the prototype will score those high too. Acceptable for a sifter (overcall > missed call), but if false positives dominate, logistic regression with labeled negatives is the next step. Not implemented here.
- Prototype quality depends on **label quality and geographic diversity** of the training set. With ~1,800 positive tiles across multiple regions, the prototype should be stable. The script should log the region distribution of contributing tiles in the cache metadata — if tiles are concentrated in a few regions, note this as a coverage caveat.

Output: `plots/extra_channel_vis/se_feasibility.png`

---

## Plot 2: Extended SAR Channels (2024 only)

### GEE source

- `COPERNICUS/S1_GRD`: VV + VH, 10 m, 2024 Jul–Sep acquisitions.
- Filter: IW mode, VV+VH polarization. Document ascending/descending choice at tile location; avoid mixing orbit directions without checking for geometric artefacts.
- Log S1 acquisition counts per aggregation window (full Jul–Sep, Jul-only, Aug–Sep-only). Coverage gaps corrupt the corresponding panel — surface this, don't hide it.
- **Aggregate in linear space, convert to dB last**: `dB = 10 * log10(linear_mean)`.

**HH/HV polarization**: S1 IW over Arctic land is VV+VH only. Check once; if absent, note in caption, no empty panels.

### Subplot layout

3x3 grid:

| | | |
|---|---|---|
| RGB (PlanetScope) | VV (2024 Jul–Sep mean, dB) | VH (2024 Jul–Sep mean, dB) |
| VV − VH (dB) | RVI | Dual-pol composite |
| VV temporal std | VH temporal std | Mid-season VV shift (Aug–Sep − Jul) |

### Channel definitions

| Channel | Computation | Interpretation |
|---|---|---|
| VV | VV, 2024 Jul–Sep linear mean → dB | Surface roughness + moisture; wet bare ground bright |
| VH | VH, 2024 Jul–Sep linear mean → dB | Volume scattering; vegetation canopy bright, bare ground dark |
| VV − VH | `VV_dB − VH_dB` | Polarization difference; high = bare, low = vegetated |
| RVI | `4 · VH_lin / (VV_lin + VH_lin)`, linear space, 0–1 | Radar Vegetation Index; low = bare |
| Dual-pol composite | R=VV_dB, G=VH_dB, B=(VV−VH)_dB, each stretched to [0,1] | Dual-pol overview |
| VV temporal std | Pixel-wise std of VV scenes in Jul–Sep 2024, on dB | Within-season surface dynamism |
| VH temporal std | Pixel-wise std of VH scenes in Jul–Sep 2024, on dB | Within-season canopy/volume dynamism |
| Mid-season VV shift | `VV_dB(Aug–Sep) − VV_dB(Jul)`, each linear-mean first | Direction of within-season change |

### Visualization notes

- dB panels: `viridis` or `gray`, 2–98% of data range, colorbars in dB.
- VV − VH: `viridis`, percentile-based stretch.
- RVI: `viridis`, vmin=0, vmax=1.
- Dual-pol composite: RGB, text annotation for channel assignment.
- Temporal std: `viridis`, percentile stretch, dB units.
- Mid-season VV shift: `RdBu_r`, centered at 0, symmetric vmin/vmax from 2–98% of absolute values.
- Polygon overlay everywhere.

### Why 2024-only

- Labels describe **state**, not change. State labels → state inputs.
- Year-over-year change is silent on **stable-active** RTS (active both years → ~zero difference).
- Operational cost (2023 coverage, co-registration, gap handling) without matching benefit.

Within-season signals (temporal std, mid-season shift) capture dynamism using 2024 data only.

### Why no InSAR coherence

The most RTS-specific SAR channel, but requires SNAP/HyP3 SLC processing — not a ready GEE product. `VV/VH temporal std` are the cheapest on-GEE proxies. If either shows polygon contrast, coherence is worth the separate infrastructure investment.

Output: `plots/extra_channel_vis/sar_extended.png`

---

## Implementation conventions

- Reuse polygon loading, overlay styling, figure-saving patterns from `extra_channel_plot.py`.
- Cache everything to `plots/extra_channel_vis/.cache/` (gitignored):
  - Per-tile SE embeddings (keyed by tile bbox + year)
  - Global PCA sample (keyed by AOI + sample size + seed + year)
  - Prototype vector + metadata (keyed by hash of contributing tile IDs + year). **Cache prototype separately** — it's the most expensive to build (all positive tiles × GEE round-trip) and should only recompute when labels change.
  - SAR channels (keyed by tile bbox + year + date window)
- `--no-cache` flag forces full refresh.
- Surface GEE / GCS failures with clear error messages naming the product or path. No silent empty panels.
- Figure DPI ≥ 150, tight layout. Shared polygon overlay colour with existing script.
- Top-of-file docstring: purpose, plot contents, GEE products, GCS paths, expected runtime, two output figures.

## Uncertainties flagged for the implementer

- **AlphaEarth band names**: verify at runtime. Collection ID `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` confirmed.
- **GEE FeatureCollection size limit**: `sampleRegions` may fail if the FeatureCollection exceeds ~5M features. If total RTS pixels > 2M, subsample before uploading. If the 2M call still fails, reduce to 1M. Log actual sample size.
- **GCS label path**: `gs://abrupt_thaw/RTS_MODEL_V2/DATA/labels/{tile_id}.tif`. Tile ID format should match `metadata.csv` `Tile_id` column (zero-padded numeric). Verify one tile loads correctly before iterating over all.
- **metadata.csv location**: `gs://abrupt_thaw/RTS_MODEL_V2/DATA/metadata.csv`. Columns per `data/data.md`: `Tile_id`, `centroid_lat`, `centroid_lon`, `TrainClass`, `RegionName`, `UIDs`.
- **Prototype region distribution**: log the count of tiles per RegionName used in prototype construction. If one region dominates (>50% of tiles), note in cache metadata and figure caption.
- **S1 ascending vs descending**: document which was used at tile location.
- **dB conversion ordering**: aggregate in linear, dB last. Applies to all aggregations. RVI computed in linear.
- **Polygon in EPSG:3857**: same as existing script. Do not reproject.