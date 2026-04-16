# SE + Extended SAR Channel Visualization

## Purpose

Two additional multi-subplot figures, extending the existing channel visualization work:

1. **Satellite Embeddings (SE) feasibility** — evaluate whether AlphaEarth Foundations embeddings can serve as a coarse sifter to eliminate impossible regions for RTS. Three complementary diagnostics in one figure: per-tile PCA (does SE locally represent RTS?), global PCA (could an unsupervised PCA-based sifter work?), and prototype cosine similarity (could a minimal supervised sifter work?).

2. **Extended SAR** — plot every SAR-derived channel that might carry RTS signal, using 2024 data only (no cross-year dependency).

Both plots follow the conventions of `extra_channel_plot.py`: PlanetScope RGB reference, RTS polygon overlaid on every spatial subplot, consistent styling.

## Environment

- **Conda env**: `rts_dataset`
- **Required packages**: earthengine-api, numpy, matplotlib, Pillow, scikit-learn, rasterio, google-cloud-storage (for reading training labels from GCS)
- **GEE auth**: must be authenticated (`earthengine authenticate`) before running
- **GCS auth**: service account or ADC configured for reading from the training bucket (for prototype construction)

## Usage

```bash
conda run -n rts_dataset python plots/extra_channel_vis/se_sar_plot.py
```

## Input Data

- **PlanetScope tile**: seven tifs in \plots
- **RTS polygon**: the geojson in \plots
- **mapping**: the csv in \plots maps the polygon with the tile

Reuse polygon-loading and overlay helpers from `extra_channel_plot.py` verbatim.

---

## Plot 1: SE Feasibility — PCA (per-tile and global) + Prototype Cosine Similarity

### Why three diagnostics, not one

Each answers a different question. Running them together gives a decision matrix rather than a single ambiguous signal.

| Diagnostic | Question answered | Method style |
|---|---|---|
| Per-tile PCA | Does SE locally represent RTS as a distinct feature within this tile? | Unsupervised, variance-driven |
| Global PCA | Would an unsupervised PCA-based sifter running on pan-Arctic SE carry RTS signal in its top components? | Unsupervised, variance-driven |
| Prototype cosine | Would the intended sifter mechanism (cosine to a prototype built from labeled RTS) actually discriminate RTS from background here? | Supervised, mean-based |

Per-tile PCA is the cheapest test — failure here means SE doesn't represent RTS at all. Global PCA is the next step up — failure here with per-tile success means the RTS direction exists but is low-variance globally. Prototype cosine is the strongest test — it directly evaluates the mechanism you would actually deploy.

### GEE source (common to all three diagnostics)

- Collection: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- 64 bands, annual composite, 10 m native resolution
- Model version v2.1
- **Embeddings are unit-length** (64-D unit sphere). This matters — see "On the geometry" below.
- Year: 2024
- Resampling to PlanetScope grid: nearest-neighbour (bilinear smears 10 m semantics)

### Diagnostic A: Per-tile PCA

1. Download SE for the tile region: shape `(64, 512, 512)`.
2. Reshape to `(n_pixels, 64)`, fit `sklearn.PCA(n_components=3)` on these pixels.
3. Project and reshape to `(3, 512, 512)`. Per-component 2–98% stretch to `[0, 1]` → PCA-RGB.
4. Record explained variance ratios.

### Diagnostic B: Global PCA

1. Define AOI: 60–74°N land-only bounding box, EPSG:4326. Use any reasonable global land mask (LSIB, MODIS land cover, SRTM extent); document the choice. **Do not skip masking** — ocean embeddings would dominate the fit.
2. Sample via `ee.Image.sample(region=aoi, numPixels=10000, scale=10, seed=42)`, extract to `(10000, 64)` numpy array.
3. Fit `sklearn.PCA(n_components=3)` on the sample.
4. Project the tile's pixels onto these axes using `global_pca.transform()`. **Must center using `global_pca.mean_`** — sklearn handles this automatically in `.transform()` but if implementing manually, easy to forget.
5. Reshape to `(3, 512, 512)`, apply per-component 2–98% stretch → PCA-RGB.

Caveat to note in the figure caption: the 60–74°N land bounding box approximates the true inference domain (Arctic boreal ∩ permafrost ∩ Planet coverage per `domain/inference_domain.md`). Over-samples temperate-edge latitudes; under-weights the northernmost permafrost zones. Adequate for feasibility, not production calibration.

### Diagnostic C: Prototype Cosine Similarity

Mechanism: build an "RTS prototype vector" by averaging the SE embeddings of labeled RTS pixels from other tiles, then score every pixel in the target tile by cosine similarity to this prototype.

Steps:

1. From `metadata.csv`, select a random sample of 30 positive tiles **excluding** the current tile. Seeded for reproducibility. Spread across regions if `metadata.csv` has RegionName populated; otherwise random.
2. For each sampled tile:
   - Load the label raster from GCS, find RTS pixels (value = 1).
   - Get the pixel coordinates and convert to EPSG:4326 for GEE sampling.
   - Pull the 2024 SE embeddings at those pixel locations (`ee.Image.sampleRegions` or equivalent).
3. Concatenate all labeled RTS embeddings across tiles: shape `(n_total_rts_pixels, 64)`. Target ~50,000 labeled pixels; if fewer, use what's available and log the count.
4. Compute prototype: mean across pixels → unit-normalize back to the sphere. **Re-normalization is critical** — averaging moves the vector off the unit sphere, and without renormalization the cosine similarity scale becomes meaningless.
5. For the target tile: compute dot product of each pixel's 64-D embedding with the prototype vector. Since both are unit-length, dot product = cosine similarity ∈ [−1, 1]. Reshape to `(512, 512)`.
6. Also compute the similarity distribution histogram across all pixels in the tile (for threshold intuition).

### On the geometry (applies to all three)

SE embeddings are constrained to the unit sphere in 64-D. This is why cosine similarity (or equivalently, dot product on unit vectors) is the *designed* metric — the model was trained such that all semantic information lives in direction, not magnitude.

- **For prototype similarity:** this is perfect. Cosine similarity is mathematically and semantically the right operation. No geometric compromise.
- **For PCA:** this is an imperfect match. PCA treats the data as Euclidean; projecting unit-sphere data into 3-D via PCA produces 3-D Euclidean points that are no longer on any sphere. Distances in the PCA projection approximate — but don't exactly preserve — angular distances in the original 64-D space. Acceptable for visualization; not ideal for quantitative downstream use. If serious downstream work is needed, spherical PCA or tangent-space projection would be more principled.

### Subplot layout

4x3 grid:

| | | |
|---|---|---|
| RGB (PlanetScope) | Per-tile PCA-RGB | Global PCA-RGB |
| Per-tile PC1 | Per-tile PC2 | Per-tile PC3 |
| Global PC1 | Global PC2 | Global PC3 |
| Prototype cosine similarity | Prototype at threshold T | Similarity histogram |

Styling:

- Individual PC panels: `gray` or `viridis`, per-panel 2–98% stretch, colorbar.
- PCA-RGB panels: include explained variance ratios in the title, e.g. `"Per-tile PCA-RGB (0.52, 0.18, 0.09)"`.
- Prototype similarity map: `RdBu_r` diverging colormap centered at 0, vmin = −1, vmax = 1, colorbar labeled "cos similarity".
- Prototype at threshold T: binary mask. Choose T as the 95th percentile of cosine similarity across the *prototype-construction pixels* (not the target tile) and note the value in the panel title. This approximates "pixels that are at least as RTS-like as the typical labeled RTS pixel."
- Similarity histogram: log-scale y-axis; overlay a vertical line at the chosen threshold T; annotate the fraction of tile pixels above T and the cosine similarity at the known polygon centroid.
- Polygon overlay on all spatial panels. No polygon overlay on histogram.
- Top of figure: brief text noting global sample size, prototype construction size (n tiles, n pixels), year.

### What to look for (reading the plot)

**PCA panels (rows 1–3):**

- Per-tile PCA-RGB shows polygon distinct from background → SE has locally usable RTS information.
- Global PCA-RGB shows polygon distinct → unsupervised PCA sifter is feasible (rare, best case).
- Per-tile succeeds but global fails → PCA is the wrong global projection; supervised alternatives indicated. This is the most likely outcome.
- Individual PC panels identify which components carry the signal. If per-tile PC1 shows large-scale gradient (e.g. water-land) while PC2 or PC3 shows the polygon, that tells you PCA is picking up the RTS direction but not as the dominant axis.

**Prototype panels (row 4):**

- Similarity map shows polygon as a coherent bright region (high similarity) relative to surrounding dark (low similarity) → prototype sifter works on this tile.
- Thresholded mask should roughly overlap the polygon, with tolerable false positives elsewhere (remember: sifter, not detector — overcalls are cheap, missed calls are expensive).
- Histogram shows clear separation between "most of the tile" mass and a tail containing the polygon pixels → the threshold choice is meaningful. If the polygon's mean similarity is in the middle of the main distribution mass, the prototype cannot separate RTS from background here.

**Decision logic:**

| Per-tile PCA | Global PCA | Prototype | Interpretation |
|---|---|---|---|
| Fail | Fail | Fail | SE doesn't represent RTS. Abandon. |
| Pass | Fail | Fail | SE has *some* local structure but it doesn't align with RTS semantically. Abandon. |
| Pass | Fail | Pass | Expected case. SE works as a supervised sifter; PCA is the wrong projection. Proceed to build the prototype sifter. |
| Pass | Pass | Pass | Best case. Even unsupervised PCA could serve. |

### Caveats specific to the prototype test

- A prototype built from 30 tiles is a **sample-size-limited** test. If those 30 tiles happen to come from one region and the target polygon is in a different region, the prototype may not generalize — failure here doesn't necessarily mean SE is unusable, only that this particular prototype underfits the diversity of RTS. Document region distribution of sampled tiles in the figure caption.
- The prototype is the **mean** of positive embeddings — it doesn't know about negatives. If RTS embeddings cluster near other bare-ground classes (gravel, river bars, fire scars), the prototype will score those high too. This is a known limitation; if false positives dominate, logistic regression (with labeled negatives) is the next step. Not implemented here.
- Prototype similarity tests the *mechanism*. If it succeeds, the actual deployment still needs pan-Arctic threshold calibration, which is a separate exercise.

Output: `plots/extra_channel_vis/se_feasibility.png`

---

## Plot 2: Extended SAR Channels (2024 only)

### GEE source

- `COPERNICUS/S1_GRD`: VV + VH, 10 m, 2024 Jul–Sep acquisitions.
- Filter: IW mode, VV+VH polarization. Document ascending/descending choice at tile location; avoid mixing orbit directions without checking for geometric artefacts.
- Log S1 acquisition counts per aggregation window (full Jul–Sep, Jul-only, Aug–Sep-only). A coverage gap in any window corrupts the corresponding panel — surface this, don't hide it.
- **Aggregate in linear space, convert to dB last**: `dB = 10 * log10(linear_mean)`. Mean of dB is wrong.

**HH/HV polarization**: S1 IW over Arctic land is VV+VH only. HH/HV in IW is rare and location-restricted. Check once; if absent, note in caption, do not create empty panels.

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
| VV − VH | `VV_dB − VH_dB` | Polarization difference; high = surface-dominated (bare), low = volume-dominated |
| RVI | `4 · VH_lin / (VV_lin + VH_lin)`, linear space, 0–1 | Radar Vegetation Index; low = less volume scattering = bare |
| Dual-pol composite | R=VV_dB, G=VH_dB, B=(VV−VH)_dB, each stretched to [0,1] | Dual-pol overview |
| VV temporal std | Pixel-wise std of VV scenes in Jul–Sep 2024, on dB | Within-season VV variability; surface dynamism |
| VH temporal std | Pixel-wise std of VH scenes in Jul–Sep 2024, on dB | Within-season VH variability; canopy/volume dynamism |
| Mid-season VV shift | `VV_dB(Aug–Sep) − VV_dB(Jul)`, each linear-mean first | Direction of within-season change |

### Visualization notes

- dB panels: `viridis` or `gray`, 2–98% of data range (not fixed), colorbars in dB.
- VV − VH: `viridis`, percentile-based stretch.
- RVI: `viridis`, vmin=0, vmax=1.
- Dual-pol composite: RGB, text annotation for channel assignment.
- Temporal std: `viridis`, percentile stretch, dB units.
- Mid-season VV shift: `RdBu_r`, centered at 0, symmetric vmin/vmax from 2–98% of absolute values.
- Polygon overlay everywhere.

### Why 2024-only (no year-over-year change)

Year-over-year change (dVV_2024-2023) was considered and dropped:

- Labels describe **state**, not change. State labels → state inputs.
- YoY change is silent on stable-active RTS (active both years → ~zero difference). Most labeled RTS are multi-year.
- Operational cost (2023 coverage, co-registration, gap handling) without matching benefit.

Within-season signals (temporal std, mid-season shift) capture dynamism using 2024 data only.

### Why no InSAR coherence

The most RTS-specific SAR channel, but requires SNAP/HyP3 SLC processing — not a ready GEE product. `VV/VH temporal std` are the cheapest on-GEE proxies. If either shows clear polygon contrast, coherence is worth the separate infrastructure investment.

Output: `plots/extra_channel_vis/sar_extended.png`

---

## Implementation conventions

- Reuse polygon loading, overlay styling, figure-saving patterns from `extra_channel_plot.py`.
- Cache GEE downloads to `plots/extra_channel_vis/.cache/` (gitignored). Cache key: AOI bbox + year window + product + (SE) model version + (global PCA) sample size and seed + (prototype) list of contributing tile IDs hashed. `--no-cache` flag forces refresh.
- Cache prototype embeddings separately from per-tile and global fits — prototype construction is the most expensive step (30 tiles × many RTS pixels × GEE round-trips) and should only rerun when the tile sample or year changes.
- Surface GEE / GCS failures with clear error messages naming the product or file. No silent empty panels.
- Figure DPI ≥ 150, tight layout. Shared polygon overlay colour with existing script.
- Top-of-file docstring: purpose, plot contents, GEE products, GCS dependencies, expected runtime, and that two figures are produced.

## Uncertainties flagged for the implementer

- **AlphaEarth band names**: verify at runtime (`A00`–`A63` historically; collection ID `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` confirmed).
- **AlphaEarth de-quantization**: via GEE, embeddings are served as 64-bit floats (already dequantized). If ever accessing via the GCS COG route, quantization handling is needed. Stay in GEE for simplicity.
- **Global PCA sampling**: 10K pixels in one `ee.Image.sample` call should complete; if it times out, reduce to 5K or split by longitude slice. Do not drop below 1K — too few for stable 64-D fit.
- **Prototype sample size**: target 30 tiles. If fewer positive tiles are available at script time, use what exists and log the count. Fewer than 10 tiles is too few — warn and consider skipping the prototype panel.
- **Prototype region distribution**: if `metadata.csv` has RegionName populated, stratify the 30-tile sample across regions. If not, random sample with fixed seed.
- **S1 ascending vs descending**: document which was used at the tile location.
- **dB conversion ordering**: aggregate in linear, dB as the final step. Applies to seasonal mean, mid-season windows, RVI computation (RVI in linear).
- **Polygon in EPSG:3857**: same as existing script. Do not reproject.