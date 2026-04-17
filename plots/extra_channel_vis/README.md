# Extra Channel Visualization — Master Doc

_Last updated: 2026-04-17_

Living single-source-of-truth for the extra-channel visualization work: auxiliary channels beyond RGB that might carry RTS-segmentation signal. Consolidates spec, methods, results, and decisions for DEM-derived, Sentinel-derived, and SE-derived channels.

---

## Current Recommendations

- **SE as auxiliary channel**: use **Approach 2 — Contrastive**. It passes both C1 (|r| ≤ 0.5 vs every Sentinel auxiliary) and C2 (no sign inversions across 7 test tiles) and produces visible polygon-vs-background separation. Held-out C1 on the full 150-tile set is still pending (see Open Items).
- **15-channel correlation**: SE-cosine is non-redundant globally. NDVI↔NDWI and SWIR↔TCW are near-duplicates in the positive regime — don't train with both of each pair. NBR and NDMI cluster at distance 0.60 from SE-cosine (above the 0.5 redundancy threshold).
- **SAR (Sentinel-1)**: 2024-only, Jul–Sep aggregation in linear space then dB. Within-season temporal std and mid-season shift are the cheapest dynamism proxies; InSAR coherence deferred.
- **ArcticDEM**: not adopted into the SE correlation study but still a candidate channel group. Plots kept for terrain-context reference.

---

## Folder Layout

```
plots/extra_channel_vis/
├── README.md                             ← this doc
├── extra_channel_plot.py                 ← DEM + Sentinel-2 per-tile figures
├── se_sar_plot.py                        ← SE feasibility + extended SAR figures
├── inputs/
│   ├── polygons.geojson                  ← RTS polygons for 7 test OIDs
│   ├── polygon_image_mapping.csv
│   └── tiles/tile_*.tif                  ← PlanetScope RGB tiles
├── dem_derived/
│   └── oid{OID}_arcticdem.png            ← 7 files, ArcticDEM derivatives
├── sentinel_derived/
│   ├── sentinel2/oid{OID}_sentinel.png   ← 7 files, S-2 indices + tasseled cap
│   └── sar_extended/oid{OID}_sar_extended.png   ← 7 files, S-1 diagnostics
├── se_derived/
│   ├── feasibility/oid{OID}_se_feasibility.png  ← 7 files (historical baseline)
│   ├── correlation/                      ← 15-channel Spearman heatmaps, dendrogram
│   └── variants/
│       ├── approach1_multiprototype/
│       ├── approach2_contrastive/
│       └── comparison/                   ← three-way plots
└── .cache/                               ← GEE / prototype caches
```

7 test OIDs: 93, 113, 136, 144, 169, 187, 262.

---

## Environment

- **Conda env**: `rts_dataset`
- **Packages**: earthengine-api, numpy, matplotlib, rasterio, scikit-learn, scipy, seaborn, google-cloud-storage, Pillow, pyyaml, tqdm
- **GEE auth**: `earthengine authenticate` (project `pdg-project-406720`)
- **GCS auth**: ADC or service account with read access to `gs://abrupt_thaw/`

---

## Data Sources Overview

| Source | Goal | Status | Key outputs |
|---|---|---|---|
| ArcticDEM v4.1 | Terrain derivatives per tile | done | `dem_derived/` |
| Sentinel-2 SR Harmonized (2024 Jul–Sep median) | Spectral + tasseled cap | done | `sentinel_derived/sentinel2/` |
| Sentinel-1 GRD (2024 Jul–Sep) | Extended SAR diagnostics | done | `sentinel_derived/sar_extended/` |
| AlphaEarth SE V1 Annual (2024) — feasibility | PCA + single-prototype cosine baseline | superseded | `se_derived/feasibility/` |
| SE 15-channel correlation | Spearman on 150 held-out tiles | done | `se_derived/correlation/` |
| SE variants (multi-prototype, contrastive, comparison) | Fix failure modes of single-prototype | done; comparison planned | `se_derived/variants/` |

---

## DEM-derived Channels

Source: `UMN/PGC/ArcticDEM/V4/2m_mosaic`. Figure is a 3×3 grid: RGB, DEM (raw), Relative Elevation, Hillshade, Curvature, Shaded Relief, Slope, Aspect, TPI.

| Channel | Computation | Notes |
|---|---|---|
| DEM | ArcticDEM elevation band | meters |
| Relative Elevation | `DEM − focalMean(radius=500m)` | positive = locally high |
| Hillshade | `ee.Terrain.hillshade(dem)` | simulated illumination |
| Curvature | `dem.convolve(Laplacian8 kernel)` | concave (+) vs convex (−) |
| Shaded Relief | `0.7·hillshade_norm + 0.3·curvature_norm` | blended |
| Slope | `ee.Terrain.slope(dem)` | degrees |
| Aspect | `ee.Terrain.aspect(dem)` | degrees, 0–360 |
| TPI | `DEM − focalMean(radius=300m)` | Topographic Position Index |

---

## Sentinel-derived Channels

### Sentinel-2 (`COPERNICUS/S2_SR_HARMONIZED`, 2024 Jul–Sep median)

2×5 figure: RGB, NIR, SWIR, NBR, NDVI, NDWI, NDMI, TCB, TCW, SAR (VV).

| Channel | Formula | Notes |
|---|---|---|
| NIR | B8 | vegetation reflectance |
| SWIR | B11 | moisture / minerals |
| NBR | (B8 − B12) / (B8 + B12) | Normalized Burn Ratio; bare soil |
| NDVI | (B8 − B4) / (B8 + B4) | vegetation greenness |
| NDWI | (B3 − B8) / (B3 + B8) | water content |
| NDMI | (B8 − B11) / (B8 + B11) | moisture |
| TCB | Tasseled Cap Brightness (Shi & Xu 2019) | bare soil exposure |
| TCW | Tasseled Cap Wetness (Shi & Xu 2019) | soil moisture |

### Sentinel-1 extended SAR (`COPERNICUS/S1_GRD`, 2024 Jul–Sep, IW, VV+VH)

3×3 figure: RGB, VV, VH, VV−VH, RVI, Dual-pol composite, VV temporal std, VH temporal std, Mid-season VV shift.

| Channel | Computation | Interpretation |
|---|---|---|
| VV | 2024 Jul–Sep linear mean → dB | surface roughness + moisture |
| VH | 2024 Jul–Sep linear mean → dB | volume scattering; canopy bright |
| VV − VH | `VV_dB − VH_dB` | polarization diff; bare (hi) vs vegetated (lo) |
| RVI | `4·VH_lin / (VV_lin + VH_lin)` | Radar Vegetation Index |
| Dual-pol composite | R=VV, G=VH, B=(VV−VH) | overview |
| VV/VH temporal std | per-pixel std over Jul–Sep, on dB | within-season dynamism |
| Mid-season VV shift | `VV_dB(Aug–Sep) − VV_dB(Jul)`, linear-mean first | direction of change |

**Why 2024-only**: labels describe state, not change. Within-season signals (temporal std, mid-season shift) capture dynamism without cross-year co-registration overhead.

**Why no InSAR coherence**: requires SNAP/HyP3 SLC processing — not a GEE product. VV/VH temporal std is the cheapest on-GEE proxy. If it shows polygon contrast, InSAR coherence becomes worth the separate infra investment.

---

## SE-derived Channels — Investigation

### Baseline feasibility (superseded, kept for reference)

Three diagnostics in one figure per tile (4×3 grid: RGB, per-tile PCA-RGB, global PCA-RGB, per-tile PC1–3, global PC1–3, prototype cosine, threshold mask, histogram):

| Diagnostic | Question | Method |
|---|---|---|
| Per-tile PCA | Does SE locally represent RTS within this tile? | Unsupervised, variance |
| Global PCA | Unsupervised sifter viability? | Unsupervised, variance (60–74°N land sample, n=10k) |
| Prototype cosine | Does cosine to labeled-RTS prototype discriminate RTS? | Supervised, mean-based |

SE = `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 64 bands, 10 m, unit vectors on the 64-D sphere. Prototype = mean of ~45k SE embeddings sampled across ~1818 labeled-positive tiles, re-normalized to unit length. Threshold T = 5th percentile of in-prototype cosines.

**Outcome**: per-tile PCA passes (shows RTS locally), global PCA fails (RTS isn't in the top 3 PCs of an unsupervised Arctic sample), prototype works on most tiles but has three failure modes:

1. **Narrow dynamic range** — all cosines compress into ~[0.86, 0.98] because SE embeddings cluster tightly on the hypersphere.
2. **Weak polygon-vs-background separation** — a single mean averages over morphological subtypes (headwall, floor, revegetating).
3. **Sign inversions** — e.g. OID 169 polygon mean falls below background mode.

This motivated the Phase 2 investigation below.

### Success Criteria

| Code | Criterion |
|---|---|
| **C1** | Spearman \|r\| ≤ 0.5 between SE-derived channel and every Sentinel auxiliary (NIR, SWIR, NBR, NDVI, NDWI, NDMI, TCB, TCW) in both all-pixel and positive-pixel regimes. |
| **C2** | Polygon-mean cosine above tile background mode on ≥ 6/7 test tiles — no more than 1 sign inversion. |

### Correlation analysis (15 channels, 150 held-out tiles)

Channel set: R, G, B, NIR, SWIR, NBR, NDVI, NDWI, NDMI, TCB, TCW, SE-cosine, SE-PC1, SE-PC2, SE-PC3. Per-tile PCA is excluded (basis drifts across tiles). Global PCA has a fixed basis and is a stable feature.

Method: Spearman pairwise in three regimes (all pixels, positive pixels, near-boundary at ±10 px). Pearson as linear-correlation reference (zero-variance column mask patched). No p-values — spatial autocorrelation breaks significance testing.

**Original single-prototype SE-cosine vs each Sentinel auxiliary:**

| Channel | r (all) | r (positive) | C1 pass |
|---|---|---|---|
| NIR | −0.44 | −0.32 | pass |
| SWIR | +0.31 | −0.23 | pass |
| NBR | −0.11 | −0.05 | pass |
| NDVI | −0.08 | **−0.53** | fails in positive regime |
| NDWI | +0.05 | **+0.58** | fails in positive regime |
| NDMI | +0.24 | +0.14 | pass |
| TCB | −0.01 | −0.15 | pass |
| TCW | −0.35 | +0.21 | pass |

Global C1 passes (max |r|=0.44 with NIR). The positive-regime failures on NDVI/NDWI reflect that SE, NDVI, and NDWI all respond to the exposed-soil / dead-vegetation signature that defines RTS.

**Dendrogram (Spearman, average linkage)**: SE-cosine joins NBR+NDMI at distance ≈ 0.60 — above the 0.5 redundancy threshold. SE carries information that is not a linear recombination of existing channels.

**Pair interpretations** (|r| > 0.7 pairs) live in `se_derived/correlation/interpretations.md` — auto-generated table with a user-fillable "Physical Reason" column.

### SE Variants

**Reference pool**: raw 64-D SE vectors for ~45k RTS pixels sampled from ~1668 labeled-positive tiles (held-out 150 excluded). No preprocessing — SE vectors are already unit-length; cosine = dot product.

#### Approach 1 — Multi-prototype

k-means over the reference pool, k ∈ {3, 5, 8, 12}, centroids re-normalized to unit length after fit. Score: `SE(x) = max_i cos(x, c_i)` over k prototypes.

| k | Silhouette |
|---|---|
| **3** | **0.287 (selected)** |
| 5 | 0.201 |
| 8 | 0.159 |
| 12 | 0.164 |

Cluster masses at k=3: 12.7% / 67.8% / 19.6% (one dominant cluster — SE embeddings are tightly concentrated).

**C2**: 6/7 correct, 1 inversion (OID 93: polygon 0.9464 vs background 0.9465 — effectively a tie). Passes target.
**C1 (on 7 test tiles, not held-out)**: NIR −0.26, NDVI −0.29, **NBR −0.79 (fails)**. Max-cosine scoring correlates strongly with NBR on positive-heavy tiles because both respond to surface disturbance.
**Dynamic range**: still [0.86, 0.98] — max-over-prototypes doesn't widen the range meaningfully.

Nearest-pixel prototype samples (`prototype_samples.png`) require the index-preserving sampler fix — see Open Items.

#### Approach 2 — Contrastive

Positive prototypes from Approach 1 (k=3). Negative pool: SE vectors at label=0 coordinates from ~200 tiles (~50k points), k-means k=3, centroids re-normalized. Score: `SE(x) = max_i cos(x, p_pos_i) − max_j cos(x, p_neg_j)`, range [−2, 2].

**C2**: **7/7 correct, 0 inversions**. Including OID 93 (polygon +0.0014, background −0.0244) and OID 169 (polygon +0.0249, background −0.0033).
**C1 (on 7 test tiles)**: NIR 0.05, NDVI 0.07, NBR 0.34 — all pass.
**Dynamic range**: centered near 0 with visible spatial structure. `oid262_variant.png` is a clean example — strong red inside polygon, blue elsewhere.

### Three-way Comparison

Outputs in [se_derived/variants/comparison/](se_derived/variants/comparison/). Per-tile 2×4 figures (`oid{OID}_three_way.png`): RGB + original single-prototype SE-cosine + Approach 1 max-cos + Approach 2 contrastive, top row as spatial maps with polygon overlay, bottom row as histograms with polygon-mean/background-mode annotations. Plus a `summary.png` + `summary.md` with C1 bars, C2 grouped bars per OID, and dynamic-range boxplots per variant.

**Pooled-7-tile summary** (see [summary.md](se_derived/variants/comparison/summary.md) for full tables):

| Variant | abs r NIR | abs r NDVI | abs r NBR | C2 sign margin (all OIDs positive?) |
|---------|----------|-----------|----------|-------------------------------------|
| original single-prototype | 0.23 | 0.27 | **0.74** | yes |
| Approach 1 multi-prototype | 0.26 | 0.30 | **0.79** | 6/7 (OID 93 effectively tied) |
| Approach 2 contrastive | 0.05 | 0.07 | 0.34 | yes |

Only Approach 2 passes C1 (abs r ≤ 0.5) on all three S2 channels, and its pooled-pixel boxplot is the only one not collapsed near 0.95. Confirms the recommendation from the individual-phase analyses above.

---

## Decision Log

- **2026-04-13** — ArcticDEM + Sentinel-2 per-tile baseline plots shipped. 7 test OIDs selected to span morphological variety.
- **2026-04-16** — SE feasibility study (per-tile PCA, global PCA, prototype cosine) complete. Single-prototype cosine exhibited the three failure modes noted above; motivated the Phase-2 investigation.
- **2026-04-17** — Dropped z-scoring. SE embeddings are unit vectors; z-scoring breaks the geometry.
- **2026-04-17** — Excluded per-tile PCA from the 15-channel correlation set — basis drifts across tiles, sign is arbitrary. Global PCA (fixed basis) kept.
- **2026-04-17** — k=3 selected for Approach 1 by silhouette (0.287).
- **2026-04-17** — Approach 2 (contrastive) is the recommended SE variant. Passes C1 and C2 on 7 test tiles with visible polygon-vs-background separation.
- **2026-04-17** — Folder reorganized by data source (`dem_derived/`, `sentinel_derived/`, `se_derived/`). Legacy `se_v2/` renamed to `se_derived/variants/`. Four scattered markdowns consolidated into this README.
- **2026-04-17** — Three-way comparison (original single-prototype + Approach 1 + Approach 2) run on all 7 test tiles. Approach 2 is the sole variant passing C1 on NIR/NDVI/NBR (all abs r ≤ 0.34) and C2 on all 7 OIDs with strictly positive polygon-mean − background-mode margin; recommendation reconfirmed.
- **2026-04-17** — Nearest-pixel rendering fix landed in [scripts/se_variants.py](../../scripts/se_variants.py). `_sample_se_at_points` now tags each feature with an `idx` property and returns `(vectors, kept_indices)` in input order; `build_reference_pool` tracks `tile_id` / `row` / `col` provenance via a new `reference_pool_se_{year}_v2` cache; `_render_nearest_crops` crops PlanetScope RGB at the actual nearest `(row, col)` with edge-padding. Added `--rebuild-reference-pool` CLI flag.

---

## Open Items

1. **Regenerate Approach 1 `prototype_samples.png`** with the nearest-pixel patch. Run `python scripts/se_variants.py --config configs/se_investigation.yaml --approach approach1 --rebuild-reference-pool` (~15–20 min of GEE `sampleRegions` on ~50k points), then re-run `--approach approach1` to overwrite the existing fallback panel with real RGB crops centered on the nearest pixel of each dominant cluster. The fix has already landed; only the rebuild + re-render remains.
2. **Re-run the 15-channel correlation** with the Pearson zero-variance mask to regenerate `se_derived/correlation/heatmap_all_pearson.png`. Full rerun costs ~65 min of GEE; defer unless needed. Spearman is the primary analysis per spec.
3. **Approach 2 C1 on the 150-tile held-out set** (currently only evaluated on the 7 test tiles). Requires either caching per-tile S2 channel stacks from Phase 1 or another full GEE pass.

---

## How to Regenerate

| Artifact | Command |
|---|---|
| DEM + Sentinel-2 per-tile plots | `conda run -n rts_dataset python plots/extra_channel_vis/extra_channel_plot.py` |
| SE feasibility + SAR per-tile plots | `conda run -n rts_dataset python plots/extra_channel_vis/se_sar_plot.py` |
| 15-channel correlation | `python scripts/channel_correlation.py --config configs/se_investigation.yaml` |
| SE variants (A1, A2) | `python scripts/se_variants.py --config configs/se_investigation.yaml --approach all` |
| Three-way SE comparison | `python scripts/se_variants.py --config configs/se_investigation.yaml --approach compare` |
| Rebuild reference pool (after nearest-pixel fix) | `python scripts/se_variants.py --config configs/se_investigation.yaml --approach approach1 --rebuild-reference-pool` |

All paths live in `configs/se_investigation.yaml`. No hardcoded paths in scripts.

---

## Implementation References

Scripts:

- [plots/extra_channel_vis/extra_channel_plot.py](extra_channel_plot.py) — DEM + Sentinel-2 per-tile figures
- [plots/extra_channel_vis/se_sar_plot.py](se_sar_plot.py) — SE feasibility + extended SAR
- [scripts/channel_correlation.py](../../scripts/channel_correlation.py) — 15-channel correlation
- [scripts/se_variants.py](../../scripts/se_variants.py) — SE variants (A1, A2, comparison)

Config: [configs/se_investigation.yaml](../../configs/se_investigation.yaml).

Caches in `.cache/` (gitignored):

- `prototype.npz` — single-prototype vector + metadata
- `global_pca.npz` — Arctic-land PCA basis
- `reference_pool_se_2024_v2.npz` — raw SE vectors with tile_id/row/col metadata (post nearest-pixel fix)
- `negative_pool_se_2024.npz` — background SE vectors
- `se_tile_<hash>.npz` — per-tile SE stacks
- `positive_tile_ids.json`, `held_out_tile_ids.json` — tile selections

---

## Caveats

- SE global PCA basis fit on a 60–74°N land-only bounding box, n=10k samples. Approximates the true inference domain (Arctic boreal ∩ permafrost ∩ PlanetScope coverage). Adequate for feasibility; not production-calibrated.
- SE prototype uses pixels from all ~1818 positive tiles. For Phase-1 correlation, the ~10% of reference pixels that fall inside the 150 held-out tiles create minor leakage; acceptable for a redundancy test.
- S1 ascending vs descending at each tile location not enforced identical. Document per tile if mixing produces geometric artefacts.
- SE embeddings are unit-length. PCA on them is imperfect (Euclidean assumption on a sphere) — fine for visualization, not quantitative use. Cosine similarity is the geometrically correct metric.
