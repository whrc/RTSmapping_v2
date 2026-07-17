# Pan-Arctic South RTS Products — ADC / PDG Handover

Handover package for the **Arctic Data Center** (archive + DOI) and the
**Permafrost Discovery Gateway** (visualization) from the 2025 Q3 pan-Arctic
South retrogressive thaw slump (RTS) mapping run. The submission is
deliberately minimal — **two data products plus one provenance file** — so the
metadata burden stays small. Everything else the pipeline produced is listed in
the appendix for reference but is *not* part of the submission.

This document lives in the repo at `deliverables/README.md` (source of truth)
and is published verbatim at
`gs://rts-mapping-v2-usw1/inference/2025q3_south/products/README.md`.

---

## 1. Submission manifest

| File | Size | What it is |
|---|---|---|
| `gs://rts-mapping-v2-usw1/inference/2025q3_south/products/probability_wmts_z10/` (+ `probability_wmts_z10.vrt`) | 12 GB, 80,159 COGs (one per tile) | **RTS probability, full resolution.** Calibrated model scores over the whole domain, re-tiled so each file is exactly one WMTS WebMercatorQuad zoom-10 tile (§2). uint8; **pixel value = probability × 250; 255 = NoData** (`prob = pixel / 250.0`). EPSG:3857 @ 4.777 m/px. |
| `gs://rts-mapping-v2-usw1/inference/2025q3_south/products/south_rts_candidates.gpkg` | 206 MB | **RTS polygon inventory.** 60,167 polygons vectorized at threshold 0.30 with no minimum mapping unit; every polygon classified by the QC-calibrated `rts_class` (§4). This is the *only* vector file: `rts_class = 'confirmed'` (19,068 polygons / 529.7 km²) is the conservative "fact map"; the rest are candidate/marginal pools for recall-sensitive uses. |
| `gs://rts-mapping-v2-usw1/inference/2025q3_south/products/region_log.json` | <1 KB | Machine-readable run provenance (canvas geometry, resolution, thresholds, tile counts). |

## 2. Raster tiling — answers to your questions

**How were the original ~1,600 shards divided?** The inference canvas
(8,388,608 × 1,606,304 px, EPSG:3857 @ 4.777314 m/px) was written as
65,536-px (2¹⁶) square COGs on a grid anchored at the canvas's north-west
corner, named `probability_{row}_{col}.tif` with canvas-relative indices.
Only grid cells containing at least one non-empty processing block were
written — 1,633 files of 3,200 possible cells (the domain is sparse: ocean
and never-inferred areas have no file). Each COG has 512-px internal tiles
and overviews, so HTTP range-request subset access already worked.

**Was a convention followed, and does it match a WMTS Tile Matrix Set?**
Nearly. The pixel grid is exactly the **WebMercatorQuad** family: the
resolution is the zoom-15 cell size (4.777314267… m), each 65,536-px shard
footprint is exactly one zoom-7 tile (313,086.068 m), and columns are aligned
with the global matrix. But shard *rows* were anchored to the region's north
edge rather than the matrix origin — ~77.7 km off the global z7 grid — so a
shard did **not** correspond to precisely one WMTS tile.

**What we did about it.** For this handover we re-cut the probability canvas
onto the exact global grid at **matrix level z10** (8,192-px / ~39 km tiles —
smaller files, per your "appropriate dimensions for easier access"):
`probability_wmts_z10/{col}_{row}.tif`, where `col` (0–1023, from 180°W) and
`row` (0–1023, from the northern matrix edge) are **global WebMercatorQuad z10
indices**. Tile bounds are exactly
`(-20037508.342789244 + col·39135.75848201, 20037508.342789244 − row·39135.75848201)`
(NW corner). All-NoData tiles are omitted. Because the source grid is aligned
to the global grid to <0.001 px, the re-cut is an exact pass-through — pixel
values and positions are unchanged (verified by pixel-identity comparison
against the original mosaic). We used WebMercatorQuad rather than a polar
scheme (e.g. UPSArcticWGS84Quad) because the data is natively EPSG:3857 —
conforming required only a grid shift, not a reprojection.

**Building a full multi-zoom tileset.** If you want a served pyramid, GDAL
can produce it directly from our VRT (we deliberately left this to your
existing workflows):

```bash
gdal raster tile --tiling-scheme WebMercatorQuad \
    probability_wmts_z10.vrt out_tiles/        # GDAL >= 3.11
# or: gdal2tiles.py --profile mercator probability_wmts_z10.vrt out_tiles/
```

Remember the scaled_uint8 decode (value/250, 255 = NoData) if the pyramid is
rendered rather than kept as data tiles.

## 3. Dataset metadata

**Abstract.** Pan-Arctic map of retrogressive thaw slump (RTS) probability
and an RTS polygon inventory for the circumpolar band ≈50–76°N, derived from
2025 Q3 (July–September) PlanetScope basemap imagery at 4.777 m resolution
(EPSG:3857). A semantic-segmentation ensemble (three EfficientNet-B5 U-Net++
models, seeds 42/43/44, predictions averaged) was applied to 41,567,572
512-px tiles; per-pixel probabilities are temperature-calibrated
(T = 0.512321) so scores are directly interpretable. The polygon inventory
contains 60,167 candidate outlines (688.2 km²) vectorized at probability 0.30
with no minimum mapping unit, each classified `confirmed` / `candidate` /
`marginal` by a QC-calibrated rule measured on 279 expert ratings.

**Methods (condensed).**
- *Training:* model v2, supervised on 2024 PlanetScope basemaps with
  expert-digitized RTS labels (details in the project repo,
  `training/training.md`).
- *Inference:* 2025 Q3 PlanetScope quads → 512-px tiles → 3-seed ensemble
  mean → temperature scaling → Gaussian-weighted overlap fusion (σ = 128 px)
  onto the region canvas; run completed 2026-07-10, git `7b7d74c`.
- *Vectorization:* threshold 0.30, seam-dissolved, no minimum mapping unit
  (technical floor 2 px ≈ 10–45 m²).
- *QC calibration:* 280-polygon stratified sample (confidence tier × size
  band) expert-rated in 2026-07; 279 verdicts scored into a precision grid
  (Wilson CIs); grid cells clearing 0.5 precision define `rts_class`
  acceptance. Measured precision of `confirmed`: 0.54–0.90 by size band.

**Coverage.** Spatial: ≈50–76°N, all longitudes (EPSG:3857 canvas y
5,711,221.9 – 13,385,040.9 m); land with 2025 Q3 PlanetScope coverage; 3 of
309,101 source quads absent (NoData). Temporal: imagery 2025-07-01 –
2025-09-30 (quarterly basemap); processing 2026-07.

**Usage caveats (important).**
1. Precision is **not monotonic in threshold** — it peaks near 0.65;
   filtering to `max_prob ≥ 0.9` yields a *worse* fact map than
   `rts_class = 'confirmed'`.
2. `confirmed` is ~54–90% precise (measured), not 100%; `medium`/`low`
   tiers are triage pools for recall-sensitive work.
3. Recall on 2025 imagery is unmeasured (no 2025 ground truth exists);
   precision comes from the 2026-07 QC sample.
4. Known false-positive modes: NoData-speckle context, small (likely
   ephemeral) water bodies, snow, mining/infrastructure look-alikes.
5. Compute areas from the supplied geodesic `area_m2` attribute — planar
   areas in EPSG:3857 are inflated ~13× at 74°N.

## 4. File-level metadata

**Raster (`probability_wmts_z10/*.tif`).** Single band, uint8, NoData 255.
`probability = value / 250.0` (0–250 valid range). COG layout: 512-px internal
tiles, nearest-resampled internal overviews, deflate compression, EPSG:3857.
One file per WebMercatorQuad z10 tile as defined in §2.

**Vector (`south_rts_candidates.gpkg`, layer `south_rts_candidates`,
polygon geometry, EPSG:3857).**

| Attribute | Type | Description |
|---|---|---|
| `rts_id` | int | stable polygon id |
| `area_m2` | float | polygon area, **geodesic** (WGS84 ellipsoid), m² |
| `perimeter_m` | float | polygon perimeter, geodesic, m |
| `centroid_lat` / `centroid_lon` | float | representative point, WGS84 degrees |
| `mean_prob` / `max_prob` | float | mean / max calibrated probability inside the polygon (0–1) |
| `area_m2_t45` / `_t65` / `_t80` | float | polygon area re-cut at thresholds 0.45 / 0.65 / 0.80 — per-object boundary-uncertainty band, m² |
| `conf_class` | text | confidence tier by `max_prob`: `high` ≥0.65 · `medium` 0.45–0.65 · `low` 0.30–0.45 |
| `rts_class` | text | QC-calibrated class: `confirmed` (19,068) · `candidate` (25) · `marginal` (41,074) — see §3 Methods |
| `nodata_frac` | float | fraction of NoData pixels in the polygon's padded bbox — soft triage hint only (real RTS can contain NoData); never used as a filter |
| `detection_scale` | text | internal detection-scale tag |
| `tile_ids` | text | comma-separated source-tile ids (provenance) |

Geometry note: every outline is the **0.30 threshold** boundary regardless of
class; the `area_m2_t*` attributes quantify shrinkage at stricter cuts.

## 5. Appendix — full product family (not part of the submission)

Everything else under
`gs://rts-mapping-v2-usw1/inference/2025q3_south/products/`, for reference
(facts and caveats in the repo's `post-inference/south_products.md`):

| File | Size | One-liner |
|---|---|---|
| `probability.vrt` + `probability_cog_shards/` | 13.6 GB | original canvas-anchored probability shards (provenance master for `probability_wmts_z10/`) |
| `mask.vrt` + `mask_cog_shards/` | 9.5 GB | binary RTS mask at threshold 0.65 |
| `south_rts.gpkg` | 62 MB | original thr-0.65 / min-blob-2000 inventory (10,984 polygons; superseded, kept for provenance) |
| `south_rts_confirmed.gpkg` | 117 MB | convenience extract: `rts_class = 'confirmed'` only |
| `south_rts_centroids.gpkg` | 13 MB | representative point per candidate polygon |
| `south_rts_attributes.csv` / `.parquet` | 12 / 5 MB | attribute table without geometry |
| `likelihood_95m.tif` | 259 MB | max-probability overview at ~95 m, embedded colormap |
| `density_10km.*`, `density_0.5deg.*` | ~296 MB | gridded expected RTS area (threshold-free abundance) + browse renders |
| `qc_sample.gpkg`, `qc_ratings.csv`, `qc_precision_grid.csv`, `qc_false_hard_negatives.gpkg`, `qc_rater.html` | ~20 MB | the 2026-07 QC campaign: rated sample, verdicts, scored precision grid, verified false positives, offline rating tool |
| `south_rts_summary.md` / `.html` | <1 MB | factsheet (totals, size/latitude distributions, precision table) |
| `rgb_chips.vrt` + `rgb_chips/` | 16.8 GB | RGB context chips of detection tiles (derived PlanetScope imagery — license-restricted, not for redistribution) |
