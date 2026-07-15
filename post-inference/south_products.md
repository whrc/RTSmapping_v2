# South 2025Q3 RTS Product Catalog — SSoT

Every product cut from the pan-Arctic **South** (≈50–76°N) 2025 Q3 inference run:
what it is, where it lives, who it serves, and what it must not be trusted for.
This file is the **single source of truth for the product family**;
`arcgis_south_products.md` is the how-to-open guide and defers here for facts.

**Provenance (shared by every product).** Model v2 = 3-seed EffB5 ensemble
(seeds 42/43/44 averaged in-pipeline — per-seed maps were never written),
temperature-calibrated **T = 0.512321** (training.md §12, on val), inference
git `7b7d74c`, 41,567,572 tiles / 2,079 shards, completed 2026-07-10; canvas
8,388,608 × 1,606,304 px @ 4.777 m EPSG:3857; 3 quads genuinely absent
(0.001% — NoData). Products bucket:
`gs://rts-mapping-v2-usw1/inference/2025q3_south/products/`

**Decode (raster products).** Pixel value = **probability × 250**
(`scaled_uint8`); **255 = NoData**. `prob = pixel / 250.0`.

## The three packages

### D1 — RTS Inventory (vector)

One dataset — the **tiered candidate inventory**, vectorized at threshold
**0.30** (min_blob 2000 px, seam-dissolved): **25,716 polygons / 639.4 km²**
(high 17,239 / 522.3 km² · medium 6,765 · low 1,712; built 2026-07-14) — in
four access forms:

| File | Form | For |
|---|---|---|
| `south_rts_candidates.gpkg` | flagship polygons + all attributes | GIS researchers; filter by `conf_class` or `max_prob` |
| `south_rts_high.gpkg` | high tier only | "just show me the slumps" — zero-decision fact map |
| `south_rts_centroids.gpkg` | representative point per polygon | pan-Arctic-zoom browsing; web maps |
| `south_rts_attributes.csv` / `.parquet` | attribute table, no geometry | pandas/R/Excel statistics, no GIS |

Attributes: `rts_id`, `conf_class`, `max_prob`, `mean_prob`, `area_m2`,
`perimeter_m` (geodesic, WGS84 ellipsoid — never measure in 3857),
`centroid_lat/lon`, `area_m2_t45/t65/t80` (the polygon's area re-cut at
stricter thresholds — a per-object boundary-uncertainty band),
`detection_scale`, `tile_ids`.

**Confidence tiers** (by `max_prob`, inclusive bounds — SSoT
`scripts/export_south_products.py:TIER_BOUNDS`):

| Tier | max_prob | Val-anchor obj-precision* | South QC precision |
|---|---|---|---|
| `high` | ≥ 0.65 | ~0.61 | *(pending Phase-B rating)* |
| `medium` | 0.45–0.65 | ~0.49–0.52 (band) | *(pending)* |
| `low` | 0.30–0.45 | marginal detections | *(pending)* |

\* from the 342-combo operating-point sweep on **2 val regions only**
(`/mnt/outputs/v1.0/object_operating_point/effb5_ensemble/`); directional, not
authoritative — the South-specific stratified QC (60/band, longitude × area
strata, `scripts/sample_qc_polygons.py`) supersedes it once rated.

The original `south_rts.gpkg` (thr 0.65, 10,984 polygons / 238.08 km²,
delivered 2026-07-11) **remains untouched** for provenance; `south_rts_high`
is its successor within the tiered family.

### D2 — Probability rasters & spatial summaries

The probability map at three scales:

| File | What | For |
|---|---|---|
| `probability.vrt` + `probability_cog_shards/*.tif` (1,633) | **full-res canvas** (raw output) | re-thresholding, sensor fusion, benchmarking |
| `likelihood_95m.tif` | max-prob at ~95 m (20× decimation), browse COG | "where should I look" reconnaissance |
| `density_10km.gpkg` + `density_10km_expected_m2.tif` | 10-km cells: threshold-free **expected RTS area** + per-tier counts/areas | regional planning, field-campaign targeting |
| `density_0.5deg.gpkg` + `density_0.5deg_expected_m2.tif` | same on a 0.5° WGS84 grid | climate / permafrost-carbon modelling |
| `mask.vrt` + `mask_cog_shards/` | binary mask at thr 0.65 | legacy/simple raster consumers |

**Expected RTS area** = Σ decoded P × geodesic pixel area over the cell.
Because probabilities are temperature-calibrated, this is an *expectation no
threshold choice can bias* — the honest abundance statistic. Built by
`scripts/aggregate_probability.py` (one streaming pass, cos²lat Mercator
correction). Canvas total: **1,037.4 km²** (both grids agree exactly) —
note this integrates *all* probability mass, including diffuse
sub-min_blob/sub-0.30 signal no polygon product carries, so it exceeds the
639.4 km² candidate-outline total by design (abundance estimate ≠ inventory
area; the mass ordering is 238 km² @0.65 mask < 639 km² @0.30 outlines <
1,037 km² expectation).

### D3 — Documentation & communication

| File | What |
|---|---|
| `south_products.md` (this file) | catalog SSoT |
| `arcgis_south_products.md` | how to open everything in ArcGIS Pro (+ GEE viewer) |
| `south_rts_summary.md` / `.html` | factsheet: totals, size/latitude distributions, hotspot figure, tier precision |

## Caveats (read before using any product)

1. **Precision is NOT monotonic in threshold.** Object precision *peaks* at
   0.65 and falls above it (confident look-alike false positives survive any
   cut while true detections fragment). Filtering `max_prob ≥ 0.9` gives a
   *worse* fact map than `conf_class = 'high'`.
2. **The low tier is a candidate pool, not an inventory.** At the val anchor,
   roughly every second `medium`/`low` polygon is a false positive. Use those
   tiers for triage, prospecting, and recall-sensitive analyses.
3. **Latitude-dependent minimum size.** min_blob is 2000 *pixels* in EPSG:3857,
   so the geodesic minimum detectable area shrinks poleward: ≈1.1 ha at 50°N →
   ≈0.27 ha at 76°N. Size-frequency analyses must account for this.
4. **Geometry is the 0.30 outline.** In the tiered inventory every polygon —
   whatever its tier — is outlined at threshold 0.30. `area_m2_t45/t65/t80`
   quantify how the object shrinks at stricter cuts.
5. **No ground truth on 2025 imagery.** All precision figures are transferred
   (val) or sampled (Phase-B QC); recall on 2025 imagery is unmeasurable.
6. **Coverage gaps.** 3 of 309,101 Planet quads absent (NoData), plus imagery
   seams/clouds inherited from the 2025 Q3 basemap.
7. **EPSG:3857 areas lie.** Use the provided geodesic `area_m2`; never
   `ST_Area` in 3857 (~13× inflation at 74°N).
