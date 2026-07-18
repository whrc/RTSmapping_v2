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

One dataset — the **MMU≈0 candidate inventory**, vectorized at threshold
**0.30** with **no minimum mapping unit** (technical floor 2 px ≈ 10–45 m²,
below ARTS P1 = 79 m²; seam-dissolved): **60,167 polygons / 688.2 km²**
(high 19,068 / 529.7 km² · medium 11,865 · low 29,234; built 2026-07-16) — in
four access forms:

| File | Form | For |
|---|---|---|
| `south_rts_candidates.gpkg` | flagship polygons + all attributes | GIS researchers; filter by `rts_class`, `conf_class`, `max_prob`, `area_m2` |
| `south_rts_high_confidence.gpkg` | `rts_class = 'high_confidence'` only | "just show me the slumps" — zero-decision fact map (model-derived tier, not human-verified) |
| `south_rts_centroids.gpkg` | representative point per polygon | pan-Arctic-zoom browsing; web maps |
| `south_rts_attributes.csv` / `.parquet` | attribute table, no geometry | pandas/R/Excel statistics, no GIS |

Attributes: `rts_id`, `conf_class`, `rts_class`, `max_prob`, `mean_prob`,
`area_m2`, `perimeter_m` (geodesic, WGS84 ellipsoid — never measure in 3857),
`centroid_lat/lon`, `area_m2_t45/t65/t80` (the polygon's area re-cut at
stricter thresholds — a per-object boundary-uncertainty band),
`nodata_frac` (fraction of NoData pixels in the polygon's padded bbox — QC
found false positives concentrate on high-NoData context; **soft triage only**,
real RTS can contain NoData), `detection_scale`, `tile_ids`.

**Confidence tiers** (by `max_prob`, inclusive bounds — SSoT
`scripts/export_south_products.py:TIER_BOUNDS`) with **measured South QC
precision** (2026-07 stratified rating, 279 verdicts, 63 unsure excluded;
`qc_precision_grid.csv` has the full tier × size grid with Wilson CIs):

| Tier | max_prob | South QC precision (by size band) |
|---|---|---|
| `high` | ≥ 0.65 | 0.90 (500–2k m², n=10) · 0.54–0.56 (larger bands) |
| `medium` | 0.45–0.65 | 0.53 (<500 m²) · 0.11–0.31 (larger bands) |
| `low` | 0.30–0.45 | 0.00–0.31 — candidate pool only |

**`rts_class` — the QC-calibrated adaptive MMU** (SSoT
`scripts/export_south_products.py:assign_rts_class`). Instead of one fixed
minimum size, acceptance depends on confidence: cells of the (tier × size)
precision grid clearing a 0.5 precision floor are accepted. Measured result:
*all* high-tier cells clear it; the only other clearing cell is medium <500 m²
(0.53). So:

| rts_class | Rule | Count | Area |
|---|---|---|---|
| `high_confidence` | `conf_class = high` (any size) | 19,068 | 529.7 km² |
| `candidate` | `conf_class = medium` and `area_m2 < 500` | 25 | ~0.01 km² |
| `marginal` | everything else | 41,074 | 158.5 km² |

There is no size cut inside `high_confidence` because size did not predict falseness
— the *smallest* measured high-tier band was the most precise (0.90). (The
high <500 m² cell is unmeasured but also empty: the inventory contains zero
such polygons.)

The original `south_rts.gpkg` (thr 0.65, min_blob 2000 px, 10,984 polygons /
238.08 km², delivered 2026-07-11) **remains untouched** for provenance;
`south_rts_high_confidence` supersedes the interim `south_rts_high.gpkg` and
the briefly-shipped `south_rts_confirmed.gpkg` (both deleted; 'confirmed' was
renamed 2026-07-17 — it wrongly implied human verification).

**QC artifacts** (kept for reproducibility and v3): `qc_sample.gpkg` (280
stratified polygons, rated), `qc_ratings.csv` (the verdicts, also uploaded to
GCS 2026-07-18 — was previously repo-only, a doc/bucket mismatch fixed in the
audit below), `qc_precision_grid.csv` (scored grid), `qc_false_hard_negatives.gpkg`
(**152 user-verified false positives — the v3 hard-negative seed set**; FP
modes: NoData context, water bodies, snow, mining/infrastructure look-alikes).

**2026-07-18 bucket audit.** Three items removed from `products/` as
unused/redundant: `mask.vrt` + `mask_cog_shards/` (binary thr-0.65 mask, fully
derivable from `probability_wmts_z10` or `south_rts.gpkg`, unused since the
tiered inventory existed); `qc_chips/` (raw chip mosaic, superseded once
`qc_rater.html` started embedding its own crops as base64 JPEGs — nothing
referenced it). `rgb_chips.vrt` + `rgb_chips/` (license-restricted PlanetScope
derivative, "not for redistribution") was **moved out of `products/`** to
`gs://rts-mapping-v2-usw1/inference/2025q3_south/internal/` — it must never
sit inside the folder `deliverables/README.md` hands to ADC/PDG.

### D2 — Probability rasters & spatial summaries

The probability map at three scales:

| File | What | For |
|---|---|---|
| `probability.vrt` + `probability_cog_shards/*.tif` (1,633) | **full-res canvas**, canvas-anchored shards (raw output / provenance master) | re-thresholding, sensor fusion, benchmarking |
| `probability_wmts_z10.vrt` + `probability_wmts_z10/*.tif` (80,159) | same probability, re-cut onto the global WebMercatorQuad z10 grid (exact pass-through, pixel-identical — see `deliverables/README.md` §2) | ADC/PDG handover; anyone who needs one-file-per-standard-tile access |
| `likelihood_95m.tif` | max-prob at ~95 m (20× decimation), embedded colormap + true block-max overviews | "where should I look" reconnaissance |
| `density_10km.gpkg` + `density_10km_expected_m2.tif` + `density_10km_browse.tif` | 10-km cells: threshold-free **expected RTS area** + per-tier counts/areas | regional planning, field-campaign targeting |
| `density_0.5deg.gpkg` + `density_0.5deg_expected_m2.tif` + `density_0.5deg_browse.tif` | same on a 0.5° WGS84 grid | climate / permafrost-carbon modelling |

`*_browse.tif` are RGBA color-relief renders (log-percentile breaks) that are
informative with zero styling; `*_expected_m2.tif` are the compute-with floats.

**Expected RTS area** = Σ decoded P × geodesic pixel area over the cell.
Because probabilities are temperature-calibrated, this is an *expectation no
threshold choice can bias* — the honest abundance statistic. Built by
`scripts/aggregate_probability.py` (one streaming pass, cos²lat Mercator
correction). Canvas total: **1,037.4 km²** (both grids agree exactly) —
note this integrates *all* probability mass, including diffuse sub-0.30
signal no polygon product carries, so it exceeds the 688.2 km²
candidate-outline total by design (abundance estimate ≠ inventory area; the
mass ordering is 238 km² @0.65 mask < 688 km² @0.30 MMU≈0 outlines <
1,037 km² expectation).

### D3 — Documentation & communication

| File | What |
|---|---|
| `south_products.md` (this file) | catalog SSoT |
| `arcgis_south_products.md` | how to open everything in ArcGIS Pro (+ GEE viewer) |
| `south_rts_summary.md` / `.html` | factsheet: totals, size distribution vs ARTS v6, latitude distribution, hotspot figure, measured precision grid |
| `deliverables/README.md` (repo) / `products/README.md` (GCS) | ADC/PDG handover: minimized submission manifest, WMTS tiling convention, dataset + file-level metadata |

## Caveats (read before using any product)

1. **Precision is NOT monotonic in threshold.** Object precision *peaks* at
   0.65 and falls above it (confident look-alike false positives survive any
   cut while true detections fragment). Filtering `max_prob ≥ 0.9` gives a
   *worse* fact map than `rts_class = 'high_confidence'`.
2. **Even `high_confidence` is ~55–90% precise, not 100%.** The measured high-tier
   precision is 0.54–0.90 depending on size band. `medium`/`low` are triage
   pools: at best every second, at worst almost no polygon is real. Use them
   for prospecting and recall-sensitive analyses only.
3. **23% of QC verdicts were `unsure`** (63/279) — excluded from the precision
   numbers, which therefore carry an extra uncertainty band (true precision
   lies between the all-false and all-rts extremes). The QC sample is
   stratified per (tier × size) cell, so pooled rates are *not* map-level
   precision.
4. **No minimum mapping unit.** Polygons go down to ~2 px (10–45 m²);
   the floor is latitude-constant in geodesic m² (the old 2000-px pixel MMU
   and its latitude bias are gone). Tiny polygons are often partial detections
   sitting on a part of a real slump — correct location, under-delineated
   extent.
5. **Geometry is the 0.30 outline.** In the tiered inventory every polygon —
   whatever its class — is outlined at threshold 0.30. `area_m2_t45/t65/t80`
   quantify how the object shrinks at stricter cuts.
6. **Known false-positive modes** (from the QC): NoData speckle, small water
   bodies, snowy scenes, high-NoData tiles, and mining/infrastructure
   look-alikes. `nodata_frac` helps triage the first mode as a **soft
   attribute — not applied as a filter anywhere in this pipeline**; no
   `rts_class` is auto-downgraded by it, by design (real RTS can contain
   NoData, and there is no QC evidence for where a safe cutoff would sit).
   **No water-body attribute exists.** A static water layer (JRC Global
   Surface Water, OSM) was considered and rejected — most of the FP water
   bodies observed in QC are small and likely ephemeral (meltwater,
   thermokarst ponds), which a static mask would miss or mis-tag; a real fix
   needs a same-epoch water mask, deferred to v3.
7. **No ground truth on 2025 imagery.** Precision is sampled (2026-07 QC);
   recall on 2025 imagery is unmeasurable.
8. **Coverage gaps.** 3 of 309,101 Planet quads absent (NoData), plus imagery
   seams/clouds inherited from the 2025 Q3 basemap.
9. **EPSG:3857 areas lie.** Use the provided geodesic `area_m2`; never
   `ST_Area` in 3857 (~13× inflation at 74°N).
