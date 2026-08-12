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

## Size parameters — which number is which

> Four different size floors in this repo are called some variant of
> "min_blob" or "MMU". Three of them are **not** the product rule. This section
> is the SSoT; quote it, and do not restate its numbers elsewhere.

### The product rule — the only one to cite as the method

> **The shipped inventory applies no minimum mapping unit.**
> `vectorize_region.py --threshold 0.30 --min-area-m2 0`. The sole floor is a
> 2-pixel technical floor to kill single-pixel noise = **2.7–22 m²** across the
> map's 46.03–76.00°N span (it is a *pixel* floor, so it shrinks poleward).
> Measured: the smallest polygon in the delivered file is **2.69 m²** — the
> 2-px floor at 76°N. **6,761 polygons (11.2%) fall below ARTS P1 = 79 m².**
> Size acceptance is not a floor at all: it is the QC-calibrated adaptive
> `rts_class` rule (below).

### Not the product rule

These appear in `configs/deployment.yaml`, the experiment ledger and the git
history. None of them produced the delivered inventory.

| Parameter | Value | Units | What it is | What it is **not** |
|---|---|---|---|---|
| `metrics.min_blob_size_px` (17 training configs → `training/metrics.py`) | 10 | px | eval-only speckle guard for val/test object metrics | never applied to a product |
| `--min-blobs` tuning grid (`tune_object_operating_point.py`) | 1, 5, 10, 20, 40, **80** | px | the Val-Realistic sweep; **80 is the grid maximum** | not a product floor |
| `vectorize_min_blob_px` (`configs/deployment.yaml`) | 2000 | px | **superseded** first-product floor → `south_rts.gpkg` | **never swept** — 25× the grid max, a 2026-06-26 user choice scored post hoc |
| `LADDER_M2` (`build_ee_app_stats.py`) | 0…20000, 24 rungs | m² | the GEE app's viewer-side display slider | not applied to the delivered file |

**One key held three values.** `configs/deployment.yaml`'s pixel floor was
`10` (2026-05-04) → `80` (2026-06-25) → `2000` (2026-06-26). All three are
legitimately "the deployment min_blob", at different dates — which is why
ledger J's frozen `obj-P 0.584` is a **min_blob 80** number while the config
reads 2000 today. The sweep that chose 80 was **19 thresholds × {1,5,10,20,40,80}
px × 3 morph radii = 342 cells**; 2000 was never in it.

### Which stage does a number act on?

The sharpest distinction, and the easiest to lose: whether a floor filters
**ground-truth labels** or **model predictions**.

| Stage | Identifiers | Acts on | Units | Shipped value |
|---|---|---|---|---|
| **A. Label / GT** | `data.min_mapping_unit_px`; `apply_min_mapping_unit()`, `clean_positive_label(min_size=)` in `data/label_cleaning.py`; `object_scorecard.py --min-mapping-unit` | **ground truth** — sub-floor positives become **255 ignore**, never background | px | **0 = OFF** |
| **B. Eval** | `metrics.min_blob_size_px`; `object_scorecard.py --min-blob`; `evaluate_test.py --region-min-blobs` | **model predictions** | px | 10 (eval); 80 & 2000 reported side by side |
| **C. Tuning** | `tune_object_operating_point.py --min-blobs` | model predictions | px | grid {1…80} |
| **D. Vectorization** | `--min-area-m2` ✅ / `vectorize_min_blob_px` ❌ | **polygons** | **m²** / px | **`--min-area-m2 0`** |

Three consequences worth stating plainly:

1. **`--min-blob` filters predictions; `--min-mapping-unit` filters ground
   truth.** They sit on the same `object_scorecard.py` command line and are not
   interchangeable — one changes what the model is credited with, the other
   changes what it is asked to find.
2. **"MMU" is overloaded.** In `data/` it is a *GT label* floor in **pixels**
   whose removals become ignore(255). In `post-inference/` it is a *polygon
   area* floor in **m²**. Same acronym, different stage, different units.
3. **The model was not trained with an MMU.** `data.min_mapping_unit_px` is `0`
   in every shipped config — "the Minimum Mapping Unit is a scoring-time
   correction, not a training change" (`configs/v1_1_seed42.yaml`). The
   **"MMU-600"** figure in the diary is a GT-scoring sensitivity check, never a
   training or product setting.

### Pixel floors converted to ground area

A pixel floor is **not** a constant area: EPSG:3857 pixel ground area is
`res² · cos²(lat)` at res = 4.777 m, so it slides ~7× across this map. Use this
table rather than re-deriving — and note that **80 px is not 79 m²**, a
near-collision with ARTS P1 that is easy to make.

| pixel floor | at 50°N | at 71.63°N (inventory median) | at 76°N |
|---|---|---|---|
| 2 px (technical, shipped) | 19 m² | 5 m² | 3 m² |
| 10 px (eval filter) | 94 m² | 23 m² | 13 m² |
| **80 px (ledger J anchor)** | 754 m² | **181 m²** | 107 m² |
| 2000 px (legacy product) | 18,857 m² | 4,533 m² | 2,671 m² |

The test-region GT sits near 70°N, so ledger J's 80 px ≈ **214 m²** there.
Never quote the cos-uncorrected product (`80 × 4.777²` = 1,826 m²) — it is an
intermediate with no physical meaning, not a ground area.

## The three packages

### D1 — RTS Inventory (vector)

One dataset — the **MMU≈0 candidate inventory**, vectorized at threshold
**0.30** with **no minimum mapping unit** (technical floor 2 px ≈ 2.7–22 m²
over the 46–76°N span, smallest shipped polygon 2.69 m²; 6,761 polygons /
11.2% below ARTS P1 = 79 m²; seam-dissolved): **60,167 polygons / 688.2 km²**
(high 19,068 / 529.7 km² · medium 11,865 · low 29,234; built 2026-07-16) — in
four access forms:

| File | Form | For |
|---|---|---|
| `south_rts_candidates.gpkg` | flagship polygons + all attributes | GIS researchers; filter by `rts_class`, `conf_class`, `max_prob`, `area_m2` |
| `south_rts_high_confidence.gpkg` | `rts_class = 'high_confidence'` only | "just show me the slumps" — zero-decision fact map (model-derived tier, not human-verified) |
| `south_rts_centroids.gpkg` | representative point per polygon | pan-Arctic-zoom browsing; web maps |
| `south_rts_attributes.csv` / `.parquet` | attribute table, no geometry | pandas/R/Excel statistics, no GIS |
| `south_rts_t65.gpkg` | the same probabilities re-cut at 0.65, MMU≈0 (see below) | the 0.65 contour as geometry — nested inside the 0.30 outlines |

Attributes: `rts_id`, `conf_class`, `rts_class`, `max_prob`, `mean_prob`,
`area_m2`, `perimeter_m` (geodesic, WGS84 ellipsoid — never measure in 3857),
`centroid_lat/lon`, `area_m2_t45/t65/t80` (the polygon's area re-cut at
stricter thresholds — a per-object boundary-uncertainty band),
`nodata_frac` (fraction of NoData pixels in the polygon's padded bbox — QC
found false positives concentrate on high-NoData context; **soft triage only**,
real RTS can contain NoData), `detection_scale`, `tile_ids`.

**Confidence tiers** (by `max_prob`, inclusive bounds — SSoT
`scripts/export_south_products.py:TIER_BOUNDS`) with **measured South QC
precision** (2026-07 stratified rating, 280 verdicts, 63 unsure excluded;
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

**Held-out test score for this rule** (2026-08-12, `scripts/score_product_rule.py`,
re-cut of the frozen `test_probs.npz`): object P/R/F1 at IoU≥0.3 —
`high_confidence` **0.854 / 0.516 / 0.644**, whole candidate inventory
**0.532 / 0.581 / 0.556**. One-region GT (215 objects), and the tile-scored
replication cannot reproduce the cross-tile seam dissolve. Both anchors (min_blob
80 and 2000) reproduce exactly as a parity gate. See the *Shipped product rule on
Test-Realistic* sub-block in `docs/experiment_ledger.md`; artifact
`/mnt/outputs/v1.0/diagnostics/product_rule_scorecard_frozen_test.json`.

The original `south_rts.gpkg` (thr 0.65, min_blob 2000 px, 10,984 polygons /
238.08 km², delivered 2026-07-11) **remains untouched** for provenance;
`south_rts_high_confidence` supersedes the interim `south_rts_high.gpkg` and
the briefly-shipped `south_rts_confirmed.gpkg` (both deleted; 'confirmed' was
renamed 2026-07-17 — it wrongly implied human verification).

**`south_rts_t65.gpkg` — the 0.65 core outlines** (built 2026-08-03 for the
public GEE app): **23,682 polygons / 259.91 km²**, the same probability shards
re-cut at threshold 0.65 with **no MMU** (`vectorize_region.py --threshold 0.65
--min-area-m2 0`, 2 px technical floor), carrying the same attribute schema.
This is the *geometry* of the 0.65 contour, which nothing else in the family
has: the candidate polygons are outlined at 0.30 and only carry `area_m2_t65`
as a number, and `south_rts.gpkg` is the 0.65 cut with min_blob 2000 px already
baked in. Pairing it with the 0.30 outlines gives a nested contour pair.

Two checks at build time, both across independent code paths:
`{p≥0.65} ⊆ {p≥0.30}` holds exactly — **0 of 23,682** cores fall outside a
candidate polygon; and Σ`area_m2` = 259.91 km² against Σ`area_m2_t65` = 257.13
km² over the candidates (**+1.08%**), the polygonized boundary versus the
pixel-fraction estimator agreeing to about a percent.

**A 0.65 cut is really a 0.648 cut.** `_polygonize_block` binarizes at
`int(round(thr × 250))`, and Python rounds halves to even, so `round(162.5)` =
**162** — decoded 0.648, not 0.650. That is one `scaled_uint8` quantization
step (1/250 = 0.004), below what the raster can resolve, and every 0.65 product
shares it (`south_mask`, `south_rts.gpkg`, `south_rts_t65.gpkg`). Its one
visible consequence: 129 cores (0.54%) sit inside candidate polygons whose
`max_prob` is exactly 0.648, which `conf_class` therefore calls `medium` — so
the 0.65 core layer is **not** a strict subset of `high_confidence`.

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

**Earth Engine assets** — public, under `projects/pdg-project-406720/assets/`.
Ingested by `scripts/ingest_ee_app_assets.py` (which pins the shapefile field
renames the `.dbf` 10-char limit forces — `centroid_lat`/`centroid_lon` would
otherwise collide, and `tile_ids` overflows the 254-char text cap and is
dropped).

| Asset | Source | Used by |
|---|---|---|
| `south_rts_candidates` | `south_rts_candidates.gpkg` | app — 0.30 contour, points, inspector |
| `south_rts_t65` | `south_rts_t65.gpkg` | app — 0.65 core contour |
| `south_density_10km` | `density_10km_expected_m2.tif` (MEAN pyramiding) | app — zoomed-out overview |
| `south_rts_centroids` | `south_rts_centroids.gpkg` | app — mid-zoom points |
| `south_rts_high_confidence`, `south_likelihood_95m`, `south_mask`, `south_rts` | earlier cuts | `ee_south_viewer.js`; kept for provenance, **not used by the app** |

`south_likelihood_95m` is retired *from the app*: ingested as a continuous
raster it gets MEAN pyramiding, so zoomed out its mean fell below the app's own
`.gte(0.3)` mask and the layer erased itself. Re-ingesting with MAX pyramiding
would make it visible but not quantitative (max-prob over a 10 km screen pixel
saturates to ~1.0 wherever one detection exists), so the overview is the
threshold-free density grid instead.

### D4 — Human-verified inventory (review campaign)

A 2–3 person team is traversing **all 60,167 candidate polygons** and recording a
`rts`/`false`/`unsure` verdict on each, replacing the sampled precision estimates
above with a census. Protocol SSoT: `post-inference/review_campaign.md`; this
entry is the catalog's pointer to it.

The queue is ordered by descending `max_prob`, so the campaign's claim at any
moment is **"every polygon with `max_prob ≥ p` is human-reviewed"** — see the
`review_agreement.json` report for the current `p`. It is stoppable at any point:
unreviewed polygons carry a null verdict and are excluded from the verified cut.

| File | What | Status |
|---|---|---|
| `internal/rgb_chips/` + `rgb_chips.vrt` | the RGB chip mosaic, **rebuilt 2026-08-03 from 29,850 → 118,586 chips (60.7 GiB)**: it had been cut for the 10,984-polygon 0.65 product and covered only 29.9% of the 0.30 candidates | `scripts/build_rgb_chips.py --gpkg south_rts_candidates.gpkg --workers 64` |
| `internal/review_crops/<rts_id>_{t,w}.jpg` | two rendered crops per polygon (tight ~3× feature, wide ~1.5 km), red outline burned in — **120,334 JPEGs / 3.2 GB**, 20 polygons (0.03%) without imagery (`no_imagery.csv`) | built by `scripts/build_review_crops.py`; **`internal/`, not `products/`** — PlanetScope-derived, not for redistribution |
| `review/manifest.parquet` | the queue: 60,467 items (60,167 coverage + 300 injected replicates) in 301 batches of 200 | built by `scripts/build_review_manifest.py`, deterministic |
| `review/verdicts/<batch_id>.jsonl` | one immutable file per completed batch | written by the review app as the campaign runs |
| `south_rts_verified.gpkg` | candidates + `qc_verdict`, `n_reviews`, `reviewers`, `agreement`, `reviewed_at` | produced by `scripts/merge_review_verdicts.py`; re-run any time |
| `south_rts_verified_true.gpkg` | `qc_verdict = 'rts'` — the human-verified inventory | ditto |
| `qc_false_hard_negatives.gpkg` | every `false` — supersedes the 152-polygon v3 hard-negative seed | ditto |
| `review_agreement.json` | coverage, per-reviewer counts, Cohen's κ on ~300 cross-reviewer replicate pairs, and the confusion matrix against the 2026-07 pass | ditto |

`review_verdicts.csv` is shaped exactly like `qc_ratings.csv`, so
`scripts/score_qc_ratings.py` re-derives the tier × size precision grid from the
census with no change — at which point the Wilson intervals in the table above
collapse and caveats 2 and 3 are superseded **for reviewed polygons**.

Reviewing is a **precision** pass over what the model found. It does not look for
missed slumps, so caveat 7's "recall on 2025 imagery is unmeasurable" stands
regardless of how far the campaign gets.

## Caveats (read before using any product)

1. **Precision is NOT monotonic in threshold.** Object precision *peaks* at
   0.65 and falls above it (confident look-alike false positives survive any
   cut while true detections fragment). Filtering `max_prob ≥ 0.9` gives a
   *worse* fact map than `rts_class = 'high_confidence'`.
2. **Even `high_confidence` is ~55–90% precise, not 100%.** The measured high-tier
   precision is 0.54–0.90 depending on size band. `medium`/`low` are triage
   pools: at best every second, at worst almost no polygon is real. Use them
   for prospecting and recall-sensitive analyses only.
3. **23% of QC verdicts were `unsure`** (63/280) — excluded from the precision
   numbers, which therefore carry an extra uncertainty band (true precision
   lies between the all-false and all-rts extremes). The QC sample is
   stratified per (tier × size) cell, so pooled rates are *not* map-level
   precision.
4. **No minimum mapping unit.** Polygons go down to the 2-px technical floor
   (2.7–22 m² over the map's latitude span; smallest shipped polygon 2.69 m²);
   the MMU dial is latitude-constant in geodesic m² (the old 2000-px pixel
   floor and its latitude bias are gone — see "Size parameters — which number
   is which"). Tiny polygons are often partial detections
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
