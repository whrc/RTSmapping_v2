# Future work — Mining new training tiles from 2025 predictions

**Status:** planned / not started. This is a design + TODO spec to pick up later; no code exists yet.
**Owner:** TBD (team of 2–3 raters + 1 implementer).
**Last updated:** 2026-07-22.

---

## Goal

Convert human-verified 2025 South detections into **new training tiles** for the next model version,
on two tracks from a single human-rating pass:

- **Positive-tile gain** — confirmed-true detections → new `TrainClass=positive` tiles (label = the
  predicted polygon). Target **~1,000 confirmed positives**, weighted toward **underrepresented
  regions**. The value here is **recall in under-covered geography**, not boundary precision —
  matched-object outlines are already near the label ceiling (prior finding:
  [[outline-refinement-killed-at-phase0]]).
- **Hard-negative mining** — confirmed-false detections → `TrainClass=negative` tiles. Target
  **~1,000–2,000**, drawn from **high/medium-confidence** FPs (the ones crossing threshold and
  hurting precision), stratified by **failure mode** and region.

Success is measured by three things, not just count:
1. **Count** — new positives / hard-negatives.
2. **Coverage** — ecoregions gained (regions going 0 → N training tiles).
3. **Representation** — reduction in per-region imbalance (Gini / max:median of per-region counts).

The mined tiles train **alongside** existing v1.0 data as a new dataset version (delta root), not as
a replacement.

### Input (already exists)
- `/mnt/outputs/inference/south/products_local/south_rts_candidates.gpkg` — **60,167 candidate
  polygons**, EPSG:3857, with `rts_id, conf_class (high≥0.65 / medium 0.45–0.65 / low 0.30–0.45),
  area_m2, centroid_lat/lon, max_prob, nodata_frac, tile_ids`.
- `.../qc_chips/rgb_chips.vrt` — RGB chip mosaic for rendering crops.
- Already-rated ids to exclude: `post-inference/qc_ratings.csv` (the 280-item precision round).

### FP failure-mode patterns (observed)
False positives cluster into recognizable patterns — **mining/anthropogenic landscapes, riverside
erosion, barren rock/outcrop, water, snow, other** (see [[south-qc-failure-modes-2026-07]]).
Capturing the pattern per false verdict lets us hit a **per-mode quota** (a few hundred *each* is the
impactful unit, not the aggregate) and yields a reusable FP taxonomy for v3.

---

## Locked decisions

- **Coordination — static per-worker shards, ~300 items/shard.** Each shard is a self-contained HTML
  page like today's `qc_rater.html`. Workers claim shards off a shared checklist, rate on-and-off /
  simultaneously, export one CSV per shard; we merge. **Zero backend** — small shards keep assignment
  flexible.
- **Imagery year — 2025 q3** for mined tiles (label bound to the image it was derived from; the
  conventional pseudo-label rule). Consequences: **recompute `normalization_stats.json`** over the
  combined 2024+2025 set for the new version; add a **`year` provenance column**; **val/test stay
  frozen (2024)** for cross-version comparability.
- **Positive label geometry** — predicted polygon interior = `1`, an outward **buffer collar
  (~1–2 px, R≈5–10 m) = `255` ignore**, rest = `0`. Honors known boundary uncertainty without
  eroding the confident core.
- **Verdict schema — `1=rts / 2=false / 3=decide-later`.**
  - `3` is **deferral, not a final judgment**: deferred items are **revisited at the very end**; any
    still-`3` after revisit collapses to **unsure** → excluded from *both* training tracks.
  - **Optional FP sub-tag:** after `2=false`, an optional keypress records the failure mode
    (`m`=mining/anthropogenic, `r`=riverside erosion, `b`=barren rock, `w`=water, `s`=snow, `o`=other)
    into an `fp_category` column. Skippable so the main pass stays low-friction.

### Sizing
- Net **~1,000 confirmed positives** + **~1,000–2,000 hard negatives** ⇒ rate **~2,000–3,500**
  polygons across 2–3 workers ⇒ **~7–12 shards of 300**. Pool default ~3,000 (configurable).
- Hard negatives are worth far more per-unit than the 20k random negatives (they hit real error
  modes). Aim **a few hundred per failure mode**; diminishing returns past ~2–3k until a retrain
  shows which modes persist. Mine them from **high/medium**-confidence FPs (a 0.31 false barely
  matters).

---

## Design overview

Pipeline:

```
south_rts_candidates.gpkg
  → [1a] coverage/representation pool sampler + shard splitter
  → [1b] per-shard self-contained rating pages  → workers rate → per-shard CSVs
  → [1c] merge CSVs (+ revisit set for leftover decide-later)
  → [2a] confirmed_positive.gpkg / confirmed_false.gpkg + increment report
  → [3]  tile production (reuse existing miners) → delta dataset root
  → wire via data.additional_roots → train next version
```

The existing **precision-QC path stays intact** (`sample_qc_polygons.py` → `qc_rater.html` →
`score_qc_ratings.py`). This work adds **sibling `mine_*` scripts** that reuse its helpers but serve a
different objective (coverage/representation, not precision estimation) and a shardable multi-worker
UI.

---

## TODO — Phase 1: Rating system

- [ ] **`scripts/mine_sample_polygons.py` (new)** — coverage/representation pool sampler + shard splitter.
  - [ ] Read candidates; exclude already-rated `rts_id`s (`post-inference/qc_ratings.csv`).
  - [ ] Assign ecoregion via point-in-polygon `sjoin` vs `domain/circumpolar_subregions.geojson`
        (reuse the `sjoin` pattern in `negative_tile_creation.py`).
  - [ ] Compute a **per-region deficit** from current training positive counts
        (`/mnt/outputs/v1.0/data_local/metadata.csv`); give underrepresented + brand-new (0-tile)
        regions a **higher sampling floor**.
  - [ ] Draw pool from **high/medium** `conf_class` + a slice of **low-confidence in underrepresented
        regions** (recall positives). Default `--pool-size 3000`.
  - [ ] **Region-mix each shard** (shuffle, then split into fixed numbered shards of
        `--shard-size 300`) so any single shard advances coverage broadly.
  - [ ] Output `mining_pool.gpkg` (+ `shard_id`, `RegionName` cols) and a `shards/` dir of per-shard
        GPKGs. Reuse `SIZE_BANDS` / `N_LON_BINS` idioms from `sample_qc_polygons.py`; seed 42.

- [ ] **`scripts/build_mining_rating_page.py` (new; adapted from `build_qc_rating_page.py`)** —
      one self-contained HTML **per shard**.
  - [ ] Reuse `_crop_bounds` / `_render_crop` (tight+wide crops, red outline, base64 JPEG) verbatim.
  - [ ] Show **`RegionName`** in the metadata line; per-region tally within the shard; shard id +
        worker name baked in.
  - [ ] **localStorage key namespaced per shard** (`mining_v1_shard{ID}`); export
        `mining_ratings_shard{ID}.csv` with schema `rts_id,qc_verdict,fp_category`.
  - [ ] Keys: `1=rts / 2=false / 3=decide-later`, arrows navigate; after `2`, optional
        `m/r/b/w/s/o` → `fp_category`; a **revisit key** (e.g. `d`) jumps to the next still-`3` item.
  - [ ] ~300-item pages stay well under the current 19 MB / 280-item page → browser-friendly.

- [ ] **Shared `CLAIMS.md` (or a Google Sheet)** listing `shard_00..shard_NN` + who claimed each —
      the entire coordination mechanism (static, no backend).

- [ ] **`scripts/merge_mining_ratings.py` (new)** — merge all `mining_ratings_shard*.csv`.
  - [ ] Dedup by `rts_id` (carry `fp_category`); on >1 rater for the same id, record **inter-rater
        agreement** and keep majority/first with a conflict flag → unified `mining_ratings.csv`.
  - [ ] Emit a **revisit set** (GPKG + a fresh `shard_revisit.html` via 1b) of all still-`decide-later`
        items for one final pass; whatever remains is treated as **unsure** downstream.

## TODO — Phase 2: Seeds, scoring, increment report

- [ ] **`scripts/score_mining_ratings.py` (new; reuses `score_qc_ratings.export_false_polygons`)**.
  - [ ] `confirmed_positive.gpkg` (`rts`) and `confirmed_false.gpkg` (`false`, carrying `fp_category`).
        Leftover `decide-later` → **unsure**, excluded from both.
  - [ ] **Region filter:** drop seeds whose ecoregion is in `val_realistic`/`val_balanced`/
        `test_realistic` (load `splits.yaml`); record excluded counts + set of **brand-new
        train-eligible regions**. Guarantees no val/test leakage.
  - [ ] **Increment report (facts only, no editorializing** — see [[reports-facts-only-no-interpretation]]):
        per-ecoregion before/after positive & negative counts; ecoregions gained (0→N, listed);
        distinct ecoregions with ≥1 positive before/after; balance metric (Gini + max:median of
        per-region positive counts, before vs after); **hard-negative breakdown by `fp_category`**
        (per-mode counts vs the ~few-hundred quota). Log val/test exclusions, decide-later drops, and
        new train regions to append.

## TODO — Phase 3: Tile production (reuse existing miners + thin glue)

Delta lives in its own root (mirror the `v1.1_delta` precedent), e.g.
`gs://rts-mapping-v2/training/v1.2_mined/{PLANET-RGB,labels,EXTRA,metadata.csv}`, wired via
`data.additional_roots`. **Do not** use `restage_v1_additive.py` — it is hardcoded to
`known_issues_v1.0.json` and not reusable for arbitrary adds.

- [ ] **Positives.**
  - [ ] Build `POSITIVE.geojson` (confirmed polygons) + `IGNORE.geojson` (buffer collars
        `poly.buffer(+R).difference(poly)`).
  - [ ] Build a **2025 `TILE_BOUNDARIES.geojson`** from the union of confirmed-positive `tile_ids`
        materialized against the 2025 inference grid (`tiles_2025q3_domain_full.csv`), each footprint
        carrying a resolvable 2025 quad path.
  - [ ] **Required 1-line edit:** `positive_tile_creation.py:name_to_blob()` hardcodes the 2024
        `_quad.tif` pattern; 2025 quads are `..._quad_file_format.tif` in `quad_index_2025q3.csv`.
        Add a full `gcs_path` column to the boundaries GeoJSON and have `name_to_blob` prefer it.
  - [ ] Run `positive_tile_creation.py` (grid-driven: burns ignore then overwrites interior with 1).

- [ ] **Negatives.**
  - [ ] Convert `confirmed_false.gpkg` → GeoJSON with `TrainClass="Negative"` (script filters on
        capital-N) + a 2025 negative grid with `delivery_location/basemap_name/grid_column/grid_row`
        synthesized from `quad_index_2025q3.csv`.
  - [ ] Run `negative_tile_creation.py` (centroid-framed; resume-appends to the delta `metadata.csv`).

- [ ] **EXTRA:** `generate_extra_tiles.py --groups all --year 2025 --metadata <delta metadata.csv>
      --rgb-dir <delta PLANET-RGB> --out-dir <delta EXTRA> --se-artifacts se_artifacts.npz`
      (no new code).

- [ ] **`Tile_ID` collision guard (~5 lines):** `load_metadata_multiroot` raises if a mined
      `Tile_ID` (geohash-12 centroid) already exists in frozen v1.0 — drop such mined rows first.

- [ ] **`splits.yaml`:** append brand-new **train-only** regions to `splits.yaml[train]`; re-run
      `data/splits.py:assert_no_region_leakage` (val/test untouched → still passes).

- [ ] **Normalization + provenance:** add `year=2025` to mined metadata rows; **recompute
      `normalization_stats.json`** over the combined set for the new version.

- [ ] **Config (new, mirror `configs/v1_1_seed43.yaml`):** add the delta local mirror to
      `data.additional_roots`; keep the frozen split; point `normalization_stats_path` at the
      recomputed file.

## TODO — Tests (CLAUDE.md Rule 4; update `tests/tests.md` in the same change)

- [ ] `tests/test_mine_sample_polygons.py` — region deficit weighting, shard-size split, region-mix,
      exclusion of already-rated ids, seed determinism.
- [ ] `tests/test_build_mining_rating_page.py` — verdict schema (`1/2/3` + `fp_category`),
      shard/worker tagging, namespaced localStorage key, per-shard CSV name+columns, `RegionName` in items.
- [ ] `tests/test_merge_mining_ratings.py` — dedup, overlap/agreement, conflict flag, `fp_category`
      carry, revisit-set emission.
- [ ] `tests/test_score_mining_ratings.py` — positive/false seed split, decide-later→unsure exclusion,
      val/test region exclusion, increment-report counts + balance metric + ecoregions-gained +
      per-`fp_category` breakdown.

---

## Files

**Adapt / reuse:**
`scripts/build_qc_rating_page.py` (`_crop_bounds`, `_render_crop`),
`scripts/sample_qc_polygons.py` (`SIZE_BANDS`, lon-bin idiom),
`scripts/score_qc_ratings.py` (`export_false_polygons`),
`scripts/positive_tile_creation.py` (+1-line `name_to_blob` fix),
`scripts/negative_tile_creation.py`, `scripts/generate_extra_tiles.py`,
`data/splits.py` (`assert_no_region_leakage`, `load_metadata_multiroot`),
`configs/v1_1_seed43.yaml` (config template).

**New:**
`scripts/mine_sample_polygons.py`, `scripts/build_mining_rating_page.py`,
`scripts/merge_mining_ratings.py`, `scripts/score_mining_ratings.py`, a delta training config,
four test files, a shared `CLAIMS.md`.

**Inputs:**
`/mnt/outputs/inference/south/products_local/south_rts_candidates.gpkg`, `.../qc_chips/rgb_chips.vrt`,
`domain/circumpolar_subregions.geojson`, `/mnt/outputs/v1.0/data_local/{metadata.csv,splits.yaml}`,
`tiles_2025q3_domain_full.csv`, `quad_index_2025q3.csv`.

---

## Verification (when executed later)

1. **Sampler:** confirm pool size, per-shard ≈300, underrepresented regions over-weighted vs their
   candidate share, already-rated ids excluded. Unit tests green.
2. **Rating pages:** open a shard from `file://`; rate with `1/2/3`, confirm autosave (reload resumes),
   per-region tally updates, optional `fp_category` recorded, EXPORT downloads the shard CSV; a second
   shard uses an independent localStorage key.
3. **Merge + score:** feed 2–3 shard CSVs (incl. a deliberate overlap) → merger → scorer; confirm
   seed counts, val/test-region exclusion, and the increment report (per-region before/after,
   ecoregions gained, balance metric, per-mode hard-negative counts).
4. **Tile production (one region first):** confirm 512×512 EPSG:3857 tiles, labels ∈ {0,1,255} with
   the ignore collar present, metadata rows have `year=2025` + correct `RegionName`, no `Tile_ID`
   collisions with v1.0.
5. **Loader smoke test:** point a config at the delta via `data.additional_roots`, run
   `scripts/check_data.py` over combined roots (normalization/collation), confirm
   `assert_no_region_leakage` passes after the `splits.yaml[train]` append.

---

## Open questions / decisions deferred to execution time

- **Deliberate overlap fraction** for inter-rater agreement (e.g. duplicate ~5% of items across two
  shards) — set when shards are cut.
- **Exact per-mode hard-negative quotas** — start at "a few hundred each," revisit after the first
  retrain shows which FP modes persist.
- **Whether to also mine confirmed positives as-is where the polygon only partially covers the RTS**
  (partial detections) — current plan relies on the ignore collar and keeps the schema at 3 verdicts;
  a `partial` verdict could be added later if partial-coverage label noise proves material.
- **Delta version label** (`v1.2_mined` is a placeholder).
