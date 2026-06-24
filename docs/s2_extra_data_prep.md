# Sentinel-2 Download & EXTRA-Channel Data Preparation — Design Doc

## 1. Why this doc
We are about to acquire and process the Sentinel-2 (S2) imagery that two models depend on, and the
choices made now (extent, CRS, bands, bucket, grid) are expensive to redo once terabytes are on disk.
This doc proposes *how* we download and preprocess that data so reviewers can catch problems before we
spend storage budget. **Yili will run the downloading/processing scripts**

Two models drive the requirements:

1. **Planet model** — the existing RTS segmentation model (PlanetScope RGB + auxiliary "EXTRA"
   channels), trained on 2024, deployed on **2025** PlanetScope basemaps over the pan-Arctic Planet
   domain (~60–74°N). It needs **2025 inference EXTRA channels** generated identically to training.

2. **Pure-S2 model** *(new)* — an RTS model that runs on Sentinel-2 directly, to cover
   **74–84°N (ARTS North)** where **no PlanetScope basemap exists**. Trained on **2024** S2 against the
   current ARTS labels; deployed on **2025** S2.

---

## 2. The data matrix (what we need)

Two regions × two years, with different jobs in each cell:

```
                 │  2024  (TRAINING-side)                │  2025  (INFERENCE-side)
─────────────────┼───────────────────────────────────────┼───────────────────────────────────────
 ARTS NORTH      │  ✅ HAVE  (in EPSG:3413 — re-export)    │  ⬇ DOWNLOAD
 74–84°N         │  → pure-S2 model TRAINING              │  → pure-S2 model INFERENCE
 (no Planet)     │                                       │
─────────────────┼───────────────────────────────────────┼───────────────────────────────────────
 ARTS SOUTH      │  ⬇ DOWNLOAD (South)                    │  ⬇ DOWNLOAD (full Planet coverage)
 (Planet domain) │  → EXTRA-for-training source          │  → Planet-model EXTRA INFERENCE
                 │  → pure-S2 model TRAINING             │  → pure-S2 model INFERENCE
```

### Data flow

```
                         ┌──────────────────────────────────────────────┐
   GEE: S2_SR_HARMONIZED │  Bulk S2 median composites (per 1°×3° tile)   │
   + QA60 mask, summer   │  → GCS, COG, EPSG:3857   (§3)                 │
   median composite      └───────────────┬──────────────────────────────┘
                                          │
                 ┌────────────────────────┴───────────────────────────┐
                 ▼                                                      ▼
   ┌──────────────────────────────┐                  ┌──────────────────────────────────┐
   │ Pure-S2 model tiles (§5)     │                  │ Planet EXTRA channels (§4)        │
   │ 512×512, EPSG:3857           │                  │ per Planet-RGB footprint, EPSG:3857│
   │ 2024+labels → train          │                  │ via generate_extra_tiles.py        │
   │ 2025 → inference             │                  │ (reuses GEE directly; see §4)      │
   └──────────────────────────────┘                  └──────────────────────────────────┘
```

> Note: the EXTRA pipeline (§4) queries GEE per Planet-RGB footprint directly; it does **not** consume
> the bulk S2 composites of §3. The bulk composites of §3 are for the **pure-S2 model** (and as an
> on-disk S2 reference). They stay consistent because **both tracks import the same recipe** from
> [`data/extra_channels.py`](../data/extra_channels.py): the Track-1 export script imports those
> constants rather than re-specifying them, exactly as the already-shipped Track-2 generator
> (`generate_extra_tiles.py`) does. Same source, same recipe → consistent results.

---

## 3. Track 1 — Bulk Sentinel-2 composite export

**Approach.** Port our two existing Colab/GEE notebooks (grid generation + gridded S2 export) into
resumable, non-Colab scripts that run on the VM (§7), changing the export CRS to **EPSG:3857**.

- **Grid:** clean **1°×3°** latitude×longitude grid, land-filtered against LSIB
  (`USDOS/LSIB_SIMPLE/2017`), the same GENERATED approach already used for ARTS North. We **drop** the
  earlier "aggregate the Planet fine grid" path — it produced oversized cells (1.3°×3.9°) and its
  edge-merge was an unimplemented stub.
- **Compositing recipe** — **imported from the SSoT, not restated.** The bulk export uses the exact
  recipe constants in [`data/extra_channels.py`](../data/extra_channels.py) §"Sentinel-2 acquisition"
  (`S2_COLLECTION = COPERNICUS/S2_SR_HARMONIZED`, `S2_WINDOW = Jul-01 → Sep-30`, `S2_CLOUD_PCT = 20`
  i.e. `CLOUDY_PIXEL_PERCENTAGE < 20`, QA60 cloud+cirrus mask on bits 10/11, **median** composite over
  `S2_BANDS`), exported via `Export.image.toCloudStorage` as deflate-compressed Cloud-Optimized
  GeoTIFF. Importing these constants — rather than re-specifying them here — is what makes the Track-1
  / Track-2 consistency claim (§2) true (CLAUDE Rule 5 SSoT, Rule 3 shared preprocessing).
- **CRS:** **EPSG:3857 everywhere** (project standard). For the Planet model this is *forced* — its
  tiles must co-register with the Planet basemaps. For the **pure-S2 North model (74–84°N)** it is
  not structurally forced (no Planet to co-register against), so adopting 3857 there is a **conscious
  call**: we accept the ~5–6× Web Mercator scale distortion at those latitudes in exchange for a
  single pipeline shared with the South. *(Alternative — EPSG:3413 for the North — was considered and
  rejected for pipeline simplicity; recorded as decision §6.6.)*
- **Bands:** **TBD — placeholder pending the channel-selection experiments.** We will lock the band
  list after those results. Expected to include at least RGB + NIR + NDVI; SWIR (B11/B12) added only if
  NBR / Tasseled-Cap make the final EXTRA list.
- **Resumability:** skip tiles already present in the GCS folder and tasks already active in the GEE
  queue; back off and retry on "Too many tasks".

**Runs:**

First a *test run on a tiny region* (claude code can decide extent and size) to visually check imagery quality.

If the visual check passed, download runs:

| Run | Region/Year | Notes |
|---|---|---|
| `s2_2024_south` | ARTS South 2024 | training-side; extent per §6.1 |
| `s2_2025_south` | ARTS South 2025 | inference-side; **full Planet coverage** |
| `s2_2025_north` | ARTS North 2025 | inference-side |
| `s2_2024_north_reexport` | ARTS North 2024 | re-export existing 3413 tiles to EPSG:3857 (§6.3) |

---

## 4. Track 2 — Planet-model inference EXTRA channels

> **Implementation update (2026-06-24).** Benchmarking killed the per-tile route at inference scale:
> `generate_extra_tiles.py` runs at **~2.5 tiles/s** (64 workers) and ~1.1 MB/tile → **~192 VM-days and
> ~45 TB** for all 41.57M tiles. The 45 TB is pure redundancy (33% tile overlap × millions of tiny
> poorly-compressing float files), not information. **New plan: derive NDVI from the bulk S2 composites
> (Track 1) and window it on-the-fly at inference** — exactly how inference already reads RGB from the
> Planet quads. No per-tile GEE, no 45 TB. The CSV-bbox generator change below still stands and is the
> right tool for **training** EXTRA / small AOIs, just not the 41.57M inference set. Follow-up (inference
> component): add an "NDVI from S2-composite window" reader alongside the RGB-quad reader.

**Reuse, don't rebuild.** `scripts/generate_extra_tiles.py` + `data/extra_channels.py` (already merged)
produce the per-tile EXTRA stack in EPSG:3857 from each PLANET-RGB footprint, querying GEE
(Sentinel-2 + Google Satellite Embedding). It is resumable, multi-threaded, and parameterized by
`--year / --groups / --rgb-dir / --out-dir` — **it was written to be exactly this inference path**
("2024 training and 2025 inference tiles produced identically").

For 2025 inference we drive it with `--year 2025` over the **2025 domain tile grid** — the same
`tiles_2025q3_domain_full.csv` the inference pipeline already produces (tile IDs + per-tile bounding
boxes, inference.md §3.2):

```
python scripts/generate_extra_tiles.py --groups <s2|all> --year 2025 \
   --metadata <tiles_2025q3_domain_full.csv> \
   --out-dir  <.../EXTRA_2025> [--se-artifacts se_artifacts.npz] --workers N
```

> The generator opens each `--rgb-dir` GeoTIFF only to read `src.bounds`
> ([generate_extra_tiles.py](../scripts/generate_extra_tiles.py) L113–114) — it never reads RGB
> pixels. Since the inference tile-grid CSV already carries those bounds, we feed bounds straight from
> the CSV instead of requiring on-disk RGB tiles. This is a **small generator change** (read the bbox
> from the CSV row rather than from an RGB file), not a new ingest.

- **Gated on the final EXTRA channel list** (channel-selection experiments are still running):
  - NDVI / S2-derived only → `--groups s2`, **no AlphaEarth needed**.
  - If Satellite-Embedding (SE) channels make the cut → also need `se_artifacts.npz` +
    AlphaEarth 2025 availability → `--groups all`.
- **Prerequisite (ours):** only the **2025 domain tile grid** (`tiles_2025q3_domain_full.csv`), which
  the inference pipeline already produces. There is **no separate per-tile Planet-RGB ingest** —
  inference reads the downloaded quads as on-the-fly windowed crops (inference.md §1, never cutting
  RGB tiles to disk), and the generator needs only the per-tile bounds the CSV already carries
  (see the note above and §6.5).
- **Scale:** ~3.4M tiles is the cost driver. The runner is thread-pooled and resumable; we shard the
  tile list across the VM's cores (and, if needed, across multiple VMs).

---

## 5. Track 3 — Pure-S2 model data prep (prep only)

*(Building/training the model itself is out of scope; this is just its data.)*

First, *a human visual check* to confirm the labels delineated on 2024 Planet aligns with 2024 Sentinel2. And 
proceed to the next step after confirmation with the desired label matchness.

- **Training tiles:** co-register **S2-2024** (North + South) with the **current ARTS labels** into
  512×512 EPSG:3857 image/label tiles, reusing the tiling logic in
  `scripts/positive_tile_creation.py` / `scripts/negative_tile_creation.py`. Label convention
  unchanged: 0 = background, 1 = RTS, 255 = ignore. Reuse is **uneven**, not a clean source swap:
  - `positive_tile_creation.py` is a near drop-in — it reads any 3-band EPSG:3857 GeoTIFF from
    `INPUT_PREFIX`, with no Planet-specific assumptions; point it at the S2 composites.
  - `negative_tile_creation.py` is **not** a clean swap — its `grid_row_to_blob()` is coupled to the
    Planet quad path scheme (`delivery_location` / `basemap_name` / `grid_column` / `grid_row`). The
    negative path needs its grid-metadata layer reworked for the S2 grid, not just a source swap.
- **Inference tiles:** tile **S2-2025** (North + South) to 512×512.
- **Normalization:** per-dataset stats via `scripts/compute_normalization_stats.py` for the S2 bands.

---

## 6. Open decisions (please weigh in)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 6.1 | **2024 South extent** | label-bearing Planet South only *vs* full Planet South | Label regions only (cheaper; 2024 is training-side) |
| 6.2 | **Final S2 bands** | — | **Placeholder — lock after channel-selection experiments** |
| 6.3 | **North-2024 CRS fix** | re-export in 3857 *vs* reproject existing 3413 tiles | Re-export (avoids resampling artifacts; cheap via GEE) |
| 6.4 | **Bucket** | keep `gs://pdg-storage-default/sentinel2/…` *vs* co-locate in PDG `gs://rts-mapping-v2` (VM region) | **DECIDED 2026-06-24:** created **`gs://rts-mapping-v2-usw1`** (us-west1 *single-region*) — refines the original (multi-region `rts-mapping-v2`) so the us-west1 inference fleet reads egress-free. S2 imagery under `S2_RGB/<job>/`. |
| 6.5 | **Track-2 footprint source** | materialize 41.5M per-tile Planet-RGB GeoTIFFs to disk *vs* drive EXTRA generation from the existing 2025 inference tile-grid CSV bboxes | Drive from `tiles_2025q3_domain_full.csv` — the generator needs only bounds, which the CSV already has. Avoids a phantom multi-TB RGB-tile ingest; costs only a small generator change |
| 6.6 | **Pure-S2 North model CRS** (74–84°N) | EPSG:3857 (shared pipeline, ~5–6× distortion) *vs* EPSG:3413 (polar-stereographic, low distortion, second pipeline) | EPSG:3857 — single pipeline shared with the South. Unlike the Planet model, 3857 is *not* forced here (no Planet to co-register against), so this is a deliberate accept of the distortion, sign off explicitly |

---

## 7. Compute

- **Bulk composite export needs no VM.** The GEE `Export.image.toCloudStorage` jobs run **server-side**
  and write COGs straight to GCS; the launcher (`scripts/export_s2_composites.py`) only submits tasks, so
  it runs in a container on the existing **`a100-8x-train` control node** (`gcloud`-driven, no new VM).
  **Run status (2026-06-24):** launched 2025 South (1799 cells) + 2025 North (272) + 2024 train footprint
  (1063, label-region §6.1) → `gs://rts-mapping-v2-usw1/S2_RGB/`; ~5–6 TB total; resumable.
- **A CPU VM is only needed later for *local* post-processing** (S2-RGB model tiling / any local NDVI
  windowing) — provision a fresh **`rts-`-prefixed** Spot CPU VM in us-west1 then, per
  [computing/infrastructure.md](../computing/infrastructure.md). (The pre-existing `download-vm` is **not
  ours** — don't use it.)

---

## 8. Out of scope

Building/training either model; running Planet or S2 inference; the 2025 Planet basemap download and
the inference tile-grid generation (both done/owned by the inference pipeline — Track 2 just reuses
the resulting `tiles_2025q3_domain_full.csv`, §6.5); rebuilding AlphaEarth artifacts unless SE
channels make the final EXTRA list. All of these are ours to do later — this doc covers only the S2
download + EXTRA/S2-model data preparation design.
