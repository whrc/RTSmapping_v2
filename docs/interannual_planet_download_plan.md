# Interannual PlanetScope acquisition, 2019–2024

**Status:** proposal, awaiting review by Heidi Rodenhizer
**Written:** 2026-08-14
**Asks of the reviewer:** the five questions in [§8](#8-open-questions-for-heidi). The quota
question (8.1) blocks everything past the pilot year.

---

## 1. Why

Pan-arctic RTS inference has been run once, on the **2025 q3** PlanetScope Global Quarterly
basemap: 41,567,572 tiles across 2,079 shards, completed 2026-07-10, yielding `south_rts.gpkg`
(10,984 polygons / 238.08 km²) and `south_rts_candidates.gpkg` (60,167 candidates).

That gives one epoch. RTS are, definitionally, a *change* process — the science question is how
fast they grow and where they initiate, and a single year cannot answer it. The next step is to
run the already-deployed model over **2019–2024 q3** on the same grid, giving a seven-epoch
series (2019–2025).

The model, the tiling, the normalization and the whole inference pipeline already exist and are
frozen. The only blocker is imagery: 2019–2024 is not in a usable place, and this repo has never
contained Planet acquisition code — imagery has always arrived pre-delivered into GCS from the
`circumpolar_planet_basemaps` workflow. Heidi holds `PL_BM_API_KEY` and `PDG_PL_ORDERS_KEY`, so
the orders can only be placed by her.

## 2. What we are asking for

Six years of **Global Quarterly q3**, filtered to the circumpolar-south domain, delivered so
that each year is byte-for-byte the same shape as the existing 2025 delivery:

| | Value |
|---|---|
| Years | 2019, 2020, 2021, 2022, 2023, 2024 (q3 only) |
| Domain | [`domain/circumpolar_south_domain.geojson`](../domain/circumpolar_south_domain.geojson) — unchanged since the 2025 run |
| Destination | `gs://pdg-planet-data/global_quarterly/<year>/q3/<col>/<row>/` |
| Order tool | `tools: [{"file_format": {"format": "COG"}}]`, exactly as 2025 |
| Resulting filename | `global_quarterly_<year>q3_mosaic_<col>-<row>_quad_file_format.tif` |
| Quads per year | **~309,100** (see §6.1 — this number needs your confirmation) |

The delivery layout is not a matter of taste. Our inference reads quads through
[`inference/quad_index.py`](../inference/quad_index.py), which matches on the literal suffix
`_quad_file_format.tif` (L36) and derives quad bounds arithmetically from the `<col>-<row>` id
(L46-50). If the new years land in that exact shape, per-year indexes need nothing but a new
`--prefix` argument and **no code in this repo changes at all**. If they land in the legacy
`_quad.tif` shape, every downstream step needs a compatibility branch.

## 3. Review of the three notebooks

Read against `HRodenhizer/circumpolar_planet_basemaps` @ `initial-download`. Line numbers are
from that branch as of 2026-08-14. These are observations, not instructions — several are
judgement calls that are yours to make.

### 3.1 The year is hardcoded in all three notebooks

| File | Line | Current |
|---|---|---|
| `1_basemap_grid_search.qmd` | 34 | `basemap_year = "2025"` |
| `2_circumpolar_south_basemap_grids.qmd` | 19 | `st_read("./data/circumpolar_basemap_grids_2025.geojson")` |
| `2_circumpolar_south_basemap_grids.qmd` | 101 | `"./data/circumpolar_south_planet_basemap_grids_2025.geojson"` (inside the commented `st_write`) |
| `3_order_basemaps.qmd` | 55 | `imagery_year = '2025'` |

Notebooks 1 and 3 already isolate this into a single variable and interpolate everywhere
downstream (`1` L87/L201, `3` L60/L82) — that pattern just needs extending to notebook 2, which
has the year written into two literal paths. Since these will be run six more times, a single
`year` variable per notebook is worth the five minutes.

### 3.2 `rename_data_files()` hardcodes 2025 in its regex — silent failure on other years

`3_order_basemaps.qmd` L67-74:

```python
def rename_data_files(old_path):
    new_path = re.sub(
        "global_quarterly_2025q3_mosaic/",
        "global_quarterly_2025q3_mosaic_",
        re.sub("/[a-z0-9-]{36}/", "/", old_path),
    )
    return new_path
```

Run this on a 2019 delivery and the inner `re.sub` still strips the order-UUID directory, but
the outer one no longer matches anything. The function returns a path that *looks* renamed, the
guard `renaming_guide.loc[old_name != new_name]` still passes because the UUID strip changed the
string, and the copy proceeds — producing objects our `_QUAD_NAME_RE` will not match. Nothing
raises. The failure surfaces days later as a short quad index.

Suggest interpolating `imagery_year` into both patterns. **This is the highest-risk item in the
review**, purely because it fails quietly rather than loudly.

Two related spots: the same literal appears at L332
(`.replace("global_quarterly_2025q3_mosaic_", "")` in the count-reconciliation cell), and your
own note at L64 already flags this function as needing rework — *"This needs to be updated to
just remove the order_id directory within the delivery directory."*

### 3.3 `NameError` in the batch-delete cell

`3_order_basemaps.qmd` L362-365:

```python
    batches = [
        original_files[i : i + batch_size]
        for i in range(0, len(remaining_original_files), batch_size)
    ]
```

`remaining_original_files` is never bound on this path — its only other occurrences (L367,
L374-375) are inside the commented-out re-run block directly below. On a first execution this
raises `NameError` before any delete happens. Suggest `len(original_files)`, which is what L363
already iterates over.

Benign in effect (it fails closed — nothing is deleted), but it stops the run at the very end of
a multi-day loop, which is the worst place to stop.

### 3.4 Notebook 2's domain path — correct as written

`2_circumpolar_south_basemap_grids.qmd` L22 reads
`"../RTSmapping_v2/domain/circumpolar_south_domain.geojson"`. That resolves correctly against a
sibling clone of `whrc/RTSmapping_v2`, which is this repo — no change needed. (Flagging it only
because our local working directory happens to be named `RTSmappingDL`, which makes the path
look stale when it isn't.)

The point worth stating is that
[`domain/circumpolar_south_domain.geojson`](../domain/circumpolar_south_domain.geojson) is
**unchanged since the 2025 run**, so all six new years get filtered against exactly the same
domain polygon as the epoch they will be compared against. That is what we want, and it is worth
not "improving" the domain mid-programme.

### 3.5 `st_write` in notebook 2 is commented out

L99-104. It produces the file notebook 3 reads at L82, so it needs uncommenting.

### 3.6 Not a defect — the dedupe makes this resumable

`3_order_basemaps.qmd` L122-127 scans `bucket.list_blobs(prefix=delivery_directory)` and
extracts `\d{1,4}-\d{1,4}` from filenames containing `quad`, then skips any quad already
delivered. This makes each year's run idempotent and safely interruptible, which matters a great
deal for a five-day loop that may hit a rate limit or a credential blip. Worth keeping as-is.

One consequence to be aware of: it keys on what is physically under the prefix, so it is only
correct if the rename pass (§3.2) has produced matchable names. A silently-failed rename would
make a resumed run re-order quads that are already present.

## 4. Per-year runbook

Steps 0–4 are yours; 5–7 are ours. Each has the check to run before moving on.

**0.** Set `year`; confirm `.env` carries `PL_BM_API_KEY` and `PDG_PL_ORDERS_KEY`.

**1.** Run notebook 1 → `circumpolar_basemap_grids_<year>.geojson`.
*Check:* the mosaic resolves to `global_quarterly_<year>q3_mosaic`; the quad count is in the
expected range; the `percent_covered` distribution has no unexpected mass at 0.

**2.** Run notebook 2 → `circumpolar_south_planet_basemap_grids_<year>.geojson`.
*Check:* row count against 2025's; the per-column count summary (L84-95); the plotted footprint
is circumpolar and not obviously holed.

**3.** Run notebook 3's order loop.
*Check:* confirm the first order returns `202` before walking away — L173-174 `sys.exit`s on a
400/401 **only at `index == 0`**; from L175 onward a 400/409/500 enters an unbounded 30-second
retry loop with no attempt ceiling, so a mid-run credential expiry looks identical to slow
progress. Then monitor by counting delivered quads under the prefix.

**4.** Run notebook 3's rename + delete passes.
*Check:* zero remaining paths matching the 36-char UUID directory pattern, and total object
count ≈ quad count × 6.

**5.** *(ours)* Build the quad index:
```bash
python scripts/build_quad_index.py --bucket pdg-planet-data \
    --prefix global_quarterly/<year>/q3/ \
    --output /mnt/outputs/inference/quad_index_<year>q3.csv
```
*Check:* row count ≈ step 2's grid count. A large shortfall here is the signature of a
silently-failed rename (§3.2) — this is the step that catches it.

**6.** *(ours)* `scripts/check_inference_normalization.py` against the training
`normalization_stats.json`, using the thresholds already fixed in `inference/inference.md` §5.4
(|Δmean| > 0.5σ_training, or |σ_sample/σ_training − 1| > 0.25), recorded per year.
Per that spec a trip means **pause and investigate**, and that gate stands. But note the
thresholds were written for a one-year 2024→2025 step; across a seven-year span some drift is
expected, and characterising it is part of the point rather than purely a failure signal. If
early years trip consistently, that is a finding to discuss, not a reason to silently widen the
threshold.

**7.** *(ours)* `generate_tile_grid.py` → `mask_tiles_to_domain.py` → `shard_tiles.py --output
gs://rts-mapping-v2-usw1/inference/<year>q3_south/`, then run the fleet.
*Check:* shard count in line with 2025's 2,079; `done/` markers reconcile against `index.json`.

## 5. Sequencing — and why your ordering and our inference overlap

### 5.1 Pilot first

**2022 runs end-to-end, alone, before anything else is ordered.** It is the gate on quota,
on the rename behaviour, and on radiometric drift. 2022 rather than 2019 because it is
mid-range: far enough from 2025 to exercise real drift, but not at the archive's early edge
where genuine Planet coverage gaps could be misread as pipeline bugs.

If the pilot is clean, the rest are ordered **2019 → 2020 → 2021 → 2023 → 2024**. 2019 goes
first because it gives the longest lever arm against 2025 if quota turns out to cut the
programme short.

### 5.2 The two halves share no resource

This is the part worth knowing on your side: **you do not need to wait for us, and we do not
need to wait for you.**

| | Your download | Our inference |
|---|---|---|
| Does the work | Planet's servers → GCS, server-side | our 8×A100 host |
| Our compute footprint | one single-threaded order loop, 1 of 96 vCPUs | 8 GPUs, `num_workers=16` |
| GCS traffic | ~31 MB/s of **writes** to `pdg-planet-data` | ~217 tiles/s of **reads** |
| State touched | `global_quarterly/<year>/q3/` | `inference/<year>q3_south/` |

`scripts/shard_tiles.py --output` takes a per-run base prefix, and everything (`shards/`,
`claims/`, `done/`, `probs/`, `logs/`) hangs beneath it, so each year is an independent run with
no shared state. Both traffic figures are orders of magnitude under any GCS bucket limit.

The rates are asymmetric in our favour — **~5.5 d/year to order, ~2.3 d/year to infer** — so
after the pilot, our inference on year *N* runs concurrently with your ordering of year *N+1*
and disappears entirely behind it. **Your ordering throughput sets the programme's wall clock.**

- fully sequential: 6 × (5.5 + 2.3) ≈ **47 days**
- pipelined: ~33 d of ordering + a 2.3 d inference tail ≈ **35 days**

### 5.3 What we are deliberately not doing

Your grid list is `arrange(year, grid_column, grid_row)`, so delivery sweeps west→east and
completed column bands are spatially contiguous — tempting to start inferring mid-year. We are
not doing that. Our tiles are 512×512 windowed reads at stride 344 that **straddle quad
boundaries**; a tile at the advancing edge that reads a not-yet-delivered quad is treated as
NoData rather than erroring, so it would silently produce degraded predictions. Doing it safely
needs a held-back one-quad buffer column plus appending to `shards/index.json`, which
`shard_tiles.py:104` writes once. That saves ~2 days out of 35 in exchange for a
silent-corruption failure mode.

## 6. Cost, time, and quota

All figures re-derived against live buckets on 2026-08-14. Method noted so you can check them.

### 6.1 Quads per year — a discrepancy we need you to settle

Your note at `3_order_basemaps.qmd` L88 says *"These orders will cover 259783 grids."* But the
2025 delivery actually indexed **309,100 unique quads**
(`wc -l /mnt/outputs/inference/quad_index_2025q3.csv` = 309,101 incl. header), across 1,951
column directories. That is ~19% more than the planning figure.

We do not know which is the right per-year expectation, and it propagates into every number
below. **All figures here use 309,100** as the conservative choice. The pilot settles it
empirically at step 2.

### 6.2 Storage

Measured average delivery size, summing three columns (500, 900, 1300) = 19,413,767,687 bytes
over 409 quads → **47.5 MB per quad** across all six delivered objects. (A single hand-picked
quad reads 52 MiB; that one is above average, which is why we sampled.)

| | |
|---|---|
| Per year | 309,100 × 47.5 MB ≈ **14.7 TB** |
| Six years | ≈ **88 TB** |
| Storage cost | ~**$1,650/month**, ~**$20k/year** (us-west1 standard, $0.020/GiB-month) |

Two levers if that becomes binding, neither recommended now: dropping provenance saves ~0.5%
(not worth it), and dropping udm2 saves ~6% — our inference path reads RGB + alpha from the quad
only and never touches udm2, but discarding it forecloses future cloud/quality masking for a
small gain. The real lever is a lifecycle rule that demotes a year to Nearline once its
inference has completed and been checked.

This draws on the budget line already reserved in `computing/infrastructure.md` §3 —
*"pan-arctic inference + EXTRA-channel generation + multi-year/ensemble runs, ~$40–55k"*. Note
the $70k credit expires **Sep 2026**, which bounds how long 88 TB can sit before it is on
someone's real budget.

### 6.3 Time

At your measured 39 grids/min: 309,100 ÷ 39 ≈ 7,926 min ≈ **5.5 days per year**, **~33 days**
for six. With inference pipelined (§5.2), ~35 days end-to-end.

### 6.4 Quota — the blocking question

Six full years at ~309,100 quads is **~1.85M quad downloads**. We have no visibility into
whether the basemap subscription has a quad, bandwidth, or cost cap that this would exhaust,
and it is not a thing to discover 60% of the way through. **We will not ask for anything past
the 2022 pilot until this is answered.**

## 7. Schedule and hardware

Indicative, assuming a clean pilot and no quota ceiling. Weeks are from the pilot start.

| Week | Ordering (yours) | Inference (ours) |
|---|---|---|
| 1 | 2022 | — |
| 2 | *(pilot gate: index, drift check, review)* | 2022 |
| 3–4 | 2019 | — |
| 4–5 | 2020 | 2019 |
| 5–6 | 2021 | 2020 |
| 6–7 | 2023 | 2021 |
| 7–8 | 2024 | 2023 |
| 8–9 | — | 2024 |

### The VM, stated plainly

We are dedicating our 8×A100 host (`a100-8x-train`, 96 vCPU) to this programme. To be accurate
about why: **the GPUs contribute nothing to the download.** Planet delivers server-side straight
into GCS — no imagery transits any machine we own, and the order loop is a single-threaded API
caller that would run just as fast on a laptop. The host earns its place for two other reasons:
it can hold a month-long order loop and the GCS-API-bound rename/copy passes at zero marginal
cost (it is already never stopped), and its otherwise-idle GPUs run the concurrent per-year
inference that makes the pipelined schedule in §5.2 work.

**Environment note:** notebook 2 is R/Quarto (`sf`, `tidyverse`, `viridis`); notebooks 1 and 3
are Python (`planet`, `geopandas`, `google-cloud-storage`, `google-cloud-storage-control`,
`gcsfs`, `python-dotenv`). None of these are in our `rts-train:v2` image, so whoever runs the
notebooks needs an environment separate from the inference container.

## 8. Open questions for Heidi

1. **Quota (blocking).** Does the basemap subscription have a quad, bandwidth, or cost cap that
   ~1.85M quad downloads would exhaust? Nothing past the 2022 pilot proceeds without this.
2. **Is 39 grids/min a Planet-side rate limit, or just the serial `requests.post` loop?** This
   is the single highest-leverage number in the plan — it sets the whole 35-day wall clock, and
   our inference can already absorb roughly double it. If the ceiling is the loop rather than
   the API, threading it would shorten everything.
3. **309,100 or 259,783 quads per year?** (§6.1.) Which is the right planning figure?
4. **Year-parameterization: your repo or ours?** Would you like the §3 fixes as a PR against
   `initial-download` (we would fork — we have read-only access), or would you rather apply
   them yourself, or just edit the year by hand for each of the six runs?
5. **Who runs each year's loop?** The key is yours, but if it would help, we can host the loop
   on our VM under credentials you control and hand you the monitoring.

---

## Appendix — how imagery flows today

```
circumpolar_planet_basemaps  (Heidi)
  nb1  Basemaps API  →  circumpolar_basemap_grids_<year>.geojson
  nb2  ∩ circumpolar_south_domain.geojson  →  circumpolar_south_planet_basemap_grids_<year>.geojson
  nb3  Orders API v2, one order per quad, delivery.google_cloud_storage
                                │
                                ▼  (server-side; never transits a VM)
        gs://pdg-planet-data/global_quarterly/<year>/q3/<col>/<row>/
                                │  nb3 rename pass strips the order-UUID directory
                                ▼
whrc/RTSmapping_v2
  scripts/build_quad_index.py   →  quad_index_<year>q3.csv
  scripts/generate_tile_grid.py →  stride-344 tile grid
  scripts/mask_tiles_to_domain.py, scripts/shard_tiles.py
                                ▼  512² windowed reads across quad boundaries
        gs://rts-mapping-v2-usw1/inference/<year>q3_south/probs/
```
