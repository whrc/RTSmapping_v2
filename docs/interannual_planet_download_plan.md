# Interannual PlanetScope acquisition, 2019–2024 — run instructions

**For:** Heidi Rodenhizer · **Status:** proposal, awaiting your review · **Written:** 2026-08-14

We are asking you to run your `circumpolar_planet_basemaps` notebooks six more times, once per
year for 2019–2024 q3. You own the Planet key and the account; we are specifying the delivery
shape because our inference pipeline consumes it. That split is the awkward part of this
request, so this document tries to be precise about exactly what to run and to keep our own
implementation out of your way — everything about what happens on our side is in
[§6–§8](#6-storage-and-the-delete-on-approval-policy), after the parts you need.

**Five questions for you in [§5](#5-questions-for-you).** The first one blocks everything past
the pilot year. If it would help, we are happy to run the loops ourselves on our VM under
credentials you control — see 5.5.

---

## 1. What we need

Six years of **Global Quarterly q3**, filtered to the circumpolar-south domain, delivered in
exactly the same shape as your existing 2025 delivery:

| | Value |
|---|---|
| Years | 2019, 2020, 2021, 2022, 2023, 2024 (q3 only) |
| Domain | `circumpolar_south_domain.geojson` — unchanged since the 2025 run |
| Destination | `gs://pdg-planet-data/global_quarterly/<year>/q3/<col>/<row>/` |
| Order tool | `tools: [{"file_format": {"format": "COG"}}]`, exactly as 2025 |
| Resulting filename | `global_quarterly_<year>q3_mosaic_<col>-<row>_quad_file_format.tif` |
| Quads per year | **~309,100** — see [5.3](#5-questions-for-you), this needs your confirmation |

The filename shape is the one thing worth not improvising on. Our index matches the literal
suffix `_quad_file_format.tif` and derives each quad's bounds from the `<col>-<row>` id, so if
the new years land in that shape, we change nothing on our side at all. If they land in the
legacy `_quad.tif` shape, every downstream step needs a compatibility branch.

## 2. Before you run: edits the notebooks need

Read against `HRodenhizer/circumpolar_planet_basemaps` @ `initial-download`, line numbers as of
2026-08-14. Items 2.1–2.3 must be done or the run breaks; 2.4 is context.

### 2.1 Un-hardcode the year (all three notebooks)

| File | Line | Currently |
|---|---|---|
| `1_basemap_grid_search.qmd` | 34 | `basemap_year = "2025"` |
| `2_circumpolar_south_basemap_grids.qmd` | 19 | `st_read("./data/circumpolar_basemap_grids_2025.geojson")` |
| `2_circumpolar_south_basemap_grids.qmd` | 101 | output path, inside the commented `st_write` |
| `3_order_basemaps.qmd` | 55 | `imagery_year = '2025'` |

Notebooks 1 and 3 already isolate this in one variable and interpolate downstream (`1` L87/L201,
`3` L60/L82). Notebook 2 has the year written into two literal paths and needs the same
treatment.

### 2.2 `rename_data_files()` hardcodes 2025 in its regex — **this one fails silently**

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

On a 2019 delivery the inner `re.sub` still strips the order-UUID directory, so the guard
`renaming_guide.loc[old_name != new_name]` still passes and the copy proceeds — but the outer
substitution matches nothing. You get objects that look renamed and are not, with nothing
raising. It surfaces days later as a short quad index.

Interpolating `imagery_year` into both patterns fixes it. The same literal also appears at L332
in the count-reconciliation cell. Your own note at L64 already flags this function as needing
rework.

### 2.3 `NameError` in the batch-delete cell

`3_order_basemaps.qmd` L362-365 iterates `range(0, len(remaining_original_files), batch_size)`,
but that name is only bound inside the commented-out re-run block below (L367, L374-375). First
execution raises. `len(original_files)` is what L363 already slices.

It fails closed — nothing gets deleted — but it stops you at the very end of a multi-day run.

### 2.4 Two things that are fine as-is

**The domain path is correct.** L22's `"../RTSmapping_v2/domain/circumpolar_south_domain.geojson"`
resolves against a sibling clone of our repo. (Our local working directory happens to be named
`RTSmappingDL`, which makes it look stale when it isn't.) Worth saying that file is unchanged
since the 2025 run, so all six years get filtered against the same polygon as the epoch they
will be compared against — worth not "improving" mid-programme.

**The dedupe makes runs resumable.** L122-127 skips quads already present under the prefix, so a
five-day loop can be interrupted and restarted freely. One caveat: it keys on what is physically
under the prefix, so it is only correct once 2.2 is fixed — a silently-failed rename would make
a resumed run re-order quads that are already there.

Also uncomment `st_write` in notebook 2 (L99-104); it produces the file notebook 3 reads at L82.

## 3. Step by step, for one year

**0.** Set the year; confirm `.env` has `PL_BM_API_KEY` and `PDG_PL_ORDERS_KEY`.

**1.** Run `1_basemap_grid_search.qmd` → `circumpolar_basemap_grids_<year>.geojson`.
*Check:* the mosaic resolves to `global_quarterly_<year>q3_mosaic`; the quad count is in range;
`percent_covered` has no unexpected mass at 0.

**2.** Run `2_circumpolar_south_basemap_grids.qmd` → `circumpolar_south_planet_basemap_grids_<year>.geojson`.
*Check:* row count against 2025's; the per-column summary at L84-95; the plotted footprint is
circumpolar and not obviously holed. **Tell us this row count** — it is the number we reconcile
against in step 5.

**3.** Run the order loop in `3_order_basemaps.qmd`. Expect ~5.5 days.
*Check:* confirm the first order returns `202` before walking away. L173-174 `sys.exit`s on a
400/401 **only at `index == 0`**; from L175 on, a 400/409/500 enters an unbounded 30-second retry
with no attempt ceiling — so a mid-run credential expiry looks exactly like slow progress. Worth
glancing at the delivered-quad count once a day rather than trusting silence.

**4.** Run the rename + delete passes in the same notebook.
*Check:* no paths left matching the 36-char UUID directory pattern, and total object count ≈
quad count × 6.

**5.** Tell us the year is ready. Everything after this is ours ([§8](#8-what-happens-on-our-side)).

## 4. Which years, in what order

**2022 runs alone, first, end to end** — including our inference and a look at the result —
before anything else is ordered. It is the gate on quota, on the 2.2 rename fix, and on
radiometric drift. 2022 rather than 2019 because it is mid-range: far enough from 2025 to
exercise real drift, but not at the archive's early edge where genuine Planet coverage gaps
could be misread as pipeline bugs.

If the pilot is clean: **2019 → 2020 → 2021 → 2023 → 2024**, back to back. 2019 goes first
because it gives the longest lever arm against 2025 if quota turns out to cut the programme
short.

Note that 2024 is a full order like the rest. `gs://abrupt_thaw` has a 2024 q3 tree that looks
substantial by directory count, but it holds only ~8.8% of 2025's quad density inside the
columns it covers, and 296 of 2025's columns are missing from it entirely — it is an ARTS-site
subset, not a mapping layer. Same for the 2019/2021/2023 trees there.

## 5. Questions for you

**5.1 Quota — blocking.** Six full years is **~1.85M quad downloads**. Does the basemap
subscription have a quad, bandwidth, or cost cap that this would exhaust? We would rather know
now than 60% of the way through. Nothing past the 2022 pilot proceeds without this.

**5.2 Is 39 grids/min a Planet-side rate limit, or just the serial `requests.post` loop?** This
is the highest-leverage unknown in the whole plan — it sets the entire wall clock, and our
inference can already absorb roughly double it. If the ceiling is the loop rather than the API,
threading it would shorten everything proportionally.

**5.3 309,100 or 259,783 quads per year?** Your note at `3_order_basemaps.qmd` L88 says the
orders "will cover 259783 grids", but the 2025 delivery actually indexed **309,100** unique
quads across 1,951 column directories — ~19% more. We do not know which is the right planning
figure; every estimate here uses the larger one. Step 2 of the pilot settles it.

**5.4 Notebook edits — your repo or ours?** Would you like §2 as a PR against `initial-download`
(we would fork; we have read-only access), would you rather apply them yourself, or is editing
the year by hand for each of six runs simplest?

**5.5 Who runs the loops?** The key is yours. But six runs at ~5.5 days each is a real
imposition, and if it helps we can host the loops on our VM under credentials you control and
hand you monitoring — you would keep the account and the key, and we would absorb the babysitting.

---

*Everything below is our side of the pipeline, recorded so the plan is auditable. None of it
needs action from you.*

## 6. Storage and the delete-on-approval policy

Measured average delivery size: three columns (500, 900, 1300) sum to 19,413,767,687 bytes over
409 quads → **47.5 MB per quad** across all six delivered objects. (A single hand-picked quad
reads 52 MiB; that one is above average, which is why we sampled.)

| | |
|---|---|
| Per year | 309,100 × 47.5 MB ≈ **14.7 TB** |
| Six years, all resident | ≈ **88 TB** |

**The imagery is not kept.** Once a year's inference map is produced and checked, the source
quads are deleted. Assuming ~2 months of review per year plus ordering and inference time —
call it 2.5 months' residency — the whole programme costs about **$4,100 in storage**, against
roughly **$20k/year** if all six years were simply held. Peak concurrent footprint is still
~88 TB for about a month, when the last year has been delivered and the first has not yet
cleared review.

What we actually keep is the probability COGs — ~0.3 TB per year at `scaled_uint8`, so ~1.8 TB
for the whole series. The bulky part is the source imagery, and it is re-orderable from Planet
if we ever need it again.

Two caveats on the deletion gate. It should be **map approved**, not map produced — re-ordering
a deleted year costs quota a second time, so deleting on anything weaker than a human sign-off
trades a cheap disk bill for an expensive re-download. And the imagery is needed *during* review,
not just during inference: the QC tooling streams RGB crops built from these quads, so the
quads must stay on Standard storage through the review window rather than being demoted to
Nearline to save the difference.

This draws on the budget already reserved in `computing/infrastructure.md` §3 —
*"pan-arctic inference + EXTRA-channel generation + multi-year/ensemble runs, ~$40–55k"*. The
$70k credit expires Sep 2026, which is a further argument for not letting years accumulate.

## 7. Schedule — your ordering and our inference overlap

You do not need to wait for us, and we do not need to wait for you. The two halves share no
resource:

| | Your ordering | Our inference |
|---|---|---|
| Does the work | Planet's servers → GCS, server-side | our 8×A100 host |
| Our compute footprint | one single-threaded order loop, 1 of 96 vCPUs | 8 GPUs, `num_workers=16` |
| GCS traffic | ~31 MB/s of **writes** | ~217 tiles/s of **reads** |
| State touched | `global_quarterly/<year>/q3/` | `inference/<year>q3_south/` |

`scripts/shard_tiles.py --output` takes a per-run base prefix and everything (`shards/`,
`claims/`, `done/`, `probs/`, `logs/`) hangs beneath it, so each year is an independent run with
no shared state. Both traffic figures are far under any GCS bucket limit.

Rates are asymmetric in our favour — **~5.5 d/year to order, ~2.3 d/year to infer** — so after
the pilot, our inference on year *N* runs concurrently with your ordering of year *N+1* and
disappears entirely behind it. **Your ordering throughput sets the programme's wall clock:**
~33 days of ordering plus a 2.3-day tail, ≈ **35 days**, against ~47 if the two were serialised.

| Week | Ordering (yours) | Inference (ours) |
|---|---|---|
| 1 | 2022 | — |
| 2 | *pilot gate: index, drift check, review* | 2022 |
| 3–4 | 2019 | — |
| 4–5 | 2020 | 2019 |
| 5–6 | 2021 | 2020 |
| 6–7 | 2023 | 2021 |
| 7–8 | 2024 | 2023 |
| 8–9 | — | 2024 |

Review windows then run ~2 months behind each inference, and each year's quads are deleted as it
clears (§6).

**What we are deliberately not doing:** your grid list is `arrange(year, grid_column, grid_row)`,
so delivery sweeps west→east and completed column bands are contiguous — tempting to start
inferring mid-year. We are not. Our tiles are 512×512 windowed reads at stride 344 that straddle
quad boundaries; a tile at the advancing edge reading a not-yet-delivered quad is treated as
NoData rather than erroring, so it would silently produce degraded predictions. Doing it safely
needs a held-back one-quad buffer column plus appending to `shards/index.json`, which
`shard_tiles.py:104` writes once. That saves ~2 days out of 35 for a silent-corruption failure
mode.

## 8. What happens on our side

After your step 4, per year:

**5.** Build the quad index:
```bash
python scripts/build_quad_index.py --bucket pdg-planet-data \
    --prefix global_quarterly/<year>/q3/ \
    --output /mnt/outputs/inference/quad_index_<year>q3.csv
```
Row count is reconciled against your step-2 grid count. A large shortfall is the signature of a
silently-failed rename (§2.2) — this is the step that catches it.

**6.** `scripts/check_inference_normalization.py` against the training `normalization_stats.json`,
using the thresholds fixed in `inference/inference.md` §5.4 (|Δmean| > 0.5σ_training, or
|σ_sample/σ_training − 1| > 0.25), recorded per year. Per that spec a trip means pause and
investigate, and that gate stands. Note the thresholds were written for a one-year 2024→2025
step; across a seven-year span some drift is expected, and characterising it is part of the
point rather than purely a failure signal. If early years trip consistently that is a finding to
discuss, not a reason to quietly widen the threshold.

**7.** `generate_tile_grid.py` → `mask_tiles_to_domain.py` → `shard_tiles.py --output
gs://rts-mapping-v2-usw1/inference/<year>q3_south/`, then run the fleet. Shard count should be
in line with 2025's 2,079; `done/` markers reconcile against `index.json`.

**8.** Review the map, then delete that year's quads (§6).

### The VM, stated plainly

We are dedicating our 8×A100 host (`a100-8x-train`, 96 vCPU) to this programme, but to be
accurate about why: **the GPUs contribute nothing to the download.** Planet delivers server-side
straight into GCS — no imagery transits any machine we own, and the order loop is a
single-threaded API caller that would run just as fast on a laptop. The host earns its place
because it can hold a month-long order loop and the GCS-API-bound rename passes at zero marginal
cost (it is already never stopped), and because its otherwise-idle GPUs run the concurrent
per-year inference that makes §7's schedule work.

**Environment:** notebook 2 is R/Quarto (`sf`, `tidyverse`, `viridis`); notebooks 1 and 3 are
Python (`planet`, `geopandas`, `google-cloud-storage`, `google-cloud-storage-control`, `gcsfs`,
`python-dotenv`). None are in our `rts-train:v2` image, so the notebooks need an environment
separate from the inference container — relevant to question 5.5.

---

## Appendix — the flow end to end

```
circumpolar_planet_basemaps  (yours)
  nb1  Basemaps API  →  circumpolar_basemap_grids_<year>.geojson
  nb2  ∩ circumpolar_south_domain.geojson  →  circumpolar_south_planet_basemap_grids_<year>.geojson
  nb3  Orders API v2, one order per quad, delivery.google_cloud_storage
                                │
                                ▼  (server-side; never transits a VM)
        gs://pdg-planet-data/global_quarterly/<year>/q3/<col>/<row>/
                                │  nb3 rename pass strips the order-UUID directory
                                ▼
whrc/RTSmapping_v2  (ours)
  scripts/build_quad_index.py   →  quad_index_<year>q3.csv
  scripts/generate_tile_grid.py →  stride-344 tile grid
  scripts/mask_tiles_to_domain.py, scripts/shard_tiles.py
                                ▼  512² windowed reads across quad boundaries
        gs://rts-mapping-v2-usw1/inference/<year>q3_south/probs/
                                ▼  review, then delete that year's quads
```

All figures re-derived against live buckets on 2026-08-14; sampling method stated inline so they
can be checked.
