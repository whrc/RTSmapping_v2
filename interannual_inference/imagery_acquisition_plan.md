# Interannual imagery acquisition — routes and mitigations

**For team discussion. Prepared 2026-08-26. One decision is open: which route we take for
the Sentinel-2 download.**

---

## 1. What we are trying to do

The delivered 2025 pan-Arctic RTS map is a single epoch. To turn it into an interannual
series we run the **frozen** deployed model over 2019–2024. Each year needs two
co-registered inputs:

- **PlanetScope q3 basemaps** — the RGB imagery the model segments (Heidi's acquisition).
- **A Sentinel-2 Jul–Sep median composite** — supplies the NDVI channel, derived on the
  fly at inference from B4/B8.

The model is frozen: every year runs the same weights that produced the delivered map.
The imagery is the only thing that varies.

**Planet is healthy.** 2019 is at 30.8%, 38.1 orders/min, **zero failures**, ETA ~3.9 days.
The remaining years are ~26 days serial. This is a hard floor on the campaign, and it
matters below because it absorbs the latency of every Sentinel-2 option.

**Sentinel-2 is blocked, for two unrelated reasons.**

---

## 2. Problem one: we are out of Earth Engine compute

Earth Engine meters processing in **EECU-hours** (Earth Engine Compute Units — one managed
worker running for one hour, like CPU-hours). Noncommercial projects get a fixed monthly
grant.

We read the real limit off the quota API rather than inferring it from symptoms:

| project | monthly grant | tier |
|---|---|---|
| `abruptthawmapping` | 1,000 EECU-hours | Contributor |
| `pdg-project-406720` | 150 EECU-hours | Community |

Our export costs **~3.5 EECU-hours per grid cell**, measured across 195 completed tasks.
A year is 1,799 cells.

| | EECU-hours | against a 1,000/month grant |
|---|---|---|
| one year | 6,374 | **6.4 months** |
| six years | 38,244 | **38 months** |

We spent 69% of a month's grant in the first 48 hours and have been throttled in
*restricted mode* since — where tasks still run, just slowly, which is why it looked like a
mysterious slowdown rather than an error.

This also retires a long-standing puzzle: pdg's batch exports "never starting" was never
about registration state. At 150 EECU-hours ≈ 42 cells/month, against a 1,799-cell year, it
was simply out of compute.

**Things that do not fix this:** more Cloud projects (each carries only its own small
grant), and more Google accounts (they add concurrency but draw on the *same* project
grant — which is precisely how we exhausted it twice as fast).

---

## 3. Problem two: our 2022 and 2023 composites have no cloud mask

The recipe masks cloud using Sentinel-2's QA60 band. **ESA stopped populating QA60 at
processing baseline 04.00.** When QA60 is all zeros, the bit tests pass everywhere and the
mask silently becomes a no-op — no error, just an unmasked median.

Maximum QA60 value over a Jul–Sep, cloud < 20 collection:

| site | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Alaska N. Slope | 2048 | 2048 | 2048 | **EMPTY** | **EMPTY** | 1024 | 2048 |
| Canada, Banks Is. | 2048 | 1024 | 2048 | **EMPTY** | **EMPTY** | 2048 | 2048 |
| W. Siberia, Yamal | 2048 | 2048 | 2048 | **EMPTY** | **EMPTY** | 2048 | 2048 |
| E. Siberia, Lena | 2048 | 2048 | 2048 | **EMPTY** | 1024 | 2048 | 2048 |

Confirmed against the actual product, not just the metadata. Rebuilding cell
`W1590_N0710` (71°N) for 2022 from an independent source, on the EE product's own grid:

| reconstruction | difference vs our export | MAE | corr |
|---|---|---|---|
| *with* a cloud mask | −0.0804 NDVI | 0.1082 | 0.9347 |
| ***without* any cloud mask** | **−0.0002** | **0.0016** | **0.9994** |

Our 2022 product *is* the unmasked composite, to 0.0016 NDVI.

**Why this matters more than it first appears.** It is not a uniform bias that cancels in a
difference. It is a **mask discontinuity placed in the middle of the study period** —
2019–2021 and 2024–2025 masked, 2022–2023 not. Any NDVI trend computed across that boundary
carries a step that is an artefact of ESA's processing history, not of the landscape, in
the one channel the interannual comparison depends on. At the cell measured the step is
~0.08 mean NDVI with a p95 of 0.55.

The 165 cells of 2022 already exported are affected and will need redoing.

---

## 4. What is already decided

| Question | Decision |
|---|---|
| **Replacement cloud mask** | **Optimise to reproduce QA60.** Pick whichever of SCL / s2cloudless best reproduces QA60-masked NDVI in years where QA60 works, then apply it to **2022–2023 only**. This keeps all seven epochs on one effective basis, stays consistent with the QA60-masked data the model was trained on, and needs no re-export of the good years. |
| **Compute efficiency** | **Do not touch the recipe.** Two levers were identified — computing only B4+B8 (provably NDVI-identical, since the median is per-band) and exporting at a coarser scale (currently ~3.75 ground metres at 68°N, finer than Sentinel-2's true 10 m). Both are declined; the composite stays byte-for-byte as the delivered 2025 map was made. **Consequence: the quota wall is a route problem, not an optimisation problem.** |
| **Sentinel-2 source** | **Open. This is the decision to make.** |

---

## 5. The decision: routes for the Sentinel-2 download

Every route produces the **same product** — same recipe, EPSG:3857, ~15 TiB/year. Storage
(~90 TiB, ~$1.8k/month) is identical across routes and is not a discriminator.

With the recipe fixed, the remaining need is **10,574 cells ≈ 37,000 EECU-hours**.

### A · Earth Engine + Partner Tier

| | |
|---|---|
| **Throughput** | 100,000 EECU-hours/month → the whole remaining campaign is **37% of one month** |
| **Time to start** | **Several weeks** — manual review, and it **may be refused** |
| **Cost** | Free |
| **Work needed** | None. Application only; the pipeline exists and is running |
| **Fidelity risk** | **None** — same code path that produced the delivered map |
| **Depends on** | Google's approval, judged on demonstrable impact |

Eligibility looks strong: nonprofit research institute, climate and permafrost work,
genuinely substantial compute. A draft application exists; it needs our institutional and
impact detail. **Submitting costs nothing and forecloses nothing** — it should go regardless
of which route we ultimately choose.

### B · Earth Engine + Contributor Tier (status quo)

| | |
|---|---|
| **Throughput** | ~282 cells/month → **~38 months** |
| **Time to start** | Zero — this is what is running now |
| **Cost** | Free |
| **Fidelity risk** | None |

**Not a viable route.** Listed so the baseline is explicit: even the 2022 pilot alone is
~5 months away, with eight A100s idle meanwhile.

### C · Earth Search / S3 (Element 84 public COGs)

| | |
|---|---|
| **Throughput** | Measured **134 MB/s** at 64 threads and still climbing → ~89 TB for six years ≈ **8 days of pulling**, plus compositing |
| **Time to start** | Days-to-weeks of build, then deterministic |
| **Cost** | VM time only — the data is AWS Open Data, egress free |
| **Work needed** | **Substantial** |
| **Fidelity risk** | Medium, partly measured |
| **Depends on** | **Nobody's approval.** The only route with no external gate |

We would have to build what Earth Engine does for us: per-pixel cloud masking, seasonal
median, mosaicking across MGRS tiles *and* UTM zones, reprojection to EPSG:3857, plus
resume / sharding / QC.

**Validation so far is encouraging but incomplete.** Geometry is exact — reflectance
correlation 0.9987, zero pixel shift. NDVI agrees to MAE ≤ 0.025 on **11 of 14 windows**.
Of the two outliers, one is fully explained (it is the QA60 gap above) and **one is not**.
The sample was also opportunistic rather than designed.

Two traps found, both of which fail *silently*:

- The widely-circulated advice to apply the BOA offset is **wrong for this data** —
  Element 84 already removed it. Applying it drives reflectance negative and clips most of
  the red band into nodata.
- ESA's reprocessing publishes the same overpass twice; **43 of 57 tile-days in 2019** carry
  two versions, which would double-weight those dates in the median.

Both are handled in our prototype, but they illustrate that this route needs careful work,
not a quick script.

### D · Earth Engine commercial

| | |
|---|---|
| **Throughput** | Paid EECU; also raises batch concurrency from ~3 to ~20 |
| **Time to start** | Sales conversation plus registration |
| **Cost** | **Unknown — rates are not published.** Note `abruptthawmapping` bills to a card, *not* the pdg credit, so this is real money |
| **Fidelity risk** | None |

Cannot be assessed without a quote. Worth one email if A and C both stall.

### E · Reduce scope

| | |
|---|---|
| **Effect** | Does not raise throughput; shrinks the requirement |
| **Example** | Three epochs (2019 · 2022 · 2024) instead of six years ≈ 19 months on Contributor — still not viable alone, but **halves whatever route we choose** |
| **Cost** | Lower temporal resolution in the science |

Not a route by itself, but it composes with all the others and is the only lever that
depends on neither Google, money, nor new code.

### Summary

| route | time to first data | total time | cost | new code | external gate |
|---|---|---|---|---|---|
| **A** Partner Tier | weeks (approval) | then ~days | free | none | Google approval |
| **B** Contributor | now | ~38 months | free | none | none |
| **C** Earth Search | days–weeks (build) | ~8 days pull | VM time | **substantial** | none |
| **D** Commercial | weeks | fast | **unknown** | none | sales + budget |
| **E** Reduce scope | — | halves any route | free | none | science call |

### One rule whichever we choose

**Do not mix sources across years.** A series built half from Earth Engine and half from
Earth Search puts a source boundary *inside* the change signal — the same class of error as
the QA60 gap we just found. One source for all seven epochs, unless per-year equivalence is
proven. Current evidence (MAE 0.009 typical, one unexplained outlier) is not strong enough.

---

## 6. Work that proceeds regardless of the route

1. **Submit the Partner Tier application.** Free, weeks of latency, forecloses nothing.
   Needs our legal entity/PI, downstream impact and policy use, and a contact.
2. **Let Planet run.** ~26 days for the remaining years, zero failures. It is the campaign's
   floor and it hides the latency of every Sentinel-2 route.
3. **Run the mask bake-off and fix 2022–2023** — required under any source.
4. **Closest-year substitution for cells with no cloud-free scenes** — already agreed, not
   yet built. Two cells have failed so far. A `source_year` column keeps substitutions
   visible rather than silently absent.
5. **Explain the remaining Earth Search outlier** — cheap now, and required before route C
   could be recommended.

## 7. Mitigations register

| Risk | Mitigation | Status |
|---|---|---|
| EE monthly compute exhausted | Partner Tier application; route decision | drafted / **open** |
| QA60 empty 2022–2023 | Bake-off → reproduce-QA60 mask, those years only | decided, unbuilt |
| 165 exported 2022 cells unmasked | Re-export once the mask is chosen | pending route |
| Cells with no cloud-free scenes | Closest-year substitution + `source_year` | agreed, unbuilt |
| Earth Search: BOA offset | Do **not** apply; verify against the 2025 product | documented |
| Earth Search: reprocessing duplicates | Dedupe per (tile, acquisition), prefer higher baseline | documented |
| Earth Search: unexplained outlier | Diagnose before recommending route C | **open** |
| Source boundary inside the change signal | One source for all epochs; no mixing without proof | rule stated |
| Planet acquisition stall | Cron alerter, two-signal liveness | live |
| Progress lost across sessions | Per-year state + generated `PROGRESS.md` | live |

## 8. How we will know it worked

- The chosen mask's agreement with QA60 is reported **per stratum** (latitude, land cover,
  cloudiness), not as one pooled number, with the losing candidates' numbers kept.
- A re-exported 2022 cell **no longer** matches an unmasked reconstruction — the current
  signature of the bug — and sits inside the spread of the QA60-masked years.
- Mean NDVI per year over a fixed cell sample shows **no step** at 2021→2022 or 2023→2024
  beyond interannual variability.
- Substituted cells are visible downstream via `source_year`.
- If route C: the gate is a **designed** stratified sample of ~10 cells rebuilt for 2025 and
  diffed against the delivered product — with the outstanding outlier explained first.

---

### Background detail

`interannual_inference/` — `ee_quota.md` (the compute numbers), `qa60_gap.md` (the mask
finding), `s2_source_evaluation.md` (Earth Search evaluation),
`ee_partner_tier_request.md` (application draft), `prototype_earthsearch_diff.py` (the
prototype that produced the measurements quoted here).
