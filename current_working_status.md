# Working Status — RTSmapping_v2

Project diary: where we are and where we're going. **Rolling** — the *Now* section is overwritten each
update and the *Just completed* step rolls forward; old detail is not accumulated here (history lives in
git and `docs/archive/`). Experiment numbers + the locked recipe are **not** duplicated here — they live
in `docs/experiment_ledger.md` (the SSoT); this doc links to it. Update ritual: see `CLAUDE.md`.

---

## Project Summary

Semantic segmentation of **Retrogressive Thaw Slumps (RTS)** in Arctic satellite imagery (60–74°N).
Train on 2024 PlanetScope Quarterly Basemap (RGB, ~3 m), deploy inference on 2025 imagery for a
pan-arctic RTS survey map. Solo research project — flat code, minimal abstraction.

**Core constraints** (non-negotiable, see `CLAUDE.md`): CRS EPSG:3857 · tile 512×512 · labels
0=bg/1=RTS/255=ignore · per-dataset z-score norm (`normalization_stats.json`) · seed 42, deterministic.

**Stack**: PyTorch 2.x + `segmentation_models_pytorch` (UNet++/EffB5), albumentations, rasterio,
geopandas, MLflow. Compute: 8× A100-80GB (`a100-8x-train`) for training; 32× L4 (us-west1) provisioned
for inference. Docker `rts-train:v2`. Data in `gs://abrupt_thaw/` + `gs://rts-mapping-v2*`.

**Dataset**: v1.0 standard (22,259 tiles; 1,718 pos / 20,541 neg), corrected leakage-free region split.
**Diagnosis**: representation-limited, not data-volume- or capacity-limited (see ledger family B).

---

## Rolling progress

### Just completed
**Earth Engine Partner Tier granted on `abruptthawmapping` (2026-09-22), expiring 2027-09-22.**
This ends the constraint that made the interannual campaign a three-year project. Verified off the
quota API rather than assumed — `monthly_eecu_usage_time` reads an effective
**180,000,000,000 EECU-s/month = 50,000,000 EECU-hours**, which is **500x the documented Partner
figure** (100,000 EECU-h) and 50,000x the Contributor tier we were on. The entire remaining S2
requirement (~10,600 cells x 12,755 EECU-s ≈ 37,000 EECU-h) is **0.074% of one month**.

Route A of `interannual_inference/imagery_acquisition_plan.md` is taken; routes B–E are closed. No
Earth Search build, no commercial quote, no scope reduction — and "do not mix sources across years"
now holds trivially, since every epoch comes from the same EE code path that made the delivered map.

**Three things replace EECU as the constraint**, and only the first is solved:
- **Concurrency** — unmeasured under Partner. Was ~3 RUNNING/user under restricted mode; commercial
  is ~20. That is the difference between 10.5 and 1.6 days per year of export. Measure it in the
  first six hours of 2019 and record it in `ee_quota.md`. The **per-user 3,000-task queue depth is
  unchanged by the tier**.
- **The QA60 gap** — untouched. 2022 and 2023 have no cloud mask, the bake-off is undone, and the
  165 cells already exported for 2022 are unmasked and must be discarded.
- **GPU quota** — `abruptthawmapping` has **none**, and no request has been filed. See below.

**Config changes this forced** (all in one commit, because `/opt/rts/RTSmapping_v2` is a live shared
checkout):
- `years` re-sequenced `[2022, …]` → **`[2019, 2020, 2021, 2024, 2022, 2023]`**. 2022 led while it
  was the drift pilot — a choice made before the QA60 gap was found, which made it the one year that
  could not start. **2019 leads now**: its Planet acquisition is complete, its QA60 is intact, and it
  is the widest baseline against 2025, so it is the year that can run end-to-end first.
- `expect.n_quads` scalar → **per-year map**, read via new `stages.expect_quads_for()`. Delivered
  counts are genuinely year-dependent (2019 **308,459**, 2022 **309,107**) because the Dove
  constellation grew across the archive. 2019 passed the 1% tolerance by 0.21%, but a thinner early
  year would have tripped a guard built to catch *filename* regressions on real coverage.
- `accounts` collapsed to one entry. This was a **live bug**, not just simplification: `accounts.2019`
  pointed at `/mnt/outputs/adc-rtsmapping/application_default_credentials.json`, which does not exist,
  and `adc_for()` falls back to `default` when a path is missing — so a run launched "under
  rtsmapping" would silently have been yyang. Two accounts on one year is exactly what that config's
  own rule forbids.
- `inference_watchdog.sh` `MAX_HOURS` 48 → **144**. 48 was sized against the launch plan's 12–29 h
  estimate; the benchmarked rate is ~4.2 tiles/s/L4, so a year is 2.2–4.8 days and 48 h would trip
  mid-run on every fleet shape except a tuned 8×A100.

### Now
**Review campaign — `unsure` adjudication package built (2026-10-15).** **5,411 polygons, 16,916
chips, 9.6 GB zip** at `E:\rts_adjudication\full\unsure_pkg_2026-10-15.zip` (outside the repo; the chips
are Planet-derived, internal only). It will go to Heidi for ArcGIS Pro review with Esri World Imagery
beside Planet; the README says only how to load the imagery and polygons. Verified with GDAL/rasterio, but
**not yet opened in ArcGIS Pro**, and Esri's resolution above 60°N is unchecked. Parsing the returned work
and merging it into `merge_review_verdicts.py` is not built. Re-run with `--exclude issued_ids.csv` once
more batches land (`review_campaign.md` §7.1).

**2019 Sentinel-2 export running, and the schedule has been re-planned around a hard limit
we are not going to get lifted.** Decision 2026-09-23: accept concurrency as it is and work
with it, rather than waiting on Google.

**What Partner Tier actually delivered.** The EECU ceiling is genuinely gone —
`monthly_eecu_usage_time` reads 180,000,000,000 EECU-s/month, a `producerOverride`, 500x the
documented Partner figure. But **batch concurrency is 1 task, project-wide**, measured over
three hours with two accounts and two years in flight. `ee.data.getProjectConfig()` still
returns `registrationState: REGISTERED_NOT_COMMERCIALLY` — the string this repo already
records as why pdg's batch exports never start. The tier change appears not to have landed
with the quota change, and no concurrency quota is exposed as a metric, so it is internal
state we can neither read nor set. Net effect: **we can consume 0.02% of the allowance.**

| | |
|---|---|
| Concurrent RUNNING, one account | 1 |
| Concurrent RUNNING, **two** accounts | **1** — unchanged |
| Throughput | 2.2 cells/hour -> **~35 days per year of data** |
| Six years, serial | **~200 days** |

**This overturned the multi-account finding.** 2026-08-25 measured concurrency as per-user
(a second account's tasks went RUNNING within one second). 2026-09-23 measured 1 RUNNING
across `rtsmapping@` and `yyang@` together. Concurrency is per **project**. Collapsing
`accounts` to one entry was right, for a reason nobody had at the time. It is also *worse*
than before the uplift, when `yyang@` alone held 3 slots.

**Two changes follow, both landed:**
- `export_s2_composites.py` `MAX_ACTIVE_TASKS` 2000 -> 2900. At 2000 this was a trap: the
  count is project-scoped but the server ceiling is per-user, so a second year throttles
  against the *first* year's backlog. 2020's driver sat in "queue full, backing off 30s"
  indefinitely behind 2019's 1,738 pending. **Run one year at a time.**
- `years` re-ordered to **[2019, 2022, 2024, 2021, 2020, 2023]**. When a year costs a month,
  the series must be useful at every prefix of the list. 2019 is the widest baseline against
  the delivered 2025; 2022 is the midpoint, and 2019+2022+2025 is the first 3-epoch series
  that can show whether change is linear. The previous order spent 70 export-days on 2020
  and 2021, clustered at one end.

**The QA60 bake-off is now the priority analysis task**, because it gates 2022 — which
should be second — and it is days of our own work, not 35 days of export. If it has not
landed when 2019 finishes, swap 2024 forward.

**Live state:** 2019 has 1,744 tasks queued under `rtsmapping@`, ~6 done. 2020's driver was
stopped (it could not progress against the 2000 cap); its ~261 submitted tasks remain queued
and will run — nothing was cancelled. **The 2019 run is unmonitored**: it was launched
directly rather than through `drive.py`, so no stage state exists, and the interannual
alerter cron is still not installed.

**Blocked, needing a human:**
- `main` merge/push is blocked by the permission classifier, so `/opt/rts/RTSmapping_v2` is
  still at `b7c2500` and has none of these changes.
- **Planet 2020** needs Heidi — `run_year.sh` requires `PL_BM_API_KEY` and
  `PDG_PL_ORDERS_KEY`, prompted interactively and never stored. ~22 days for the four
  remaining years, fully independent of S2.
- **GPU quota** in `abruptthawmapping` is still zero and still unrequested.

Also corrected: `pdg_migration.md` records per-user ADC as provisioned for `yyang@` on
2026-08-28. It was not there. Only `rtsmapping@` had one; `adc-heidi` is an incomplete login
with no credential file. `yyang@` ADC was created 2026-09-23. Same failure mode as the
migration's own recurring lesson — recorded as done, never re-opened.

### Future plans

**Downstream is the real risk, and it is not a code problem.** ~178 TB of imagery will land in a
project with **zero GPU quota and no credit**, billing to a card rather than the expired $70k PDG
credit. Everything else is ready and verified: the frozen 3-seed ensemble is MD5-identical and
anchored to the `model_checkpoint_sha` in production manifests (threshold 0.65, temperature
0.512321); `rts-infer:v1` survived in the new Artifact Registry; and the whole execution path —
`shard_tiles.py`, `claim.py`, `run_inference_worker.py`, `create_inference_fleet.sh`,
`inference_watchdog.sh`, `assemble_region.py`, `vectorize_region.py` — is built, drilled and already
repointed at `abruptthawmapping`.

1. **File the GPU quota request now, concurrently with the export.** Requests take days to weeks and
   cost nothing to hold; every day of S2 export is a free day of latency to hide them behind. Values
   and rationale are in `computing/infrastructure.md`. **Ask for `PREEMPTIBLE_NVIDIA_L4_GPUS=32`
   this time** — Spot is ~3× cheaper, the claim queue makes preemption safe, and budget was not a
   constraint when Spot was last declined. It is now.
2. **Run 2019 end-to-end to a delivered map** before starting 2020's inference. It exercises all
   twelve stages against the new project for the first time, and produces the merge/vectorize
   wallclock the cost model is missing — **no timing for either stage is recorded anywhere**.
3. **Decide the per-year QC protocol before the first year reaches the gate**, not at it. 2025 South
   was a 60,467-item census ≈ 134 person-hours. Six years of that is the largest cost in the
   campaign and no quota grant touches it.
4. **Revisit storage tiering at a checkpoint.** Acquiring hot is the decision taken; at full
   footprint that is ~$3.5–3.8k/month. The existing policy — Coldline immediately, Archive once a
   year's map is approved — takes it to roughly a tenth. `inference/` stays excluded regardless:
   41.5M ~2.5 KB objects would each bill a 128 KiB minimum.
5. **Deferred**: North S2-RGB model; v3 backlog (re-stage, MAE SSL, multi-scale — see ledger
   "Deferred to v3").

*(The previous Phases 0–4 inference plan that stood here was stale: that work shipped, and the 2025
South run reconciled 2026-07-10.)*

---

### 2025 South review campaign — context imagery, neighbours, and a claim TTL that never fired (2026-09-24)

A parallel workstream to whatever leads the *Now* section above; it owns no part of that schedule.
Evidence and detail live in `post-inference/review_campaign.md` §4.1, §4.3 and §6.2 — only the
decisions are here.

**The claim TTL had never fired.** The app's identity held `roles/storage.objectCreator`, which
grants create-if-absent and neither overwrite nor delete, so heartbeats 500'd invisibly into a
`fetch` with no `.catch`, and stale reclaims swallowed their 403 for seven weeks. 18 batches (3,600
polygons) were stranded, and because the oldest was `b00001` the honest contiguous-prefix headline
sat at `max_prob >= 1.0` while 29% of the inventory was rated. A `storage.objects.delete` role
conditioned to the claims prefix was granted and verified live.

**The context view was part-blank for ~87% of polygons, and it was not a download gap.** The wide
crop is 10x the feature; chips existed only for tiles a *detection* sits in. Of the 502,555 tiles the
wide views needed but lacked, **98.8% sat on quads already in the bucket** — never chipped. Only
5,999 had no quad, and just **two quad cells** are fully enclosed holes (75.65N/112.94E,
71.50N/52.47E, 91 tiles between them). The acquisition is sound; the chipping scope was too narrow.

**Rebuilt and cut over.** `build_rgb_chips.py --wide-context` selects tiles by the wide crop window;
621,158 chips written with 0 errors, all 240,668 crops re-rendered, cutover verified by
`scripts/gcs_parity.py` (**PARITY PASS**). Polygons without full context fell from ~87% to **4.09%**;
severe gaps (>50% of the view missing) from 15.3% to **1.2%**. Remaining gaps are drawn as labelled
diagonal striping rather than black, because black is indistinguishable from dark water.

**Verdicts already recorded stand, by decision.** On 240 rated high-confidence polygons, wide-crop
blackness tracked the verdict (`false` 47% against 13% for clean crops, p<0.0001) and the gap
survived controlling for polygon area — but blackness is largely a proxy for polygon size, so it is
suggestive, not established. ~4,300 high-confidence polygons were rated with most of their context
missing; the caveat is recorded in the campaign spec rather than acted on.

**Follow-up (2026-09-28): the striping itself was firing on valid imagery.** The operator found
crops captioned `NO IMAGERY` over ground that opened complete in QGIS. Planet's 8-bit basemaps clamp
deep shadow and dark water to exactly `(0,0,0)` with `alpha=255`; the chip writer wrote NoData as 0
too, so `max == 0` caught both. Pooled over seven sampled quads, **68.8% of striped pixels were valid
imagery** — including the riverbank shadow sitting on a polygon's own downslope edge. Fixed by
reserving 0: valid imagery is floored at 1 (`write_rgb_chip`), which makes `render_crop`,
`has_imagery` and `imagery_fraction` correct unchanged. **Review-only** — inference masks by the
quad's alpha band, and training's identical conflation sits behind `data.nodata_handling`, default
off and set by no config, so it never reached the delivered model. Re-chipped and re-rendered 2026-09-28: partial context **4.09% → 0.15%** (92
polygons), no-imagery 20 → **0**, both uploads byte-verified, app serving `?v=3`. Detail in `post-inference/review_campaign.md` §4.4.

**Open**: the two enclosed quad holes are worth raising with the acquisition side. The app is plain
HTTP with no sign-in (§10.3) and Chrome's HTTPS-Upgrade already makes it unreachable in Incognito —
if that enforcement reaches normal windows the campaign goes dark for every reviewer.

---

## Pointers

- **Experiments / recipe / findings** → `docs/experiment_ledger.md` (SSoT)
- **Visual report** → `docs/report.html` (generated from the ledger)
- **Optimization opportunities + fairness audit** → `docs/optimization_roadmap.md`
- **Inference pipeline** → `inference/inference.md` · **infra/budget** → `computing/infrastructure.md`
- **Pre-2026-06-25 dated status / decisions / dev-log** → `docs/archive/working_status_pre-rolling_2026-06-25.md`
