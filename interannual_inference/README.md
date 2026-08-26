# Interannual inference

Six years of PlanetScope q3 basemaps (2019–2024) run through the **frozen** deployed
model, to turn the single-epoch 2025 RTS map into an interannual series.

That is ~6 years × 12 stages, spread over months, split between two people
(Heidi holds the Planet key and runs acquisition; we run everything downstream).
This directory is the thing that remembers where it all is.

A proper Python package (`interannual_inference`), so the modules import each other
normally. Its sibling `planetscope-download/` is Heidi's acquisition side — hyphenated
because those are standalone scripts that never import one another.

## The one-minute version

```bash
python interannual_inference/status.py                        # where is everything?
python interannual_inference/status.py --year 2022            # drill into one year
python interannual_inference/drive.py --year 2022             # run that year until a human is needed
```

`drive.py` never walks past a human gate, a failure, or a stage someone else owns.
Re-running it is always safe: finished stages are skipped, not repeated.

## What it is (and is not)

It is a **state file plus a driver**. Every stage shells out to a script that
already existed and that we already ran by hand — `build_quad_index.py`,
`export_s2_composites.py`, `shard_tiles.py`, `launch_south_inference.sh`. Nothing
here reimplements pipeline logic, and no stage does anything you could not do
yourself with the command it prints under `--dry-run`.

It is **not** a workflow engine, and it does not manage compute. The inference run
is still supervised by `scripts/launch_south_inference.sh` with its own crash-loop
guard, stall watchdog and `STOP` sentinel.

## The stages

| stage | what runs it | notes |
|---|---|---|
| `acquire` | **Heidi**, on the acquisition VM | `planetscope-download/run_year.sh` |
| `s2_export` | us, GEE batch | **the critical path** — see below |
| `quad_index` | us | reconciles against orders placed |
| `s2_index` | us | |
| `drift_check` | us | **human gate** |
| `tile_grid` | us | grid + domain mask, ~25 min |
| `shard` | us | ~2,079 shards of 20k tiles |
| `infer` | us, 8×A100 | ~2.5 days/year |
| `reconcile` | us | shards done must equal shards total, exactly |
| `merge` · `vectorize` · `qc` | us, by hand | tracked here, driven from `post-inference/` |

`s2_export` has **no prerequisite** — it does not wait on Planet. That is the single
most important scheduling fact in the campaign: the S2 export is the long pole
(measured 10.9 days for 2025_south, sharing GEE with two other exports), so it must
be started as early as possible and run alongside acquisition, not after it.

## The export ceiling, and the only thing that lifts it

Earth Engine limits us in three places, and **they do not all have the same scope** —
getting that wrong cost a day:

| limit | scope | how we know |
|---|---|---|
| Monthly EECU allowance → restricted mode | **per project** | read off the quota API: 3.6M EECU-s/mo on `abruptthawmapping` (Contributor), 540k on pdg (Community) — which is why pdg sat restricted while `abruptthawmapping` ran, same user, same moment |
| Task-queue depth (3,000) | **per user** | 1,799 queued on one project + ~440 each on three others died at 3,002 |
| Concurrent RUNNING (~3) | **per user** | 1,321 tasks pending across three other projects held **0 RUNNING for 20 min** while `abruptthawmapping` kept all 3 slots, and the running year's rate never moved off 6.6 cells/hr |

So **more projects do nothing** — that was tried and reverted. At ~3 concurrent tasks per
user, one year is ~10.5 days and six years is ~64 days serial.

### …but concurrency is not what is actually stopping us

**Measured 2026-08-26.** The export costs **12,755 EECU-s per cell**, so one year is 22.9M
EECU-s against a Contributor-tier allowance of **3.6M/month** — **6.4 months of quota per
year of data, 38 months for six years.** We spent 69% of the month in the first 48 hours and
have been throttled in restricted mode since. Past the cap tasks still run, just slowly, which
is why this looked like a slowdown rather than an error.

**Partner Tier (100,000 EECU-hours/month) fits the entire remaining campaign in 37% of one
month's allowance** and is the only lever that changes the schedule. Application drafted at
`ee_partner_tier_request.md`; the full numbers are in **[`ee_quota.md`](ee_quota.md)**.

### More *accounts* raise concurrency — but they cannot create quota

Two accounts get their own ~3 slots each, which is real. What they do **not** do is add
EECU: both draw on the same project's monthly allowance, so the effect of the second account
was to reach the cap in half the time. Keep two for when quota is available; adding more
would only spend it faster.

#### Two is still the right number

**Proven 2026-08-25.** With `yyang@` already holding all 3 slots, `rtsmapping@` submitted
two tasks that went RUNNING within **one second**, taking the project total to 5 — even
though they were queued behind 1,676 of yyang's pending tasks, which rules out a
queue-position artifact. Both years now hold 3 RUNNING each.

**We deliberately stopped at two accounts** (user decision, 2026-08-25). Adding more
would speed up S2 but not the campaign, because the two halves are already balanced:

| | remaining | finishes in |
|---|---|---|
| Planet acquisition (serial — Heidi holds one API key) | 2019 + four more years | ~27 days |
| S2 export (2 accounts, ~10.5 d/year) | 2022 + 2019 running, then four in two rounds | ~31 days |

A third account would cut S2 to ~21 days and make Planet the binding constraint at ~27 —
a net campaign gain of only ~4 days, for the cost of another person's Google credential
sitting on a shared VM. Not worth it. Revisit only if Planet acquisition speeds up.

And note the campaign is **pipelined per year**, not gated on the tail: a year's
inference can start as soon as *that* year's Planet and S2 have both landed. 2022's
Planet is already complete, so 2022 inference is unblocked the moment its S2 finishes.

### Who could be added, if that ever changes

No IAM changes needed for any of these — all are provisioned on `abruptthawmapping`:

| account | Earth Engine role | `serviceusage.services.use` |
|---|---|---|
| `yyang@` | `earthengine.admin` | ✅ via `serviceUsageConsumer` |
| `rtsmapping@` | *(none needed)* | ✅ via `editor` — **verified** it can init EE and submit |
| `ryoung@` (Rob) | `earthengine.writer` | ✅ via `serviceUsageConsumer` |
| `hrodenhizer@` (Heidi) | `earthengine.admin` | ✅ via `owner` |

### One year per person — never the same year twice

Assign a whole year to each account. The launcher skips cells already **delivered**, but
not ones merely **in flight**, so two people on the same year would submit duplicate
tasks and burn the shared 3,000-slot queue for nothing. One year each needs no code and
cannot collide.

Everyone can run on this VM under their own credentials, which keeps logs and monitoring
in one place:

```bash
CLOUDSDK_CONFIG=/mnt/outputs/adc-<name> gcloud auth application-default login
CLOUDSDK_CONFIG=/mnt/outputs/adc-<name> \
  python interannual_inference/run_stage.py --year <YYYY> --stage s2_export
```

⚠️ That ADC file is that person's Google credential. Anyone with sudo on this VM can read
it — the same caveat as the Planet API key. If that is not acceptable, run from your own
machine instead; the export is server-side either way and no imagery transits your laptop.

## Human gates

Two stages stop and wait for a person, and the driver will not pass them:

- **`drift_check`** — is this year's imagery radiometrically like what the model was
  deployed on? Sign off with
  `python interannual_inference/run_stage.py --year YYYY --stage drift_check --sign-off`.
- **`qc`** — is the resulting map good enough to call delivered?

### The drift baseline is the 2025 *quad* sample, not the training stats

`inference.md` §5.4's thresholds were written for tile samples. Comparing a random
**whole-quad** sample against `normalization_stats.json` mis-fires, because those
stats come from 17,951 curated RTS-centric training tiles while random quads also
cover water, snow and bare rock. **2025 — the imagery the delivered map was made
from — trips the blue channel at +0.256 against the training stats**, essentially
as hard as 2022's +0.295. Against the right baseline (2025 quads, same sampling)
2022's worst mean drift is 0.095σ against a 0.5σ threshold.

So `drift_check` compares against `paths.quad_baseline`
(`drift_report_2025q3_control.csv`). A trip there means something real.

## Marking work we did not do

```bash
python interannual_inference/run_stage.py --year 2019 --stage acquire --mark-done   # Heidi finished
python interannual_inference/run_stage.py --year 2022 --stage merge  --mark-done
```

Evidence is re-read from the artifact when you mark, so the ledger stays truthful
even for stages the driver never ran.

## Alerting

`interannual_inference/alert.py` runs from cron and posts to the same Slack webhook the
acquisition alerter uses. It announces four things, **once each**: a failed stage, a
gate reached, a stage gone quiet, and a year completed.

"Gone quiet" needs **two** signals — a stale heartbeat *and* a probe that has not
advanced. A stale heartbeat alone is not enough: the GEE queue legitimately idles and
the inference run lists its shards for a long time at startup. This is the same rule,
and the same reasoning, as `planetscope-download/alert_if_stopped.py`.

`interannual_inference/notify.py` is a deliberate ~20-line copy of that script's Slack helper
rather than a shared import. `alert_if_stopped.py` is **live in cron right now**
watching Heidi's acquisition, and refactoring a running alert path to save ten lines
is a bad trade. The webhook file is shared, so there is still only one secret.

## Files

| file | role |
|---|---|
| `config.yaml` | the values — years, buckets, paths, expected counts, thresholds |
| `stages.py` | the stage table: order, prerequisites, commands, evidence, probes |
| `state.py` | per-year state JSON — atomic writes, GCS mirror, heartbeats |
| `run_stage.py` | run exactly one stage, with all the checks |
| `drive.py` | walk a year through the chain, stopping at gates |
| `status.py` | the year × stage matrix |
| `alert.py` / `notify.py` | cron alerting |
| `ee_quota.md` | **the EECU numbers** — tiers, per-cell cost, what the schedule really depends on |
| `ee_partner_tier_request.md` | Partner Tier application draft, awaiting a human to submit |
| `s2_source_evaluation.md` | Earth Search / S3 COGs vs the EE export — measured throughput, the traps, the gate |
| `qa60_gap.md` | **QA60 is empty in 2022–2023** — those composites have no cloud mask |
| `prototype_earthsearch_diff.py` | prototype behind those two docs; re-derives their numbers |

State lives at `/mnt/outputs/interannual_inference/state/<year>.json` (mirrored to
`gs://rts-mapping-v2-usw1/interannual_inference/state/`), logs at
`/mnt/outputs/interannual_inference/logs/<year>/<stage>.log`. Neither is in the repo.
