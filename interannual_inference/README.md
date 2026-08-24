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

State lives at `/mnt/outputs/interannual_inference/state/<year>.json` (mirrored to
`gs://rts-mapping-v2-usw1/interannual_inference/state/`), logs at
`/mnt/outputs/interannual_inference/logs/<year>/<stage>.log`. Neither is in the repo.
