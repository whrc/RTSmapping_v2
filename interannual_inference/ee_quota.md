# Earth Engine quota — resolved 2026-09-22

> ## RESOLVED. `abruptthawmapping` is on **Partner Tier**, expiring **2027-09-22**.
>
> The EECU wall that made this the campaign's controlling constraint is gone. Read off
> the quota API, not inferred from the docs:
>
> ```
> $ gcloud alpha services quota list --service=earthengine.googleapis.com \
>     --consumer=projects/801926669176 --format=json   # monthly_eecu_usage_time
> effectiveLimit  180,000,000,000 EECU-s/month  =  50,000,000 EECU-hours/month
>                 (producerOverride; defaultLimit is int64 max)
> ```
>
> That is **500x the documented Partner figure** (100,000 EECU-h) and **50,000x** the
> Contributor tier this project was on. The whole remaining campaign — ~10,600 cells,
> ~37,000 EECU-hours — is **0.074% of one month's allowance**.
>
> **What binds now, in order:** the **per-user 3,000-task queue depth** (unchanged by the
> tier), **concurrent RUNNING** (was ~3 under restricted mode; unmeasured under Partner —
> measure it and record it below), and the **expiry**. The allowance is monthly and does
> **not** bank, so anything EE-dependent, including any future re-export, must land before
> **2027-09-22**.
>
> **BUT THE CAMPAIGN IS STILL BLOCKED — the uplift looks incomplete.** Measured 2026-09-23
> over ~3 hours of live export: **concurrent RUNNING is 1**, project-wide, and the project
> still reports `registrationState: REGISTERED_NOT_COMMERCIALLY`. See "Measured under
> Partner" below. Do not read this banner as "solved".
>
> The rest of this document is the measurement that justified the application. It is kept
> because the per-cell cost is still the number every schedule estimate is built on.

**Measured 2026-08-26.** The S2 export is the interannual run's critical path, and its
ceiling was not concurrency — it was the **monthly EECU allowance of the Earth Engine
noncommercial tier**. This document records the numbers, because every scheduling
decision in the campaign followed from them.

## What the tiers actually give

[Noncommercial tiers](https://developers.google.com/earth-engine/guides/noncommercial_tiers):

| tier | EECU-hours/month | EECU-seconds/month | how you get it |
|---|---|---|---|
| Community | 150 | 540,000 | default |
| Contributor | 1,000 | 3,600,000 | **self-service**, [Manage tier](https://console.cloud.google.com/earth-engine/configuration/manage-tier) |
| Partner | 100,000 | 360,000,000 | application, manual review, several weeks |

Read straight off our projects, so this is not inference from the docs:

```
$ gcloud alpha services quota list --service=earthengine.googleapis.com \
    --consumer=projects/<n> --format=json   # metric monthly_eecu_usage_time
pdg-project-406720   540,000 EECU-s =   150 EECU-hours  -> Community
abruptthawmapping  3,600,000 EECU-s = 1,000 EECU-hours  -> Contributor
```

`daily_eecu_usage_time` is unlimited. The monthly figure is the only EECU cap that binds.

**Restricted mode is a throttle, not a stop.** Past the cap, tasks still run at reduced
throughput — which is exactly what both projects have been doing, and why the symptom
looked like a mysterious slowdown rather than an error.

## What our export costs

195 completed cells on `abruptthawmapping`, from `batchEecuUsageSeconds`:

| | EECU-s |
|---|---|
| per cell, mean | **12,755** (≈3.5 EECU-hours) |
| per cell, median | 8,796 |
| one year (1,799 cells) | **22,946,245** |
| six years | **137,677,470** |

## The consequence

| against | one year of data | all six years |
|---|---|---|
| Community (540k/mo) | 42.5 months | 255 months |
| **Contributor (3.6M/mo) — where we are** | **6.4 months** | **38 months** |
| Partner (360M/mo) | 1.9 days of allowance | **37% of one month** |

So on Contributor the sustainable rate is ~282 cells/month, and the interannual run is a
**three-year** project. On Partner the entire remaining requirement — 10,417 cells,
132,868,835 EECU-s, 36,908 EECU-hours — fits inside **37% of a single month's allowance**,
and the schedule reverts to being bounded by concurrency (~10.5 days/year/account) and by
Heidi's Planet acquisition.

**Partner Tier is the only lever that changes the schedule.** Everything else we tried
is a treadmill:

- **More projects** — refuted 2026-08-25 (concurrency is per-user), and now doubly so:
  each project carries its own 150 or 1,000 EECU-hours, i.e. ~0.16 years of data. You
  cannot assemble six years out of Community-tier projects, and borrowing other teams'
  projects would spend *their* allowance to do it.
- **More accounts** — real (concurrency is per-user) but it spends one project's shared
  monthly EECU faster, it does not create any. This is precisely what happened: two
  accounts drew 691 EECU-hours in 48 hours, 69% of the month, and tripped the cap.
- **Contributor upgrade** — already taken; `abruptthawmapping` is on it.

## History this explains

`pdg-project-406720`'s batch exports "never starting" was never a registration mystery.
It is on **Community**: 150 EECU-hours ≈ 42 cells/month, against a 1,799-cell year. The
2025_south export that produced the delivered map is what exhausted it, and every export
attempted there since has been running in deep restricted mode.

## Near-term prediction (falsifiable) — SUPERSEDED

The prediction was: around **2026-09-01** `abruptthawmapping` resets to a fresh 1,000
EECU-hours, delivers roughly **280 cells quickly**, then throttles again. It was never
tested to conclusion — the Partner uplift landed on 2026-09-22 and moved the ceiling by
five orders of magnitude, so the throttle it predicted can no longer occur. The model of
the constraint (per-project monthly EECU, per-user concurrency and queue depth) was
confirmed by the uplift behaving exactly as a per-project quota change.

## Ask — GRANTED

`interannual_inference/ee_partner_tier_request.md` was submitted and **approved**:
Partner Tier on `abruptthawmapping` (proj# 801926669176) for one year, **expiring
2027-09-22**. See the banner at the top for the measured limit.

## Measured under Partner (2026-09-23) — concurrency did NOT improve

The 2019 export was launched under `rtsmapping@`, and 2020 under `yyang@` an hour later,
specifically to test whether concurrency is per-user. It is not.

```
09:59  PENDING  636  RUNNING 1                 (2019 only, rtsmapping@)
10:36  PENDING 1741  RUNNING 1  SUCCEEDED 2
11:58  PENDING 1738  RUNNING 1  SUCCEEDED 5
12:06  PENDING 1999  RUNNING 1  SUCCEEDED 5    (2020 submitting under yyang@)
12:39  PENDING 1999  RUNNING 1  SUCCEEDED 6
```

| | |
|---|---|
| Concurrent RUNNING, one account | **1** |
| Concurrent RUNNING, **two accounts** | **1** — unchanged |
| Throughput | 6 cells in 2.8 h = **2.2 cells/hour** |
| Implied schedule | **~35 days per year of data, ~200 days for six** |
| EECU actually consumable | ~49,000 EECU-s/h = ~35M/month against an allowance of 180,000,000,000 = **0.02%** |

**Three conclusions.**

1. **Concurrency is per PROJECT, not per user** — this overturns the 2026-08-25 finding that
   a second account adds slots. Two accounts held 1 RUNNING between them. Multi-account is
   not a lever any more, so collapsing `accounts` to one entry was right (for a reason that
   was not the one given at the time).
2. **It is worse than before the uplift.** Contributor-in-restricted-mode measured ~3
   RUNNING for `yyang@` alone. We now get 1 across two accounts.
3. **`registrationState` is still `REGISTERED_NOT_COMMERCIALLY`** —
   `ee.data.getProjectConfig()` returns nothing else. This repo already records that string
   as the reason pdg's *"batch exports never start"*. The EECU override landed
   (`monthly_eecu_usage_time` = 180,000,000,000, a `producerOverride`) but no concurrency or
   registration change came with it, and **no concurrency quota is exposed as a metric** —
   `gcloud alpha services quota list` shows only EECU, `bigquery_slot_usage_time` and
   `read_requests`. So concurrency is governed by internal tier state we cannot read or set.

**The ask back to Google:** the quota override landed but the tier change appears not to
have. Quote `registrationState: REGISTERED_NOT_COMMERCIALLY`, the
`monthly_eecu_usage_time` override value, and that batch concurrency is 1 task project-wide
with ~2,000 tasks queued.

**Do not use `ee.data.getTaskStatus()`** for any of this — it returns `UNKNOWN` for jobs
that have succeeded. `ee.data.listOperations()` is authoritative, and is **project-scoped,
not user-scoped**: both accounts see the same task list.

## Operational trap found the same day

`scripts/export_s2_composites.py`'s own `MAX_ACTIVE_TASKS = 2000` blocked the 2020
submission once 2019's queue plus 2020's reached 2,000 project-wide (`GEE task queue full
(2000 active); backing off 30s`). It is a **client-side** guard counting **project-scoped**
operations, so running two years concurrently makes each one throttle the other long before
the real per-user 3,000 server ceiling. Raise it, or do not run two years at once.
