# Earth Engine quota — the campaign's controlling constraint

**Measured 2026-08-26.** The S2 export is the interannual run's critical path, and its
ceiling is not concurrency — it is the **monthly EECU allowance of the Earth Engine
noncommercial tier**. This document records the numbers, because every scheduling
decision in the campaign follows from them.

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

## Near-term prediction (falsifiable)

The EECU allowance is monthly. Around **2026-09-01** `abruptthawmapping` should reset to a
fresh 1,000 EECU-hours, deliver roughly **280 cells quickly**, and then throttle again.
If that does not happen, this model of the constraint is wrong and should be revisited.

## Ask

`interannual_inference/ee_partner_tier_request.md` — drafted, needs a human to submit
via the [Earth Engine help page](https://www.earthengine.app/). Approval takes several
weeks, so it is worth filing before the September allowance is spent, not after.
