# PlanetScope basemap acquisition (2019–2024)

Orders Planet Global Quarterly q3 basemap quads for the circumpolar-south
domain and has them delivered straight into `gs://pdg-planet-data`.

**Heidi — start here.** This is your `circumpolar_planet_basemaps` workflow
ported into this repo, with the changes you asked for in the PR #61 review
(2026-08-17) already applied. You supply the Planet key and start the run;
everything else is set up. [§Quick start](#quick-start) is the short version.

The full rationale, schedule, cost and the decisions behind all of this live in
[`docs/interannual_planet_download_plan.md`](../docs/interannual_planet_download_plan.md).

---

## What changed from your notebooks

| Your notebook | Here | Why |
|---|---|---|
| `1_basemap_grid_search.qmd` | `search_basemap_grids.py` | year is an argument |
| `2_circumpolar_south_basemap_grids.qmd` (R) | `filter_to_domain.py` (geopandas) | ported off R so the VM needs one language and one environment |
| `3_order_basemaps.qmd` | `order_basemaps.py` | bounded retries, faster listing, progress file, stall watchdog |
| the rename + delete cells | `tidy_rename.py`, **optional** | see below |
| — | `run_year.sh` | supervises the whole year; survives you disconnecting |
| — | `check_status.py` | progress without attaching to tmux |

**The rename is no longer required.** It existed so the bucket read tidily.
Our indexer (`inference/quad_index.py`) lists recursively and matches on the
*basename*, so it reads Planet's raw delivery — order-UUID directories and all —
exactly the same as the flattened layout. That removes the crash-recovery
problem you'd been fighting and one of the two slow listings. `tidy_rename.py`
is still there if you ever want the tidy layout; it now derives its work from
bucket state on every run, so re-running after a failure just recomputes what's
left, and it deletes per-object with 404 tolerated instead of the batch delete
that aborted on one missing file.

**Your other three changes.** Filename matching is now
`(\d+)-(\d+)_quad[^/]*\.tif` — it covers `_quad.tif`, `_quad_file_format.tif`,
the bare `<col>-<row>_` raw-delivery form, and whatever Planet adds next
(verified against live 2019 and 2025 objects). The prior-delivery listing asks
for object names only instead of full metadata. And the retry policy is
[below](#when-something-goes-wrong).

---

## Prerequisites

- **VM access.** `gcloud compute ssh a100-8x-train --zone us-central1-a --project pdg-project-406720`.
  Needs a `roles/compute.osLogin` binding on your account — we've requested it. You do not need sudo.
- **Your two keys**, typed when prompted. Nothing is written to disk.

## Quick start

```bash
tmux new -s planet                              # so the run survives you disconnecting
cd /home/ext_yyang_woodwellclimate_org/RTSmappingDL   # the shared checkout, not your home dir
./planetscope-download/run_year.sh 2022
# checks its environment, then prompts for PL_BM_API_KEY and PDG_PL_ORDERS_KEY
# Ctrl-b then d  -> detach; the run keeps going
```

The repo is a **shared checkout you can read but not write**, so nothing is
written into it. Outputs go to `/mnt/outputs/planetscope-download/`
(`data/`, `status/`, `logs/`), and the scripts run under a shared virtualenv at
`/mnt/outputs/planetscope-venv` because the system python has no geopandas.
Both are found automatically; override with `PSD_WORK` and `PSD_PYTHON` if you
ever need to.

`run_year.sh` checks all of that **before** asking for your keys, so a setup
problem costs you nothing but a re-run.

Come back any time with `tmux attach -t planet`, or just:

```bash
python3 planetscope-download/check_status.py
```

```
  year               done     pct   ordered   skipped  failed  ord/min   eta_h  heartbeat
  2019  309,100/309,100    100.0%   309,100         0       0     38.4     0.0  120 min ago  complete
  2022   41,203/309,100     13.3%    41,203         0       7     39.1   114.2  1 min ago
```

A finished year is marked `complete`. An **incomplete** year whose heartbeat is
older than ~5 minutes prints `STALE`, which means the process died without the
supervisor restarting it — check `/mnt/outputs/planetscope-download/logs/`.

**2022 first, alone.** It's the pilot — it settles the retry behaviour, the
delivery layout and radiometric drift before we commit five more years. After
it's checked: 2019 → 2020 → 2021 → 2023 → 2024.

## Alerts

A cron job checks every 10 minutes and posts to Slack when a year **stops and
needs a human**, and once more when a year finishes. The supervisor already
restarts crashes and stalls by itself, so this only fires for the three things
it cannot fix: an expired Planet key, a crash loop, or a VM reboot.

**One-time setup** — paste your Slack incoming-webhook URL into the file below.
It is deliberately not in git:

```bash
umask 077
echo 'https://hooks.slack.com/services/XXX/YYY/ZZZ' > /mnt/outputs/planetscope-download/slack_webhook
```

Until that exists the checker stays silent and simply logs to
`/mnt/outputs/planetscope-download/logs/alerts.log` when something needs
attention. Test it any time with:

```bash
/mnt/outputs/planetscope-venv/bin/python planetscope-download/alert_if_stopped.py --dry-run
```

A year is only counted as stopped when its heartbeat is stale **and** no
ordering process is running — on resume the loop lists the delivery prefix
before its first heartbeat, which can take a while, and alerting on that would
cry wolf on every restart.

## Stopping

```bash
touch /mnt/outputs/planetscope-download/status/STOP   # prevents the next restart
rm /mnt/outputs/planetscope-download/status/STOP      # then re-run run_year.sh to resume
```

**The sentinel alone will not stop a run in progress** (learned the hard way,
2026-08-28). `run_year.sh` checks it at the top of its supervision loop — that is,
*between* restarts of the ordering step. But the ordering step is the multi-day one,
and `order_basemaps.py` never looks at the sentinel, so "stops after the current
step" can mean "in forty hours".

To stop now, set the sentinel **first**, then end the ordering process — the sentinel
is what stops the supervisor restarting it:

```bash
touch /mnt/outputs/planetscope-download/status/STOP
pkill -f order_basemaps.py        # sudo if someone else started it
```

Resuming is always safe. Steps 1 and 2 skip if their output exists, and step 3
lists what's already delivered and skips those quads. A quad whose order was
in flight when you stopped is either delivered (and skipped) or not (and
re-ordered) — at worst you pay for one duplicate.

## When something goes wrong

`run_year.sh` restarts the order loop on any crash or stall, with a
crash-loop guard so it stops rather than spinning. What you might see:

| Symptom | What it means | What to do |
|---|---|---|
| `FATAL rc=2 (bad or expired credentials)` | 401 from Planet. We fail fast rather than retrying — auth can't be retried into working | Re-run `run_year.sh` with a fresh key. Delivered quads are skipped, so it picks up where it stopped |
| `exited 3 ... (stall watchdog)` | No order completed for 15 min — a hung socket | Nothing; it restarts itself. Recurring stalls are worth telling us about |
| `N quads failed after 5 attempts` | Transient errors that outlasted the backoff | Nothing; `run_year.sh` sweeps them up automatically at the end. `check_status.py` prints the manual command if you want it |
| `5 consecutive fast failures — stopping` | Something is systematically broken | Check `/mnt/outputs/planetscope-download/logs/orders_<year>.log` and send it to us |
| `preflight failed` | Missing deps, or `PSD_WORK` not writable | The message says which and how to fix it. No keys were asked for, so just re-run |
| `Planet rejected the API key (HTTP 401)` | Bad or expired `PL_BM_API_KEY` | Re-run with the right key |

Retries are bounded: **5 attempts with 30s→8min backoff** on transient statuses
(400/409/429/5xx), then the quad is recorded in
`/mnt/outputs/planetscope-download/status/failed_orders_<year>.csv` and the loop
**carries on**. A five-day run is
never abandoned over one quad. 401 is the exception — it stops immediately.

## Note on the keys

They're read into the shell's environment and never written to disk, so
restarts inherit them without asking you again. Two consequences worth knowing:

- **A VM reboot loses them** and you'll need to start `run_year.sh` again.
- Anyone with sudo on the VM can read them out of process memory. That includes
  us. No arrangement on a machine we administer changes that — it's why you're
  rotating the key when the runs finish.

## After a year finishes

Ours, not yours — listed so you can see where the numbers you record go:

```bash
python scripts/build_quad_index.py --bucket pdg-planet-data \
    --prefix global_quarterly/<year>/q3/ \
    --output /mnt/outputs/inference/quad_index_<year>q3.csv \
    --expect-quads <the count step 2 printed>
```

`--expect-quads` is why step 2 logs `RECORD THIS: N quads ordered` — if the
index comes up short against it the build fails loudly, which is how we'd catch
a filename regime the matcher doesn't cover. Then the normalization drift check,
tile grid, sharding, inference, review — and the year's quads move to Archive
storage once its map is approved.

## Files

```
planetscope-download/
├── run_year.sh              ← start here; supervises one year end to end
├── search_basemap_grids.py  ← step 1: which quads exist for this year
├── filter_to_domain.py      ← step 2: clip to circumpolar-south; prints the count to record
├── order_basemaps.py        ← step 3: place the orders (the multi-day one)
├── check_status.py          ← progress for every year
├── tidy_rename.py           ← optional cosmetic flattening; not needed by the pipeline
├── alert_if_stopped.py      ← cron: Slack alert when a year stops or finishes
└── requirements.txt

/mnt/outputs/planetscope-download/     ← runtime outputs (outside the repo)
├── data/                    ← step 1 and 2 geojson outputs
├── status/                  ← <year>.json progress, failed_orders_<year>.csv, STOP
└── logs/                    ← per-step logs

/mnt/outputs/planetscope-venv/         ← the interpreter the scripts run under
```
