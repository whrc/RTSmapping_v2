# Interannual PlanetScope acquisition, 2019–2024 — run instructions

**For:** Heidi Rodenhizer · **Status:** approved 2026-08-17 (PR #61) · **Written:** 2026-08-14

Acquire six more years of PlanetScope q3 basemaps — 2019–2024 — so the deployed model can be run
interannually rather than on the single 2025 epoch. Heidi owns the Planet key and account; we
specify the delivery shape because our inference pipeline consumes it, and we build and operate
the workflow on the VM so her involvement per year is supplying the key and starting the run.

**The runnable article is [`planetscope-download/`](../planetscope-download/README.md)** — her
three notebooks ported into this repo with the changes from her review applied. This document is
the reasoning behind it: what we asked for and why, the decisions taken, and the cost and
schedule. §6 records the decisions; [§7–§9](#7-storage-and-archiving) are our side of the
pipeline.

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
| Quads per year | **309,100** (confirmed by Heidi, [6.3](#6-decisions)) |

An earlier draft of this document claimed that matching the 2025 shape meant "we change nothing
on our side at all". Heidi's review made that false, and rightly: Planet's filenames differ by
year, so pinning our matcher to one literal suffix pushed the cost of that onto her. Three
changes went into `inference/quad_index.py` instead —

* the quad matcher is now `(\d+)-(\d+)_quad[^/]*\.tif`, covering `_quad.tif`,
  `_quad_file_format.tif`, the bare `<col>-<row>_` raw-delivery form and whatever Planet adds
  next. Verified against live 2019 and 2025 objects: identical match set on 2025 (220/220 in
  column 500), and the 2019 legacy archive now indexes where it previously returned nothing;
* the dead `udm2_path` column is gone — it was written into every index and read by nothing;
* `build_quad_index.py --expect-quads` reconciles the index against the ordered count and fails
  loudly, so a filename regime we do not cover surfaces as an error rather than a short index.

The delivery shape is therefore no longer a constraint on the acquisition side. We still ask for
the COG `file_format` tool so the six years match 2025 byte-for-byte, but nothing breaks if a
year arrives differently.

## 2. Running this on the VM — access, keys, and long sessions

Answering the questions from our thread directly, including the ones where the answer is "no".

### 2.1 Can the VM use a key that stays on your laptop? No.

There is no forwarding mechanism for API keys — SSH agent forwarding carries SSH keys only, and
nothing equivalent exists for a bearer token like `PL_BM_API_KEY`. The process that calls
`api.planet.com` runs on the VM for ~5.5 days, so the key has to be in *that process's*
environment, on the VM. Any design that avoids this would mean the ordering loop runs on your
laptop, which is the thing we are trying to get away from.

Same for `PDG_PL_ORDERS_KEY`. Worth noting that one is transmitted to Planet by design — it is
embedded in the order JSON at `order_info["delivery"]["google_cloud_storage"]["credentials"]`
so their servers can write into the bucket.

### 2.2 Is there a managed secret store we can use? Not on this project.

Google Secret Manager would be the textbook answer, and we checked: it does not work here.
The Secret Manager API is not enabled on `pdg-project-406720`; this VM's service account holds
only `devstorage.read_only` scope, so it could not call the API even if it were; changing
instance scopes requires stopping and restarting the VM, which we will not do (it is an 8×A100
instance we won a stockout for and cannot re-acquire); and PDG project IAM is org-managed, so we
cannot grant a secret-accessor binding ourselves regardless.

So there is no managed vault in play. The two workable options are below.

### 2.3 Two ways to supply the key, and what each actually protects

**Option A — typed per session, never written to disk (our recommendation).** You SSH in, start
a `tmux` session, and run a wrapper that prompts for both keys with `read -s` and exports them
into that shell only. Nothing touches disk, nothing enters shell history, and the values vanish
when the process exits. The loop runs inside `tmux`, so it survives you disconnecting — which
is the whole point of moving to the VM (see 2.5).

**Option B — a `.env` file in your home directory, mode `600`, deleted when the year finishes.**
Identical to what you do locally; `%load_ext dotenv` / `%dotenv` reads `.env` from the working
directory and behaves no differently on a VM. Simpler, and it survives a process restart without
retyping. The cost is that it exists on disk, so it would be captured by any disk snapshot taken
during that window.

**What both actually guarantee.** This VM has other users on it (OS Login is enabled
project-wide, so you would get your own POSIX account and home directory automatically on first
login). We verified that `/proc/<pid>/environ` is mode `0400` owned by the process user, and a
`600` file is likewise unreadable by others — so **neither option is visible to other ordinary
users on the box**.

**What neither guarantees, stated plainly:** anyone with sudo on this VM can read your key —
out of process memory under Option A, out of the file under Option B. That includes me. No
arrangement on a machine we administer changes that, and it would be wrong to imply otherwise.
What the setup does prevent is the key reaching git, logs, a shared image, or other users; and
it can be removed when the run ends. Beyond that, this is a trust arrangement of the same kind
as handing a collaborator the key directly — the VM does not make it more than that.

### 2.4 Rotation — one key, rotated at the end

Planet has no self-service key rotation, and Heidi confirmed it will not issue a second key
while the first is live: **one key at a time, full stop.** So the separate project-scoped key an
earlier draft suggested is not available, and the mitigation is the one she proposed — the
existing key is used for the duration and replaced as soon as the runs finish.

That makes the ~35-day schedule in §8 load-bearing rather than merely convenient: the exposure
window is exactly as long as the programme takes, so finishing promptly *is* the security
control. It also raises the stakes on the sudo caveat in 2.3, which is why that paragraph stays
in this document rather than being softened.

### 2.5 What we will have ready before you log in

The point of the VM is that the ordering loop keeps running while your laptop is off, or on
Windows in ArcGIS. So we will set up, ahead of your first session:

- Your OS Login access to `a100-8x-train` (`us-central1-a`, `pdg-project-406720`). This needs a
  `roles/compute.osLogin` binding for your account — PDG IAM is org-managed, so if we cannot
  grant it ourselves we will get it requested. You do not need sudo.
- A single Python environment. Notebook 2 was R/Quarto; porting it to geopandas
  (`filter_to_domain.py`) means the VM needs one language rather than two.
- `run_year.sh`, a `tmux`-friendly supervisor: one command per year, detaches cleanly, restarts
  the order loop on any crash or stall, and stops rather than spinning on a crash loop.
- `check_status.py`, so progress is a command rather than an attached session.

Net effect: involvement per year is supplying the key, starting the run, and checking on it —
not babysitting a five-day process. Heidi approved requesting the `osLogin` binding
(2026-08-17); that request is the one thing that could still hold up a start date.

## 3. What changed from the notebooks — and why

Read against `HRodenhizer/circumpolar_planet_basemaps` @ `initial-download`, line numbers as of
2026-08-14. **All of this is already applied** in
[`planetscope-download/`](../planetscope-download/README.md); it is recorded here as the audit
trail for why the ported scripts differ from the originals. Heidi's review (2026-08-17) added
two changes of her own, in 3.5 and 3.6.

### 3.1 Un-hardcode the year (all three notebooks)

| File | Line | Currently |
|---|---|---|
| `1_basemap_grid_search.qmd` | 34 | `basemap_year = "2025"` |
| `2_circumpolar_south_basemap_grids.qmd` | 19 | `st_read("./data/circumpolar_basemap_grids_2025.geojson")` |
| `2_circumpolar_south_basemap_grids.qmd` | 101 | output path, inside the commented `st_write` |
| `3_order_basemaps.qmd` | 55 | `imagery_year = '2025'` |

Notebooks 1 and 3 already isolate this in one variable and interpolate downstream (`1` L87/L201,
`3` L60/L82). Notebook 2 has the year written into two literal paths and needs the same
treatment.

### 3.2 `rename_data_files()` hardcodes 2025 in its regex — **this one fails silently**

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

### 3.3 `NameError` in the batch-delete cell

`3_order_basemaps.qmd` L362-365 iterates `range(0, len(remaining_original_files), batch_size)`,
but that name is only bound inside the commented-out re-run block below (L367, L374-375). First
execution raises. `len(original_files)` is what L363 already slices.

It fails closed — nothing gets deleted — but it stops you at the very end of a multi-day run.

### 3.4 Two things that are fine as-is

**The domain path is correct.** L22's `"../RTSmapping_v2/domain/circumpolar_south_domain.geojson"`
resolves against a sibling clone of our repo. (Our local working directory happens to be named
`RTSmappingDL`, which makes it look stale when it isn't.) Worth saying that file is unchanged
since the 2025 run, so all six years get filtered against the same polygon as the epoch they
will be compared against — worth not "improving" mid-programme.

**The dedupe makes runs resumable.** L122-127 skips quads already present under the prefix, so a
five-day loop can be interrupted and restarted freely. One caveat: it keys on what is physically
under the prefix, so it is only correct once 3.2 is fixed — a silently-failed rename would make
a resumed run re-order quads that are already there.

Also uncomment `st_write` in notebook 2 (L99-104); it produces the file notebook 3 reads at L82.

### 3.5 Bounded retries, and never abandoning the run *(her change 4)*

Her original loop `sys.exit`s on a 400/401 only at `index == 0`; from index 1 on, a 400/409/500
enters an unbounded 30-second retry with no ceiling, so a mid-run credential expiry is
indistinguishable from slow progress. She asked for a cap — "perhaps no more than x times? 3?"

We went further, because a flat cap still aborts a five-day run over one quad:

* **401 fails fast.** Auth is the one error retrying cannot fix, and under the typed-per-session
  key model the fix is a restart with a fresh key — which resumes where it stopped, since
  delivered quads are skipped.
* **400/409/429/5xx back off exponentially**, five attempts, 30s→8min. This covers the transient
  Planet↔Google failures she has actually seen, where retrying always worked.
* **On exhaustion the quad is recorded and the loop continues.** `run_year.sh` then sweeps the
  recorded failures up automatically, reading the CSV so the sweep skips the bucket listing.

### 3.6 Dropping the rename — which removes two of her four problems *(her changes 2 and 3)*

The rename existed so the bucket reads tidily by hand. Our pipeline never needed it:
`build_quad_index` lists recursively (no delimiter) and matches on the **basename**, so Planet's
raw delivery — order-UUID directories and all — indexes identically to the flattened layout, and
repeat orders dedupe by newest object.

Dropping it takes out the crash-recovery redesign she was dreading *and* one of the two slow
listings she flagged, rather than solving either. `tidy_rename.py` keeps it available as
cosmetic clean-up, rewritten to derive its work from bucket state on every run (so re-running
after a failure recomputes what is left, with no checkpoint file) and to delete per-object with
404 tolerated instead of the batch delete that aborted on one missing file.

The surviving listing — the prior-delivery scan — now requests object names only rather than full
metadata. It runs once per invocation, so its cost is a slow resume rather than a slow loop.

**Verification status.** This was confirmed by reading `inference/quad_index.py` and by the test
suite, not by observing a live raw delivery: the 2025 bucket has already been fully renamed, so
no un-renamed object survives to check against. It is therefore the **first check of the 2022
pilot** — after ~20 orders land, index the raw prefix and confirm the count matches the orders
placed. If the raw form differs from what the original rename regex implies, 3.1's widened
matcher already absorbs it.

## 4. Step by step, for one year

The operational runbook lives in
[`planetscope-download/README.md`](../planetscope-download/README.md) and is the one to follow.
In outline, one command supervises all three steps:

```bash
tmux new -s planet
./planetscope-download/run_year.sh 2022      # prompts for both keys, then runs steps 1-3
```

| Step | Script | Check before moving on |
|---|---|---|
| 1 | `search_basemap_grids.py` | mosaic resolves to `global_quarterly_<year>q3_mosaic`; quad count in range; no unexpected mass of `percent_covered == 0` |
| 2 | `filter_to_domain.py` | row count against 2025's; per-column min/median/max; **record the printed count** — step 5 reconciles against it |
| 3 | `order_basemaps.py` | first order returns `202`; then `check_status.py` rather than watching the log |
| 4 | *(dropped — the rename is no longer required, see §3.2)* | |
| 5 | ours: `build_quad_index.py --expect-quads <step 2 count>` | a short index means a filename regime the matcher misses; the build fails loudly rather than silently under-indexing |

Steps 1 and 2 skip if their output exists and step 3 skips already-delivered quads, so
interrupting and resuming is always safe.

## 5. Which years, in what order

**2022 runs alone, first, end to end** — including our inference and a look at the result —
before anything else is ordered. It is the gate on the raw-delivery check (3.6), on the retry
policy (3.5), and on radiometric drift. 2022 rather than 2019 because it is mid-range: far enough from 2025 to
exercise real drift, but not at the archive's early edge where genuine Planet coverage gaps
could be misread as pipeline bugs.

If the pilot is clean: **2019 → 2020 → 2021 → 2023 → 2024**, back to back. 2019 goes first
because it gives the longest lever arm against 2025 if anything cuts the programme short. (Quota
turned out not to be a constraint — §6.1 — but sequencing by scientific value costs nothing.)

Note that 2024 is a full order like the rest. `gs://abrupt_thaw` has a 2024 q3 tree that looks
substantial by directory count, but it holds only ~8.8% of 2025's quad density inside the
columns it covers, and 296 of 2025's columns are missing from it entirely — it is an ARTS-site
subset, not a mapping layer. Same for the 2019/2021/2023 trees there.

## 5a. Outcome — 2022, the pilot year (completed 2026-08-24)

The pilot's job was to settle retry behaviour, delivery layout and drift before committing five more
years. It did, and all three answers were useful.

**309,109 quads ordered in 131.6 h at 39.1 orders/min, 0 failed, 0 supervisor restarts.** The retry
policy absorbed **92 transient failures**, and the composition matters: **84 of them were HTTP 400s
carrying a GCS 500/503**. That validates [§3.5](#35-bounded-retries-and-never-abandoning-the-run-her-change-4)'s
decision to retry 400s rather than treat them as client errors — a strict reading of the status code
would have abandoned 84 quads. Two quads Planet never delivered (`1610-1516`, and `622-1604` which
returned an empty manifest) = **0.0006 %**, against ~0.05 % missing over land in 2025.

**The rename really was unnecessary — now observed, not merely reasoned.** `build_quad_index` run
against the raw delivery returned **309,107 quads from 1,854,688 objects**, reconciling against orders
placed at **0.00 % off**, with paths of the form
`…/2022/q3/0/1515/<uuid>/global_quarterly_2022q3_mosaic/0-1515_quad_file_format.tif`. This was
[§3.6](#36-dropping-the-rename--which-removes-two-of-her-four-problems-her-changes-2-and-3)'s stated
first check and the one claim there that had only ever been verified by reading code. **Heidi never
has to run the rename or delete passes** — the two changes she found hardest.

*(The 2022 radiometric-drift question turned out to be a mis-calibrated gate rather than bad imagery;
that finding is `inference/inference.md` §5.4, which owns it.)*

**2019 is the second year** and was stopped mid-run at 215,443 / 308,686 (69.8 %, 0 failed) on
2026-08-28 — not a fault, but because it was ordering into `pdg-planet-data`, which is retired on
2026-08-31. It resumed on `rts-ops` against `gs://rts-arctic-usw1` with the same two keys; see
`computing/cutover_runbook.md` §3, and §5b below for how it finished.

## 5b. Outcome — 2019 (completed 2026-09-02), and its 227-quad coverage gap

**All 308,686 ordered, 0 outstanding failures.** The resumed leg ran 41.7 h at 37.4 orders/min:
93,421 newly ordered, 215,263 already present, 2 failed after five attempts each. The two
(`1803-1563`, `1803-1564`) were swept with `--retry-failed` at 10:21:50 and both delivered — the
retry path works, and this is the first time it has been exercised.

**Planet delivered 308,459 distinct quads from 1,850,759 objects — 227 short of what was ordered
(0.074 %).** 2022's shortfall was **2** (0.0006 %), so 2019 is ~113× the pilot's rate and the gap
needed explaining rather than absorbing.

**It is a real coverage gap, not a lost delivery.** The two hypotheses have different shapes and the
data is unambiguous:

| Test | Result |
|---|---|
| Missing quads with ≥1 missing 4-neighbour | **215 / 227 = 94.7 %** — scattered order failures would sit near 0 % |
| Geometry | diagonal stripes (`1127-1583, 1128-1581, 1129-1579, 1130-1577, 1131-1575, 1132-1573`) — col +1, row −2, the signature of an orbit track |
| Spread | 136 distinct columns × 63 distinct rows, in clusters across the domain — not one contiguous blob, and not a time-localised burst |

2019 flew a far smaller Dove constellation than 2022, and this is 60–74 °N; thin early-constellation
basemap coverage is the expected result, and the geometry says so directly.

**Ruled out — the migration.** The 2019 run ordered into `pdg-planet-data` until 08-31 16:39 and into
`rts-arctic-usw1` after, and Planet delivers asynchronously, so a delivery landing in PDG after the
final sync would have been stranded there. It did not happen: the 09-01 audit walked all **5,000,891**
PDG objects live with **0 missing** (`computing/pdg_migration.md` §5c), and a direct set-difference of
the two buckets' 2019 quad ids returned **0 in PDG but not in the destination**. For a quad to have
slipped the window it would have needed a >22 h delivery lag.

> **Downstream: 2019's `--expect-quads` is `308459`, not `308686`.** The reconciliation guard exists to
> catch short deliveries, and here it would fire on a real and permanent gap. Carry the *delivered*
> count, and expect 2020–2021 to sit between 2019 and 2022 as the constellation grew.

**Bug found while reading this: `--retry-failed` clobbers the year's status file.**
`order_basemaps.py:271` builds `Progress` from `len(grids)` and always writes
`status/{year}.json`, so the 2-quad sweep overwrote 2019's record with
`{"n_total": 2, "n_done": 2, "pct_done": 100.0}`. `alert_if_stopped.py` then read that, wrote
`.done_2019` and posted a completion message saying **"build the quad index with
`--expect-quads 2`"**. The year's real progress record is gone and the Slack instruction is wrong by
five orders of magnitude. A retry sweep should write `status/{year}_retry.json` — the alerter globs
`[0-9][0-9][0-9][0-9].json`, so that alone stops a sweep masquerading as its year.

## 6. Decisions

Answered by Heidi in the PR #61 review, 2026-08-17.

**6.1 Quota — no cap.** *"We have unlimited basemap tiles… I don't even think they track basemap
downloads."* The ~1.85M quad orders are not a constraint, and the programme is unblocked. An
earlier draft treated this as the blocking question; it was the right thing to ask and the
answer is the best available one.

**6.2 Threading — unknown, worth a bounded test.** *"I'm not sure which part is the slow part…
if you wanted, we could try threading and see if it speeds up."* `order_basemaps.py --workers N`
exists for exactly this. Pilot experiment: 200 quads serial, then 200 at `--workers 8`, compare
wall time. **Decision rule fixed up front so it is not a judgement call later:** keep threading
only if it beats serial by more than 1.5×; cap concurrency at 8, since her 409 handling already
anticipates concurrent orders. If Planet is the ceiling, as she suspects, we stay serial and the
~35-day schedule stands.

**6.3 309,100 quads per year.** *"The note probably never got updated when we switched the domain
to include areas outside of the ArcticDEM. We can take the length of the file as the truth."* The
259,783 figure in her notebook predates dropping the ArcticDEM restriction. Locked at 309,100;
`filter_to_domain.py` prints the real count per year and `build_quad_index.py --expect-quads`
reconciles against it.

**6.4 Code lives here.** *"Perhaps it is better to put it in this one so that any future runs will
have everything from start to finish in one place… I would rather not apply the changes myself."*
Ported into [`planetscope-download/`](../planetscope-download/README.md), edits applied by us.

**6.5 Option A, with the key rotated at the end.** *"Planet does not allow multiple API keys at
one time, so I will plan to use the current key for the download process and then replace it as
soon as the process is done."* See §2.4. She also approved requesting the `osLogin` binding now.

---

*Everything below is our side of the pipeline, recorded so the plan is auditable. None of it
needs action from you.*

## 7. Storage and archiving

Measured average delivery size: three columns (500, 900, 1300) sum to 19,413,767,687 bytes over
409 quads → **47.5 MB per quad** across all six delivered objects. (A single hand-picked quad
reads 52 MiB; that one is above average, which is why we sampled.)

| | |
|---|---|
| Per year | 309,100 × 47.5 MB ≈ **14.7 TB** |
| Six years | ≈ **88 TB** (~82,000 GiB) |

An earlier draft proposed deleting each year's quads once its map was approved, on the reasoning
that Planet is the archive of record. Heidi pushed back: *"Assuming we maintain the Planet
license, it is re-orderable. I still think some sort of long-term storage plan makes sense."*
That conditional is the whole argument — deletion silently bets the series on a licence renewal
nobody has committed to.

Pricing the alternatives out settles it:

| Tier | Six years | Retrieval |
|---|---|---|
| Standard | $1,640/mo · $19,677/yr | — |
| Coldline | $328/mo · $3,935/yr | $0.02/GB |
| **Archive** | **$98/mo · $1,181/yr** | $0.05/GB |

**Each year lifecycle-transitions to Archive once its map is approved**, and stays there.
$1,181/yr for the whole series removes the licence dependency for roughly a quarter of what the
delete plan would have cost in the interim, and about a sixteenth of holding it on Standard.
Two caveats worth recording: Archive has a 365-day minimum storage duration, so a year deleted
early still bills the remainder; and re-reading a full year costs ~$700 in retrieval, which
makes it a rare recovery operation rather than a workflow step.

The quads stay on **Standard through inference and review** — the QC tooling streams RGB crops
from them, so they are live until the map is signed off. Only then does the transition fire.

What we keep regardless is the probability COGs: ~0.3 TB per year at `scaled_uint8`, ~1.8 TB for
the series. The bulky half is the re-orderable half.

This draws on the budget reserved in `computing/infrastructure.md` §3 — *"pan-arctic inference +
EXTRA-channel generation + multi-year/ensemble runs, ~$40–55k"*.

## 8. Schedule — your ordering and our inference overlap

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

Review windows then run ~2 months behind each inference, and each year's quads transition to
Archive as it clears (§7).

**What we are deliberately not doing:** your grid list is `arrange(year, grid_column, grid_row)`,
so delivery sweeps west→east and completed column bands are contiguous — tempting to start
inferring mid-year. We are not. Our tiles are 512×512 windowed reads at stride 344 that straddle
quad boundaries; a tile at the advancing edge reading a not-yet-delivered quad is treated as
NoData rather than erroring, so it would silently produce degraded predictions. Doing it safely
needs a held-back one-quad buffer column plus appending to `shards/index.json`, which
`shard_tiles.py:104` writes once. That saves ~2 days out of 35 for a silent-corruption failure
mode.

## 9. What happens on our side

After your step 4, per year:

**5.** Build the quad index:
```bash
python scripts/build_quad_index.py --bucket pdg-planet-data \
    --prefix global_quarterly/<year>/q3/ \
    --output /mnt/outputs/inference/quad_index_<year>q3.csv
```
Row count is reconciled against your step-2 grid count. A large shortfall is the signature of a
silently-failed rename (§3.2) — this is the step that catches it.

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

**8.** Review the map, then transition that year's quads to Archive storage (§7).

### The VM, stated plainly

We are dedicating our 8×A100 host (`a100-8x-train`, 96 vCPU) to this programme, but to be
accurate about why: **the GPUs contribute nothing to the download.** Planet delivers server-side
straight into GCS — no imagery transits any machine we own, and the order loop is a
single-threaded API caller that would run just as fast on a laptop. The host earns its place
because it can hold a month-long order loop at zero marginal cost (it is already never stopped),
and because its otherwise-idle GPUs run the concurrent per-year inference that makes §8's
schedule work.

**Environment:** the acquisition scripts need only
`planetscope-download/requirements.txt` — geopandas, shapely, pandas, requests and
google-cloud-storage. Porting notebook 2 off R (§3, `filter_to_domain.py`) removed the R/Quarto
half, so the VM needs one language rather than two, and the `planet` SDK dependency went with it
(order state is read over the same HTTP session that places the orders).

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
                                ▼  review, then Archive that year's quads
```

All figures re-derived against live buckets on 2026-08-14; sampling method stated inline so they
can be checked.
