# Cutover — the last 24 hours before PDG access closes

**All PDG access closes Monday 2026-08-31.** Everything below assumes the bulk copy is done and
gate rows 2, 4, 5, 7 are already recorded in [pdg_migration.md](pdg_migration.md) §5. This file is
the *ordering*, and the ordering is the whole point: several steps are only correct in one
sequence, and getting them wrong loses data rather than merely wasting time.

> **The rule this doc exists to enforce:** a producer's parity can only be proven against a
> **stopped** source. Everything here is arranged so each producer is frozen, drained, verified,
> and repointed *once* — never measured while still being written.

---

## 0. The three live writers

Nothing else in the migration can lose data. These can:

| Writer | Writes to | Owned by | Frozen at |
|---|---|---|---|
| ~~S2 export driver~~ | `rts-mapping-v2-usw1/S2_RGB/` | us, `tmux` on the master | **stopped 2026-08-28** (§2) |
| Planet order loop | `pdg-planet-data/global_quarterly/` | Heidi | §3 |
| Review app | `rts-mapping-v2-usw1/inference/2025q3_south/review/` | reviewers, continuously | §4 |

Each is *live*, so each already mismatches its copy. **That is the source moving, not the copy
failing** — but the two acquisition legs move in different ways, and the difference decides what
the final sync has to do (measured 2026-08-28):

- **S2 rewrites in place.** Counts are *identical* (14,780 = 14,780) and 823 MB of bytes differ, and
  the drift grows between measurements. A count-only check passes this falsely. The delta-sync
  **must** carry `--overwrite-when=different`; a plain "copy what's missing" sync would copy nothing
  and leave stale composites behind.
- **Planet appends.** Counts differ (src 4,953,262 vs dst 4,939,100) and the destination count is
  *exactly* what the `planet-quads` job reported copying, so nothing was dropped — 14,162 new quads
  (+97.2 GB) landed after the job listed the source.

So a **count** difference on `planet` before the freeze is expected; on `s2` it would not be.

---

## 1. Before you start (any time Sunday)

```powershell
# Nothing here mutates anything. Do it early so surprises surface early.
python scripts/verify_migration_parity.py --json parity_pre.json
gcloud compute instances list --project=abruptthawmapping
gcloud compute instances list --project=pdg-project-406720
```

`experiments`, `ee_mirror`, `ee_staging` and `interannual` already MATCH exactly (2026-08-28) and
should still. `s2` and `planet` mismatch until frozen, in the two distinct ways §0 describes. If an
object **count** differs on any *frozen* prefix, stop and find out why before going further.

## 2. S2 export — DONE 2026-08-28, abandoned on instruction

**No longer a cutover step.** The export is stopped and will not be restarted; resuming it
(and against which bucket) is a later decision, deliberately deferred.

Stopping the driver was not enough, and this is the part worth remembering: the driver only
*submits* `Export.image.toCloudStorage` tasks — Earth Engine's servers do the writing. Killing it
left **2,000 PENDING tasks** queued (373 for 2019, the rest 2022) that would have gone on writing
into `rts-mapping-v2-usw1` right up to deletion, and failed after it. So:

```
kill <run_stage.py>                      # the interannual driver
sudo docker stop <export container>      # the driver runs inside it
ee.data.cancelOperation(name)  × 2,000   # the queue, which is the real writer
```

All 2,000 cancelled, 0 failures, no active EE task remains. `S2_RGB/` is now **frozen**, so its
parity can be measured for real.

A second thing surfaced while cancelling: `abruptthawmapping` is in Earth Engine **restricted
mode** ("exceeded the compute quota of its noncommercial tier"). That is why 1,999 tasks sat
PENDING behind a single RUNNING one — the export was not going to finish on this schedule
whatever we did. Anything that resumes S2 has to resolve the EE quota first.

Remaining for this leg — one delta-sync against the frozen source, then verify:

```powershell
gcloud transfer jobs create gs://rts-mapping-v2-usw1 gs://rts-arctic-usw1 `
    --project=abruptthawmapping --name=s2-final --include-prefixes="S2_RGB/" `
    --overwrite-when=different
```
```bash
python scripts/gcs_parity.py --src gs://rts-mapping-v2-usw1/S2_RGB --dst gs://rts-arctic-usw1/S2_RGB
```

The pre-freeze measurement showed **identical counts (14,780) with 5 objects differing** on
size/MD5, all under `2019_south/` — exactly the cells being rewritten. `--overwrite-when=different`
is what carries them; a "copy what's missing" sync would copy nothing.

## 3. Freeze Planet acquisition (needs Heidi)

Heidi stops her order loop, then:

```powershell
gcloud transfer jobs create gs://pdg-planet-data gs://rts-arctic-usw1 `
    --project=abruptthawmapping --name=planet-final --overwrite-when=different
```
```bash
python scripts/verify_migration_parity.py --pair planet
```

She then restarts against the new bucket. **This is a one-flag change** — same Planet API key, same
`PDG_PL_ORDERS_KEY`, same CSV, because nothing was renamed and `planet-orders@` now has
`objectUser` on the new bucket:

```bash
./planetscope-download/run_year.sh --year 2019 --bucket rts-arctic-usw1
```

(The default in `order_basemaps.py` is now `rts-arctic-usw1` too, so a bare invocation is also safe.)

## 4. Cut the reviewers over

**Order matters here more than anywhere else.** Claims and verdicts live in the bucket, so a verdict
submitted to the old app after the final sync is simply lost.

1. **Stop the old app** — this is the freeze:
   ```powershell
   gcloud compute instances stop rts-review-vm --project=pdg-project-406720 --zone=us-west1-a
   ```
2. **Final sync of the review state:**
   ```powershell
   gcloud transfer jobs create gs://rts-mapping-v2-usw1 gs://rts-arctic-usw1 `
       --project=abruptthawmapping --name=review-final `
       --include-prefixes="inference/2025q3_south/review/" --overwrite-when=different
   ```
3. **Now run the gate-row-7 check that was deliberately skipped** — it mutates state, so it could
   not be run before this point:
   ```bash
   curl -s http://34.83.225.204/api/next            # claim
   curl -s -X POST http://34.83.225.204/api/submit -H 'Content-Type: application/json' -d '{...}'
   curl -s -X POST http://34.83.225.204/api/submit -H 'Content-Type: application/json' -d '{...}'   # expect 409
   ```
4. **Tell the reviewers**: <http://34.83.225.204/>. They need nothing else — no account, no install,
   the name is remembered in `localStorage`. Send the heads-up *before* Monday so the switch is just
   a link.
5. Retire **both** old deployments — the VM and the Cloud Run service that was superseded on
   2026-08-04 but never torn down.

> The URL changes and cannot be preserved: `8.229.247.193` is a PDG static address, and static IPs
> do not transfer between projects.

## 5. The gate

Every row of [pdg_migration.md](pdg_migration.md) §5 must pass — with §2's and §3's parity measured
against the **frozen** sources, not the provisional numbers. Only then does §6 run.

## 6. Teardown — irreversible

Follow [pdg_migration.md](pdg_migration.md) §6 exactly. The two easy mistakes:

- **Disable soft-delete before deleting buckets.** 7-day retention means deleted objects keep
  billing, which is the opposite of the point.
- **`pdg-planet-data` is handed back, not deleted.** It is PDG's bucket; deleting our
  `global_quarterly/` prefix would mean ~5.5 M destructive API calls in someone else's project for
  no saving of ours. Tell Luigi/Todd it is theirs to dispose of.

## 7. If the deadline arrives with work outstanding

Priority order, most to least costly to lose:

1. **Anything local-only on the master** — already drained (2022 tile lists, `south_t65`); the MAE
   corpus is a recorded, deliberate loss.
2. **Review verdicts** — human effort, unreproducible. §4 before everything else.
3. **Delivered Planet quads and S2 composites** — expensive but re-orderable/re-exportable.
4. **Everything else** — reproducible from what is already migrated.

Storage in PDG bills until access closes, so an unfinished copy is a *data* problem, not a cost one.
