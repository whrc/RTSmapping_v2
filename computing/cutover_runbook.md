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
| ~~Planet order loop~~ | `pdg-planet-data/global_quarterly/` | Heidi | **stopped 2026-08-28** (§3); delivery tail still draining |
| ~~Review app~~ | `rts-mapping-v2-usw1/inference/2025q3_south/review/` | reviewers, continuously | **cut over 2026-08-28** (§4) |

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

## 3. Planet acquisition — stopped 2026-08-28, restart is Heidi's

The 2019 run was live and ordering into `pdg-planet-data`, the bucket that dies on the 31st. It
would have finished ~Sunday with about 18 hours to spare, which is not enough margin to be worth
having. Stopped at **215,443 / 308,686 orders placed (69.8 %), 0 failed**.

Stopping needed two actions, not one — see the acquisition README's *Stopping* section: the STOP
sentinel only stops the *supervisor restarting* the ordering step, so the sentinel goes first and
the ordering process is then ended.

**Order matters for what comes next, and it is not the obvious order.**

1. **Wait for the delivery tail.** Orders placed ≠ quads delivered: Planet writes into the bucket
   asynchronously, so `pdg-planet-data` keeps growing for a while after the loop stops. Count
   `global_quarterly/2019/q3/` until it stops climbing.
2. **Delta-sync PDG → new bucket**, so the destination holds everything already delivered:
   ```powershell
   gcloud transfer jobs create gs://pdg-planet-data gs://rts-arctic-usw1 `
       --project=abruptthawmapping --name=planet-final --overwrite-when=different
   ```
3. **Only then does Heidi restart**, on `rts-ops`:
   ```bash
   gcloud compute ssh rts-ops --zone=us-west1-a --project=abruptthawmapping --tunnel-through-iap
   tmux new -s planet && cd /opt/rts/RTSmapping_v2
   ./planetscope-download/run_year.sh 2019
   ```

Step 2 **must** precede step 3. `order_basemaps.py` decides what to skip by listing the *delivery*
bucket. Restart against the new bucket before the sync and it cannot see the quads that are still
only in PDG, so it re-orders them — burning Planet quota to re-buy imagery we already own.

No new credentials: same `PL_BM_API_KEY`, same `PDG_PL_ORDERS_KEY`, and `--bucket` is not passed
because `order_basemaps.py` already defaults to `rts-arctic-usw1`. Heidi holds `roles/owner` on
`abruptthawmapping`, which carries both OS Login and IAP tunnel access, so nothing needs granting.

Two traps for her: `rts-ops` has **no external IP**, so it is `gcloud compute ssh ...
--tunnel-through-iap` and never plain `ssh`; and her Linux username changes from
`ext_hrodenhizer_woodwellclimate_` to `hrodenhizer_woodwellclimate_org`, because she is in-org on
the new project and was out-of-org on PDG. That asymmetry presents as `Permission denied
(publickey)`.

Finally: her acquisition state (`data/`, `status/`, `logs/`) is mirrored to
`gs://rts-arctic-usw1/planetscope-download/` and restored on `rts-ops`, and the venv, work dir,
Slack webhook and alert cron are in place there — none of which existed before 2026-08-28.

## 4. Cut the reviewers over — DONE 2026-08-28

**Order matters here more than anywhere else.** Claims and verdicts live in the bucket, so a verdict
submitted to the old app after the final sync is simply lost. Stop first, then sync.

1. **Stopped the old app** — `gcloud compute instances stop rts-review-vm --project=pdg-project-406720
   --zone=us-west1-a`. This is the freeze; nothing else is safe until it is done.
2. **Diffed the two stores before copying anything**, which is the step this section originally
   lacked:
   ```bash
   python scripts/gcs_parity.py --src gs://rts-mapping-v2-usw1/inference/2025q3_south/review                                 --dst gs://rts-arctic-usw1/inference/2025q3_south/review
   ```
   Result: **nothing missing, nothing differing, 5 objects extra at the destination.** Reviewers had
   already been cut over and one had submitted a batch on the new app, so the new store was *ahead*
   of the old one. **The final sync was correctly a no-op.**

   > The instinct this section used to encode — fire an `--overwrite-when=different` sync old → new —
   > is the wrong shape once the new app is live. Diff first. Verdicts are unreproducible human
   > effort, and "sync the old one over the new one" is how you lose them.

3. **The deferred end-to-end check.** Two things the original recipe got wrong: the submit endpoint
   is `POST /api/batch`, not `/api/submit`; and a duplicate submit is **200, not 409** — it is
   idempotent by design (`review/app.py`: *"a retry is accepted, not doubled"*). The **409** comes
   from `GET /api/batch/{id}` when reopening an already-submitted batch.

   It also should not fabricate a verdict. `merge_review_verdicts.py` pools every
   `verdicts/*.jsonl` into the verified inventory, so a synthetic submission contaminates the
   product. What the migration actually needs to prove is that this deployment reads and writes the
   *new* bucket — a claim marker proves that without inventing science:

   ```bash
   curl -s "http://34.83.225.204/api/next?reviewer=migration-check"   # claim -> b00046, 201 items
   gcloud storage ls gs://rts-arctic-usw1/.../review/claims/b00046    # the write landed
   curl -o /dev/null -w "%{http_code}" http://34.83.225.204/api/batch/b00044   # 409 submitted
   curl -o /dev/null -w "%{http_code}" http://34.83.225.204/api/batch/b99999   # 404 unknown
   gcloud storage rm gs://rts-arctic-usw1/.../review/claims/b00046    # release it
   ```

   **Delete the claim afterwards.** `STALE_AFTER_S` is one week, so a claim left behind blocks that
   batch from every reviewer until next Friday.

4. **Reviewers told**: <http://34.83.225.204/> — done. They need nothing else: no account, no
   install, the name is remembered in `localStorage`.
5. Retire **both** old deployments — the VM (now stopped) and the Cloud Run service that was
   superseded on 2026-08-04 but never torn down.

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
