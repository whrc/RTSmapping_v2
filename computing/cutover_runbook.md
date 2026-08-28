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
| S2 export driver | `rts-mapping-v2-usw1/S2_RGB/` | us, `tmux` on the master | §2 |
| Planet order loop | `pdg-planet-data/global_quarterly/` | Heidi | §3 |
| Review app | `rts-mapping-v2-usw1/inference/2025q3_south/review/` | reviewers, continuously | §4 |

Each is *live*, so each already shows a byte mismatch against its copy — object counts match, byte
totals do not. **That is the source moving, not the copy failing.** S2 measured +780 MB drift on
2026-08-28 with counts identical at 14,780.

---

## 1. Before you start (any time Sunday)

```powershell
# Nothing here mutates anything. Do it early so surprises surface early.
python scripts/verify_migration_parity.py --json parity_pre.json
gcloud compute instances list --project=abruptthawmapping
gcloud compute instances list --project=pdg-project-406720
```

Expect `inference`, `experiments`, `ee_mirror`, `ee_staging`, `interannual` to MATCH, and `s2` /
`planet` to MISMATCH on bytes only. If an object **count** differs on a frozen prefix, stop and
find out why before going further.

## 2. Freeze the S2 export

```bash
ssh a100-8x-train
tmux ls                      # find the export session
tmux attach -t 0             # Ctrl-C the driver, or touch its STOP sentinel
```

The export is resumable and 2022 is only ~10 % delivered, so stopping costs nothing but the
in-flight cells. **Do not** kill the whole tmux server — Heidi's work may share it.

Then delta-sync and verify against the now-frozen source:

```powershell
gcloud transfer jobs create gs://rts-mapping-v2-usw1 gs://rts-arctic-usw1 `
    --project=abruptthawmapping --name=s2-final --include-prefixes="S2_RGB/" `
    --overwrite-when=different
```
```bash
python scripts/verify_migration_parity.py --pair s2      # must now MATCH on bytes too
```

Restart the export **against the new bucket** only after that passes. `interannual_inference/config.yaml`
already points at `gs://rts-arctic-usw1`, so on `rts-ops`:

```bash
cd /opt/rts/RTSmapping_v2 && git pull            # picks up the repointed config
python interannual_inference/drive.py --year 2022
```

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
