# PDG wind-down — migration to `abruptthawmapping` (Sept 2026)

**Status: IN PROGRESS.** Runbook, decision record, and the parity evidence that gates
deletion. Started 2026-08-27.

GCP funding for `pdg-project-406720` ends **2026-09-07**. Everything this project runs on
lives there. This document records what moved, where it went, why, and what was checked
before anything was deleted.

> **Abort position:** every step is a *copy*, never a move. Until §6 runs, the PDG originals
> are intact and the migration can be abandoned with no loss. Only §6 is irreversible.

---

## 1. What was there

Measured 2026-08-27 (`storage.googleapis.com/storage/v2/total_bytes`, live-object series).

| Source | Size | Objects | Fate |
|---|---|---|---|
| `pdg-planet-data/global_quarterly/{2019,2022,2025}` | 34.5 TB | ~5.5 M | copy → Coldline |
| `rts-mapping-v2-usw1/S2_RGB/` | 18.7 TiB | ~9 k | copy → Coldline (`2024_train` Standard) |
| `rts-mapping-v2-usw1/inference/` | ~210 GB | ~41.7 M | copy → Standard |
| `rts-mapping-v2` | 254 GB | — | copy → Standard, restructured |
| `rts-mapping-v2-usc1/ee_mirror/` | 13 GB | ~1.6 k | copy → **us-central1 only** |
| `rts-mapping-v2-usc1/staging/` | 247 GB | — | **not copied** — stale Banks slice, disposable |
| `a100-8x-train` local disks | 741 GB + 21 GB | — | audit, upload remainder, delete VM |
| `pdg-artifact-registry` (`rts-train:v2`, `rts-infer:v1`, `rts-review:v1`) | 10.8 GB | — | push to new repo |
| EE assets `projects/pdg-project-406720/assets/*` | small | 4 | re-ingest |

**Not ours, untouched:** `download-vm`, `gke-water-cluster-*`, `pdg-storage-default`,
`argo-filestore`, the `lake_drainage_test` image.

`infrastructure.md` used to list `/mnt/argo_filestore` (2 TB PDG Filestore) as our persistent
shared storage. Checked 2026-08-27: **not mounted on the master, holds nothing of ours.**
Nothing was stranded there; the doc was stale.

## 2. Decisions

1. **Compute pauses.** `abruptthawmapping` has zero A100/L4 quota and no credit, and the
   interannual campaign was 9 % into 2022's S2 export and 3 % into 2019's — it could not
   finish before the cliff. No GPU quota requested. **Inference is paused indefinitely.**
   Acquisition (Planet + S2 export) continues: it is server-side work costing almost nothing
   in GCP, and it is the campaign's critical path.
2. **Storage tiered by lifecycle**, not kept uniformly hot. ~$230–300/mo instead of ~$1,100.
3. **The rating campaign ran to the last moment** and cut over at the end, rather than
   pausing for the migration.
4. **All 41.5 M per-tile probability COGs migrated** — no dropping, no tarballs.
5. **The office PC (`ARCHITECTURE`) becomes the control node**; unattended work moves to a
   small always-on `rts-ops` VM. See [README.md §2](README.md).

### Two findings worth keeping

- **Do not tier the per-tile probs.** Nearline/Coldline/Archive bill a **128 KiB minimum per
  object**. 41.5 M objects averaging ~2.5 KB would bill as 5.3 TB rather than 100 GB —
  Coldline would cost roughly 10× Standard. The obvious optimisation is backwards here.
- **`ee_mirror` must stay in us-central1.** Earth Engine's `loadGeoTIFF` reads US-CENTRAL1
  buckets only. That bucket exists for that reason and nothing else; folding it into the
  us-west1 bucket would silently break the published map.

## 3. Path history — old → new

Old paths are embedded in MLflow run metadata and in archived run configs, and are **not**
rewritten. This table is how you read them.

| Old | New |
|---|---|
| `gs://rts-mapping-v2/training/…` | `gs://rts-arctic-us/training/…` |
| `gs://rts-mapping-v2/label_sources/…` | `gs://rts-arctic-us/labels/…` |
| `gs://rts-mapping-v2/RTS_MODEL_V2/…` | `gs://rts-arctic-us/experiments/v1.0/…` |
| `gs://rts-mapping-v2/RTS_MODEL_V1_1/…` | `gs://rts-arctic-us/experiments/v1.1/…` |
| `gs://rts-mapping-v2/RTS_MODEL_V21/…` | `gs://rts-arctic-us/experiments/v2.1/…` |
| `gs://rts-mapping-v2/RTS_MODEL_V2_scale05/…` | `gs://rts-arctic-us/experiments/scale05/…` |
| `gs://rts-mapping-v2/runs/…` | `gs://rts-arctic-us/experiments/v1.0/runs/…` |
| `gs://pdg-planet-data/global_quarterly/<y>/q3/…` | `gs://rts-arctic-usw1/imagery/planet_q3/<y>/…` |
| `gs://rts-mapping-v2-usw1/S2_RGB/<y>_<region>/…` | `gs://rts-arctic-usw1/imagery/s2_composites/<y>_<region>/…` |
| `gs://rts-mapping-v2-usw1/inference/…` | `gs://rts-arctic-usw1/inference/…` *(prefixes unchanged below the bucket)* |
| `gs://rts-mapping-v2-usc1/ee_mirror/…` | `gs://rts-arctic-usc1/ee_mirror/…` *(unchanged below the bucket)* |

`inference/` keeps its internal layout deliberately. Those prefixes are read by `claim.py`,
the crop server's prefix check, `chip_index`, the batch manifest and the shard queue;
renaming them would mean code changes plus a full re-verification of the review stack during
the tightest week of the deadline, in exchange for tidiness. The bucket rename alone already
touches every one of those paths.

## 3b. Day-1 audit — what on the master is genuinely local-only

Run 2026-08-27, read-only. The headline "741 GB + 21 GB on local disks" is misleading: most
of it is either mirrored already or regenerable by an existing convention. Comparing each
local tree against its GCS mirror:

| Local tree | Local | GCS mirror | On GCS | Reading |
|---|---|---|---|---|
| `v1.0/runs` | 223 GB | `RTS_MODEL_V2/runs/` | 50 GB | Expected — the mirror is deliberately *slim* (`best_deployment.pth` + config + summary + log + figures). The gap is `resume_latest-*.pth`, local-only by convention |
| `v2.1/PRETRAIN_CORPUS` | **189 GB** | — | — | **The one real decision.** The 295,429-tile MAE corpus. The v2.1 programme is closed negative, and the corpus is rebuildable from v1.0 data + the corpus builder |
| `v2.1/runs` | 45 GB | `RTS_MODEL_V21/` | 5.7 GB | 12 × `resume_latest-*.pth` = 48 GB, regenerable training state |
| `inference/south/review` | 68 GB | `inference/2025q3_south/internal/` | 67 GiB | Mirrored — the crop/chip archive |
| `inference/south/products_local` | 15 GB | `inference/2025q3_south/products/` | 25 GiB | Mirrored |
| `inference/tiles_2022q3_*.csv` | **7 GB** | — | — | **Not mirrored.** The 2022 tile grid; small, regenerable in ~25 min, but cheap to keep |
| `v1.0/data_local` | 100 GB | `training/v1.0/` | 44 GiB | Staged working copy; the mirror is the SSoT |
| `nvme_scratch/south_t65` | 21 GB | `inference/2025q3_south/products/` | — | The 0.65-contour build; **confirm object-by-object before deleting** — no audit has ever covered this disk |

So Phase E's real payload is **~7 GB that must be uploaded** (the 2022 tile lists), one
**189 GB judgement call** (the MAE corpus), and a verification pass over everything else.
That is a very different afternoon from "drain 762 GB".

**The MAE corpus decision.** Precedent exists: the five 26 GB sat-7B dead-arm checkpoints were
deliberately dropped in the 2026-07-16 cleanup, with the rationale recorded in
`artifact_inventory.md`. The corpus is the same shape of decision at 189 GB — a closed
programme's input, reproducible from data we are keeping. Coldline would cost ~$0.76/month,
so this is not really about money; it is about whether a rebuild is ever wanted.
**Left for the user; not assumed either way.**

## 4. Runbook

Detailed phases, dates and rationale: the approved migration plan. Condensed here as the
operational sequence.

- **A (27–28 Aug)** — request the human-blocked items (Planet delivery SA key, `roles/editor`
  and `roles/iap.tunnelResourceAccessor` for `yyang@`, licence confirmation); notify PDG;
  create the three buckets and the `rts` Artifact Registry repo as `rtsmapping@`; create
  `rts-ops`; copy the deployment packages first and verify their MD5s; run the throughput
  probe and the read-only local-disk audit.
- **B (28 Aug – 1 Sep)** — bulk copy, largest first, from `rts-ops`.
- **C (1–3 Sep)** — per producer: stop submitting → drain → freeze → delta-sync → verify
  parity on the frozen source → repoint → restart → watch it work. One at a time.
- **D (2–4 Sep)** — `rts-review`; push the four images.
- **E (3–5 Sep)** — drain the master's disks; re-ingest the EE assets and republish the app.
- **F (5 Sep)** — the §5 gate. 7 Sep held as slack.
- **G (6 Sep)** — cut reviewers over; delete PDG VMs, IP, Cloud Run, images, our buckets.

### Copy method

`gcloud storage cp -r -n` / `rsync` run **from `rts-ops`** under `yyang@`'s ADC — that
identity already has read on the PDG buckets and write on `abruptthawmapping`, so no IAM
change is needed, which matters in an org-managed project where we hold no `setIamPolicy`.
Copies are server-side; the VM only orchestrates. The two high-object-count legs are chunked
by natural sub-prefix (shard id, quad column) at ~32-way concurrency rather than run as one
giant `rsync`, so they are resumable and independently verifiable.

Same-location pairs throughout (US→US, us-west1→us-west1, us-central1→us-central1), so there
is no network egress charge. Expect ~**$235** of Class-A operations for ~47 M object writes.

## 5. Verification — the gate before deletion

Nothing in §6 runs until every row passes. *(To be filled in as the migration proceeds.)*

| # | Check | Result |
|---|---|---|
| 1 | Parity per prefix — object count and total bytes, source vs destination. Acquisition prefixes must use the **frozen** Phase-C measurement, not the provisional Phase-B one | — |
| 2 | `gcloud storage hash` on 200 random objects per leg, plus *all* of `packages/seed{42,43,44}` and `normalization_stats.json` | — |
| 3 | Frozen-model reload — migrated packages reproduce the recorded 3-seed anchor (`anchors_all_match: true`) | — |
| 4 | No dead references — repo grep for old bucket/project names returns only historical prose | — |
| 5 | Cold start on `rts-ops` over IAP; `status.py` reproduces the campaign grid; a dry-run stage resolves; survives a reboot | — |
| 6 | Live acquisition unbroken — `check_status.py` advancing into the new bucket, `ord/min` back at 38–39 | — |
| 7 | Review app end-to-end on the new host — rater page, 301 batches / 60,167 items, a crop served, manifest still 404, claim → submit → idempotent retry → 409 | — |
| 8 | Public EE map renders from the new assets and the new usc1 mirror | — |
| 9 | Master drained — nothing durable local-only, `/mnt/nvme_scratch` included | — |
| 10 | Billing — no `rts-*` resource left in PDG; `abruptthawmapping` line items as expected | — |
| 11 | Docs true — the [README.md](README.md) registry matches `gcloud compute instances list` | — |

## 6. Teardown (irreversible — only after §5)

1. Stop cron and `tmux` on the master; final delta-sync; re-list each source prefix once more
   for anything that landed late.
2. Cut reviewers to the new URL, then retire `rts-review-vm`.
3. Delete `a100-8x-train` (**only** after its disks are drained and signed off) and
   `rts-review-vm`; release `rts-review-vm-ip`; delete the Cloud Run service `rts-review`.
4. Delete our three images from `pdg-artifact-registry`; leave `lake_drainage_test`.
5. **Disable soft-delete first** — 7-day retention means deleted objects keep billing — then
   delete `rts-mapping-v2`, `rts-mapping-v2-usw1`, `rts-mapping-v2-usc1`.
6. **`pdg-planet-data`: hand back, do not delete.** It is PDG's bucket, its funding lapses
   with the project regardless, and deleting our `global_quarterly/` prefix would mean ~5.5 M
   destructive API calls in someone else's project for no saving of ours. Tell Luigi/Todd it
   is theirs to dispose of; offer to run the delete if they want it.
7. Confirm no `rts-*` VM, disk, address, Cloud Run service or bucket remains in PDG.
