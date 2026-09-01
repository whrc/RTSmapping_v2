# Working Status — RTSmapping_v2

Project diary: where we are and where we're going. **Rolling** — the *Now* section is overwritten each
update and the *Just completed* step rolls forward; old detail is not accumulated here (history lives in
git and `docs/archive/`). Experiment numbers + the locked recipe are **not** duplicated here — they live
in `docs/experiment_ledger.md` (the SSoT); this doc links to it. Update ritual: see `CLAUDE.md`.

---

## Project Summary

Semantic segmentation of **Retrogressive Thaw Slumps (RTS)** in Arctic satellite imagery (60–74°N).
Train on 2024 PlanetScope Quarterly Basemap (RGB, ~3 m), deploy inference on 2025 imagery for a
pan-arctic RTS survey map. Solo research project — flat code, minimal abstraction.

**Core constraints** (non-negotiable, see `CLAUDE.md`): CRS EPSG:3857 · tile 512×512 · labels
0=bg/1=RTS/255=ignore · per-dataset z-score norm (`normalization_stats.json`) · seed 42, deterministic.

**Stack**: PyTorch 2.x + `segmentation_models_pytorch` (UNet++/EffB5), albumentations, rasterio,
geopandas, MLflow. Compute: 8× A100-80GB (`a100-8x-train`) for training; 32× L4 (us-west1) provisioned
for inference. Docker `rts-train:v2`. Data in `gs://abrupt_thaw/` + `gs://rts-mapping-v2*`.

**Dataset**: v1.0 standard (22,259 tiles; 1,718 pos / 20,541 neg), corrected leakage-free region split.
**Diagnosis**: representation-limited, not data-volume- or capacity-limited (see ledger family B).

---

## Rolling progress

### Just completed
**PDG migration complete (2026-08-26 → 09-01).** 60.19 TB / 47,553,531 objects, four buckets, the
rating app, three Docker images and the EE assets behind the published map, all moved off
`pdg-project-406720` before its access closed on 2026-08-31. Runbook `computing/pdg_migration.md`;
teardown `computing/teardown.ps1`.

**The gate closed at 11 of 11.** Row 1 compared *every* object rather than a sample — `planet`
5,000,891 obj / 39.46 TB and `s2` 14,780 / 20.72 TB byte-identical, `inference` `missing = 0` in all
2,161 chunks with the 5,381-object surplus reconciling to the byte. Row 3 anchored the frozen model
to `model_checkpoint_sha` recorded in production manifests on 2026-07-07, not to the copy. Row 7 cut
reviewers over with zero verdict loss — the new app was already *ahead* of the old, so the runbook's
prescribed old→new sync would have been the wrong instinct and was correctly a no-op. Row 8 was
verified against the published bundle after the first publish silently served the old script at
HTTP 200: **EE Apps publish from a saved script path, not the editor buffer.**

**Five findings that outlive the migration.**
- **Every inventory was wrong, five times, the same way** — a prefix listed but never opened.
  `usc1/staging/` (2,661 obj / 265 GB) was caught with the delete already running; `pdg-storage-default`
  was cleared on its top-level names and hid three of our 2024 RTS archives in `working/` (§5d).
  Never clear a bucket on its top-level prefix names.
- **The parity gate had a hole exactly where it hurt.** GCS stores **no `md5Hash` for composite
  objects**, so an MD5-only comparison saw `"" == ""` and passed the biggest objects on *size alone*.
  `gcs_parity.py` now falls back to CRC32C and reports rather than passes when no checksum is shared;
  25 tests.
- **The drain audit measured the wrong thing** — it ordered by size, so it missed four small,
  unreproducible trees (Heidi's acquisition state, the 128 MB MLflow store, `multiscale_poc_eval`, and
  the two VM-creation scripts that existed only on the VM they created). "What is large" and "what is
  unreproducible" are different questions; only the second gates deletion.
- **A dead bucket path was visible to the public.** `ee_south_app.js:524` was not a comment but a
  `ui.Label` advertising `gs://rts-mapping-v2-usw1/...` to every visitor.
- **Killing a driver does not stop Earth Engine.** `Export.image.toCloudStorage` only *submits*;
  2,000 PENDING tasks stayed queued writing into the dying bucket after the S2 driver was killed. All
  cancelled, 0 failures. `abruptthawmapping` is in EE **restricted mode**, which is why 1,999 sat
  behind one RUNNING — anything resuming S2 must fix that quota first.

**Nothing of ours remains in PDG.** Every live service API re-checked 2026-09-01: no instance, disk,
address, Cloud Run service, image, firewall rule or IAM binding of ours; Functions/Dataproc/Cloud SQL
APIs were never enabled; all 9 PDG Earth Engine assets have counterparts in `abruptthawmapping`.
`pdg-planet-data` is PDG's and **verified safe for them to delete** — all 5,000,891 objects present in
`gs://rts-arctic-usw1`, 0 missing / 0 differing. `rts-mapping-v2-usw1` is emptying unattended under an
age-0 lifecycle rule (20.9 TB → 56 GB in its first pass); it needs nobody and finishes whether or not
we still hold access.

### Now
**Planet 2019 acquisition, running on `rts-ops` in the new project.** 267,548 / 308,686 (86.7 %),
52,283 ordered, 2 failed, ~37 orders/min, ETA ~18 h. It is entirely inside `abruptthawmapping` and
survives PDG's closure.

One artefact worth knowing before it confuses someone again: on restart the displayed percentage
**drops** — `list_delivered()` finds every already-delivered quad up front (215,263 here, logged at
start), but `order_basemaps.py` increments `skipped` one row at a time as the loop walks the 308,686-row
grid. Six minutes after the 08-31 restart it read 45.2 %; it caught up within the hour and `skipped`
settled at exactly 215,263. Seeding `n_skipped` with `len(delivered)` at startup would fix the display
— hold that change until 2019 finishes, because `/opt/rts/RTSmapping_v2` is the live checkout the run
reads from (`pdg_migration.md` §4a).

**Next**, once 2019 completes: build its quad index with `--expect-quads 308686`, then Phase 0 of the
interannual run below.

### Future plans (inference phase — full plan in `.claude/plans/elegant-exploring-lemur.md`)
1. **Phase 0** — finish the 2025_south S2 export (GEE-throttled), build + upload the `s2_index` (NDVI
   windowing) + domain↔S2 coverage audit with a residual-gap policy; build `scripts/inference_progress.py`
   (terminal dashboard + Claude-watcher JSON).
2. **Phase 1** — GCS shard-claim queue: `scripts/shard_tiles.py` (spatial-contiguous shards), `inference/
   claim.py` (atomic claim/done/stale-reclaim), `scripts/run_inference_worker.py`; refactor `inference.py`
   body into `run_inference(tiles_df, …)`; tests (atomic mutual-exclusion, reclaim, done-skip, exactly-once).
3. **Phase 2** — rebuild `rts-infer:v1` (current code + cv2 baked + MLflow `<3.0` per requirements; push via
   ADC); build the 3 per-seed deployment packages; create the output-bucket layout (hierarchical shard-scoped
   prefixes, one manifest/shard).
4. **Phase 3** — fleet provisioning (`create_inference_fleet.sh`, `rts-infer-{1..4}`) + pre-flight: L4 quota,
   1-VM startup test, drift check, Banks Island RGB+NDVI end-to-end + multi-VM claim collision check +
   kill/restart drill, benchmark → shard size + output dtype + ETA.
5. **Phase 4** — launch 40 workers (explicit go), monitor + auto-stop watchdog (stops `rts-infer-*` only;
   never the A100 master), end-of-run exactly-once coverage reconciliation; then stop (not delete) the L4 fleet.
6. **Deferred (per user)**: post-inference vectorization → products (cut from retained prob COGs, no GPU rerun);
   North S2-RGB model; v3 backlog (re-stage, MAE SSL, multi-scale — see ledger "Deferred to v3").

---

## Pointers

- **Experiments / recipe / findings** → `docs/experiment_ledger.md` (SSoT)
- **Visual report** → `docs/report.html` (generated from the ledger)
- **Optimization opportunities + fairness audit** → `docs/optimization_roadmap.md`
- **Inference pipeline** → `inference/inference.md` · **infra/budget** → `computing/infrastructure.md`
- **Pre-2026-06-25 dated status / decisions / dev-log** → `docs/archive/working_status_pre-rolling_2026-06-25.md`
