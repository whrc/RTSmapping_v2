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
**SE-PC2 decomposition + SAM2 re-run honestly (branch `pc2-and-sam2-investigation`, 2026-07-29/30).**
Two loose ends closed, plus a silent infrastructure bug found and fixed.

**PC2 — CLOSED NEGATIVE.** SE_PCA had only ever been trained as an atomic 3-band pack (0.8736/0.8571),
and the 8-band QC showed **SE_PCA2 is the only SE band tracking NDVI** (Spearman −0.765), so the pack's
mediocre score could have been PC2 carrying signal while PC1/PC3 diluted it. Five arms tested that:
PC1 **0.8720** · PC2 **0.8640** · PC3 **0.8776** (pack 0.8736) · NDVI+PC2 **0.8865** · NDVI+PC3 **0.8653**
(NDVI-alone 0.8879, all seed 42). Three negatives: no single PC is the carrier (all within ±0.010 of the
pack, inside G), PC2 alone does not match NDVI, and PC2+NDVI ties NDVI-alone. The −0.765 correlation is a
shared **response** to exposed soil, not shared **information**. `EXTRA=[NDVI]` unchanged. Single seed by
design — every arm is at or below its comparator, so a 3-seed confirm would spend 10 runs re-establishing
a consistent null.

**SAM2 — 0.5558 → 0.9030 (3-seed), still loses to EffB5 0.9218.** The recorded "non-competitive" was
right in direction and wrong by an order of magnitude, and every reason attached to it was false. The old
run used the **pre-lock recipe** (`boundary_handling: none` changes the *val labels*, so 0.5558 vs 0.9218
was never like-for-like), and three premises in its config were disproved by direct audit: Hiera *does*
expose `.blocks` (16, one level under timm's `features_only` wrapper) so LLRD was always possible; it
*does* expose `.patch_embed` so NDVI was always possible; and the weights *did* load (202/202 tensors,
never verified before). An 11-cell factorial then attributed the recovery: **encoder LR is the whole story
(~+0.34** — flat 1.0× → gentle uniform 0.1×; the original destroyed the encoder every epoch after
unfreeze, hitting `pixel_iou=0.0` twice), **LLRD is NOT the fix (−0.028**, contrary to expectation),
**normalization is real (+0.068** ImageNet vs z-score, vs +0.019 for DINOv3), and **NDVI is an interaction
not a main effect** (+0.0055 under gentle LR, **−0.044** under LLRD — pairing NDVI only with LLRD, as the
battery first did, would have concluded NDVI harms SAM2). A 24-cell frozen-feature probe
(`scripts/probe_frozen_features.py`) additionally showed **Hiera does not improve with scale** and that
**512 beats its native 896**, killing the resolution hypothesis. Key caveat recorded: probe AP does *not*
predict trained score — use it for triage only. **EffB5 stays deployed**; family E closed for a defensible
reason rather than a broken run. Full attribution: `docs/experiment_ledger.md` §E-SAM2.

**Infra bug (silent data loss).** `run_gpu_pool.sh` had passed a **host** path as the container's
`--out-dir`/`MLFLOW_TRACKING_URI` since the NVMe-mode commit; in default (non-NVMe) mode that path does not
exist inside the container, so every run wrote into its ephemeral layer and `docker run --rm` destroyed it
on exit. The only symptom was one `tee: No such file or directory` line at startup. Cost **3 completed
SAM2 runs (~25 GPU-h)**; 7 in-flight runs were recovered with `docker cp`. Fixed by adding `HOTROOT_C`
(container view) alongside `HOTROOT` (host view), verified by writing through the container and reading
back on the host. **Anything launched via the pool between that commit and 2026-07-29 lost its artifacts.**

Prior: **C1–C4 hypothesis-test battery COMPLETE — 23 runs, manuscript CSV shipped (2026-07-25).** The
capacity-vs-representation-vs-labels battery (design: `docs/future_work/experiments_hypothesis_test.md`)
ran back-to-back on 8×A100 (~40 h, **all 23 `status=completed`, 0 crashes**) on the hardened configs.
Deliverable **`outputs/metric_robustness.csv`** (`scripts/export_metric_robustness.py`): 45 per-seed rows +
30 mean/std aggregates over 15 conditions, secondary metrics (IoU/F1/obj-P/R/F1) read from the MLflow store
at each run's pr-auc `best_epoch`. Ledger: **23 rows added** (fam B/D/E), `sync_experiments.py` zero-drift.
Computed 3-seed means (pr-AUC geomean, gate G=0.0112): **C1** ViT-L-RGB locked 0.9177 ≈ EffB5+NDVI 0.9218
(Δ within σ); NDVI-effect +0.044 for EffB5 but +0.0014 for ViT-L. **C2** capacity B0 0.860 → B3 0.906 →
B5 0.912 → B7 0.903 (rise then flat/down). **C3** data budget — ViT-L+NDVI 25/50/100% = 0.865/0.870/0.919;
EffB5+NDVI 25/50/75/100% = 0.793/0.855/0.884/0.912. The A/B/C outcome-classification (design §1) is left
for the manuscript — the ledger/CSV state facts only. Engineering landed alongside: **permanent config
validator** (loud-fails on unknown `early_stopping`/section keys; the silent-typo class that had voided
`start_epoch` overrides) + 18 configs corrected + 4 tests; ViT-L `start_epoch=45` overfit-tail trim; boot
disk resized **500 G→1 TB online** (no VM stop, A100 capacity preserved); all 8 local SSDs → **RAID0 2.9 TB
`/mnt/nvme_scratch`** (provisioned, not yet wired into the pool — see *Now*).

Prior: **v2.1 SSL-pretraining program CLOSED in the negative (branch `v2.1-pretraining`, 2026-07-18).** The full
MAE program ran end to end: corpus (295,429 tiles, 4-ch RGB+NDVI south-only, ~188 GB), 80-epoch DINOv3-L
MAE continue-pretrain on 8×A100 (92,320 steps, recon loss 1.016 → 0.0763), then the 3-seed arm-(c) gate.
**Result: arctic MAE actively harms the encoder** — 0.8173/0.8155/0.8090, mean **0.8139** vs the arm-(b)
sat493m baseline 0.9191 → **Δ −0.105, 0/3 seeds positive**, ≈9× G in the *negative* direction. Not a null,
a regression. Comparison verified fair (flattened locked recipe field-by-field; `encoder_init` loaded
318/318 tensors). Leading hypothesis: catastrophic forgetting — MAE pixel-reconstruction overwrites
sat493m's discriminative features with reconstruction-oriented ones (arm-c peaks came *later*, ep45–50 vs
ep35–40). **EffB5 remains the deployed encoder**; family-E encoder null now extended to domain-adapted ViT
weights and closed. Epoch-20/40/60 MAE checkpoints kept on GCS if the length-ablation is ever revisited.
Details: `docs/experiment_ledger_v21.md` (Finding A).

Two engineering fixes landed alongside: (1) **early-stop floor lowered** `start_epoch` 101→65 in
`configs/baseline.yaml` + the v2.1 base — locked-recipe peaks land ep35–50, so the 101 floor only burned
compute (`max_epochs` deliberately left at 300: it sets the cosine LR horizon, so changing it would change
the recipe); (2) **resume no longer overrides config** — `EarlyStopping.load_state_dict` was restoring
`patience`/`min_delta`/`start_epoch` from the checkpoint, silently voiding the 101→65 change on the gate
resume (all 3 seeds ran to ep105, not the expected ~ep90). It now restores observation state only and logs
any config/checkpoint divergence; 3 regression tests added.

Prior: **Disk-cleanup audit on the A100 master (2026-07-16).** Every durable artifact verified on / uploaded to
GCS (run-mirror parity 123/123 incl. the previously unmirrored scale_ndvi + sat-7B runs; new prefixes
`RTS_MODEL_V1_1/`, `RTS_MODEL_V2_scale05/eval/`, `RTS_MODEL_V2/inference_inputs/`, `training/v1.1_delta/`;
full drift-refresh of all existing mirrors), then ~275G of verified GCS copies + regenerables deleted —
root fs 94% → 37% used, scratch NVMes cleared. Two deliberate non-archives (5× 26G sat-7B dead-arm
weights; `_archive/v2-alpha`) recorded with rationale in `computing/artifact_inventory.md` (the
"where is everything" map, re-surveyed and re-dated).

Prior: **South pan-Arctic inference + products delivered (2026-07-07 to 2026-07-11, see `docs/archive/` for
full detail).** Full South inference run completed (2,079/2,079 shards, 41,567,572 tiles reconciled
exactly, 3 quad-level coverage gaps ≈0.001%) after three fork/stall/crash-loop fixes shipped; post-
inference pipeline delivered `south_rts.gpkg` (10,984 polygons / 238.08 km²) + probability/mask VRTs
over 1,633 super-tile COGs. Master `a100-8x-train` freed for the manuscript-gap work below.

Prior: **Pre-launch full audit (branch `audit-prelaunch`, `docs/inference_launch_audit.md`).** Comprehensive
audit before the pan-Arctic inference launch, against two goals: (1) past decisions rest on solid facts,
(2) model + machinery are scientifically & engineering sound. **Science side: clean** — every headline
number re-derived from primary artifacts (3-seed 0.9218, ensemble 0.9393/T 0.512321, test 0.584/0.437/0.500,
MMU-600 floor 0.159), splits spatially leakage-free (0 overlapping cross-split tiles), norm-stats/EMA
weights/calibration chain all verified, v1.1-wash verdict sound; ledger sync zero-drift. **Machinery side:
10 defects found, 6 blockers**, all in the fleet/execution path and only reachable by loading GCS packages
+ booting a real L4 VM: gs:// packages unloadable, fleet-startup GPU-visibility bug, no `--shm-size` (every
worker crashed), SA scopes/IAM (all writes 403), retired DLVM image family, `--metadata` comma parse, no
docker on the image, `pretrained=True` HF-hub pull, heartbeat starvation, reboot non-idempotency. All fixed
(8 commits) + `rts-infer:v1` rebuilt/pushed (digest `7772dbc7…`) + 2 IAM grants (our bucket + read on
`pdg-planet-data`). **Headline benchmark: real L4 rate ~4.2 tiles/s (3-model ensemble, GPU-bound) → full run
~2–5 days, NOT the plan's 12–29 h.** E9 smoke passed (Banks AOI, NDVI parity 0.95–0.97, detections sane,
output uint8 ~8 KB vs float32 570 KB/tile). Also: S2 export was actually **1,797/1,799 done** (diary's 76%
was stale), 2 cells relaunched (GEE PENDING); `s2_index` built + uploaded; drill VM stopped (not deleted).
Prior: **data-v1.1 closed out (branch `data-v1.1`, ledger N + N-retrain).** Two deliverables: (1) the **Minimum
Mapping Unit metric fix** — the real object-score win (test invisible floor 0.223→0.159, obj-F1 0.526→0.560
at MMU600, precision invariant, no retrain); (2) the **v1.1 data-correctness retrain** (+28 restored pos,
−49 black neg, vjn7 promotion) which came back an **ability WASH**: calibration-free test pixel PR-AUC
0.9976≈0.9970 and fair val-optimal obj-F1 tie (v1.0 0.567/v1.1 0.562 val; 0.627 vs 0.607 test, ≈noise).
The apparent val/object drops were confounds — a −29-black-negative val-set change and a calibration
mismatch (v1.1's optimal threshold is 0.45, not the deployed 0.65). v1.1 does show a **tighter val−test gap
(0.045 vs 0.060)** and a **precision lean**, both mild positives, but not enough to beat v1.0.
**Decision: keep v1.0 deployed; retain v1.1 (cleaner labels + checkpoints) for the next real modeling
change** (shipping it would need its own thr≈0.45 calibration). Prior: **Minimum Mapping Unit metric
correction.** Object-wise scoring counted every GT
component as a full object while predictions are size-filtered (deploy min_blob 2000) — so any GT
`< min_blob*iou_thr = 600 px` was a structurally-guaranteed false negative and inflated the Finding-K
invisible floor. Domain-expert re-diagnosis: 0–50 px = rasterization artefacts, 50–400 px = real but
boundary-clipped slump tails (body off-tile). Fix = mark sub-Minimum-Mapping-Unit positives as ignore
(255) uniformly via one shared `apply_min_mapping_unit` (data/label_cleaning.py), wired into the loader
(`RTSDataset.min_mapping_unit_px`, loss + live metric) and the cached-npz scoring path
(`object_scorecard.py`/`analyze_residual_errors.py` `--min-mapping-unit`); default off, **no retrain**.
Free 3-seed re-score at the deploy point (self-check True): **obj-precision invariant** (0.793 val /
0.768 test), obj-recall +3.2 pt val / +4.1 pt test, invisible floor 0.280→0.231 val, 0.223→0.159 test.
6 new tests (`test_gt_mmu_scoring.py`), suite green. Track B (restore 28 positives + drop 49 black + promote
`vjn7wxyufczs`) in progress. Prior: **multi-scale inference (§6.3/§7.3) + multiscale-poc merged** — the
inference pipeline now does
per-tile multi-scale fusion when `deployment.yaml.scales` has >1 entry (default stays `[1.0]`, deploy path
unchanged): `InferenceTileDataset(scales=…)` reads each scale (scale s<1 = bbox expanded 1/s×, the §6.3
context read; NDVI at the same expanded bbox), and `inference/runner.fuse_scale_probs` averages over valid
scales on the 1× grid (§7.3; scale-s centre-cropped + upsampled), output NoData = the 1× footprint (§5.3).
Mirrors the validated `evaluate_multiscale_poc._fuse`. **Capability only** — deploying `scales:[1.0,0.5]`
stays a separate decision (calibration + §6.4 test gate; POC gate-3 fusion-recall failed). 7 new tests; full
suite **338 green**. Also merged the **multiscale-poc** branch (family M: 0.5× re-stage + multi-root loader +
3-seed joint training; gates 1+2 pass, gate-3 fail). Prior: v3 object-scorecard bias/variance diagnosis
(F_in 14% / F_held 28% → bake-off; ledger K); inference Phases 1–3 (orchestration + packages + `rts-infer:v1`
+ fleet scripts).

<!-- NOW:BEGIN -->
### Now
**2022 pilot passed; interannual campaign under way — `campaign/` (branch `interannual-campaign`,
2026-08-24).** The acquisition machinery built in August works: 2022 is downloaded, indexed and
drift-checked, and the coordinator that will carry the remaining five years is built and tested.
Runbooks: `campaign/README.md` (us) · `planetscope-download/README.md` (Heidi).

- **2022 acquisition complete, 0 escalations.** 309,109 quads ordered in 131.6 h at 39.1 orders/min,
  **0 failed**, 0 supervisor restarts, **92 transient failures absorbed** by the retry policy — 84 of
  them HTTP 400s carrying GCS 500/503, which validates retrying 400s rather than treating them as
  client errors. Two quads Planet never delivered (`1610-1516`; `622-1604` returned an empty
  manifest) = 0.0006%, vs ~0.05% missing over land in 2025.
- **The rename really was unnecessary — now observed, not just reasoned.** `build_quad_index` against
  the raw delivery returned **309,107 quads from 1,854,688 objects**, reconciling against orders
  placed at **0.00% off**, with paths like
  `…/2022/q3/0/1515/<uuid>/global_quarterly_2022q3_mosaic/0-1515_quad_file_format.tif`. This was the
  pilot's stated first check and the one claim that had only ever been verified by code reading.
  **Heidi never has to run the rename or delete passes** — the two changes she found hardest.
- **The §5.4 drift gate was mis-calibrated, not 2022.** 2022's blue channel tripped at σ-ratio
  +0.295 (threshold 0.25) — but running the identical check on **2025**, the imagery the delivered
  map was made from, trips the same channel at **+0.256**. The gate compares random *whole quads*
  against `normalization_stats.json`, which came from 17,951 curated RTS-centric tiles, so any year
  reads as wider. Against the right baseline (2025 quads, same sampling, now committed as
  `inference/quad_baseline_2025q3.csv`) 2022's worst mean drift is **0.095σ against 0.5σ** and worst
  σ-ratio **+0.047**. `inference.md` §5.4 corrected; the gate was fixed, not overridden.
- **The real blocker for 2022 inference was NDVI, not imagery.** The deployed ensemble is RGB+NDVI,
  read on the fly from year-matched S2 composites — and only `2024_train`/`2025_north`/`2025_south`
  exist. Measured: `2025_south` took **10.9 days for 1,799 cells / 15.02 TiB** while sharing GEE with
  two other exports. **This makes the S2 export the campaign's critical path, not the Planet
  download** — and since it depends only on the fixed domain, it runs alongside acquisition rather
  than after it. Decisions: export each year (reusing 2025 NDVI would bake 2025 vegetation into a
  2022 map, destroying the signal we are measuring); keep all six years on Standard (~90 TiB).
- **`campaign/` coordinates the remaining work.** A per-year state file plus a driver that shells out
  to the scripts we already run by hand; 12 stages from `acquire` to `qc`, with **human gates** at
  `drift_check` and `qc` that the driver will not pass. Finished work is never repeated (re-running
  `infer` is 2.5 A100-days), a stage with unready inputs refuses and names what is missing, and a
  detached launcher exiting 0 is not treated as the work finishing. Alerting is announce-once on two
  signals (stale heartbeat AND no progress), so a queuing GEE export cannot cry wolf.
  30 tests, GPU/Docker/network-free. Cron: `/etc/cron.d/rts-campaign`.
- **Open risk being watched:** the pdg EE project has warned *"exceeded the compute quota of its
  noncommercial tier … restricted mode"* since 2026-08-03. A 2-cell smoke export launched 2026-08-24
  sat **PENDING with an empty project queue and zero EECU consumed** — batch export is exactly the
  compute that warning covers. The whole interannual programme is gated on this; resolve before
  committing six years of exports.
- Next: EE verdict → launch the 2022 export → `s2_index` → `shard` → `infer` (~2.5 days, 8×A100,
  master-only now the 32×L4 fleet is gone). Heidi starts **2019** in parallel; her acquisition runs
  underneath the S2 critical path. Campaign critical path ≈ 40–65 days, set by S2.

**Also live: collaborative review campaign — a 2–3 person team can now traverse all 60,167 candidates
(branch `review-campaign`, 2026-08-03).** The South inventory has no human verification; everything
in `south_products.md` traces back to one 280-polygon stratified sample. This builds the machinery to
replace that extrapolation with a census. Protocol SSoT: `post-inference/review_campaign.md`.

- **Queue ordered by descending `max_prob`**, so the campaign's claim at any moment is *"every
  polygon with `max_prob ≥ p` is human-reviewed"* — a statement in the attribute the product family
  already exposes, and one that holds at any stopping point (60,167 items ≈ 134 person-hours, so
  stopping early is the expected case, not the failure case). Item order is shuffled *within* each
  batch against rater response bias; the reviewer brief says plainly that the head of the queue is
  **not** the precision peak (caveat 1: precision peaks at 0.65 and falls above it).
- **Assignment reuses `inference/claim.py` unchanged** — the same `if_generation_match=0` atomic
  claim, heartbeat and stale-reclaim that ran the pan-Arctic inference queue. A review batch is
  claimed exactly like an inference shard, so two reviewers can never get one batch and a
  walked-away reviewer's batch returns to the pool.
- **No imagery ever lands on the host.** Crops are streamed out of the bucket through the app
  (`/crop/<key>`, prefix-checked so it cannot read anything else); nothing is written to disk. FastAPI
  + the existing keyboard UI rather than Gradio.
- **The blocker found on the way: the chip archive covered only 29.9% of the inventory.**
  `internal/rgb_chips/` (29,850 chips) was built for the 10,984-polygon 0.65 product, not the 60,167
  candidates, which reference **118,586** tiles — so 17,980 of 60,167 polygons (452 of 688 km²) had
  imagery and the rest would have been served blank crops. Fixed by re-running
  `build_rgb_chips.py` against `south_rts_candidates.gpkg`; the script gained `--workers`,
  skip-existing resume, an atomic write-then-rename (resume skips existing files, so a truncated
  chip would be skipped forever), and a `-input_file_list` VRT build (118k paths overflow argv).
  **88,736 chips written in 13 min at 64 workers, 0 errors.** The crop archive that followed is
  **120,334 JPEGs / 3.2 GB with only 20 polygons (0.03%) lacking imagery**, versus the ~70% the old
  archive would have produced.
- **Crop rendering was 53× too slow until indexed.** Reading a crop from the 29,850-source mosaic VRT
  costs ~2.1 s because GDAL scans the source list per window — 62 h for the archive. `chip_index`
  parses the VRT once and each crop is read from a micro-VRT of the 2–3 chips it touches: 0.070
  s/polygon of worker time — the full 60,167 rendered in ~7 min on 90 workers.
- **Two real bugs the tests caught before any data moved**: a replicate clamped into its own batch
  (one `rts_id` twice in a batch, which a verdict map keyed by id cannot represent), and a blank-crop
  detector that counted antialiased red outline as imagery — replaced by a source-pixel probe.
- **76 new unit tests** — 75 across 5 new files (crops/manifest/store/app/merge) plus a chip-write
  atomicity regression. Full suite **456 → 532 passed, 2 skipped**.
- **Verified beyond the suite**, all on real artifacts: the refactored offline rater rebuilds the
  shipped `qc_rater.html` to a **byte-identical MD5**; the claim loop runs against real GCS
  (claim → heartbeat → 200-verdict submit → idempotent retry → 409 → progress); replaying the 2026-07
  ratings through the merge reproduces their **65 rts / 152 false / 63 unsure** split and writes the
  60,167-row verified layer with the 59,887 unreviewed polygons null; and `score_qc_ratings.py`
  reproduces the published `qc_precision_grid.csv` byte-identically.

**Host: Cloud Run behind IAP — and a corrected premise.** The host was first chosen on throughput
grounds (per-item round-trips ≈ 100 person-hours), and that reasoning was **wrong**: the rating loop
is entirely client-side — `rate()`/`nav()`/`show()` make no network calls, `/api/next` returns all 200
items at once, verdicts buffer in `localStorage`, one POST submits. Rating itself is **zero
requests**. The real bottleneck is the human judgement, as the user pointed out, so the decision was
re-made on cost, credentials and setup:

- **~$0** (Cloud Run free tier covers the traffic, scales to zero) vs $9/mo HF PRO or ~$13/mo VM.
- **No service-account key, and no signing permission either.** Crops are streamed through the app,
  so the runtime SA needs only `objectViewer`/`objectCreator` on the one bucket — see the permission
  wall below, which is what forced this.
- **Direct IAP** (GA 2026, no load balancer): reviewers sign in with their existing Woodwell Google
  accounts, and **verdict attribution becomes authenticated** rather than a typed name — it feeds the
  inter-rater κ, so a client-supplied name is now ignored whenever an assertion is present, and a
  forged assertion is a 401, never a silent fallback.
- Claiming is already atomic in GCS, so the app is **safe to autoscale by construction**.

Since 2026 HF requires a paid plan for any compute Space; only static Spaces are free, and a static
Space cannot hold credentials (its *secrets* are readable from the browser). `build_hf_space.sh`
keeps that path working as a documented fallback. Comparison: `review_campaign.md` §10.1.

Verified locally: one image serves `PORT=8080` (Cloud Run) and the default 7860 (HF); `/api/me`
behaves in all three identity cases; a **forged assertion is rejected with 401 against Google's real
key server**. 21 tests in the app suite.

**The deploy hit a permission wall, and half of it was designed away.** The first run of
`deploy_cloud_run.sh` failed on `iam.serviceAccounts.setIamPolicy`. Probing the project showed the
shortfall is general: the operating account holds **no `setIamPolicy` of any kind** here — not on the
service account, not on the Cloud Run service, not on IAP — though it *can* create service accounts,
build, deploy, and administer the bucket (`review_campaign.md` §10.3 has the full table).

- **Signing is gone rather than waited on.** It was the only thing needing the refused SA binding, so
  crops now stream through the app instead. This also removes the one link that could never be tested
  off-platform: the container was run against the **real bucket** and served a real crop (560×560
  JPEG, 30,647 bytes, HTTP 200) on `objectViewer` alone, while the manifest, a `..` traversal and a
  missing crop each returned 404. Bonus: a signed URL is a bearer token, so proxying also puts the
  licence-restricted crops behind the same auth as everything else. Cost is 169 ms/crop measured from
  outside the bucket's region, against ~8 s/item of human judgement, five prefetched ahead.
- **The IAP brand — the prerequisite flagged as possibly needing an admin — already exists.** Moot.
- **A third wall at the image push:** Cloud Build runs as the compute default SA, which cannot write
  to `pdg-artifact-registry`. The operating account can, so the script now builds locally and pushes
  directly — Cloud Build is out of the path, and the deployed image is byte-for-byte the tested one.
- **Two bindings genuinely need an admin** and cannot be designed away, because they *are* the access
  control: IAP service agent → `run.invoker`, and each reviewer → `iap.httpsResourceAccessor`. The
  script no longer aborts on them; it completes the deploy and reprints each refused binding verbatim
  for whoever holds project admin.

Cloud Run deployed and proven to run there (<https://rts-review-bl6ow4qsaa-uw.a.run.app>, revision
`rts-review-00001`, `Ready`, IAP on): its logs show the container starting under `rts-review-app@` and
reading the campaign with just the bucket roles — *"campaign loaded: 60467 items in 301 batches"*.
But it is unreachable, and **"just make it public" turns out to be the same wall**: tried, and
`--allow-unauthenticated` is `roles/run.invoker` for `allUsers`, refused by the identical
`run.services.setIamPolicy`. There is no way to open a Cloud Run service from this account.

**Live for the team on an open VM instead — <http://8.229.247.193/> (2026-08-04).** A VM's front door
is a firewall rule, not an IAM policy, and firewall rules the operators *do* control. So
`scripts/deploy_review_vm.sh` builds `rts-review-vm` (`e2-small`, `us-west1-a`, static IP, tag
`rts-review` open on `tcp:80` to `0.0.0.0/0`), shipping the image through the campaign bucket because
granting the VM's SA registry access would be yet another blocked IAM write. Verified over the public
IP: rater page 200, **301 batches / 60,167 items**, a crop served as a 30,647-byte JPEG, the manifest
still 404 through `/crop`, and no batch claimed during the checks. **Reviewers need only the link** —
no account, no install, no SSH.

The trade, taken with the user's eyes open: **no authentication** (the unlisted URL *is* the access
control), **self-declared attribution** — so κ means "these two names disagreed", not a verified
identity — and **no TLS** on a bare IP. Cloud Run stays deployed and IAP-fronted; both hosts read the
same GCS state, so if an admin ever runs the four refused bindings, switching hosts is just a
different URL and the queue carries over untouched.

**First real batch landed 2026-08-04** — `b00000`, 200/200 rated and submitted, 15.30 km², claim →
verdicts → `done` all correct in GCS, and the 7 items rated into `b00001` stayed in the browser
exactly as designed (nothing partial is ever written). Two changes the user asked for off the back of
it:

- **A claim lasts one week** (`STALE_AFTER_S = 604800.0` seconds; first set to `inf`, then given a
  finite TTL on 2026-08-04). The inference queue reclaimed a shard after 30 min because a crashed
  worker's shard was pure loss; a reviewer is not a crashed worker — their part-rated batch is in
  `localStorage` and still good tomorrow, and reclaiming it would have the same 200 polygons rated
  twice. A week covers a weekend while still letting a truly abandoned batch return by itself;
  `review_campaign.md` §6.1 documents the manual release for getting one back sooner.
- **An outline toggle** (`o`, or the button). The red outline is drawn *into* the pixels by
  `render_crop`, so this could not be a client-side switch — the archive now carries four JPEGs per
  polygon instead of two (`_t_plain` / `_w_plain`, +3 GB) and the toggle swaps between them. Sticky
  across items and reloads, so judging a batch unoutlined is a deliberate mode, not a per-item peek.

**Open — nothing blocking.** The campaign can start now. One measurement gap worth knowing:
`reviewed_at` is stamped once at submit, identical across a batch's 200 rows, so per-item pace is not
recoverable from the verdicts — only batch pace (claim → submit).

---

**Size-parameter disambiguation — the 10 / 80 / 2000 confusion, root-caused and closed (2026-08-12).**
Manuscript drafts kept fusing the `min_blob` numbers. The cause was not carelessness:
`configs/deployment.yaml`'s pixel floor **held all three values in sequence** — `10` (e630266,
2026-05-04) → `80` (db38892, 2026-06-25, *live when the frozen ledger-J one-shot ran*, which is why
J's obj-P 0.584 is a min_blob-80 number) → `2000` (757825a, 2026-06-26). All three are legitimately
"the deployment min_blob", at different dates, and **none of them is the shipped rule** — the
delivered inventory uses the different-unit geodesic MMU (`--min-area-m2 0`, m²).

- **One name, four parameters, now separated.** `min_blob_size_px` → **`vectorize_min_blob_px`** in
  `deployment.yaml`, read through `utils.config.vectorize_min_blob_px()` which falls back to the old
  key (still in the GCS packages) with a warning. The eval-stage `metrics.min_blob_size_px: 10` and
  every label-side identifier keep their names — they were already correctly scoped.
- **The shipped rule is now the default code path.** `vectorize_region.py --min-area-m2` defaults to
  `0.0`; reproducing the superseded product needs an explicit `--legacy-min-blob-px`. Previously a
  bare re-run silently rebuilt the legacy `south_rts.gpkg` floor.
- **A wrong number was propagating to 4 files.** "technical floor 2 px ≈ **10–45 m²**" — the `45` had
  simply dropped the `cos²(lat)` correction (2 × 4.777² = 45.6 m² is the *equator* value). Checked
  against the shipped attribute table: the true 2-px floor is **2.7–22 m²** over the 46–76°N span and
  the smallest delivered polygon is **2.69 m²** (the floor at 76°N). **6,761 polygons (11.2%)** sit
  below ARTS P1 = 79 m².
- **A spec contradicted the code.** `post-inference.md` described the vectorization filter as the
  10-px *eval* parameter from `baseline.yaml`; the code reads `deployment.yaml`. Corrected.
- **New SSoT:** `south_products.md` §"Size parameters — which number is which" — deliberately
  asymmetric (product rule alone first, everything else demoted under "Not the product rule"), plus
  a px→m² conversion table (**80 px is not 79 m²**; it is 181 m² at the inventory's median latitude)
  and a stage table separating **label/GT** floors from **prediction** floors. `object_scorecard.py`'s
  `--min-blob` (predictions) and `--min-mapping-unit` (ground truth) are opposite sides of the match;
  `data.min_mapping_unit_px` is **0/OFF** in every shipped config, and "MMU-600" is a GT-scoring
  sensitivity check, never a training or product setting.
- **Verified by cold reader, not by inspection.** A fresh subagent given the manuscript's questions
  against the *pre-fix* repo answered the MMU as "2.7–**45** m²" and, having computed the true minimum
  2.69 m² itself, **explained the contradiction away** as "a nominal mid-latitude figure" — the docs
  beat the data. Post-fix, the same probe answers all seven questions correctly. Parity held: both
  frozen anchors reproduce exactly (`anchors_all_match: true`).

Prior: 
**Public GEE app redesigned + the 0.65 contour built as a product (branch `gee-app-redesign`,
2026-08-03).** The app had three layers and two problems: the 95 m likelihood surface was invisible
zoomed out and coarse zoomed in, and nothing exposed the threshold or size filtering. Code and data
are done; **the app itself still needs a manual paste + Update in the Code Editor** — publishing is a
UI action, so it cannot be scripted from here.

- **Why the 95 m layer failed (one root cause, both symptoms).** Ingested as a continuous raster it
  gets **MEAN** pyramiding — EE discards the file's own block-max overviews — so zoomed out the mean
  fell below the layer's own `.gte(0.3)` mask and it erased itself. Retired rather than repaired:
  MAX pyramiding would make it visible but not quantitative (max-prob over a 10 km screen pixel
  saturates to ~1.0 wherever one detection exists). The overview is now the threshold-free
  **10 km expected-area** grid, which is a continuous field and survives pyramiding.
- **New product `south_rts_t65.gpkg` — 23,682 polygons / 259.91 km²** (19 min on 90 cores, no new
  code: `vectorize_region.py --threshold 0.65 --min-area-m2 0`). The 0.65 contour existed nowhere as
  geometry before — candidates are outlined at 0.30 and only carry `area_m2_t65` as a number, and
  `south_rts.gpkg` has min_blob 2000 px baked in. Two cross-code-path checks: **0 of 23,682** cores
  fall outside a 0.30 candidate, and Σ`area_m2` agrees with Σ`area_m2_t65` to **+1.08 %**.
- **A 0.65 cut is really a 0.648 cut.** `int(round(0.65 × 250))` hits Python's round-half-to-even →
  **162**, not 163. One `scaled_uint8` quantization step (0.004), shared by every 0.65 product. Its
  one visible consequence: 129 cores (0.54 %) sit in candidates whose `max_prob` is exactly 0.648,
  which `conf_class` calls `medium` — so the 0.65 layer is **not** a strict subset of
  `high_confidence`. Recorded in `south_products.md`, not "fixed".
- **The min_blob answer.** The shipped inventory applies **no MMU at all** (2 px technical floor);
  `min_blob_size_px: 2000` only ever applied to the legacy `south_rts.gpkg`. And 2000 px is a *pixel*
  count, so its ground floor slides ≈7× with latitude — **18,860 m² at 50°N → 2,671 m² at 76°N**,
  and **4,532 m²** at this inventory's median latitude (71.63°N), which is what the app's preset uses.
- **Three new public EE assets** (`south_rts_candidates`, `south_rts_t65`, `south_density_10km`) via
  the new `scripts/ingest_ee_app_assets.py`, which pins the shapefile field renames the `.dbf`
  10-char limit forces (`centroid_lat`/`centroid_lon` collide; `tile_ids` overflows the 254-char cap).
- **Performance is the app's design constraint, not an afterthought.** Every count and area in the
  panel comes from ladders precomputed by `scripts/build_ee_app_stats.py` (7 unit tests), so dragging
  the size slider costs **zero** server round-trips; `.evaluate()` survives in one place, the click
  inspector. Outlines use `paint()` not `style()`; only the density layer paints at the opening zoom;
  the zoom handler returns early within a band.
- **Noted, not acted on:** every EE call warns that `pdg-project-406720` *"has exceeded the compute
  quota of its noncommercial tier and is currently in restricted mode"*. Ingestion tasks still ran
  fine. The published app runs under `abruptthawmapping`, not pdg, so viewer compute should be
  unaffected — but this is worth a look before any EE-heavy work.

Prior: **ArcticDEM revisit CLOSED — terrain is real but redundant with NDVI (branch `arcticdem-revisit`,
8 runs, 2026-07-31).** ArcticDEM had been dropped at the channel-design stage on 2026-05-02 and **never
trained**; this tested it as the last physically-distinct modality (terrain, not spectra). Full numbers
and design in ledger Finding **§D-DEM**; availability facts in `docs/arcticdem_diagnostic.md`.
**Deployed path unchanged: EffB5 + NDVI, 0.9218.** 8×A100 now idle.

- **Δ_single (RGB+DEM vs RGB) = +0.0318**, 3/3 sign-consistent, 2.8× G → passes both lock criteria.
  The **first non-NDVI channel in family D to clear G at all** (NBR, TC, SE_PCA, SE_PROTO were nulls).
- **Δ_stack (RGB+NDVI+DEM vs RGB+NDVI) = +0.0015** = 0.13× G → fails. NDVI alone is worth +0.0498 on
  the same tiles. **DEM does not enter the deployed stack**, so the 28 % domain shrink and a terrain
  inference reader never have to be paid for.
- Reading: terrain and NDVI are two routes to the same discrimination — an RTS scar is both bare (low
  NDVI) and concave (terrain) — and NDVI gets there more cheaply and over the whole domain.

Two Stage-1 measurements drove the design, and both are worth carrying forward:
1. **The DEM predates the labels everywhere** — 0.0 % of tiles observed after 2021.5 (median `maxdate`
   2020.6) vs 2024-refined labels. No scar leakage is possible, so a win would be real terrain signal;
   the channel is simply 3–4 years stale.
2. **DEM coverage is itself a label cue** — 100 % of positive tiles have ArcticDEM vs 78.3 % of
   negatives. Missing DEM is NaN → post-norm 0, so a model could clear the gate by reading "DEM absent"
   as a negative prior without using terrain. Hence the new `splits.tile_allowlist` key: all four arms
   run on the 17,796 covered tiles, keeping **all** positives. **These four scores are therefore not
   comparable to the 0.9218 full-set SOTA** — that is why the deployed recipe is re-measured as
   `dem_ndvi_control` rather than reused.

Also fixed on the way through: the terrain derivatives are computed in **ground metres**, not EPSG:3857
map units. The pre-existing visualization prototype (`plots/extra_channel_vis`) uses map units, which
understates slope by 1/cos φ — measured against real GEE data as 2.02× at 60.4° N rising to 4.13× at
76.0° N, i.e. a latitude fingerprint. Visualization-only, so no past result is affected, but do not port
that code into a training path.

**Carried forward from the ArcticDEM work:**
1. **Δ_stack rests on one seed and is under-determined** — this experiment measured the terrain arm's
   seed spread at σ = 0.0178 (4× the RGB control's 0.0044), so a single-seed difference carries sd
   ≈ 0.018, larger than G. +0.0015 is a *failure to demonstrate* a stack gain, not a demonstration of
   none. A 4-run 3-seed confirm on the NDVI pair would settle it; not run, because the screen policy
   only confirms arms within reach of G.
2. **Terrain arms converge late and unevenly** — `best_epoch` 35/70/50 vs control 35/45/30, with seed 43
   still improving at epoch 70. Don't template a stop schedule for terrain work from the RGB arms.
3. **`EXTRA_DEM/` (58 GB, 17,796 tiles) is on the boot disk** — reusable if terrain is ever revisited
   (e.g. against a *pure-RGB* North model where no NDVI exists); delete it if not.
4. **New reusable machinery** — `splits.tile_allowlist` (restrict every split to a common tile subset),
   `scripts/slice_normalization_stats.py` (per-arm stats without perturbing shared constants), and
   `computing/Dockerfile.dataprep` (the GEE-capable image; `rts-train:v2` has no `earthengine-api`).

**Carried forward from the earlier PC2/SAM2 work:**
1. **The pool bug is fixed but its blast radius is worth a check** — any run launched through
   `run_gpu_pool.sh` between the NVMe-mode commit (`9d6f0b2`) and 2026-07-29 wrote its artifacts into a
   container layer that `--rm` then deleted. The C1–C4 battery predates that commit and is intact, but if
   any other wave ran in that window its run dirs are gone (MLflow + checkpoints alike).
2. **Five PC arms have metrics but no deployment checkpoint** — they were stopped manually once past peak
   (ES could not fire; they inherited `phase0c`'s `start_epoch: 101`), and their `run_summary.json` is
   *reconstructed* from MLflow history, flagged in a `provenance` field. Fine as ledger evidence, not
   reusable as weights.
3. **New template configs to reuse, not re-derive** — `fm_sam2_ndvi_multlow.yaml` is the best SAM2 recipe
   found (gentle 0.1× uniform encoder LR + ImageNet norm + NDVI); the general lesson for any future
   foundation encoder is *tame the encoder LR first, and do not assume LLRD transfers from the ViT arms*.

**Still parked on a user call (unchanged):** (1) **NVMe pool integration** — `/mnt/nvme_scratch` (RAID0,
2.9 TB) is provisioned and `run_gpu_pool.sh` supports `NVME_OUT`, but the default path stays on the boot
disk; wiring it needs a pool restart + smoke-test. (2) Whether `build_report.py` should gain **C1–C4
charts** — a report-feature change, not bookkeeping; the manuscript consumes the CSV directly.

Prior: **v2.1 closed; branch `v2.1-pretraining` up for PR (2026-07-18).** The SSL program reached a decisive
verdict (see *Just completed*) and needs no further compute — all 8 A100s are idle again. The branch
carries the full `pretraining/` component, the v2.1 ledger + report, the two early-stop fixes, and 21
green unit tests; nothing in it changes the deployed v2.0 path (EffB5 stays the encoder).

Prior: **ADC/PDG handover + public GEE app (branch `deliverables-handover`, PR #52, 2026-07-17/18).**
Three deliverables shipped for the South product handover: **(A) WMTS-conformant probability
tiles** — the 1,633 canvas-anchored shards were already exactly WebMercatorQuad z15-resolution /
z7-footprint but row-offset ~77.7 km from the global grid, so `scripts/retile_wmts_z10.py` re-cut
the canvas into **80,159 COGs (12 GB), each precisely one global z10 tile** (alignment <0.001 px ⇒
exact pass-through; 12 sampled tiles pixel-identical to source; 5 grid-math tests). **(B) Handover
README** (`deliverables/README.md` = repo SSoT, published byte-identical at `products/README.md`):
minimized 3-item submission (z10 tiles + `south_rts_candidates.gpkg` + `region_log.json`), the
tiling-convention answers for ADC (incl. gdal pyramid recipe), Abstract/Methods/Coverage, live-schema
attribute dictionary, full-family appendix. **(C) Public GEE app** (`post-inference/ee_south_app.js`):
assets-only (no live mosaic) — `south_likelihood_95m` + `south_rts_high_confidence` (19,068) +
`south_rts_centroids` (60,167), all ingested + public. **Rename landed everywhere** (user call:
'confirmed' wrongly implied human verification): `rts_class` 'confirmed' → **'high_confidence'** in
code SSoT, GCS data rewritten in place (old gpkg deleted, factsheet regenerated), EE assets
re-ingested, docs. Suite 409 green. **Blocked on an Owner grant**: app publishing needs
`roles/earthengine.appsPublisher` on `abruptthawmapping` (yyang has `earthengine.admin`, which
covers assets but not apps; Editor isn't enough either — both denied empirically). One-line grant
by gfiske/hrodenhizer/spotter, then user publishes → URL
`https://abruptthawmapping.projects.earthengine.app/view/south-rts-map`.

Prior: **South products v2 — QC-calibrated adaptive MMU (branch `south-products-v2-adaptive-mmu`, 2026-07-17).**
User rated the 280-polygon tier×size QC sample via the new offline HTML rater
(`scripts/build_qc_rating_page.py`; the GEE rater lost a full round to session-auth 500s): 65 rts /
152 false / 63 unsure (unsure excluded). Scored grid (`scripts/score_qc_ratings.py`, Wilson CIs):
precision is monotone in tier (high 0.54–0.90, medium 0.11–0.53, low 0.00–0.31) and **not** in size —
the smallest measured high band (500–2k m²) is the most precise cell (0.90), vindicating MMU≈0.
**`rts_class` rule locked with user** (option 1, tier+extension): confirmed = all high (19,068 / 529.7 km²) ·
candidate = medium <500 m² (25) · marginal = rest (41,074). SSoT
`scripts/export_south_products.py:assign_rts_class`. New: `south_rts_confirmed.gpkg` (replaces
`south_rts_high.gpkg`), `nodata_frac` soft-triage attribute (FPs concentrate on NoData/water/snow/mining
context — soft only, real RTS contain NoData), `qc_false_hard_negatives.gpkg` (152 verified FPs = v3
hard-negative seed, noted in ledger K-family backlog). Factsheet + catalog rewritten around the measured
grid; MMU≈0 inventory (60,167 / 688.2 km², thr 0.30) is the flagship. Shipping: re-export → density
re-join → factsheet → GCS upload + stale cleanup → PR.

Prior: **Tiered South probability products SHIPPED (branch `south-probability-products`, 2026-07-14).** The
probability canvas is now a three-package product family (catalog SSoT: `post-inference/south_products.md`;
plan `now-the-final-product-delegated-locket.md`). **D1 tiered inventory:** `south_rts_candidates.gpkg` —
**25,716 polygons / 639.4 km²** at thr 0.30 (`vectorize_region --threshold`, windowed polygonize of the
1,633 prob COG shards, no re-assemble), classed by max_prob (**high ≥0.65: 17,239 / 522.3 km²** · medium
6,765 · low 1,712) + `area_m2_t45/t65/t80` per-object boundary bands; plus high-only / centroids /
csv+parquet forms. Reconciliation: all 10,984 delivered thr-0.65 polygons intersect a high candidate
(0 orphans). **D2:** `likelihood_95m.tif` browse surface (exact block-max via new `downsample_max.py` —
`gdalwarp -r max` bled NoData-edge values 251–254 onto coverage seams, fixed); `density_10km` + `density_0.5deg`
grids with threshold-free **expected RTS area = 1,037.4 km²** (Σ calibrated P × geodesic px area; both grids
agree exactly; expectation > 639 km² outlines > 238 km² @0.65 mask — integrates diffuse sub-detection mass).
**D3:** catalog + `south_rts_summary.{md,html}` factsheet (hotspot map reproduces known RTS geography).
All uploaded to `gs://rts-mapping-v2-usw1/inference/2025q3_south/products/`.

**Open question — what the freed 8×A100 node does next.** The v2.1 result closes the "better encoder via
SSL" avenue, so the remaining high-leverage levers are the ones the v2.0 ledger still defers: **hard-negative
mining** (K, gated on first-inference outputs — now available from the South run, and the South QC pass
already rated a pool of false positives that would seed it) and the **conditional** scale-TTA / ensemble
decisions. No work started on either; needs a decision before compute is committed.
<!-- NOW:END -->

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
