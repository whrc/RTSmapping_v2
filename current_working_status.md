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
**C1–C4 hypothesis-test battery COMPLETE — 23 runs, manuscript CSV shipped (2026-07-25).** The
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
**C1–C4 battery done; deliverables written; 8×A100 idle again (2026-07-25).** `outputs/metric_robustness.csv`
(the manuscript-consumed stats) + the 23 ledger rows + regenerated `docs/report.html` are all in place;
scores are zero-drift against `run_summary.json`. **Two items parked on a user call, not started:** (1) the
**NVMe pool integration** — `/mnt/nvme_scratch` (RAID0, 2.9 TB) is provisioned but `run_gpu_pool.sh` still
writes run/MLflow output to the boot disk; wiring it (NVME_OUT mode + rsync keepers back to durable
`/mnt/outputs`, move HF cache) needs a pool restart to take effect and a smoke-test, so it waits until the
next training wave. (2) Whether `build_report.py` should gain **C1–C4 charts** — the report renders the
curated v2 build-up/findings view and does not enumerate the battery, so surfacing it visually is a
report-feature change, not bookkeeping. The manuscript itself consumes the CSV directly.

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
