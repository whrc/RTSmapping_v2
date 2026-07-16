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
**Disk-cleanup audit on the A100 master (2026-07-16).** Every durable artifact verified on / uploaded to
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
**v2.1 SSL-pretraining — MAE pretrain RUNNING on 8×A100 (branch `v2.1-pretraining`, 2026-07-16).**
v2.0 frozen; the idle node runs the DINOv3-L MAE program (plan `now-all-the-training-agile-puffin.md`,
spec `pretraining/pretraining.md`, ledger `docs/experiment_ledger_v21.md`). Direction: **MAE
continue-pretrain DINOv3-Large, not a convnet** (locked UNet++ decoder is incompatible with ConvNeXt —
stride-4 stem → 0-channel skip; DINOv3-L already integrated in family E + baselined 0.9191). Gate arms:
(a) EffB5 0.9218 · (b) DINOv3-L sat493m 0.9191 [both existing] · (c) DINOv3-L + arctic-MAE, 3 seeds;
SSL helped iff (c)−(b) ≥ G=0.0112 sign-consistent.

**Done:** all code (corpus builder, ViT-MAE `pretraining/mim_model.py`, DDP trainer `scripts/pretrain.py`,
`_load_encoder_init` hook, 3 ft configs, `sync_experiments.py --ledger`); **caught+fixed a leakage bug**
(exclusion polys were EPSG:3413 vs 3857 tiles → 0 exclusions; now reprojected, 7.4M eval-region tiles
correctly excluded); **corpus materialized** (295,429 tiles, 4-ch RGB+NDVI south-only, ~188 GB, stats
RGB[64.8,65.7,46.3]/NDVI 0.453). **Pretrain launched** (container `v21_pretrain`, detached): 80 epochs,
global batch 256, grad-checkpointing + batch 32/GPU (SimMIM full-seq ViT-L@512 is memory-heavy; OOM'd at
64), ~1 s/step measured → **ETA ~25 h** (~2026-07-17 07:00 UTC), loss decreasing. 18 unit tests green.

**Next (auto when pretrain done):** fine-tune the 3 arm-(c) seeds (`configs/v21/ft_dinov3l_arctic_seed*`
→ `encoder_final.pt`), harvest via `sync_experiments.py --ledger docs/experiment_ledger_v21.md`, verdict
in the v2.1 ledger. Corpus not yet GCS-mirrored (local-only; rebuildable) — optional durability upload.
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
