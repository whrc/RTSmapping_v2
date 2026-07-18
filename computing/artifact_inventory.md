# Artifact & Bucket Inventory

**Single source of truth for "what artifact lives where."** Every durable artifact produced since project
start → its bucket/path, region, owning project, and whether it's source-of-truth or derived/backup.
Companion to `infrastructure.md §4` (bucket facts) — this doc is the *artifact → location* map.

> Surveyed 2026-06-26; **re-surveyed 2026-07-16 (disk-cleanup audit)** — all durable artifacts verified
> on GCS, local copies of GCS-durable data deleted. When you produce a new durable artifact or move one,
> update this table.

---

## 1. Buckets at a glance

| Bucket | Project | Region | Role |
|---|---|---|---|
| `gs://abrupt_thaw/` | abruptthawmapping (non-PDG) | US multi-region | **Original/legacy data** + the v2-alpha training source (`RTS_MODEL_V2/DATA/`). Reading from PDG VMs crosses projects → egress. |
| `gs://rts-mapping-v2/` | PDG | US multi-region | **Compute-adjacent SSoT** — training data (`training/v1.0/`), run + MLflow mirrors (`RTS_MODEL_V2/`), and the planned `backups/`. |
| `gs://rts-mapping-v2-usw1/` | PDG | **us-west1** | **Sentinel-2 composites** (`S2_RGB/{2024_train,2025_south}`) — NDVI source + S2 model. **Inference I/O** prefix (to create): `inference/2025q3_south/`. |
| `gs://pdg-planet-data/` | PDG | **us-west1** | **2025 Planet basemap quads** (`global_quarterly/`) — pan-arctic inference input. **Shared PDG data — read-only, not ours.** |

---

## 2. Artifact → location

| Artifact | Where (SSoT) | Region/proj | Backup / mirror | Notes |
|---|---|---|---|---|
| **Training imagery + labels (v1.0)** | `gs://rts-mapping-v2/training/v1.0/{PLANET-RGB,labels,metadata.csv,splits.yaml,normalization_stats*.json}` | PDG / US | local `/mnt/outputs/v1.0/data_local/` | The working training set. `abrupt_thaw/RTS_MODEL_V2/DATA/` is the upstream/legacy source. |
| **Legacy arrays** | `gs://abrupt_thaw/{maxar_rgb,rts_labels}*.npy`, `CAMS/`, hashed dirs | abruptthaw / US | — | Pre-v2 / exploratory; not on the v2 path. |
| **Training runs (checkpoints, configs, logs, figures)** | `gs://rts-mapping-v2/RTS_MODEL_V2/runs/` (**mirror is now the only full copy of checkpoints**) | PDG / US | local `/mnt/outputs/v1.0/runs/<run>/` (configs/logs/summaries only) | **123/123 mirrored (2026-07-16):** 120 here + 3 `multiscale_poc_*` in `RTS_MODEL_V2_scale05/runs/`. Slim convention: `best_deployment.pth` + config + run_summary + train.log + figures. **Deliberate exceptions (2026-07-16):** the 5 dead-arm sat-7B runs (`fm_dinov3sat_7b_{lrtest,tuned_a,tuned_b,tuned_b_seed43,tuned_b_seed44}`) are mirrored *without* their 26 GB weights (arm lost, verdict in ledger; retrainable from config+seed). `fm_dinov3sat_7b_frozen` weights ARE on GCS. All local `.pth` (incl. `resume_latest-*`) deleted 2026-07-16. |
| **MLflow tracking** | local `/mnt/outputs/v1.0/mlflow/` + `/mnt/outputs/mlflow/` | A100 master | `gs://rts-mapping-v2/RTS_MODEL_V2/mlflow/` | UI served from the master. |
| **Calibration report** (Phase D) | local `/mnt/outputs/v1.0/calibration/effb5_trivialaug/` | A100 master | `gs://rts-mapping-v2/RTS_MODEL_V2/calibration/` (2026-06-26) | — |
| **Test-Realistic metrics** (shipped #) | local `/mnt/outputs/v1.0/test_realistic/effb5_ensemble_metrics.json` | A100 master | `gs://rts-mapping-v2/RTS_MODEL_V2/test_realistic/` (2026-06-26) | The one-shot v2 number (ledger J). |
| **Object operating-point report + val_probs** | local `/mnt/outputs/v1.0/object_operating_point/effb5_ensemble/` | A100 master | `gs://rts-mapping-v2/RTS_MODEL_V2/object_operating_point/` (2026-06-26) | `val_probs.npz` ~1.1 GB; restore verified byte-identical. |
| **Deployment packages** (3 seeds) | `gs://rts-mapping-v2-usw1/inference/2025q3_south/packages/seed{42,43,44}/` (2026-06-26) | PDG / us-west1 | local `/mnt/outputs/v1.0/deploy_packages/seed{42,43,44}/` | Run-dir packaged (`package_model.py --run-dir`) from `aug_trivialaugment_deploy{,_seed43,_seed44}`; verified load + ensemble-consistent + fuse. |
| **Inference tile list + quad index** | local `/mnt/outputs/inference/{tiles_2025q3_domain_full.csv,quad_index_2025q3.csv}` | A100 master | back up in −1.3 | 41.57M tiles / 309,101 quads. |
| **S2 index** (NDVI windowing) | **built** → `gs://rts-mapping-v2-usw1/inference/2025q3_south/s2_index_2025_south.csv` | PDG / us-west1 | local `/mnt/outputs/inference/` | From `scripts/build_s2_index.py`. |
| **Shard-claim queue** | `gs://rts-mapping-v2-usw1/inference/2025q3_south/shards/{index.json,shard_*.csv}` | PDG / us-west1 | (regenerable from the tile list) | **Built 2026-07-07 for the South launch**: 41,567,572 tiles → 2,079 shards of ≤20,000 (`scripts/shard_tiles.py`). |
| **Inference output** (prob COGs, claims, manifests) | **run output** → `gs://rts-mapping-v2-usw1/inference/2025q3_south/{probs,claims,done,logs}/` | PDG / us-west1 | (is the durable product) | scaled_uint8 COGs (deploy default, ~40 GB); shard-scoped `probs/<shard_id>/` prefixes (pre-mortem #1). 41,551,451 tiles → COGs + 16,121 all-NoData; reconciled 2026-07-10. |
| **South products** (polygons + prob mosaic) | `gs://rts-mapping-v2-usw1/inference/2025q3_south/products/{south_rts.gpkg,probability.vrt,mask.vrt,*_cog_shards/}` | PDG / us-west1 | (is the durable product) | **2026-07-11: 10,984 RTS polygons / 238.08 km²** + 1,633 scaled_uint8 super-tile COGs. From `assemble_region.py` + `vectorize_region.py`. Open guide: `post-inference/arcgis_south_products.md`. |
| **Docker image** | training: `…/rts-train:v2`; inference: `…/rts-infer:v1` (**rebuilt+pushed 2026-07-07 for South launch**, `rts.git_sha=7b7d74c`, digest `sha256:03a9a69b…`) | PDG Artifact Registry / us-west1 | — | `rts-infer:v1` is self-contained — current code incl. `inference/` (async prob-COG writes + cached GCS client = the 12× throughput fix; **+ forkserver DataLoader + stall watchdog = the South fork-safety fixes**), cv2 patch baked, MLflow 2.22.5 (pin `<3.0`); **no runtime sed/mount/pip**. Built locally + pushed via ADC (Cloud Build blocked, §8). `:v2` kept for training reproducibility. |
| **Docs / ledger / report** | repo (`docs/`, `current_working_status.md`) + GitHub `whrc/RTSmapping_v2` | — | git | `report.html` is gitignored (regenerated). |
| **v1.1 program outputs** (runs, diagnostics, object_operating_point, mlflow, logs) | `gs://rts-mapping-v2/RTS_MODEL_V1_1/` (2026-07-16) | PDG / US | local `/mnt/outputs/v1_1/` | Same substructure as `RTS_MODEL_V2/`. |
| **v1.1_delta training data** | `gs://rts-mapping-v2/training/v1.1_delta/` (2026-07-16) | PDG / US | local `/mnt/outputs/v1.1_delta/data_local/` | Delta-label set (32 MB). |
| **scale05 training data** | `gs://rts-mapping-v2/training/v1.0_scale05/` | PDG / US | — (**local copy deleted 2026-07-16**, re-stage from GCS) | PLANET-RGB + EXTRA + labels, "-05" tile IDs. |
| **scale05 runs / logs / qc / eval** | `gs://rts-mapping-v2/RTS_MODEL_V2_scale05/{runs,logs,qc,eval}/` | PDG / US | local run dirs (slim leftovers) | `eval/` = `multiscale_poc_eval` seed dirs + baseline (uploaded 2026-07-16). |
| **2025 inference inputs** (tile list, quad index, S2 index, grid scripts+logs, domain overlay) | `gs://rts-mapping-v2/RTS_MODEL_V2/inference_inputs/` (2026-07-16) | PDG / US | local `/mnt/outputs/inference/` (top-level files) | `tiles_2025q3_domain_full.csv` (3.4 GB, 41.57M tiles) + `quad_index_2025q3.csv`. |
| **Banks products** | `gs://rts-mapping-v2-usw1/inference/banks/products/` | PDG / us-west1 | — (local `out/`, `probs_local/` deleted 2026-07-16 after byte-parity check) | Incl. `banks_rts_parallel.gpkg` (uploaded 2026-07-16). Intermediate `blocks/` regenerable from `banks/probs/`. |

### Deliberately NOT archived (deleted 2026-07-16, user decision)

| What | Why it's gone | If ever needed again |
|---|---|---|
| 5× 26 GB sat-7B dead-arm `best_deployment.pth` | Losing arm of the resolved sat-DINOv3-vs-EffB5 confound re-run; finding recorded in `docs/experiment_ledger.md` | Retrain from the mirrored config + seed (~1 GPU-day each) |
| `/mnt/outputs/_archive/v2-alpha` (27 GB) | Pre-v1.0 outputs trained on the destroyed dataset; superseded by the v1.0 program, marked disposable in the outputs README | Not reproducible — accepted loss |
| Local South probs copy (`/mnt/scratch/probs`, 163 GB) + `south_out` (39 GB), scale05 `data_local` (40 GB), south/banks product copies (~27 GB) | Verified copies of durable GCS objects (rsync parity / byte-identical) | Re-download from the GCS paths above |
| HF cache `vit_7b_patch16_dinov3.sat493m` (26 GB), resume checkpoints (21 GB), gcloud logs (27 GB) | Regenerable caches / training state / log spam | Re-download / retrain |

---

## 3. Local `/mnt/outputs` (A100 master — local disk, NOT durable)

**As of the 2026-07-16 cleanup, nothing durable lives only on local disk.** Every artifact in §2 is on
GCS; local dirs are working copies or regenerable scratch. The full drift-refresh (all mirrored v1.0
subdirs incl. `mlflow`, `mlflow_combined`, top-level `mlflow` → `RTS_MODEL_V2/mlflow_toplevel/`,
`staging`, `logs` + stray top-level run logs) ran 2026-07-16. Kept hot locally on purpose:
`v1.0/data_local` (42 GB — the v2.1 fine-tunes read it; mirror of `training/v1.0`), Docker images,
DINOv3-L HF cache. Root disk after cleanup: **37% used (~308 GB free)**; scratch NVMes cleared.
See `/mnt/outputs/README.md` for the layout.
