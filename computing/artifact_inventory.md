# Artifact & Bucket Inventory

**Single source of truth for "what artifact lives where."** Every durable artifact produced since project
start → its bucket/path, region, owning project, and whether it's source-of-truth or derived/backup.
Companion to `infrastructure.md §4` (bucket facts) — this doc is the *artifact → location* map.

> Surveyed 2026-06-26. When you produce a new durable artifact or move one, update this table.

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
| **Training runs (checkpoints, configs, logs, figures)** | local `/mnt/outputs/v1.0/runs/<run>/` | A100 master (local disk) | `gs://rts-mapping-v2/RTS_MODEL_V2/runs/` | **110/110 mirrored (2026-06-26).** Mirror is *slim* by convention: `best_deployment.pth` + config + run_summary + train.log + figures; `resume_latest-*.pth` are local-only (regenerable training state). |
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

---

## 3. Local `/mnt/outputs` (A100 master — local disk, NOT durable)

Source-of-truth — **backed up to `gs://rts-mapping-v2/RTS_MODEL_V2/` (2026-06-26):** `v1.0/runs`
(110/110 slim), `v1.0/calibration`, `v1.0/test_realistic`, `v1.0/object_operating_point`. Still
local-only (back up before any disk cleanup): `inference/` (2025 tile lists + quad index, ~3.7 GB),
`v1.0/staging` (norm stats — also travel with `training/v1.0`). `v1.0/mlflow` is a UI mirror; the
score SoT is each run's `run_summary.json`. Derived/disposable (NOT backed up): `mlflow*`,
`report.html`, `qc/`, `hf_cache/`, `bench/`, `s2_qc/`, `worktrees/`, `_archive/`, scratch (`_du.txt`,
`_tmpcheck.txt`, `_paper.txt`, `upload(irrelevant)/`), S2 export logs. See `/mnt/outputs/README.md`
for the current layout + SoT-vs-derived split.
