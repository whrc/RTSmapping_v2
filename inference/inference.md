# RTS Segmentation Model v2: Inference Pipeline

> **Implementation status (2026-06-12).** Phase-1 single-scale pipeline implemented and
> smoke-tested end-to-end on real 2025-Q3 quads: `inference/{quad_index,tiles,predictor,writer}.py`
> + `scripts/{build_quad_index,generate_tile_grid,inference,merge_predictions,vectorize_predictions}.py`.
> §3.1 correction discovered at implementation: the 2025 data in `gs://rts-arctic-usw1` is **not**
> pre-cut tiles but Planet **quad deliveries** — 4096×4096 uint8 RGBA quads on the zoom-15 mosaic
> grid (2048×2048 grid over EPSG:3857; alpha band = NoData; per-quad UDM2 + metadata; a quad may
> appear under several order UUIDs). Tiles are therefore 512×512 **windowed reads** that may
> straddle quad boundaries (`inference/tiles.py` mosaics intersecting quads per tile). The §14
> calibration-mismatch assertion is implemented (`inference/predictor.py:assert_runtime_matches_package`).
> **Multi-scale inference implemented (2026-07-03)** — `inference/runner.py` (`fuse_scale_probs`,
> `_run_inference_multiscale`) + `InferenceTileDataset(scales=…)` do the §6.3 context-expanded per-scale
> reads and §7.3 arithmetic-mean fusion when `deployment.yaml.scales` has >1 entry (default stays `[1.0]`).
> The multiscale training POC is ledger family M (gates 1+2 pass, gate-3 fusion-recall fail); **whether to
> deploy `scales:[1.0,0.5]` remains a separate decision** needing calibration + the §6.4 test-side gate.
> Deferred, per spec gates: TTA validation (§7.4/8.5b). Smoke
> evidence: `/mnt/outputs/inference/` (dev package from phase0c_seed42 with DEV-ONLY threshold).
>
> **EXTRA=NDVI implemented (2026-06-24).** The locked v2 recipe is RGB+NDVI; NDVI is windowed
> on the fly from the bulk S2 composites (§3.3) — `inference/s2_index.py` +
> `inference.tiles.read_ndvi_tile`, stacked and normalized through the shared `apply_norm`
> (CLAUDE Rule 3). `scripts/inference.py --s2-index` is required when the deployment package
> declares EXTRA channels.

## 1. Inference Objective

Deploy the trained segmentation model for pan-arctic inference (60-74°N) on 2025 PlanetScope basemap imagery to produce an RTS survey map. The pipeline prioritizes **precision over recall** to minimize false alarms in the final product.

The data and model operation in inference should exactly match those in training. The best 'recipe' will be provided once the training and experiments are done.
---

## 2. Infrastructure

### 2.1 Compute Environment

**Region + throughput — SSoT is `computing/infrastructure.md` §4 (corrected 2026-07-07).** Inference runs
on the secured **us-central1 A100 master (`a100-8x-train`) reading directly from the us-west1 buckets** —
no data staging. The *first* bottleneck was the **output write** (sync per-tile COG write + a new
`storage.Client` per tile → 2.8 t/s at 0% util); fixed in code (async prob-COG writes + a cached GCS
client) → the write no longer stalls the GPU. **At the actual launch (2026-07-07, `rts-infer:v1` git_sha
`7b7d74c`, 3-model ensemble) the master sustains ~12 t/s/A100 → ~5 d**, cross-region reads being the
now-exposed second bottleneck (the 33 t/s write-fix figure was in-region). *(Withdrawn as the run's
blocker: the "move 14 TB to `us-central1`" co-location plan — not worth it mid-run; an in-region fleet
is the real ~2.7× lever if needed. See the infrastructure SSoT for the full correction.)*

**Fleet scaled to 32× L4 (2026-06-17, user decision).** `g2-standard-96` carries **8× L4** (the max L4
per single G2 VM), so **32× L4 = 4 × `g2-standard-96`** — there is no single-VM 16/32-L4 option; an N-L4
fleet is always ⌈N/8⌉ VMs. Doubling 16→32 L4 halves wallclock at ~constant GPU-hours/cost (the pass is
embarrassingly parallel + resumable).

| Resource | Specification |
|----------|---------------|
| Cloud | Google Cloud Platform (`pdg-project-406720`) |
| Region | **us-central1 master** reading us-west1 buckets directly (corrected 2026-07-07; the rows below are the *superseded* us-west1 L4-fleet design — SSoT `computing/infrastructure.md` §4). |
| VM fleet | **The 8-A100 master alone** (launched 2026-07-07). An in-region L4/spot fleet would ~2.7× throughput (reads become in-region) but is **optional** — the master reads us-west1 cross-region. *(Historical: 4× `g2-standard-96` = 32× L4.)* |
| Throughput / wallclock | **MEASURED + TUNED: ~24 tiles/s/A100** (3-model ensemble, cross-region reads, **num_workers=16**; ~217 t/s aggregate) → **~2.3 d** for the **41.57M** tiles on the 8-A100 master. *(8 workers was I/O-bound at ~12 t/s/~5 d; the ~61 idle vCPUs let `--num-workers 8→16` ~double it by hiding cross-region read latency, near the in-region ~33 t/s ceiling. The write fix first lifted 2.8→~12. SSoT: `computing/infrastructure.md` §region/throughput.)* |
| Storage | `gs://rts-arctic-usw1/inference/2025q3_south/` (single-region **us-west1**; Planet quads and `S2_RGB` now live in this same bucket → egress-free) — outputs, deployment packages, queue markers. (Supersedes the earlier planned `woodwell-rts-inference-arts-south` — one fewer bucket; see `computing/artifact_inventory.md`.) |
| Orchestration | **Self-balancing GCS shard-claim queue** (decided 2026-06-25; `inference/claim.py` + `scripts/shard_tiles.py` + `scripts/run_inference_worker.py`). The domain tile list is spatially sorted + split into many contiguous shards (`shards/*.csv` + `index.json`); each worker (one per GPU, **8 on the A100 master + 8 per L4 VM**) atomically claims the next free shard (`if_generation_match=0`), so the heterogeneous A100+L4 fleet auto-balances. Done markers are the source of truth; stale claims are reclaimed → preemption/stragglers just resume. **Fork-safety (2026-07-07):** the DataLoader uses a `forkserver` start method (`runner._make_loader`) so workers don't inherit the parent's gRPC/CUDA threads — the fix for the probabilistic Banks GPU-0 fork deadlock. A per-shard **stall watchdog** (`runner._start_stall_watchdog`, `inference.stall_timeout_s=900`) `os._exit(3)`s a worker that wedges so its claim goes stale + is reclaimed, and a **host supervisor** (`scripts/launch_south_inference.sh`, one worker/GPU) restarts any non-zero exit with a crash-loop guard → a silent single-GPU failure self-heals. |
| Collaboration | PDG workflow optimization team (Luigi/Todd) |

### 2.2 Storage Structure

```
gs://rts-arctic-usw1/inference/2025q3_south/   # single-region us-west1 (global_quarterly/ + S2_RGB/ share this bucket)
├── packages/                         # the 3 ensemble deployment packages (built 2026-06-26)
│   └── seed{42,43,44}/
│       ├── weights.pth               # EMA weights only (see training.md §4.3)
│       ├── normalization_stats.json  # channel-name bindings R,G,B,ndvi (training.md §4.5)
│       ├── model_config.yaml         # architecture, backbone, channels, data.tile_size
│       ├── deployment_config.yaml    # threshold, temperature, tta, precision, scales, fusion
│       ├── run_metadata.json         # git_sha, seed, checkpoint epoch/metric, channel_names
│       └── requirements_frozen.txt
├── shards/                           # scripts/shard_tiles.py output (the shard universe)
│   ├── index.json                    # {n_tiles, n_shards, shards:[{shard_id, n_tiles}]}
│   └── shard_<NNNNNN>.csv            # contiguous tile lists
├── claims/<shard_id>                 # atomic claim locks (inference/claim.py)
├── done/<shard_id>                   # done markers (source of truth on restart)
├── probs/<shard_id>/<tile_id>.tif    # probability COGs (§9.1, scaled_uint8 deploy default); shard-scoped to avoid GCS write-hotspotting
├── logs/<shard_id>.json              # one manifest per shard (not per-tile markers)
├── merged/                           # post-inference: merged probability rasters (§4.3)
└── vectors/                          # post-inference: vectorized polygons (§9.3)
```
Inputs read from elsewhere (not under this prefix): 2025 Planet quads `gs://rts-arctic-usw1/global_quarterly/2025/q3/`; S2 composites for NDVI `gs://rts-arctic-usw1/S2_RGB/2025_south/`.

This section owns the post-calibration deployment-package layout. MLflow-side artifacts produced during training (per-epoch metrics, figures, `run_summary.md`, etc.) are spec'd in `training/experiments.md §1.3`; on-disk checkpoint payloads (`best_deployment.pth`, `resume_latest-*.pth`) in `training.md §4.3`.

Note: `scripts/package_model.py` renames the training-time `best_deployment.pth` to `weights.pth` when assembling this deployment package — same EMA state dict, new filename.

### 2.3 Docker Environment

**Base Image**: Same as training — see `computing/docker_training.md` for the authoritative Dockerfile and base image.

**Additional Inference Requirements**:

| Package | Purpose |
|---------|---------|
| google-cloud-storage | GCS bucket access |
| geopandas | Vector operations |
| shapely | Geometry handling |
| pyproj | Coordinate transformations |

**Docker Configuration for Inference**:

| Flag | Purpose |
|------|---------|
| `--gpus all` | Enable GPU access |
| `-v /path/to/cache:/cache` | Local cache for tiles |
| `--env GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` | GCS authentication |

**GCS Authentication**:
1. Create service account with Storage Object Viewer and Storage Object Creator roles
2. Download JSON key file
3. Mount key file into container and set environment variable

---

## 3. Input Data

### 3.1 Source Imagery

| Attribute | Value |
|-----------|-------|
| Product | Global Quarterly PlanetScope Basemap |
| Year | 2025 |
| Quarter | Q3 (July-September) |
| Bands | RGB |
| Resolution | 4.77 m projected (EPSG:3857; Web Mercator zoom 15; constant in projected space). Ground sample varies with latitude (see `training.md §8.3`). |
| Coverage | Permafrost ∩ Planet (`domain/circumpolar_south_domain.geojson`), ~45–76°N — see §3.2 |
| CRS | EPSG:3857 |

### 3.2 Coverage & tile count (measured 2026-06-17)

Replaces the earlier ~3.4M/~7.5M projected estimates with the actual scan of the 2025 download.

| Parameter | Value (measured) |
|-----------|------------------|
| Quad index | **309,100 quads** under `gs://rts-arctic-usw1/global_quarterly/2025/q3/` (`scripts/build_quad_index.py` → `/mnt/outputs/inference/quad_index_2025q3.csv`) |
| Raw coverage extent | 45.5–76°N, all longitudes; **land-focused** — Planet visual basemaps omit open ocean, so the quad grid is already land-only |
| Inference domain | `domain/circumpolar_south_domain.geojson` (= permafrost ∩ Planet, per `domain/inference_domain.md`; EPSG:3413) — **20.68M km²**; parent permafrost region (`circumpolar_domain.geojson`) = 21.34M km² |
| **Tile count** (stride 344, domain-masked) | **41,567,572 tiles** (`scripts/generate_tile_grid.py` → `scripts/mask_tiles_to_domain.py` centroid-in-domain → `tiles_2025q3_domain_full.csv`) |
| Coverage completeness | real missing quads over land ≈ **0.05%** (143 enclosed cells of 309,100); ocean correctly absent. The grid is built from the quad index, so **ocean + gaps carry zero tiles — the count already excludes them** (no NoData/ocean reduction left to apply) |
| Runtime (32× L4, no-TTA, 1 scale) | **~8–25 h**, ≈270–770 GPU-hr, ~$170–500 (see §2.1) |

Overlay of the four components (permafrost region · Planet coverage · inference domain · tile-count
region) at `/mnt/outputs/inference/domain_overlay.png` (built by `scripts/plot_domain_overlay.py`;
coverage-gap QA by `scripts/check_coverage_gaps.py`).

> **Domain note:** the inference domain extends to ~45°N at the boreal margin (not just 60–74°N) and is
> bounded on the north (~74–76°N) by PlanetScope's coverage guarantee — the permafrost region north of
> that (`circumpolar_north_domain.geojson`) has no 2025 Planet basemap and is excluded.

### 3.3 EXTRA channel source — NDVI from S2 composites (on the fly)

The locked v2 recipe is **RGB + NDVI**. Per-tile EXTRA materialization was abandoned for inference
scale (~45 TB / ~192 VM-days for 41.57M tiles; diary 2026-06-24). Instead NDVI is derived **on the fly**,
mirroring how RGB is mosaicked from Planet quads:

| Item | Value |
|------|-------|
| Source | Bulk Jul–Sep `s2_sr_composite` (`data/extra_channels`), bands B4,B3,B2,B8, EPSG:3857, 10 m, COG — exported by `scripts/export_s2_composites.py` to `gs://rts-arctic-usw1/S2_RGB/2025_south/` |
| Index | `scripts/build_s2_index.py` → `inference/s2_index.py` (cell bounds + GCS path; one-time GCS scan) |
| Reader | `inference.tiles.read_ndvi_tile`: window intersecting cells, `NDVI=(B8−B4)/(B8+B4)` (band 1 / band 4), bilinear-resample 10 m → tile grid, mosaic; no-coverage → NaN |
| Consistency | Same composite recipe + NDVI formula as training (`s2_image`); NDVI is scale-invariant so the /10000 cancels. Stacked as channel 4 and normalized via the shared `apply_norm` — NaN → 0, identical to training EXTRA (CLAUDE Rule 3) |
| Invocation | `scripts/inference.py --s2-index s2_index_2025_south.csv` (required when the package declares EXTRA) |

The S2 composite is read per tile alongside the Planet quads, so the §11.3 quad-cache (A1.1) should
cache composite reads too. NoData: only the **RGB** mask drives the output NoData (§5.3); NDVI gaps are
neutralized to 0, not propagated.

---

## 4. Tiling Strategy

### 4.1 Tile Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Tile size | 512×512 pixels | Matches training tile size |
| Spatial coverage | ~2.4 km × 2.4 km projected (512 × 4.77 m) | Web Mercator zoom 15; constant in projected space, shrinks with latitude on the ground |
| CRS | EPSG:3857 | Consistent with training |
| Format | GeoTIFF | Preserves georeferencing |

### 4.2 Overlap Configuration (math-derived)

Overlap exists so that an RTS straddling a tile boundary is captured fully in *some* tile. At tile size T = 2442 m projected (512 px × 4.77 m), an RTS of length L (in projected meters) fits entirely in at least one tile iff stride **S ≤ T − L**.

Known RTS-size distribution (source: training label statistics):

| RTS population | Max bbox edge L | Required stride S = T−L | Stride in pixels (S/4.77) | Overlap p = 1 − S/T | Compute multiplier (1/(1−p))² |
|----------------|-----------------|-------------------------|---------------------------|---------------------|-------------------------------|
| 99.5% | ≤ 800 m | ≤ 1642 m | ≤ 344 px | ≥ 33% | 2.22× |
| 99.9% | ≤ 1300 m | ≤ 1142 m | ≤ 239 px | ≥ 53% | 4.59× |

**Default: stride 344 px (~33% overlap), persisted as `configs/deployment.yaml.inference.stride_px`** — chosen for the 99.5% RTS-size row above. Change the config to retune.

**Overlap rationale**: An RTS split at a tile boundary may show only floor in tile A and only headwall in tile B. At the default stride, intermediate tiles will contain both features for >99.5% of the RTS size distribution.

Flow:
1. Single pass at the configured `inference.stride_px` across all filtered tiles.
2. Merge per §4.4 → regional probability rasters.
3. Threshold candidate polygons.

### 4.3 Overlap Aggregation

Fusion method: **distance-from-tile-center weighted average**, Gaussian weighting with σ = 128 px in tile coordinates, normalized per pixel. **Implementation refinement (2026-06-12, found by the tiny-area validation):** the weight is a separable *edge-zeroed* Gaussian (per-axis `g(i) = exp(−(i−c)²/2σ²) − g_edge`, `w = g⊗g`) rather than the plain radial form — the radial Gaussian retains weight ≈ 0.135 at the tile edge, so contributions appear/disappear discontinuously across seams (measured ~7× elevated probability gradients on seam lines). Zeroing the edge makes contributions fade in continuously; σ and the center-trust rationale are unchanged. Side effect: a pixel covered only by tiles' outermost row/column (the 1-px ring at an unchunked AOI boundary) has zero total weight → NoData.

Rationale: edge-of-tile predictions come from locations where the model has seen fewer surrounding pixels within *this* tile. Center-of-tile predictions are more trustworthy. Max fusion (taking the highest probability across tiles) is recall-biased and contradicts §1's precision-over-recall goal; averaging preserves calibration.

Implementation: per-tile probability rasters persist to GCS first, then a separate regional merge pass computes the weighted average per output pixel.

**Note on training loss**: this edge-down-weighting is an **inference-only fusion decision** across multiple predictions of the same physical pixel. Training loss weights all pixels uniformly — weighting by tile position during training would teach the model to ignore edges, which would hurt exactly the inference scenario where edge predictions are being averaged in. Orthogonal decisions.

### 4.4 Tile Grid Generation

The inference tile grid is **pre-filtered externally** (land-only, permafrost zones) before the inference pipeline runs. The inference code receives a pre-filtered tile list and processes it as-is — no filtering logic inside the inference container.

1. Define bounding box for inference region (or per-region bounding boxes)
2. Apply land/permafrost filtering externally (outside this pipeline)
3. Generate tile grid using `configs/deployment.yaml.inference.stride_px`
4. Save filtered tile grid as CSV with tile IDs and bounding boxes → this is the `--tile-list` input to the inference script

---

## 5. Normalization

### 5.1 Loading Statistics

**Critical**: Use the exact normalization statistics from training. Training-inference consistency on normalization is codified in `training.md §4.1` and §4.5.

1. Load `normalization_stats.json` from the deployment package.
2. **Assert channel-name binding** (`training.md §4.5`): `stats["rgb"]["channel_names"] == ["R", "G", "B"]` and, if EXTRA channels are declared, `stats["extra"]["channel_names"] == [c.name for c in model_config["channels"]["extra"]]`. Prevents silent position-vs-name mismatches if the 2025 basemap API changes band ordering. Abort inference on mismatch.
3. Apply mean subtraction and std division per channel using the name-bound stats (not positional).

### 5.2 Application

Use the exact normalization methods and statistics identically to training.

### 5.3 NoData Handling

Per `training.md §4.4`, the training side labels NoData pixels as ignore=255 so the model never receives gradient signal from them. The inference side mirrors this:

| Case | Treatment |
|------|-----------|
| Full-NoData tile | Skipped at the tile-list stage. Manifest-logged with reason `"all_nodata"`. |
| Partial-NoData tile | Predict normally (substitute per-channel training mean for NoData pixels before normalization, matching training). After prediction, **mask the output**: `pred_raster[input_nodata_mask] = -1.0` (the NoData value declared in §9.1). |

Rationale: the model output on NoData input is undefined. Propagating NoData through to the probability raster ensures downstream overlap aggregation (§4.4) and vectorization (§9.3) treat those pixels correctly.

### 5.4 Pre-deployment drift check

Before running full inference on a new region, run `scripts/check_inference_normalization.py` (owned by the training team) against a sample of 2025 tiles from that region:
- Computes per-channel mean/std on the 2025 sample.
- Compares to `normalization_stats.json`.
- Reports drift as `|Δmean| / σ_training` and `|σ_sample / σ_training − 1|` per channel.
- **Concern thresholds**: |Δmean| > 0.5σ_training OR |σ_sample / σ_training − 1| > 0.25. If tripped, pause deployment and investigate — likely distribution shift from 2024 to 2025 imagery, a region-specific radiometric issue, or a basemap-API change.

**Which baseline — tiles vs whole quads (added 2026-08-24, interannual campaign).** The thresholds
above are calibrated for a sample of **pre-cut tiles**. `--quad-index` mode samples **random whole
quads** instead, which is the only option for a year that exists before any tiles are cut. Those two
samples are not comparable against the same baseline: `normalization_stats.json` was computed on
17,951 curated **RTS-centric** training tiles, whereas random quads also cover open water, snow and
bare rock. A random-quad sample of *any* year therefore reads as wider —

| sample vs `normalization_stats.json` | worst σ-ratio | verdict |
|---|---|---|
| 2025 quads (the imagery the delivered map was made from) | **+0.256** (B) | trips |
| 2022 quads | **+0.295** (B) | trips |

so the gate as written does not discriminate. **For a whole-quad sample, compare against the 2025
quad baseline** (`/mnt/outputs/inference/drift_report_2025q3_control.csv`), same sampling method and
`--sample-seed`. On that baseline 2022's worst mean drift is 0.095σ and worst σ-ratio +0.047 — i.e.
2022 is radiometrically consistent with the deployed-on imagery. `campaign/stages.py::_drift_evidence`
implements this comparison; the baseline path is `campaign.yaml:paths.quad_baseline`.

---

## 6. Multi-Resolution Inference

### 6.1 Rationale

RTS range from ~50m to 2+ km. A single resolution cannot optimally detect all sizes:
- Native 4.77 m projected: Good for small-medium RTS, may miss context for large RTS
- Downscaled: Larger effective field of view captures large RTS

### 6.2 Scale Configuration

| Scale | Effective Resolution (projected) | Field of View (projected) | Target RTS |
|-------|----------------------------------|---------------------------|------------|
| 1.0 | 4.77 m (native) | 2.4 km | Small-medium (50m-500m) |
| 0.5 | 9.55 m | 4.9 km | Medium-large (200m-1km) |

**Phase 1 default: scale 1.0 only.** Multi-scale deployment is gated by a feasibility test (§6.4). Training is at scale 1.0 only per `training.md §8.3`, but the fractal nature of earth features plus the encoder's multi-scale receptive fields suggest scale-0.5 inference *may* work without retraining. Test before assuming.

### 6.3 Multi-Scale Procedure

For each tile location:

**Scale 1.0 (native)**:
1. Load 512×512 tile at native resolution
2. Normalize using training statistics
3. Run inference → probability map P_1.0

**Scale 0.5** (only if §6.4 gate passes):
1. Load 1024×1024 region centered on tile location
2. Downsample to 512×512 (bilinear interpolation)
3. Normalize using training statistics
4. Run inference → probability map at 512×512
5. Upsample prediction back to 1024×1024
6. Crop center 512×512 → P_0.5

**Edge case — basemap boundary.** A scale-0.5 fetch needs 1024 × 1024 projected pixels centered on the tile. If any side of that window falls outside the basemap coverage of the input region (geographic edge or NoData border wider than 256 px), **skip scale 0.5 for that tile** and treat its scale-0.5 prediction as NoData in the §7.3 fusion. Do not pad with reflection or zeros — the model has not seen such patterns. The §7.3 valid-scales rule degrades gracefully to scale-1.0-only for these edge tiles.

### 6.4 Multi-Scale Feasibility Gate

Multi-scale evaluation is **optional and deferred**: the canonical Test-Realistic result comes from `scripts/evaluate_test.py` at `scales: [1.0]` (see `training.md §4.6`). Multi-scale runs **after** the 1× number is locked, lives in this Phase 2 inference pipeline, and is gated as below — it never runs inside `evaluate_test.py`.

Run once per trained model, post-calibration, pre-deployment. Owned by `scripts/inference_feasibility.py` (Phase 1 Step 8.5). Procedure:

1. Run scale-0.5 inference on the val set using the baseline (scale-1.0-trained) model.
2. Average-fuse with cached scale-1.0 val predictions (per §7.3).
3. Compute three measurements at the calibrated threshold:
   - PR-AUC on the **large-RTS subset** (bbox > 500 m).
   - PR-AUC on the full val set.
   - Global false-positive-rate delta vs scale-1.0-only.

**Decision gate** — ship multi-scale if **both**:
- Large-RTS PR-AUC gain ≥ +2%
- Global FP-rate delta ≤ +10%

Otherwise keep `scales: [1.0]`. Context-expanded training (fetch 2× physical area, downsample to 512) is a Phase-1.5 consideration triggered only if the gate fails *and* post-inference analysis identifies large-RTS recall as the primary precision bottleneck.

The gate's outcome is written into `deployment_config.yaml.scales` and the feasibility report is attached to the MLflow run.

---

## 7. Test-Time Augmentation (TTA)

### 7.1 Configuration

| Setting | Transforms | Speed Multiplier |
|---------|------------|------------------|
| Disabled | None | 1× |
| Minimal | Identity, hflip | 2× |
| Standard | Identity, hflip, vflip, rot180 | 4× |

**Recommendation**: For pan-arctic inference, use **Minimal TTA** (2×) as balance between accuracy and compute cost. Full TTA on the §3.2 tile count is expensive.

### 7.2 TTA Procedure

For each input tile:
1. Original → predict → P_orig
2. Horizontal flip → predict → flip back → P_hflip
3. Average: P_tta = (P_orig + P_hflip) / 2

### 7.3 Combining TTA with Multi-Scale

Order of operations (matches §8.2 step 2):
1. For each scale:
   a. For each TTA transform:
      - Run model → raw **logits**.
      - Apply temperature scaling: `scaled_logits = logits / temperature` (per `training.md §12.1`).
      - Apply sigmoid: `probs = sigmoid(scaled_logits)`.
      - Apply the inverse TTA transform to the probability map.
   b. Average TTA probability maps within this scale (arithmetic mean).
2. **Average probability maps across scales** (arithmetic mean over **valid** scales — see NoData rule below), then apply the calibrated threshold for the binary mask.

Temperature scaling **must be applied to logits before sigmoid**, not to probabilities. Folding temperature into the per-pass sigmoid keeps the math consistent with the calibration definition in `training.md §12.1`.

Max fusion was the original spec but biases toward recall (any scale says "positive" → positive), directly contradicting §1's precision-over-recall priority. Arithmetic averaging preserves probability calibration and lets the threshold do its job.

**NoData handling during scale fusion**: a per-pixel scale prediction is treated as NoData when it equals `−1.0` (the §5.3 sentinel) **or** when it falls inside the input NoData mask of that scale's tile fetch. Per-pixel fusion rule: arithmetic mean over the valid scales for that pixel. If all scales are NoData at a pixel, the fused output is `−1.0`.

Total inference passes per tile location: n_scales × n_tta_transforms

| Configuration | Passes per Location |
|---------------|---------------------|
| 1 scale, no TTA (Phase 1 default) | 1 |
| 1 scale, minimal TTA | 2 |
| 2 scales, no TTA | 2 |
| 2 scales, minimal TTA | 4 |
| 2 scales, standard TTA | 8 |

### 7.4 TTA Cost–Benefit

Pan-arctic cost analysis on A100 (~$3.67/hr on-demand). **Note:** the wallclock / GPU-hr / cost
columns below were computed against the *superseded* ~7.5M-tile estimate; the measured count is
**41.57M** (§3.2, stride 344), so scale those absolute figures **~5.5×**. The per-config
*ratios* (passes/tile, throughput) — the point of this table — are unchanged.

| Config | Passes/tile | Throughput (tiles/s) | Wallclock @ ~7.5M (superseded est.) | GPU-hrs | Cost |
|--------|-------------|----------------------|------------------|---------|------|
| No TTA | 1 | ~150 | 14 hr | 14 | ~$50 |
| Minimal (identity, hflip) | 2 | ~75 | 28 hr | 28 | ~$100 |
| Standard (identity, hflip, vflip, rot180) | 4 | ~37 | 56 hr | 56 | ~$210 |
| Full D4 (8 symmetries) | 8 | ~19 | 110 hr | 110 | ~$400 |

Against the $70K training+inference budget, all four configs are affordable — the choice is driven by **precision preservation at the calibrated threshold**, not cost. TTA averaging can either improve calibration (good) or pull confident positives below the threshold (bad for precision-over-recall).

**TTA is validated before deployment, not assumed**: Step 8.5b of Phase 1 measures val PR-AUC and precision@threshold under each TTA config using the cached val predictions. Ship the cheapest config that (a) gains ≥ 1% PR-AUC *and* (b) drops precision by ≤ 0.5% at the calibrated threshold. Default in `configs/deployment.yaml`: `tta: none`.

---

## 8. Batch Inference

### 8.1 Batching Strategy

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 64-128 | Tune based on GPU memory |
| Tile loading | Async prefetch | Overlap I/O with compute |
| GPU utilization target | >90% | Monitor with nvidia-smi |

### 8.2 Inference Loop

1. **Initialize**:
   - Load deployment package directory (see §2.2). Required files: `weights.pth`, `normalization_stats.json`, `model_config.yaml`, `deployment_config.yaml`.
   - Build model per `model_config.yaml`; load `weights.pth` into the model state dict (already EMA — see `training.md §4.3`).
   - Load `normalization_stats.json`; assert channel-name binding per §5.1.
   - Load `deployment_config.yaml`: `threshold`, `temperature`, `tta`, `precision`, `torch_compile`, `scales`, `fusion`. These must match the values used during calibration (`training.md §4.6`).
   - `model.eval()`; if `torch_compile: true`, run `torch.compile(model)` here.
2. **Tile iteration** (sequence per §7.3):
   - Load batch of tiles from GCS (with prefetching).
   - Handle NoData per §5.3 (skip full-NoData tiles; mean-substitute partial NoData before normalization).
   - Normalize batch per §5.2.
   - For each scale in `scales`:
     - For each TTA transform (per `tta`):
       - Forward pass → raw logits.
       - Apply temperature: `scaled_logits = logits / temperature`.
       - Apply sigmoid: `probs = sigmoid(scaled_logits)`.
       - Apply inverse TTA transform to the probability map.
     - Average TTA probability maps within this scale.
   - Fuse across scales per §7.3 (arithmetic mean over valid scales).
   - The **probability raster is written pre-threshold**; the calibrated threshold is applied separately to produce the binary mask (§9.2).
   - Mask NoData in output raster per §5.3 (`pred[nodata_mask] = -1.0`).
   - Save probability tile to GCS.
3. **Progress tracking**: Log completed tiles, estimated time remaining.
4. **Checkpointing**: Save progress every N tiles for resumability.

### 8.3 Resumability

The inference job must be resumable after interruption:
1. Maintain manifest of completed tiles in `inference_log.json`
2. On restart, load manifest and skip completed tiles
3. Use atomic writes to GCS (write to temp, then rename)
---

## 9. Output Specification

### 9.1 Probability Raster

| Attribute | Value |
|-----------|-------|
| Format | Cloud-Optimized GeoTIFF (COG) |
| Data type | **scaled_uint8** (deploy default) — prob×250 → uint8 [0,250], NoData 255; ~8 KB/tile, ~0.3 TB full run, re-threshold precision 0.004. `float32` (NoData −1.0, exact, ~570 KB/tile, ~24 TB) is the alternative. Set by `configs/deployment.yaml:inference.output_dtype`. |
| Valid range | [0.0, 1.0] (uint8: [0,250] → ÷250) |
| NoData sentinel | scaled_uint8: **255** · float32: **−1.0** (out-of-range; uniquely identifies NoData) |
| CRS | EPSG:3857 |
| Resolution | 4.77 m projected (native; Web Mercator zoom 15) |
| Compression | Deflate |

Encode/decode live in `inference/writer.py` (`write_probability_tile(…, dtype=)` / `read_probability_tile` — SSoT; `merge_predictions.py` decodes either encoding). `output_dtype` is **not** a §14 calibration-bound key (it does not affect thresholds/probabilities, only their on-disk quantization).

### 9.2 Binary Mask

| Attribute | Value |
|-----------|-------|
| Format | Cloud-Optimized GeoTIFF (COG) |
| Data type | UInt8 |
| Values | 0 (background), 1 (RTS) |
| NoData value | 255 |
| CRS | EPSG:3857 |
| Resolution | 4.77 m projected (native; Web Mercator zoom 15) |
| Compression | Deflate |

Threshold applied: Use calibrated threshold from training (documented in model config).

### 9.3 Vector Output

| Attribute | Value |
|-----------|-------|
| Format | GeoPackage (.gpkg) |
| Geometry | Polygon (MultiPolygon for fragmented) |
| CRS | EPSG:3857 |

**Attributes per polygon**:

| Field | Type | Description |
|-------|------|-------------|
| rts_id | Integer | Unique identifier |
| area_m2 | Float | Polygon area in square meters (geodesic) |
| perimeter_m | Float | Polygon perimeter in meters (geodesic) |
| centroid_lat | Float | Centroid latitude (WGS84) |
| centroid_lon | Float | Centroid longitude (WGS84) |
| mean_prob | Float | Mean probability within polygon |
| max_prob | Float | Maximum probability within polygon |
| detection_scale | String | Scale(s) that detected this RTS |
| tile_ids | String | Comma-separated tile IDs containing this RTS |

### 9.4 Inference Metadata

Save with each inference run:

**inference_log.json**:

| Field | Description |
|-------|-------------|
| model_version | Model identifier (e.g., `rts-v2-seed42`) |
| deployment_package_path | `gs://` URI of the deployment package directory |
| model_checkpoint_sha | SHA256 of `weights.pth` |
| inference_date | ISO timestamp |
| basemap_version | 2025-Q3 |
| scales_used | e.g., `[1.0]` or `[1.0, 0.5]` (per §6.4 gate) |
| fusion_method | `weighted_mean` \| `max` \| `consensus` (default `weighted_mean`) |
| tta_config | `none` \| `minimal` \| `standard` \| `full` |
| precision | `bf16` \| `fp16` \| `fp32` (must match calibration) |
| torch_compile | boolean (must match calibration) |
| threshold | Calibrated threshold from `deployment_config.yaml` |
| temperature | Calibrated temperature (§12.1 of training.md) |
| stride_px | value used at run time, mirrors `configs/deployment.yaml.inference.stride_px` |
| overlap_aggregation | `gaussian_weighted_mean`, σ from `configs/deployment.yaml.inference.fusion_sigma_px` |
| n_tiles_processed | Total tiles |
| n_tiles_skipped_nodata | Tiles skipped per §5.3 |
| n_tiles_with_detection | Tiles with any RTS prediction |
| total_rts_area_km2 | Sum of predicted RTS area |
| processing_time_hours | Wall clock time |
| gpu_type | e.g., "NVIDIA H100" |

---

## 10. Quality Control

### 10.1 Sanity Checks During Inference

| Check | Action if Failed |
|-------|------------------|
| Tile has valid data (not all NoData) | Skip tile, log warning |
| Prediction values in [0, 1] | Clip and log error |
| Tile georeferencing valid | Stop and investigate |
| GPU memory stable | Reduce batch size |

### 10.2 Post-Inference Validation

Performed before releasing results (detailed in post-inference.md):
- Visual inspection of sample predictions
- Comparison with known RTS locations
- False positive analysis
- Regional performance assessment

---

## 11. Performance Optimization

### 11.1 I/O Optimization

| Technique | Description |
|-----------|-------------|
| Tile caching | Cache frequently accessed tiles locally |
| Prefetching | Load next batch while current batch processes |
| COG format | Cloud-Optimized GeoTIFF enables efficient partial reads |
| Batch GCS operations | Upload predictions in batches, not per-tile |

### 11.2 GPU Optimization

| Technique | Description |
|-----------|-------------|
| Mixed precision | BF16 on A100/H100 (preferred — no dynamic loss scaling); FP16 fallback on older GPUs. Must match `training.md §4.6` calibration precision. The operative source of truth is `configs/deployment.yaml.precision`; both training-time AMP and inference read from there. |
| Batch size tuning | Maximize GPU utilization |
| Multiple streams | Overlap data transfer and compute |
| Model compilation | **Opt-in only.** `torch.compile()` changes numerics slightly; if enabled at deployment but disabled during calibration (or vice versa), the calibrated threshold is systematically wrong. Phase 1 baseline: `torch_compile: false`. Enable only when a benchmark demonstrates > 15% throughput gain *and* calibration is re-run with compile enabled. |

### 11.3 Measured Throughput (2026-06-12)

A100-80GB, bf16, no TTA, scale 1.0, 552-tile Banks Island AOI, quads streamed from GCS via
windowed reads, no caching:

Extrapolations below use the *superseded* ~7.5M-tile estimate; the measured count is **41.57M**
(§3.2), so scale them **~5.5×** (e.g. the 8-worker row → ~1100 GPU-h ≈ 137 h on the 8×A100 node).

| Configuration | Tiles/s (measured) | Extrapolation @ ~7.5M (superseded est.) |
|---------------|--------------------|----------------------------|
| batch 64, 4 workers | 6.4 | — |
| **batch 64, 8 workers** | **10.5** | **~198 GPU-h ≈ 25 h on the 8×A100 node** |
| batch 128, 8 workers | 6.9 | — |
| batch 128, 16 workers | 7.3 | — |

The pipeline is **GCS-read-bound, not GPU-bound** (workers 4→8 nearly doubled throughput;
larger batches did not help): at stride 344 each quad is re-opened by ~36 overlapping tiles
with no reuse. **Highest-value optimization before the production run**: quad-level caching —
a per-worker LRU of decoded quads, or restructuring the loop to process all tiles of a
quad-block per fetch; expected ~10–30× (toward the original ~150 tiles/s estimate ⇒ a full
pass at ~14 GPU-h). Cheaper GPUs (L4) are viable since the GPU idles at current throughput.

**Implemented (2026-06-25, `inference/tiles.py`):** the three I/O fixes are now in code —
(a) a per-worker LRU of **open** rasterio handles (`_OpenDatasetCache`) so an overlapping
quad is opened once, not ~36×, and GDAL's block cache serves the repeated windows;
(b) an **STRtree spatial index** (`_BBoxIndex`) replacing the O(N) per-tile boolean scan over
the ~309k-row quad/S2 index; (c) **spatial tile ordering** (`_spatial_sort`) so a batch's
tiles share quads and hit the cache. All three are bit-identical to the prior mask path
(candidates re-filtered with the exact strict-overlap test in original order;
`tests/test_inference_pipeline.py::test_read_tile_hits_path_identical_to_mask`).
**Still TODO — Tier-2 re-benchmark:** rerun the 552-tile Banks Island AOI on the L4/A100 VM
with real 2025 quads, confirm tiles/s rises toward the ~150 ceiling and GPU util climbs, and
replace the table above. Consider raising `GDAL_CACHEMAX` and `--num-workers` once GPU-bound
(§11.2). Quad-handle cache size is `_OPEN_CACHE_SIZE` (default 16/worker).

---

## 12. Workflow Integration

### 12.1 PDG Workflow

The inference pipeline integrates with the existing PDG (Permafrost Discovery Gateway) workflow infrastructure developed for DARTS inference.

**Integration points**:
- Input: Basemap tiles from GCS
- Output: Prediction tiles and vectors to GCS
- Logging: Compatible format for PDG monitoring
- Parallelization: Workflow handles VM orchestration

### 12.2 Docker Entry Point

The inference container exposes a CLI interface for PDG workflow integration:

```bash
python scripts/inference.py --config configs/deployment.yaml --tile-list tiles.csv
```

- `--config`: `configs/deployment.yaml` — single source for threshold, temperature, scales, tta, precision, torch_compile, fusion, stride_px, fusion_sigma_px (see §2.2 deployment package)
- `--tile-list`: CSV file with tile IDs and bounding boxes to process (pre-filtered by PDG/RTS team)
- Output: Prediction tiles written to GCS path defined in config; `inference_log.json` updated on completion

### 12.3 Parallelization Strategy

**Tile-level parallelism** (managed by PDG workflow):
1. RTS team generates the full filtered tile grid (CSV)
2. PDG team (Luigi/Todd) partitions the CSV into chunks and spawns VMs
3. Each VM runs the inference container with its assigned tile list chunk
4. RTS team merges outputs after all chunks complete

**Within-VM parallelism**:
- Single GPU processes tiles in batches
- Multiple CPU workers handle I/O prefetching
- No multi-GPU within single VM (simplifies code)

### 12.4 Coordination

| Responsibility | Owner |
|----------------|-------|
| Tile grid generation (filtered CSV) | RTS team |
| VM orchestration + tile partitioning | PDG team (Luigi/Todd) |
| Inference Docker container | RTS team |
| Output merging | RTS team |
| Quality control | RTS team |

**Interface contract** (to finalize with PDG team):
- Input: `configs/deployment.yaml` + `tiles.csv` (tile_id, bbox columns)
- Output: Prediction tiles at `{config.output_path}/{tile_id}.tif`; log at `{config.output_path}/inference_log.json`

---

## 13. Inference Checklist

### Pre-Inference
- [ ] Model artifacts uploaded to GCS (model, normalization stats, config)
- [ ] Docker image built and pushed to container registry
- [ ] Tile grid generated and validated
- [ ] GCS permissions configured (service account)
- [ ] Test inference on small region successful
- [ ] Throughput estimate matches budget

### During Inference
- [ ] Progress monitoring active
- [ ] GPU utilization >90%
- [ ] No error accumulation in logs
- [ ] Checkpoint saves working

### Post-Inference
- [ ] All tiles processed (compare manifest to grid)
- [ ] Merged rasters generated
- [ ] Vectorization complete
- [ ] Metadata logged
- [ ] Sanity checks passed
- [ ] Ready for quality control (post-inference.md)

---

## 14. Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| OOM errors | Batch size too large | Reduce batch size |
| Slow inference | I/O bottleneck | Enable prefetching, use local cache |
| Inconsistent predictions | Wrong normalization | Verify normalization_stats.json hash |
| Missing tiles in output | Job interrupted | Check manifest, restart from checkpoint |
| Global FP rate ≫ val reported | Train-inference distribution shift | Run `scripts/check_inference_normalization.py` on a 2025 sample (§5.4) and compare per-channel drift vs `normalization_stats.json`. If drift is real, consider histogram matching or retraining with 2025 data included. |
| Regional FP rate ≫ val reported | Region has characteristics under-represented in val | Collect 50–100 hand-labelled negatives from that region; calibrate a region-specific threshold per `training.md §6.4`. Do not re-run the global calibration — re-running on the same val set cannot fix a regional bias. |
| Calibration-deployment mismatch | Precision / TTA / compile differ between calibration and run | Verify `inference_log.json.precision`, `.tta_config`, `.torch_compile` match the deployment package's `deployment_config.yaml`. Inference aborts at startup on mismatch — implemented in `inference/predictor.py:assert_runtime_matches_package` (called from `inference/runner.py:build_context`, so both the CLI and the queue worker enforce it; covered by `tests/test_inference_pipeline.py::test_runtime_package_mismatch_aborts`). |
| Predictions all zero | Model loading error | Verify `weights.pth` SHA256 in run log matches deployment package; confirm EMA weights loaded (not random-init). Test on a known-positive val tile first. |