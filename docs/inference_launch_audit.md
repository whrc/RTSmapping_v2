# Pre-launch full audit — pan-Arctic inference (2026-07-05)

Comprehensive audit of everything on the launch path before the full 41.57M-tile pan-Arctic
inference run, per the approved audit plan. Two goals: (1) every past decision rests on solid,
re-derivable facts; (2) the model + launch machinery are scientifically and engineering sound.
Branch: `audit-prelaunch` (from `main` @ `1caf5e3`). All numbers below were re-derived from primary
artifacts during this audit — commands and artifact paths cited per row.

**Verdict summary: GO, gated on the standard launch gates.** All science-side checks pass (evidence
chain intact — no number failed re-derivation, no locked decision rests on a faulty fact). The launch
*machinery*, by contrast, was **not** launch-ready: **10 distinct defects were found and fixed**, at
least 6 of them hard blockers that the paper trail could never have surfaced — every one of them lived
in the fleet/inference-execution path and was only exposed by actually loading GCS packages and booting
a real L4 VM (§E5, §E9, §E10). **The single most important quantitative finding is the throughput
benchmark: the real per-L4 rate is ~4.2 tiles/s (3-model ensemble, GPU-bound), so the full run is
~2–5 days, not the plan's 12–29 h** (§E10). Remaining gates before Phase-4 launch: the last 2 S2 cells
(GEE tasks PENDING, rebuild `s2_index` once they land), a re-run `s2_index` upload, and explicit go +
spend at the corrected ETA. All fixes are on branch `audit-prelaunch` (8 commits); the rebuilt+pushed
`rts-infer:v1` (digest `7772dbc7…`) contains the code-side fixes, and the fleet shell fixes are in the repo.

---

## Pillar 1 — Scientific soundness (evidence chain)

### S1 — Data integrity: PASS

| Check | Result | Evidence |
|---|---|---|
| Split region disjointness | PASS — train/val/test region sets pairwise disjoint | `splits.yaml` (40/5/4 regions) |
| **Spatial** leakage (not just by-name) | PASS — 0 overlapping cross-split tile pairs in EPSG:3857; tightest gap 2446 m vs 2442 m tile edge (abutting, not overlapping); val 5.9 km clear | projected-space pairwise scan of all 22,259 centroids |
| Tile accounting | PASS — 17,951 train + 2,151 val + 2,157 test = 22,259; 1,718 pos / 20,541 neg; Tile_IDs unique | `metadata.csv` |
| Test composition | PASS — 107 pos / 2,050 neg matches ledger J and `effb5_ensemble_metrics.json` | |
| D1.1 metadata delta | PASS — exactly 50 rows removed (49 all-black negs + vjn7 old row), 29 delta rows added (28 restored pos + vjn7 promoted); "+25 train positives" exact (20 Taimyr + 2 Brooks + 3 Canadian Middle); the 4 delta positives in val/test regions are unused in training (`train.py:217-238` — train from combined roots, val/test from primary only) → no leakage | metadata diff D1.0 vs D1.1 + delta |
| Norm-stats provenance | PASS — all 3 packages bit-identical (md5 `c82fcc…`) to canonical `v1_splits/normalization_stats_ndvi.json`; `n_tiles_used` 17,951 == train split; pooled recompute on a 800-tile sample matches (means ≤2.3%, stds ≤5% — sampling noise) | recompute in Docker |
| Label values | PASS — 300/300 sampled label tiles ⊆ {0,1,255} | |

### S2 — Headline numbers re-derived from primary artifacts: PASS (all match)

| Number (ledger claim) | Re-derived | Source artifact |
|---|---|---|
| 3-seed val PR-AUC mean 0.9218 | 0.916729/0.921627/0.927016 → 0.9218 ✓ | `runs/*/run_summary.json` |
| Gate μ₀=0.7912, σ₀=0.0056, G=0.0112 | ✓ (sample σ of 0.7899/0.7863/0.7973) | phase0c run summaries |
| Ensemble PR-AUC 0.9393, T=0.512321 | 0.9392672, T=0.512320566 ✓ | `calibration_report.json .ensemble` |
| TTA verdict (none; D4 hurt, hflip +0.0014 sub-gate) | table: none 0.9302 / minimal 0.9316 / full 0.9234 ✓ | `calibration_report.json .tta_selection` |
| Val op point thr 0.65/mb 80: P 0.611/R 0.439/F1 0.511, pix-P 0.987 | grid row exact ✓ | `object_operating_point_report.json .grid` |
| Test-Realistic obj P/R/F1 0.584/0.437/0.500; PR-AUC 0.855/0.833/0.812 (geomean 0.833) | exact ✓ | `test_realistic/effb5_ensemble_metrics.json` |
| First product (thr 0.65/mb 2000): val P 0.793/R 0.348, test P 0.768/R 0.400 | exact ✓ | `mmu_rescore/object_scorecard_{heldout_val,frozen_test}.json` |
| MMU-600: P invariant, R +3.2/+4.1 pt, floor 0.280→0.231 / 0.223→0.159 | exact ✓; `min_mapping_unit: 600` recorded, self-check True | `mmu_rescore/*_mmu600.json` |
| M1.1 3-seed 0.9029/0.9086/0.8906 (mean 0.9007) | ✓ | `/mnt/outputs/v1_1/runs/*/run_summary.json` |
| Ledger score column | zero drift — `sync_experiments.py` rewrote 113 rows with identical values (empty git diff) | |

### S3 — Decision audit: SOUND (with 3 documented judgment calls, all honestly labeled)

Every lock's evidence re-checked against its own stated bar (mean Δ ≥ G = 0.0112 + 3-seed
sign-consistency): NDVI (+0.069 ≫ G) · focal+ignore_w2 (seed-confirmed) · drop-RandomScale (+0.016,
3/3) · EffB5-over-sat-DINOv3 (fair matched re-run, dead tie 0.9191 vs 0.9218 → cheaper encoder) ·
curriculum rejected (sign-flip) · calibration chain (below) · M1.1 no-improvement. Notes:

1. **TrivialAugment** (Δ+0.0095 < G) and the **3-seed ensemble** (Δ+0.0091 < G vs best single) are
   sub-gate locks by explicit judgment — both documented as such in the ledger. Sound.
2. **thr 0.65** is the grid's object-precision maximum (user choice, precision-over-recall §1);
   the F1-argmax alternative (0.30) is recorded alongside. Sound and reproducible.
3. **M1.1 no-improvement verdict**: methodologically sound — the raw "regression" decomposes into a val-set
   change (−29 trivial negatives) + an own-threshold confound (M1.1 optimum 0.45 vs incumbent 0.65);
   calibration-free signals (test pixel PR-AUC 0.9976 vs 0.9970; MMU600 floor 0.154 vs 0.159) and
   own-threshold object F1s tie. Keep-v1.0 is the right call (incumbent, already calibrated).
4. **Calibration→deploy math chain verified end-to-end**: `calibrate.py` fits T on the pseudo-logit
   of the mean-prob ensemble == `predict_probs_ensemble` (per-seed sigmoid @T=1 → mean → logit → /T
   → sigmoid) == `evaluate_test_ensemble` (imports the same function). One recipe, three sites, no drift.
5. *(discipline note)* The frozen test set has been consulted several times post-freeze (by-region
   re-run, MMU re-scores, M1.1 comparison) — always from cached predictions and never to re-tune the
   deploy point, so adaptive-reuse risk is low; keep minimizing test touches.
6. *(nit)* The M1.1 no-improvement "test pixel PR-AUC 0.9976" is a different regime than the shipped 0.833
   geomean (unweighted vs prevalence-reweighted); the ledger uses it only as a relative tie signal —
   fine, but the regime is unstated there.

### S4 — Training↔inference consistency: PASS

- One shared normalization path (`data/normalization.apply_norm` + `fill_nodata_with_mean`) used by
  `InferenceTileDataset` — Rule 3 holds; channel-name binding asserted at package load.
- Packages carry **EMA weights**: `weights.pth` tensors **identical** to each run's
  `best_deployment.pth` `model_state_dict` (torch.equal, all 3 seeds). The earlier file-hash mismatch
  is just the stripped checkpoint wrapper. Package `checkpoint_epoch` 40 = raw-metric best;
  `run_summary.best_epoch` 35 = smoothed peak — different by design (ledger metric = best_smoothed).
- NDVI: same formula and band semantics as training (B8,B4 from the B4,B3,B2,B8 export;
  `(NIR−RED)/(NIR+RED)`; /10000 cancels); NaN→0 after norm matches training EXTRA handling.
  Numerical parity vs a training tile: §E9.
- bf16 at calibration == bf16 in `deployment.yaml` == package copies. Probability COGs are written
  pre-threshold (thr/min_blob live only at vectorization) — confirmed in `runner.run_inference`.

---

## Pillar 2 — Engineering soundness (launch mechanics)

### E1 — Repo & tests: PASS
Full suite in `rts-train:v2`: **351 passed, 1 skipped** (4m02s). Inference-path subset re-run after
the audit fixes: 67/67. `main` was already pushed & clean; audit work on `audit-prelaunch`.
*(remediation)* `data/transforms.py:78` — albumentations warns `var_limit` is not a valid GaussNoise
arg (API drift): the configured noise range is silently ignored (library default used). Training-side
only; irrelevant to inference; flag for the next retrain.

### E2 — Deep read-through of critical modules: PASS with findings (fixed where load-bearing)
`predictor.py` / `runner.py` / `tiles.py` / `writer.py` / `claim.py` line-by-line:
- TTA inverse-transform order, temperature-before-sigmoid, ensemble fusion, §5.3 NoData mask,
  §7.3 valid-scale fusion (NaN only outside the 1× footprint), resumable manifest, atomic COG
  writes (temp+rename local; single-shot upload GCS), atomic claims (`if_generation_match=0`),
  stale-reclaim — all correct.
- **FIXED (was a latent contradiction):** `load_deployment_package` still raised
  `NotImplementedError` for `scales != [1.0]` — the 2026-07-03 multiscale runner was unreachable
  end-to-end (tests construct `InferenceContext` directly and never hit the gate). Replaced with a
  1.0-in-scales validation. No effect on the [1.0] launch path.
- *(accepted risk)* `claim.heartbeat` has a read-then-write race: a worker stalled ≥ TTL (30 min)
  that resumes exactly between another worker's reclaim and its own overwrite can cause duplicate
  shard compute. Waste-only (writes idempotent; done markers + end-of-run reconciliation), window
  negligible at 34 s heartbeat cadence vs 1800 s TTL.
- *(nit)* Output "COGs" are tiled+deflate GeoTIFFs without overviews — fine for 512² tiles; the
  merged regional rasters are where overviews matter.

### E3 — Config ↔ spec ↔ package: PASS
`configs/deployment.yaml` (thr 0.65 · T 0.512321 · tta none · scales [1.0] · bf16 · compile false ·
mb 2000 · stride 344 · σ 128 · batch 64) is **byte-identical** to all 3 packages'
`deployment_config.yaml`. §14 runtime assertion exists (`predictor.py:assert_runtime_matches_package`,
called from `build_context` → both CLI and worker), covers precision/tta/compile/scales/T/thr, and is
tested. **FIXED doc drift:** the stale §14 "TODO(impl): no code enforces it" note and the ledger
recipe-table's "calibration still null" line.

### E4 — Deployment packages: PASS
All 3 local packages complete (6 files), calibration non-null and identical, channel binding
R,G,B,ndvi, weights load into the built EffB5 arch, tensors == run checkpoints (above). GCS copies at
`gs://rts-mapping-v2-usw1/inference/2025q3_south/packages/seed{42,43,44}/` **hash-identical** to local
(md5, every file). `weights.pth` SHA256: seed42 `e0ffdbff…9924`, seed43 `335aeedd…60fa`,
seed44 `fa4866cd…c738`.

### E5 — Docker image `rts-infer:v1`: WAS STALE → REBUILT
The registry image (built 2026-06-26) predated the multiscale merge: `inference/runner.py`,
`inference/tiles.py`, `data/dataset.py`, `data/splits.py`, `training/metrics.py` differed from HEAD;
`data/label_cleaning.py` and 12 newer scripts absent. Diff analysis: the inference-path deltas were
**purely multiscale additions** — at `scales: [1.0]` the baked deploy path was functionally identical
(worker/claim/predictor/writer identical), so no numerics risk existed; rebuilt anyway per policy from
the fixed branch (`rts.git_sha` label now baked), pushed, re-diffed clean → §E9 smoke runs inside it.
Entrypoint `python -u`, cv2 patch baked, zero runtime patches.

### E6 — Input data: PASS (S2 at 1,797/1,799 → sweep relaunched)

| Check | Result |
|---|---|
| Tile list | 41,567,572 rows exact; **0 duplicate tile_ids** (full scan); bbox = 2446.0 m everywhere sampled (n=41,568); stride-344 alignment exact at zoom-15 resolution (0 off-grid) |
| Quad index | 309,100 quads (+1 header — the launch plan's "309,101" counted the header); sample paths readable |
| S2 export | **1,797/1,799 cells present** (99.9% — the diary's "76%, ETA 07-07" was stale; the export completed early). Missing: `E0450_N0680`, `E1530_N0480` → relaunched (see fix). Banks Island cells all present |
| **FIXED (blocker):** S2 resume regex | `_EE_TILE_SUFFIX` required a leading dash EE shard names don't have → matched nothing → a resume sweep would have re-exported ~1,600 cells. Fixed pattern verified against all 9,652 objects (1,797 distinct cells, 0 non-conforming); corrected launcher then computed **2/1799 todo** and launched exactly 2 GEE tasks |
| s2_index | built from the export → `/mnt/outputs/inference/s2_index_2025_south.csv` (+ uploaded); coverage audit in §E9 |
| Residual-gap policy | per spec §3.3, NDVI gaps neutralize to 0 (channel mean) and only the RGB mask drives output NoData — so uncovered-S2 tiles degrade to RGB-effective, they are not dropped. With ≤2 cells outstanding this is a non-issue at launch; the per-shard manifest records nothing S2-specific (remediation: none needed — policy now stated here and in §3.3) |

### E7 — Orchestration & fleet scripts: PASS after one live-reproduced blocker fix

- `shard_tiles.py` — spatial sort reuses `_spatial_sort`; exactly-once partition (dry-run §E10).
- Claim TTL 1800 s vs heartbeat every progress tick (~512 tiles ≈ 30–60 s) — no premature reclaims.
- Worker ↔ bucket layout ↔ `inference_progress.py` share one contract (`shards/ claims/ done/
  probs/<shard>/ logs/`).
- Watchdog: `^rts-infer-[0-9]+$` regex enforced per-VM, `run_active` sentinel gate, 48 h backstop,
  stop-not-delete, never touches the master. Sound.
- **FIXED (blocker, reproduced live):** `inference_fleet_startup.sh` set `CUDA_VISIBLE_DEVICES=$g`
  alongside `--gpus device=$g`. The NVIDIA runtime renumbers the pinned GPU to 0 inside the
  container, so for g≥1 the env var pointed at a nonexistent index — reproduced on the master:
  `torch.cuda.is_available() == False`. **7 of 8 workers per VM would have crashed at fleet launch**
  (the exact pre-mortem-#4 silent-idle scenario). Fix: drop the env var; add `--worker-id
  host:gpuN` for monitor attribution.

### E8 — Infra / permissions / cost: PASS
- Live us-west1 quota: `NVIDIA_L4_GPUS 1/32` (phantom L4 persists → 31 schedulable → fleet default
  N=3 correct; 4th VM only after it clears), `CPUS 46/480` (not binding).
- Buckets: `rts-mapping-v2-usw1` US-WEST1 co-located with `pdg-planet-data` US-WEST1 ✓ (egress-free
  fleet reads); write+delete probe under our prefix OK; Planet bucket readable from the master.
  (The fleet VMs' own SA is validated live in §E10.)
- Cost model unchanged from the launch plan (≈$750–2,200/pass); bytes/tile measured in §E9/E10
  refresh the storage estimate.

### E9 — GPU smoke on the master (inside the rebuilt image, no repo mount): PASS
3,225-tile Banks Island AOI (cut from the production tile grid around a 24-positive training cluster),
run with the **baked image code only**, the **3 GCS packages**, real 2025 quads + real S2 NDVI:

| Check | Result |
|---|---|
| Run | 3,225/3,225 tiles, 0 skipped; **~13.0 tiles/s** on 1 A100 (3-model ensemble, cross-region reads, 8 DL workers) |
| **gs:// package loading** | **FAILED on first run** → blocker #3 found & fixed (see Fixes); re-run green |
| COG spec | Float32 · EPSG:3857 · NoData −1.0 · values ∈ [0,1] (0 violations in 129 sampled) · max prob 0.997 |
| Detection plausibility | max prob ≥0.65 within 500 m of **15/24** known RTS (62.5%); <0.30 for 8/24 (33%); median 0.81 — matches the Finding-K val typology (60.6% detected / 28% invisible) on 2025 imagery |
| **NDVI parity** (pre-mortem #3) | pixel corr **0.95–0.97** vs training EXTRA tiles at identical bounds; means/stds within ~0.02 across the 2024↔2025 year change; coverage 100% |
| §14 guard | deliberately tripped (runtime tta=minimal vs package none) → aborts with the right message |
| Offline load | package loads under `HF_HUB_OFFLINE=1` (post-fix — no HF hub dependency at startup) |
| §5.4 drift | trips vs **global** stats (RGB Δ≈1.4–1.5σ, NDVI Δ≈0.96σ) — but like-for-like (2024 Banks training tiles vs 2025 Banks reads) shows **no temporal drift**: RGB 117/98/78 → 108/92/73, NDVI 0.200 → 0.243 (≈0.3σ). The trip is single-region deviation from global stats — §5.4 needs a domain-representative sample, not one AOI (remediation #6) |
| **Output size** | Float32+deflate ≈ **570 KB/tile → ~23.7 TB full-run (~2× the plan's ~12 TB)**. Re-encoded as scaled-uint8 (prob×250, NoData 255): **~8 KB/tile → ~0.3 TB (71×)** with re-threshold precision 0.004. **Recommendation: switch the probability output to scaled-uint8 before launch** (needs a small `writer.py` dtype option + §9.1 spec update — user decision) |

### E10 — Live fleet pre-flight (`rts-infer-1`): the highest-value audit step — **7 fleet-path defects found live, all fixed**

Drill setup: a 200k-tile staging queue (10 × 20k spatially-contiguous shards from the production grid,
`gs://…/inference/staging_preflight/`), real quads + S2 NDVI + the 3 GCS packages.

**Defects found by actually booting a VM (none were reachable by static review or unit tests):**

| # | Defect | Fix (commit) |
|---|---|---|
| 1 | `CUDA_VISIBLE_DEVICES=$g` × `--gpus device=$g` → 7/8 workers see no GPU (reproduced on the master before any VM) | drop env; `--worker-id host:gpuN` (`8eccb1b`) |
| 2 | DLVM family `common-cu123` retired by Google → create fails | `common-cu129-ubuntu-2204-nvidia-580`, verified live (`ea1f3f2`) |
| 3 | `--metadata` parses commas → the comma-separated `packages=` value could never pass | `^\|^` alternate-delimiter (`ea1f3f2`) |
| 4 | new DLVM family ships driver + nvidia-ctk but **no docker** → startup exit 127 | startup installs docker.io + configures NVIDIA runtime when missing (`ea1f3f2`) |
| 5 | default-SA VMs get legacy `devstorage.read_only` scopes → all GCS writes 403 even with IAM; SA also had **no** IAM on our bucket | always `--scopes cloud-platform`; granted `roles/storage.objectAdmin` on `gs://rts-mapping-v2-usw1` to the compute SA (`38a46df`) |
| 6 | startup not reboot-idempotent (`docker run --name` vs previous boot's exited container) | `docker rm -f` before run (`38a46df`) |
| 7 | fleet SA 403 on `gs://pdg-planet-data` quad reads (master had only ever read it via user ADC) | **IAM change on a shared-project bucket (review):** granted `roles/storage.objectViewer` to the project's own default compute SA — read-only, reversible, the designed PDG workflow |
| 8 | **no `--shm-size`** on worker containers → DataLoader `/dev/shm` fills → every worker crashes `No space left on device` (the burst-then-stall pattern seen all through the drill) | `--shm-size 16g` (`1212431`) |
| 9 | claim heartbeat was progress-tick-only → an **active** worker's shard got reclaimed before its first 512-tile tick | wall-clock heartbeat thread every 240 s (`9b07b0e`, unit-tested) |

Also live: **us-west1 g2-standard-96 STOCKOUT in all 3 zones** (pre-mortem #5) — `g2-standard-48`
(4×L4) available in us-west1-a and used for the drill. **Production fleet sizing must plan for
6–8 × g2-standard-48 as the fallback shape** (same 24–32 L4 total).

**Queue mechanics verified live:** 4 L4 + 2 A100 workers on one queue → 6 claims, 6 distinct owners
(`rts-infer-1:gpuN` / `a100-master:gpuN`), zero collisions; probability COGs flowing to the
shard-scoped prefixes. Kill drill: killed the worker holding `shard_000005` mid-shard → claim
orphaned, reclaimable post-TTL; an aggressive-TTL (120 s) reclaim worker **stole an *active*
worker's shard** whose first heartbeat hadn't fired yet (heartbeats were progress-tick-only, first
tick = 512 tiles) → **fixed with a wall-clock heartbeat thread every 240 s**
(`work_loop(heartbeat_every_s=…)`, unit-tested, commit `9b07b0e`; final image digest `7772dbc7…`).
Duplicate compute remains waste-only (idempotent writes, done-markers single-writer).

**Defect #8 — the burst-then-stall root cause (blocker):** the startup `docker run` had **no
`--shm-size`**. With `--num-workers 8` the DataLoader shares tensors via `/dev/shm`; docker's 64 MB
default filled within a shard and every worker died with `No space left on device` (torch shm write).
This produced exactly the intermittent partial-output pattern seen through the drill. Fixed (`1212431`,
`--shm-size 16g` matching the master). After the fix all 4 L4 workers ran stably at **GPU 100%**.

**Benchmark (post-shm-fix, 4× L4 on g2-standard-48, co-located us-west1, 3-model ensemble, bf16):**

| Quantity | Measured |
|---|---|
| Per-L4-worker throughput | **~4.2 tiles/s** steady-state (instantaneous 512→3072); **GPU-bound at 100%** — the 3-model ensemble is 3 forward passes/tile, not I/O-bound as the plan assumed |
| Compressed bytes/tile | ~570 KB Float32 (§E9) → **scaled-uint8 ~8 KB recommended** |
| **Full-run ETA (41.57M tiles)** | **This is the headline benchmark correction.** 24 L4 → **~4.8 d**; 28 L4 (stockout fallback shape) → **~4.1 d**; 28 L4 **+ 8 A100** (~13 t/s each, cross-region) → **~2.2 d**. The plan's 12–29 h assumed 10–25 t/s/worker; the real ensemble rate is ~4 t/s/L4. **Launch planning must budget ~2–5 days of fleet spend, and should include the A100 master as a worker** (halves wallclock) |

**Reconciliation:** drill was a capability/throughput validation, not a full drain (staging queue torn
down after the benchmark). Exactly-once is proven by the unit tests + the live collision-free multi-VM
claims; the full-run end-of-run reconciliation (`done` count == n_shards, tiles == 41,567,572) runs at
Phase-4 launch. **`rts-infer-1` stopped (TERMINATED, not deleted); staging prefix cleaned; no rogue fleet VMs.**

---

## Fixes applied (branch `audit-prelaunch`, 8 commits)

| # | Severity | Fix | Commit | Re-verified by |
|---|---|---|---|---|
| 1 | **Blocker** | fleet-startup GPU visibility bug (CVD × `--gpus`) | `8eccb1b` | live repro; §E10 1-VM drill |
| 2 | **Blocker** | S2 launcher resume regex (would re-export ~1,600 cells) | `8eccb1b` | corrected launcher → 2/1799 todo |
| 3 | Medium | predictor stale `scales!=[1.0]` gate (dead multiscale path) | `8eccb1b` | inference tests green |
| 4 | Low | doc drift: §14 TODO note; ledger "calibration null" | `8eccb1b` | — |
| 5 | **Blocker** | gs:// deployment packages unloadable (`Config not found: gs:/…`) | `90248b7` | §E9 smoke + regression test |
| 6 | Medium | package models built `pretrained=True` → HF-hub pull at every worker startup | `d1393dd` | loads under `HF_HUB_OFFLINE=1` |
| 7 | **Blocker** | DLVM family retired · `--metadata` comma parse · no docker on image | `ea1f3f2` | §E10 VM boots + workers claim |
| 8 | **Blocker** | default-SA legacy read-only scopes + missing bucket IAM (all writes 403) | `38a46df` + IAM | live writes succeed post-fix |
| 9 | Low | startup not reboot-idempotent (`docker run --name` clash) | `38a46df` | clean VM reset |
| 10 | Medium | claim heartbeat starvation (active worker's shard reclaimed) | `9b07b0e` | unit test + drill |
| 11 | **Blocker** | no `--shm-size` → DataLoader crashes every worker | `1212431` | 4 workers stable at GPU 100% |
| — | — | fleet SA read on `gs://pdg-planet-data` (IAM, read-only, shared bucket) | IAM | live quad reads succeed |
| — | — | `rts-infer:v1` rebuilt from the fixed branch + pushed (digest `7772dbc7…`) | — | re-diff clean; §E9/E10 run on it |

## Remediation list (non-blocking, not fixed here)

1. `sync_experiments.py` only harvests `/mnt/outputs/v1.0/runs` — the `v1_1_*` ledger rows are
   never machine-checked (they verified manually). Add multi-root harvest or symlink the run dirs.
2. `data/transforms.py:78` GaussNoise `var_limit` silently ignored (albumentations API drift) —
   revisit at the next retrain; the deployed models trained with whatever the library default was,
   so this documents reality rather than changing it.
3. Claim-heartbeat read-then-write race (§E2) — accepted; end-of-run reconciliation covers it.
4. Test-set touch discipline (§S3.5) — keep future test consultations to frozen-cache re-scores.
5. Output GeoTIFFs lack overviews — add at the merged-raster stage, not per-tile.
6. §5.4 drift-check procedure: compare a **domain-representative** 2025 sample (or per-region vs
   per-region), not a single AOI vs global stats — a single extreme region trips the gate spuriously
   (seen at Banks Island, §E9).
7. `work_loop` **exits** when every remaining shard is claimed-but-stale-ineligible (e.g. after a crash
   storm all claims are <TTL old) — with `--restart=on-failure` not firing on exit-0, a fleet can go
   idle ~TTL minutes until manually restarted (seen in the drill). Consider an opt-in
   `--wait-when-empty` sleep-retry loop; the monitor's per-VM tiles/s alarm covers detection meanwhile.
8. Test isolation: `tests/test_run_inference_worker.py` imports fail when run standalone
   (`scripts/` on sys.path makes `scripts/inference.py` shadow the `inference` package; fine in full-suite
   order). Rename the script or de-shadow imports.

## Cleanup / archive candidates — **awaiting user confirmation, nothing moved**

Repo root: `normalization_stats.json` (stale 2026-05-27 RGB-only stats, 3,638 tiles — nothing
references it; deploy uses `v1_splits/normalization_stats_ndvi.json`).

Closed-campaign one-off scripts (→ `docs/archive/` or a tagged removal commit):
`diagnose_outline_quality.py` (outline refinement killed at Phase 0) · `channel_correlation.py`,
`channel_correlation_8band.py`, `se_variants.py`, `build_se_artifacts.py`, `plot_extra_channels.py`
(family D closed) · `audit_v1_1.py`, `build_v1_1_dataset.py` (family N closed) ·
`qc_scale05_contact_sheet.py`, `scale05_tile_creation.py`, `evaluate_multiscale_poc.py` (family M
closed; POC verdict recorded) · `seed_recall_noise.py`, `score_insample_train.py` (K diagnostics
done) · `validate_v21_positives.py`, `add_ARTS_uids.py`, `validate_new_ARTS_uids.py`,
`apply_region_hotfix.py`, `regenerate_labels_no_ignore.py` (data-prep one-offs) ·
`run_ablation_queue.sh`, `negative_tile_run.sh`, `positive_tile_run.sh` (training-phase shells).
KEEP (pending campaigns): `make_invisible_contact_sheet.py` (D1 label audit), `probe_change_signal.py`
(D2, awaiting 2023 imagery), `inference_feasibility.py` (§6.4 gate, deferred).

Stale plan files in `~/.claude/plans/` (all superseded except `elegant-exploring-lemur.md` and this
audit's plan): 11 files. `docs/superpowers/plans/2026-05-02-pre-smoke-fixes.md` (May-era).

## Track B (M1.1) swap-cost — for the record (decision already taken: keep M1.0)

Shipping M1.1 instead would require: its own calibration (T + thr ≈ 0.45 on Val-Realistic) →
repackage 3 seeds → re-upload → re-smoke (§E9). Roughly a day of work, no retrain needed
(checkpoints exist). Everything model-independent in this audit (E5–E8, E10) carries over. The
ledger's no-improvement verdict means there is no ability upside; the case for a swap would be purely the
cleaner labels — revisit only with the next real modeling change.

## `inference.md §13` pre-inference checklist reconciliation

| §13 item | Status |
|---|---|
| Model artifacts on GCS | ✅ E4 (hash-verified, 3 packages) |
| Docker image built + pushed | ✅ E5 (rebuilt from fixed branch; digest `7772dbc7…`) |
| Tile grid generated + validated | ✅ E6 (41,567,572 rows, unique, stride-exact) |
| GCS permissions | ✅ E8 (master) + E10 (fleet SA scopes + bucket IAM fixed live) |
| Test inference on small region | ✅ E9 (Banks AOI, real NDVI, detections + COGs sane) |
| Throughput estimate vs budget | ✅ E10 — **corrected to ~4.2 t/s/L4 → ~2–5 d full run** (not 12–29 h) |

## What still gates the launch (all standard, none new-blocking)

1. **Last 2 S2 cells** `E0450_N0680`, `E1530_N0480` — GEE tasks PENDING (queued, not failed). When they
   land, re-run `build_s2_index.py` + re-upload `s2_index_2025_south.csv` (the ~8,500 tiles over
   `E0450_N0680` otherwise run RGB-effective per the §3.3 gap policy — acceptable, not a blocker).
2. **Explicit go + spend at the corrected ETA** (~2–5 days, ~$1–3k) — the benchmark moved this materially
   vs the plan, so it warrants a fresh sign-off.
3. **Fleet sizing decision** given the g2-standard-96 stockout: 6–8× g2-standard-48, and whether to add
   the 8 A100 master workers (halves wallclock; cross-region egress is minor).
4. **Output dtype decision**: scaled-uint8 (~0.3 TB) vs Float32 (~24 TB) — a small `writer.py` change if yes.
5. Cleanup/archive list above — **awaiting user confirmation before anything moves.**
