# Optimization Roadmap & Experiment-Fairness Audit

Living doc. Created 2026-06-24. Covers improvement/optimization opportunities across **every**
aspect of the pipeline (training efficiency, resource utilization, model accuracy, inference
efficiency/accuracy, infrastructure), an evidence assessment of the "data-limited" diagnosis, and a
fairness/validity audit of the experiments and decisions to date.

Scope note: this is an *analysis & prioritization* doc — **no code/config changes are implied by its
existence**. Each lever below carries an explicit pre-commit verification; implementing any of them
(especially A1.1) is a separate, approved follow-up. Decisions already settled are in the
[Appendix](#appendix--off-the-table-already-tried-or-decided) so nothing here re-opens them.

Method: 3 parallel codebase sweeps + manual verification of every load-bearing claim against source
(`inference/tiles.py`, the `configs/fm_dinov3*.yaml` foundation configs, `scripts/train.py` LLRD path,
and the `docs/experiment_ledger.md` split labels).

Impact/effort ratings are rough (high/med/low). "Co-priority": inference efficiency and
training/accuracy are weighted equally per the current phase.

---

## Part A — Optimization opportunities by aspect

### A1. Inference efficiency & accuracy — *deployment-critical*

The pan-arctic run is **41.57M tiles / 309,100 quads** (`inference/inference.md` §3.2) and the
pipeline is **GCS-read-bound, not GPU-bound** (§11.3: workers 4→8 nearly doubled throughput, larger
batches did not). This is where the largest uncaptured wins are.

| ID | Opportunity | Impact / effort | Current state |
|----|-------------|-----------------|---------------|
| **A1.1** | **Per-worker quad LRU cache** | **high / med** | `read_tile` (`inference/tiles.py:102-103`) calls `rasterio.open()` for every intersecting quad on **every** tile — stateless. At stride 344 (~33% overlap) each quad is re-opened ~36×. §11.3 estimates **10–30×** from quad caching (toward the original ~150 t/s ⇒ a full pass ≈14 GPU-h vs the measured ~137 GPU-h @ 10.5 t/s). |
| **A1.2** | **Spatial hit-test** | med / low | `inference/tiles.py:94` filters the 309k-row quad index with a pandas boolean scan **per tile** → ~1.3e13 comparisons across 41.57M tiles, on dataloader-worker CPU. Replace with an STRtree / grid-bucket lookup. Complements A1.1. |
| **A1.3** | **NDVI at inference (readiness gap)** | high / med | `InferenceTileDataset` (`inference/tiles.py:97,157-176`) emits **RGB only (3 bands)**, but the locked v2 model is **RGB+NDVI**. The on-the-fly-NDVI-from-bulk-S2-composites design (launched 2026-06-24) is not yet wired into the inference path. **Required before deployment.** Reuse the A1.1 cache for the S2 composites too. |
| **A1.4** | **Benchmark A1.1 on a subregion (P0)** | — / low | The 15–43 t/s co-located estimate (`inference.md` §2.1) is speculative. Measure cache uplift + hit-rate + worker memory on a ~1–5k-tile AOI, then re-estimate wallclock/cost before the full run. |
| **A1.5** | **Douglas–Peucker vector simplification** | med / low | `post-inference/post-inference.md` §6 specifies ~1px DP simplify; `scripts/vectorize_predictions.py` doesn't implement it → 20–50% smaller vectors with no accuracy loss if tuned. |

**Already optimal / off the table here:** batch & worker sweep (8 workers won, bigger batches hurt,
`inference.md` §11.3), bf16, inference-time `torch.compile` (gated on a calibration re-run; numerics
shift), multi-scale (zero-shot transfer failed; the training POC — ledger family M — passed gates 1+2 but failed gate 3 fusion-recall, so deploy stays `scales:[1.0]`). **Pending milestones (not optimizations):**
TTA decision + temperature/threshold calibration on Val-Realistic.

### A2. Training efficiency & resource utilization

`base_v2_fast` already bought ~2× throughput via the stop-schedule audit (median 40 wasted epochs
removed). Remaining levers are smaller and **measure-first**.

| ID | Opportunity | Impact / effort | Current state |
|----|-------------|-----------------|---------------|
| **A2.1** | **`torch.compile` on the training forward** | high-if-stable / med | Not used in training (`scripts/train.py`); 10–20% typical for UNet++/EffB5. Must validate with EMA + bf16 + curriculum before adopting. |
| **A2.2** | **Profile augmentation worker time** | med / med | elastic (p=0.3) + CLAHE (p=0.2) in `data/transforms.py` are CPU-heavy. If they dominate worker time, lower p or swap to cheaper ops. Prerequisite: per-tile timing — data-driven decision, not a blind change. |
| **A2.3** | **Dataloader latency check** | low / low | Confirm `prefetch_factor=2` / `num_workers=8` aren't starving the GPU; tune from a measured batch-latency number. |

**Off / already correct:** single-GPU + fleet-parallel ablations (DDP would invalidate per-run LR
calibration and isn't needed at bs≤32 on A100-80GB); EMA / sampler / metrics are already cheap;
gradient checkpointing & accumulation only if batch-size pressure appears (it hasn't); bf16/fp16
auto-selection (`_configure_precision`, `scripts/train.py:139`) is correct.

### A3. Model accuracy

Largely tapped — see [Part B](#part-b--is-accuracy-really-data-limited). The recommendation is to
**not open new architecture/loss/aug ablations** and instead:

- **A3.1 — Finish in-flight screens** (DINOv3-sat ±NDVI, ViT-7B frozen probe, SAM2/Hiera,
  RandAugment/TrivialAugment) and **seed-confirm** any that clear (3-seed mean Δ≥G *and*
  sign-consistency). Already planned.
- **A3.2 — Calibration milestone** (temperature + threshold + D4-TTA decision on Val-Realistic).
  **Blocking** for a valid deployment; must use identical precision/TTA/scales as deployment
  (the `assert_runtime_matches_package` guard already enforces this at run time).
- **A3.3 — Ensemble decision at final lock** (F4 averaging of top seeds/configs vs the ×k cost).
- **A3.4 — The real accuracy headroom is DATA, not modeling.** Hard-negative mining post-inference and
  the deferred train re-stage (+28 pos / −49 black / 564 degraded). MAE pretraining on unlabeled
  Arctic quads is the one idle-GPU lever that directly attacks the diagnosed regime (user-gated).

### A4. Infrastructure & ops

| ID | Opportunity | Impact / effort | Current state |
|----|-------------|-----------------|---------------|
| **A4.1** | **Rebuild `rts-train` image with `earthengine-api` + pinned `requirements.txt`** | med / low | Currently runtime-installed (flagged TODO, `computing/infrastructure.md` §6) → fragile + slow start. |
| **A4.2** | **MLflow concurrency-safe backend** | low now / med later | Local per-VM file store (`/mnt/outputs/mlflow`) is not concurrency-safe. Fine for fleet-parallel single runs; a shared SQLite/Postgres + GCS-artifact backend is needed before any multi-node/DDP or merged live tracking. |
| **A4.3** | **Spot/preemptible for bulk inference** | — | Already decided (resumable via `inference_log.json`); keep. Budget is **not** a binding constraint ($70k credit, expires Sep 2026). |

---

## Part B — Is accuracy really data-limited?

The Phase-2 scaling curve is the **weakest** leg of the argument (it's stale & leaky — see
[C2](#part-c--experiment-fairness--validity-audit)). The diagnosis nonetheless holds, on
**convergent, independent corrected-split evidence**: when several orthogonal levers *all* fail to
move the metric, the bottleneck is information/data, not the model.

| Lever pulled | Result | Source | Implication |
|---|---|---|---|
| Capacity **down** (EffB3) | no-win 0.9050 vs 0.9123 | ledger / status 2026-06-21 | not under-capacity |
| Capacity **up** (EffB7) | dropped — overfit risk on plateau | `experiment_ledger.md` | not under-capacity |
| More **features** (full 8-band) | loses to NDVI-only (0.869 vs 0.8985) | ledger Phase 4 | input signal saturated |
| Better **representation** (DINOv3 web) | RGB edge 0.873 > 0.830 **vanishes** w/ NDVI (0.9120 ≈ 0.9123) | ledger row 73 | not representation-limited once the cheap feature is in |
| **Augmentation** (mixing ×4) | copy-paste / mosaic / cutmix / mixup all no-win | `experiment_ledger.md` | can't manufacture more signal |
| Train–val gap | 0.05–0.17 (< 0.4) | status 2026-06-15 | well-fit, not overfitting |
| Positive count | ~1,718 pos / 22,259 | status (D1.0 snapshot) | rare-object / data-scarce regime |
| Seed noise vs effect sizes | σ_corrected ≈ 0.02 ≳ most remaining Δ | status 2026-06-20 | remaining signal is below the noise floor |

**Honest caveat:** the *quantitative* plateau (slope −0.12, gap 0.05/0.17) was computed 2026-06-15,
**before** the 2026-06-16 leak fix, and the scaling re-run is marked "redo (paused)"
(`docs/v1.0_rebaseline.md:78`); an earlier conflicting slope of ≈4.4 is flagged in
`docs/v21_staleness_audit.md:31`. The qualitative regime is robust; the number is not.
**Cheap re-confirm:** recompute the train–val gap on the corrected-split v2 baseline — read it off
existing MLflow runs, no new training.

---

## Part C — Experiment fairness & validity audit

### Strengths (verified — keep doing these)
- **Foundation/ViT comparisons are fair.** The `configs/fm_dinov3*.yaml` runs use layer-wise LR decay
  (`llrd_decay: 0.75`, implemented at `scripts/train.py:742-747` via
  `training/freeze.build_llrd_param_groups`), a lower `base_lr 2e-4`, full-LR backbone with a 20-epoch
  freeze, `early_stopping.start_epoch 70`, and they inherit `phase0c_seed42` (max_epochs 300 — so they
  are **not** truncated by base_v2_fast's aggressive stop). There is also a web-DINOv3
  native-ImageNet-norm **control** to de-confound web-vs-satellite. This is careful design.
- **Conservative lock rule.** Every lock requires a 3-seed mean Δ≥G **and** sign-consistency across
  seeds (rejected curriculum r20_pf33 on a sign flip; locked drop-RandomScale 3/3).
- **Test-Realistic is touched once, at the very end** — avoids adaptive test-set overfitting.
- **Gate metric** = PR-AUC geomean over [5, 10, 20] ratios — chosen to avoid recall-happy bias.

### Validity gaps (real — recommended *cheap re-confirms*, not re-running everything)
- **C1 — Gate calibrated on the leaky split.** μ₀/σ₀ and G=0.0112 come from the leaky baseline;
  corrected-split σ ≈ 0.02 (≈4×). Locks are protected (3-seed + sign), but the **single-seed screen**
  near G is unreliable, so secondary single-seed rankings (round-1 EXTRA additions, some aug arms) sit
  within noise. **Recommend:** widen the screen band to ~2σ_corrected (≈0.04), or require a
  seed-confirm before treating any single-seed delta as real. (NDVI and drop-RandomScale survive
  either way.)
- **C2 — The data-plateau scaling number is stale & leaky** (see Part B caveat). **Recommend:**
  re-read the corrected-split train–val gap (free); only re-run scale_25/50/75 if a gate decision
  hinges on it.
- **C3 — Load-bearing leaky-split locks were not re-litigated.** focal·ignore_w2 (boundary width) and
  "UNet++ beats FPN/DeepLabV3+/PSPNet/MAnet" were decided on the leaky split and carried into the
  corrected baseline without re-test. Low risk (robust levers) but a genuine gap. **Recommend:** one
  corrected-split A/B of ignore_w ∈ {none, 2, 3} on the locked recipe to confirm the boundary win
  persists.
- **C4 — `base_v2_fast` "gate-neutral" is lightly validated.** The 48-run retrospective (runs peak
  ~ep52) is good population evidence, but the truncation (max_epochs 300→120, start_epoch 101→45,
  patience 8→5) was confirmed gate-neutral on a *single* NDVI fastcheck. Low risk for the EffB5 family;
  foundation runs correctly opt out. **Recommend:** spot-check one more locked-recipe seed's best-epoch
  < 120.
- **C5 — Pervasive single-seed screening** (acknowledged). Fine as a *screen* given abundant compute;
  just don't let single-seed deltas drive narrative claims without the C1 band.

---

## Prioritized shortlist

1. **A1.1 quad cache + A1.3 NDVI-at-inference + A1.4 benchmark** — deployment-critical; the project's
   single biggest uncaptured efficiency win.
2. **A3.2 calibration** + finishing **A3.1** screens — blocking for a valid deployment.
3. **C1 / C3 cheap re-confirms** — protect the locked decisions' validity at low cost.
4. **A2.1 torch.compile** + **A2.2 augmentation profiling** — training throughput; measure-first.
5. **A4.1 Docker image rebuild** — removes a known operational fragility.

---

## Appendix — off the table (already tried or decided)

Do **not** re-propose these; they're settled (see `docs/experiment_ledger.md`):
focal + ignore_w2 · UNet++/EfficientNet-B5 · NDVI-only + F0 channel-stack (heavy fusion F2/F3/F5 all
lose) · drop-RandomScale · mixing augs (4/4 no-win) · curriculum sweep (within noise) · capacity
up/down · weight-decay grid · SegFormer / EffB7 · soft-label boundary · pseudo-labeling · multi-scale
zero-shot · per-dataset z-score normalization (Arm A) · DDP (would invalidate LR calibration) ·
inference batch/worker sweep (8 workers won).

---

## Verification (how each lever is validated before commitment)

- **A1.1 / A1.2** — subregion benchmark (cache on/off): throughput, hit-rate, worker memory; adopt only
  on a measured ≥2× gain.
- **A1.3** — inference smoke producing a 4-channel tensor whose NDVI matches the training NDVI stats
  (CLAUDE.md Rule 3 train/inference parity).
- **A2.1 / A2.2** — timed train-smoke (epoch wall-clock, per-aug timing) before/after; adopt only on a
  measured win.
- **C1 / C3 / C4** — read existing MLflow runs (gap, best-epoch) + at most one small corrected-split
  A/B; no full re-runs.
- **This doc** — cross-checked against `docs/experiment_ledger.md` so nothing in the Appendix is
  re-proposed.
