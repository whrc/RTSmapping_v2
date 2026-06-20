# Experiment Ledger

Master chronological registry of **every** training run in RTSmapping_v2, with status and score.
Living doc — update when runs finish, launch, or change state. SSoT for "what has been tried."

**Metric:** `val_realistic_pr_auc_geomean` (= `best_smoothed` in each `run_summary.md`). Higher is better.
Source of truth: `/mnt/outputs/v1.0/runs/<name>/run_summary.md` (finished) and `.../logs/<name>.log` (live).

**Status legend:** ✅ done · 🔵 ongoing (live best so far) · ⏳ queued · ⏸️ deferred · ❌ dropped/crashed.

> ⚠️ **Scores are NOT comparable across the split boundary.** Phases 0/2/3/5 ran on the **leaky**
> region split (relative comparisons only, by design). Phases 4/10 run on the **corrected leakage-free**
> split — their absolute numbers sit higher and must be compared *within phase* against the in-phase
> control (Phase-4 RGB control = 0.830), not against earlier phases. Final test is scored once, honestly,
> on the corrected split (Step 5).

_Last refreshed: 2026-06-20 09:00._

---

## Master table (chronological by launch)

| # | Started | Experiment | Phase / purpose | Split | Score | Status |
|---|---------|-----------|-----------------|-------|------:|:------:|
| 1 | 06-13 08:00 | phase0b_lr_frozen | 0b LR probe (frozen) | leaky | 0.0008 | ❌ degenerate |
| 2 | 06-13 11:12 | phase0b_lr_unfrozen | 0b LR probe (unfrozen) | leaky | — | ❌ crashed |
| 3 | 06-13 23:37 | phase0a_arm_a | 0a norm arm A (z-score) | leaky | 0.667 | ✅ |
| 4 | 06-14 01:46 | phase0a_arm_b | 0a norm arm B | leaky | 0.626 | ✅ |
| 5 | 06-14 02:40 | phase0a_arm_c | 0a norm arm C | leaky | 0.670 | ✅ |
| 6 | 06-14 04:09 | phase0c_seed42 | 0c baseline seed 42 | leaky | 0.790 | ✅ |
| 7 | 06-14 04:16 | phase0c_seed43 | 0c baseline seed 43 | leaky | 0.786 | ✅ |
| 8 | 06-14 05:35 | phase0c_seed44 | 0c baseline seed 44 | leaky | 0.797 | ✅ |
| 9 | 06-15 01:22 | phase2_scale_25 | 2 data-scale 25% | leaky | 0.764 | ✅ |
| 10 | 06-15 04:27 | phase2_scale_50 | 2 data-scale 50% | leaky | 0.772 | ✅ |
| 11 | 06-15 09:35 | phase2_scale_75 | 2 data-scale 75% | leaky | 0.792 | ✅ |
| 12 | 06-15 14:11 | phase3_loss_compound_2to1 | 3 loss sweep | leaky | 0.793 | ✅ |
| 13 | 06-15 14:11 | phase3_loss_tversky_3to7 | 3 loss sweep | leaky | 0.590 | ✅ |
| 14 | 06-15 14:20 | phase3_loss_tversky_2to8 | 3 loss sweep | leaky | 0.073 | ❌ collapsed |
| 15 | 06-15 14:26 | phase3_loss_compound_1to2 | 3 loss sweep | leaky | 0.800 | ✅ |
| 16 | 06-15 15:10 | phase3_loss_compound_1to1 | 3 loss sweep | leaky | 0.788 | ✅ |
| 17 | 06-16 06:33 | phase3_bd_focal_ignore_w1 | 3 boundary factorial | leaky | 0.787 | ✅ |
| 18 | 06-16 06:56 | phase3_loss_compound_1to2_seed43 | 3 loss seed-confirm | leaky | 0.787 | ✅ |
| 19 | 06-16 06:57 | **phase3_bd_focal_ignore_w2** | 3 boundary factorial | leaky | **0.805** | ✅ **winner** |
| 20 | 06-16 07:15 | phase3_bd_compound_1to2_ignore_w2 | 3 boundary factorial | leaky | 0.800 | ✅ |
| 21 | 06-16 08:00 | phase3_bd_focal_ignore_w3 | 3 boundary factorial | leaky | 0.797 | ✅ |
| 22 | 06-16 09:10 | phase3_bd_compound_1to2_ignore_w1 | 3 boundary factorial | leaky | 0.800 | ✅ |
| 23 | 06-16 14:37 | phase3_bd_compound_1to2_ignore_w3 | 3 boundary factorial | leaky | 0.802 | ✅ |
| 24 | 06-17 00:06 | phase5_arch_fpn | 5 arch sweep | leaky | 0.794 | ✅ |
| 25 | 06-17 01:52 | phase5_arch_manet | 5 arch sweep | leaky | 0.621 | ✅ |
| 26 | 06-17 02:05 | phase3_bd_focal_ignore_w2_seed44 | 3 winner seed-confirm | leaky | 0.805 | ✅ |
| 27 | 06-17 02:11 | phase3_bd_focal_ignore_w2_seed43 | 3 winner seed-confirm | leaky | 0.820 | ✅ |
| 28 | 06-17 04:25 | phase3_bd_compound_1to2_ignore_w3_seed43 | 3 runner-up confirm | leaky | 0.817 | ✅ |
| 29 | 06-17 10:13 | phase3_bd_compound_1to2_ignore_w3_seed44 | 3 runner-up confirm | leaky | 0.815 | ✅ |
| 30 | 06-15 16:13 | phase5_arch_deeplabv3plus | 5 arch sweep | leaky | 0.788 | 🔵 early-stop tail |
| 31 | 06-16 11:27 | phase5_arch_pspnet | 5 arch sweep | leaky | 0.729 | 🔵 early-stop tail |
| 32 | — | phase5_arch_segformer | 5 arch sweep | leaky | — | ❌ dropped (no run) |
| 33 | 06-17 11:48 | phase4_extra_rgb_baseline | 4 EXTRA control | corrected | 0.830 | 🔵 early-stop tail |
| 34 | 06-17 12:50 | **phase4_extra_full** (8-band) | 4 EXTRA | corrected | **0.876** | 🔵 early-stop tail |
| 35 | 06-18 06:39 | **phase4_extra_ndvi** | 4 EXTRA | corrected | **0.888** | ✅ **best single** |
| 36 | 06-18 06:48 | phase4_extra_tc (tasseled-cap) | 4 EXTRA | corrected | 0.868 | 🔵 early-stop tail |
| 37 | 06-18 07:49 | phase4_extra_nbr | 4 EXTRA | corrected | 0.847 | ✅ |
| 38 | 06-18 08:35 | phase4_extra_se_proto | 4 EXTRA | corrected | 0.847 | ✅ |
| 39 | 06-18 09:05 | phase4_extra_se_pca | 4 EXTRA | corrected | 0.874 | ✅ |
| 40 | 06-18 13:52 | phase10_curric_r30_pf50 | 10 curriculum (Step 2a) | corrected | 0.853 | ✅ |
| 41 | 06-18 13:52 | **phase10_curric_r20_pf33** | 10 curriculum (Step 2a) | corrected | **0.894** | ✅ best cell (s42) |
| 42 | 06-18 15:23 | phase10_curric_base | 10 curriculum control | corrected | 0.879 | ✅ |
| 43 | 06-18 | phase10_curric_r30_pf33 | 10 curriculum (Step 2a) | corrected | 0.853 | ✅ |
| 44 | 06-19 | phase10_curric_r20_pf33_seed43 | 10 curric seed-confirm | corrected | 0.901 | ✅ |
| 45 | 06-19 | phase10_curric_r20_pf33_seed44 | 10 curric seed-confirm | corrected | **0.859** | ✅ ⚠ low (variance) |
| 46 | 06-19 | phase4_extra_se_pca_seed43 | 4 EXTRA seed-confirm | corrected | 0.857 | ✅ (s42 was 0.874) |
| 47 | 06-19 | phase4_extra_ndvi_seed43 | 4 EXTRA seed-confirm | corrected | ~0.90 | 🔵 live (raw peak) |
| 48 | 06-19 | phase4_extra_ndvi_seed44 | 4 EXTRA seed-confirm | corrected | ~0.92 | 🔵 live (raw peak) |
| 49 | 06-19 | phase4_extra_full_seed43 | 4 EXTRA seed-confirm | corrected | ~0.884 | 🔵 live (raw peak) |
| 50 | 06-19 | phase4_extra_full_seed44 | 4 EXTRA seed-confirm | corrected | ~0.864 | 🔵 live (raw peak) |
| 51 | 06-19 | phase4_extra_ndvi_fastcheck | 0.1 stop-fix validation | corrected | ~0.898 | 🔵 gate-neutral check (≈ s42 0.888) |
| 52 | 06-20 | aug_ref | 3A aug control | corrected | — | 🔵 early ep |
| 53 | 06-20 | aug_p0_geom_only | 3A photometric-off | corrected | — | 🔵 early ep |
| 54 | 06-20 | aug_p1_no_clahe | 3A drop CLAHE | corrected | — | 🔵 early ep |
| 55 | 06-20 | aug_scale_off | 3B RandomScale off | corrected | — | ⏳ queued |
| 56 | 06-20 | aug_p3_photo_x15 | 3A photometric ×1.5 | corrected | — | ⏳ queued |
| 57 | 06-20 | aug_pad_ignore | 3B pad-ignore fix (fill_mask) | corrected | — | ⏳ queued |

---

## Queued (configs ready, awaiting a free GPU)

| Experiment | Step | Notes |
|-----------|------|-------|
| aug_scale_off / aug_p3_photo_x15 | 3A/B | tail of the aug-study wave (auto-dispatch as GPUs free) |
| aug_pad_ignore | 3B | PadIfNeeded `fill_mask=255` A/B (vs aug_ref); auto-dispatch follow-on |

**Code-pending (not yet config-launchable):** Stage-3A mixing augs (copy-paste / mosaic / cutmix /
mixup, curated RandAug/TrivialAug, annealing); Stage-0.2 bootstrap 1:50/1:100 high-ratio metric;
Stage-0.3 v1.0 re-stage (+28 pos / −49 black); heavy fusion **F3** (dual-encoder late) + **F5**
(residual cross-modal attention) classes. Step-3 channel×fusion selection follows the fusion verdict.

---

## Campaign stages & deferred (planned, not yet running)

Second-wave campaign plan: `.claude/plans/elegant-exploring-lemur.md`; roadmap in `current_working_status.md`.

| Item | Stage / decision | Notes |
|------|------------------|-------|
| Heavy fusion **F3** (dual-encoder late) + **F5** (residual cross-modal attn, JSTARS) | **Stage 1** — to build | F0–F2 already run; pick F\* across all, then channel selection under F\* |
| Foundation **DINOv3** (LLRD/LP-FT) → +best-EXTRA | **Stage 2** | DINOv3 RGB running; +EXTRA if it beats EffB5 |
| Foundation **SAM2** · **EffB3** cheap probe | **Stage 2** | SAM3 image-incompatible (py3.12/torch2.7); EffB3 = capacity-DOWN probe on a plateau |
| **Augmentation study** — copy-paste / mosaic / cutmix / mixup / RandAug / TrivialAug / annealing | **Stage 3A** — config arms running; mixing-aug code pending | domain fact: PlanetScope basemap RGB is CV-optimized, not surface reflectance → full toolbox; exclude shadow-cue scramblers; precision-guard = shadow safety net |
| **Multi-scale** — RandomScale A/B + pad-ignore (running); D4-TTA; scale-TTA | **Stage 3B** — tested, not assumed | scale-TTA gated on a scale-transfer test; context-expansion deferred to post-inference |
| **Calibration** (temp+threshold) · **ensemble** · **3-seed final lock → Test-Realistic once → ship** | **Stage 3C** | ensemble decided at final-lock (F4 + top-k vs ×k cost) |
| **Bootstrap 1:50/1:100 high-ratio eval** | **Stage 0.2** | secondary deployment-aligned readout; primary gate stays [5,10,20] |
| **v1.0 re-stage** (+28 pos / −49 black, train-only) | **Stage 0.3** | preserves the ecoregion split |
| **MAE** ViT-B/16 SSL pretraining | **Stage 5** — end-stage, parallel w/ inference | go/no-go = linear-probe beats random-init; → next iteration (v2); does NOT gate v1 |
| **Hard-negative mining** (manual) | **Stage 4** — post first inference | feeds the next iteration |
| Reg grid wd×aug (6.3) · SegFormer / EffB7 (8.2) | ❌ dropped | trigger never fired / low value on a plateau |
| Pseudo-labeling · val-negative growth | ⏸️ backup only | confirmation-bias risk / only if the bootstrap readout becomes decisive |

---

## Cluster takeaways (best-in-cluster)

- **Loss (Phase 3, leaky):** focal + `ignore_w2` is the boundary winner (0.805, seed-confirmed
  0.805–0.820). compound_1to2 + `ignore_w3` close runner-up (0.802–0.817). Tversky variants collapse.
- **Architecture (Phase 5, leaky):** UNet++ baseline (0.790) ≥ FPN (0.794) > DeepLabV3+ (0.788) >
  PSPNet (0.729) ≫ MANet (0.621). No CNN decoder beats UNet++ → arch stays UNet++/EffB5.
- **EXTRA channels (Phase 4, corrected):** NDVI-alone (0.888) ≈ full 8-band (0.876) ≈ SE-PCA (0.874)
  ≫ RGB control (0.830). NBR/SE-proto (+0.017) are weak. → NDVI is the efficient ceiling; the open
  question (Step 3) is whether a channel **combination** + better **fusion** beats NDVI-alone.
  **Seed-confirmed (3 seeds):** NDVI is robust and high — s42 0.888, s43 ~0.90, s44 ~0.92 (live raw
  peaks) — clearly beating RGB by ~0.06 (≫ any plausible σ). NDVI is a **real, low-variance win**.
- **Curriculum (Phase 10, corrected):** r20_pf33 best cell single-seed 0.894, **but seed-confirm is
  high-variance: 0.894 / 0.901 / 0.859 → mean ≈0.885 vs base 0.879 (Δ≈0.006), within std ≈0.021.**
  The curriculum "win" is **not distinguishable from seed noise** at 3 seeds — treat as unconfirmed.
- **⚠ Gate vs measured variance (open):** the corrected-split seed std (~0.02) is **~4× the leaky-split
  σ₀=0.0056** that set the gate G=0.0112. Many single-seed "wins" near G (curriculum, se_pca) are within
  noise. Decision pending: widen G toward ~2σ_corrected (≈0.04) and/or **require a seed-confirm before
  any lock**. NDVI's ~0.06 margin survives either way.
- **Stop-schedule fix (Stage 0.1, audit 2026-06-19):** all 48 prior runs peaked by ~ep52 then trained a
  median 40 wasted epochs (41% of GPU-h, overfitting tail), best checkpoint unchanged. New `base_v2_fast`
  (patience 8→5, start_epoch 101→45, max_epochs 300→120) is **gate-neutral** — validated by `fastcheck`
  (~0.898 ≈ original NDVI 0.888). ~2× throughput; all second-wave runs inherit it.
