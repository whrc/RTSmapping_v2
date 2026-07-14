# Experiment Ledger — RTSmapping_v2

**This file is the experiments SSoT.** Every training run, the locked recipe, the findings, and the
dropped ideas live here and nowhere else. `docs/report.html` is the generated analytical/visual view of
this file; `current_working_status.md` is the project diary. To update, follow the ritual in `CLAUDE.md`.

**Metric:** `val_realistic_pr_auc_geomean` = `best_smoothed` (higher is better). The **`score` column of
the run table is machine-harvested** from each run's `run_summary.json` by `scripts/sync_experiments.py`
— do not hand-edit it. Everything else is agent-edited.

**Split warning:** scores are **not comparable across the split boundary.** Families A/B/C-loss/E-decoder
ran on the **leaky** region split (relative comparisons only). D/F/G/I and the encoder runs use the
**corrected** leakage-free split (absolute numbers sit higher; compare within-family vs the in-family
control, not across phases). Test-Realistic is scored once, honestly, on the corrected split.

**Status:** done · seed-confirm · running · incomplete (killed before maturity) · crashed · collapsed ·
degenerate · killed (ran to a verdict but failed) · lr-test.

**Families:** A Baseline/gate · B Data · C Loss/boundary · D Channels/fusion · E Architecture/encoder ·
F Augmentation · G Sampling · H Calibration/TTA · I Final-lock/Test · J Deploy/inference · K Deferred ·
M Multi-scale (0.5x context-expanded training POC) · N Data-v1.1 (MMU metric fix + correctness retrain).

<!-- GATE:BEGIN -->
## Gate

| μ₀ | σ₀ (leaky) | **G = max(2σ₀, 0.01)** | σ (corrected) |
|----|-----------|------------------------|---------------|
| 0.7912 | 0.0056 | **0.0112** | ≈0.012 |

**Lock policy:** a change is locked only if a 3-seed confirm shows **mean Δ ≥ G** *and*
**sign-consistency across all 3 seeds** (G alone is a single-seed *screen*). Sign-consistency is the
decisive test — it separated drop-RandomScale (+0.016, 3/3 → locked) from curriculum r20_pf33 (+0.006,
sign-flipped → rejected).
<!-- GATE:END -->

---

<!-- RUN-TABLE:BEGIN — `score` is harvested by scripts/sync_experiments.py; do not hand-edit that column. One run-dir name per row. -->
## Master run table

| name | fam | split | score | status | note |
|------|:---:|:-----:|------:|:------:|------|
| phase0a_arm_a | A | leaky | 0.6666 | done | norm arm A (per-dataset z-score) — LOCKED |
| phase0a_arm_b | A | leaky | 0.6261 | done | norm arm B (x255+ImageNet) |
| phase0a_arm_c | A | leaky | 0.6697 | done | norm arm C (x255-only) |
| phase0b_lr_frozen | A | leaky | 0.0008 | degenerate | LR probe (frozen) |
| phase0b_lr_unfrozen | A | leaky | — | crashed | LR probe (unfrozen) — diverged |
| phase0c_seed42 | A | leaky | 0.7899 | done | 3-seed baseline → μ₀=0.7912, σ₀=0.0056, G=0.0112 |
| phase0c_seed43 | A | leaky | 0.7863 | done | 3-seed baseline |
| phase0c_seed44 | A | leaky | 0.7973 | done | 3-seed baseline |
| phase2_scale_25 | B | leaky | 0.7636 | done | data-scale 25% |
| phase2_scale_50 | B | leaky | 0.7720 | done | data-scale 50% |
| phase2_scale_75 | B | leaky | 0.7916 | done | data-scale 75% → plateau (slope flat) |
| scale_ndvi_25 | B | corrected | 0.7429 | done | data-scale 25%, locked recipe (RGB+NDVI) |
| scale_ndvi_25_seed43 | B | corrected | 0.8412 | done | seed-confirm |
| scale_ndvi_25_seed44 | B | corrected | 0.7946 | done | seed-confirm → 3-seed mean 0.7929, σ=0.0402 |
| scale_ndvi_50 | B | corrected | 0.8628 | done | data-scale 50%, locked recipe |
| scale_ndvi_75 | B | corrected | 0.8732 | done | data-scale 75%, locked recipe |
| phase3_loss_compound_1to1 | C | leaky | 0.7878 | done | loss sweep |
| phase3_loss_compound_2to1 | C | leaky | 0.7933 | done | loss sweep |
| phase3_loss_compound_1to2 | C | leaky | 0.7998 | done | near-miss (Δ<G) → carried to boundary factorial |
| phase3_loss_compound_1to2_seed43 | C | leaky | 0.7872 | done | seed-confirm |
| phase3_loss_tversky_3to7 | C | leaky | 0.5902 | done | Tversky 3:7 |
| phase3_loss_tversky_2to8 | C | leaky | 0.0729 | collapsed | Tversky 2:8 collapse |
| phase3_bd_focal_ignore_w1 | C | leaky | 0.7872 | done | boundary factorial |
| phase3_bd_focal_ignore_w2 | C | leaky | 0.8046 | done | **boundary winner → LOCKED (focal·ignore_w2)** |
| phase3_bd_focal_ignore_w2_seed43 | C | leaky | 0.8200 | done | seed-confirm |
| phase3_bd_focal_ignore_w2_seed44 | C | leaky | 0.8054 | done | seed-confirm |
| phase3_bd_focal_ignore_w3 | C | leaky | 0.7973 | done | boundary factorial |
| phase3_bd_compound_1to2_ignore_w1 | C | leaky | 0.7996 | done | boundary factorial |
| phase3_bd_compound_1to2_ignore_w2 | C | leaky | 0.8003 | done | boundary factorial |
| phase3_bd_compound_1to2_ignore_w3 | C | leaky | 0.8025 | done | close runner-up |
| phase3_bd_compound_1to2_ignore_w3_seed43 | C | leaky | 0.8170 | done | seed-confirm |
| phase3_bd_compound_1to2_ignore_w3_seed44 | C | leaky | 0.8153 | done | seed-confirm |
| ablation_noignore_ndvi_seed42 | C | corrected | 0.8727 | done | ignore-region ablation (train-only) — ignore helps |
| ablation_noignore_ndvi_seed43 | C | corrected | 0.9214 | done | seed-confirm |
| ablation_noignore_ndvi_seed44 | C | corrected | 0.9012 | done | seed-confirm (mean 0.8984, Δ−0.014 vs deploy) |
| phase4_extra_rgb_baseline | D | corrected | 0.8297 | done | **RGB control (corrected-split anchor 0.830)** |
| phase4_extra_ndvi | D | corrected | 0.8879 | done | **best single channel; 3-seed mean 0.8985** |
| phase4_extra_ndvi_seed43 | D | corrected | 0.8965 | done | seed-confirm |
| phase4_extra_ndvi_seed44 | D | corrected | 0.9111 | done | seed-confirm |
| phase4_extra_ndvi_fastcheck | D | corrected | 0.8934 | done | base_v2_fast stop-fix validation (gate-neutral) |
| phase4_extra_full | D | corrected | 0.8763 | done | full 8-band ≈ NDVI-alone |
| phase4_extra_full_seed43 | D | corrected | 0.8619 | done | seed-confirm |
| phase4_extra_full_seed44 | D | corrected | 0.8678 | done | seed-confirm |
| phase4_extra_nbr | D | corrected | 0.8469 | done | single-group NBR |
| phase4_extra_tc | D | corrected | 0.8683 | done | single-group tasseled-cap |
| phase4_extra_se_pca | D | corrected | 0.8736 | done | single-group SE-PCA |
| phase4_extra_se_pca_seed43 | D | corrected | 0.8571 | done | seed-confirm |
| phase4_extra_se_proto | D | corrected | 0.8468 | done | single-group SE-prototype |
| phase4_extra_ndvi_nbr | D | corrected | 0.8559 | done | greedy round-1 NDVI+NBR (no add) |
| phase4_extra_ndvi_tc | D | corrected | 0.8996 | done | greedy round-1 NDVI+TC (no add) |
| phase4_extra_ndvi_sepca | D | corrected | 0.8963 | done | NDVI+SE-PCA |
| phase4_extra_ndvi_sepca_seed43 | D | corrected | 0.8949 | done | seed-confirm |
| phase4_extra_ndvi_sepca_seed44 | D | corrected | 0.9088 | done | seed-confirm (mean 0.900 ≈ NDVI-alone) |
| phase4_extra_ndvi_seproto | D | corrected | 0.8984 | done | NDVI+SE-proto |
| phase4_extra_ndvi_seproto_seed43 | D | corrected | 0.8988 | done | seed-confirm |
| phase4_extra_ndvi_seproto_seed44 | D | corrected | 0.8966 | done | seed-confirm (mean 0.898 ≈ NDVI-alone) |
| phase4_f1_full | D | corrected | 0.8903 | done | F1 smart-stem-init (8-band) ≈ NDVI-alone |
| phase4_f1_ndvi_seproto | D | corrected | 0.8921 | done | F1 (pair) ≈ NDVI-alone |
| phase4_f2_full | D | corrected | 0.8268 | done | F2 channel-attn collapses on 8-band |
| phase4_f2_ndvi_seproto | D | corrected | 0.8891 | done | F2 (pair) < NDVI-alone |
| phase4_f3_full | D | corrected | 0.8184 | done | F3 dual-encoder late fusion — loses to F0 |
| phase4_f5_full | D | corrected | 0.8480 | done | F5 residual cross-modal attn (JSTARS) — loses to F0 |
| phase4_f5_ndvi_seproto | D | corrected | 0.8544 | done | F5 (pair) — loses to F0 |
| phase5_arch_fpn | E | leaky | 0.7939 | done | decoder sweep — ties UNet++ |
| phase5_arch_deeplabv3plus | E | leaky | 0.7878 | done | decoder sweep |
| phase5_arch_pspnet | E | leaky | 0.7288 | done | decoder sweep |
| phase5_arch_manet | E | leaky | 0.6213 | done | decoder sweep (worst) |
| effb3_deploy | E | corrected | 0.9050 | done | capacity-down probe — no-win (cheaper fallback) |
| phase4_fm_dinov3_rgb | E | corrected | 0.8734 | done | web-DINOv3 ViT-B RGB (z-score norm) |
| phase4_fm_dinov3_rgb_lrtest | E | corrected | — | lr-test | LR range test |
| fm_dinov3_rgb_imagenet | E | corrected | 0.8923 | done | web-DINOv3 RGB + ImageNet norm (de-confound) |
| phase4_fm_dinov3_ndvi | E | corrected | 0.9121 | done | web-DINOv3+NDVI **ties EffB5** → generic FM not the lever |
| fm_sam2_rgb | E | corrected | 0.5558 | done | SAM2/Hiera — non-competitive |
| fm_dinov3sat_7b_frozen | E | corrected | 0.4747 | killed | sat-7B frozen — diverged, non-competitive |
| fm_dinov3sat_7b_lrtest | E | corrected | — | lr-test | decoder LR range test — clean up to ~2e-4, explodes ~4e-4 |
| fm_dinov3sat_7b_tuned_a | E | corrected | 0.7435 | done | honest re-tune, decoder_phase2_start_epoch=10 (scheduler fix) |
| fm_dinov3sat_7b_tuned_b | E | corrected | 0.7585 | done | honest re-tune, decoder_phase2_start_epoch=20 — winner |
| fm_dinov3sat_7b_tuned_b_seed43 | E | corrected | 0.7508 | done | seed-confirm |
| fm_dinov3sat_7b_tuned_b_seed44 | E | corrected | 0.7588 | done | seed-confirm → 3-seed mean 0.7561 |
| fm_dinov3sat_l_rgb | E | corrected | 0.9200 | done | sat-DINOv3 ViT-L RGB, 3-seed |
| fm_dinov3sat_l_rgb_seed43 | E | corrected | 0.9195 | done | seed-confirm |
| fm_dinov3sat_l_rgb_seed44 | E | corrected | 0.9003 | done | seed-confirm (mean 0.9133 ≈ EffB5) |
| fm_dinov3sat_l_ndvi | E | corrected | 0.9234 | done | sat-DINOv3 ViT-L +NDVI, seed42 (**off-recipe — confounded**) |
| fm_dinov3sat_l_ndvi_seed43 | E | corrected | 0.9199 | done | seed-confirm |
| fm_dinov3sat_l_ndvi_seed44 | E | corrected | 0.9150 | done | seed-confirm (mean 0.9194, Δ+0.0071 sub-gate) |
| fm_dinov3sat_l_ndvi_seed42_rerun | E | corrected | — | crashed | off-recipe rerun (superseded by _locked) |
| fm_dinov3sat_l_ndvi_locked | E | corrected | 0.9221 | done | **FAIR sat re-run (locked recipe), seed42 — peak ep40** |
| fm_dinov3sat_l_ndvi_locked_seed43 | E | corrected | 0.9286 | done | seed-confirm — peak ep40 |
| fm_dinov3sat_l_ndvi_locked_seed44 | E | corrected | 0.9067 | done | seed-confirm — peak ep35 (**mean 0.9191 ≈ EffB5 0.9218 → TIE → deploy EffB5**) |
| aug_ref | F | corrected | 0.8661 | done | aug control |
| aug_ref_seed43 | F | corrected | 0.8468 | done | seed-confirm |
| aug_ref_seed44 | F | corrected | 0.8808 | done | seed-confirm (ref mean 0.865) |
| aug_p0_geom_only | F | corrected | 0.7936 | done | geometric-only → photometric matters (−0.072) |
| aug_p1_no_clahe | F | corrected | 0.8541 | done | drop CLAHE (−0.012, within noise) |
| aug_p3_photo_x15 | F | corrected | 0.8658 | done | photometric ×1.5 ≈ ref (no gain) |
| aug_pad_ignore | F | corrected | 0.8527 | done | pad-ignore fix — downscale itself hurts, not the pad |
| aug_scale_off | F | corrected | 0.8862 | done | **drop RandomScale → LOCKED (+0.016, 3/3)** |
| aug_scale_off_seed43 | F | corrected | 0.8673 | done | seed-confirm |
| aug_scale_off_seed44 | F | corrected | 0.8892 | done | seed-confirm (mean 0.881) |
| aug_copypaste_deploy | F | corrected | 0.8930 | done | mixing aug — worst (breaks shadow/context cues) |
| aug_mosaic_deploy | F | corrected | 0.9069 | done | mixing aug — no-win |
| aug_cutmix_deploy | F | corrected | 0.9014 | done | mixing aug — no-win |
| aug_mixup_deploy | F | corrected | 0.9028 | done | mixing aug — no-win (family 4/4 struck out) |
| aug_randaugment_deploy | F | corrected | 0.9089 | done | shadow-safe pool — no-win (−0.0034) |
| aug_trivialaugment_deploy | F | corrected | 0.9167 | done | **TrivialAugment → LOCKED color stage (3-seed mean 0.9218)** |
| aug_trivialaugment_deploy_seed43 | F | corrected | 0.9216 | done | seed-confirm |
| aug_trivialaugment_deploy_seed44 | F | corrected | 0.9270 | done | seed-confirm |
| aug_anneal_deploy | F | corrected | 0.9074 | done | aug-strength annealing — no-win |
| aug_anneal_deploy_seed43 | F | corrected | 0.9192 | done | seed-confirm |
| aug_anneal_deploy_seed44 | F | corrected | 0.9204 | done | seed-confirm (mean 0.9157, Δ+0.0034 sub-gate) |
| phase10_curric_base | G | corrected | 0.8786 | done | curriculum control |
| phase10_curric_r20_pf33 | G | corrected | 0.8945 | done | best cell — but seed-confirm = noise → rejected |
| phase10_curric_r20_pf33_seed43 | G | corrected | 0.9013 | done | seed-confirm |
| phase10_curric_r20_pf33_seed44 | G | corrected | 0.8587 | done | seed-confirm (sign-flipped → noise) |
| phase10_curric_r30_pf33 | G | corrected | 0.8528 | done | curriculum cell |
| phase10_curric_r30_pf50 | G | corrected | 0.8530 | done | curriculum cell |
| phase_lock_ndvi_bd_curric | I | corrected | 0.9063 | done | early lock attempt — superseded by deploy_v1 |
| deploy_v1_ndvi_seed42 | I | corrected | 0.9144 | done | **final-lock 3-seed (v2 recipe) — reference baseline** |
| deploy_v1_ndvi_seed43 | I | corrected | 0.9068 | done | final-lock 3-seed |
| deploy_v1_ndvi_seed44 | I | corrected | 0.9156 | done | final-lock 3-seed (mean 0.9123) |
| multiscale_poc_seed42 | M | corrected | 0.9165 | done | multiscale POC — gates 1+2 pass, gate 3 fail (family-M finding) |
| multiscale_poc_seed43 | M | corrected | 0.9284 | done | multiscale POC seed-confirm |
| multiscale_poc_seed44 | M | corrected | 0.9282 | done | multiscale POC seed-confirm |
| v1_1_seed42 | N | corrected | 0.9029 | done | v1.1 correctness retrain — ability WASH (val set −29 black neg; own-thr 0.45) |
| v1_1_seed43 | N | corrected | 0.9086 | done | v1.1 correctness retrain seed-confirm |
| v1_1_seed44 | N | corrected | 0.8906 | done | v1.1 correctness retrain seed-confirm (mean 0.9007; keep v1.0 deployed) |<!-- RUN-TABLE:END -->

---

<!-- RECIPE-TABLE:BEGIN -->
## v2 final recipe (locked)

| Component | Locked choice | Fam | Evidence |
|---|---|:---:|---|
| Channels | RGB + **NDVI** (4-ch), F0 early channel-stack | D | NDVI 0.8985 ≫ RGB 0.830; greedy adds nothing; F1/F2 tie, F3/F5 lose |
| Normalization | per-dataset z-score (Arm A) | A | Arm A > B/C |
| Decoder | **UNet++** (dense skips) | E | UNet++ ≥ FPN > DeepLabV3+ > PSPNet ≫ MAnet — none beats it |
| Encoder | **EffB5** (UNet++/EfficientNet-B5) | E | fair re-run: sat-DINOv3 ViT-L **ties** (0.9191 vs 0.9218, obj metrics equal) → no benefit, EffB5 ~4× cheaper |
| Loss + boundary | focal + **ignore_w2** | C | boundary factorial, seed-confirmed |
| Sampling | default balanced (no curriculum) | G | curriculum rejected (sign-flipped) |
| Augmentation | geometric + **TrivialAugment** − **RandomScale** | F | drop-scale +0.016 3/3; TrivialAugment 0.9218 |

**Training schedule** (reproduce-training only, not a deploy param): `base_v2_fast` — patience 5,
start_epoch 45, max_epochs 120; bf16; seeds 42/43/44; deterministic.

**Deploy/inference** (`configs/deployment.yaml`): threshold 0.65 (object-tuned, H.2) · temperature
0.512321 · TTA none — locked by H calibration 2026-06-25; stride 344 px (~33% overlap); overlap fusion =
distance-from-center Gaussian σ=128 px; NDVI windowed on-the-fly from S2 composites (inference.md §3.3/§4.3).
<!-- RECIPE-TABLE:END -->

<!-- BUILDUP-TABLE:BEGIN -->
## Recipe build-up (PR-AUC, `best_smoothed`, 3-seed)

| Step (cumulative) | PR-AUC | Δ step | Δ vs baseline |
|---|---:|---:|---:|
| RGB baseline | 0.830 | — | — |
| + NDVI | 0.8985 | +0.069 | +0.069 |
| + boundary ignore_w2 + drop-RandomScale (= deploy_v1) | 0.9123 | +0.014 | +0.082 |
| + TrivialAugment (current EffB5 recipe) | 0.9218 | +0.0095 | +0.092 |
| sat-DINOv3 encoder (fair re-run, same recipe) | 0.9191 | −0.003 (tie) | — → **EffB5 deployed** (sat no benefit, 4× costlier) |
<!-- BUILDUP-TABLE:END -->

---

<!-- FINDINGS:BEGIN -->
## Findings (per family)

**A — Baseline & gate.** A reproducible 3-seed baseline (μ₀=0.7912, σ₀=0.0056) sets the winner gate
G=0.0112. Re-measured on the corrected split, σ≈0.012 (~2×) → every lock needs mean Δ≥G **and** 3-seed
sign-consistency. Normalization Arm A (per-dataset z-score) locked.

**B — Data scaling (leaky split, relative-only).** More labeled data is **not** the near-term lever: v1.0
is on a data plateau (25→100%: 0.764→0.792, flat slope) and the model is well-matched to its data
(train/val IoU gap 0.05/0.17 ≪ 0.4). No new labels are coming → squeeze the fixed ~1.5k positives via
representation, not volume. This is the central diagnosis: **representation-limited, not data-volume- or
capacity-limited.**

**B — Data scaling, corrected split + locked recipe (2026-07-11 — softens the finding above).**
Rerunning the same data-scaling sweep on the corrected leakage-free split with the locked RGB+NDVI
recipe (`scale_ndvi_25/50/75` vs the existing `aug_trivialaugment_deploy` 100% point) gives a
**materially rising curve, not a plateau**: 25%→50%→75%→100% = **0.7929 (3-seed mean, σ=0.0402) → 0.8628
→ 0.8732 → 0.9218 (3-seed mean, σ=0.0042)**. End-to-end Δ = **+0.1289 ≈ 11.5× G** — far outside seed noise.
The rise is uneven: 25→50% (+0.070 mean / +0.120 seed42-only) and 75→100% (+0.049 mean / +0.044
seed42-only) are large; only 50→75% (+0.010) sits near-flat at-gate. The train/val pixel-IoU gap
(same §8.1 methodology) **shrinks monotonically** with more data — at the best checkpoint: 25%≈0.32
(mean of 3 seeds: 0.345/0.293/0.328) → 50%=0.226 → 75%=0.155 → 100%=0.080 (at final epoch: ≈0.39 →
0.330 → 0.293 → 0.190) — the opposite of "well-matched"; the low-data models measurably overfit more.
Seed variance at 25% (σ=0.040) is also ~3.6× the 100%-point's σ=0.0042, another data-volume-linked
signal. **Verdict: the "representation-limited, not data-volume-limited" claim from the leaky split
does not hold on the honest split — data volume is a real, substantial lever, especially 25→50% and
75→100%.** The leaky-split row above is kept for the relative/gate-history record but is superseded as
the honest data-scaling read.

**C — Loss & boundary.** The win is the **boundary treatment, not the loss.** Focal alone is the loss
winner; precision-skewed Tversky collapses (2:8 → 0.073). Adding a boundary-ignore band clears the gate
— **focal·ignore_w2 0.8046** (seed-confirmed 0.805–0.820), tied with compound 1:2·ignore_w3
(0.802–0.817); focal·w2 chosen for simplicity. The ignore-region ablation (train-only) confirms ignore
helps (mean 0.8984, Δ−0.014 vs deploy; caveat: positives overwrite ignore, so not a clean counterfactual).

**D — Channels & fusion.** A single well-chosen channel (**NDVI**) is the biggest representation win,
and more is not better. NDVI-alone 3-seed **0.8985 ≫ RGB 0.830** (+0.07, ≫σ) and > full 8-band (0.869).
Greedy forward from NDVI adds nothing (+SE-PCA 0.900, +TC 0.900, +SE-proto 0.898, +NBR 0.856 — none
clears G). Heavy fusion **loses**: F3-full 0.818, F5-full 0.848, F5-pair 0.854 (all ≪ NDVI-alone; below
even the F0 stack). **LOCKED: EXTRA=[NDVI], F0 early channel-stack.**

**E — Architecture & encoder (RESOLVED 2026-06-25 → EffB5).** Capacity is not the lever, and neither is
the encoder. No CNN decoder beats UNet++ (FPN 0.794 ties > DeepLabV3+ 0.788 > PSPNet 0.729 ≫ MAnet 0.621);
EffB3 capacity-down no-win (0.9050). Generic web-DINOv3+NDVI ties EffB5 (0.9121 vs 0.9123). The
satellite-pretrained DINOv3 ViT-L *looked* like a breakout off-recipe (+NDVI 0.9194 + big object metrics)
— **but that was the confound**: sat ran on `phase0c` (boundary-none val labels, no TrivialAugment) and was
compared to the pre-TrivialAugment EffB5 0.9123. The **fair re-run on the full locked recipe** (ignore_w2 +
drop-RandomScale + TrivialAugment, *identical* val labels) settles it — **a dead tie**:

| 3-seed, **locked recipe** (identical val labels) | PR-AUC | pixel-IoU | obj-F1 |
|---|---:|---:|---:|
| EffB5 (= `aug_trivialaugment_deploy`) | **0.9218** | 0.612 | 0.438 |
| sat-DINOv3 ViT-L (= `fm_dinov3sat_l_ndvi_locked`) | 0.9191 | 0.612 | 0.437 |
| Δ (sat − EffB5) | −0.0027 | ≈0 | ≈0 |

> **Verdict: DEPLOY EffB5.** On the matched recipe the satellite ViT-L gives **no benefit** on any metric,
> and EffB5 (CNN) is ~4× cheaper/faster across the 41.57M-tile pass. The off-recipe "sat edge"
> (+0.13 IoU / +0.07 obj-F1 on boundary-none labels) **collapses** once both use the same recipe + val
> labels — a textbook confound, caught by the fair A/B. SAM2 (0.556) and sat-7B-frozen (0.475)
> non-competitive. The sat re-run (`fm_dinov3sat_l_ndvi_locked*`) ran to ep60, peaked ep35–40
> (best_smoothed 0.9221/0.9286/0.9067), terminated in the overfit tail — a complete verdict.

**E — sat-7B frozen, honestly re-tuned (2026-07-12 — the original 0.4747 was a bug, not a capacity
verdict).** Root-caused the original `fm_dinov3sat_7b_frozen` failure from raw MLflow history: no NaNs
(`train_nan_steps=0` throughout), but `lr_decoder` sat flat at 1e-3 for all 29 logged epochs — the
scheduler's frozen-phase branch (`freeze_backbone_epochs≥max_epochs`, the only way to keep a 6.7B-param
encoder permanently frozen on an A100-80GB) also silences the decoder's own warmup/cosine schedule for
the *entire* run, so the decoder never annealed; it collapsed to background-only prediction (pixel_iou→
0.0002) right after the epoch-11 curriculum hardening. Fixed `training/scheduler.py` with a new
`decoder_phase2_start_epoch` key that lets the decoder anneal on its own early timeline while the
backbone stays permanently frozen (fully backward-compatible — defaults to the old behavior). Also fixed
a **second, unrelated disk-bloat bug** in `training/checkpoint.py::save_resume`: it serialized the full
model including the frozen 6.7B-param encoder on every rotation (`keep_last_n=3`) — harmless for small
encoders but ~40GB/run for this one; now omits `encoder.*` when the encoder is currently frozen (restored
for free by `build_model`'s pretrained load on resume). An LR range test (`fm_dinov3sat_7b_lrtest`) showed
the decoder loss is clean up to ~2e-4 and explodes from ~4e-4 — validating `base_lr=2e-4` (unchanged) and
motivating a lower `frozen_lr=1e-4` (down from the original buggy 1e-3). A 2-point sweep over
`decoder_phase2_start_epoch` (10 vs 20, seed 42) gave stable training with **no collapse** in both:
**0.7435** (start=10) and **0.7585** (start=20, winner) — both ran to `max_epochs=120` without early
stopping (still slowly improving, no_improve 1–3). 3-seed confirm of the winner (seeds 42/43/44 → **0.7585 / 0.7508 / 0.7588, mean 0.7561**) plus the fair
matched-recipe table (identical val labels, same as the sat-ViT-L-vs-EffB5 table above):

| 3-seed, locked recipe (identical val labels) | PR-AUC | pixel-IoU | obj-F1 | train wall-clock (h/seed) |
|---|---:|---:|---:|---:|
| EffB5 (= `aug_trivialaugment_deploy`) | **0.9218** | 0.612 | 0.438 | ≈8.65 |
| sat-DINOv3 7B frozen, honestly tuned (= `fm_dinov3sat_7b_tuned_b*`) | 0.7561 | 0.343 | 0.382 | ≈28.48 |
| Δ (7B tuned − EffB5) | **−0.1657** | −0.269 | −0.056 | ≈3.3× slower |

Δ is sign-consistent and enormous across all 3 seeds individually (−0.163 / −0.171 / −0.163) — nowhere
near the gate in either direction.

> **Verdict: fixed and honestly tuned, this sat-7B frozen linear-probe now trains stably (no collapse,
> clears the old 0.556/0.4747 floor by a wide margin) but loses decisively to EffB5 on every metric —
> PR-AUC, pixel-IoU, and object-F1 — while costing ~3.3× more wall-clock to train. This was not a
> training artifact (the original 0.4747 was); with the bug fixed, honest tuning gives the encoder every
> fair chance, and the capacity-null still holds — now on a referee-proof basis. DEPLOY EffB5 stands.**

**F — Augmentation.** Not a plateau-breaker, but two cheap wins lock in: **drop RandomScale** (+0.016,
3/3 seeds) and replace the hand-tuned color stack with **TrivialAugment** (3-seed mean 0.9218, 3/3 >
deploy; Δ+0.0095 just under G but locked by judgment — a parameter-free auto-policy that's lighter and
consistent). Photometric aug matters (geometric-only craters to 0.794, −0.072). All other arms fail:
mixing family 4/4 (copy-paste 0.893 worst, cutmix 0.901, mixup 0.903, mosaic 0.907), RandAugment 0.909,
aug-anneal 0.916.

**G — Sampling.** Default balanced sampling is sufficient — the curriculum r20_pf33 "win" (0.894
single-seed) did not survive seeds (0.894/0.901/0.859, sign-flipped) → rejected as noise.

**H — Calibration & TTA (DONE 2026-06-25, on Val-Realistic).** `scripts/calibrate.py` on the 3 EffB5 seeds:
**TTA → none** (D4-TTA *hurt* 0.9302→0.9234; hflip +0.0014 < 1% gate). **Temperature T≈0.51–0.54** (<1 — the
focal-trained model is *under*-confident, so calibration sharpens logits; threshold lands low ~0.12–0.16).
Per-seed PR-AUC-geomean 0.9161 / 0.9216 / 0.9302 (P≈0.80/R≈0.86–0.88). **3-seed mean-prob ensemble = 0.9393**
(P=0.800/R=0.896). **Deploy = the 3-seed ensemble** (T=0.5123, tta=none → `configs/deployment.yaml`):
chosen for robustness against an unlucky single seed (0.916 vs 0.930), not the marginal +0.0091 (which is
sub-gate).

**H.2 — Object operating point (DONE 2026-06-25, `scripts/tune_object_operating_point.py`, Val-Realistic).**
The calibrate.py threshold (0.1224) is tuned for *pixel* precision and is the **wrong operating point for an
object product**: at thr 0.1224 / min_blob 10 the ensemble scores obj-F1 **0.304** (obj-P 0.189, 443 FP objects,
424 of them no-overlap speckle). Sweeping threshold × min_blob × morph-close picks the **obj-F1 argmax at the
pixel-P≥0.8 floor: thr 0.30 + min_blob 80 + morph off** → obj-F1 **0.567** (obj-P 0.489 / obj-R 0.674, pixel-P
0.931, FP objects 443→93). Robust plateau (obj-F1≈0.56 over thr 0.30–0.35; morph radius 0/1/2 identical → off).
**DEPLOYED operating point (user choice 2026-06-25, precision-leaning): thr 0.65 + min_blob 80 + morph off** →
the object-precision MAXIMUM of the grid, **obj-P 0.611 / obj-R 0.439 (obj-F1 0.511), pixel-P 0.987** (beyond
0.65 both obj-P and obj-R fall — recall collapses faster than precision rises). Chosen for a pan-Arctic survey
favouring few false slumps over completeness (project §1 precision-over-recall). The obj-F1-argmax alternative
was thr 0.30 (obj-F1 0.567, obj-P 0.489/obj-R 0.674). Frozen in `deployment.yaml` before the one-shot;
Test-Realistic gives the honest number.

**I — Final lock.** **Encoder = EffB5** (fair sat-DINOv3 re-run tied, 0.9191 vs 0.9218, equal object metrics →
no benefit at ~4× cost). v2 recipe (RGB+NDVI · F0 · focal·ignore_w2 · default sampling · aug−RandomScale ·
**TrivialAugment** · base_v2_fast) 3-seed **0.9218** (= `aug_trivialaugment_deploy`, 3 clean checkpoints),
deployed as a **3-seed ensemble** (calibration in H) at the precision-leaning operating point (H.2).

**J — Test-Realistic (SHIPPED v2 NUMBER, one-shot 2026-06-25).** 3-seed EffB5 ensemble, frozen op-point
(thr 0.65 · min_blob 80 · T 0.5123 · tta none · scale 1.0), held-out test split (107 pos / 2050 neg tiles):
**object precision 0.584 / recall 0.437 / F1 0.500** (IoU≥0.3) · pixel IoU 0.432 / F1 0.604 · **PR-AUC
0.855 / 0.833 / 0.812 at 1:200 / 500 / 1000** (geomean 0.833, deployment-realistic prevalence). The op-point
generalized cleanly — object metrics ≈ Val (obj-F1 0.511→0.500, obj-R 0.439→0.437, obj-P 0.611→0.584), no
val-overfit. (PR-AUC looks below the Val 0.9393 only because that was at 1:5/10/20; this is the harder, correct
1:200–1000 regime.) `/mnt/outputs/v1.0/test_realistic/effb5_ensemble_metrics.json`. Touched once — frozen.
Remaining to ship: package the 3 per-seed deployment packages → Phase E (bucket/fleet) → Phase F (pre-flight
→ full pan-Arctic inference).

**K — Residual-error diagnostics (pre-S2, 2026-06-29, `scripts/analyze_residual_errors.py`, Val-Realistic).**
On the cached 3-seed ensemble val predictions (2151 tiles, 132 GT objects; deployed scale T=0.5123).
Facts only.

*Signal typology* — per GT object, max predicted prob inside its footprint, binned at low_thr 0.30 /
deploy_thr 0.65: **detected_at_deploy 80 (60.6%) · recoverable_below_deploy 15 (11.4%) · perception_invisible
37 (28.0%)**. So the **perception-invisible floor = 0.280** (max prob < 0.30 → no signal; min_blob/IoU-
independent), and 11.4% of GT objects carry sub-deploy-threshold signal (max prob in [0.30, 0.65)). Invisible
objects are not all small: area px p50 929 / p90 4424 / max 13360. Invisibles concentrate in **Canadian Low
Arctic tundra 29/118** and **Northeast Siberian coastal tundra 8/12**.

*Per-region scoring* (object machinery = `training.metrics`; ALL roll-up at thr 0.65/min_blob 10/morph 0 =
obj 58/74/74 = the `object_operating_point_report.json` grid row → parity verified). At parity min_blob 10:
**Canadian Low Arctic tundra** (118 GT obj, the bulk) obj-P/R/F1 0.44/0.47/0.46, pix-R 0.52; **Northeast
Siberian coastal tundra** (12 GT obj) **obj-R 0.00 — 12/12 missed** (worst scorable region); **Beringia
lowland tundra** 2/2 (n=2). **Interior Alaska-Yukon lowland taiga** and **Novosibirsk Islands Arctic desert**
have **0 GT positives** (negative-only val regions → not scorable; recall/F1 reported null, not 0). At the
product point (min_blob 2000): Canadian 0.51, Beringia 0.67, Northeast Siberian 0.00.

Caveat: 132 val objects, only 3 regions with positives (one n=2) → qualitative, not proportions with CIs.
Artifact: `/mnt/outputs/v1.0/diagnostics/residual_errors_report.json`.

*Per-region Test-Realistic* (2026-06-29, `scripts/evaluate_test.py --by-region`, deterministic re-run of the
one-shot, `test_probs.npz` cached so test is never re-touched again). **Determinism verified:** at min_blob 80
the ALL roll-up reproduces frozen J **exactly** — obj-P 0.5839 / R 0.4372 / F1 0.5000. **Geographic
concentration:** all **215 test GT objects fall in a single region — Northwest Russian-Novaya Zemlya tundra**;
the other 3 test regions (Cook Inlet taiga, Northern Canadian Shield taiga, West Siberian taiga) have **0 GT
positives** (specificity-only — they exercise false-positive behaviour, not detection). So the shipped test
detection metric is a **one-region** number, not a multi-region pan-Arctic average (Val is likewise dominated
by one region, 118/132 objects in Canadian Low Arctic). **First-product point on test** (thr 0.65 / min_blob
2000, recomputed from the cache per the `deployment.yaml` note): obj-P **0.768** / R **0.400** / F1 **0.526**
(tp 86 / fp 26 / fn 129), vs the min_blob-80 anchor 0.584 / 0.437 / 0.500. Artifacts:
`/mnt/outputs/v1.0/test_realistic/effb5_ensemble_by_region.json`, `/mnt/outputs/v1.0/diagnostics/test_probs.npz`.

**M — Multi-scale (0.5x context-expanded training POC, started 2026-07-02).** Motivation: the v2.0
model does not transfer zero-shot to 2× GSD (scale-0.5 tiny-AOI test: 0 blobs vs 9, hot-region IoU
0.000 — `docs/inference_validation.md`), and the inference.md §6.4 gate presumes multi-scale must come
from training. Design (user decisions): full re-stage of the dataset at 0.5x (~9.55 m/px, 1024-px
native windows bilinear-downsampled to 512, matching the inference scale-0.5 read) · ignore features
touching a refined positive **auto-converted to positive** (they were ignore for lack of within-tile
context; 115/168 convert) · joint dual-scale training, 3 seeds. New guards at 0.5x: unrefined-ARTS
255 (known-but-undelineated RTS in the 4× context must not train as background), sub-pixel (<10 px)
positives → 255. Staged: `gs://rts-mapping-v2/training/v1.0_scale05` — **21,934 tiles (1,491 pos /
20,443 neg)**; splits resolve to train 17,679 / val_realistic 2,155 (93 pos) / test 2,100. Provenance:
staging vectors reproduce 57/60 sampled v1.0 label tiles exactly (3 misses = later hotfix edits).
Norm: v1.0 stats reused; measured drift mean −1.2..−1.8%, std −5..−7% (bilinear variance loss —
accepted, keeps the 1x baseline unconfounded). Ignore share in 0.5x positives: 0.71% of px (vs 3.40%
positive px) — the context-expansion did remove most ignore need. **Pre-registered gates:**
(1) *1x no-regression*: 3-seed mean best_smoothed within seed noise of the TrivialAugment baseline
0.9218 (Δ ≥ −0.01); (2) *0.5x capability*: 0.5x-val F1 ≥ 0.7× same-model 1x val F1 (zero-shot today
collapses to ~0) + tiny-AOI `--scale05` blobs reappear; (3) *helps-final-performance*: §7.3
average-fusion of 1x+0.5x val predictions beats 1x-only object recall by > seed spread (0.052) at no
F1 loss. Runs: `multiscale_poc_seed{42,43,44}` (locked recipe + `data.additional_roots` delta only).

*Results (2026-07-03, `scripts/evaluate_multiscale_poc.py`, artifacts `/mnt/outputs/multiscale_poc_eval/`).*
**Gate 1 — PASS.** 3-seed best_smoothed 0.9165 / 0.9284 / 0.9282, mean **0.9244** vs baseline 0.9218
(per-seed Δ vs the matching TrivialAugment seed: −0.0002 / +0.0068 / +0.0012 — all within noise, none
below −0.01). Adding 17,679 0.5x train tiles does not hurt 1x; the +0.0026 mean is below G, not a lock.
**Gate 2 — PASS 3/3.** Same-checkpoint 0.5x val (2,155 tiles / 93 pos): geomean 0.8234 / 0.8166 / 0.8183
vs the 1x-only baseline checkpoint's **0.7500**; obj-F1 ratio to own 1x = 0.787 / 0.808 / 0.781 (gate
≥0.7; baseline = 0.588). Note: on 2024 same-domain tiles the baseline degrades rather than fully
collapses — the tiny-AOI zero-blob result compounded 2025 imagery + the uncalibrated v2.0 dev
checkpoint. Multiscale training closes most of the 0.5x gap regardless.
**Gate 3 — FAIL (pre-registered).** §7.3 average-fusion on the 1x val grid (2,096/2,151 tiles covered):
fused object recall never exceeds 1x-only at any matched threshold (best-F1 points: R 0.674/0.630/0.704
fused vs 0.704/0.630/0.756 1x). Fused geomean +0.0189 / +0.0095 / −0.0016 and best-F1 Δ +0.039 / −0.008
/ −0.013 — sign-flipped, mean < G. Fusion trades recall for precision; no recall gain > seed spread.
Consistent with Finding K: residual val misses are perception-invisible, not FOV-limited (val objects
are small/medium, one dominant region). Deferred with the rest of inference validation: tiny-AOI
`--scale05` 2025 rerun (needs the S2-gated s2_index for NDVI-at-inference over Banks Island).

**N — Minimum Mapping Unit metric correction (2026-07-04, `data.apply_min_mapping_unit`, free re-score).**
Object-wise scoring counted every GT connected-component as a full object regardless of size, while
predictions are size-filtered (`_filter_small_blobs`, deploy min_blob 2000). At min_blob 2000 / iou_thr
0.3 any GT `< min_blob*iou_thr = 600 px` is structurally unmatchable → a guaranteed false negative, and
also inflates the Finding-K perception-invisible floor. Domain-expert diagnosis: 0–50 px GT blobs are
rasterization artefacts; 50–400 px are real RTS pixels but the boundary-clipped tail of a slump whose
body is in the neighbouring tile (negligible pixel-IoU weight, full object-count weight). Fix: mark
sub-Minimum-Mapping-Unit positive components as ignore (255) uniformly at load (loss + live metric) and
at scoring — one shared `apply_min_mapping_unit` (pure size floor, no fill/close), default off
(reproducibility preserved), no retrain. Frozen 3-seed ensemble re-score at the deploy point (thr 0.65 /
min_blob 2000), scorecard self-check True at every value:

| MMU px | Val obj-P/R/F1 · floor (132 GT) | Test obj-P/R/F1 · floor (215 GT) |
|---|---|---|
| 0 (off) | 0.793 / 0.348 / 0.484 · 0.280 | 0.768 / 0.400 / 0.526 · 0.223 |
| 50 | 0.793 / 0.357 / 0.492 · 0.271 | 0.768 / 0.410 / 0.534 · 0.205 |
| 400 | 0.793 / 0.374 / 0.508 · 0.244 | 0.768 / 0.430 / 0.551 · 0.170 |
| 600 | 0.793 / 0.380 / 0.514 · 0.231 | 0.768 / 0.441 / 0.560 · 0.159 |

**Object precision is invariant** (0.793 val / 0.768 test at every value): the floor only removes
unmatchable-GT false negatives, never a false positive. MMU 50 (artefacts only) barely moves; the bulk
of the correction is the 50–600 px edge-partial band. Excluded set at 600: val 11/132 (8.3%), test
20/215 (9.3%). The Finding-K perception-invisible floor 0.280 → 0.231 (val) and the shipped test floor
0.223 → 0.159 (−6.4 pt, ~29% of the "floor" was un-scoreable slivers). Artifacts:
`/mnt/outputs/v1.0/staging/data_v1_1_audit/mmu_rescore/` (per-MMU scorecards + `excluded_audit_{val,test}.json`).

**N-retrain — v1.1 data-correctness 3-seed retrain = ability WASH (2026-07-04, `v1_1_seed{42,43,44}`).**
v1.1 applied three unambiguous row-level fixes (small-blob deletion dropped — superseded by the MMU
metric fix): +28 restored positives (hotfixed regions), −49 all-black negatives, +`vjn7wxyufczs`
promotion (test-region label correction). **Training-relevant delta = +25 train pos / −16 train black
neg = 0.14% of 17,951 train tiles** → predicted within seed noise. Staged multiroot (v1.0 symlink
primary + 29-tile delta root; the 28 restores needed NDVI regenerated via GEE, S2-2024, scale-matched
mean 0.440). Same locked recipe (`aug_trivialaugment_deploy`); only the data differs.

*Raw gate looked like a regression, but both "worse" signals are confounds:*
- val best_smoothed 0.9029/0.9086/0.8906 (mean **0.9007**) vs v1.0 0.9167/0.9216/0.9270 (**0.9218**) —
  **confounded**: v1.1 val_realistic lost 29 all-black (trivial) negatives (2151→2122), a different,
  harder prevalence-conditioned val set.
- object P/R at the fixed deploy point (thr 0.65 on v1.0's T=0.512321) dropped ~6 pt — **pure
  calibration**: v1.1's val-optimal threshold is **0.45, not 0.65** (`tune_object_operating_point`).

*Calibration-free / fairly-calibrated metrics say TIE:*
- Temperature-invariant test **pixel PR-AUC: v1.1 0.9976 ≈ v1.0 0.9970**; MMU600 invisible floor
  0.154 vs 0.159 (v1.1 hair better).
- Each model at its **own** val-tuned operating point (v1.0 thr0.3/mb80, v1.1 thr0.45/mb80):

  | | VAL obj-F1 | TEST obj-F1 | val−test gap |
  |---|---|---|---|
  | v1.0 | 0.5669 (P.489/R.674) | 0.6272 (P.668/R.591) | −0.060 |
  | v1.1 | 0.5615 (P.570/R.553) | 0.6069 (P.701/R.535) | −0.045 |

  VAL F1 tied (0.567 vs 0.562); TEST F1 v1.0 +0.020 (≈ seed noise). **v1.1 has a tighter val−test gap
  (0.045 vs 0.060)** — more honest generalization — and **leans precision** (higher P / lower R at both
  splits), a bias aligned with the precision-leaning deploy (min_blob 2000).

**Verdict:** v1.1 model ability = v1.0 (the 0.14% delta did not move it); the apparent drop was
calibration + a val-set change, not a regression. **Deploy decision: keep v1.0** (incumbent, calibrated
at thr 0.65); retain the v1.1 cleaner-label dataset + checkpoints for the next real modeling change
(shipping v1.1 would need its own thr≈0.45 calibration first). The object-score win from the data-v1.1
effort is finding N (the MMU metric fix), not the retrain. Artifacts: `/mnt/outputs/v1_1/`
(runs, diagnostics, object_operating_point) + `/mnt/outputs/v1.0/staging/data_v1_1_audit/`.
<!-- FINDINGS:END -->

---

## Dropped & discussed-but-didn't-land

| Idea | Fam | Verdict / why |
|---|:---:|---|
| Curriculum r20_pf33 | G | tested → rejected (within seed noise, sign-flipped) |
| RandomScale downscale aug | F | tested → dropped (removing it +0.016, all seeds) |
| Mixing augs (copy-paste / mosaic / cutmix / mixup) | F | tested → no-win (4/4; copy-paste worst) |
| RandAugment · aug-strength annealing | F | tested → no-win (sub-gate) |
| F2 channel-attention (8-band) | D | tested → collapsed (0.827) |
| F3 dual-encoder / F5 cross-modal attn | D | tested → lose to F0 (heavy fusion extracts less than the stack) |
| EffB3 capacity-down | E | tested → no-win (0.9050); kept as a cheaper deploy fallback only |
| SAM2 / sat-7B-frozen | E | tested → non-competitive (0.556 / 0.475) |
| Web-DINOv3 + EXTRA | E | tested → ties EffB5 once NDVI is in → generic FM not the lever |
| §6.5 loss×wd×curriculum interaction | C/F/G | dropped (moot after wd dropped + curriculum rejected + loss locked) |
| wd × aug regularization grid | F | dropped (over-parameterization trigger never fired) |
| SegFormer (mit_b5) · EffB7 · UNet3+ | E | dropped (low value / overfit risk on a plateau / condition unmet) |
| YOLO / Mask R-CNN | E | rejected (paradigm mismatch) |
| SAM3 | E | dropped (image incompatible — py3.12/torch2.7) |
| Re-run Phase 2 on 3500 positives | B | moot (no new labels) |
| Pseudo-labeling / self-training | K | backup only (confirmation-bias risk) |
| Soft-label boundary handling | C | deferred (ignore covers annotation noise for v2) |

**Deferred to v3 (K):** v1.0 re-stage (+28 pos / −49 black) · hard-negative mining (post first
inference) · MAE SSL pretraining (user-go, end-stage). **Conditional:** scale-TTA · ensemble (decided
at final lock) · context-expansion multi-scale (post-inference map review).
