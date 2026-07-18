# Experiment Ledger — RTSmapping_v2.1 (SSL pretraining program)

**This file is the v2.1 experiments SSoT.** The v2.0 ledger (`docs/experiment_ledger.md`) is frozen;
this ledger owns everything for the self-supervised-pretraining program: run registry, gate, findings,
and verdicts. Spec: `pretraining/pretraining.md`. Update ritual as in `CLAUDE.md` (harvest with
`scripts/sync_experiments.py --ledger docs/experiment_ledger_v21.md`).

**Metric:** fine-tune runs (`ft_*`) use `val_realistic_pr_auc_geomean` = `best_smoothed`, identical to
v2.0 — the **`score` column is machine-harvested**; do not hand-edit it. Pretraining runs (`pretrain_*`)
have no comparable score; their `score` cell stays `—` and their truth is the recon-loss curve in MLflow
plus the downstream `ft_*` result.

**Isolation:** MLflow experiment `rts-segmentation-v2.1` · configs `configs/v21/` · run dirs
`/mnt/outputs/v2.1/runs/<run_name>` · GCS `gs://rts-mapping-v2/RTS_MODEL_V21/` (corpus under
`PRETRAIN_CORPUS/`). All fine-tunes use the corrected split and the locked v2 recipe verbatim, changing
only encoder backbone + init.

**Corpus hygiene:** pretraining tiles are label-free, and tiles intersecting val/test region footprints
are excluded at corpus build time anyway, so no evaluation pixel is ever seen during SSL.

**Corpus scope limitation (decided 2026-07-15):** the corpus is **4-ch RGB+NDVI, south-only** —
restricted to the S2-covered footprint because pan-arctic NDVI does not yet exist. Channel-identical to
the v2/fine-tune input; geographically narrower than pan-arctic (no high-arctic / Siberian-north terrain).
Alternatives (RGB-only pan-arctic; export pan-arctic NDVI first) were considered and deferred.

<!-- GATE:BEGIN -->
## Gate

Inherited from v2.0: **G = 0.0112**, single-seed screen; lock/verdict needs 3-seed **mean Δ ≥ G** *and*
3/3 sign consistency.

**Arms & decision rules (DINOv3-L MAE):**
| arm | encoder init |
|---|---|
| (a) | EffB5 locked baseline = **0.9218** (existing, no rerun) |
| (b) | DINOv3-L sat493m, fair recipe = **0.9191** (existing family-E run `fm_dinov3sat_l_ndvi_locked`, no rerun) |
| (c) | DINOv3-L + **arctic MAE continue-pretrain** (this program), 3 seeds |

- **SSL helped** iff (c) − (b) ≥ G (0.0112), 3/3 seed signs. (Isolates the domain-adaptation effect:
  same encoder + recipe, only the init differs.)
- **Deployable for v3** iff (c) also beats 0.9218 by ≥ G, 3/3 signs.
- A clean null is a recorded finding: it extends the family-E capacity/encoder null to *domain-adapted*
  ViT pretraining — the one lever family E never tested.
<!-- GATE:END -->

---

<!-- RUN-TABLE:BEGIN — `score` is harvested by scripts/sync_experiments.py; do not hand-edit that column. One run-dir name per row. -->
## Master run table

| name | arm | split | score | status | note |
|------|:---:|:-----:|------:|:------:|------|
| pretrain_dinov3l_arctic | — | — | — | done | MAE continue-pretrain, 8×A100, 80 ep / 92,320 steps, recon loss 1.016 → **0.0763**; encoder_final.pt (1.2 GB) |
| ft_dinov3l_arctic_seed42 | c | corrected | 0.8173 | done | peak ep50 |
| ft_dinov3l_arctic_seed43 | c | corrected | 0.8155 | done | peak ep45 |
| ft_dinov3l_arctic_seed44 | c | corrected | 0.8090 | done | peak ep50 (**mean 0.8139 vs arm-b 0.9191 → Δ −0.1052, 0/3 signs → FAILS gate**) |<!-- RUN-TABLE:END -->

---

## Findings

**A — Arctic MAE continue-pretraining actively HARMS the sat493m encoder. Program closed 2026-07-18.**

| arm | encoder init | best_smoothed |
|---|---|---|
| (a) | EffB5 locked baseline | 0.9218 |
| (b) | DINOv3-L sat493m, fair recipe | 0.9191 |
| (c) | DINOv3-L + arctic MAE (80 ep) | **0.8139** (0.8173 / 0.8155 / 0.8090) |

Δ(c−b) = **−0.1052**, **0/3** seeds positive — not a null but a large, sign-consistent *regression*,
≈9× G in the negative direction. Δ(c−a) = −0.1079. Gate fails on both rules; nothing is deployable.

*The comparison is fair.* `configs/v21/_ft_dinov3l_base.yaml` is a flattened copy of the arm-(b) locked
recipe, verified field-by-field: same `data_root` (`/outputs/v1.0/data_local`), same DINOv3 norm stats,
`multi_scale p=0.0`, `auto_policy trivialaugment`, `boundary_handling: ignore` w2, batch 16, identical
LR/LLRD/freeze schedule, same corrected-split val labels. `encoder_init` loaded **318/318 tensors**
(missing=0, unexpected=0). Only the encoder init differs.

*Mechanism (hypothesis, not tested):* catastrophic forgetting. 80 epochs of MAE pixel-reconstruction at
lr 1.5e-4 over 295k tiles appears to overwrite sat493m's discriminative semantic features with
reconstruction-oriented ones. MAE's objective is known to yield weaker dense-transfer features than the
DINO-style pretraining sat493m already carries, so continue-pretraining a strong discriminative
checkpoint with MAE trades away exactly what made it useful. Consistent with the arm-(c) peaks arriving
*later* (ep45–50 vs the baseline's ep35–40): a weaker init needing more adaptation.

*Not diagnosed further.* Epoch-20/40/60 MAE checkpoints are preserved on GCS
(`gs://rts-mapping-v2/RTS_MODEL_V21/pretrain_dinov3l_arctic/`), so a future "does damage scale with
pretraining length?" ablation is cheap to run if the question is ever revisited. Decision 2026-07-18:
**close the MAE program** rather than spend more compute chasing a −0.105 starting point.

*Scope caveat:* the corpus is south-only 4-ch (see limitation above), so this result speaks to
MAE-on-sat493m for this corpus and schedule — not to SSL in general.

**Consequence for v2.0/v3:** the family-E encoder verdict is unchanged — **EffB5 stays the deployed
encoder**. This extends the family-E encoder null to the one lever it never tested (domain-adapted ViT
weights), and closes it in the negative.

## Dropped & discussed-but-didn't-land

| Idea | Verdict / why |
|---|---|
| EffB5-native masked pretraining | rejected at design time — SE global pooling leaks mask info |
| ConvNeXt / ResNet convnet MIM | rejected 2026-07-15 — locked UNet++ decoder is incompatible with ConvNeXt (stride-4 stem → 0-channel skip stage at every depth); ResNet works with UNet++ but has no timm MIM weights. Pivoted to DINOv3-L (ViT-native masking, already integrated + baselined) |
| MAE from scratch | rejected — corpus (south-only) too small; continue-pretrain from the sat493m init instead |
