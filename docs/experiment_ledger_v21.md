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
<!-- RUN-TABLE:END -->

---

## Findings

*(none yet)*

## Dropped & discussed-but-didn't-land

| Idea | Verdict / why |
|---|---|
| EffB5-native masked pretraining | rejected at design time — SE global pooling leaks mask info |
| ConvNeXt / ResNet convnet MIM | rejected 2026-07-15 — locked UNet++ decoder is incompatible with ConvNeXt (stride-4 stem → 0-channel skip stage at every depth); ResNet works with UNet++ but has no timm MIM weights. Pivoted to DINOv3-L (ViT-native masking, already integrated + baselined) |
| MAE from scratch | rejected — corpus (south-only) too small; continue-pretrain from the sat493m init instead |
