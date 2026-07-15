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

<!-- GATE:BEGIN -->
## Gate

Inherited from v2.0: **G = 0.0112**, single-seed screen; lock/verdict needs 3-seed **mean Δ ≥ G** *and*
3/3 sign consistency.

**Arms & decision rules (Stage 1):**
| arm | encoder init |
|---|---|
| (a) | EffB5 locked baseline = **0.9218** (existing, no rerun) |
| (b) | ConvNeXt-B ImageNet (control: encoder-swap effect) |
| (c) | ConvNeXt-B FCMAE-IN (off-the-shelf `convnextv2_base.fcmae`) |
| (d) | ConvNeXt-B arctic continue-pretrain (FCMAE-lite on the corpus) |

- **SSL helped** iff (d) − (c) ≥ G, 3/3 signs.
- **Deployable for v3** iff (d) also beats 0.9218 by ≥ G, 3/3 signs.
- **Stage-2 (ViT-MAE) go/no-go:** GO if SSL helped; weak-GO (single-seed screen only) if (d) − (c) > 0
  on 3/3 seeds but < G; NO-GO if (d) ≤ (c) — record the null and stop.
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
| EffB5-native masked pretraining | rejected at design time — SE global pooling leaks mask info; no reference implementation; ConvNeXt swap is the standard path |
| ViT-MAE from scratch | rejected at design time — 1–2M tiles too small; continue-pretrain from satellite/IN MAE weights instead |
