# Archive — v2-alpha / v2.1-alpha preliminary datasets (superseded 2026-06-13)

## Version scheme (canonical)
- **Model / project = `v2`** — "RTS Segmentation Model v2" (bucket `RTS_MODEL_V2/…`). A model-
  generation axis, **not** a dataset version.
- **`v2-alpha`** — the original Model-v2 development dataset (the frozen ~15,528-tile snapshot,
  `metadata_phase0c.csv` / `splits_phase0c.yaml`). Unstable/preliminary. **Destroyed by an external
  in-place bucket rewrite on 2026-06-12** (versioning Suspended → unrecoverable).
- **`v2.1-alpha`** — the regenerated, in-progress drop that followed (still mid-production, batch
  labels churning). Also preliminary.
- **`v1.0`** — the **first STANDARD dataset** = **`batch1 + batch2 + batch3`** (cleaned/QC'd, the new
  stable ground; `gs://rts-mapping-v2/training/v1.0/`). From v1.0 on, the dataset is defined by its
  **batch composition** (add batch4/5… → v1.1, v1.2…).

Everything in this folder is from the **alpha phase** and run on `v2-alpha` data. **None of its
numbers apply to v1.0** — the dataset changed entirely, so every metric/gate/curve was re-measured.
Kept only as a historical record; the methodology/locked-config *decisions* live in the current
living specs (`training/experiments.md`, `training/training.md`, `docs/baseline_unetpp_effb5.md`).

## What's here
- `phase0_baseline.md` — alpha Phase-0 calibration (μ₀=0.5683, σ₀=0.0125, gate G=0.025).
- `phase2_data_scaling.md` — alpha Phase-2 data-scaling curve (0.5361→0.5607).
- `devlog_v2-alpha.md` — condensed dev-log of the alpha phase.

## Also removed in the 2026-06-13 cleanup (recoverable from git, not copied here)
- 18 pre-made experiment configs that pointed at the deleted `v2-alpha` snapshot
  (`abl_loss_*`, `phase2_scale_*`, `phase3_*`, `phase5_*`). Per `training/experiments.md §11.1`
  phase configs are recreated on demand when each phase's winner locks.

## Recovery
Full pre-cleanup state is preserved at git tag **`v2-alpha-archive`**:
```
git show v2-alpha-archive:configs/phase3_loss_compound_2to1.yaml   # any removed file
git checkout v2-alpha-archive -- <path>                            # restore one file
```
