# Archive — v2.0 / v0.x preliminary work (superseded 2026-06-13)

Everything here is from the **preliminary phase** of the project, run on the **v2.0 / v0.2
training dataset** that was **destroyed by an external in-place bucket rewrite on 2026-06-12**
(`gs://abrupt_thaw/.../TRAINING_DATA`, versioning Suspended → unrecoverable). The data team then
regenerated the dataset; **v1.0** (`gs://rts-mapping-v2/training/v1.0/`) is the new standard ground.

**None of the numbers in these documents apply to v1.0** — the dataset changed (15,528 → 22,259
cleaned tiles, different positives and negatives), so every metric, gate, and curve was re-measured
from scratch. These files are kept purely as a historical record of the preliminary calibration and
ablations; nothing in them was migrated forward (the methodology and locked-config *decisions* live
in the living specs `training/experiments.md`, `training/training.md`, and
`docs/baseline_unetpp_effb5.md`, which remain current).

## What's here
- `phase0_baseline.md` — preliminary Phase-0 calibration (μ₀=0.5683, σ₀=0.0125, gate G=0.025) on v2.0.
- `phase2_data_scaling.md` — preliminary Phase-2 data-scaling curve (0.5361→0.5607) on v2.0.

## Also removed in the 2026-06-13 cleanup (recoverable from git, not copied here)
- 18 pre-made experiment configs that pointed at the deleted v2.0 snapshot
  (`metadata_phase0c.csv` / `splits_phase0c.yaml`): `abl_loss_{compound,tversky}`,
  `phase2_scale_{25,50,75}`, `phase3_*` (10), `phase5_*` (3). Per `training/experiments.md §11.1`
  phase configs are created on demand as each phase's winner locks, so these are regenerated when
  the v1.0 ablations run — not archived.

## Recovery
The complete pre-cleanup state (all files above + the full v2.0 dev-log) is preserved at git tag
**`v2.0-preliminary-archive`**:

```
git show v2.0-preliminary-archive:configs/phase3_loss_compound_2to1.yaml   # any removed file
git checkout v2.0-preliminary-archive -- <path>                            # restore one file
```
