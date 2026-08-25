# Interannual run — progress

> **Generated file — do not hand-edit.** Written from the per-year state at
> `/mnt/outputs/interannual_inference/state/<year>.json`, which is the source of
> truth. Refresh with `python interannual_inference/status.py --write-progress`;
> `run_stage.py` also rewrites it after every stage transition.

Last updated: 2026-08-25T09:23:42+00:00

✅ done · 🔄 running · ⏸ awaiting human sign-off · ❌ failed · · not started

| year | acquire | s2_export | quad_index | s2_index | drift_check | tile_grid | shard | infer | reconcile | merge | vectorize | qc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **2022** | ✅ | · | ✅ | · | · | · | · | · | · | · | · | · |

## Evidence

- **2022** — `quad_index` evidence_error division by zero

## Earth Engine project per year

Quota and concurrency are per-project, so each year exports on its own.

| year | EE project |
|---|---|
| 2019 | `ee-proj-b` |
| 2022 | `ee-proj-a` |
