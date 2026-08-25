# Interannual run — progress

> **Generated file — do not hand-edit.** Written from the per-year state at
> `/mnt/outputs/interannual_inference/state/<year>.json`, which is the source of
> truth. Refresh with `python interannual_inference/status.py --write-progress`;
> `run_stage.py` also rewrites it after every stage transition.

Last updated: 2026-08-25T10:41:47+00:00

✅ done · 🔄 running · ⏸ awaiting human sign-off · ❌ failed · · not started

| year | acquire | s2_export | quad_index | s2_index | drift_check | tile_grid | shard | infer | reconcile | merge | vectorize | qc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **2019** | 🔄 14% | 🔄 0% | · | · | · | · | · | · | · | · | · | · |
| **2020** | · | · | · | · | · | · | · | · | · | · | · | · |
| **2021** | · | · | · | · | · | · | · | · | · | · | · | · |
| **2022** | ✅ | 🔄 7% | ✅ | · | ✅ | ✅ | ✅ | · | · | · | · | · |
| **2023** | · | · | · | · | · | · | · | · | · | · | · | · |
| **2024** | · | · | · | · | · | · | · | · | · | · | · | · |

## Evidence

- **2019** — `acquire` n_ordered 38,597, n_total 308,686, n_failed 0, n_done 38,597 · `acquire` 42,518/308,686 · `s2_export` n_cells 0, n_cells_expected 1,799 · `s2_export` 1/1,799, ETA 1798h
- **2020** — `s2_export` n_cells 0, n_cells_expected 1,799
- **2021** — `s2_export` n_cells 0, n_cells_expected 1,799
- **2022** — `acquire` n_ordered 309,109, n_total 309,109, n_failed 0, n_done 309,109 · `s2_export` n_cells 0, n_cells_expected 1,799 · `s2_export` 124/1,799, ETA 335h · `quad_index` n_quads 309,107 · `drift_check` worst_mean_drift_sigma 0.0954, worst_std_ratio 0.0471, baseline inference/quad_baseline_2025q3.csv · `tile_grid` n_tiles 41,568,231 · `shard` n_shards 2,079, n_tiles 41,568,231

## Earth Engine project per year

Quota and concurrency are per-project, so each year exports on its own.

| year | EE project |
|---|---|
| 2019 | `abruptthawmapping` |
| 2020 | `abruptthawmapping` |
| 2021 | `abruptthawmapping` |
| 2022 | `abruptthawmapping` |
| 2023 | `abruptthawmapping` |
| 2024 | `abruptthawmapping` |
