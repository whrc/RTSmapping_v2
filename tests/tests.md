# Tests

Living doc for the test suite. **Update this file whenever you add, remove, or meaningfully change a test.** Keep the per-file tables and the coverage-gap section in sync with the code.

---

## Purpose

Phase 0 verification has two tiers (see the plan: `.claude/plans/now-read-the-document-optimized-lynx.md`):

- **Tier 1 — `pytest tests/`**: runs on synthetic fixtures, no GCS, no GPU, ~3 s. Guards code correctness and contracts. **Must be green before any real-data work.**
- **Tier 2 — `scripts/check_data_content.py` + `scripts/check_data.py`**: runs on the real v2.0 bucket once it finalizes. Guards data correctness.

**Green pytest ≠ "this works on real imagery"** — it means the plumbing doesn't crash and the invariants hold on canned input. Real-data surprises (CRS mismatches, radiometric drift, missing EXTRA bands, etc.) are caught by Tier 2, not here.

---

## Running

```bash
# Activate the venv on the L4 VM
source ~/ml-env/bin/activate

# Full suite
pytest tests/ -v

# One file
pytest tests/test_sampler.py -v

# One test
pytest tests/test_sampler.py::test_sampler_determinism_across_epochs -v
```

Deps: `pytest`, `rasterio`, `pandas`, `pyyaml`, `numpy`, `scipy`, `albumentations`, `torch` (CPU is fine). Install via `pip install -r requirements.txt` (torch separately via the CUDA index per `computing/vm_instruction.md`).

---

## Fixtures

Defined in [conftest.py](conftest.py).

| Fixture | What you get | Notes |
|---|---|---|
| `synthetic_dataset` | Temp dir laid out like `gs://.../training/v2.0/`: 4 regions × 3 tiles = 12 tiles (8 Positive, 4 Negative), 64×64 GeoTIFFs in `PLANET-RGB/`, `EXTRA/` (4-band), `labels/`, plus `metadata.csv` and `splits.yaml`. | Returns `{root, metadata_df, splits}`. 64×64 instead of 512×512 for speed. |

Fresh temp dir per test — no cross-test state leakage.

---

## Strictness legend

- **real** — exercises an actual invariant or contract; a genuine bug would fail it.
- **shallow** — smoke test; only catches egregious mistakes (typos, imports, empty returns).
- **placeholder** — present but known to be weak; flagged in "Coverage gaps" for future work.

---

## Test inventory

### [test_splits.py](test_splits.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_load_metadata_and_splits` | CSV + YAML parse, required columns exist, `TrainClass` ∈ {Positive, Negative} | shallow |
| `test_get_tile_ids_returns_correct_counts` | `train` → 6 tile IDs; `class_filter="Positive"` → 4 | real |
| `test_no_region_leakage_passes_on_clean` | Disjoint splits don't raise (strips `val_balanced` which intentionally duplicates `val_realistic` regions) | real |
| `test_no_region_leakage_fails_on_overlap` | Region in two splits → `ValueError` mentioning the region | real |
| `test_split_summary_counts` | Per-split positive/negative counts match the fixture | real |

### [test_normalization.py](test_normalization.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_welford_matches_numpy` | 8-chunk streamed Welford ≈ `np.mean`/`np.std` at atol/rtol 1e-4 | real |
| `test_build_stats_no_extra` | No `"extra"` block when `extra=None` | shallow |
| `test_build_stats_with_extra_variable_channels` | Arbitrary EXTRA channel names (`"custom_signal"`) survive | real — flexible-EXTRA guarantee |
| `test_save_load_roundtrip` | JSON write → read preserves values | shallow |
| `test_stats_to_arrays_rgb_only` | `with_extra=False` returns RGB only | shallow |
| `test_stats_to_arrays_with_extra` | Concatenation order: RGB first, then EXTRA in declared order | real |

### [test_sampler.py](test_sampler.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_parse_schedule_sorted` | Schedule dict parses and sorts by epoch range | real |
| `test_ratio_for_epoch` | Epoch bucket lookup + clamp-to-last on out-of-range | real |
| `test_sampler_ratio_1_1` | batch=8, ratio 1:1 → exactly 4 pos / 4 neg | real |
| `test_sampler_ratio_1_5_shifts_distribution` | batch=12, ratio 1:5 → exactly 2 pos / 10 neg | real |
| `test_sampler_determinism_across_epochs` | Same seed+epoch → identical sequence; different epoch → different sequence | real — reproducibility lock |
| `test_sampler_requires_both_classes` | Zero negatives → `ValueError("both classes")` | real |

### [test_dataset.py](test_dataset.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_parse_extra_spec_empty` | `None` and `[]` both → `[]` | shallow |
| `test_parse_extra_spec_flexible_names` | Arbitrary names parsed, band indices preserved | real — flexible-EXTRA guarantee |
| `test_parse_extra_spec_rejects_missing_keys` | Missing `name` or `band` → `ValueError` | real |
| `test_dataset_rgb_only` | `(3, 64, 64) float32` image, `(64, 64) int64` label, str `tile_id` | real — end-to-end plumbing |
| `test_dataset_with_variable_extra` | Bands [0, 2] + arbitrary names → `(5, 64, 64)` | real — flexible-EXTRA end-to-end |
| `test_dataset_label_values_in_set` | Every label's unique values ⊂ {0, 1, 255} | real |
| `test_boundary_dilation_adds_ignore` | Width=2 dilation creates 255 band and preserves interior 1s | real |

---

## Coverage gaps (known)

Deliberately deferred — most are better caught by Tier 2 against real data than by more synthetic fixtures. Don't close these by adding fake tests; address them when real data lands or when a bug motivates it.

1. **`scripts/create_splits.py` constraint solver** — no test for ecoregion-diversity, test-positive-minimum, or drift-tolerance enforcement. Would need a synthetic GeoJSON + metadata combination. Close this when the solver misbehaves on real domain regions.
2. **Augmentation pipeline behavior** — `test_dataset.py` uses `build_eval_transforms()` (no-op). Bugs in `build_train_transforms()` aren't caught by pytest. Caught instead by `scripts/check_data.py` previews (Tier 2).
3. **Normalization-through-dataset** — `RTSDataset` currently only runs against the "zero mean, unit std" fallback in tests. The full path (load JSON → `stats_to_arrays` → subtract/divide in `__getitem__`) isn't tested end-to-end.
4. **`BalancedBatchSampler.__len__`** — the integer returned is not asserted to match the number of batches actually yielded. If DataLoader relies on `__len__`, an off-by-one would slip through.
5. **Error paths on corrupted rasters** — unreadable GeoTIFF, wrong CRS, size mismatch between RGB/label/EXTRA. All raise somewhere in `RTSDataset` but no test exercises those branches.
6. **Numerical edge cases in Welford** — single distribution, realistic scale. No adversarial `1e10 + 1e-10` test for catastrophic cancellation. Fine for imagery in practice.
7. **Malformed metadata** — missing column, bad `TrainClass`, duplicate `Tile_id`. `load_metadata()` has the guards but they're not exercised.

---

## Conventions for adding tests

1. **Name the file after the module under test**: `test_<module>.py` (no `test_utils_config.py` for multi-module tests — split them).
2. **One assertion concern per test function.** If you need to check five things, write five tests; small focused failures are easier to diagnose than one big test that says "something broke."
3. **Use the `synthetic_dataset` fixture** when you need a real on-disk dataset. Don't manufacture paths by hand in each test.
4. **Tests must be GPU-free and GCS-free.** Anything needing a real bucket goes in a Tier 2 script, not pytest.
5. **Prefer `pytest.raises(ErrType, match="…")` over `except: pass`** — assert that the error message mentions the relevant identifier.
6. **Seed RNGs** for any randomized test (`np.random.default_rng(42)`, `random.Random(seed)`). Never leave test outcomes dependent on unseeded randomness.
7. **No network, no sleep, no disk writes outside `tmp_path`.** pytest's `tmp_path` fixture cleans up; `Path.cwd()` writes do not.
8. **Update this file**:
   - Add your test to the relevant inventory table above with a strictness rating.
   - If you knowingly leave something untested, add it to "Coverage gaps".
   - If you added a new fixture, document it in the Fixtures table.

---

## Dev log

- 2026-04-22 — Initial suite: 24 tests across 4 files, all green. Covers Phase 0 data pipeline. See plan for context.
