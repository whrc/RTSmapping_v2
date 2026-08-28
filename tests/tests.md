# Tests

Living doc for the test suite. **Update this file whenever you add, remove, or meaningfully change a test.** Keep the per-file tables and the coverage-gap section in sync with the code.

---

## Purpose

Two-tier verification:

- **Tier 1 — `pytest tests/`**: runs on synthetic fixtures, no GCS, no GPU. Fast suite ~12 s, plus the end-to-end train-smoke at ~130 s. Guards code correctness and contracts. **Must be green before any real-data work.**
- **Tier 2 — real-data scripts**: runs on the real v2-alpha bucket on the L4 VM.
    - Phase 0 data checks: `scripts/check_data_content.py` (bucket structure) + `scripts/check_data.py` (DataLoader preview).
    - Phase 1 training smoke: `python scripts/train.py --config configs/smoke.yaml` (2 epochs on a subset of real regions; inference.md §6.4 gate).
    - Inference validation: `scripts/validate_inference_tiny.py` (real 2025 quads + GPU; 11 PASS/FAIL checks on overlap/stitching/fusion/NoData/resume — results in `docs/inference_validation.md`).

**Green pytest ≠ "this works on real imagery"** — it means the plumbing doesn't crash and the invariants hold on canned input. Real-data surprises (CRS mismatches, radiometric drift, missing EXTRA bands, OOM on real tile sizes) are caught by Tier 2, not here.

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
| `synthetic_dataset` | Temp dir laid out like `gs://.../training/v2-alpha/`: 4 regions × 3 tiles = 12 tiles (8 positive, 4 negative), 64×64 GeoTIFFs in `PLANET-RGB/`, `EXTRA/` (4-band), `labels/`, plus `metadata.csv` and `splits.yaml`. | Returns `{root, metadata_df, splits}`. 64×64 instead of 512×512 for speed. |

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
| `test_load_metadata_and_splits` | CSV + YAML parse, required columns exist, `TrainClass` ∈ {positive, negative} | shallow |
| `test_get_tile_ids_returns_correct_counts` | `train` → 6 tile IDs; `class_filter="positive"` → 4 | real |
| `test_no_region_leakage_passes_on_clean` | Disjoint splits don't raise (strips `val_balanced` which intentionally duplicates `val_realistic` regions) | real |
| `test_no_region_leakage_fails_on_overlap` | Region in two splits → `ValueError` mentioning the region | real |
| `test_split_summary_counts` | Per-split positive/negative counts match the fixture | real |
| `test_load_tile_allowlist_parses_and_rejects_empty` | One id per line, blanks ignored; empty file → `ValueError` | real — `splits.tile_allowlist` |

### [test_normalization.py](test_normalization.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_welford_matches_numpy` | 8-chunk streamed Welford ≈ `np.mean`/`np.std` at atol/rtol 1e-4 | real |
| `test_build_stats_no_extra` | No `"extra"` block when `extra=None` | shallow |
| `test_build_stats_with_extra_variable_channels` | Arbitrary EXTRA channel names (`"custom_signal"`) survive | real — flexible-EXTRA guarantee |
| `test_save_load_roundtrip` | JSON write → read preserves values | shallow |
| `test_stats_to_arrays_rgb_only` | `with_extra=False` returns RGB only | shallow |
| `test_stats_to_arrays_with_extra` | Concatenation order: RGB first, then EXTRA in declared order | real |
| `test_fill_nodata_inference_convention_chw_perpixel_float` | Shared `fill_nodata_with_mean` on the inference path (CHW float32, per-pixel mask broadcast across channels) fills exact (unrounded) mean; rest untouched — Rule-3 train/inference parity | real |
| `test_fill_nodata_rounds_for_integer_raster` | uint8 raster → mean rounded to dtype (on-disk raw-value contract) | real |
| `test_build_norm_arrays_modes_and_clip` | `build_norm_arrays` maps the stats `mode`/`clip`/`scale` to per-channel arrays (RGB plain z-score; EXTRA zscore-with-clip vs fixed_scale) — data.md §9 | real |
| `test_apply_norm_zscore_clip_and_fixed_scale` | `apply_norm` clips before z-score on zscore channels; divides by `scale` (no z-score) on fixed_scale (SE_PROTO) channels | real — the §9 dispatch contract |
| `test_apply_norm_neutralizes_nonfinite_extra` | Non-finite EXTRA pixels (SE coverage gaps, NDVI/NBR div-zero) → 0 post-norm (channel mean / no-signal), so NoData can't propagate NaN into the network — fixes the Phase-4 validation crash (2026-06-17) | real |
| `test_apply_norm_rgb_only_matches_plain_zscore` | No EXTRA/modes ⇒ dispatch == plain `(x-μ)/σ` (backward-compat) | real |
| `test_build_stats_dict_records_modes` | `build_stats_dict` carries `extra_modes`/`extra_clips`/`extra_scales` into the `extra` block | shallow |

### [test_extra_channels.py](test_extra_channels.py)

EXTRA derivation SSoT (`data/extra_channels.py`). SE + ArcticDEM math only — Earth Engine is
mocked or bypassed (`dem_derivatives` is pure numpy/scipy; the grid builders are pure arithmetic).

| Test | Checks | Strictness |
|---|---|---|
| `test_band_norm_mode` | `band_norm_mode` returns "zscore" for NDVI/SE_PCA/TC and "fixed_scale" for SE_PROTO; unknown band → `ValueError` | real — §9 SSoT |
| `test_se_bands_projection_and_cosine` | With `fetch_se_raw` mocked: `se_bands` returns {2,3,4,5} of shape (H,W); SE_PROTO ∈ [-1,1]; SE_PCA1 == manual `flat @ component[0]` projection | real — SE derivation math |
| `test_se_bands_nan_propagates` | A no-coverage (NaN) SE pixel yields NaN SE bands; finite pixels stay finite (matches S2 NaN handling) | real |
| `test_se_bands_zero_vector_is_nan` | A no-coverage SE pixel arriving as an all-zero vector (not NaN) → NaN SE_PCA *and* SE_PROTO, so `(0-pca_mean)@comps.T` can't leak a nonzero artifact | real — B1 NoData contract |
| `test_group_bands_have_no_duplicate_indices` | `GROUP_BANDS` indices are unique and cover `0..N_EXTRA_BANDS_DEM-1` — `band_norm_mode` resolves by index, so a collision would silently mis-normalize | real — §9 SSoT |
| `test_dem_slope_is_in_ground_degrees` | 1 m rise per pixel at 1 m/px → exactly 45° | real — terrain math |
| `test_dem_slope_scales_with_ground_scale_not_map_scale` | Doubling ground m/px halves `tan(slope)` | **real — regression for the Web-Mercator map-unit bug** (2.0–4.1× latitude-dependent slope error) |
| `test_dem_curvature_sign_and_planar_zero` | Curvature ≈ 0 on a plane; > 0 everywhere in a concave bowl (slump-floor sign convention) | real |
| `test_dem_relative_elevation_zero_on_uniform_slope_center` | On a plane, elevation − own focal mean ≈ 0 at tile centre | real |
| `test_dem_derivatives_shape_and_bands` | Returns exactly `DEM_BAND_IDX`, each `(H,W)` float32 with the halo cropped | real |
| `test_dem_void_stays_nan_and_does_not_poison_the_window` | An ArcticDEM void is NaN in **every** band at that pixel (not displaced by the gradient stencil), while >90 % of the tile stays finite | real — NoData contract |
| `test_nan_uniform_filter_matches_plain_mean_without_nans` | NaN-aware box mean == `scipy.uniform_filter` when there are no NaNs | real |
| `test_ground_scale_shrinks_with_latitude` | `ground_scale_m` == map scale × cos(lat) at 60°/74° N | real |
| `test_coarse_grid_pads_by_the_relev_radius_in_ground_metres` | Coarse grid is square, centred, and pads ≥ `DEM_RELEV_RADIUS_M` of *ground* so the focal window is never edge-extended | real |
| `test_halo_grid_grows_symmetrically` | Halo grid grows `2*DEM_HALO_PX` per axis with the origin shifted correctly, same CRS/pixel size | real |

### [test_generate_extra_tiles.py](test_generate_extra_tiles.py)

Covers the CSV-bbox footprint source added for the 2025 inference EXTRA handoff (doc §6.5) — pure logic only; the GEE fetch is not exercised.

| Test | Checks | Strictness |
|------|--------|------------|
| `test_load_ids_and_bounds_inference_schema` | `tile_id,minx,miny,maxx,maxy` CSV → ids + correct `{id: bounds}` map | real |
| `test_load_ids_training_schema_has_no_bounds` | Legacy `Tile_ID` CSV (no bbox cols) → ids only, bounds `None` (falls back to `--rgb-dir`) | real |
| `test_profile_from_bounds_coregisters` | Profile is EPSG:3857, 512², 8-band float32; transform maps pixel (0,0)→(minx,maxy) and (512,512)→(maxx,miny) | real — co-registration contract |
| `test_write_bands_creates_then_resumes` | First write creates 8-band NaN stack + fills NDVI; tile "done" for `--groups s2` only once {0,1,6,7} all non-NaN | real — resumability |
| `test_dem_sidecar_is_12_bands_and_leaves_canonical_alone` | `--groups dem` writes a 12-band tile with 8–11 filled and 0–7 NaN, while an 8-band canonical tile stays "complete" at its own count (else a DEM run would mark all 22,259 EXTRA tiles stale) | real — sidecar contract |
| `test_needs_work_flags_wrong_band_count` | An 8-band file can't satisfy a 12-band request — the reason DEM goes to a sidecar instead of being appended in place | real |
| `test_read_band_roundtrips_ndvi` | `--copy-ndvi-from` reads band 0 verbatim, so sidecar NDVI is bit-identical to canonical (no GEE re-query, no drift vs the comparator) | real — comparability |

### [test_export_s2_composites.py](test_export_s2_composites.py)

Covers the bulk S2 export grid/domain geometry (doc §3); EE + GCS not exercised.

| Test | Checks | Strictness |
|------|--------|------------|
| `test_latlon_grid_aligns_and_covers` | `latlon_grid` cells are origin-aligned `dlon×dlat` and cover the bbox corners | real |
| `test_cell_id_deterministic_and_sign_safe` | `cell_id` stable + sign-safe (`W1500_N0740`, `E0000_S0025`); distinct corners → distinct ids | real |
| `test_domain_cells_keeps_only_intersecting` | Cells filtered to those intersecting the (reprojected) domain polygon; clip ⊆ domain ∩ cell; far cell excluded | real |
| `test_queue_full_error_is_recognised` | the per-user ceiling arrives as a message, not a typed exception | real |
| `test_submit_with_backoff_waits_out_a_full_queue` | a full queue is a wait, not a failure | real — three years' launches died at 3,002 tasks because the per-project slot check cannot see the per-user ceiling |
| `test_submit_with_backoff_reraises_other_errors` | a real export bug surfaces at once | real — otherwise it would be retried for a day |
| `test_submit_with_backoff_gives_up_eventually` | a queue that never drains is not waited on forever | real |

### [test_sampler.py](test_sampler.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_parse_schedule_sorted` | Schedule dict parses and sorts by epoch range | real |
| `test_ratio_for_epoch` | Epoch bucket lookup + clamp-to-last on out-of-range | real |
| `test_sampler_ratio_1_1` | batch=8, ratio 1:1 → exactly 4 pos / 4 neg | real |
| `test_sampler_ratio_1_5_shifts_distribution` | batch=12, ratio 1:5 → exactly 2 pos / 10 neg | real |
| `test_sampler_determinism_across_epochs` | Same seed+epoch → identical sequence; different epoch → different sequence | real — reproducibility lock |
| `test_sampler_requires_both_classes` | Zero negatives → `ValueError("both classes")` | real |
| `test_filter_train_positive_subset_keeps_negatives_intact` | `_filter_train_positive_subset` (in `scripts/train.py`): subset_pct=25 keeps 25% of positives, all negatives untouched | real — Phase 0 §3.2 + Phase 2 §5.1 contract |
| `test_filter_train_positive_subset_is_deterministic` | Two invocations with the same input give the same output (seed=42 hard-coded) | real — reproducibility |
| `test_filter_train_positive_subset_full_pct_no_op` | subset_pct=100 keeps every tile | shallow — boundary case |

### [test_config.py](test_config.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_load_config_without_base_unchanged` | no `base:` key → behavior identical to before | shallow |
| `test_base_merge_nested_override` | `base: baseline.yaml` inherits all keys; overrides win at any nesting depth; sibling keys preserved; `base` key consumed | real — config-inheritance contract |
| `test_base_merge_lists_replace_not_concat` | lists replace wholesale (no concat surprises) | real |
| `test_missing_base_raises` | dangling base path → FileNotFoundError naming both files | real |
| `test_chained_base_rejected` | base-of-base → ValueError (one level only, by design) | real |
| `test_deep_merge_does_not_mutate_inputs` | merge is pure | real |
| `test_validate_accepts_only_known_top_level_keys` | a valid training cfg passes `validate_training_config` | real |
| `test_validate_rejects_top_level_early_stopping` | the recurring bug — top-level `early_stopping:` (silently ignored by train.py) → ValueError naming `training.early_stopping` | real — guards a known GPU-h-wasting foot-gun |
| `test_validate_rejects_unknown_key_and_lists_all_stray` | every stray top-level key is listed in the error | real |
| `test_validate_schema_matches_base_recipe_keys` | canonical `base_v2_fast.yaml` validates cleanly; its keys ⊆ allow-list (schema stays in sync with the recipe) | real |
| `test_no_new_configs_inherit_the_frozen_phase0c_schedule` | exactly 68 configs declare `base: phase0c_seed42.yaml` — a new one would inherit the FROZEN `start_epoch: 101` floor | real — guards the recurring stop-schedule regression |
| `test_vectorize_min_blob_px_prefers_new_key` | `vectorize_min_blob_px` is read when present | shallow |
| `test_vectorize_min_blob_px_reads_legacy_key_with_warning` | deployment packages already on GCS carry the pre-2026-08-12 `min_blob_size_px`; still resolves, and warns | real — back-compat contract for shipped packages |
| `test_vectorize_min_blob_px_new_key_wins_over_legacy` | new key takes precedence when both are present | real |
| `test_vectorize_min_blob_px_default_when_absent` | neither key → caller's default | shallow |
| `test_deployment_yaml_uses_the_renamed_key` | `configs/deployment.yaml` carries `vectorize_min_blob_px` and no longer the eval-colliding `min_blob_size_px` | real — guards the name split that caused the 10/80/2000 manuscript confusion |

### [test_dataset.py](test_dataset.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_parse_extra_spec_empty` | `None` and `[]` both → `[]` | shallow |
| `test_parse_extra_spec_flexible_names` | Arbitrary names parsed, band indices preserved | real — flexible-EXTRA guarantee |
| `test_parse_extra_spec_rejects_missing_keys` | Missing `name` or `band` → `ValueError` | real |
| `test_dataset_rejects_soft_labels` | `boundary_handling="soft_labels"` raises `NotImplementedError` (deferred to a later iteration, training.md §5.5) | real — guards a config option that isn't wired to code |
| `test_dataset_rejects_unknown_boundary_handling` | Unknown value (e.g. `"bogus"`) raises `ValueError` | real |
| `test_dataset_rgb_only` | `(3, 64, 64) float32` image, `(64, 64) int64` label, str `tile_id`; negative tiles return synthetic all-zero label (no label file needed) | real — end-to-end plumbing |
| `test_dataset_with_variable_extra` | Bands [0, 2] + arbitrary names → `(5, 64, 64)` | real — flexible-EXTRA end-to-end |
| `test_dataset_label_values_in_set` | Every label's unique values ⊂ {0, 1, 255} | real |
| `test_mixing_is_seed_reproducible` | With copy-paste `p=1`: same `(seed, idx)` → identical `__getitem__` output; a different `seed` changes it (proves the mixing RNG is seeded, not fixed) | real — B2 reproducibility lock |
| `test_boundary_dilation_adds_ignore` | Width=2 dilation creates 255 band and preserves interior 1s | real |
| `test_substitute_nodata_all_band_zero_becomes_ignore_and_mean` | §4.4: all-band-zero pixel → label 255 + per-channel mean; single-band dropout → mean substitution only (label kept); non-zero untouched | real — pure-function NoData logic |
| `test_substitute_nodata_noop_when_no_zeros` | No zeros → rgb and label returned unchanged | real |
| `test_init_raises_on_rgb_channel_name_mismatch` | RTSDataset refuses stats with permuted RGB channel names (training.md §4.5) | real — Critical C1 (2026-05-02) |
| `test_init_raises_on_extra_channel_name_mismatch` | RTSDataset refuses stats with mis-ordered EXTRA channel names | real — Critical C1 (2026-05-02) |
| `test_read_with_retry_recovers_from_transient_failure` | Transient GCS/VSI read error is retried; eventual success returned (no real backoff) | real — guards against single transient read crashing a multi-hour run (2026-06-04) |
| `test_read_with_retry_raises_after_exhausting_attempts` | Persistently corrupt tile fails all 4 attempts and raises `RuntimeError` naming the tile id | real — corrupt tiles surface loudly, not silently |
| `test_dataset_per_tile_data_root_column` | Multiscale POC: a `data_root` metadata column routes a tile's RGB/EXTRA/label paths to its own root; tiles without it fall back to the ctor `data_root`; both load end-to-end | real — multi-root plumbing (2026-07-02) |

### [test_mixing.py](test_mixing.py)

Sample-mixing augs (`data/mixing.py`, family F). Pure array ops; synthetic tiles + fake sampler.
The sampler callback is `sample_fn(positive_only, rng)` — the rng is threaded through so source-tile
selection is reproducible under the run seed (B2).

| Test | Checks | Strictness |
|---|---|---|
| `test_copy_paste_adds_positives_and_preserves_dtype` | copy-paste pastes an instance: positives appear, shape/dtype preserved | real — the rare-object lever |
| `test_copy_paste_no_source_instances_is_identity` | negative source ⇒ identity (nothing to paste) | real — edge case |
| `test_copy_paste_preserves_ignore_pixels` | pasted positives never overwrite ignore(255) | real — label-integrity guard |
| `test_copy_paste_rgb_only_path` | works with `extra=None` | real — RGB-only compat |
| `test_mosaic_output_shape_and_density` | mosaic returns tile_size, valid labels, positives present | real |
| `test_cutmix_swaps_patch` | source patch (incl. positives) enters target | real |
| `test_mixup_blends_and_unions_labels` | pixel blend + label union of both positive blocks | real |
| `test_augmenter_off_by_default_is_identity` | no `mixing` config ⇒ bit-identical passthrough (default-off guarantee) | real — protects existing runs |
| `test_augmenter_copy_paste_p1_fires` | `copy_paste.p=1` ⇒ op fires (positives added) | real — dispatch guard |

### [test_transforms.py](test_transforms.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_color_aug_does_not_touch_extra` | EXTRA channels are bit-identical after color-only augmentation (training.md §9.2) | real — Critical C3 (2026-05-02) |
| `test_geometric_aug_applies_to_extra_and_mask` | HorizontalFlip applies to RGB, EXTRA, and mask jointly | real |
| `test_extra_none_path_still_works` | RGB-only call path (no `extra` kwarg) preserved through the split | real — backward-compat for baseline RGB-only |
| `test_pad_mask_ignore_default_is_background` | Default RandomScale pad border in the mask is background (0) — documents the baked-in baseline | real — Stage 3B A/B control |
| `test_pad_mask_ignore_true_labels_border_ignore` | `multi_scale.pad_mask_ignore: true` labels the pad border ignore (255), not background | real — Stage 3B pad-fix (albumentations 2.x `fill_mask`) |
| `test_auto_policy_default_none_is_handtuned` | No `auto_policy` (and explicit-null) ⇒ color stage is **exactly** the baseline op list `[RandomBrightnessContrast, HueSaturationValue, GaussNoise, CLAHE]` (structural bit-identity), EXTRA untouched | real — the locked-baseline contract |
| `test_trivialaugment_runs_preserves_shape_and_mask` | `mode=trivialaugment` → color stage is exactly `[OneOf]`; runs; shape/dtype preserved; mask + EXTRA untouched | real — family-F auto-policy contract |
| `test_randaugment_runs_with_num_ops` | `mode=randaugment, num_ops=2` → color stage is exactly `[SomeOf]`; runs; mask untouched | real — family-F auto-policy contract |
| `test_auto_policy_pool_excludes_shadow_scramblers` | op pool omits solarize/invert/posterize/equalize/channel-shuffle/grayscale (shadow-cue safety) | real — the RTS shadow-safety guard |
| `test_auto_policy_invalid_mode_raises` | unknown `auto_policy.mode` → `ValueError` | shallow |
| `test_annealed_magnitude_schedule` | `_annealed_magnitude` = start at ep≤1, end at ep≥end_epoch, monotone-decreasing between | real — anneal schedule contract |
| `test_set_epoch_noop_without_anneal` | no `anneal` block ⇒ `set_epoch` leaves the auto-policy magnitude + structure unchanged (off-by-default) | real — identity guard |
| `test_set_epoch_anneals_magnitude` | `anneal` block ⇒ `set_epoch(1)` magnitude > `set_epoch(end_epoch)`; exact start/end values; stage still runs | real — family-F annealing contract |

### [test_train_helpers.py](test_train_helpers.py)

Module-level helpers in `scripts/train.py` (CPU, no training loop). Added 2026-06-22.

| Test | Checks | Strictness |
|---|---|---|
| `test_deploy_state_dict_live_weights_when_no_ema` | `_deploy_state_dict(model, ema=None)` → live weights (the freeze-phase / permanently-frozen-probe path that would otherwise never write a deployment checkpoint) | real — guards the frozen-encoder checkpoint fix |
| `test_deploy_state_dict_uses_ema_and_restores_model` | with EMA present → captures swapped-in EMA weights and restores the live model after | real — EMA-path superset guarantee |

### [test_models.py](test_models.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_build_model_rgb_only_output_shape` | `(B, 1, 512, 512)` from UNet++/EffB5 on RGB-only config | real |
| `test_build_model_with_extra_channels` | 7-channel (RGB+4 EXTRA) forward pass returns correct shape | real — flexible-EXTRA in models |
| `test_output_bias_initialized_to_class_prior` | Final-conv bias at prior=0.5 equals 0.0 | real — focal-paper init |
| `test_output_bias_for_imbalanced_prior` | prior=0.01 → bias ≈ -log(99) | real |
| `test_output_is_logits_not_probabilities` | Random-input outputs span beyond [0, 1] | real — logits contract |
| `test_invalid_bias_prior_rejected` | Prior outside (0, 1) → `ValueError` | shallow |
| `test_build_model_segformer_output_shape_and_bias` | SegFormer (mit_b5) builds, returns (B,1,H,W) logits at input res, and the class prior flows through its `.segmentation_head[0]` bias (fair arch comparison) | real — guards the new architecture branch (2026-06-06) |
| `test_build_model_smp_decoder_sweep` | §8.2 arch sweep: each smp decoder (`deeplabv3plus`/`fpn`/`pspnet`/`manet`) builds on EffB5, returns (B,1,H,W) logits, shares the `.segmentation_head[0]` bias path (parametrized) | real — guards the decoder-sweep branch (2026-06-15) |
| `test_unknown_architecture_rejected` | Unsupported arch (`bogusnet`) → clear `ValueError` (`segformer` is now supported) | shallow |
| `test_fusion_default_and_ensemble_are_plain_models` | `model.fusion` absent (=early) and `ensemble` both build a normal single-encoder model (F4 averaging is eval-side) | real — fusion default + back-compat (2026-06-18) |
| `test_fusion_stem_init_zeroes_extra_input_channels` | F1: encoder stem-conv weights zero on EXTRA channels (≥3), nonzero RGB | real — guards the F1 init |
| `test_fusion_stem_init_invariant_to_extra_at_init` | F1: at init, changing only EXTRA channels leaves the output unchanged (epoch-0 == RGB-only) | real — the F1 contract |
| `test_fusion_chan_attn_shape_gate_and_delegation` | F2: wrapper returns (B,1,H,W) logits, delegates `.encoder`/`.segmentation_head` (bias-init flows through), gate per-channel in (0,1) | real — guards F2 + freeze/bias compat |
| `test_fusion_chan_attn_param_groups_split` | F2: gate params land in the non-encoder (decoder) group so freeze schedule + backbone-LR target correctly | real — F2 × freeze.py integration |
| `test_fusion_unknown_rejected` | Unsupported `model.fusion` → clear `ValueError` | shallow |
| `test_heavy_fusion_shape_delegation_and_uses_extra` | F3/F5: (B,1,H,W) logits, delegates `.encoder`/`.segmentation_head`, has 2nd `.extra_encoder`, output depends on EXTRA channels | real — guards F3/F5 dual-encoder wiring |
| `test_heavy_fusion_rejects_rgb_only` | F3/F5 with no EXTRA channels → `ValueError` | shallow — guard |
| `test_build_model_foundation_rgb` | `arch='foundation'` (DINOv3 ViT) builds, (B,1,H,W) logits, class-prior bias flows through `.segmentation_head[0]` | real — guards the FM branch (2026-06-18) |

### [test_foundation.py](test_foundation.py)

Forward-path tests for `models/foundation.py` (FoundationSegmenter: DINOv3/ViT encoder → simple feature pyramid → FPN decoder → logits). CPU, `pretrained=False`. Added 2026-06-18 (second-wave Step 4).

| Test | Asserts | Strictness |
|---|---|---|
| `test_foundation_forward_shape` | ViT `forward_intermediates` → pyramid → decoder → `(B,1,H,W)` at input res | real — the core ViT→dense chain |
| `test_foundation_taps_four_blocks_incl_deepest` | 4 evenly-spaced block taps, deepest included, sorted | real — pyramid-diversity contract |
| `test_foundation_exposes_encoder_and_head` | `.encoder` (freeze/LLRD) + `.segmentation_head[0]` bias-init compatible | real — integration hooks |
| `test_foundation_output_is_logits` | random-input outputs span beyond [0,1] | shallow — logits contract |
| `test_foundation_extra_channels_forward_shape` | RGB+EXTRA (in_channels=4): patch-embed widened to 4, forward → `(B,1,H,W)` | real — guards the DINOv3+EXTRA adapter |
| `test_foundation_extra_channels_zero_init_is_rgb_only_at_init` | EXTRA channels zero-init ⇒ epoch-0 invariant to EXTRA (fair F1-style start) | real — fairness/init guarantee |
| `test_sam2_hierarchical_forward_shape` | SAM2/Hiera (`sam2_hiera_tiny`): native {/4,/8,/16,/32} pyramid → 1×1 proj (4) → FPN → `(B,1,H,W)` | real — guards the hierarchical foundation branch (2026-06-22) |
| `test_sam2_exposes_encoder_and_head` | `.encoder.parameters()` (LP-FT/freeze) + `.segmentation_head[0]` bias-init compatible | real — integration hooks for the no-LLRD path |
| `test_sam2_rejects_extra_channels` | SAM2/Hiera path is RGB-only (stem not exposed) → `in_channels≠3` raises `NotImplementedError` | shallow — guard |
| `test_sam2_features_only_wrapper_exposes_inner_model` | timm's `features_only` wrapper DOES expose `.model` → HieraDet with `.blocks` (16) and `.patch_embed` | real — pins the reach-through that disproves the "no LLRD / no EXTRA possible" premise `fm_sam2_rgb` was built on (2026-07-28) |
| `test_sam2_pretrained_weights_are_actually_loaded` | `pretrained=True` changes **every** encoder tensor vs random init (audited: 202/202, 33,947,328 params) | real — closes the "nothing ever verified SAM2 weights load" gap; skipped unless the weights are already in the local HF cache, so the module's no-download contract holds (2026-07-28) |
| `test_dinov3_sat_vitl_forward_shape` | Satellite DINOv3 ViT-L (`vit_large_patch16_dinov3.sat493m`) builds via the isotropic-ViT path → `(B,1,H,W)`; non-hierarchical + has `.blocks` (LP-FT/LLRD apply) | real — guards the satellite-DINOv3 branch (2026-06-22) |

### [test_losses.py](test_losses.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_focal_loss_matches_hand_computed` | logit=0, γ=2, α=0.25 → FL = 0.25²·ln2 | real — reference value |
| `test_focal_loss_zero_at_perfect_prediction` | logit=30 on positive → ≈ 0 | real |
| `test_focal_loss_ignore_mask_respected` | ignore=255 pixels don't contribute to mean | real — ignore contract |
| `test_focal_loss_finite_gradient_at_extreme_logits[±30, y∈{0,1}]` | Finite gradient across logit range | real — numerical stability |
| `test_dice_loss_perfect_prediction_near_zero` | Confident correct → dice ≈ 1, loss ≈ 0 | real |
| `test_dice_loss_empty_mask_stable` | All-negative tile with eps > 0 → finite loss | real — edge case |
| `test_tversky_reduces_to_dice_at_half_half` | Tversky(0.5, 0.5, ε) == Dice(2ε) algebraic identity | real — generalization check |
| `test_tversky_beta_greater_alpha_penalizes_fps_more` | β>α loss > α>β loss on FP-heavy input | real |
| `test_compound_loss_weighted_sum` | Compound equals λ_f·focal + λ_d·dice | real |
| `test_build_loss_dispatch[focal|dice|tversky|compound]` | Dispatcher returns the right class per config | shallow |
| `test_build_loss_unknown_raises` | Unknown name → `ValueError` | shallow |

### [test_ema.py](test_ema.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_ema_init_matches_model_state` | Shadow is a clone of initial params | shallow |
| `test_ema_update_converges_to_target` | With constant target, shadow converges | real — update math |
| `test_ema_swap_in_restores_live_weights` | Context manager puts EMA in, then puts live back | real — swap contract |
| `test_ema_swap_in_restores_on_exception` | Live restored even if caller raises | real — error path |
| `test_ema_state_dict_roundtrip` | save → fresh instance → load reproduces shadow | real — resumption |

### [test_scheduler.py](test_scheduler.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_phase1_holds_frozen_lr_on_both_groups` | Phase 1 LR = frozen_lr for all groups | real |
| `test_phase2_decoder_linear_warmup` | Decoder LR linearly ramps over warmup_epochs | real |
| `test_phase2_backbone_linear_warmup_shorter` | Backbone independent short warmup, then plateau until decoder joins | real — plan risk #17 |
| `test_cosine_anneal_reaches_min_lr_at_max_epoch` | LR at max_epochs = min_lr | real |
| `test_cosine_lr_between_peak_and_min_during_decay` | Interior cosine LR strictly in (min_lr, base_lr), monotone decreasing | real |
| `test_cosine_exact_halfway_at_t_over_tmax_0p5` | Mid-cosine LR brackets (base_lr + min_lr)/2 | real |
| `test_phase1_epoch_zero_handled_safely` | epoch=0 treated as Phase 1, no crash | shallow |
| `test_lr_range_test_endpoints_and_log_midpoint` | lr_range_test: step 0 → lr_min, last step → lr_max, midpoint → geometric mean | real — Phase 0 §3.2 implementation |
| `test_phase2_backbone_lr_scale_applied` | LLRD: a backbone group's per-epoch LR is multiplied by its `lr_scale`; groups without it (decoder/legacy) unaffected | real — §8.2a LLRD × scheduler (2026-06-18) |
| `test_lr_range_test_applies_same_lr_to_all_groups` | All param groups receive the same LR under range-test mode | real |
| `test_lr_range_test_rejects_invalid_bounds` | lr_min ≥ lr_max → `ValueError` | shallow — guard |
| `test_unknown_scheduler_raises` | Unknown `scheduler:` value → `ValueError` | shallow — dispatch guard |
| `test_decoder_phase2_start_epoch_defaults_to_freeze_epochs` | Omitting `decoder_phase2_start_epoch` reproduces the pre-existing flat-frozen behavior exactly | real — backward-compat guard |
| `test_decoder_anneals_early_while_backbone_stays_permanently_frozen` | With `freeze_backbone_epochs≥max_epochs` (permanent freeze, e.g. a 7B ViT), setting `decoder_phase2_start_epoch` lets the decoder warmup/anneal on its own early timeline while the backbone group stays pinned at frozen_lr | real — fixes the fm_dinov3sat_7b_frozen flat-LR collapse (2026-07-10) |

### [test_early_stopping.py](test_early_stopping.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_monotone_increase_always_improving` | Strict monotone gain → no-improve counter stays 0 | real |
| `test_plateau_triggers_stop_after_patience` | Flat smoothed metric past patience → stopped=True | real |
| `test_start_epoch_gates_stopping_but_not_best_tracking` | Stop suppressed pre-start_epoch; best still tracked | real — plan risk #5 |
| `test_min_delta_ignores_noise` | Gains below min_delta don't reset counter | real |
| `test_missing_metric_key_raises` | Metric name absent from dict → `KeyError` | real |
| `test_state_dict_restores_observation_state` | save/load reproduces history + best + counters | real |
| `test_load_state_dict_keeps_config_hyperparameters` | Resume takes patience/min_delta/start_epoch from **config**, not the checkpoint | real — regression, v2.1 gate resume silently kept a stale start_epoch=101 |
| `test_lowered_start_epoch_takes_effect_after_resume` | End-to-end: a resumed run stops on the new, lower start_epoch | real — regression |

### [test_checkpoint.py](test_checkpoint.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_save_deployment_contains_contracted_fields` | best_deployment.pth has model_state_dict, channel_names, git_sha, trained_with, etc. (no separate stats hash; channel-name binding is the integrity guarantee per training.md §4.5) | real — training.md §4.3 |
| `test_save_resume_contains_full_state` | resume_latest-*.pth carries live+ema+optimizer+scheduler+scaler+epoch+es+rng | real |
| `test_save_resume_omits_encoder_when_frozen` | `encoder.*` keys dropped from `live_state_dict` when the whole encoder is frozen (untouched pretrained weights — avoids re-serializing a multi-billion-param encoder every rotation) | real — 2026-07-10 disk-exhaustion fix |
| `test_save_resume_keeps_encoder_when_trainable` | Once unfrozen (diverged from pretrained), `encoder.*` keys are saved in full | real — regression guard |
| `test_resume_rotation_keeps_last_n` | Beyond keep_last_n=2, only newest 2 snapshots survive | real |
| `test_update_best_tracks_smoothed_monotone` | update_best returns True only on strict improvement | real |

### [test_metrics.py](test_metrics.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_filter_small_blobs_drops_undersized` | min_size=4 drops 1-px speckle | real — plan risk for object FPs |
| `test_filter_small_blobs_passthrough_when_min_leq_one` | min_size=1 preserves input | shallow |
| `test_match_objects_empty_both` | (0, 0) preds vs GT → (0, 0, 0) | real — edge case |
| `test_match_objects_empty_pred_positive_tile` | Empty pred, 2 GT → FN=2 | real — plan §6.2 edge case |
| `test_match_objects_empty_gt_negative_tile` | 3 preds, empty GT → FP=3 | real |
| `test_match_objects_greedy_confidence_sort` | Higher-conf prediction wins the GT | real |
| `test_detail_tp_fp_fn_parity_with_match_objects` | `_object_match_detail` (report-only scorecard) tp/fp/fn bit-identical to `_match_objects` (frozen gate path) | real — drift guard |
| `test_detail_clean_one_to_one_no_split_no_merge` | Clean 1↦1 → n_splits=0, n_merges=0, matched IoU=1.0 | real |
| `test_detail_split_one_gt_two_preds` | One GT, two pred blobs → n_splits=1 (over-segmentation) | real |
| `test_detail_merge_two_gt_one_pred` | Two GT, one pred blob → n_merges=1 (under-segmentation) | real |
| `test_accumulator_perfect_prediction_pixel_iou_one` | Exact match → all metrics 1.0 | real |
| `test_accumulator_ignore_index_masks_pixels` | ignore pixels contribute nothing to TP/FP/FN | real — ignore contract |
| `test_accumulator_speckle_fp_filtered` | 1-px FP below min_blob_size doesn't count | real |
| `test_accumulator_pr_auc_ranges_between_zero_and_one` | PR-AUC in [0, 1]; geomean equals single-ratio value | real |
| `test_accumulator_no_positive_tiles_produces_zero_pr_auc` | No-positive val → PR-AUC=0.0 gracefully | real — edge case |
| `test_bootstrap_off_by_default` | No `bootstrap_ratios` → no boot keys emitted (default frozen) | real — Stage 0.2 guard |
| `test_bootstrap_emits_mean_and_ci_within_range` | Enabled → per-ratio mean/lo/hi in [0,1], lo ≤ mean ≤ hi | real — Stage 0.2 bootstrap readout |
| `test_bootstrap_does_not_change_gate_metric` | Enabling bootstrap leaves the gate geomean bit-identical (separate RNG) | real — eval-freeze guarantee |

### [test_object_scorecard.py](test_object_scorecard.py)

Object-level scorecard + applicability probes (Phase 0 of the v3 object-improvement plan). All report-only / synthetic; modules import `training.metrics` → torch (test dep). `make_invisible_contact_sheet` imports matplotlib lazily (in `render`), so `find_invisible_objects` is testable without it.

| Test | Checks | Strictness |
|---|---|---|
| `test_detail_counts_match_object_counts_and_flag_split` | `object_detail_counts` obj tp/fp/fn == `object_counts`; 1-GT/2-pred tile → n_splits≥1 | real — parity + split |
| `test_detail_counts_merge` | One pred spanning two GTs → n_merges≥1 | real — merge |
| `test_bootstrap_point_and_ci_ordering` | Tile-cluster bootstrap: point P/R match counts; lo ≤ point ≤ hi in [0,1] | real |
| `test_bootstrap_empty_region_is_none` | Empty region → None point/CI (not spurious 0) | real — edge case |
| `test_bootstrap_deterministic` | Same seed → identical CIs | real — reproducibility |
| `test_geometry_summary_basic` | Matched-pair IoU median/quantiles | real |
| `test_geometry_summary_empty_none` | No matches → None | shallow |
| `test_build_scorecard_selfcheck_and_signals` | End-to-end: self-check (detail==score_by_region), recall=3/5, precision=3/4, splits+merges flagged, invisible counted, low-sample + per-region invisible_floor | real — instrument integration |
| `test_sampler_caps_dense_regions_keeps_sparse` | Region-stratified sample caps dense regions, keeps sparse, drops none, deterministic + sorted | real |
| `test_sampler_cap_above_sizes_keeps_all` | cap > region sizes → all candidates kept | shallow |
| `test_change_probe_bright_blank_and_excludes_detected` | D2: invisible object above ambient change → bright, flat → blank; detected objects excluded; change_blank_fraction correct | real — change-arm go/no-go |
| `test_change_probe_no_invisible_objects` | No invisible objects → n=0, blank-fraction None | real — edge case |
| `test_seed_noise_stats` | Cross-seed mean/std/spread of aggregate obj metrics | real |
| `test_seed_noise_handles_none_metric` | None metric values dropped before stats | shallow |
| `test_find_invisible_objects_selects_only_below_threshold` | Contact-sheet selection: only max_prob<thr GT objects, with correct area + bbox | real — D1 audit selection |

### [test_score_product_rule.py](test_score_product_rule.py)

Scoring of the **shipped adaptive product rule** (`scripts/score_product_rule.py`) on a frozen prediction cache — the rule the delivered South map actually uses (0.30 contour → per-polygon `max_prob` → `conf_class`/`rts_class`), as opposed to the fixed-min-size anchors of ledger J/K. Synthetic 32×32 tiles, report-only, no GPU/GCS; imports `training.metrics` → torch (test dep). The real-cache parity gate against ledger J/K is a Tier-2 check (Coverage gaps #8).

| Test | Checks | Strictness |
|---|---|---|
| `test_cut_points_match_the_product_arithmetic` | `vector_cut_u8(0.30)==75` (round, as `_polygonize_block`); `tier_cut_u8(0.65)==163` (ceil, since `conf_class` tests the *decoded* `u8/250`); 162/250 < 0.65 ≤ 163/250 | real — the 1/250 grid arithmetic |
| `test_tier_boundary_is_on_the_quantised_grid[4 cases]` | conf_class decided by the **quantised** max, not the raw float: 0.6499→u8 162→medium (excluded), 0.652→u8 163→high, 0.99→high, 0.50→medium | real — the core subtlety; guards the documented t65-vs-high_confidence mismatch |
| `test_floor_drops_one_pixel_blobs_and_keeps_two` | `vectorize_region`'s `max(2, …)` technical floor: 1-px blob dropped, 2-px kept | real |
| `test_floor_is_applied_before_the_tier_test` | Ordering: a lone 0.99 pixel must NOT become a high_confidence object (floor in `_polygonize_block` precedes banding in `export_south_products`) | real — ordering guard |
| `test_high_tier_is_a_subset_and_geometry_row_preserves_identity` | Medium-only blob in candidates but not high; `geom065` keeps the same object count with a smaller outline (36 px skirt → 4 px core) | real — row-relationship invariant |
| `test_edge_touching_counts_border_blobs` | `n_pred_edge` counts only border-touching blobs (measures the un-replicable seam-dissolve population) | real |
| `test_ignore_regions_are_excluded_from_predictions` | 255-ignore suppresses predictions, matching the J/K `valid` convention | real |
| `test_anchor_tile_counts_match_object_counts` | Anchor path obj tp/fp/fn bit-identical to `object_counts`; pixel counts intentionally post-filter (≤ pre-filter) and coincide at `min_blob=1` | real — parity with the ledger machinery |
| `test_aggregate_row_sums_counts_and_flags_low_sample` | Count roll-up, P/R, `n_pred_objects`, geometry; None recall for 0-GT regions; `low_sample` boundary at exactly 5 GT | real |
| `test_load_product_constants_reads_the_shipped_rule` | `ast` reader returns `TIER_BOUNDS==(0.45,0.65)` / `CANDIDATE_MAX_AREA_M2==500.0` from `export_south_products.py` without importing it (geopandas absent from the image) | real — SSoT guard |
| `test_load_product_constants_raises_on_missing_name` | Missing constant → `KeyError`, never a silent local fallback | real — SSoT guard |

### [test_gt_mmu_scoring.py](test_gt_mmu_scoring.py)

Minimum Mapping Unit (sub-MMU positive → ignore) *metric/loss semantics* — validates the data-v1.1 fix composing through the existing 255-ignore machinery (the primitive itself is covered by `test_label_cleaning.py`). Synthetic, CPU-only; imports `training.metrics` → torch + `losses` (test deps).

| Test | Checks | Strictness |
|---|---|---|
| `test_apply_min_mapping_unit_off_is_identity` | `mmu_px≤1` returns the input object unchanged (reproducibility-preserving default) | real — guards the off-default |
| `test_sub_mmu_gt_plus_correct_pred_zero_fp_zero_fn` | sub-MMU GT + correct pred: without floor → 1 FN; with floor → (0,0,0) — the core ignore assertion | real — core semantics |
| `test_real_object_survives_mmu` | ≥3000px GT + matching pred → 1 TP unchanged at mmu=600 (no over-exclusion) | real |
| `test_straddle_pred_still_fp_on_background` | Pred straddling a sub-MMU sliver + genuine background → exactly 1 FP (background FP not masked away) | real — FP-carve-out edge case |
| `test_scorecard_self_check_holds_with_mmu` | Clean labels once → `build_scorecard` parity self-check stays True (all 3 paths see identical labels) | real — parity regression guard |
| `test_pixel_metrics_and_loss_ignore_sub_mmu` | Real-object pixel counts unmoved; sliver's 4px leave the FN count; focal loss invariant to logits under the 255 sliver | real — pixel + loss ignore |

### [test_freeze.py](test_freeze.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_freeze_backbone_disables_grad_on_encoder_only` | Encoder params requires_grad=False; decoder untouched | real |
| `test_unfreeze_backbone_restores_grad` | After unfreeze, all encoder params trainable again | real |
| `test_build_param_groups_partitions_by_id` | Every model param appears in exactly one named group | real |
| `test_build_param_groups_lrs_set` | Decoder/backbone LRs + weight_decay set as requested | shallow |
| `test_optimizer_respects_frozen_encoder` | After freeze + step, encoder weights unchanged | real — integration check |
| `test_build_llrd_param_groups_decay_and_coverage` | LLRD (§8.2a): per-layer `lr_scale` increases stem→top (top=1.0), decoder group=1.0, every model param covered exactly once | real — guards LLRD grouping (2026-06-18) |
| `test_build_llrd_rejects_bad_decay` | `llrd_decay` outside (0,1] → `ValueError` | shallow |

### [test_mlflow_utils.py](test_mlflow_utils.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_flatten_params_nested_dict` | Dotted-key flatten handles dict + list + scalar | real |
| `test_flatten_params_truncates_long_values` | Values > 500 chars truncated (MLflow limit) | real |
| `test_config_sha_deterministic_and_order_independent` | Same config (any key order) → same SHA | real |
| `test_config_sha_changes_on_value_difference` | Value delta produces different SHA | real |

### [test_visualizations.py](test_visualizations.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_prediction_preview_grid_writes_png` | Grid writes a non-empty PNG | shallow |
| `test_pr_curves_at_ratios_handles_zero_positives` | Ratio with all-negative labels doesn't crash | real — edge case |
| `test_probability_histogram_log_scale_safe` | All-zero input doesn't break log-scale y | real |
| `test_confusion_matrix_pixel_subsampled` | Scaled counts + writes PNG | shallow |
| `test_pick_preview_tiles_pass1_partitions_positives_and_negatives` | 3 pos + 3 neg disjoint, from correct TrainClass | real — preview contract |

### [test_package_model.py](test_package_model.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_null_threshold_rejected` | deployment config with threshold=null → `ValueError` | real — plan Step 8 guard |
| `test_null_temperature_rejected` | temperature=null → `ValueError` | real |
| `test_both_null_rejected_together` | Both null → error mentions threshold first | shallow |
| `test_both_set_accepted` | Properly calibrated config passes guard | shallow |
| `test_package_from_rundir_writes_all_files` | no-MLflow run-dir packaging writes all 6 files; run_metadata carries source/seed/git_sha/channel_names/epoch from the checkpoint+config | real — Phase 2 deploy-package path (the 3 ensemble seeds were built this way) |
| `test_package_from_rundir_rejects_uncalibrated` | null threshold/temperature → `ValueError` (guard also on the run-dir path) | real |
| `test_package_from_rundir_missing_input_raises` | absent checkpoint/config/norm-stats → `FileNotFoundError` | shallow |

> **Coverage note:** the run-dir packaging path (`package_model_from_rundir`,
> used to build the 3 deployment packages from `/mnt/outputs/v1.0/runs/<run>/`)
> is now unit-tested end-to-end above. The **MLflow**-run path
> (`package_model`, run resolution + artifact download) is still exercised only
> via `_assert_calibration_complete`; it relies on real MLflow runs at deploy
> time. The deploy seeds did not use it (they are run-dir packaged).

### [test_calibrate.py](test_calibrate.py)

Covers the GPU-free math in `scripts/calibrate.py` (Phase D calibration). The
forward-pass / checkpoint-load paths need GPU + real checkpoints and are
exercised by the live calibration run, not here.

| Test | Checks | Strictness |
|---|---|---|
| `test_logit_sigmoid_roundtrip` | `_sigmoid(_logit(p)) ≈ p` | shallow |
| `test_fit_temperature_recovers_known_scaling` | logits inflated ×2 → fitted T ∈ (1.6, 2.4) | real |
| `test_fit_temperature_bounds` | T stays within the [0.25, 5] search bounds | shallow |
| `test_pr_auc_geomean_separable_is_high` | separable signal → geomean > 0.8; higher ratio ≤ lower | real |
| `test_pr_auc_geomean_empty_returns_zero` | no positive tiles → 0.0 | shallow |
| `test_precision_at_threshold_monotone` | raising the threshold does not lower precision (separable) | real |
| `test_select_threshold_meets_target_when_separable` | finds a threshold with precision ≥ 0.8, `target_met=True` | real |
| `test_select_threshold_falls_back_to_f1_when_unreachable` | unreachable target → max-F1 fallback, `target_met=False` | real |

> **Coverage gap (acknowledged 2026-06-25):** the forward-pass collection
> (`collect_probs`, `load_checkpoint`), TTA selection, and single-vs-ensemble
> orchestration in `main()` are validated only by the live Val-Realistic run
> (`/mnt/outputs/v1.0/calibration/effb5_trivialaug/`), whose fresh PR-AUC
> reproduced each run's training `best_smoothed` (parity check). Not unit-tested.

### [test_inference_pipeline.py](test_inference_pipeline.py)

Inference pipeline (`inference/` + grid/merge entry scripts), GPU-free. Fixtures: synthetic 512px RGBA quads written on the **real zoom-15 mosaic grid** (production grid constants, small rasters), so quad-bounds math is exercised against an observed Planet quad coordinate; plus synthetic 4-band S2 composite COGs (B4,B3,B2,B8 export order) for the EXTRA=NDVI windowed reader.

| Test | Checks | Strictness |
|---|---|---|
| `test_quad_bounds_match_observed_planet_quad` | grid math reproduces real quad 0-1515's GCS bounds | real — grid anchor |
| `test_grid_constants_consistent` | GRID_N × QUAD_SIZE_M = world extent; resolution derivation | shallow |
| `test_load_quad_index_validates_columns` | missing index columns → ValueError | shallow |
| `test_read_tile_interior` | full-quad window read, no NoData | real |
| `test_read_tile_straddles_quads` | tile spanning 2 quads mosaics correctly; alpha=0 → NoData | real — §4.1 quad-straddling |
| `test_read_tile_outside_coverage_is_all_nodata` | no indexed quad → all-NoData | real — §5.3 |
| `test_is_missing_object_distinguishes_gap_from_transient` | "does not exist"/"no such file" → gap (skip); "HTTP 503" → transient (retry) | real — the missing-quad classifier |
| `test_read_tile_missing_quad_degrades_to_nodata` | quad in the index but absent from the bucket → NoData footprint, no crash (the gap that stalled South) | real — §5.3 robustness |
| `test_read_ndvi_missing_cell_degrades_to_nan` | absent S2 cell → NDVI stays NaN, no crash | real — §5.3 robustness |
| `test_dataset_normalizes_and_mean_substitutes` | NoData pixels mean-substituted pre-z-score (normalize to 0); valid pixels z-scored | real — §5.3/§4.4 parity |
| `test_dataset_flags_all_nodata` | all-NoData tile flagged for skip+manifest | real |
| `test_dataset_rejects_missing_columns` | tile-list schema guard | shallow |
| `test_load_s2_index_validates_columns` | missing S2-index columns → ValueError | shallow |
| `test_read_ndvi_tile_value_and_coregistration` | NDVI=(B8−B4)/(B8+B4) from a composite cell, resampled to the tile grid; no-coverage (zeroed) → NaN | real — §5 NDVI reader / Rule 3 |
| `test_read_ndvi_tile_outside_coverage_is_nan` | tile with no intersecting S2 cell → all-NaN | real — §5 |
| `test_dataset_with_ndvi_extra_stacks_and_neutralizes` | RGB+NDVI → 4-ch image; NDVI z-scored via shared `apply_norm`; no-coverage NaN → 0 | real — §5 / Rule 3 EXTRA parity |
| `test_dataset_extra_requires_s2_index` | EXTRA declared but no s2_index → ValueError | shallow |
| `test_dataset_rejects_non_ndvi_extra` | EXTRA other than ndvi → NotImplementedError | shallow |
| `test_tile_grid_counts_and_determinism` | deterministic ids, unique, every tile intersects a quad, tile = 512 px | real — §4.4 |
| `test_tile_grid_aoi_filter` | AOI subset preserves global grid alignment | real |
| `test_tta_inverse_correctness` | flip/rot-equivariant model ⇒ TTA mean == identity pass (exposes wrong inverse) | real — §7.2 |
| `test_temperature_applied_to_logits_before_sigmoid` | sigmoid(logit/T), not T on probabilities | real — §7.3 |
| `test_tta_pass_counts` | none/minimal/standard/full = 1/2/4/8 passes | shallow |
| `test_predict_probs_rejects_unknown_tta` | bad tta config → ValueError | shallow |
| `test_ensemble_single_member_equals_predict_probs` | 1-member ensemble at T reduces to `predict_probs(·, T)` (fused-prob temperature inversion is exact) | real — Phase-D ensemble recipe |
| `test_ensemble_mean_prob_then_temperature_math` | 2 const-logit models → mean of per-model sigmoids, then T on the fused logit (matches deployment.yaml recipe) | real |
| `test_ensemble_identical_members_equal_single` | N identical members == single model | shallow |
| `test_ensemble_empty_raises` | empty model list → ValueError | shallow |
| `test_gcs_package_path_is_staged` | gs:// package dir routes through `_stage_gcs_package`, not the local config loader | real — regression: fleet workers crashed on gs:// packages (2026-07-05 audit) |
| `test_runtime_package_mismatch_aborts` | runtime vs package precision/tta mismatch aborts; null defers | real — §14 calibration-mismatch guard |
| `test_probability_tile_roundtrip` | float32, NoData −1.0, EPSG:3857 roundtrip | real — §9.1 |
| `test_scaled_uint8_roundtrip_precision_and_nodata` | scaled_uint8 (prob×250/NoData 255): on-disk uint8+255, decode within 0.004, NoData preserved | real — §9.1 scaled_uint8 encoding |
| `test_read_probability_tile_reads_float32` | `read_probability_tile` auto-detects float32 encoding (returns as-is) | real — §9.1 decode dispatch |
| `test_merge_decodes_scaled_uint8_tiles` | `merge_tiles` decodes scaled_uint8 COGs → same mean 0.5 + NoData strip as float32 | real — §4.3 merge dtype-agnostic |
| `test_binary_mask_roundtrip` | uint8, NoData 255 roundtrip | real — §9.2 |
| `test_manifest_resume_skips_completed` | restart resumes from inference_log.json; skip reasons + counts kept | real — §8.3 |
| `test_gaussian_weights_peak_center_symmetric` | σ=128 weight grid peaks center, symmetric | shallow |
| `test_merge_weighted_average_and_nodata` | equal-weight overlap → exact mean; NoData strip falls back to valid tile | real — §4.3 |
| `test_merge_ignores_missing_tiles` | absent (skipped) tile rasters don't break the merge | real |
| `test_read_tile_scale05_expands_fov` | scale=0.5 decimated read: 2× ground bbox → same px dims, values survive bilinear | real — §6.2 scale path |
| `test_read_tile_scale05_nodata_stays_crisp` | alpha resampled nearest: NoData boundary exact under decimation | real |
| `test_bbox_index_matches_boolean_mask` | `_BBoxIndex` STRtree hits == boolean-mask rows (interior/straddle/outside) | real — §11.3 spatial hit-test parity |
| `test_read_tile_hits_path_identical_to_mask` | spatial-index `hits=` path byte-identical to full-scan mask path | real — §11.3 cache must not change pixels |
| `test_open_dataset_cache_reuses_handle` | two reads of one quad → `rasterio.open` called once (per-worker LRU) | real — §11.3 quad-cache |
| `test_spatial_sort_permutes_without_dropping_tiles` | `_spatial_sort` keeps the tile set, groups same-quad tiles contiguously | real — §11.3 cache-locality ordering |
| `test_crop_center_upsample_recovers_uniform_center` | `_crop_center_upsample` crops the centre `frac` + bilinear-resizes (§6.3 scale-s→1× map) | real — multiscale geometry |
| `test_fuse_two_scales_averages_where_both_valid` | §7.3 arithmetic mean where both scales valid (0.8,0.4→0.6) | real — §7.3 fusion |
| `test_fuse_falls_back_to_1x_where_05_invalid` | 0.5× invalid → 1×-only (§6.3 graceful degradation) | real — §6.3 |
| `test_fuse_partial_05_coverage_mixes_per_pixel` | per-pixel mix: covered→mean, uncovered→1× | real — §7.3 per-pixel |
| `test_fuse_all_invalid_is_nan` | all scales NoData → NaN (runner masks to −1.0) | real — §7.3/§5.3 |
| `test_multiscale_dataset_yields_per_scale_images` | `scales=[1.0,0.5]` item carries per-scale image+valid; 0.5× reads the 2×-expanded bbox | real — §6.3 context read |
| `test_run_inference_multiscale_writes_fused_cog` | end-to-end `run_inference` multiscale dispatch → fused COG (const model 0.5 at both scales → fused 0.5) | real — §6.3/§7.3 integration |

> Not covered (deliberate): `scripts/inference.py` main loop is exercised by the
> Tier-2 real-data smoke (see inference.md §13 pre-inference checklist), not unit
> tests — it is thin glue over the tested modules.

### [test_vectorize_predictions.py](test_vectorize_predictions.py)

`scripts/vectorize_predictions.py` — mask→polygon vectorization with the
deployment `min_blob_size_px` object filter + windowed prob pixel-stats.
GPU-free; synthetic mask/prob rasters.

| Test | Checks | Strictness |
|---|---|---|
| `test_min_blob_filter_drops_small_and_keeps_large` | min_blob=2000 keeps the 3600px blob, drops the 144px one; compact rts_id; windowed `mean_prob`==0.8; CRS 3857 | real — object filter + rasterio-1.4 window-rounding regression |
| `test_no_filter_keeps_both_blobs` | min_blob=0 vectorizes both blobs | real — filter off path |

### [test_vectorize_region.py](test_vectorize_region.py)

`scripts/vectorize_region.py` — parallel block-mask polygonize + cross-seam
dissolve (post-inference.md §9.3 at region scale; mandatory for South where the
merged mask exceeds RAM). Validated bit-for-bit against the monolithic
`vectorize_predictions` on Banks (3010 polys / 69.42 km², both). GPU-free; two
synthetic adjacent block masks sharing a seam.

| Test | Checks | Strictness |
|---|---|---|
| `test_seam_split_slump_reassembles_and_survives_min_blob` | a slump split into two 200px halves (each < min_blob 300) reassembles to one 400px polygon and is KEPT — proves min_blob is applied AFTER the dissolve, no double-count | real — the core seam-stitch invariant |
| `test_scaled_uint8_prob_raster_decodes_mean_prob` | with a scaled_uint8 prob COG, `mean_prob` decodes to 0.8 (not raw 200) | real — regression of `de981b2` |
| `test_min_blob_zero_keeps_all_including_tiny` | min_blob=0 → interior + tiny + reassembled seam slump all kept | real — filter-off path |
| `test_threshold_mode_matches_mask_mode` | `threshold=0.65` on `probability_*.tif` shards reproduces the mask-mode result, incl. window seams (`window_px=50` splits the seam slump) | real — threshold-mode ≡ mask-mode equivalence |
| `test_threshold_mode_scaled_u8_nodata_not_above_threshold` | scaled_uint8 NoData 255 (> any scaled thr) excluded — no polygons over the NoData sea | real — the NoData-vs-threshold trap |
| `test_threshold_mode_lower_threshold_recovers_lower_prob_blob` | prob-0.5 blob invisible at thr 0.65, present at 0.30 with decoded `max_prob` | real — the permissive-product path |
| `test_multi_threshold_area_attributes` | `area_m2_t45/t65/t80` = geodesic area × fraction of in-polygon pixels ≥ t (half-0.9/half-0.5 blob) | real — boundary-uncertainty attributes |
| `test_min_area_m2_is_latitude_invariant` | identical-px blobs at lat 0 vs 60°N: geodesic 5,000 m² MMU keeps only the low-lat one (px filter can't separate); px prefilter didn't pre-drop either | real — the geodesic-MMU contract |
| `test_parallel_record_matches_serial` | workers=1 ≡ workers=4 on all `_record` attributes | real — parallel-stats regression guard |
| `test_arithmetic_tile_join_matches_scan` | stride-grid arithmetic `tile_ids` ≡ bbox scan on a holey t{col}_{row} grid, incl. sub-stride bounds | real — replaces the 41.5M-row scan |

### [test_aggregate_probability.py](test_aggregate_probability.py)

`scripts/aggregate_probability.py` — one-pass streaming aggregation of the
probability shards into threshold-free density grids (D2 products): per-cell
expected RTS area Σ decoded P × geodesic pixel area (cos²lat Mercator
correction) on a metric 3857 grid and a 0.5° WGS84 grid. GPU-free; tiny
synthetic scaled_uint8 shards with analytic ground truth.

| Test | Checks | Strictness |
|---|---|---|
| `test_expected_area_near_equator_matches_analytic` | Σ expected ≈ 0.5·100·res² for a P=0.5 blob at lat≈0; NoData sea contributes 0 | real — the core expectation math |
| `test_expected_area_at_60N_applies_cos2_correction` | same blob at 60°N shrinks by cos²(60°)=0.25 | real — geodesic correction |
| `test_blobs_land_in_their_own_cells` | blobs 2 cells apart occupy exactly 2 cells at the right offsets | real — metric binning |
| `test_half_degree_grid_bins_by_lonlat` | blob at (0.3°E, 0.1°N) lands in the containing 0.5° cell, analytic total | real — WGS84 binning |
| `test_multi_shard_sums_are_additive` | 2 shards sum; 3857 and 0.5° grids agree on the canvas total | real — the plan's independent-total check |
| `test_write_grids_products_and_per_class_join` | cell GPKGs (valid cells only) + GeoTIFFs written; `n_<class>`/`rts_m2_<class>` joined into the centroid's cell | real — product writer |

### [test_downsample_max.py](test_downsample_max.py)

`scripts/downsample_max.py` — exact block-max downsample for the browse
likelihood surface (replaces `gdalwarp -r max`, which bled NoData-edge values
251–254 > the 250 ceiling onto coverage seams). GPU-free; synthetic rasters.

| Test | Checks | Strictness |
|---|---|---|
| `test_block_max_and_nodata_semantics` | per-block max of valid pixels; all-NoData block stays 255; mixed block ignores 255; valid output ≤ 250 | real — the artifact regression |
| `test_non_divisible_edges_are_padded` | 50×30 @ factor 20 → 3×2; partial blocks don't fabricate values; transform scales by factor | real — edge handling |

### [test_retile_wmts_z10.py](test_retile_wmts_z10.py)

`scripts/retile_wmts_z10.py` — re-cuts the probability canvas into COGs that
each correspond to precisely one WMTS WebMercatorQuad z10 tile (the ADC
handover requirement). Grid math only — no rasters, GPU-free. (The raster
path is verified operationally: sampled output tiles are pixel-compared
against the source mosaic on every production run.)

| Test | Checks | Strictness |
|---|---|---|
| `test_grid_constants_are_consistent` | tile size = 8192 px × z15 res; 1024 tiles span the world exactly | real — grid arithmetic |
| `test_corner_tiles_span_the_world` | tiles (0,0) and (1023,1023) hit the matrix corners (z10 is 1024×1024) | real — caught the 2048-row bug |
| `test_adjacent_tiles_share_edges_exactly` | neighbouring tiles share edge coords bit-exactly (index-based edges) | real — caught float-drift edges |
| `test_candidate_tiles_cover_a_bbox_and_only_it` | bbox inside one tile → that tile; 2×2 straddle → exactly 4 | real — shard→candidate mapping |
| `test_candidate_tiles_clamped_to_matrix_extent` | whole-world bbox clamps to 1024×1024 indices | real — bounds guard |

### [test_export_south_products.py](test_export_south_products.py)

`scripts/export_south_products.py` — packages the raw thr-0.30 candidates into
the four D1 access forms (flagship tiered GPKG / high_confidence / centroids /
csv+parquet attribute table) with the QC-calibrated `rts_class` and the
`nodata_frac` soft-triage attribute. GPU-free; synthetic GPKG + tiny raster.

| Test | Checks | Strictness |
|---|---|---|
| `test_conf_class_boundaries_are_inclusive` | max_prob 0.45→medium, 0.65→high (inclusive bounds), below→low | real — tier SSoT |
| `test_rts_class_qc_calibrated_rule` | high_confidence = all high; candidate = medium <500 m² (inclusive/exclusive edge); marginal = rest | real — the locked 2026-07 QC rule |
| `test_export_products_writes_four_access_forms` | 4 files incl. `south_rts_high_confidence.gpkg` (no stale `south_rts_high`); rts_class column; representative points INSIDE a C-shaped polygon; csv/parquet without geometry | real — packaging correctness |
| `test_nodata_frac_from_probability_raster` | fraction of 255s in the (padded) bbox: clean → 0.0, half-NoData straddle → 0.5 | real — the soft-filter math |
| `test_export_products_plumbs_nodata_frac` | `prob_raster=` plumbs `nodata_frac` into flagship gpkg + csv | plumbing |

### [test_score_qc_ratings.py](test_score_qc_ratings.py)

`scripts/score_qc_ratings.py` — rated QC verdicts → precision per (tier ×
size band) with Wilson CIs → the adaptive-MMU acceptance grid. GPU-free.

| Test | Checks | Strictness |
|---|---|---|
| `test_precision_grid_counts_and_wilson` | counts, unsure excluded+reported, Wilson bounds bracket p, accept at floor | real — the calibration math |
| `test_empty_cells_are_reported_not_dropped` | full tier×band grid; unmeasured cells n=0/NaN and never accepted | real — the no-silent-acceptance guard |
| `test_export_false_polygons_hard_negative_seed` | rated-false polygons (only) export with geometry + verdict + tier/size in 3857 — the v3 hard-negative seed | real — join + filter correctness |

### [test_build_qc_rating_page.py](test_build_qc_rating_page.py)

`scripts/build_qc_rating_page.py` — offline single-file HTML rater (embedded
JPEG crops, localStorage autosave, CSV download) replacing the GEE rater whose
per-polygon tile loads were the rating bottleneck. GPU-free.

| Test | Checks | Strictness |
|---|---|---|
| `test_page_embeds_images_and_rating_machinery` | 2 data-URI crops per polygon; ITEMS JSON; localStorage/keydown/export tokens | real — generator contract |
| `test_tiny_polygon_gets_minimum_context_window` | tight ≥250 m, wide ≥1.5 km, centred | real — context floors |

Crop geometry + rendering now live in `review/crops.py`, shared with the review
campaign so both surfaces show pixel-identical views; this file's second test
imports `crop_bounds` from there. The refactor was verified by rebuilding the
shipped `qc_rater.html` and confirming a byte-identical MD5.

### [test_build_ee_qc_rater.py](test_build_ee_qc_rater.py)

`scripts/build_ee_qc_rater.py` — generates the GEE Code Editor rating app
with embedded WGS84 outlines + chip COG URIs. GPU-free.

| Test | Checks | Strictness |
|---|---|---|
| `test_rater_embeds_features_and_chip_uris` | FEATURES JSON parses, chip URIs from tile_ids on the usc1 mirror prefix, rings in lon/lat, loadGeoTIFF/Export present | real — generator contract |

### [test_build_ee_app_stats.py](test_build_ee_app_stats.py)

`scripts/build_ee_app_stats.py` — precomputes the public app's MMU retention
ladders so the size slider needs no server-side aggregation. GPU-free;
synthetic 9-polygon inventory.

| Test | Checks | Strictness |
|---|---|---|
| `test_retention_starts_whole_and_decreases` | MMU 0 keeps every polygon and the whole area; both series monotone non-increasing | real — ladder contract |
| `test_retention_filter_is_inclusive_at_the_threshold` | `area_m2 >= mmu`, matching `ee.Filter.gte` and `vectorize_region`'s geodesic filter | real — off-by-one guard |
| `test_tier_series_sum_to_the_whole_inventory` | per-tier ladders sum to the pooled ladder at every step (the disjointness the app relies on to combine tiers client-side) | real — the app's core assumption |
| `test_non_exhaustive_conf_class_is_rejected` | an unclassified polygon raises rather than silently vanishing from the readout | real — fail-loud |
| `test_min_blob_ground_area_shrinks_with_latitude` | pixel MMU ground area is `res²·cos²(lat)`; ~7× spread across the domain | real — the latitude-bias claim the panel makes |
| `test_ladder_carries_arts_p1_and_the_representative_min_blob` | 79 m² and the derived min_blob value are ladder members, so both presets land exactly | real — preset contract |
| `test_write_js_emits_a_parseable_literal` | `var APP_STATS = {…};` round-trips through JSON.parse equal to the input | real — generator contract |

### [test_sample_qc_polygons.py](test_sample_qc_polygons.py)

`scripts/sample_qc_polygons.py` — fixed-seed stratified QC sample (n per
conf_class band, longitude × area strata) with an empty `qc_verdict` column
for the ArcGIS rating pass. GPU-free; synthetic 600-polygon gdf.

| Test | Checks | Strictness |
|---|---|---|
| `test_sample_counts_and_verdict_column` | exactly n per band, empty verdicts, seed-reproducible | real — sample contract |
| `test_sample_spreads_across_longitude` | each band hits ≥5 of 6 longitude bins | real — the cross-region-variation guard |
| `test_small_band_returns_all_its_polygons` | band smaller than quota returned whole | real — degenerate band |

### [test_prob_writer.py](test_prob_writer.py)

`inference/runner.py::_ProbWriter` — background thread-pool prob-COG writer that
un-blocks the GPU from the per-tile GCS upload (the A100 throughput bottleneck;
benchmark 2026-07-07: 2.8 → 33 t/s on the real worker once writes went async +
the GCS client was cached). GPU-free.

| Test | Checks | Strictness |
|---|---|---|
| `test_writes_all_tiles_and_marks_done_after_success` | all tiles written + marked done via the pool | real — async write correctness |
| `test_backpressure_caps_inflight` | pending writes never exceed `max_inflight` on any submit | real — bounded-memory backpressure |
| `test_write_error_propagates_and_tile_not_marked_done` | a failed write re-raises on the owning thread; the tile is NOT marked done (crash-safe resume) | real — the done-only-after-success invariant |

Also covers the South-readiness fork-safety fix in `inference/runner.py` (2026-07-07):
`_make_loader` (forkserver worker start, avoiding the fork+gRPC deadlock that stranded Banks GPU-0).
The stall-watchdog tests moved to [test_watchdog.py](test_watchdog.py) when the watchdog was
lifted into `utils/` for the acquisition loop (2026-08-18).

| Test | Checks | Strictness |
|---|---|---|
| `test_make_loader_uses_forkserver_only_with_workers` | num_workers>0 → ForkServerContext; num_workers=0 → in-process (None) | real — the deadlock fix |
| `test_stall_watchdog_disabled_is_noop` | `stall_timeout_s<=0` returns a no-op stop fn, starts no thread | shallow — off-switch |
| `test_stall_watchdog_does_not_kill_while_progressing` | a live `last_active` never triggers os._exit | real — no false positives |
| `test_stall_watchdog_exits_process_on_hard_stall` | a stale `last_active` os._exit(3)s (asserted in a subprocess) | real — the self-heal trigger |

### [test_quad_drift.py](test_quad_drift.py)

`scripts/check_inference_normalization.py` quad mode — the interannual drift check. The original
script only read pre-cut training-layout tiles, so there was no way to drift-check a year that
exists only as 4096x4096 RGBA quads. GPU-free and GCS-free: synthetic RGBA quads on disk.

| Test | Checks | Strictness |
|------|--------|------------|
| `test_nodata_is_excluded_from_stats` | a half-NoData quad still reports the true mean | real — including alpha-0 zeros would halve the mean and fake radiometric drift on every coastal quad |
| `test_fully_nodata_quad_contributes_nothing` | an empty quad does not skew the sample | real |
| `test_unreadable_quad_is_skipped_not_fatal` | one bad object does not abort a 300-quad pass | real |
| `test_drift_flags_a_real_shift` | a >0.5σ mean shift trips `concerning`, and only on the shifted channel | real — the gate itself |

### [test_interannual_inference.py](test_interannual_inference.py)

`interannual_inference/` — the multi-year inference coordinator (6 years x 12 stages, two people, months of
wall clock). Its only job is to be trustworthy about what has and has not run, so the tests
concentrate on the ways it could *lie*: losing state on a crashed write, re-running finished work,
starting a stage whose inputs are not ready, walking past a human gate, or alerting either never
or every ten minutes forever. No Docker, no GCS, no network — stage commands are stubbed to
`true`/`false` and every path lives under `tmp_path`.

| Test | Checks | Strictness |
|------|--------|------------|
| `test_state_round_trip` | state survives save/load | shallow |
| `test_save_is_atomic_no_tmp_left_behind` | no `.tmp` orphan after a write | real — a half-written state file strands a whole year |
| `test_new_stage_added_later_reads_as_pending` | the stage table may grow mid-campaign | real |
| `test_set_stage_rejects_unknown_status` | typo'd status is refused, not stored | real |
| `test_running_sets_heartbeat_done_sets_finished` | timestamps follow the transition | real — the alerter reads these |
| `test_refuses_stage_with_unmet_prereq` | refuses, and *names* the missing stage | real |
| `test_runs_once_prereq_is_done` | the gate opens when it should | real |
| `test_done_stage_is_a_noop_without_force` | finished work is never repeated | real — re-running `infer` is 2.5 A100-days |
| `test_force_reruns_a_done_stage` | the deliberate override works | real |
| `test_failure_records_exit_code_and_log` | a failure says where to look | real |
| `test_external_stage_is_not_run_here` | `acquire` is Heidi's; the driver refuses it | real |
| `test_dry_run_executes_nothing` | `--dry-run` prints, does not run | real |
| `test_gate_stage_ends_blocked_not_done` | a gate stage stops at `blocked` | real — the whole point of gating |
| `test_blocked_gate_blocks_its_dependents` | inference cannot start behind an unsigned drift check | real |
| `test_sign_off_clears_the_gate` | a human can release it | real |
| `test_detached_stage_stays_running_after_its_launcher_exits` | launcher rc=0 != work finished | real — GEE/nohup both exit immediately |
| `test_quad_evidence_counts_rows_not_lines` | header is not counted as a quad | real |
| `test_acquire_evidence_reads_the_order_loop_status` | reads Heidi's status JSON shape | real |
| `test_drift_evidence_compares_against_the_quad_baseline` | drift is measured vs the 2025 *quad* sample | real — vs training-tile stats even 2025 trips the gate |
| `test_s2_export_passes_the_EE_project_not_the_gcs_one` | `--project` (EE quota) is not `GOOGLE_CLOUD_PROJECT` (GCS billing) | real — pdg cannot run batch exports at all; conflating these broke the first launch |
| `test_docker_wrapper_sets_the_gcs_billing_project` | the GCS billing project reaches the container | real — omitting it fails with "Project was not passed" |
| `test_unset_ee_project_refuses_loudly` | no authorised EE project fails fast | real — must never silently borrow another team's pdg-wg-* quota |
| `test_ee_project_guard_names_both_real_options` | the error names the two legitimate fixes and warns off pdg-wg-* | real |
| `test_evidence_failure_is_recorded_not_raised` | a missing artifact is data, not a crash | real |
| `test_matrix_shows_one_row_per_year_and_marks_each_stage` | the status grid renders | shallow |
| `test_cell_shows_percentage_when_a_probe_reported` | live progress reaches the cell | shallow |
| `test_detached_stage_making_progress_shows_no_heartbeat_warning` | a GEE launcher exiting is normal, not a fault | real — otherwise every export run looks broken |
| `test_detached_stage_with_no_progress_still_warns` | the warning survives where it matters | real |
| `test_dry_run_does_not_block_on_a_running_detached_stage` | `--dry-run` always returns | real — a running detached stage used to fall into the polling loop and hang |
| `test_shard_does_not_wait_on_the_s2_export` | sharding reads no imagery | real — would idle ~11 days for nothing |
| `test_infer_still_requires_the_s2_index` | NDVI prerequisite moved, did not vanish | real |
| `test_year_detail_names_the_gate` | a blocked year says what is wanted | real |
| `test_failure_alerts_once_not_every_tick` | announce-once, not every cron tick | real — the reason anyone keeps reading the channel |
| `test_gate_alert_names_the_stage` | the alert is actionable | real |
| `test_year_complete_alerts` | a finished year is announced | shallow |
| `test_stale_heartbeat_alone_is_not_stuck` | a still-advancing stage is not "stuck" | real — the false-alarm guard |
| `test_stale_heartbeat_plus_no_progress_is_stuck` | both signals together do alert | real |
| `test_first_sighting_never_alerts` | nothing to compare against yet | real — would cry wolf on every restart |
| `test_fresh_heartbeat_is_never_stuck` | a live stage is never flagged | real |

### [test_watchdog.py](test_watchdog.py)

`utils/watchdog.py::start_stall_watchdog` — the shared process-level stall guard. Lifted out of
`inference/runner.py` (2026-08-18) so the acquisition order loop can use it: both have the same
failure mode, where a wedged call holds its claim and reports no error rather than crashing.
GPU-free; the hard-stall cases run in a subprocess so `os._exit` cannot kill pytest.

| Test | Checks | Strictness |
|------|--------|------------|
| `test_disabled_is_noop` | `timeout_s <= 0` returns a no-op stop fn (tests / single-shot CLI) | real — opt-out path |
| `test_does_not_kill_while_progressing` | a 0.5 s timeout does not fire while `last_active` keeps moving | real — no false positives |
| `test_exits_process_on_hard_stall` | a stale `last_active` exits 3, the code the inference supervisor restarts on | real — the whole point of the module |
| `test_exit_code_is_configurable` | `exit_code=` is honoured, so callers can distinguish stall from crash | real — supervisor contract |

### [test_planetscope_download.py](test_planetscope_download.py)

`planetscope-download/` — the acquisition scripts ported from Heidi Rodenhizer's
`circumpolar_planet_basemaps` notebooks, carrying the four changes from her PR #61 review
(2026-08-17). Network-free and GCS-free: the Planet API is a scripted fake session, so the retry
policy is exercised without waiting on real back-off (`time.sleep` is patched out).

| Test | Checks | Strictness |
|------|--------|------------|
| `test_202_succeeds_first_try` | the happy path places exactly one request | real |
| `test_401_fails_fast_without_retrying` | auth failure raises immediately instead of burning 5 attempts | real — retrying a dead key only delays the fix |
| `test_transient_status_retries_then_succeeds` | 500 → 409 → 202 recovers within the attempt budget | real — the failures Heidi actually sees |
| `test_retries_are_bounded_then_recorded_as_failed` | exhaustion returns `failed` after `MAX_ATTEMPTS`, so the loop continues | real — a 5-day run is never abandoned over one quad |
| `test_non_retryable_status_gives_up_immediately` | a 404 is not retried | real |
| `test_connection_errors_are_retried` | transport exceptions are retried like transient statuses | real |
| `test_order_payload_matches_the_2025_delivery_shape` | quad id, COG `file_format` tool, delivery bucket/prefix, and a non-None timeout on every call | real — payload drift would change delivered filenames |
| `test_progress_counts_and_writes_status` | counters and `pct_done` are right; the status file lands | real |
| `test_status_write_failure_does_not_kill_the_run` | an unwritable status path is warned, not raised | real — monitoring must not take down the job |
| `test_watchdog_timestamp_advances_on_progress` | `last_active` is bumped per completed quad | real — wires the loop to the watchdog |
| `test_filter_to_domain_clips_and_derives_columns` | the R→geopandas port drops out-of-domain quads and derives `delivery_location` | real — a wrong prefix mis-delivers the imagery |
| `test_filter_to_domain_sorts_by_column_then_row` | column-major order preserved (the original `arrange`) | real — delivery order is spatially contiguous |
| `test_filter_to_domain_survives_an_empty_clip` | an all-miss grid returns an empty frame with the schema intact | real — regression, this KeyError'd in the first end-to-end smoke |
| `test_flatten_name_strips_uuid_and_folds_mosaic_dir` | the optional tidy-up's path mapping | real |
| `test_flatten_name_is_idempotent` | re-running the tidy-up is a no-op, not a second mangling | real — it is resumed by re-running |
| `test_flattened_names_still_index` | both raw and flattened names match `_QUAD_NAME_RE` | real — the claim that the rename is optional |
| `test_finished_year_is_not_reported_stale` | a 100%-done year reads `complete`, not `STALE` | real — regression; a finished run stops heartbeating by design, and the false alarm sends the operator chasing a process that succeeded |
| `test_incomplete_and_quiet_is_reported_stale` | an unfinished year gone quiet is flagged | real — the alarm that matters |
| `test_live_run_is_flagged_neither` | a progressing run is unflagged | real — no false positives |
| `test_runtime_outputs_default_outside_the_repo` | `DEFAULT_WORK` / `DEFAULT_STATUS_DIR` are not under the checkout | real — regression; the shared checkout is read-only to collaborators, which broke Heidi's first run |
| `test_psd_work_env_var_overrides_the_default` | `PSD_WORK` relocates runtime output | real — the documented escape hatch |
| `test_silent_while_the_run_is_healthy` | no alert while ordering is progressing | real — a checker that chatters trains you to ignore it |
| `test_silent_when_stale_but_process_still_alive` | a live process suppresses the alarm during the slow startup listing | real — otherwise it cries wolf on every restart |
| `test_alerts_when_stale_and_no_process` | both signals together do fire, with the resume command | real — the case the cron exists for |
| `test_alert_is_not_repeated_every_tick` | one alert per incident, not one per 10 min | real — dedupe |
| `test_recovery_rearms_the_alert` | a resumed run clears the marker so the next stop alerts again | real — the failure mode of naive dedupe |
| `test_completion_alerts_once` | a finished year alerts once, carrying `--expect-quads` | real — hands off to the next step |


### [test_assemble_region.py](test_assemble_region.py)

`scripts/assemble_region.py` — blocked windowed merge that assembles a whole
region's per-tile prob COGs into one mosaicked COG without holding the
(200k×310k) canvas in RAM (post-inference.md §7). GPU-free; synthetic
constant-value overlapping tile COGs.

| Test | Checks | Strictness |
|---|---|---|
| `test_blocked_merge_matches_single_shot` | blocked `merge_window` reconstruction == single-shot `merge_tiles` (incl. NoData mask) at 3 block sizes — the seamlessness guarantee | real — §7 mosaic == merge |
| `test_cog_grid_mosaic_matches_single_cog` | the parallel super-tile-COG grid + `.vrt` (`cog_tile_px>0`, the South-scale path) reads back pixel-identical to the monolithic single-COG path (`cog_tile_px=0`) | real — grid is a scale/perf change only; skips w/o GDAL CLI |
| `test_iter_blocks_tiles_the_canvas_without_gaps` | `iter_blocks` partitions the canvas exactly once (no gap/overlap) | real — block grid |

### [test_build_rgb_chips.py](test_build_rgb_chips.py)

`scripts/build_rgb_chips.py` — generates RGB "underlying tile" context chips
for the ArcGIS Pro QC package, but only for the tiles a detected RTS polygon's
`tile_ids` column references (not the whole region). Reuses the real
`inference.tiles.read_tile` quad-windowing path. GPU-free; synthetic gpkg +
one synthetic RGBA quad COG.

| Test | Checks | Strictness |
|---|---|---|
| `test_collect_flagged_tile_ids_dedupes_across_polygons` | comma-separated `tile_ids` across rows dedupe into one set | real |
| `test_collect_flagged_tile_ids_empty_gpkg_returns_empty_set` | zero-polygon gpkg → empty set, no crash | shallow |
| `test_build_tile_bboxes_returns_only_requested_ids` | join against the tile-list CSV returns exactly the requested ids with correct bounds | real |
| `test_build_tile_bboxes_raises_on_missing_tile_id` | a `tile_id` referenced by the gpkg but absent from the tile list raises (surfaced data-integrity mismatch, not silently dropped) | real |
| `test_write_rgb_chip_is_georeferenced_uint8_and_matches_quad_values` | `write_rgb_chip` → `read_tile` end-to-end: output is a 3-band uint8 GeoTIFF, EPSG:3857, correct bounds, pixel values match the source quad | real — exercises the actual inference read path |
| `test_chip_write_is_atomic` | a failed write leaves neither a truncated `.tif` nor a `.partial` | real — the resume path skips any existing file, so a truncated chip would be skipped forever |

Gained `--workers`, skip-existing resume, and a `-input_file_list` VRT build on
2026-08-03 for the review campaign: the 0.30 candidate inventory references
118,586 tiles (vs 29,850 chipped for the 0.65 product), and 118k paths overflow
argv. See `post-inference/review_campaign.md` §4.1.

### [test_claim.py](test_claim.py)

`inference/claim.py` — the GCS-atomic shard-claim queue for the dual-fleet run (plan Phase 1). GPU-free, network-free: a FakeBucket emulates GCS `if_generation_match=0` create-if-absent atomicity + listing + download + delete; a clock is injected for staleness.

| Test | Checks | Strictness |
|---|---|---|
| `test_two_workers_never_win_one_shard` | atomic create-if-absent → exactly one of two contenders wins; claim records the winner | real — the core "use both fleets" invariant |
| `test_claim_next_skips_done_and_returns_first_free` | `claim_next` skips done shards, returns first free | real |
| `test_done_skip_on_restart` | a shard with a done marker is never reprocessed | real — §8.3 resumability |
| `test_mark_done_clears_claim` | `mark_done` writes done marker + drops the claim | real |
| `test_fresh_claim_is_not_reclaimed` | a claim younger than the TTL is not stolen | real |
| `test_stale_claim_is_reclaimed_and_reassigned` | a crashed worker's stale claim (heartbeat > TTL) is reclaimed + reassigned | real — straggler/preemption recovery |
| `test_heartbeat_keeps_claim_fresh` | heartbeat refreshes the claim so a long shard isn't seen as stale | real |
| `test_heartbeat_does_not_steal_others_claim` | heartbeat by a non-owner is a no-op | real — ownership guard |
| `test_reclaim_absent_claim_is_false` | reclaiming a missing claim returns False | shallow |

### [test_gcs_parity.py](test_gcs_parity.py)

`scripts/gcs_parity.py` — the object-level parity check that gates deletion of the PDG buckets (`computing/pdg_migration.md` §5). GPU-free, network-free: a FakeClient serves canned listings. The tool compares *every* object by walking both lexicographic listings in lockstep, so these tests pin each way a copy can be wrong rather than trusting a sample.

| Test | Checks | Strictness |
|---|---|---|
| `test_split_uri` (4 cases) | `gs://bucket/prefix` parsing, trailing slash, bare bucket | shallow |
| `test_split_uri_rejects_non_gs` | a local path raises rather than being silently treated as a URI | real |
| `test_entries_are_named_relative_to_the_prefix` | two prefixes at different absolute paths yield identical entry lists | real — the premise the whole comparison rests on |
| `test_entries_skips_directory_placeholders` | zero-byte `dir/` objects are not counted as data | real |
| `test_entries_tolerates_null_size_and_md5` | absent size/MD5 in the projection degrade to `0`/`""` instead of raising | shallow |
| `test_identical_listings_match` | a clean copy passes; counts and bytes agree | real |
| `test_both_empty_is_a_pass` | a legitimately empty prefix is not a failure | shallow |
| `test_missing_object_is_reported_by_name` | a dropped object fails and is named | real |
| `test_missing_object_at_the_end_is_caught` | the walk drains the longer side rather than stopping at the shorter | real — the classic lockstep off-by-one |
| `test_missing_object_at_the_start_is_caught` | divergence at the first element is caught | real |
| `test_extra_object_at_destination_is_reported` | an unexpected object at the destination fails | real |
| `test_truncated_object_is_caught_by_size` | same count, wrong bytes — the partial write | real |
| `test_corrupt_object_is_caught_by_md5` | same count *and* same bytes; only the MD5 differs | real — the failure a sampling check would miss |
| `test_every_corruption_is_caught_not_just_a_sample` | 500/500 corrupt objects counted, reporting capped at `MAX_REPORTED` | real — pins full coverage vs capped display |
| `test_reported_names_are_capped_but_the_walk_completes` | capping the report never truncates the counts | real |
| `test_main_exits_zero_on_a_clean_copy` | exit 0 on parity | real — this is what the gate reads |
| `test_main_exits_nonzero_on_a_bad_copy` | exit 1 on corruption | real |
| `test_main_exits_nonzero_on_an_empty_destination` | exit 1 when nothing arrived | real |

### [test_shard_tiles.py](test_shard_tiles.py)

`scripts/shard_tiles.make_shards` — splits the tile list into spatially-contiguous shards (plan Phase 1). GPU-free, I/O-free (the pure split logic; the CLI's GCS writes are thin glue).

| Test | Checks | Strictness |
|---|---|---|
| `test_every_tile_in_exactly_one_shard` | union of shards == input, no dup/drop | real — exactly-once coverage invariant |
| `test_shard_count_and_sizes` | ceil(N/size) shards; full shards + remainder; sizes sum to N | real |
| `test_shard_ids_are_sequential_and_padded` | ids `shard_000000…` sequential | shallow |
| `test_shards_concatenate_to_spatial_order` | shards in order == `_spatial_sort` (cache-locality contract) | real — §11.3 |
| `test_single_shard_when_size_exceeds_count` | size > N → one shard with all tiles | shallow |
| `test_nonpositive_shard_size_rejected` | shard_size <= 0 → ValueError | shallow |

### [test_run_inference_worker.py](test_run_inference_worker.py)

`scripts/run_inference_worker.work_loop` — the queue-drain loop (plan Phase 1). GPU-free: real `ClaimStore` over the in-memory fake bucket + a stub `process_shard` (the inference body itself, `inference.runner.run_inference`, is covered by the pipeline tests + the Tier-2 real-data smoke).

| Test | Checks | Strictness |
|---|---|---|
| `test_single_worker_drains_all_exactly_once` | claims every shard in order, once; all marked done | real — drain completeness |
| `test_mark_done_only_after_process` | at process time the shard is claimed but not yet done (crash-mid-shard leaves a reclaimable claim, not a false done) | real — crash-safety ordering |
| `test_resume_skips_already_done` | pre-done shards are skipped on (re)start | real — §8.3 resume |
| `test_two_workers_cover_all_disjointly` | A (capped) + B drain cooperatively → union complete, intersection empty | real — multi-VM exactly-once |
| `test_max_shards_stops_early` | `--max-shards` stops after N | shallow |
| `test_time_based_heartbeat_during_slow_shard` | claim heartbeat refreshed by wall-clock thread while process_shard runs | real — an active worker's shard was reclaimed live when heartbeats depended on progress ticks (2026-07-05 drill) |

### [test_inference_progress.py](test_inference_progress.py)

`scripts/inference_progress` pure math (plan Phase 0/4 monitor). GPU-free, network-free, injected clock (the GCS listing + dashboard rendering around it is thin glue, exercised live against the bucket).

| Test | Checks | Strictness |
|---|---|---|
| `test_progress_counts_and_pct` | shards done/active/remaining + tiles done (exact, from per-shard counts) + per-host | real |
| `test_progress_recent_rate_and_eta` | windowed tiles/s + ETA; empty → rate 0, ETA None | real |
| `test_progress_falls_back_to_average_rate` | no completions in window → since-start average rate | real |
| `test_progress_flags_stale_worker` | claim heartbeat older than window → stale (silent-idle alarm, pre-mortem #4) | real |
| `test_progress_aggregates_per_host` | active claims grouped by worker host (per-VM view) | real |
| `test_s2_counts_and_eta` | S2 cells done/remaining/pct, launched passthrough, cells/hr + ETA | real |
| `test_s2_empty_has_no_eta` | no cells yet → rate 0, ETA None | shallow |

### [test_tune_object_operating_point.py](test_tune_object_operating_point.py)

Tier-1 object operating-point tuner (`scripts/tune_object_operating_point.py`, report-only). GPU-free; synthetic prob/label maps with known objects. Load-bearing test is parity with the training object metric.

| Test | Checks | Strictness |
|---|---|---|
| `test_parity_with_validation_accumulator` | at defaults (thr 0.5, min_blob 10, no morph) the tuner's obj_tp/fp/fn == `ValidationAccumulator` for identical input | real — parity guarantee |
| `test_min_blob_filters_small_predictions` | raising `min_blob_size` drops sub-size FP blobs | real |
| `test_morph_closing_merges_fragments` | morph-close radius bridges a 1-px gap → fragment FP removed (1 TP, 0 FP/FN) | real — morphology lever |
| `test_decompose_categories` | object-error decomposition routes a no-overlap FP → `fp_no_overlap`, an unpredicted GT → `fn_missed` | real |
| `test_evaluate_grid_shape_and_threshold_monotonicity` | grid yields one row per cell; raising threshold past the prob drops obj_tp to 0 | real |

### [test_pretrain_corpus.py](test_pretrain_corpus.py)

v2.1 SSL corpus sampling/exclusion/quality logic (`pretraining/corpus.py`), CPU-only, no GCS.

| Test | Checks | Strictness |
|---|---|---|
| `test_filter_to_s2_footprint_keeps_only_intersecting` | Envelope pre-filter + STRtree exact-intersect keeps only tiles over an S2 cell | real — defines the south-covered corpus domain |
| `test_drop_excluded_removes_tiles_over_polygons` | Tiles intersecting a val/test polygon are dropped | real — leakage hygiene |
| `test_drop_excluded_empty_tree_is_noop` | No exclusion polygons → all tiles kept | shallow |
| `test_load_exclusion_polygons_reprojects_from_3413` | Subregions GeoJSON is EPSG:3413 (polar stereo); polygons reprojected to 3857 before the tree, else intersections silently miss → eval-region leakage | real — leakage-hygiene regression (caught pre-build 2026-07-15) |
| `test_load_exclusion_polygons_no_reproject_when_already_3857` | CRS already 3857 → geometry unchanged | shallow |
| `test_stratified_sample_balances_across_strata` | Two far-apart clusters → roughly even draw per stratum | real |
| `test_stratified_sample_returns_all_when_target_exceeds_pool` | `n_target ≥ pool` → returns everything | shallow |
| `test_stratified_sample_oversamples_marked_tiles` | `oversample_mask` tiles drawn first within a stratum | real — near-label oversampling |
| `test_quality_ok_rejects_high_nodata_and_empty_ndvi` | >50% NoData or all-NaN NDVI → rejected | real — corpus quality filter |

### [test_pretrain_mim.py](test_pretrain_mim.py)

v2.1 MAE masking + item loading (`pretraining/mim_dataset.py`), CPU-only.

| Test | Checks | Strictness |
|---|---|---|
| `test_random_patch_mask_ratio_and_shape` | Exactly `round(ratio·N)` patches masked, right shape/dtype | real |
| `test_random_patch_mask_varies_with_generator` | Different RNG → different mask | shallow |
| `test_expand_mask_upsamples_to_pixels` | Patch mask → pixel mask block-upsampled correctly | real |
| `test_mae_patchify_shape_and_layout` | `MaskedAutoencoderViT.patchify` shape + per-patch pixel layout (top-left patch → row 0) | real — target-construction correctness (full ViT forward/backward is GPU-smoke-covered, not in the CPU suite) |
| `test_dataset_item_shapes_and_nan_neutralization` | 4-ch image + patch mask shapes; NaN NDVI neutralized to 0 by `apply_norm` | real — training/inference norm parity |

### [test_train_smoke.py](test_train_smoke.py)

End-to-end training loop on the synthetic fixture (~130 s, still Tier 1 — no GCS, no GPU). Asserts the hardened criteria from the plan Step 7a.

An autouse fixture `_isolate_mlflow_tracking_uri` clears `MLFLOW_TRACKING_URI` before each test: MLflow 3.x's `set_tracking_uri()` writes that env var into `os.environ`, and since these tests call `train.main()` in-process (and train.py prefers the env var over cfg), the first test's tracking URI would otherwise leak into later tests and send their runs to the wrong store — `test_mlflow_run_written` / `test_train_iou_logged` failed only in full-suite order without it (2026-06-26).

| Test | Checks | Strictness |
|---|---|---|
| `test_run_produces_log_file` | train.log exists, validation ran at least once | shallow |
| `test_figures_written` | prob_hist / confusion / pr_curves PNGs produced | real — figure plumbing |
| `test_deployment_checkpoint_contract` | best_deployment.pth has all contracted keys | real — training.md §4.3 end-to-end |
| `test_resume_checkpoint_rotation` | resume_latest-*.pth exists post-training | real |
| `test_no_nan_in_model_params` | Final EMA weights all finite | real — numerical guard |
| `test_mlflow_run_written` | MLflow directory populated | shallow |
| `test_train_iou_logged` | `train_iou` logged per epoch ∈ [0,1] (needed for experiments.md §5.4 data-scaling gap + §8.1 Phase-5 gate) | real — guards the train-metric add (2026-06-07) |
| `test_ema_divergent_from_live_after_training` | EMA ≠ live weights after unfreeze (exercises update path) | real — plan risk #15 |
| `test_prediction_shows_response_on_positive_region` | max pred prob > 0.1 on a positive tile (collapse guard) | real — plan risk (mode collapse) |
| `test_train_smoke_resume_then_continue` | Resume from epoch-2 snapshot for 1 more epoch; EMA shadow is restored and continues decaying (key set unchanged, post-resume ≠ saved) | real — Important I5 (2026-05-02); guards EMA-restore-on-resume audit fix |
| `test_select_preview_tiles_uses_fixed_list` | A `preview_tiles.yaml` UID list is used verbatim (intersected with val, order preserved) | real — fixed preview contract (2026-06-05) |
| `test_select_preview_tiles_is_seed_independent` | Same fixed list → identical previews for seed 42 vs 43 (cross-experiment comparability) | real — guards the seed-coupling bug fix |
| `test_select_preview_tiles_falls_back_when_none_in_val` | If no configured tile is in val, fall back to the seeded heuristic | real — graceful fallback |
| `test_resume_ema_shadow_on_model_device` | `_resume_from` moves the restored EMA shadow to the model's device; `ema.update` must not raise cpu/cuda mismatch (skipped without CUDA) | real — regression for 2026-06-11 A100 resume crash |

### [test_review_crops.py](test_review_crops.py)

`review/crops.py` (crop geometry + rendering, shared with the offline pack) and
`chip_index` in `scripts/build_review_crops.py`. GPU-free; builds a real 2-chip
VRT with `gdalbuildvrt`.

| Test | Checks | Strictness |
|---|---|---|
| `test_crop_bounds_apply_the_minimum_context_floors` | tight ≥250 m, wide ≥1.5 km for a tiny feature | real — context floors |
| `test_crop_bounds_scale_with_a_large_feature` | above the floor the crop is 3×/10× the feature | real |
| `test_crop_bounds_are_square_and_centred` | square, centred on a non-square feature | real |
| `test_index_bounds_match_the_chips_own_bounds` | indexed bounds equal rasterio's own per file | real — a wrong index silently selects no chips and yields blank crops |
| `test_index_paths_are_absolute_and_readable` | VRT-relative sources resolved to absolute, openable paths | real — workers run from another cwd |
| `test_index_lists_each_chip_once_not_once_per_band` | band-1 sources only; all 3 bands would triple the index | real |
| `test_has_imagery_is_true_over_a_populated_chip` | probe sees real pixels | real |
| `test_has_imagery_is_false_over_a_nodata_chip` | an all-NoData chip reads as absent | real — the coverage report depends on it |
| `test_has_imagery_is_false_off_the_mosaic` | boundless fill reads as absent, not as dark imagery | real — replaced a colour-based check that mistook antialiased outline for content |
| `test_render_crop_returns_a_jpeg` | output is JPEG-encoded | shallow |
| `test_outline_false_renders_the_same_view_without_the_red` | `outline=False` drops the red outline pixels and keeps the imagery | real — counts red pixels, not bytes; the toggle's whole premise |

### [test_review_manifest.py](test_review_manifest.py)

`scripts/build_review_manifest.py` — the review campaign's queue construction
(`post-inference/review_campaign.md` §3, §5). GPU-free, network-free.

| Test | Checks | Strictness |
|---|---|---|
| `test_every_polygon_appears_exactly_once_as_coverage` | the coverage invariant — no polygon skipped, none reviewed twice by accident | real — the census claim rests on it |
| `test_batches_are_full_except_the_last` | batch sizes 300/300/300/100 for 1,000 items | real |
| `test_batch_and_item_ids_never_collide_within_a_batch` | (batch_id, rts_id) unique — a duplicate would collapse two items into one verdict | real — caught a live bug (replicate clamped into its own batch) |
| `test_batches_are_cut_in_descending_probability_order` | each batch's floor ≥ the next batch's ceiling | real — the "every polygon with p ≥ x is reviewed" headline |
| `test_item_order_within_a_batch_is_shuffled` | a batch is not probability-sorted internally; `seq` is still 0..n-1 | real — response-bias defence |
| `test_replicates_are_injected_into_later_batches` | every replicate sits exactly `REPLICATE_OFFSET` batches after its source | real — caught the clamping bug |
| `test_replicates_are_spread_through_the_campaign` | replicate batches span the queue rather than clustering | real — else kappa measures one slice |
| `test_zero_replicates_is_allowed` | `n_replicates=0` yields a pure coverage queue | shallow |
| `test_rebuild_is_identical` | deterministic under seed 42 | real — rebuildable manifest |
| `test_duplicate_input_ids_are_rejected` | a duplicated `rts_id` in the inventory is an error | real |

### [test_review_store.py](test_review_store.py)

`review/store.py` — campaign queue + verdict store over `inference/claim.py`
(§6, §9). Network-free; reuses `test_claim.py`'s FakeBucket.

| Test | Checks | Strictness |
|---|---|---|
| `test_two_reviewers_never_get_the_same_batch` | atomic claiming across reviewers | real — the core collaboration invariant |
| `test_queue_is_served_in_probability_order` | batches handed out in ascending id = descending probability | real |
| `test_exhausted_queue_returns_none` | a finished campaign reports completion | real |
| `test_a_claim_outlives_any_rating_session` | a two-day-old claim is still held | real — a reviewer's part-rated batch must not be handed to someone else overnight |
| `test_a_claim_older_than_the_ttl_returns_to_the_pool` | past one week the batch is presumed abandoned and re-served | real |
| `test_the_ttl_is_one_week_in_seconds` | `STALE_AFTER_S == 604800.0` | real — a wrong unit here silently steals live batches |
| `test_a_released_claim_is_re_servable` | deleting the claim object frees the batch before the TTL | real |
| `test_fresh_claim_is_not_stolen` | a live claim is respected | real |
| `test_submitted_batch_is_never_re_served` | done markers survive restart | real |
| `test_submit_is_idempotent` | a retried submit neither doubles nor overwrites | real — double-click / retry safety |
| `test_submit_rejects_an_id_from_another_batch` | cross-batch verdicts are an error | real |
| `test_submit_rejects_an_unknown_verdict` | vocabulary is enforced | real |
| `test_submit_rejects_an_unknown_batch` | unknown batch is an error | shallow |
| `test_injected_items_are_flagged_in_the_record` | replicates are marked in the JSONL, so the merge can separate them | real |
| `test_batch_items_are_in_presentation_order_with_crop_keys` | shuffled order preserved; all four crop keys (outlined + plain) derived from the prefix | real |
| `test_headline_counts_only_a_contiguous_prefix` | finishing a later batch does not inflate the headline claim | real — the product's honesty guard |
| `test_progress_counts_coverage_items_not_injected_ones` | injected items excluded from coverage totals | real |

### [test_review_app.py](test_review_app.py)

`review/app.py` — the campaign's HTTP contract (§6). Network-free: FakeBucket
plus a stubbed IAP verifier, via FastAPI's `TestClient`.

| Test | Checks | Strictness |
|---|---|---|
| `test_claim_returns_items_with_crop_urls` | items carry proxy URLs and never leak raw object keys | real — the "no imagery on the host" design |
| `test_crop_streams_the_jpeg` | `/crop/<key>` returns the bytes as `image/jpeg`, cacheable | real — the whole imagery path |
| `test_crop_refuses_objects_outside_the_crop_prefix` | verdicts/manifest are unreachable through the proxy, including via `..` | real — otherwise the proxy reads the whole bucket |
| `test_crop_of_a_missing_object_is_404_not_500` | the 20 no-imagery polygons degrade gracefully | real |
| `test_two_reviewers_get_different_batches` | claiming is atomic through the API | real |
| `test_exhausted_campaign_reports_null_batch` | the UI's completion signal | real |
| `test_reopen_serves_a_held_batch_again` | browser-reload resume returns the same items | real — reload safety |
| `test_reopen_of_a_submitted_batch_is_a_conflict` | 409 so the UI can discard stale local verdicts | real |
| `test_reopen_of_an_unknown_batch_is_404` | unknown batch is not a 500 | shallow |
| `test_submit_persists_and_is_idempotent` | written=True then written=False | real |
| `test_bad_verdict_is_a_400_not_a_500` | store `ValueError` maps to a client error | real |
| `test_progress_tracks_submissions` | progress reflects a completed batch | real |
| `test_index_serves_the_rater` | the UI is served at `/` | shallow |
| `test_iap_identity_is_used_when_present` | a verified IAP assertion sets the reviewer | real |
| `test_client_cannot_override_the_iap_identity` | a supplied name is ignored behind IAP | real — attribution feeds κ and the audit trail, so it must not be typeable |
| `test_claim_is_recorded_under_the_iap_identity` | the verdict JSONL carries the authenticated address, not the client's | real |
| `test_an_unverifiable_assertion_is_rejected` | a forged assertion 401s rather than falling back to the typed name | real — the fallback would otherwise be a bypass |
| `test_supplied_name_is_used_when_there_is_no_iap` | local runs and the offline pack still work | real |
| `test_no_identity_at_all_is_a_403` | no name and no assertion is refused | real |
| `test_me_reports_no_identity_rather_than_failing` | `/api/me` answers before the UI has a name | real — the UI calls it first |

### [test_review_merge.py](test_review_merge.py)

`scripts/merge_review_verdicts.py` — pooling verdicts into the verified
inventory plus the agreement statistics (§7–§8). GPU-free, network-free.

| Test | Checks | Strictness |
|---|---|---|
| `test_kappa_matches_a_hand_computed_table` | kappa = 0.4 for a 2×2 with po=0.7, pe=0.5 | real — pins the statistic against a hand computation |
| `test_kappa_is_one_for_perfect_agreement` | identical series → 1.0 | real |
| `test_kappa_is_nan_when_one_label_is_universal` | degenerate case is NaN, not a spurious 1.0 | real — would otherwise flatter the campaign |
| `test_kappa_of_an_empty_sample_is_nan` | no pairs → NaN | shallow |
| `test_replicate_pairs_join_a_replicate_to_its_coverage_verdict` | pairing carries both verdicts and both reviewers | real |
| `test_a_replicate_without_its_coverage_verdict_is_dropped` | unpaired replicates are not counted | real |
| `test_coverage_verdict_wins_over_the_replicate` | precedence rule; `n_reviews`/`reviewers` populated | real — decides what the product says |
| `test_agreement_is_true_when_both_reviewers_match` | agreement flag set | real |
| `test_unreplicated_polygons_have_no_agreement_value` | NaN rather than a fabricated True | real |
| `test_a_stray_id_is_an_error_not_a_silent_drop` | a verdict outside the manifest raises | real — data-integrity guard |
| `test_duplicate_coverage_verdicts_keep_the_latest` | deterministic conflict resolution | real |
| `test_a_partial_campaign_merges_only_what_was_rated` | partial campaigns produce partial products | real — stop-anytime design |
| `test_an_empty_campaign_merges_to_an_empty_frame` | schema preserved when nothing is rated | shallow |
| `test_report_counts_coverage_and_excludes_injected_from_the_total` | denominators exclude injected rows | real — else "% reviewed" is wrong |
| `test_kappa_is_only_computed_across_different_reviewers` | self-agreement is not reported as inter-rater | real — the statistic's meaning |
| `test_report_compares_against_the_2026_07_pass` | drift check against the prior solo ratings | real |
| `test_read_verdicts_reads_every_batch_file` | all batch JSONLs are pooled | real |
| `test_read_verdicts_of_an_empty_campaign_is_an_empty_frame` | empty campaign has a usable schema | shallow |
| `test_duplicate_coverage_verdicts_do_not_inflate_the_pair_count` | joining on a non-unique id must not multiply the agreement sample | real — guards a silent kappa inflation |

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
8. **Object-scorecard real-data execution (Tier 2)** — `test_object_scorecard.py` covers all report-only *logic* on synthetic arrays, but the GPU/real-data runs are not in pytest: (a) `score_insample_train.py` `_build_cache` (ensemble inference over the train sample → `*_probs.npz`) is exercised only on the L4 VM; (b) `probe_change_signal.py` needs an L4-built `*_change.npz` (|ΔNDVI| over the 2024−2023 train year-pair for the val tiles) — its construction is imagery-layout-dependent and not yet written; (c) parity of the scorecard's aggregate against the frozen Finding-K numbers is a Tier-2 check on the real val/test caches.

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
- 2026-04-23 — Phase 1 additions: 81 new tests across 10 files covering models, losses, EMA, scheduler, metrics, checkpointing, freeze/unfreeze, early stopping, MLflow utilities, visualizations, deployment-package guards, and an end-to-end training smoke. Fast suite 105 tests (~12 s), plus the train-smoke at ~130 s. Total 113 tests. All green.
- 2026-06-30 — Object-scorecard instrument (v3 object-improvement plan, Phase 0): +4 `_object_match_detail` tests in `test_metrics.py` (tp/fp/fn parity with the frozen gate path + split/merge/geometry), and new `test_object_scorecard.py` (14 tests) covering `object_detail_counts`, per-region bootstrap CIs, `_geometry_summary`, `build_scorecard` self-check, the region-stratified train sampler, the D2 change-signal probe, and the seed-noise aggregator. All report-only/synthetic; verified green off-VM under a torch stub (real torch on the L4 runs them in the full suite). Tier-2 execution gaps recorded in Coverage gaps #8.
- 2026-07-04 — Minimum Mapping Unit fix (data-v1.1): new `test_gt_mmu_scoring.py` (6 tests) validating `apply_min_mapping_unit` composing through the 255-ignore machinery — sub-MMU GT + correct pred → (0,0,0); real object survives; straddle keeps 1 FP; `build_scorecard` parity self-check holds; pixel counts unmoved + focal loss invariant to logits under the 255 sliver. The primitive itself stays covered by `test_label_cleaning.py`. All synthetic/CPU. Green under real torch (`test_gt_mmu_scoring.py` + `test_label_cleaning.py` + `test_object_scorecard.py` = 28 passed; `test_dataset.py` + `test_metrics.py` = 35 passed).
- 2026-08-12 — Shipped-product-rule scoring: new `test_score_product_rule.py` (14 tests) for `scripts/score_product_rule.py`, which scores the delivered adaptive rule (0.30 contour → `max_prob` → `conf_class`/`rts_class`) on the frozen test cache alongside the ledger-J/K anchors. Concentrates on the failure modes the replication invites: the 1/250 quantisation boundary that makes the 0.65 tier cut `u8 >= 163` rather than the `u8 >= 162` of a raster cut, the 2-px technical floor and its ordering *before* banding, and `load_product_constants` reading `export_south_products.py`'s constants via `ast` (geopandas is absent from the scoring image, so importing it is not an option and retyping the constants would fork the SSoT). Also pins the intentional pixel-count basis change vs `object_counts` (final mask, not pre-filter). All synthetic/CPU. Green in `rts-train:v2` (14 passed).
- 2026-08-27 — PDG migration parity gate: new `test_gcs_parity.py` (21 tests) for `scripts/gcs_parity.py`, the check that decides whether the PDG buckets may be deleted. The tool was first written to MD5-sample 200 objects per leg; the test that corrupted one object in twenty exposed why that is worthless here — a 200-object sample over the 41.7M-object `probs/` leg would essentially never touch the bad one. Rewritten as a constant-memory lockstep walk of both lexicographic listings, which is both stronger (every object compared) and simpler (no reservoir, no RNG, no seed flag). The tests pin each failure mode separately — missing at start/middle/end, extra, truncated, corrupt-but-same-size — because they exercise different branches of the three-way merge. All synthetic/CPU. Green in `rts-train:v2` (21 passed); full suite **668 passed, 2 skipped** (620 before the `interannual-campaign` merge landed its 48). Fixed en route: `test_quad_drift.py` hardcoded `REPO = Path("/w")`, so the whole suite failed collection unless the container happened to mount the repo at `/w` — now derived from `__file__` like every other test.
- 2026-07-07 — ArcGIS Pro QC package (Banks Island team review): new `test_build_rgb_chips.py` (5 tests) for `scripts/build_rgb_chips.py`, which generates RGB "underlying tile" context chips for the ArcGIS Pro QC package — only for the tiles a detected RTS polygon references, reusing `inference.tiles.read_tile`. All synthetic/GPU-free. Full suite 356 passed, 1 skipped (pre-existing) + these 5 = 361 green.

## `scripts/verify_frozen_model.py` — not a pytest test

Added 2026-08-28 for the PDG migration gate (`computing/pdg_migration.md` §5 row 3). Checks the
migrated deployment packages against the `model_checkpoint_sha` every production shard manifest
recorded on 2026-07-07, so the anchor is the run that made the delivered map rather than the PDG
copy of it. Operational, not part of the suite: it needs live ADC and reads a live bucket, so it has
no place in a GPU-free offline run. Listed here so nobody adds it to CI looking for missing coverage.
Object-level parity is `scripts/gcs_parity.py`, which *is* covered (`test_gcs_parity.py`, 21 tests).

**A duplication caught and reverted the same day.** I wrote two more verifiers —
`verify_migration_parity.py` (counts + bytes per prefix) and `sample_hash_check.py` (reservoir MD5
sample per leg) — without noticing `gcs_parity.py` already existed and did both, over *every*
object. Worse, `gcs_parity.py`'s own docstring records that the 200-object sample was written first
and rejected, because the test that corrupts 1 object in 20 showed a sample cannot cover a 42 M-object
prefix. Both were deleted and the gate re-measured with `gcs_parity.py`. Checking `scripts/` before
writing a script would have cost a minute.
