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

EXTRA derivation SSoT (`data/extra_channels.py`). SE math only — Earth Engine is mocked.

| Test | Checks | Strictness |
|---|---|---|
| `test_band_norm_mode` | `band_norm_mode` returns "zscore" for NDVI/SE_PCA/TC and "fixed_scale" for SE_PROTO; unknown band → `ValueError` | real — §9 SSoT |
| `test_se_bands_projection_and_cosine` | With `fetch_se_raw` mocked: `se_bands` returns {2,3,4,5} of shape (H,W); SE_PROTO ∈ [-1,1]; SE_PCA1 == manual `flat @ component[0]` projection | real — SE derivation math |
| `test_se_bands_nan_propagates` | A no-coverage (NaN) SE pixel yields NaN SE bands; finite pixels stay finite (matches S2 NaN handling) | real |
| `test_se_bands_zero_vector_is_nan` | A no-coverage SE pixel arriving as an all-zero vector (not NaN) → NaN SE_PCA *and* SE_PROTO, so `(0-pca_mean)@comps.T` can't leak a nonzero artifact | real — B1 NoData contract |

### [test_generate_extra_tiles.py](test_generate_extra_tiles.py)

Covers the CSV-bbox footprint source added for the 2025 inference EXTRA handoff (doc §6.5) — pure logic only; the GEE fetch is not exercised.

| Test | Checks | Strictness |
|------|--------|------------|
| `test_load_ids_and_bounds_inference_schema` | `tile_id,minx,miny,maxx,maxy` CSV → ids + correct `{id: bounds}` map | real |
| `test_load_ids_training_schema_has_no_bounds` | Legacy `Tile_ID` CSV (no bbox cols) → ids only, bounds `None` (falls back to `--rgb-dir`) | real |
| `test_profile_from_bounds_coregisters` | Profile is EPSG:3857, 512², 8-band float32; transform maps pixel (0,0)→(minx,maxy) and (512,512)→(maxx,miny) | real — co-registration contract |
| `test_write_bands_creates_then_resumes` | First write creates 8-band NaN stack + fills NDVI; tile "done" for `--groups s2` only once {0,1,6,7} all non-NaN | real — resumability |

### [test_export_s2_composites.py](test_export_s2_composites.py)

Covers the bulk S2 export grid/domain geometry (doc §3); EE + GCS not exercised.

| Test | Checks | Strictness |
|------|--------|------------|
| `test_latlon_grid_aligns_and_covers` | `latlon_grid` cells are origin-aligned `dlon×dlat` and cover the bbox corners | real |
| `test_cell_id_deterministic_and_sign_safe` | `cell_id` stable + sign-safe (`W1500_N0740`, `E0000_S0025`); distinct corners → distinct ids | real |
| `test_domain_cells_keeps_only_intersecting` | Cells filtered to those intersecting the (reprojected) domain polygon; clip ⊆ domain ∩ cell; far cell excluded | real |

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
| `test_state_dict_roundtrip` | save/load reproduces history + counters | real |

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

Also covers the South-readiness fork-safety fixes in `inference/runner.py` (2026-07-07):
`_make_loader` (forkserver worker start, avoiding the fork+gRPC deadlock that stranded Banks GPU-0) and `_start_stall_watchdog` (os._exit a wedged worker so its shard is reclaimed).

| Test | Checks | Strictness |
|---|---|---|
| `test_make_loader_uses_forkserver_only_with_workers` | num_workers>0 → ForkServerContext; num_workers=0 → in-process (None) | real — the deadlock fix |
| `test_stall_watchdog_disabled_is_noop` | `stall_timeout_s<=0` returns a no-op stop fn, starts no thread | shallow — off-switch |
| `test_stall_watchdog_does_not_kill_while_progressing` | a live `last_active` never triggers os._exit | real — no false positives |
| `test_stall_watchdog_exits_process_on_hard_stall` | a stale `last_active` os._exit(3)s (asserted in a subprocess) | real — the self-heal trigger |

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
- 2026-07-07 — ArcGIS Pro QC package (Banks Island team review): new `test_build_rgb_chips.py` (5 tests) for `scripts/build_rgb_chips.py`, which generates RGB "underlying tile" context chips for the ArcGIS Pro QC package — only for the tiles a detected RTS polygon references, reusing `inference.tiles.read_tile`. All synthetic/GPU-free. Full suite 356 passed, 1 skipped (pre-existing) + these 5 = 361 green.
