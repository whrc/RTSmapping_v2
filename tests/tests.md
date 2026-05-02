# Tests

Living doc for the test suite. **Update this file whenever you add, remove, or meaningfully change a test.** Keep the per-file tables and the coverage-gap section in sync with the code.

---

## Purpose

Two-tier verification:

- **Tier 1 — `pytest tests/`**: runs on synthetic fixtures, no GCS, no GPU. Fast suite ~12 s, plus the end-to-end train-smoke at ~130 s. Guards code correctness and contracts. **Must be green before any real-data work.**
- **Tier 2 — real-data scripts**: runs on the real v2.0 bucket on the L4 VM.
    - Phase 0 data checks: `scripts/check_data_content.py` (bucket structure) + `scripts/check_data.py` (DataLoader preview).
    - Phase 1 training smoke: `python scripts/train.py --config configs/smoke.yaml` (2 epochs on a subset of real regions; inference.md §6.4 gate).

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
| `test_filter_train_positive_subset_keeps_negatives_intact` | `_filter_train_positive_subset` (in `scripts/train.py`): subset_pct=25 keeps 25% of positives, all negatives untouched | real — Phase 0 §3.2 + Phase 2 §5.1 contract |
| `test_filter_train_positive_subset_is_deterministic` | Two invocations with the same input give the same output (seed=42 hard-coded) | real — reproducibility |
| `test_filter_train_positive_subset_full_pct_no_op` | subset_pct=100 keeps every tile | shallow — boundary case |

### [test_dataset.py](test_dataset.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_parse_extra_spec_empty` | `None` and `[]` both → `[]` | shallow |
| `test_parse_extra_spec_flexible_names` | Arbitrary names parsed, band indices preserved | real — flexible-EXTRA guarantee |
| `test_parse_extra_spec_rejects_missing_keys` | Missing `name` or `band` → `ValueError` | real |
| `test_dataset_rejects_soft_labels` | `boundary_handling="soft_labels"` raises `NotImplementedError` (deferred to v2.1, training.md §5.5) | real — guards a config option that isn't wired to code |
| `test_dataset_rejects_unknown_boundary_handling` | Unknown value (e.g. `"bogus"`) raises `ValueError` | real |
| `test_dataset_rgb_only` | `(3, 64, 64) float32` image, `(64, 64) int64` label, str `tile_id` | real — end-to-end plumbing |
| `test_dataset_with_variable_extra` | Bands [0, 2] + arbitrary names → `(5, 64, 64)` | real — flexible-EXTRA end-to-end |
| `test_dataset_label_values_in_set` | Every label's unique values ⊂ {0, 1, 255} | real |
| `test_boundary_dilation_adds_ignore` | Width=2 dilation creates 255 band and preserves interior 1s | real |
| `test_init_raises_on_rgb_channel_name_mismatch` | RTSDataset refuses stats with permuted RGB channel names (training.md §4.5) | real — Critical C1 (2026-05-02) |
| `test_init_raises_on_extra_channel_name_mismatch` | RTSDataset refuses stats with mis-ordered EXTRA channel names | real — Critical C1 (2026-05-02) |

### [test_transforms.py](test_transforms.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_color_aug_does_not_touch_extra` | EXTRA channels are bit-identical after color-only augmentation (training.md §9.2) | real — Critical C3 (2026-05-02) |
| `test_geometric_aug_applies_to_extra_and_mask` | HorizontalFlip applies to RGB, EXTRA, and mask jointly | real |
| `test_extra_none_path_still_works` | RGB-only call path (no `extra` kwarg) preserved through the split | real — backward-compat for baseline RGB-only |

### [test_models.py](test_models.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_build_model_rgb_only_output_shape` | `(B, 1, 512, 512)` from UNet++/EffB5 on RGB-only config | real |
| `test_build_model_with_extra_channels` | 7-channel (RGB+4 EXTRA) forward pass returns correct shape | real — flexible-EXTRA in models |
| `test_output_bias_initialized_to_class_prior` | Final-conv bias at prior=0.5 equals 0.0 | real — focal-paper init |
| `test_output_bias_for_imbalanced_prior` | prior=0.01 → bias ≈ -log(99) | real |
| `test_output_is_logits_not_probabilities` | Random-input outputs span beyond [0, 1] | real — logits contract |
| `test_invalid_bias_prior_rejected` | Prior outside (0, 1) → `ValueError` | shallow |
| `test_unknown_architecture_rejected` | Unsupported arch → clear `ValueError` | shallow |

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
| `test_lr_range_test_applies_same_lr_to_all_groups` | All param groups receive the same LR under range-test mode | real |
| `test_lr_range_test_rejects_invalid_bounds` | lr_min ≥ lr_max → `ValueError` | shallow — guard |
| `test_unknown_scheduler_raises` | Unknown `scheduler:` value → `ValueError` | shallow — dispatch guard |

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
| `test_accumulator_perfect_prediction_pixel_iou_one` | Exact match → all metrics 1.0 | real |
| `test_accumulator_ignore_index_masks_pixels` | ignore pixels contribute nothing to TP/FP/FN | real — ignore contract |
| `test_accumulator_speckle_fp_filtered` | 1-px FP below min_blob_size doesn't count | real |
| `test_accumulator_pr_auc_ranges_between_zero_and_one` | PR-AUC in [0, 1]; geomean equals single-ratio value | real |
| `test_accumulator_no_positive_tiles_produces_zero_pr_auc` | No-positive val → PR-AUC=0.0 gracefully | real — edge case |

### [test_freeze.py](test_freeze.py)

| Test | Checks | Strictness |
|---|---|---|
| `test_freeze_backbone_disables_grad_on_encoder_only` | Encoder params requires_grad=False; decoder untouched | real |
| `test_unfreeze_backbone_restores_grad` | After unfreeze, all encoder params trainable again | real |
| `test_build_param_groups_partitions_by_id` | Every model param appears in exactly one named group | real |
| `test_build_param_groups_lrs_set` | Decoder/backbone LRs + weight_decay set as requested | shallow |
| `test_optimizer_respects_frozen_encoder` | After freeze + step, encoder weights unchanged | real — integration check |

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

> **Coverage gap (acknowledged 2026-05-01):** these four tests exercise only
> `_assert_calibration_complete` (~6 LOC). The end-to-end packaging path
> (MLflow run resolution, `weights.pth` extraction from
> `best_deployment.pth`, `model_config.yaml` + `deployment_config.yaml`
> assembly) is not unit-tested; it relies on real MLflow runs at deploy time.
> The training smoke test does not call `package_model.main()`. Close this gap
> when packaging misbehaves on a real run, or by feeding the smoke test's
> synthetic MLflow run into `package_model()` end-to-end.

### [test_train_smoke.py](test_train_smoke.py)

End-to-end training loop on the synthetic fixture (~130 s, still Tier 1 — no GCS, no GPU). Asserts the hardened criteria from the plan Step 7a.

| Test | Checks | Strictness |
|---|---|---|
| `test_run_produces_log_file` | train.log exists, validation ran at least once | shallow |
| `test_figures_written` | prob_hist / confusion / pr_curves PNGs produced | real — figure plumbing |
| `test_deployment_checkpoint_contract` | best_deployment.pth has all contracted keys | real — training.md §4.3 end-to-end |
| `test_resume_checkpoint_rotation` | resume_latest-*.pth exists post-training | real |
| `test_no_nan_in_model_params` | Final EMA weights all finite | real — numerical guard |
| `test_mlflow_run_written` | MLflow directory populated | shallow |
| `test_ema_divergent_from_live_after_training` | EMA ≠ live weights after unfreeze (exercises update path) | real — plan risk #15 |
| `test_prediction_shows_response_on_positive_region` | max pred prob > 0.1 on a positive tile (collapse guard) | real — plan risk (mode collapse) |
| `test_train_smoke_resume_then_continue` | Resume from epoch-2 snapshot for 1 more epoch; EMA shadow is restored and continues decaying (key set unchanged, post-resume ≠ saved) | real — Important I5 (2026-05-02); guards EMA-restore-on-resume audit fix |

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
- 2026-04-23 — Phase 1 additions: 81 new tests across 10 files covering models, losses, EMA, scheduler, metrics, checkpointing, freeze/unfreeze, early stopping, MLflow utilities, visualizations, deployment-package guards, and an end-to-end training smoke. Fast suite 105 tests (~12 s), plus the train-smoke at ~130 s. Total 113 tests. All green.
