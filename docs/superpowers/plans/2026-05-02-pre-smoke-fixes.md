# Pre-Smoke-Test Fixes + Config Matrix Slim-Down

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the necessary code-review fixes (channel-name assertion, EXTRA-safe augmentation, normalization-clip documentation, resume-flow test) and slim the `configs/` matrix down to `baseline.yaml` + `deployment.yaml` so per-phase configs are constructed on demand. After this plan, the branch is ready for the L4 real-data smoke test.

**Architecture:** Two-track plan in one sequence. **Track A** (Task 1) deletes 15 pre-made phase configs and updates the two docs that reference them. This deletion dissolves Critical issue C2 (`output_bias_prior: 0.5` reverted in 14 configs) without per-file edits. **Track B** (Tasks 2–5) lands the remaining Critical and selected Important fixes from the 2026-05-02 code review. Final task runs the full verification.

**Tech Stack:** Python 3.10, PyTorch, albumentations, rasterio, pytest. No new dependencies.

---

## Background

Code review of branch `phase1-training-loop` returned 3 Critical + 9 Important issues. User decisions on scope:
- **C1** (channel-name binding never asserted at training load) → fix.
- **C2** (`output_bias_prior: 0.5` re-surfaced in 14 phase configs) → dissolved by deleting those configs.
- **C3** (color/radiometric augmentations applied to EXTRA channels) → fix now (safest).
- **I1** (`normalization.{rgb,extra}_clip_percentiles` is dead config) → keep keys, document as unimplemented (no clipping work).
- **I5** (resume flow has no automated test) → include.
- All other Important / Minor items → defer to a post-smoke housekeeping plan.

User intent on configs: "every decision is controlling the next yaml, don't pre-make late phase yamls. Delete all yamls and only keep the beginning one. Construct yaml one by one when run experiments." `phase0a` is documented in `training/experiments.md §Phase 0a` as a procedure (3 RGB-arm runs); Arm A is `baseline.yaml` as-is, Arms B/C will be created on demand. So the kept set is `baseline.yaml` + `deployment.yaml` only.

## File Structure

**Modified:**
- [data/dataset.py](../../../data/dataset.py) — `RTSDataset.__init__`: assert channel-name agreement after `load_stats(...)`.
- [data/transforms.py](../../../data/transforms.py) — split into `build_train_transforms` returning a `TrainTransform` callable that applies color-only stage to RGB then geometric stage to stacked array.
- [scripts/check_inference_normalization.py](../../../scripts/check_inference_normalization.py) — already asserts RGB names; extend to assert EXTRA names when present.
- [configs/baseline.yaml](../../../configs/baseline.yaml) — comment `rgb_clip_percentiles` / `extra_clip_percentiles` as reserved-but-unimplemented.
- [training/experiments.md](../../../training/experiments.md) — §11.1 "Config naming": replace listing of phase configs with on-demand convention.
- [current_working_status.md](../../../current_working_status.md) — update "17 configs" line and add 2026-05-02 dev-log entry.
- [tests/test_dataset.py](../../../tests/test_dataset.py) — add channel-name-assertion tests.
- [tests/test_train_smoke.py](../../../tests/test_train_smoke.py) — add `test_train_smoke_resume_then_continue`.
- [tests/tests.md](../../../tests/tests.md) — log new tests.

**Created:**
- [tests/test_transforms.py](../../../tests/test_transforms.py) (new) — EXTRA-isolation regression test for the augmentation split.

**Deleted:**
- `configs/phase0_lr_test_frozen.yaml`
- `configs/phase0_lr_test_unfrozen.yaml`
- `configs/phase0_seed42.yaml`
- `configs/phase0_seed43.yaml`
- `configs/phase0_seed44.yaml`
- `configs/phase2_scale_25.yaml`
- `configs/phase2_scale_50.yaml`
- `configs/phase2_scale_75.yaml`
- `configs/phase2_scale_100.yaml`
- `configs/phase3_loss_compound_1_1.yaml`
- `configs/phase3_loss_compound_1_2.yaml`
- `configs/phase3_loss_compound_2_1.yaml`
- `configs/phase3_loss_tversky_02_08.yaml`
- `configs/phase3_loss_tversky_03_07.yaml`
- `configs/se_investigation.yaml`

---

## Task 1: Slim config matrix to baseline + deployment

**Why this is first:** Deleting these 15 configs removes the surface area where C2 lives. Done first so subsequent tasks don't have to think about per-phase configs.

**Files:**
- Delete: 15 config files listed above
- Modify: `training/experiments.md:411-432`
- Modify: `current_working_status.md:30-37`

- [ ] **Step 1: Verify the deletion list against current state**

```bash
ls configs/
```

Expected: 17 files visible. `baseline.yaml` and `deployment.yaml` to be kept; the other 15 to be deleted. If anything other than these 17 files is present, stop and reconcile with the plan author before proceeding.

- [ ] **Step 2: Delete the 15 phase configs**

```bash
git rm configs/phase0_lr_test_frozen.yaml \
       configs/phase0_lr_test_unfrozen.yaml \
       configs/phase0_seed42.yaml \
       configs/phase0_seed43.yaml \
       configs/phase0_seed44.yaml \
       configs/phase2_scale_25.yaml \
       configs/phase2_scale_50.yaml \
       configs/phase2_scale_75.yaml \
       configs/phase2_scale_100.yaml \
       configs/phase3_loss_compound_1_1.yaml \
       configs/phase3_loss_compound_1_2.yaml \
       configs/phase3_loss_compound_2_1.yaml \
       configs/phase3_loss_tversky_02_08.yaml \
       configs/phase3_loss_tversky_03_07.yaml \
       configs/se_investigation.yaml
```

- [ ] **Step 3: Verify only baseline + deployment remain**

```bash
ls configs/
```

Expected output: `baseline.yaml  deployment.yaml`

- [ ] **Step 4: Confirm no test or script references the deleted configs**

```bash
grep -rn "phase0_lr_test\|phase0_seed4\|phase2_scale_\|phase3_loss_\|se_investigation" tests/ scripts/ data/ training/ models/ losses/ utils/ 2>/dev/null
```

Expected output: no matches in `tests/`, `scripts/`, or any code directory. Matches in `training/experiments.md` and `current_working_status.md` are OK and will be addressed in the next two steps. Any matches outside of those two files mean an unexpected reference exists — investigate before continuing.

- [ ] **Step 5: Update `training/experiments.md` §11.1 to on-demand-config convention**

Replace lines 416–432 (everything from the `### 11.1 Config naming` heading through the end of that subsection, **including** the `Phase configs are self-contained...` paragraph) with the following block:

```markdown
### 11.1 Config naming

Each experiment is one YAML file in `configs/`. The repository commits only the
two configs needed to start any experiment chain:

```
configs/baseline.yaml    ← Phase 0 baseline (and Phase 0a Arm A)
configs/deployment.yaml  ← post-calibration deployment config (per inference.md §2.2)
```

All phase-specific configs are created **on demand**, one per experiment, as the
predecessor phase locks. Naming convention when created:

```
configs/phase0a_arm_b.yaml             ← Phase 0a §Arms — x/255 + ImageNet stats
configs/phase0a_arm_c.yaml             ← Phase 0a §Arms — x/255 only
configs/phase0_lr_test_frozen.yaml     ← Phase 0 §3.2 frozen-phase LR range
configs/phase0_lr_test_unfrozen.yaml   ← Phase 0 §3.2 unfrozen-phase LR range
configs/phase0_seed{42,43,44}.yaml     ← Phase 0 §3.3 multi-seed baseline
configs/phase2_scale_{25,50,75,100}.yaml  ← Phase 2 §5.1 — N% positives
configs/phase3_loss_<family>.yaml      ← Phase 3 §6.1 — per loss-family candidate
configs/phase4_extra_<group_name>.yaml ← Phase 4 §7.1 — per EXTRA group
configs/final_seed{42,43,44}.yaml      ← Final §9 multi-seed lock
```

Each new config copies the prior phase's winner hyperparameters into a fresh
file, then changes only the keys this experiment is testing. This avoids
the drift class that pre-made placeholder configs introduced (audit
2026-05-01: 12 configs deleted; review 2026-05-02: `output_bias_prior` had
reverted in 14 of the remaining configs).
```

(Use the Edit tool with `old_string` matching the existing §11.1 block exactly.)

- [ ] **Step 6: Update `current_working_status.md` Status section**

Replace the line at `current_working_status.md:33`:

```
  - **Config matrix slimmed**: deleted 12 placeholder configs (phase3_boundary_*, phase4_extra_*, final_seed*); these will be created per-phase as each predecessor locks. Remaining: baseline + deployment + 5 phase0 + 4 phase2 + 5 phase3_loss + se_investigation = 17 configs.
```

with:

```
  - **Config matrix slimmed (2026-05-02)**: deleted 15 remaining placeholder configs (phase0_*, phase2_*, phase3_loss_*, se_investigation). Repository commits only `configs/baseline.yaml` + `configs/deployment.yaml`; per-phase configs are created on demand when each experiment fires. See `training/experiments.md §11.1`.
```

- [ ] **Step 7: Append a 2026-05-02 entry to the Dev Log**

Append at the end of `current_working_status.md`:

```markdown
- 2026-05-02 — Phase 1 code-review pass + pre-smoke prep. Code-reviewer surfaced 3 Critical (C1 channel-name binding never asserted at training load; C2 `output_bias_prior: 0.5` reverted in 14 configs; C3 color/radiometric augmentations applied to EXTRA channels) plus 9 Important. Plan `docs/superpowers/plans/2026-05-02-pre-smoke-fixes.md` lands C1, C3, I1 (document `clip_percentiles` as unimplemented), I5 (resume regression test), and dissolves C2 by deleting the 15 pre-made phase configs. Other Important items (I2, I4, I6, I7, I8, I9 + Minor) deferred to post-smoke housekeeping plan.
```

- [ ] **Step 8: Commit**

```bash
git add configs/ training/experiments.md current_working_status.md
git commit -m "configs: slim matrix to baseline + deployment; per-phase on demand

Removes 15 pre-made phase configs (phase0_*, phase2_*, phase3_loss_*,
se_investigation). Per-phase configs are created on demand as each
experiment fires, copying the prior phase's winner hyperparameters.

Side-effect: dissolves Critical C2 from the 2026-05-02 code review
(output_bias_prior: 0.5 had reverted in all 14 phase configs; only
baseline.yaml retained the audit-fixed 0.005)."
```

---

## Task 2: C1 — channel-name binding assertion in `RTSDataset`

**Why this matters:** training.md §4.5 mandates that train and inference both assert the stats file's channel order matches the consumer's expected order. The assertion was missing from training-side load. This is the single most consequential train-inference contract violation; even with baseline RGB-only it landmines Phase 4 EXTRA configs. Reviewer cited [data/dataset.py:95-103](../../../data/dataset.py#L95-L103).

**Files:**
- Modify: `data/dataset.py:95-103` (the `if norm_stats_path is not None` block)
- Modify: `scripts/check_inference_normalization.py` (extend to also assert EXTRA names when present)
- Test: `tests/test_dataset.py` (new tests for the assertion)

- [ ] **Step 1: Write the failing test for RGB-name mismatch**

Add to `tests/test_dataset.py` (at the end of the file):

```python
def test_init_raises_on_rgb_channel_name_mismatch(synthetic_dataset, tmp_path):
    """RTSDataset must refuse stats whose RGB channel_names != ['R', 'G', 'B']."""
    import json

    from data.dataset import RTSDataset
    from data.transforms import build_eval_transforms

    bad_stats = {
        "dataset_version": "test",
        "computed_date": "2026-05-02T00:00:00Z",
        "n_tiles_used": 10,
        "rgb": {
            "channel_names": ["B", "G", "R"],   # permuted!
            "mean": [128.0, 128.0, 128.0],
            "std": [40.0, 40.0, 40.0],
        },
    }
    bad_path = tmp_path / "bad_stats.json"
    bad_path.write_text(json.dumps(bad_stats))

    with pytest.raises(ValueError, match=r"RGB channel_names"):
        RTSDataset(
            tile_ids=synthetic_dataset["splits"]["train"][:1],
            metadata=synthetic_dataset["metadata_df"],
            data_root=synthetic_dataset["root"],
            rgb_dir="PLANET-RGB",
            extra_dir="EXTRA",
            labels_dir="labels",
            extra_channels=[],
            norm_stats_path=str(bad_path),
            transform=build_eval_transforms(),
            tile_size=64,
        )


def test_init_raises_on_extra_channel_name_mismatch(synthetic_dataset, tmp_path):
    """When extra_channels is set, EXTRA channel_names must match exactly."""
    import json

    from data.dataset import ExtraChannel, RTSDataset
    from data.transforms import build_eval_transforms

    extra_channels = [
        ExtraChannel(name="ndvi", band=0),
        ExtraChannel(name="nir", band=1),
    ]
    # Stats file lists EXTRA names in REVERSED order vs the config:
    bad_stats = {
        "dataset_version": "test",
        "computed_date": "2026-05-02T00:00:00Z",
        "n_tiles_used": 10,
        "rgb": {
            "channel_names": ["R", "G", "B"],
            "mean": [128.0, 128.0, 128.0],
            "std": [40.0, 40.0, 40.0],
        },
        "extra": {
            "channel_names": ["nir", "ndvi"],   # reversed!
            "mean": [0.5, 0.0],
            "std": [0.1, 0.3],
        },
    }
    bad_path = tmp_path / "bad_stats_extra.json"
    bad_path.write_text(json.dumps(bad_stats))

    with pytest.raises(ValueError, match=r"EXTRA channel_names"):
        RTSDataset(
            tile_ids=synthetic_dataset["splits"]["train"][:1],
            metadata=synthetic_dataset["metadata_df"],
            data_root=synthetic_dataset["root"],
            rgb_dir="PLANET-RGB",
            extra_dir="EXTRA",
            labels_dir="labels",
            extra_channels=extra_channels,
            norm_stats_path=str(bad_path),
            transform=build_eval_transforms(),
            tile_size=64,
        )
```

(`pytest` is already imported at the top of `tests/test_dataset.py`; if not, add `import pytest`.)

- [ ] **Step 2: Run the new tests, expect FAIL**

```bash
cd /home/ext_rtsmapping_woodwellclimate_o/RTSmappingDL
python -m pytest tests/test_dataset.py::test_init_raises_on_rgb_channel_name_mismatch \
                  tests/test_dataset.py::test_init_raises_on_extra_channel_name_mismatch -v
```

Expected: both FAIL — `RTSDataset.__init__` doesn't raise; the bad stats are accepted and `stats_to_arrays` happily concatenates the wrong-order means/stds.

- [ ] **Step 3: Implement the assertion in `RTSDataset.__init__`**

Replace the block in `data/dataset.py` from line 95 (`if norm_stats_path is not None:`) through line 103, with:

```python
        if norm_stats_path is not None:
            stats = load_stats(norm_stats_path)
            # training.md §4.5: assert channel-name agreement before any vector
            # arithmetic. Catches the "R-stats applied to G-channel" failure
            # mode where compute_normalization_stats was re-run after the
            # config's EXTRA order changed but the consumer expects the old order.
            expected_rgb = ["R", "G", "B"]
            actual_rgb = list(stats.get("rgb", {}).get("channel_names", []))
            if actual_rgb != expected_rgb:
                raise ValueError(
                    f"normalization stats RGB channel_names {actual_rgb!r} "
                    f"does not match expected {expected_rgb!r}"
                )
            if extra_channels:
                expected_extra = [c.name for c in extra_channels]
                actual_extra = list(stats.get("extra", {}).get("channel_names", []))
                if actual_extra != expected_extra:
                    raise ValueError(
                        f"normalization stats EXTRA channel_names {actual_extra!r} "
                        f"does not match config order {expected_extra!r}"
                    )
            self.mean, self.std = stats_to_arrays(stats, with_extra=bool(extra_channels))
        else:
            # Permitted for smoke tests; real runs must supply stats.
            logger.warning("RTSDataset created without normalization stats; output will be unnormalized")
            n_channels = 3 + len(extra_channels)
            self.mean = np.zeros(n_channels, dtype=np.float32)
            self.std = np.ones(n_channels, dtype=np.float32)
```

- [ ] **Step 4: Run the new tests, expect PASS**

```bash
python -m pytest tests/test_dataset.py::test_init_raises_on_rgb_channel_name_mismatch \
                  tests/test_dataset.py::test_init_raises_on_extra_channel_name_mismatch -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full `test_dataset.py` to confirm no existing tests broke**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: all tests PASS (existing 7 + 2 new = 9, give or take depending on what's already in the file).

- [ ] **Step 6: Extend `scripts/check_inference_normalization.py` to also assert EXTRA names**

The script already asserts `rgb_names == ["R", "G", "B"]` at line 163. Extend so EXTRA names (when present in the stats file) are also surfaced. After the existing RGB assertion (line 166), and BEFORE the `data_root = args.data_root or ...` line (line 168), insert:

```python
    # When EXTRA stats are present, log their channel order so any drift-script
    # consumer is aware. (Strict mismatch with a config consumer's expected order
    # is enforced inside RTSDataset.__init__; this script just reports.)
    if "extra" in training_stats:
        extra_names = list(training_stats["extra"].get("channel_names", []))
        logger.info("Training stats include EXTRA channels: %s", extra_names)
```

- [ ] **Step 7: Commit**

```bash
git add data/dataset.py scripts/check_inference_normalization.py tests/test_dataset.py
git commit -m "data: assert normalization-stats channel names at RTSDataset init

Closes Critical C1 from the 2026-05-02 code review. training.md §4.5
mandates the assertion; it was missing from the training-side load
path. Adds two regression tests (RGB permuted, EXTRA reversed).
check_inference_normalization.py now also logs EXTRA channel order
when present."
```

---

## Task 3: C3 — split RGB-color and stacked-geometric augmentation pipelines

**Why this matters:** training.md §9.2 says color/radiometric augmentations apply only to RGB. The current `build_train_transforms` declares `additional_targets={"extra": "image"}` so albumentations applies `RandomBrightnessContrast`, `HueSaturationValue`, `GaussNoise`, and `CLAHE` to EXTRA bands too. Latent on RGB-only baseline; corrupts any EXTRA-using run silently. Reviewer cited [data/transforms.py:78](../../../data/transforms.py#L78).

**Approach:** `build_train_transforms` returns a callable `TrainTransform` object with the same call signature as the current Compose (`(image=, extra=, mask=)`). Internally, when `extra` is provided, the callable applies a color-only Compose to RGB, then a geometric Compose to RGB+EXTRA+mask. When `extra` is None, the two composes still run sequentially, but it doesn't matter to the downstream caller. `build_eval_transforms` is unchanged (no augmentation, no problem).

This keeps the `RTSDataset.__getitem__` call site stable (line 144: `aug = self.transform(image=rgb, extra=extra, mask=label)` — works because `TrainTransform.__call__` accepts the same kwargs).

**Files:**
- Modify: `data/transforms.py` (replace `build_train_transforms`)
- Test: `tests/test_transforms.py` (new) — assert that with `extra` provided and color probabilities = 1.0, EXTRA pixels are bit-identical pre/post; geometric ops still apply to both.

- [ ] **Step 1: Write the failing test for EXTRA isolation under color-only augmentation**

Create `tests/test_transforms.py` (new file):

```python
"""Regression tests for data/transforms.py.

Focus: EXTRA channels must NOT receive color/radiometric augmentation
(training.md §9.2). Geometric augmentation (flips, rotations, scale,
elastic) DOES apply to EXTRA.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.transforms import build_train_transforms


def _aug_cfg(*, color_p: float, geo_p: float) -> dict:
    """Build an augmentation config with all color ops at color_p and all geometric
    ops at geo_p. Multi-scale stays off (would change tile shape)."""
    return {
        "geometric": {
            "rot90_p": geo_p, "hflip_p": geo_p, "vflip_p": geo_p,
            "shift_scale_rotate": {"shift": 0.1, "scale": 0.1, "rotate": 30, "p": geo_p},
            "elastic": {"alpha": 120, "sigma": 6, "p": 0.0},   # leave elastic off; deterministic shapes only
            "shear": {"shear_degrees": 10, "p": 0.0},
        },
        "color": {
            "brightness": 0.5, "contrast": 0.5, "saturation": 0.5,
            "brightness_contrast_p": color_p,
            "gaussian_noise": {"var_limit": [10, 50], "p": color_p},
            "clahe": {"clip_limit": 4.0, "tile_grid": [8, 8], "p": color_p},
        },
        "multi_scale": {"scale_range": [1.0, 1.0], "p": 0.0},
    }


def _make_inputs(seed: int):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    extra = rng.standard_normal((64, 64, 2)).astype(np.float32)
    mask = rng.integers(0, 2, size=(64, 64), dtype=np.uint8)
    return rgb, extra, mask


def test_color_aug_does_not_touch_extra():
    """With color_p=1.0 and geo_p=0.0, EXTRA must come back bit-identical."""
    rgb, extra, mask = _make_inputs(seed=42)

    transform = build_train_transforms(tile_size=64, aug_cfg=_aug_cfg(color_p=1.0, geo_p=0.0))
    out = transform(image=rgb, extra=extra, mask=mask)

    assert out["extra"].shape == extra.shape
    assert out["extra"].dtype == extra.dtype
    np.testing.assert_array_equal(out["extra"], extra,
        err_msg="EXTRA pixels were modified by color-only augmentation")
    # Mask is also untouched by color ops.
    np.testing.assert_array_equal(out["mask"], mask)
    # RGB SHOULD have changed (color_p=1.0 means at least one op fires).
    assert not np.array_equal(out["image"], rgb), \
        "RGB unchanged despite color_p=1.0 — color stage may not be running"


def test_geometric_aug_applies_to_extra_and_mask():
    """With geo_p=1.0 (HorizontalFlip among others), EXTRA and mask must transform together."""
    # Disable color so we can isolate the geometric effect.
    rgb, extra, mask = _make_inputs(seed=7)
    cfg = _aug_cfg(color_p=0.0, geo_p=0.0)
    cfg["geometric"]["hflip_p"] = 1.0   # only hflip fires; deterministic
    cfg["geometric"]["rot90_p"] = 0.0
    cfg["geometric"]["vflip_p"] = 0.0
    cfg["geometric"]["shift_scale_rotate"]["p"] = 0.0

    transform = build_train_transforms(tile_size=64, aug_cfg=cfg)
    out = transform(image=rgb, extra=extra, mask=mask)

    np.testing.assert_array_equal(out["image"], np.ascontiguousarray(rgb[:, ::-1, :]))
    np.testing.assert_array_equal(out["extra"], np.ascontiguousarray(extra[:, ::-1, :]))
    np.testing.assert_array_equal(out["mask"], np.ascontiguousarray(mask[:, ::-1]))


def test_extra_none_path_still_works():
    """Existing RGB-only call path (no extra kwarg) must still work."""
    rgb, _, mask = _make_inputs(seed=11)
    transform = build_train_transforms(tile_size=64, aug_cfg=_aug_cfg(color_p=0.0, geo_p=0.0))
    out = transform(image=rgb, mask=mask)
    np.testing.assert_array_equal(out["image"], rgb)
    np.testing.assert_array_equal(out["mask"], mask)
    assert "extra" not in out
```

- [ ] **Step 2: Run the new tests, expect FAIL on `test_color_aug_does_not_touch_extra`**

```bash
python -m pytest tests/test_transforms.py -v
```

Expected: `test_color_aug_does_not_touch_extra` FAILS (current implementation applies color to EXTRA via `additional_targets={"extra": "image"}`). The other two tests should PASS against the existing implementation.

- [ ] **Step 3: Replace `build_train_transforms` in `data/transforms.py`**

Replace the entire `build_train_transforms` function (current lines 26–79) with:

```python
class TrainTransform:
    """Two-stage augmentation: color-only on RGB, then geometric on RGB+EXTRA+mask.

    Designed to honor training.md §9.2: color/radiometric ops apply only to
    RGB; EXTRA and mask must not see brightness/contrast/saturation/noise/CLAHE.
    Geometric ops apply to all three.
    """

    def __init__(self, color_stage: A.Compose, geometric_stage: A.Compose):
        self._color = color_stage
        self._geo = geometric_stage

    def __call__(self, *, image, extra=None, mask):
        # Stage 1: color ops on RGB only. mask passed through (color stage
        # also includes mask as a target so the mask key flows through).
        color_out = self._color(image=image, mask=mask)
        rgb = color_out["image"]
        mask = color_out["mask"]
        # Stage 2: geometric ops on RGB+EXTRA+mask together.
        if extra is not None:
            geo_out = self._geo(image=rgb, extra=extra, mask=mask)
            return {"image": geo_out["image"], "extra": geo_out["extra"], "mask": geo_out["mask"]}
        geo_out = self._geo(image=rgb, mask=mask)
        return {"image": geo_out["image"], "mask": geo_out["mask"]}


def build_train_transforms(tile_size: int, aug_cfg: dict[str, Any]) -> TrainTransform:
    """Training-time augmentation. Returns a TrainTransform callable.

    Color stage runs on RGB only. Geometric + multi-scale stage runs on
    RGB+EXTRA+mask together via additional_targets.
    """
    geo = aug_cfg["geometric"]
    col = aug_cfg["color"]
    ms = aug_cfg["multi_scale"]

    color_stage = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=col["brightness"],
            contrast_limit=col["contrast"],
            p=col["brightness_contrast_p"],
        ),
        A.HueSaturationValue(
            sat_shift_limit=int(col["saturation"] * 100),
            p=col["brightness_contrast_p"],
        ),
        A.GaussNoise(
            var_limit=tuple(col["gaussian_noise"]["var_limit"]),
            p=col["gaussian_noise"]["p"],
        ),
        A.CLAHE(
            clip_limit=col["clahe"]["clip_limit"],
            tile_grid_size=tuple(col["clahe"]["tile_grid"]),
            p=col["clahe"]["p"],
        ),
    ])

    geometric_stage = A.Compose(
        [
            A.RandomRotate90(p=geo["rot90_p"]),
            A.HorizontalFlip(p=geo["hflip_p"]),
            A.VerticalFlip(p=geo["vflip_p"]),
            A.ShiftScaleRotate(
                shift_limit=geo["shift_scale_rotate"]["shift"],
                scale_limit=geo["shift_scale_rotate"]["scale"],
                rotate_limit=geo["shift_scale_rotate"]["rotate"],
                p=geo["shift_scale_rotate"]["p"],
                border_mode=0,
            ),
            A.ElasticTransform(
                alpha=geo["elastic"]["alpha"],
                sigma=geo["elastic"]["sigma"],
                p=geo["elastic"]["p"],
            ),
            A.Affine(shear=(-geo["shear"]["shear_degrees"], geo["shear"]["shear_degrees"]),
                     p=geo["shear"]["p"]),
            A.RandomScale(
                scale_limit=(ms["scale_range"][0] - 1.0, ms["scale_range"][1] - 1.0),
                p=ms["p"],
            ),
            A.PadIfNeeded(min_height=tile_size, min_width=tile_size, border_mode=0),
            A.CenterCrop(height=tile_size, width=tile_size),
        ],
        additional_targets={"extra": "image"},
    )

    return TrainTransform(color_stage, geometric_stage)
```

`build_eval_transforms()` stays as it is at line 82.

- [ ] **Step 4: Re-run the transforms tests, expect all PASS**

```bash
python -m pytest tests/test_transforms.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Re-run the full data + smoke test suite to confirm no regression**

```bash
python -m pytest tests/test_dataset.py tests/test_train_smoke.py -v
```

Expected: all tests PASS. (`RTSDataset.__getitem__` calls `self.transform(image=rgb, extra=extra, mask=label)` which `TrainTransform.__call__` accepts identically.)

- [ ] **Step 6: Update `tests/tests.md` to log the new tests**

Append to the appropriate section (or create a new "## test_transforms.py" subsection if none exists) — exact format should match the existing inventory style (look at how other test files are listed):

```markdown
| `test_color_aug_does_not_touch_extra` | EXTRA channels are bit-identical after color-only augmentation | real — Critical C3 |
| `test_geometric_aug_applies_to_extra_and_mask` | HorizontalFlip applies to RGB, EXTRA, and mask jointly | real |
| `test_extra_none_path_still_works` | RGB-only call path (no extra kwarg) is preserved | real |
| `test_init_raises_on_rgb_channel_name_mismatch` | RTSDataset refuses stats with permuted RGB names | real — Critical C1 |
| `test_init_raises_on_extra_channel_name_mismatch` | RTSDataset refuses stats with mis-ordered EXTRA names | real — Critical C1 |
```

(Adapt to the existing `tests/tests.md` table format. If it uses different columns or strictness ratings, match those.)

- [ ] **Step 7: Commit**

```bash
git add data/transforms.py tests/test_transforms.py tests/test_dataset.py tests/tests.md
git commit -m "data: split augmentation into color-only-RGB and stacked-geometric stages

Closes Critical C3 from the 2026-05-02 code review. training.md §9.2
mandates color/radiometric ops apply only to RGB; previous Compose
declared additional_targets={'extra': 'image'} which silently fed
brightness/contrast/saturation/noise/CLAHE to EXTRA bands.

build_train_transforms now returns a TrainTransform callable that
runs color-only Compose on RGB, then geometric+multi-scale Compose
on RGB+EXTRA+mask together. Call site in RTSDataset.__getitem__ is
unchanged.

Adds tests/test_transforms.py with 3 tests; the EXTRA-isolation
test fails against the old implementation."
```

---

## Task 4: I1 — document `clip_percentiles` config keys as unimplemented

**Why this matters:** `configs/baseline.yaml:79-83` declares `rgb_clip_percentiles: [0.1, 99.9]` and `extra_clip_percentiles: [0.1, 99.9]` as if outlier clipping is in effect. `scripts/compute_normalization_stats.py` never reads these keys; `data/normalization.py:WelfordChannelStats.update` consumes raw values. This is a documentation lie, not a code bug, but it will mislead the next reader. User decision: keep keys, mark them reserved.

**Files:**
- Modify: `configs/baseline.yaml:78-83` (the `normalization:` block)

- [ ] **Step 1: Edit `configs/baseline.yaml` to mark the keys reserved**

Replace lines 76-83:

```yaml
# ---------------------------------------------------------------------------
# Normalization (per-dataset stats; see data/data.md §5)
# ---------------------------------------------------------------------------
normalization:
  # One-off percentage clipping of outliers before mean/std computation.
  # Set after inspecting histograms with compute_normalization_stats.py --histograms-only.
  rgb_clip_percentiles: [0.1, 99.9]
  extra_clip_percentiles: [0.1, 99.9]
```

with:

```yaml
# ---------------------------------------------------------------------------
# Normalization (per-dataset stats; see data/data.md §5)
# ---------------------------------------------------------------------------
normalization:
  # RESERVED — currently unimplemented. compute_normalization_stats.py runs
  # Welford on raw values (no clipping). Keys retained so that adding
  # percentile-clipping in v2.1 does not require a config-schema change.
  # If real-data stats reveal saturation/zero-fill skew, implement clipping
  # in scripts/compute_normalization_stats.py and update data/data.md §5.
  rgb_clip_percentiles: [0.1, 99.9]
  extra_clip_percentiles: [0.1, 99.9]
```

- [ ] **Step 2: Verify the file still parses**

```bash
python -c "import yaml; yaml.safe_load(open('configs/baseline.yaml'))"
```

Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add configs/baseline.yaml
git commit -m "configs: mark normalization clip_percentiles keys as reserved/unimplemented

Closes Important I1 from the 2026-05-02 code review. The keys were
implying outlier protection that compute_normalization_stats.py does
not actually apply. Keeping the keys (with a comment) so v2.1
implementation does not require a schema change."
```

---

## Task 5: I5 — resume-flow regression test

**Why this matters:** The 2026-05-01 audit landed a fix for "EMA shadow weights silently fall back to live weights on resume" — a direct §10.2 violation. There is no automated test guarding the fix. A regression would be invisible until a real-data run resumed and validation silently used the wrong weights for the rest of training. Reviewer marked this as Important I5.

**Files:**
- Modify: `tests/test_train_smoke.py` (add `test_train_smoke_resume_then_continue`)
- Modify: `tests/tests.md` (log the new test)

- [ ] **Step 1: Add the resume test to `tests/test_train_smoke.py`**

Append at the end of `tests/test_train_smoke.py`:

```python
def test_train_smoke_resume_then_continue(synthetic_dataset, tmp_path, monkeypatch):
    """Resume from a 2-epoch run for 1 more epoch; assert EMA shadow is restored.

    Guards Important I5 from the 2026-05-02 code review and the underlying
    audit fix that restores EMA state on resume (was silently falling back to
    live weights — a direct §10.2 violation).
    """
    # First run: 2 epochs.
    cfg = _build_smoke_cfg(synthetic_dataset["root"], tmp_path / "mlruns")
    cfg["training"]["max_epochs"] = 2
    cfg_path = tmp_path / "smoke_initial.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "run_initial"
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--config", str(cfg_path),
        "--device", "cpu",
        "--out-dir", str(out_dir),
    ])
    import train  # scripts/train.py is on sys.path
    rc = train.main()
    assert rc == 0

    # Find the latest resume snapshot.
    resume_files = sorted((out_dir / "checkpoints").glob("resume_latest-*.pth"))
    assert resume_files, "no resume snapshot from initial run"
    resume_path = resume_files[-1]

    saved = torch.load(resume_path, map_location="cpu", weights_only=False)
    saved_ema_sd = saved.get("ema_state_dict")
    if saved_ema_sd is None:
        pytest.skip("EMA was not constructed in the initial run; resume-of-EMA path not exercised")

    # Second run: resume + 1 more epoch.
    cfg2 = dict(cfg)
    cfg2["training"]["max_epochs"] = 3   # +1 epoch beyond the resume point
    cfg2_path = tmp_path / "smoke_resume.yaml"
    cfg2_path.write_text(yaml.safe_dump(cfg2))

    out_dir2 = tmp_path / "run_resume"
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--config", str(cfg2_path),
        "--device", "cpu",
        "--out-dir", str(out_dir2),
        "--resume", str(resume_path),
    ])
    rc2 = train.main()
    assert rc2 == 0

    # Verify the resumed run's first-validation EMA matches the saved EMA
    # (i.e. resume restored the shadow, not silently fell back to live).
    # We assert this indirectly via the resume-snapshot the second run wrote:
    # right after resume, before the EMA decays again, the new resume snapshot's
    # first epoch must have an EMA close to the saved one.
    new_resume_files = sorted((out_dir2 / "checkpoints").glob("resume_latest-*.pth"))
    assert new_resume_files
    new_saved = torch.load(new_resume_files[-1], map_location="cpu", weights_only=False)
    new_ema_sd = new_saved["ema_state_dict"]
    assert new_ema_sd is not None, "EMA dropped after resume — regression of audit fix"

    # Floating-point parameter count should match between the two checkpoints.
    saved_keys = {k for k, v in saved_ema_sd.items() if v.dtype.is_floating_point}
    new_keys = {k for k, v in new_ema_sd.items() if v.dtype.is_floating_point}
    assert saved_keys == new_keys, "EMA state_dict keys changed across resume"

    # The EMA shadow at end-of-epoch-3 should differ from the one at end-of-epoch-2
    # (decay continued working) — strongest signal that EMA is alive, not stuck.
    diff_found = False
    for k in saved_keys:
        if not torch.equal(saved_ema_sd[k], new_ema_sd[k]):
            diff_found = True
            break
    assert diff_found, (
        "Post-resume EMA shadow is bit-identical to the saved one across an "
        "extra epoch of training — resume likely did not restart the EMA decay."
    )
```

- [ ] **Step 2: Run the new test, expect PASS (the audit fix is already in place)**

```bash
python -m pytest tests/test_train_smoke.py::test_train_smoke_resume_then_continue -v
```

Expected: PASS. (If it FAILS, the audit fix has regressed already — investigate `scripts/train.py:_resume_from` before continuing.)

- [ ] **Step 3: Run the full smoke suite to confirm no regression**

```bash
python -m pytest tests/test_train_smoke.py -v
```

Expected: all 7 (existing 6 + 1 new) PASS.

- [ ] **Step 4: Update `tests/tests.md`**

Add the new test to the `test_train_smoke.py` section (matching the existing inventory format):

```markdown
| `test_train_smoke_resume_then_continue` | Resume from epoch-2 snapshot for 1 more epoch; EMA shadow is restored and continues decaying | real — Important I5 |
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_train_smoke.py tests/tests.md
git commit -m "tests: add resume-then-continue smoke test

Closes Important I5 from the 2026-05-02 code review. Guards the
audit fix that restores EMA shadow weights on resume — a regression
would be invisible without a test (validation would silently use the
wrong weights for the rest of training)."
```

---

## Task 6: Final verification

**Files:** none modified; this is the gate before declaring the branch smoke-ready.

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/ext_rtsmapping_woodwellclimate_o/RTSmappingDL
python -m pytest tests/ -v
```

Expected: all tests PASS. New test count: 122 (prior) + 2 (Task 2) + 3 (Task 3) + 1 (Task 5) = 128.

If any test fails, stop and investigate before moving on. Do not "make it pass" by weakening the assertion — diagnose the root cause.

- [ ] **Step 2: Re-run the spec-drift grep gates from the 2026-05-01 audit**

```bash
grep -rn "stride.*245\|2-pass\|model\.input_size\|configs/inference\.yaml\|md5\|MD5" \
  configs/ data/ training/ inference/ scripts/ tests/ models/ losses/ utils/ \
  CLAUDE.md current_working_status.md 2>/dev/null | grep -v "^Binary file" || echo "OK — no matches"
```

Expected: no matches (the line `OK — no matches` should print). Anything else is drift; reconcile before declaring complete.

- [ ] **Step 3: Confirm the config matrix is exactly two files**

```bash
ls configs/
```

Expected output: `baseline.yaml  deployment.yaml`

- [ ] **Step 4: Confirm baseline.yaml's `output_bias_prior` is the audit-fixed value**

```bash
grep "output_bias_prior" configs/baseline.yaml
```

Expected: `  output_bias_prior: 0.005`. (Anything else means C2 is not actually dissolved.)

- [ ] **Step 5: Confirm `data/dataset.py` has the channel-name assertion**

```bash
grep -n "RGB channel_names\|EXTRA channel_names" data/dataset.py
```

Expected: at least two matches (the two `ValueError` messages from Task 2).

- [ ] **Step 6: Confirm `data/transforms.py` has the two-stage class**

```bash
grep -n "class TrainTransform\|color_stage\|geometric_stage" data/transforms.py
```

Expected: matches showing `class TrainTransform`, `_color`, `_geo` references.

- [ ] **Step 7: Verify `current_working_status.md` records the work**

```bash
grep "2026-05-02" current_working_status.md
```

Expected: at least one match (the dev-log entry from Task 1, Step 7).

- [ ] **Step 8: Final commit + branch status**

This task adds no new commits unless a verification step revealed a missed change. If everything passes, just verify the branch state:

```bash
git status
git log --oneline phase1-training-loop ^main | head -10
```

Expected: clean working tree; the new commits from Tasks 1–5 visible at the head.

The branch is now ready for the L4 real-data smoke test.

---

## Self-Review Checklist (run before declaring plan complete)

- [x] Every Critical from the 2026-05-02 code review (C1, C2, C3) has a task that closes it.
- [x] Every Important the user asked for (I1, I5) has a task that closes it.
- [x] Every task ends with a commit step.
- [x] Every code-changing step shows the actual code (no "implement validation" placeholders).
- [x] File paths in tests match the new structure (`tests/test_transforms.py` is new; others are extensions).
- [x] Function signatures referenced in later tasks match what earlier tasks define (`TrainTransform.__call__(image=, extra=, mask=)` matches the dataset.py call site at line 144).
- [x] Verification task at the end runs the full suite + drift gates + config-matrix sanity check.
- [x] Plan respects CLAUDE.md "flat-not-nested" — `TrainTransform` is one class, not a hierarchy; no factories.
