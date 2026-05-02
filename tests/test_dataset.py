"""Unit tests for data/dataset.py — shape/dtype/channel flexibility/boundary handling."""

from __future__ import annotations

import numpy as np
import torch

from data.dataset import RTSDataset, parse_extra_spec
from data.splits import get_tile_ids, load_metadata, load_splits_yaml
from data.transforms import build_eval_transforms, dilate_label_boundary


BASE_AUG_CFG = {
    "geometric": {
        "rot90_p": 0.0, "hflip_p": 0.0, "vflip_p": 0.0,
        "shift_scale_rotate": {"shift": 0.1, "scale": 0.2, "rotate": 45, "p": 0.0},
        "elastic": {"alpha": 120, "sigma": 6, "p": 0.0},
        "shear": {"shear_degrees": 10, "p": 0.0},
    },
    "color": {
        "brightness": 0.2, "contrast": 0.2, "saturation": 0.2,
        "brightness_contrast_p": 0.0,
        "gaussian_noise": {"var_limit": [10, 50], "p": 0.0},
        "clahe": {"clip_limit": 4.0, "tile_grid": [8, 8], "p": 0.0},
    },
    "multi_scale": {"scale_range": [0.5, 1.0], "p": 0.0},
}


def test_parse_extra_spec_empty():
    assert parse_extra_spec(None) == []
    assert parse_extra_spec([]) == []


def test_parse_extra_spec_flexible_names():
    spec = parse_extra_spec([
        {"name": "ndvi", "band": 0},
        {"name": "some_custom_band", "band": 2},
    ])
    assert len(spec) == 2
    assert spec[0].name == "ndvi"
    assert spec[1].name == "some_custom_band"
    assert spec[1].band == 2


def test_parse_extra_spec_rejects_missing_keys():
    import pytest
    with pytest.raises(ValueError, match="'name' and 'band'"):
        parse_extra_spec([{"name": "ndvi"}])


def test_dataset_rejects_soft_labels(synthetic_dataset):
    """boundary_handling='soft_labels' is deferred to v2.1 (training.md §5.5);
    construction must raise rather than silently fall through to none."""
    import pytest
    ds = synthetic_dataset
    metadata = load_metadata(f"{ds['root']}/metadata.csv")
    splits = load_splits_yaml(f"{ds['root']}/splits.yaml")
    ids = get_tile_ids("train", metadata, splits)
    with pytest.raises(NotImplementedError, match="soft_labels"):
        RTSDataset(
            tile_ids=ids, metadata=metadata, data_root=ds["root"],
            rgb_dir="PLANET-RGB", extra_dir="EXTRA", labels_dir="labels",
            extra_channels=[], norm_stats_path=None,
            transform=build_eval_transforms(),
            tile_size=64,
            boundary_handling="soft_labels",
        )


def test_dataset_rejects_unknown_boundary_handling(synthetic_dataset):
    import pytest
    ds = synthetic_dataset
    metadata = load_metadata(f"{ds['root']}/metadata.csv")
    splits = load_splits_yaml(f"{ds['root']}/splits.yaml")
    ids = get_tile_ids("train", metadata, splits)
    with pytest.raises(ValueError, match="boundary_handling"):
        RTSDataset(
            tile_ids=ids, metadata=metadata, data_root=ds["root"],
            rgb_dir="PLANET-RGB", extra_dir="EXTRA", labels_dir="labels",
            extra_channels=[], norm_stats_path=None,
            transform=build_eval_transforms(),
            tile_size=64,
            boundary_handling="bogus",
        )


def test_dataset_rgb_only(synthetic_dataset):
    ds = synthetic_dataset
    metadata = load_metadata(f"{ds['root']}/metadata.csv")
    splits = load_splits_yaml(f"{ds['root']}/splits.yaml")
    ids = get_tile_ids("train", metadata, splits)

    dataset = RTSDataset(
        tile_ids=ids, metadata=metadata, data_root=ds["root"],
        rgb_dir="PLANET-RGB", extra_dir="EXTRA", labels_dir="labels",
        extra_channels=[], norm_stats_path=None,
        transform=build_eval_transforms(),
        tile_size=64,
    )
    item = dataset[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["image"].dtype == torch.float32
    assert item["label"].shape == (64, 64)
    assert item["label"].dtype == torch.int64
    assert isinstance(item["tile_id"], str)


def test_dataset_with_variable_extra(synthetic_dataset):
    """Vary the EXTRA channel selection via config — dataset adapts automatically."""
    ds = synthetic_dataset
    metadata = load_metadata(f"{ds['root']}/metadata.csv")
    splits = load_splits_yaml(f"{ds['root']}/splits.yaml")
    ids = get_tile_ids("train", metadata, splits)

    # Pick 2 of the 4 available EXTRA bands with arbitrary names.
    extra_channels = parse_extra_spec([
        {"name": "band_a", "band": 0},
        {"name": "band_c", "band": 2},
    ])
    dataset = RTSDataset(
        tile_ids=ids, metadata=metadata, data_root=ds["root"],
        rgb_dir="PLANET-RGB", extra_dir="EXTRA", labels_dir="labels",
        extra_channels=extra_channels, norm_stats_path=None,
        transform=build_eval_transforms(),
        tile_size=64,
    )
    item = dataset[0]
    assert item["image"].shape == (5, 64, 64)  # 3 RGB + 2 EXTRA


def test_dataset_label_values_in_set(synthetic_dataset):
    ds = synthetic_dataset
    metadata = load_metadata(f"{ds['root']}/metadata.csv")
    splits = load_splits_yaml(f"{ds['root']}/splits.yaml")
    ids = get_tile_ids("train", metadata, splits)

    dataset = RTSDataset(
        tile_ids=ids, metadata=metadata, data_root=ds["root"],
        rgb_dir="PLANET-RGB", extra_dir="EXTRA", labels_dir="labels",
        extra_channels=[], norm_stats_path=None,
        transform=build_eval_transforms(),
        tile_size=64,
    )
    for i in range(len(dataset)):
        label = dataset[i]["label"].numpy()
        assert set(np.unique(label).tolist()).issubset({0, 1, 255})


def test_boundary_dilation_adds_ignore():
    label = np.zeros((32, 32), dtype=np.uint8)
    label[10:20, 10:20] = 1
    out = dilate_label_boundary(label, width=2, ignore_index=255)
    assert (out == 255).sum() > 0
    # Interior of the square should still be 1.
    assert (out[14:16, 14:16] == 1).all()


# ---------------------------------------------------------------------------
# C1 (2026-05-02 review): channel-name binding asserted at training load.
# training.md §4.5 mandates these checks; they were missing on the training side.
# ---------------------------------------------------------------------------


def test_init_raises_on_rgb_channel_name_mismatch(synthetic_dataset, tmp_path):
    """RTSDataset must refuse stats whose RGB channel_names != ['R', 'G', 'B']."""
    import json

    import pytest

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

    metadata = load_metadata(f"{synthetic_dataset['root']}/metadata.csv")
    splits = load_splits_yaml(f"{synthetic_dataset['root']}/splits.yaml")
    ids = get_tile_ids("train", metadata, splits)

    with pytest.raises(ValueError, match=r"RGB channel_names"):
        RTSDataset(
            tile_ids=ids[:1],
            metadata=metadata,
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

    import pytest

    from data.dataset import ExtraChannel

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

    metadata = load_metadata(f"{synthetic_dataset['root']}/metadata.csv")
    splits = load_splits_yaml(f"{synthetic_dataset['root']}/splits.yaml")
    ids = get_tile_ids("train", metadata, splits)

    with pytest.raises(ValueError, match=r"EXTRA channel_names"):
        RTSDataset(
            tile_ids=ids[:1],
            metadata=metadata,
            data_root=synthetic_dataset["root"],
            rgb_dir="PLANET-RGB",
            extra_dir="EXTRA",
            labels_dir="labels",
            extra_channels=extra_channels,
            norm_stats_path=str(bad_path),
            transform=build_eval_transforms(),
            tile_size=64,
        )
