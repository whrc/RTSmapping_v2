"""Unit tests for the quad-mode drift check, using synthetic RGBA quads."""
import sys, importlib.util
from pathlib import Path
import numpy as np, rasterio, pytest
from rasterio.transform import from_bounds
REPO = Path("/w"); sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("cin", REPO/"scripts"/"check_inference_normalization.py")
cin = importlib.util.module_from_spec(spec); spec.loader.exec_module(cin)


def _quad(path, rgb_value, alpha_valid_frac=1.0, size=64):
    a = np.zeros((4, size, size), dtype=np.uint8)
    a[0], a[1], a[2] = rgb_value
    n_valid = int(size * alpha_valid_frac)
    a[3, :n_valid, :] = 255                      # rest is NoData
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size,
                       count=4, dtype="uint8",
                       transform=from_bounds(0, 0, 1, 1, size, size), crs="EPSG:3857") as d:
        d.write(a)


def test_nodata_is_excluded_from_stats(tmp_path):
    """Half a quad is NoData. Including those zeros would halve the mean --
    the exact way a coastal quad would fake radiometric drift."""
    p = tmp_path / "q.tif"; _quad(p, (100, 100, 100), alpha_valid_frac=0.5)
    stats = cin.compute_sample_stats(cin._iter_quad_arrays([str(p)]), n_channels=3)
    assert stats["mean"] == pytest.approx([100, 100, 100], abs=0.01)
    assert stats["std"] == pytest.approx([0, 0, 0], abs=0.01)


def test_fully_nodata_quad_contributes_nothing(tmp_path):
    good, empty = tmp_path / "g.tif", tmp_path / "e.tif"
    _quad(good, (50, 60, 70)); _quad(empty, (0, 0, 0), alpha_valid_frac=0.0)
    stats = cin.compute_sample_stats(cin._iter_quad_arrays([str(good), str(empty)]), n_channels=3)
    assert stats["mean"] == pytest.approx([50, 60, 70], abs=0.01)


def test_unreadable_quad_is_skipped_not_fatal(tmp_path):
    good = tmp_path / "g.tif"; _quad(good, (10, 20, 30))
    stats = cin.compute_sample_stats(
        cin._iter_quad_arrays([str(tmp_path/"missing.tif"), str(good)]), n_channels=3)
    assert stats["mean"] == pytest.approx([10, 20, 30], abs=0.01)


def test_drift_flags_a_real_shift():
    train = {"channel_names": ["R","G","B"], "mean": [54.3,54.4,30.5], "std": [35.9,27.5,27.9]}
    same = {"mean": [54.3,54.4,30.5], "std": [35.9,27.5,27.9], "n_pixels": 1}
    assert not cin.compute_drift(same, train, ["R","G","B"])["concerning"].any()
    shifted = {"mean": [54.3+0.6*35.9, 54.4, 30.5], "std": [35.9,27.5,27.9], "n_pixels": 1}
    d = cin.compute_drift(shifted, train, ["R","G","B"])
    assert bool(d.loc[d.channel=="R","concerning"].iloc[0])      # >0.5 sigma
    assert not d.loc[d.channel!="R","concerning"].any()
