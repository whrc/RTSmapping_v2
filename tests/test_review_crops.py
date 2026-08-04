"""Unit tests for review/crops.py + the chip index in build_review_crops.py.

The chip index exists because reading a crop from the 29,850-source mosaic VRT
costs ~2 s of source scanning; the index lets each crop be read from a
micro-VRT of the chips it touches. That is only sound if the indexed bounds are
exactly the chips' bounds, so these tests check the index against rasterio's own
reading of the same files.

Spec: `post-inference/review_campaign.md` §4.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from review.crops import (TIGHT_MIN_M, WIDE_MIN_M, crop_bounds, has_imagery,
                          render_crop)
from scripts.build_review_crops import chip_index

CHIP_M = 512 * 4.777  # one 512 px chip at the deployment GSD


def _chip(path, minx, miny, fill: int) -> None:
    data = np.full((3, 512, 512), fill, dtype="uint8")
    with rasterio.open(
            path, "w", driver="GTiff", height=512, width=512, count=3,
            dtype="uint8", crs="EPSG:3857", nodata=0,
            transform=from_bounds(minx, miny, minx + CHIP_M, miny + CHIP_M,
                                  512, 512)) as dst:
        dst.write(data)


@pytest.fixture
def mosaic(tmp_path):
    """Two side-by-side chips (one bright, one all-NoData) plus their VRT."""
    a, b = tmp_path / "t0_0.tif", tmp_path / "t1_0.tif"
    _chip(a, 0.0, 0.0, 200)
    _chip(b, CHIP_M, 0.0, 0)  # nodata=0 → this chip is empty
    vrt = tmp_path / "chips.vrt"
    subprocess.run(["gdalbuildvrt", str(vrt), str(a), str(b)], check=True,
                   stdout=subprocess.DEVNULL)
    return vrt


# --- crop geometry --------------------------------------------------------
def test_crop_bounds_apply_the_minimum_context_floors():
    tight, wide = crop_bounds((100.0, 100.0, 110.0, 110.0))
    assert tight[2] - tight[0] == pytest.approx(TIGHT_MIN_M)
    assert wide[2] - wide[0] == pytest.approx(WIDE_MIN_M)


def test_crop_bounds_scale_with_a_large_feature():
    """Above the floor, the crop is a multiple of the feature, not a constant."""
    tight, wide = crop_bounds((0.0, 0.0, 1000.0, 1000.0))
    assert tight[2] - tight[0] == pytest.approx(3000.0)
    assert wide[2] - wide[0] == pytest.approx(10000.0)


def test_crop_bounds_are_square_and_centred():
    tight, _ = crop_bounds((0.0, 0.0, 400.0, 100.0))
    assert (tight[2] - tight[0]) == pytest.approx(tight[3] - tight[1])
    assert (tight[0] + tight[2]) / 2 == pytest.approx(200.0)
    assert (tight[1] + tight[3]) / 2 == pytest.approx(50.0)


# --- chip index -----------------------------------------------------------
def test_index_bounds_match_the_chips_own_bounds(mosaic):
    idx = chip_index(str(mosaic)).sort_values("minx", ignore_index=True)
    assert len(idx) == 2
    for row in idx.itertuples():
        with rasterio.open(row.path) as src:
            assert row.minx == pytest.approx(src.bounds.left, abs=1e-6)
            assert row.miny == pytest.approx(src.bounds.bottom, abs=1e-6)
            assert row.maxx == pytest.approx(src.bounds.right, abs=1e-6)
            assert row.maxy == pytest.approx(src.bounds.top, abs=1e-6)


def test_index_paths_are_absolute_and_readable(mosaic):
    """VRT source paths are relative to the VRT; workers need absolute ones."""
    for path in chip_index(str(mosaic))["path"]:
        assert path.startswith("/")
        with rasterio.open(path):
            pass


def test_index_lists_each_chip_once_not_once_per_band(mosaic):
    """Every band repeats the same source list; counting all of them triples it."""
    assert len(chip_index(str(mosaic))) == 2


# --- imagery probe --------------------------------------------------------
def test_has_imagery_is_true_over_a_populated_chip(mosaic):
    with rasterio.open(mosaic) as src:
        assert has_imagery(src, (10.0, 10.0, 1000.0, 1000.0))


def test_has_imagery_is_false_over_a_nodata_chip(mosaic):
    with rasterio.open(mosaic) as src:
        assert not has_imagery(src, (CHIP_M + 10, 10.0, CHIP_M + 1000, 1000.0))


def test_has_imagery_is_false_off_the_mosaic(mosaic):
    """Boundless reads fill with 0, which must read as absent, not as dark."""
    with rasterio.open(mosaic) as src:
        assert not has_imagery(src, (-9000.0, -9000.0, -8000.0, -8000.0))


# --- rendering ------------------------------------------------------------
def test_render_crop_returns_a_jpeg(mosaic):
    from shapely.geometry import box

    with rasterio.open(mosaic) as src:
        jpg = render_crop(src, [box(100, 100, 300, 300)],
                          (0.0, 0.0, 1000.0, 1000.0), png_px=64)
    assert jpg[:3] == b"\xff\xd8\xff"  # JPEG SOI marker


def test_outline_false_renders_the_same_view_without_the_red(mosaic):
    """The toggle's second copy: same imagery, no burned-in outline.

    Checked on pixels, not bytes — a JPEG that merely differs could differ for
    any reason. The outlined render must carry strongly red pixels that the
    plain one does not.
    """
    import io

    import numpy as np
    from PIL import Image
    from shapely.geometry import box

    geom, crop = box(300, 300, 700, 700), (0.0, 0.0, 1000.0, 1000.0)
    with rasterio.open(mosaic) as src:
        outlined = render_crop(src, [geom], crop, png_px=64)
        plain = render_crop(src, [geom], crop, png_px=64, outline=False)

    def n_red(jpg: bytes) -> int:
        a = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")).astype(int)
        return int(((a[..., 0] > 150) & (a[..., 1] < 90)
                    & (a[..., 2] < 90)).sum())

    assert n_red(outlined) > 0
    assert n_red(plain) == 0
