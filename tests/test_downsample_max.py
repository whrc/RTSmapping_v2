"""Exact block-max downsampler (scripts/downsample_max.py) for the browse
likelihood surface — replaces gdalwarp -r max, whose kernel bleeds NoData-edge
artifacts (values 251–254 > the 250 encoding ceiling) into the output.
GPU-free; small synthetic scaled_uint8 rasters.
"""

from __future__ import annotations

import numpy as np
import rasterio

from inference.writer import write_probability_tile
from scripts.downsample_max import downsample_max

RES = 1.0


def _write(tmp_path, arr):
    p = str(tmp_path / "prob.tif")
    h, w = arr.shape
    write_probability_tile(p, arr.astype(np.float32), (0.0, 0.0, float(w), float(h)),
                           dtype="scaled_uint8")
    return p


def test_block_max_and_nodata_semantics(tmp_path):
    """factor-20 blocks: max of valid pixels; all-NoData block stays 255;
    a mixed block ignores NoData; output never exceeds 250."""
    a = np.full((40, 40), -1.0)
    a[0:20, 0:20] = 0.2          # block (0,0): solid 0.2
    a[5, 5] = 0.9                #   ... with a 0.9 peak → max 0.9
    a[25, 25] = 0.5              # block (1,1): single valid pixel in NoData
    # block (0,1) and (1,0): all NoData
    src = _write(tmp_path, a)
    out = str(tmp_path / "max20.tif")
    downsample_max(src, out, factor=20, workers=2)
    with rasterio.open(out) as d:
        assert d.nodata == 255
        m = d.read(1)
    assert m.shape == (2, 2)
    assert m[0, 0] == 225        # 0.9 × 250
    assert m[1, 1] == 125        # 0.5 × 250
    assert m[0, 1] == 255 and m[1, 0] == 255
    assert m[m != 255].max() <= 250


def test_overviews_preserve_peaks_and_colormap_embedded(tmp_path):
    """Overview levels must be built with MAX resampling — a single sparse
    peak must survive to the coarsest level (the nearest-resampling defect
    that made likelihood_95m display blank when zoomed out) — and a color
    table must be embedded so the file opens colormapped."""
    a = np.full((400, 400), -1.0)
    a[3, 3] = 0.02                # near-zero background blob
    a[200, 200] = 1.0             # one sparse peak
    src = _write(tmp_path, a)
    out = str(tmp_path / "max2.tif")
    downsample_max(src, out, factor=2, workers=2)
    with rasterio.open(out) as d:
        assert d.overviews(1), "no overviews built"
        coarsest = d.overviews(1)[-1]
        ov = d.read(1, out_shape=(d.height // coarsest, d.width // coarsest))
        assert ov[ov != 255].max() == 250, "peak lost in overviews"
        cmap = d.colormap(1)
        assert cmap[250][0] > 150 and cmap[250][1] < 90   # deep red at 1.0
        assert min(cmap[0][:3]) > 230                     # near-white at 0


def test_non_divisible_edges_are_padded(tmp_path):
    """A 50×30 raster at factor 20 → 3×2 output; edge blocks use the partial
    window (padding must not fabricate values)."""
    a = np.full((50, 30), -1.0)
    a[45, 25] = 0.8              # lives in the bottom-right partial block
    src = _write(tmp_path, a)
    out = str(tmp_path / "max20.tif")
    downsample_max(src, out, factor=20, workers=2)
    with rasterio.open(out) as d:
        m = d.read(1)
        t = d.transform
    assert m.shape == (3, 2)
    assert m[2, 1] == 200        # 0.8 × 250
    assert (m != 255).sum() == 1
    assert t.a == 20.0 * RES     # output georeferencing scales by factor
