"""Crop geometry + rendering shared by every RTS review surface.

One definition of "what a reviewer sees for a polygon", so the offline pack
(`scripts/build_qc_rating_page.py`) and the campaign app
(`scripts/build_review_crops.py` → `review/app.py`) show pixel-identical views.
Two crops per polygon: a tight one (~3× the feature) to judge the object and a
wide one (~1.5 km) to judge its context.

Spec: `post-inference/review_campaign.md` §4.
"""

from __future__ import annotations

import io

import numpy as np
from rasterio import windows
from rasterio.enums import Resampling

TIGHT_MIN_M, TIGHT_PAD = 250.0, 3.0    # tight view: 3× feature, ≥250 m
WIDE_MIN_M, WIDE_PAD = 1500.0, 10.0    # wide view: 10× feature, ≥1.5 km

Bounds = tuple[float, float, float, float]


def crop_bounds(b: Bounds) -> tuple[Bounds, Bounds]:
    """(tight, wide) square crop bounds centred on a feature's bbox.

    Args:
        b: feature bounds ``(minx, miny, maxx, maxy)`` in EPSG:3857.

    Returns:
        The tight and wide square crop bounds, same CRS.
    """
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    ext = max(b[2] - b[0], b[3] - b[1])

    def sq(side: float) -> Bounds:
        h = side / 2
        return (cx - h, cy - h, cx + h, cy + h)

    return (sq(max(TIGHT_MIN_M, ext * TIGHT_PAD)),
            sq(max(WIDE_MIN_M, ext * WIDE_PAD)))


def render_crop(src, geoms, crop: Bounds, png_px: int = 560,
                outline: bool = True) -> bytes:
    """Windowed read of the chip mosaic → JPEG bytes, optionally outlined.

    Args:
        src: an open rasterio dataset over the RGB chip mosaic (EPSG:3857).
        geoms: shapely geometries to outline in red.
        crop: square crop bounds from :func:`crop_bounds`.
        png_px: output edge length in pixels.
        outline: draw the red outline. False renders the bare imagery, which
            is what the app's outline toggle swaps to — the outline is burned
            into the pixels here, so it cannot be turned off client-side.

    Returns:
        JPEG-encoded image bytes. JPEG, not PNG: photographic chips compress
        ~7× smaller, which is what keeps both the single-file offline page and
        the 120k-object crop archive manageable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    win = windows.from_bounds(*crop, transform=src.transform)
    img = src.read(out_shape=(src.count, png_px, png_px), window=win,
                   boundless=True, fill_value=0,
                   resampling=Resampling.bilinear)
    fig = plt.figure(figsize=(png_px / 100, png_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.moveaxis(img, 0, -1), extent=(crop[0], crop[2], crop[1],
                                               crop[3]))
    if outline:
        for g in geoms:
            parts = g.geoms if g.geom_type.startswith("Multi") else [g]
            for p in parts:
                x, y = p.exterior.xy
                ax.plot(x, y, color="red", linewidth=1.4)
    ax.set_xlim(crop[0], crop[2])
    ax.set_ylim(crop[1], crop[3])
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=100, pil_kwargs={"quality": 82})
    plt.close(fig)
    return buf.getvalue()


def has_imagery(src, crop: Bounds) -> bool:
    """True if the mosaic carries any non-NoData pixel over ``crop``.

    Checked on the source pixels, not the rendered JPEG: the burned-in red
    outline and its antialiasing make a rendered "empty" crop indistinguishable
    from a dark real one. A cheap 32×32 decimated read is enough to answer it.
    """
    win = windows.from_bounds(*crop, transform=src.transform)
    probe = src.read(1, window=win, out_shape=(32, 32), boundless=True,
                     fill_value=0, resampling=Resampling.nearest)
    return bool(probe.max() > 0)
