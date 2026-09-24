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
from collections.abc import Iterable

import numpy as np
from rasterio import windows
from rasterio.enums import Resampling

TIGHT_MIN_M, TIGHT_PAD = 250.0, 3.0    # tight view: 3× feature, ≥250 m
WIDE_MIN_M, WIDE_PAD = 1500.0, 10.0    # wide view: 10× feature, ≥1.5 km

# Other candidate polygons falling inside a crop are outlined in this colour so
# a reviewer can see the cluster the model found. Deliberately not red: the red
# outline marks *the* polygon under judgement and must stay the salient one.
# Cyan because it does not occur in Arctic terrain (browns, tans, greens, dark
# water), and because its red channel is low enough that the outline-toggle test
# in tests/test_review_crops.py still measures only the target's red.
NEIGHBOUR_COLOR = "#3fd0ff"

# Where the mosaic carries nothing, a boundless read fills 0 and the crop shows
# black — indistinguishable from dark ground or shadowed water once JPEG has
# had it, which is how the 2026-08 archive shipped part-blank context views that
# nobody questioned. Absence is drawn as an obvious synthetic stripe instead, so
# a reviewer can tell "nothing was imaged here" from "this is dark".
NO_DATA_DARK = (58, 61, 69)
NO_DATA_LIGHT = (78, 82, 91)
NO_DATA_STRIPE_PX = 8
# Below this the gap is a sliver at the frame edge; labelling it would nag.
NO_DATA_LABEL_MIN = 0.02

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
                outline: bool = True, neighbours: Iterable = ()) -> bytes:
    """Windowed read of the chip mosaic → JPEG bytes, optionally outlined.

    Args:
        src: an open rasterio dataset over the RGB chip mosaic (EPSG:3857).
        geoms: shapely geometries to outline in red.
        crop: square crop bounds from :func:`crop_bounds`.
        png_px: output edge length in pixels.
        outline: draw the red outline. False renders the bare imagery, which
            is what the app's outline toggle swaps to — the outline is burned
            into the pixels here, so it cannot be turned off client-side.
        neighbours: other candidate geometries inside ``crop``, drawn thinner
            and in :data:`NEIGHBOUR_COLOR` beneath ``geoms``. Suppressed along
            with the red outline when ``outline`` is False, so the unoutlined
            view stays completely unmarked. Last in the signature because
            `scripts/build_qc_rating_page.py` calls this positionally.

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
    rgb = np.moveaxis(img, 0, -1).astype(np.uint8).copy()
    gap = rgb.max(axis=-1) == 0
    if gap.any():
        ii, jj = np.nonzero(gap)
        stripe = ((ii + jj) // NO_DATA_STRIPE_PX) % 2 == 0
        rgb[ii[stripe], jj[stripe]] = NO_DATA_DARK
        rgb[ii[~stripe], jj[~stripe]] = NO_DATA_LIGHT

    fig = plt.figure(figsize=(png_px / 100, png_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, extent=(crop[0], crop[2], crop[1], crop[3]))
    if gap.mean() > NO_DATA_LABEL_MIN:
        ax.text(0.015, 0.015, f"NO IMAGERY  {gap.mean():.0%}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=max(6, png_px // 70), color="#d6dae3",
                bbox=dict(facecolor="#23262c", edgecolor="none", pad=2.0,
                          alpha=0.85))
    if outline:
        # Neighbours first, so the target's red is never drawn under them.
        for g in neighbours:
            parts = g.geoms if g.geom_type.startswith("Multi") else [g]
            for p in parts:
                x, y = p.exterior.xy
                ax.plot(x, y, color=NEIGHBOUR_COLOR, linewidth=0.9, alpha=0.7)
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


def imagery_fraction(src, crop: Bounds, probe_px: int = 32) -> float:
    """Fraction of ``crop`` the mosaic actually carries, 0.0–1.0.

    The wide view asks for ≥1.5 km of context, which can reach past the chip
    archive; the missing part renders as :func:`render_crop`'s black fill and
    is indistinguishable from dark ground once JPEG-compressed. Measuring it on
    the source pixels is the only honest way to know, and it is what
    :func:`has_imagery`'s boolean cannot say: *how much* is there.

    Args:
        src: an open rasterio dataset over the RGB chip mosaic (EPSG:3857).
        crop: square crop bounds from :func:`crop_bounds`.
        probe_px: edge length of the decimated read; 32 is enough to resolve
            a missing chip, which is never a thin sliver.

    Note:
        Reads every band and calls a pixel present if any band is non-zero, so
        it agrees exactly with the gap :func:`render_crop` stripes. The older
        :func:`has_imagery` probes band 1 alone; it is left that way because
        its answer only gates the all-or-nothing `no_imagery.csv` report.
    """
    win = windows.from_bounds(*crop, transform=src.transform)
    probe = src.read(window=win, out_shape=(src.count, probe_px, probe_px),
                     boundless=True, fill_value=0,
                     resampling=Resampling.nearest)
    return float((probe.max(axis=0) > 0).mean())
