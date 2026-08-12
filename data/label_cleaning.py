"""Label cleaning primitive (SSoT) — remove vector→raster artifacts from RTS labels.

Shared by `scripts/clean_labels.py` (writes cleaned label tiles for a new dataset
version) and `scripts/object_scorecard.py --gt-clean` (applies the identical cleaning
to cached GT at scoring time, for the free frozen-model re-score). One function, one
definition of "clean", so the training labels and the evaluation labels can never drift.

The operation, in order (order matters — see `clean_positive_label`):
  1. fill interior holes in the positive (==1) mask,
  2. small morphological close (bridge hairline fragment splits),
  3. drop connected components smaller than `min_size` (the Minimum Mapping Unit, MMU)
     → set those pixels to `removed_value` (255 = ignore, not background).

Design decisions (data-v1.1, from the team): removed slivers → **ignore (255)** not
background (don't penalise the model on ambiguous sub-MMU blobs); close radius small
(default 1) to avoid merging genuinely distinct adjacent slumps; existing 255 ignore is
preserved (close/fill only move 0↔1).

SCOPE — "MMU" here is a **ground-truth label** floor in **pixels**, applied before
scoring/training. It is NOT the product's minimum mapping unit, which is a **polygon
area** floor in **m²** (`vectorize_region.py --min-area-m2`). Nor is it a prediction
filter: `object_scorecard.py --min-mapping-unit` calls into here and filters GT, while
that script's `--min-blob` filters predictions. `data.min_mapping_unit_px` is **0 (OFF)**
in every shipped training config — the MMU is a scoring-time correction, not a training
change. Full disambiguation of the four size parameters that share these names:
`post-inference/south_products.md` §"Size parameters — which number is which".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage


@dataclass
class CleanStats:
    n_blobs_before: int
    n_blobs_after: int
    n_removed_blobs: int
    px_removed_to_ignore: int
    px_holes_filled: int
    px_bridged: int

    def as_dict(self) -> dict:
        return asdict(self)


def clean_positive_label(
    label: np.ndarray,
    *,
    min_size: int,
    close_radius: int = 1,
    fill_holes: bool = True,
    removed_value: int = 255,
    positive_value: int = 1,
) -> tuple[np.ndarray, CleanStats]:
    """Clean the positive class of one label tile. Returns (cleaned_label, stats).

    Order is deliberate: fill + close run **before** the size filter, so fragments of
    one real slump that rasterisation split get merged and the combined blob clears
    ``min_size`` — while genuinely isolated sub-MMU slivers are still removed. Existing
    ignore (``!= 0`` and ``!= positive_value``, e.g. 255) is preserved; fill/close only
    move background↔positive.

    Args:
        label: (H, W) int array with values {0 bg, positive_value RTS, 255 ignore}.
        min_size: MMU in pixels — positive blobs smaller than this are removed.
        close_radius: morphological-close iterations (0 = no close).
        fill_holes: fill interior holes in positive blobs.
        removed_value: value written where sub-MMU blobs are removed (255 = ignore).
        positive_value: the RTS class value (1).

    Returns:
        (cleaned_label, CleanStats). ``label`` is not mutated.
    """
    orig = label
    pos = orig == positive_value
    ignore_mask = (orig != 0) & (orig != positive_value)  # pre-existing 255 etc.

    n_blobs_before = int(ndimage.label(pos)[1])

    filled = ndimage.binary_fill_holes(pos) if fill_holes else pos.copy()
    px_holes_filled = int((filled & ~pos).sum())

    if close_radius > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        closed = ndimage.binary_closing(filled, structure=struct, iterations=close_radius)
    else:
        closed = filled
    # Bridged pixels: newly-positive due to fill+close (0→1 in the gaps).
    px_bridged = int((closed & ~filled).sum())

    lbl, n = ndimage.label(closed)          # components AFTER fill+close (fragments merged)
    if n > 0:
        sizes = ndimage.sum(closed, lbl, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= min_size
        kept_mask = keep[lbl]
        n_kept = int(keep[1:].sum())
        n_removed = n - n_kept
    else:
        kept_mask = np.zeros_like(closed)
        n_kept = 0
        n_removed = 0
    # Pixels of the closed-positive belonging to a removed (sub-MMU) blob → ignore.
    removed_pos = closed & ~kept_mask

    out = orig.copy()
    out[kept_mask] = positive_value          # cleaned positive (incl. filled/bridged)
    out[removed_pos] = removed_value         # sub-MMU slivers → ignore
    out[ignore_mask] = orig[ignore_mask]     # preserve pre-existing ignore verbatim

    stats = CleanStats(
        n_blobs_before=n_blobs_before,
        n_blobs_after=n_kept,
        n_removed_blobs=n_removed,
        px_removed_to_ignore=int(removed_pos.sum()),
        px_holes_filled=px_holes_filled,
        px_bridged=px_bridged,
    )
    return out, stats


def apply_min_mapping_unit(
    label: np.ndarray,
    mmu_px: int,
    *,
    ignore_index: int = 255,
    positive_value: int = 1,
) -> np.ndarray:
    """Relabel positive components below the Minimum Mapping Unit to ignore.

    A *pure size floor*: positive connected-components smaller than ``mmu_px`` are
    set to ``ignore_index`` (255) so the loss and every metric treat them
    identically as ignore (never a false negative, never a false positive). Unlike
    :func:`clean_positive_label`'s data-hygiene use, this deliberately does **no**
    hole-fill and **no** morphological close — it must not alter GT geometry or
    merge boundary-clipped edge tails into their neighbouring bodies.

    The Minimum Mapping Unit is the smallest object size we commit to
    mapping/scoring; sub-Minimum-Mapping-Unit positives are rasterization
    artefacts or the un-inferable tail of a slump whose body lives off-tile
    (see the data-v1.1 metric-correctness decision).

    Args:
        label: (H, W) int array with values {0 bg, positive_value RTS, 255 ignore}.
        mmu_px: Minimum Mapping Unit in pixels; ``<= 1`` is a no-op (returns the
            input unchanged — the reproducibility-preserving default).
        ignore_index: value written where sub-Minimum-Mapping-Unit blobs are removed.
        positive_value: the RTS class value.

    Returns:
        The cleaned label (a new array when ``mmu_px > 1``; the input itself when off).
    """
    if mmu_px <= 1:
        return label
    cleaned, _ = clean_positive_label(
        label,
        min_size=mmu_px,
        close_radius=0,
        fill_holes=False,
        removed_value=ignore_index,
        positive_value=positive_value,
    )
    return cleaned
