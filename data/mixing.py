"""Sample-mixing augmentations (copy-paste / mosaic / cutmix / mixup).

These need access to *more than one* tile, so — unlike the per-sample albumentations
pipeline in `data/transforms.py` — they live here and are applied inside
`RTSDataset.__getitem__` on the raw `(rgb, extra, label)` arrays BEFORE boundary
dilation + the geometric/colour transform (so pasted/mixed pixels get the same
downstream treatment). All ops are **compositional / geometric** (no photometric
shadow-cue scrambling), so they are domain-safe for RTS (training.md §9.2 / campaign
plan family F). Every op is **config-gated and default-off** (`p=0` ⇒ identity), so
existing runs are unchanged until an arm is explicitly enabled.

Rare-object motivation: only ~1,513 positive tiles → copy-paste (paste real RTS
instances onto other tiles) and mosaic (raise per-image RTS density) are the
top data-limit levers.

Arrays: `rgb` (H,W,3) uint8 · `extra` (H,W,N) float32 or None · `label` (H,W) uint8
in {0, 1, ignore_index}. Source tiles are supplied by a `sample_fn` callback so this
module stays free of I/O and is unit-testable with synthetic arrays.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage

# A sampler returns one source tile: (rgb HWC uint8, extra HWC float32 | None, label HW uint8).
SampleFn = Callable[[bool], tuple[np.ndarray, "np.ndarray | None", np.ndarray]]


def _blend(dst: np.ndarray, src: np.ndarray, alpha) -> np.ndarray:
    """alpha-blend src onto dst (alpha broadcast over channels); dtype preserved."""
    a = alpha[..., None] if (dst.ndim == 3 and np.ndim(alpha) == 2) else alpha
    out = dst.astype(np.float32) * (1.0 - a) + src.astype(np.float32) * a
    return out.astype(dst.dtype)


def copy_paste(
    rgb, extra, label, src_rgb, src_extra, src_label, rng,
    *, max_instances: int = 3, blend_sigma: float = 2.0, ignore_index: int = 255,
):
    """Paste up to `max_instances` RTS instances from a source tile onto the target.

    Instances are connected components of `src_label == 1`. Each is pasted at a random
    in-bounds offset with a Gaussian-feathered alpha edge (avoids hard-seam shortcuts).
    The target label is set to 1 under the pasted instance; existing ignore pixels are
    preserved. EXTRA channels are pasted in lockstep when present.
    """
    H, W = label.shape
    inst, n = ndimage.label(src_label == 1)
    if n == 0:
        return rgb, extra, label
    rgb, label = rgb.copy(), label.copy()
    extra = None if extra is None else extra.copy()
    k = int(rng.integers(1, max_instances + 1))
    for inst_id in rng.permutation(np.arange(1, n + 1))[:k]:
        ys, xs = np.where(inst == inst_id)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        ph, pw = y1 - y0, x1 - x0
        if ph >= H or pw >= W:
            continue  # instance larger than the tile — skip
        mask = (inst[y0:y1, x0:x1] == inst_id).astype(np.float32)
        alpha = ndimage.gaussian_filter(mask, sigma=blend_sigma)
        alpha = np.clip(alpha, 0.0, 1.0) * mask  # feather inward only; 0 outside the instance
        oy = int(rng.integers(0, H - ph + 1))
        ox = int(rng.integers(0, W - pw + 1))
        sl = (slice(oy, oy + ph), slice(ox, ox + pw))
        rgb[sl] = _blend(rgb[sl], src_rgb[y0:y1, x0:x1], alpha)
        if extra is not None and src_extra is not None:
            extra[sl] = _blend(extra[sl], src_extra[y0:y1, x0:x1], alpha)
        placed = mask > 0.5
        keep = label[sl] != ignore_index           # don't overwrite ignore pixels
        label[sl] = np.where(placed & keep, np.uint8(1), label[sl])
    return rgb, extra, label


def mosaic(tiles, rng, tile_size: int, ignore_index: int = 255):
    """2×2 montage of 4 tiles on a 2S canvas, then a random S×S crop near the seam.

    Raises per-image RTS density when the 4 tiles are positive-biased (the caller picks).
    """
    S = tile_size
    canvas_rgb = np.zeros((2 * S, 2 * S, 3), dtype=np.uint8)
    n_extra = None if tiles[0][1] is None else tiles[0][1].shape[2]
    canvas_extra = None if n_extra is None else np.zeros((2 * S, 2 * S, n_extra), np.float32)
    canvas_lab = np.zeros((2 * S, 2 * S), dtype=np.uint8)
    quad = [(0, 0), (0, S), (S, 0), (S, S)]
    for (r, c), (t_rgb, t_extra, t_lab) in zip(quad, tiles):
        canvas_rgb[r:r + S, c:c + S] = t_rgb
        canvas_lab[r:r + S, c:c + S] = t_lab
        if canvas_extra is not None and t_extra is not None:
            canvas_extra[r:r + S, c:c + S] = t_extra
    cy = int(rng.integers(S // 2, S + S // 2 + 1))
    cx = int(rng.integers(S // 2, S + S // 2 + 1))
    y0, x0 = cy - S // 2, cx - S // 2
    sl = (slice(y0, y0 + S), slice(x0, x0 + S))
    extra_out = None if canvas_extra is None else canvas_extra[sl]
    return canvas_rgb[sl].copy(), (None if extra_out is None else extra_out.copy()), canvas_lab[sl].copy()


def cutmix(rgb, extra, label, src_rgb, src_extra, src_label, rng, *, ignore_index: int = 255):
    """Replace a random rectangle of the target with the same-coords patch from a source."""
    H, W = label.shape
    lam = float(rng.uniform(0.2, 0.6))            # patch area fraction
    rh, rw = int(H * np.sqrt(lam)), int(W * np.sqrt(lam))
    if rh == 0 or rw == 0:
        return rgb, extra, label
    y0 = int(rng.integers(0, H - rh + 1)); x0 = int(rng.integers(0, W - rw + 1))
    sl = (slice(y0, y0 + rh), slice(x0, x0 + rw))
    rgb, label = rgb.copy(), label.copy()
    extra = None if extra is None else extra.copy()
    rgb[sl] = src_rgb[sl]; label[sl] = src_label[sl]
    if extra is not None and src_extra is not None:
        extra[sl] = src_extra[sl]
    return rgb, extra, label


def mixup(rgb, extra, label, src_rgb, src_extra, src_label, rng, *, alpha: float = 0.2,
          ignore_index: int = 255):
    """Convex pixel blend of two tiles. Label = union of positives; ignore if either ignore
    (mixup masks are ill-defined → take the conservative union/ignore rule)."""
    lam = float(rng.beta(alpha, alpha))
    rgb_out = _blend(rgb, src_rgb, np.float32(1.0 - lam))
    extra_out = extra if (extra is None or src_extra is None) else _blend(extra, src_extra, np.float32(1.0 - lam))
    ign = (label == ignore_index) | (src_label == ignore_index)
    lab = np.maximum((label == 1), (src_label == 1)).astype(np.uint8)
    lab[ign] = ignore_index
    return rgb_out, extra_out, lab


class MixingAugmenter:
    """Stochastically applies at most one mixing op per sample, per config probabilities.

    `sample_fn(positive_only)` returns one source tile; the dataset supplies it as a
    closure over its tile reads. Default config (all p=0) ⇒ identity passthrough.
    """

    def __init__(self, aug_cfg: dict, sample_fn: SampleFn, tile_size: int, ignore_index: int = 255):
        m = (aug_cfg or {}).get("mixing", {}) or {}
        self.cp = m.get("copy_paste", {}) or {}
        self.mosaic = m.get("mosaic", {}) or {}
        self.cutmix = m.get("cutmix", {}) or {}
        self.mixup = m.get("mixup", {}) or {}
        self.sample_fn = sample_fn
        self.tile_size = tile_size
        self.ignore_index = ignore_index

    @property
    def enabled(self) -> bool:
        return any(float(c.get("p", 0.0)) > 0.0 for c in (self.cp, self.mosaic, self.cutmix, self.mixup))

    def __call__(self, rgb, extra, label, rng):
        # priority order; at most one op fires per sample (they conflict spatially)
        if float(self.cp.get("p", 0.0)) > 0.0 and rng.random() < self.cp["p"]:
            s_rgb, s_extra, s_lab = self.sample_fn(True)
            return copy_paste(rgb, extra, label, s_rgb, s_extra, s_lab, rng,
                              max_instances=int(self.cp.get("max_instances", 3)),
                              blend_sigma=float(self.cp.get("blend_sigma", 2.0)),
                              ignore_index=self.ignore_index)
        if float(self.mosaic.get("p", 0.0)) > 0.0 and rng.random() < self.mosaic["p"]:
            tiles = [(rgb, extra, label)] + [self.sample_fn(True) for _ in range(3)]
            return mosaic(tiles, rng, self.tile_size, self.ignore_index)
        if float(self.cutmix.get("p", 0.0)) > 0.0 and rng.random() < self.cutmix["p"]:
            s_rgb, s_extra, s_lab = self.sample_fn(False)
            return cutmix(rgb, extra, label, s_rgb, s_extra, s_lab, rng, ignore_index=self.ignore_index)
        if float(self.mixup.get("p", 0.0)) > 0.0 and rng.random() < self.mixup["p"]:
            s_rgb, s_extra, s_lab = self.sample_fn(False)
            return mixup(rgb, extra, label, s_rgb, s_extra, s_lab, rng,
                         alpha=float(self.mixup.get("alpha", 0.2)), ignore_index=self.ignore_index)
        return rgb, extra, label
