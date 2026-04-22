"""BalancedBatchSampler with curriculum-aware pos:neg ratio.

Per training.md §7:
  - Each batch has (pos_per_batch) positive tiles and (neg_per_batch) negative tiles.
  - Negatives-per-positive ratio steps at epoch boundaries per curriculum_schedule.
  - Deterministic: shuffle seeded from (seed + epoch).

Usage:
    sampler = BalancedBatchSampler(
        tile_ids=train_ids,
        metadata=metadata,
        batch_size=32,
        schedule=cfg["sampling"]["curriculum_schedule"],
        seed=cfg["seed"],
    )
    for epoch in range(max_epochs):
        sampler.set_epoch(epoch)
        for batch_indices in sampler:
            ...

Yields lists of dataset-index positions, so wrap with torch.utils.data.DataLoader
as batch_sampler=sampler.
"""

from __future__ import annotations

import logging
import random
from typing import Iterator

import pandas as pd
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


def parse_curriculum_schedule(schedule: dict[str, int]) -> list[tuple[int, int, int]]:
    """Convert {'1-10': 1, '11-30': 5, ...} → [(lo, hi, ratio), ...] sorted by lo."""
    parsed = []
    for key, ratio in schedule.items():
        lo_str, hi_str = key.split("-")
        parsed.append((int(lo_str), int(hi_str), int(ratio)))
    parsed.sort()
    return parsed


def ratio_for_epoch(schedule: list[tuple[int, int, int]], epoch: int) -> int:
    """Return neg:pos ratio for a 1-indexed epoch. Clamps to first/last bucket if out of range."""
    for lo, hi, ratio in schedule:
        if lo <= epoch <= hi:
            return ratio
    # Out of range — use nearest bucket.
    if epoch < schedule[0][0]:
        return schedule[0][2]
    return schedule[-1][2]


class BalancedBatchSampler(Sampler[list[int]]):
    """Yields lists of dataset-position indices, one per batch.

    Each batch interleaves positives and negatives at the current curriculum ratio.
    For ratio r: batch has ~1/(1+r) positives and ~r/(1+r) negatives.
    """

    def __init__(
        self,
        tile_ids: list[str],
        metadata: pd.DataFrame,
        batch_size: int,
        schedule: dict[str, int],
        seed: int = 42,
        epoch: int = 1,
        drop_last: bool = True,
    ):
        self.tile_ids = list(tile_ids)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.drop_last = drop_last
        self._schedule = parse_curriculum_schedule(schedule)

        # Map tile_id → dataset index (the list order) and classify positives/negatives.
        id_to_idx = {tid: i for i, tid in enumerate(self.tile_ids)}
        class_by_id = metadata.set_index("Tile_id")["TrainClass"].to_dict()

        self._pos: list[int] = []
        self._neg: list[int] = []
        for tid, idx in id_to_idx.items():
            cls = class_by_id.get(tid)
            if cls == "Positive":
                self._pos.append(idx)
            elif cls == "Negative":
                self._neg.append(idx)
            else:
                raise ValueError(f"Tile {tid} has TrainClass {cls!r}, expected Positive/Negative")

        if not self._pos or not self._neg:
            raise ValueError(
                f"BalancedBatchSampler needs both classes; got {len(self._pos)} pos, "
                f"{len(self._neg)} neg"
            )

    def set_epoch(self, epoch: int) -> None:
        """Caller must invoke this before each epoch so the ratio advances deterministically."""
        self.epoch = int(epoch)

    def _current_ratio(self) -> int:
        return ratio_for_epoch(self._schedule, self.epoch)

    def _batch_split(self) -> tuple[int, int]:
        """Return (n_pos, n_neg) per batch given the current ratio."""
        ratio = self._current_ratio()
        n_pos = max(1, self.batch_size // (1 + ratio))
        n_neg = self.batch_size - n_pos
        return n_pos, n_neg

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)

        pos_pool = list(self._pos)
        neg_pool = list(self._neg)
        rng.shuffle(pos_pool)
        rng.shuffle(neg_pool)

        n_pos_per_batch, n_neg_per_batch = self._batch_split()
        # An epoch = one pass over the positives at this ratio. Negatives are
        # sampled with replacement if exhausted (there are usually many more).
        n_batches = len(pos_pool) // n_pos_per_batch
        if not self.drop_last and len(pos_pool) % n_pos_per_batch:
            n_batches += 1

        pi = ni = 0
        for _ in range(n_batches):
            # Slice positives; refill with reshuffle if we ever run out (rare — drop_last=True by default).
            if pi + n_pos_per_batch > len(pos_pool):
                rng.shuffle(pos_pool)
                pi = 0
            pos_chunk = pos_pool[pi : pi + n_pos_per_batch]
            pi += n_pos_per_batch

            # Slice negatives, refill by reshuffle when exhausted.
            if ni + n_neg_per_batch > len(neg_pool):
                rng.shuffle(neg_pool)
                ni = 0
            neg_chunk = neg_pool[ni : ni + n_neg_per_batch]
            ni += n_neg_per_batch

            batch = pos_chunk + neg_chunk
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        n_pos_per_batch, _ = self._batch_split()
        return len(self._pos) // n_pos_per_batch
