"""Synthetic fixtures that mimic the v2.0 GCS layout on disk.

A few small tiles + metadata.csv + splits.yaml laid out exactly like the real
bucket, but with 64x64 rasters (not 512x512) for speed. Tests that care about
tile size pass --tile-size accordingly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

# Make `data/`, `utils/`, `scripts/` importable when running pytest from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Synthetic tile factory
# ---------------------------------------------------------------------------

SIZE = 64
CRS = "EPSG:3857"
TRANSFORM = from_origin(0, 0, 3.0, 3.0)  # 3m pixels in Mercator


def _write_rgb(path: Path, rng: np.random.Generator, with_positive: bool) -> None:
    arr = rng.integers(0, 256, size=(3, SIZE, SIZE), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=SIZE, height=SIZE, count=3,
        dtype="uint8", crs=CRS, transform=TRANSFORM, compress="lzw",
    ) as dst:
        dst.write(arr)


def _write_extra(path: Path, rng: np.random.Generator, n_bands: int) -> None:
    arr = rng.standard_normal((n_bands, SIZE, SIZE)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=SIZE, height=SIZE, count=n_bands,
        dtype="float32", crs=CRS, transform=TRANSFORM, compress="lzw",
    ) as dst:
        dst.write(arr)


def _write_label(path: Path, with_positive: bool) -> None:
    arr = np.zeros((SIZE, SIZE), dtype=np.uint8)
    if with_positive:
        arr[20:40, 20:40] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=SIZE, height=SIZE, count=1,
        dtype="uint8", crs=CRS, transform=TRANSFORM, compress="lzw",
    ) as dst:
        dst.write(arr, 1)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> dict:
    """Lay out a minimal v2.0-style dataset under tmp_path.

    Returns dict with keys: root, metadata_df, splits, tile_ids_by_split.
    """
    root = tmp_path / "v2.0"
    (root / "PLANET-RGB").mkdir(parents=True)
    (root / "EXTRA").mkdir()
    (root / "labels").mkdir()

    rng = np.random.default_rng(42)

    # 4 regions × 3 tiles each = 12 tiles. Mix of pos/neg.
    regions = ["region_A", "region_B", "region_C", "region_D"]
    rows = []
    for ri, region in enumerate(regions):
        for ti in range(3):
            tid = f"{ri}{ti:03d}"   # e.g. "0000", "0001", "1000" ...
            # Make half the tiles per region positive.
            is_pos = (ti < 2)  # 2 pos, 1 neg per region → 8 pos, 4 neg
            _write_rgb(root / "PLANET-RGB" / f"{tid}.tif", rng, is_pos)
            _write_extra(root / "EXTRA" / f"{tid}.tif", rng, n_bands=4)
            _write_label(root / "labels" / f"{tid}.tif", with_positive=is_pos)
            rows.append({
                "Tile_id": tid,
                "centroid_lat": 65.0 + ri,
                "centroid_lon": -150.0 + ti,
                "TrainClass": "Positive" if is_pos else "Negative",
                "RegionName": region,
                "UIDs": f"uid_{tid}" if is_pos else "",
            })
    df = pd.DataFrame(rows)
    df.to_csv(root / "metadata.csv", index=False)

    splits = {
        "train":           ["region_A", "region_B"],
        "val_balanced":    ["region_C"],
        "val_realistic":   ["region_C"],
        "test_realistic":  ["region_D"],
    }
    with (root / "splits.yaml").open("w") as f:
        yaml.safe_dump(splits, f)

    return {
        "root": str(root),
        "metadata_df": df,
        "splits": splits,
    }
