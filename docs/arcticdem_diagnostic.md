# ArcticDEM availability diagnostic — coverage and acquisition dates

_Run 2026-07-30. Facts only: measured numbers, no recommendations._

Reproduce with [scripts/arcticdem_diagnostic.py](../scripts/arcticdem_diagnostic.py):

```
# coverage (geopandas only — rts-train:v2)
python scripts/arcticdem_diagnostic.py --part coverage \
    --data-root /outputs/v1.0/data_local --out /outputs/v1.0/analysis/arcticdem_coverage.json
# dates + per-tile datamask (needs earthengine-api — rts-dataprep:v1)
python scripts/arcticdem_diagnostic.py --part dates \
    --data-root /outputs/v1.0/data_local --out /outputs/v1.0/analysis/arcticdem_dates.json \
    --allowlist-out /outputs/v1.0/staging/v1_splits/arcticdem_covered_tiles.txt
```

Source asset: `UMN/PGC/ArcticDEM/V4/2m_mosaic`, bands
`elevation, count, mad, mindate, maxdate, datamask`. `mindate`/`maxdate` are integer
days since 2000-01-01 (verified on a sample tile: `maxdate` 7463 → 2020.4).

`UMN/PGC/ArcticDEM/V4/2m` (strips) does not exist as a GEE asset; the V3 strip
collection does (260,741 images, with `acqDate1`/`acqDate2`). It was not needed —
the V4 mosaic's own date bands answer the question directly and per pixel.

---

## 1. Acquisition dates — the DEM predates the labels everywhere

Per-tile mean of `maxdate` (the most recent contributing strip) over each tile
footprint, all 22,259 v1.0 tiles, `reduceRegions` at 30 m.

| Statistic | Value |
|---|---|
| Tiles with any ArcticDEM data | 17,796 / 22,259 |
| Median `maxdate` year (all tiles) | **2020.6** |
| Median `maxdate` year (positive tiles) | 2021.2 |
| Fraction with `maxdate` ≥ 2024 | **0.0000** |
| Fraction with `maxdate` ≥ 2020 | 0.974 (positives) |
| `mindate` year range across tiles | 2008.1 – 2021.5 |

`maxdate` year histogram, tiles with DEM:

| Year | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 |
|---|---|---|---|---|---|---|---|---|---|---|
| Tiles | 4 | 5 | 9 | 56 | 164 | 568 | 778 | 1,573 | 7,358 | 7,281 |

No tile has an ArcticDEM observation later than 2021.5. Labels were refined on
2024 PlanetScope imagery, so the terrain under every tile was last observed
**3–4 years before the label epoch**.

## 2. Coverage — area

Area ratio of the domain pair in `domain/` (`*_ArcticDEM.geojson` is the domain
intersected with ArcticDEM coverage, per `domain/inference_domain.md`), measured in
EPSG:6931 (NSIDC EASE-Grid 2.0 North, equal-area).

| Domain | Area (km²) | With ArcticDEM (km²) | Fraction |
|---|---|---|---|
| `circumpolar_south_domain` (deployed) | 20,683,693 | 14,850,561 | **0.7180** |
| `circumpolar_domain` | 21,340,174 | 15,506,891 | 0.7267 |

## 3. Coverage — training tiles

Per-tile, from the mosaic's `datamask` (authoritative). A tile counts as covered
when the reducer returns a non-null value over its footprint.

| Group | n | With DEM | Fraction |
|---|---|---|---|
| All tiles | 22,259 | 17,796 | 0.7995 |
| **positive** | 1,718 | 1,718 | **1.0000** |
| **negative** | 20,541 | 16,078 | **0.7827** |
| train | 17,951 | 14,597 | 0.8132 |
| val_realistic | 2,151 | 1,469 | **0.6829** |
| test_realistic | 2,157 | 1,730 | 0.8020 |
| val_balanced | 2,151 | 1,469 | 0.6829 |

Positive tiles by split: train 1,513/1,513, val_realistic 98/98,
test_realistic 107/107 — all 1.0000.

`val_realistic` and `val_balanced` share the same 5 regions in `splits.yaml`, hence
identical rows.

### Polygon vs datamask

The polygon test (`--part coverage`, centroid `within`
`circumpolar_south_domain_ArcticDEM.geojson`) gives 17,555/22,259 = 0.7887 overall
but only 0.9400 for positives, against datamask's 1.0000. The polygon is the
domain **∩ ArcticDEM ∩ permafrost ∩ Planet**, so a centroid can fall outside it for
reasons unrelated to ArcticDEM. The datamask numbers in §3 are the ones that
describe DEM availability per tile.

## 4. Label correlation of the coverage mask

From §3: coverage is 1.0000 on positives and 0.7827 on negatives.

| Split | Tiles without DEM | Positives without DEM |
|---|---|---|
| train | 3,354 / 17,951 (0.1868) | 0 / 1,513 |
| val_realistic | 682 / 2,151 (0.3171) | 0 / 98 |
| test_realistic | 427 / 2,157 (0.1980) | 0 / 107 |

In `val_realistic`, 682 of 2,053 negatives (33.2%) have no ArcticDEM and no
positive tile lacks it.

Artifacts: `/mnt/outputs/v1.0/analysis/arcticdem_{coverage,dates}.json`;
DEM-covered tile-id allowlist (17,796 ids)
`/mnt/outputs/v1.0/staging/v1_splits/arcticdem_covered_tiles.txt`.
