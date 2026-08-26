# Evaluating Earth Search / S3 as the S2 source

**Question (2026-08-26):** replace the Earth Engine export with windowed COG reads from
Element 84's Earth Search, per a colleague's `download_many()` script?

**Verdict: the *source* is right and probably the way out of the quota wall. The *script*
is not a drop-in — it is a scene downloader, and what we need is a compositor.** Everything
below is measured, not read off a datasheet.

## Why the source is attractive

Same ESA L2A product, public COGs in `sentinel-cogs` (AWS Open Data, us-west-2), no token
signing, no EECU, no quota. Measured from this VM:

| | |
|---|---|
| open a remote COG | 0.46 s |
| 1 stream, windowed read | 13 MB/s |
| 8 / 32 / 64 threads | 34 / 55 / **134 MB/s** (still climbing — not saturated) |

Volume we would actually pull, for **red + NIR only** (see below) plus SCL, ~14 clear
overpasses/cell, 1,799 cells:

| | uncompressed |
|---|---|
| one year | 12.7 TB (2022) – 18.0 TB (2019) |
| **six years** | **~89 TB** |

At the measured 134 MB/s that is **~7.7 days** of pulling for all six years; on a
better-provisioned VM, less. Against Earth Engine's **38 months** on the Contributor tier
([`ee_quota.md`](ee_quota.md)), that is the entire argument.

## What the script does not do

It downloads scenes. Our product is a **cloud-masked Jul–Sep median composite**, one
GeoTIFF per 1°×3° cell in EPSG:3857 (`data/extra_channels.s2_sr_composite`). The gap:

| needed | in the script? |
|---|---|
| per-pixel cloud mask | ✗ — scene-level `eo:cloud_cover` only; SCL not fetched |
| median over the season | ✗ — writes each scene separately |
| mosaic across MGRS tiles | ✗ — "nothing is mosaicked" (6 tiles/cell, measured) |
| mosaic across **UTM zones** | ✗ — one box returned both `5WPS` and `6WVB` |
| reproject to EPSG:3857 | ✗ — deliberately "never warp" |
| dedupe reprocessed scenes | ✗ — **see below** |
| resume / shard / QC | ✗ |

That is the half Earth Engine was doing for us. It is buildable, but it is a pipeline plus
tests plus a validation campaign — not a swap.

## Two concrete bugs for *our* use

**1. Reprocessed duplicates are not deduped — and 2019–2021 are full of them.**
ESA's Collection-1 reprocessing republished the archive under new baselines. For one 3°×1°
Arctic box, Jul–Sep, cloud < 20:

```
2019: 186 items / 6 tiles / 57 tile-days -> 43 of 57 tile-days have TWO items
      S2A_5WPS_20190928_0_L2A  baseline 02.13  offset  0
      S2A_5WPS_20190928_1_L2A  baseline 05.00  offset -0.1
2022:  67 items / 6 tiles                     -> no duplicates (all baseline 04.00)
```

Consequences: ~75% wasted bandwidth in 2019; the same overpass entering the median
**twice**, silently double-weighting whichever dates happened to be reprocessed; and because
the output dir is keyed on `{datetime}_{tile}` with an `os.path.exists` skip, which version
wins depends on cloud-cover sort order. Deduping to one item per (tile, acquisition) is
mandatory, and *preferring the higher baseline* is the defensible rule.

Note this asymmetry falls exactly across our comparison years — 2019–2021 duplicated,
2022+ not — so getting it wrong would manufacture an interannual signal.

**2. `_scale_offset`'s silent fallback is one schema migration from firing.**
The BOA offset handling is *correct today* — I verified `raster:bands[0] = {scale: 0.0001,
offset: -0.1}` is served, and the script applies it per item, which correctly handles 2019's
mixed 02.13 (offset 0) / 05.00 (offset −0.1) scenes. But the `except` branch returns
`offset = 0.0` silently. Earth Search also exposes a STAC 1.1 `bands` field (currently
`None`); if `raster:bands` is ever dropped, every post-2022 scene shifts by 1000 DN = 0.1
reflectance **with no error**, producing a fake 2021→2022 step in NDVI. Any port of this must
**fail loudly** on missing offset metadata rather than default to zero.

## The scientific risk: the mask changes

Earth Search serves **no QA60**. Assets are `scl` (20 m) and the reflectance bands — I
listed them. Our recipe masks with QA60; SCL is arguably a *better* mask, but it is a
**different** one, so the composite, the NDVI, and therefore the model's input distribution
all shift.

**The model is frozen.** It was trained and validated on QA60-masked EE composites, and the
delivered 2025 map was produced from them. Swapping the mask silently violates
training–inference consistency (CLAUDE Rule 3). It may well be fine — but it has to be
demonstrated, not assumed.

## The gate before any switch

We already own the answer: `gs://rts-mapping-v2-usw1/S2_RGB/2025_south/` is 1,799 cells of
EE-composited truth for the same recipe.

> Rebuild ~10 well-chosen 2025 cells (a cloudy one, a snowy one, a coastal one, one
> spanning a UTM-zone boundary) through the Earth Search path and diff NDVI against the
> existing EE export, pixel for pixel.

Cheap, and it decides everything: if NDVI agrees within tolerance the switch is safe and the
quota problem is over; if it does not, we have learned that the frozen model would be seeing
different inputs — which is worth knowing regardless of which path we take.

## Cheaper thing to do first, either way

**Inference reads only 2 of the 4 exported bands.** `inference/tiles.py:35-36` sets
`S2_RED_BAND = 1` (B4) and `S2_NIR_BAND = 4` (B8); green and blue exist for the pure-S2 RGB
model, not for the Planet+NDVI interannual run. Dropping them from the export should cut
EECU roughly in half — ~282 → ~560 cells/month on the Contributor tier — for free, today.

Not a fix (still 3.2 months of quota per year of data), but it doubles whatever quota we
have, it is independent of this decision, and it should be measured rather than assumed:
export one cell 2-band and 4-band and compare `batchEecuUsageSeconds`.

Caveat: a 2-band file breaks `S2_NIR_BAND = 4`. Keep the 2025 path working — the band
indices must become a property of the index, not a module constant.

## Recommendation

1. Measure the 2-band EECU saving and, if it holds, apply it to the running exports.
2. Run the 10-cell 2025 reproduction diff.
3. Keep the Partner Tier application moving regardless — it is free, it is weeks of latency,
   and if the diff fails it is the only remaining lever.
