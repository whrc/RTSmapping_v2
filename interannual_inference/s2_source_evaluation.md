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

**2. The BOA offset handling is backwards — and it silently destroys the data.**

This is the script's headline feature, and I had it wrong in my first pass: the metadata
*is* served, but applying it is incorrect for these COGs.

`sentinel-cogs` DN are **already** in the pre-baseline-04.00 convention — Element 84
removes the +1000 before staging. The STAC `raster:bands` offset of −0.1 describes the
*original ESA product*, not the file you just opened. Proof, from a baseline-05.11 tile:

```
raw DN red: mean 730.8   p5 44   p95 2412
  DN/10000        = +0.0731     <- matches Earth Engine (0.0623 on the same cell)
  (DN-1000)/10000 = -0.0269     <- negative reflectance, physically impossible
```

A 5th percentile of **44 DN** settles it: if +1000 were present, no pixel could sit below
1000. Fitting my reconstruction against the EE product gave `ES = 1.01·EE − 0.0999` with
the offset applied, and an intercept of ~0.0001 without it.

Two consequences, both silent:

- Reflectance goes negative, and NDVI's denominator `(nir + red)` then crosses zero, so
  NDVI explodes rather than degrading gracefully. My first run produced NDVI mean +0.685
  against EE's +0.189, with correlation 0.15.
- Worse, the script writes `np.clip(a + shift, 0, 65535)` into a band whose **nodata is
  0**. Every pixel darker than 0.1 reflectance — which over Arctic land is most of the
  red band — is clipped to 0 and becomes indistinguishable from nodata.

Any port must **verify the convention against a known-good product**, not trust the
metadata. Ours is `S2_RGB/2025_south/`.

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

## Validation: does the Earth Search path reproduce the EE product?

Prototype compositor (STAC search → dedupe reprocessed duplicates → SCL mask → per-item
offset → median → `WarpedVRT` straight onto the EE product's own EPSG:3857 grid, so the
EE side is never resampled). 1024×1024 windows, cells present in both exports.

Geometry and radiometry are **exact**: reflectance correlation 0.9987 (red) / 0.9977
(nir), best alignment at dy=0, dx=0 over a ±3 px search. There is no reprojection or
co-registration problem.

NDVI agreement over 14 windows:

| | |
|---|---|
| 11 of 14 windows | MAE 0.0005 – 0.025, corr ≥ 0.975 |
| `W1530_N0580` 2025 | MAE 0.072, corr **0.54** |
| `W1590_N0710` 2022 | MAE 0.108, mean −0.080, corr 0.93 |

**Not a clean pass.** The 2022 outlier is fully explained — it is the QA60 gap
([`qa60_gap.md`](qa60_gap.md)), and disabling my mask reproduces EE to MAE 0.0016, which
is what proves EE's 2022 product is unmasked. The 2025 outlier is **not yet explained**
and needs to be before anyone relies on this path.

Also note the sample is opportunistic, not designed: a centred 1024² window often lands
on water or outside coverage, and 10 of 24 attempts returned too little data to compare.
A real gate wants windows chosen for land cover and cloudiness, not for convenience.

## Where this leaves the decision

The source is sound and the arithmetic still favours it heavily — ~8 days of pulling
versus 38 months of quota. What the prototype changes is the estimate of *effort*: the
reference script is not a starting point so much as a cautionary one, since both of its
non-trivial behaviours (offset, dedupe) are wrong for our data in ways that fail silently.

Remaining before a switch could be recommended:

1. Explain the `W1530_N0580` 2025 outlier.
2. Choose a cloud mask on evidence (QA60 vs SCL vs s2cloudless on Arctic cells in a year
   where all three exist) — required for the EE path too, per `qa60_gap.md`.
3. Re-run the gate on a designed sample once 1 and 2 are settled.
