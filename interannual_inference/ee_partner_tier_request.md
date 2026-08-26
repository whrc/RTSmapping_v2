# Earth Engine Partner Tier — application draft

**Status: DRAFT, not submitted.** Submit via the [Earth Engine help page](https://www.earthengine.app/)
(Partner Tier application). Manual review, several weeks.

**Project:** `abruptthawmapping` (project number 801926669176), currently Contributor Tier.

Sections marked **[CONFIRM]** need your input — they are institutional or impact claims I
should not invent. Everything else is measured and is safe to send as written.

---

## Organization

Woodwell Climate Research Center — an independent nonprofit climate research institute.
**[CONFIRM]** the exact legal name / 501(c)(3) status wording, the PI to name, and whether
this should be submitted under the Permafrost Discovery Gateway collaboration.

Meets the "nonprofit, university research group, or government organization" criterion.

## What we are doing

Pan-Arctic mapping of **retrogressive thaw slumps** (RTS) — the abrupt permafrost-thaw
landslides that expose buried organic carbon to decomposition — across the circumpolar
permafrost domain, 60–74°N.

We train a deep-learning semantic-segmentation model (EfficientNet-B5 U-Net++ ensemble) on
PlanetScope basemap imagery with a Sentinel-2-derived NDVI channel, and apply it at 3 m
resolution over the full domain. The single-epoch 2025 map is delivered. We are now running
the **frozen** deployed model over **2019–2024** to convert that snapshot into a
**seven-year interannual time series** — which is what makes the dataset usable for
*rates* of abrupt thaw rather than a one-time inventory.

**[CONFIRM]** downstream use: publications, the Permafrost Discovery Gateway, data release
plans, any policy or assessment process (e.g. Arctic Report Card, IPCC-facing syntheses)
that consumes this. Partner Tier review explicitly prioritises demonstrable on-the-ground
and policy impact, so this section carries the most weight.

## Why it needs Earth Engine

Earth Engine supplies the NDVI channel the model requires: a cloud-masked July–September
median Sentinel-2 surface-reflectance composite (`COPERNICUS/S2_SR_HARMONIZED`, QA60 mask,
`CLOUDY_PIXEL_PERCENTAGE < 20`), exported as B4/B3/B2/B8 GeoTIFFs in EPSG:3857 over
**1,799 land grid cells** (1°×3°) covering the circumpolar southern domain, **once per year**.

The composite must be **year-matched**. Substituting one year's NDVI into another year's
inference would bake that year's vegetation state into the map and destroy the very
interannual signal we are measuring, so there is no way to compute this once and reuse it.

## Compute requirement — measured, not estimated

From `batchEecuUsageSeconds` over 195 completed export tasks on `abruptthawmapping`:

| | EECU-seconds | EECU-hours |
|---|---|---|
| per grid cell (mean of 195) | 12,755 | 3.5 |
| per year (1,799 cells) | 22,946,245 | 6,374 |
| **remaining requirement (10,417 cells, 6 years)** | **132,868,835** | **36,908** |

Against the Contributor Tier's 1,000 EECU-hours/month, one year of imagery takes **6.4
months** and the six-year series takes **38 months**. We consumed 691 EECU-hours — 69% of
the monthly allowance — in the first 48 hours and have been in restricted mode since.

Under Partner Tier's 100,000 EECU-hours/month the entire remaining requirement is **37% of
a single month's allowance**, and the series completes in weeks rather than years.

This is a **finite, bounded** request: 2019–2024 is the whole scope. Steady-state use
afterwards is roughly one year of imagery annually (~6,400 EECU-hours) to extend the series.

## What we have already done to reduce the ask

- Exports are **resumable** and skip cells already delivered, so no cell is ever computed twice.
- The domain is masked to land ∩ permafrost ∩ Arctic-boreal before export; ocean and
  out-of-domain pixels are never computed.
- The model is **frozen** across all years — one export per year, no re-runs for retraining.
- We verified that spreading work across additional Cloud projects does not help and
  declined to spend other research groups' allowances on it.

## Contact

**[CONFIRM]** name, email, institutional affiliation of the submitter.
