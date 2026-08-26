# The QA60 gap: our 2022 and 2023 composites have no cloud mask

**Measured 2026-08-26. This is independent of where we get Sentinel-2 from, and it
affects the interannual series directly.**

## The recipe assumes QA60 exists

`data/extra_channels.s2_sr_composite` masks clouds with:

```python
qa = img.select("QA60")
m = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
return img.updateMask(m)
```

If `QA60` is all zeros, both bit tests pass everywhere, `m` is all-true, and
`updateMask` masks **nothing**. The function degrades silently to a no-op — no error,
no warning, just an unmasked median.

## QA60 is empty for 2022–2023

ESA stopped populating QA60 at processing baseline 04.00 and reinstated it later.
Maximum QA60 value over a Jul–Sep, cloud < 20 collection, four widely separated
Arctic sites:

| site | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Alaska N. Slope | 2048 | 2048 | 2048 | **EMPTY** | **EMPTY** | 1024 | 2048 |
| Canada, Banks Is. | 2048 | 1024 | 2048 | **EMPTY** | **EMPTY** | 2048 | 2048 |
| W. Siberia, Yamal | 2048 | 2048 | 2048 | **EMPTY** | **EMPTY** | 2048 | 2048 |
| E. Siberia, Lena | 2048 | 2048 | 2048 | **EMPTY** | 1024 | 2048 | 2048 |

## Confirmed against the delivered product, not just the metadata

Rebuilding cell `W1590_N0710` (71°N) for 2022 from Earth Search COGs on the EE
product's exact grid:

| reconstruction | NDVI mean | vs EE mean | MAE | corr |
|---|---|---|---|---|
| with a per-pixel cloud mask | +0.1023 | **−0.0804** | 0.1082 | 0.9347 |
| **with no cloud mask at all** | **+0.1825** | **−0.0002** | **0.0016** | **0.9994** |

EE's 2022 composite *is* the unmasked composite, to 0.0016 NDVI. That is as direct a
confirmation as we can get.

## Why it matters here specifically

This is not a uniform bias that cancels in a difference. It is a **mask discontinuity
placed exactly in the middle of our study period**: 2019–2021 and 2024–2025 are
QA60-masked, 2022–2023 are not. Any NDVI trend we compute across that boundary carries a
step that is an artefact of ESA's processing history, not of the landscape — in the one
input channel the interannual comparison depends on. At the cell measured, the step is
~0.08 mean NDVI with a p95 of 0.55.

The 165 cells of 2022 already exported are affected.

## The awkward part

The delivered 2025 map was built with QA60, and QA60 *worked* in 2025. So:

- Leaving the recipe alone keeps 2025 reproducible and leaves 2022–2023 unmasked.
- Changing the mask (SCL, or `MSK_CLDPRB`/s2cloudless — both available for every year)
  fixes 2022–2023 but makes every year differ from the basis of the delivered map.
- Making the series internally consistent really means **re-exporting all years,
  including 2025, under one mask** — which is another 6 years of EECU, and so runs
  straight back into [`ee_quota.md`](ee_quota.md).

That last point is a substantive argument for the Earth Search route
([`s2_source_evaluation.md`](s2_source_evaluation.md)): re-exporting everything under a
consistent mask is unaffordable on the Contributor tier and roughly a week of pulling
from S3.

## Open, and needed before choosing a replacement mask

SCL is **not** validated here as the right mask. At 71°N it moved NDVI by −0.08, and SCL
is known to be unreliable at high latitude and low sun angle, where it confuses snow,
bright bare ground and cloud. Before adopting any mask we should compare QA60, SCL and
s2cloudless against each other on Arctic cells in a year where **all three** are
available (2019–2021 or 2024–2025), and pick on evidence.
