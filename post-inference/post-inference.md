# RTS Segmentation Model v2: Post-Inference Processing

## 1. Objective

Turn the per-region probability rasters produced by the inference stage into the
final RTS product — a clean, vectorized, geodesically-measured polygon layer — and
quality-control it. Post-inference is everything **after** the inference stage has
written per-region probability COGs; it owns scale fusion, thresholding,
vectorization, geometric measurement, QC, and (offline) evaluation.

## 2. Inputs (from the inference stage)

| Input | Source | Spec |
|-------|--------|------|
| Probability raster(s) | `scripts/inference.py` + `scripts/merge_predictions.py` | `inference.md §9.1` — Float32 COG, NoData `-1.0`, EPSG:3857, 4.77 m |
| Run manifest | `inference_log.json` | `inference.md §9.4` — carries `scales_used`, `fusion_method`, `threshold`, `temperature`, `overlap_aggregation` |

The output **format** SSoT for the binary mask and vector layer lives in
`inference.md §9.2`/`§9.3`; this document is the SSoT for the **algorithms** that
produce them.

## 3. Stage ownership (resolves the overlap-vs-merge ambiguity)

There are three distinct "merge" operations; keep them separate:

| Operation | Owner | Method | Status |
|-----------|-------|--------|--------|
| **Tile-overlap stitching** (adjacent 33%-overlapping tiles within one scale) | **inference** (`inference.md §4.3`, `merge_predictions.py`) | Gaussian distance-from-center **weighted mean** (σ = `fusion_sigma_px` = 128) | implemented |
| **Scale fusion** (combine per-scale rasters) | **post-inference** (§4) | pixel-wise **max** | spec only — single-scale today (`scales=[1.0]`, gated by `inference.md §6.4`) |
| **Region mosaicking** (assemble chunked regions into a product) | **post-inference** (§7) | adjacency only; regions are chunked to be non-overlapping | spec only |

> **Decision (resolved):** tile-overlap stitching is **Gaussian weighted mean**, not
> pixel-wise max. The earlier "max for overlapping tiles" note is superseded — the
> tiny-area validation (`docs/inference_validation.md`, 2026-06-12) showed the plain
> radial Gaussian left a discontinuous on/off contribution at tile edges (seam
> probability gradients 6.9× background); the edge-zeroed Gaussian fixed it (1.7×).
> Pixel-wise **max** is reserved for **scale fusion** (a detection-union across FOVs),
> where there is no seam-continuity requirement.

## 4. Scale Fusion (multi-scale only)

When `scales_used` has more than one entry, combine the per-scale probability
rasters pixel-wise:

```
P_final(x, y) = max(P_1.0(x, y), P_0.5(x, y))   # over valid (non-NoData) scales
```

**Rationale:** if any scale confidently detects RTS, keep it (detection-union);
per-scale thresholds control precision. NoData (`-1.0`) contributes nothing — a
pixel is NoData in the fused output only if NoData in every scale.

> **Two fusion options (both retained).** The inference pipeline can fuse **in-pipeline** —
> `inference/runner.fuse_scale_probs` writes a **single §7.3 arithmetic-mean** fused probability
> COG per tile (the path implemented + validated against the family-M POC). This §4 **downstream
> MAX fusion** over separately-written per-scale rasters is the alternative (detection-union): use
> it when per-scale COGs are kept as distinct products and you want max-not-mean. Pick one per run;
> they are not chained.
>
> **Not active yet either way:** the v1.0/v2.0 model does **not** transfer zero-shot to 2× GSD
> (`docs/inference_validation.md` scale-0.5 experiment: 9 → 0 blobs); the family-M training POC passed
> gates 1+2 but **failed gate 3** (mean-fusion recall). Multi-scale at inference is gated by
> `inference.md §6.4`; until that gate passes, `scales=[1.0]` and both fusion paths are no-ops.

## 4b. Tiered products (decision 2026-07-14)

The single-threshold product family is extended with a **tiered candidate
inventory** cut at threshold 0.30 (`vectorize_region.py --threshold`, windowed
polygonize of the probability COG shards — no mask re-assemble), classed
`high ≥ 0.65 / medium ≥ 0.45 / low ≥ 0.30` by `max_prob`. Rationale: object
precision *peaks* at the deployed 0.65 (residual FPs are confident
look-alikes), while 11.4% of val GT slumps carry sub-0.65 signal — so the
conservative view comes from attribute filtering, not a higher cut, and the
permissive view recovers real misses. Threshold-free density grids
(Σ calibrated P × geodesic px area, `aggregate_probability.py`) summarize the
canvas without any threshold. Catalog + caveats: `south_products.md` (SSoT).

## 5. Thresholding → Binary Mask

Apply the **calibrated** threshold (`deployment_config.yaml.threshold`, from
training.md §12 calibration on val) to the (fused) probability raster:

```
mask = (P_final >= threshold).astype(uint8)   # 0/1
mask[P_final < 0] = 255                        # NoData propagates (inference.md §9.2)
```

Output per `inference.md §9.2` (UInt8 COG, NoData 255).

## 6. Vectorization

Polygonize `mask == 1` and attach the `inference.md §9.3` attribute schema.

- **Polygonize:** `rasterio.features.shapes` (or `gdal_polygonize`) on the binary
  mask, EPSG:3857. Implemented in `scripts/vectorize_predictions.py` (inference
  branch / PR #19).
- **Simplify:** raw polygonization is stair-stepped (one vertex per pixel edge).
  Apply Douglas–Peucker (`shapely.simplify(tolerance, preserve_topology=True)`) to
  cut vertex count and de-pixelate the boundary.
  - **Decision to make:** simplify `tolerance`. Start at **~1 pixel (≈4.77 m)** and
    tune against QC (§8) — too high erodes small/narrow slumps, too low keeps the
    stair-steps. Record the chosen value in the run manifest.
- **Filter:** the SHIPPED rule is the **geodesic MMU** — `vectorize_region.py
  --min-area-m2` in **m²** (default `0` = no MMU, leaving only a 2-px technical
  floor), latitude-constant. The alternative `--legacy-min-blob-px` uses the
  package's `vectorize_min_blob_px` (2000 px), which produced the **superseded**
  `south_rts.gpkg` only. Neither is `metrics.min_blob_size_px` (= 10 px), which
  is an **eval-stage** filter and never touches a product. Full disambiguation:
  `post-inference/south_products.md` §"Size parameters — which number is which".

## 7. Geometric Measurement (geodesic — mandatory)

EPSG:3857 inflates area ~13× at 74°N. **Never** compute area/perimeter from 3857
coordinates. Reproject each polygon to WGS84 and measure on the ellipsoid:

```python
from pyproj import Geod
geom_wgs = gpd.GeoSeries([geom], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
area, perim = Geod(ellps="WGS84").geometry_area_perimeter(geom_wgs)
area_m2, perimeter_m = abs(area), perim
```

(`area_m2`, `perimeter_m`, `centroid_lat/lon` per `inference.md §9.3`.) This matches
`vectorize_predictions.py` on PR #19.

## 8. Quality Control

Deployment imagery (2025) has **no ground truth**, so QC here is distributional and
qualitative — quantitative scoring is §9.

| Check | What it catches |
|-------|-----------------|
| NoData propagation | `mask==255 ⇔ P<0`; no spurious polygons over NoData |
| Seam artifacts | polygon density / probability gradient spikes on tile-stitch lines (regression of the §3 fix) |
| Area distribution | implausible giant or sub-`min_blob_size` polygons; compare the area histogram against the v2.1 training-positive distribution |
| Edge effects | the 1-px zero-weight ring at unchunked AOI boundaries (`docs/inference_validation.md`) — must vanish once regions overlap-chunk |
| Geodesic sanity | spot-check a few `area_m2` vs a manual equal-area reprojection |
| Visual overlay | the `validate_inference_tiny.py` overlay (RGB + tile outlines + probability + threshold contours + seam zoom) on a sample region |

## 9. Evaluation (offline, on the held-out test split)

The deployment map cannot be scored directly. Quantitative evaluation uses the
**frozen v1.0 test split** (`scripts/evaluate_test.py`, training.md §12) — the same
pipeline scored on labelled tiles:

- **Object-level:** match predicted blobs to label blobs at IoU ≥
  `object_iou_threshold` (= 0.3), with **eval-stage** `metrics.min_blob_size_px`
  (= 10 px) filtering *predictions* — not the product's MMU; report
  precision/recall/F1.
- **Pixel/region-level:** PR-AUC at the realistic neg:pos ratios `[5, 10, 20]`
  (`metrics.pr_auc_ratios`) — the same `val_realistic_pr_auc_geomean` vocabulary as
  the locked Phase-0 gate (`docs/phase0_baseline.md`), so test scores are
  interpretable against μ₀ = 0.7912.

> Evaluation runs the **calibrated** package (threshold/temperature from §12) so the
> reported precision matches what the deployment threshold produces.

## 10. Outputs & Chunking

Produce the product **per region** (not a single pan-arctic raster):

- `gs://<bucket>/inference/<basemap>/<region>/probability.tif` (Float32 COG)
- `…/<region>/mask.tif` (UInt8 COG)
- `…/<region>/rts.gpkg` (polygons, §6–§7 schema)
- `…/<region>/inference_log.json` (provenance, `inference.md §9.4`)

Per-region chunking eases distribution, enables parallelism, and allows
region-specific QC. Regions are chunked with overlap so the §3 edge ring is interior
to a neighbour and discarded at mosaic time.
