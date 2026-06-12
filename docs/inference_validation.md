# Tiny-area inference validation (overlap, stitching, critical ops)

2026-06-12 · branch `inference-pipeline` · harness: `scripts/validate_inference_tiny.py`
(Tier-2: real 2025-Q3 quads, GPU 0, dev package `phase0c_seed42` with **uncalibrated**
threshold 0.5 / temperature 1.0).

## Setup

- **Corner AOI**: 10×10 km centered on zoom-15 quad 4-corner **(338, 1622)** =
  71.86°N, −120.59°W (Banks Island — 34 v2.1 positives in the surrounding 2×2 quads),
  so AOI tiles straddle quad boundaries in x, y, and diagonally. 56 tiles at stride 344.
- **Edge AOI**: top of the northernmost available quad in column 337 — beyond it there is no
  quad, guaranteeing a NoData region.
- Reproduce: `quad_index.csv` built from bucket columns 337–338;
  `python scripts/validate_inference_tiny.py --quad-index … --package … --out-root …
  --metadata metadata_v21.csv`. Artifacts in `/mnt/outputs/inference/validation/`.

## Results — 11 checks, 0 FAIL (after 2 bugs fixed, see below)

| # | Check | Result | Numbers |
|---|---|---|---|
| 1 | Coverage accounting | PASS | no gaps; interior histogram 1×: 929k px, 2×: 1.45M, 4×: 564k — the designed 1/2/4 pattern at 33% overlap |
| 2a | Stitching vs offset-grid reference | PASS | merged value at production-seam locations vs direct center predictions: mean abs diff = 0.0006, p99 = 0.0050 over 1.3M px |
| 2b | Seam gradients | PASS | p99 grad on seam columns 0.00138 vs 0.00081 off-seam (1.7×; was **6.9×** before the fusion fix) |
| 3 | Fusion vs brute force | PASS | max diff 8.7e-11 at 20 random pixels (vectorized accumulation == per-pixel NumPy) |
| 4 | Quad-straddle reads | PASS | 4-corner tile (4 quads) + 2 edge tiles pixel-identical to independent per-quad reads; no NoData stripe at quad seams |
| 5 | NoData propagation | PASS | beyond-coverage all −1.0; valid ∈ [0,1]; mask 255 ⇔ prob −1 |
| 6 | Determinism | PASS | two full runs byte-identical (same batching) |
| 7 | Resume equivalence | PASS | interrupt-at-20 + resume: value-equivalent, max Δprob = 0.0062 — **bf16 batch-shape jitter**, not a resume bug (see notes) |
| 8 | TTA sanity | PASS | minimal == (identity+hflip)/2 algebra holds; mean Δ vs none = 0.00025 |
| 9 | Geo-alignment | PASS | merged raster res 4.7773 m, bounds == AOI, EPSG:3857; canvas **2576×2920 px = 12.3×14.0 km**. `validation_overlay.png` (2×2): RGB + tile outlines (cyan) + quad-boundary cross (yellow); probability + stitch lines (green); threshold contours; **400px seam-zoom** — the hottest blob sits on a seam intersection with smooth probability across both stitch lines. Regenerate without GPU: `--overlay-only` |
| 10 | Detection plausibility | INFO | 9 blobs ≥ 0.1; top-5 peaks 0.107–0.145, each within **0.87–1.14 km** of a v2.1 positive centroid (uncalibrated v2.0 model — qualitative only) |

## Bugs found and fixed by this validation

1. **Quad filename parsing** (`inference/quad_index.py`): some deliveries use a flat layout
   embedding the mosaic name in the filename
   (`…/338/1474/global_quarterly_2025q3_mosaic_338-1474_quad_file_format.tif`) instead of an
   order-UUID subdirectory — the old suffix-strip parse crashed the index build. Fixed with an
   anchored regex; unit test `test_quad_name_regex_handles_both_delivery_layouts`.
2. **Seam discontinuity in fusion** (`scripts/merge_predictions.py`): the plain radial Gaussian
   keeps weight exp(−2) ≈ 0.135 at tile edges, so a tile's contribution switches on/off
   discontinuously across stitch lines → measured 6.9× elevated probability gradients on seams.
   Replaced with a separable **edge-zeroed** Gaussian (same σ=128); seam gradients now 1.7× of
   background. Spec note added to `inference.md §4.3`. Side effect: 1-px NoData ring at
   unchunked AOI boundaries (zero total weight) — irrelevant once PDG chunks overlap.

## Notes / known characteristics

- **bf16 batch-shape jitter**: identical tiles inferred in different batch shapes differ by up
  to ~0.006 probability (different CUDA kernels). Consequences: (a) resume/restart is
  value-equivalent but not byte-identical; (b) this is the reproducibility floor for bf16
  deployment — calibrated thresholds should not be trusted beyond ~0.01 granularity, and any
  exact-reproduction need (audits) requires fp32 or fixed batching. Same order as TTA-induced
  changes (0.00025 mean).
- The coverage design at stride 344 leaves tile-center pixels single-covered (33% overlap is
  about capturing whole RTS in *some* tile, §4.2 — not about multi-covering every pixel).
- Model response on 2025 imagery is low everywhere (max ~0.19 in this AOI) — expected for the
  uncalibrated v2.0 dev checkpoint; the blob-to-known-RTS proximity (~1 km) is encouraging but
  not an evaluation.
