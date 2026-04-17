# SE Channel Investigation

Implementation spec for evaluating SE (spectral embedding) as a weak auxiliary channel for RTS segmentation.

**Phase 1 — Correlation test.** Quantify which channels are redundant with SE-cosine and with each other.
**Phase 2 — SE-cosine improvements.** Two principled approaches to make SE-cosine a complementary signal.

---

## 0. Success Criteria

| Code | Criterion |
|------|-----------|
| C1 | Spearman \|r\| ≤ 0.5 between SE-derived channel and every Sentinel auxiliary (NIR, SWIR, NBR, NDVI, NDWI, NDMI, TCB, TCW) in both all-pixel and positive-pixel regimes |
| C2 | Polygon-mean cosine above tile background mode on ≥ 6/7 test tiles (OID 93, 113, 136, 144, 169, 187, 262) — no sign inversions |

---

## Phase 1 — Correlation Test

### 1.1 Channel Set (15 channels)

Per-tile PCA is excluded: the basis is fit per tile, so PC semantics drift across tiles and the sign is arbitrary. Per-tile PCA is a useful diagnostic visualization but not a stable feature. Global PCA has a fixed basis and is usable as a feature.

| Group | Channels | Count |
|-------|----------|-------|
| PlanetScope RGB | R, G, B | 3 |
| Sentinel-2 spectral | NIR, SWIR | 2 |
| Sentinel-2 indices | NBR, NDVI, NDWI, NDMI | 4 |
| Sentinel-2 tasseled cap | TCB, TCW | 2 |
| SE reference-anchored | SE-cosine | 1 |
| SE global PCA | PC1, PC2, PC3 | 3 |

### 1.2 Data

| Parameter | Value |
|-----------|-------|
| Held-out set | Carve 100–200 tiles from the 1818 SE prototype tiles, random selection, stratified by positive/negative |
| Prototype handling | Use existing prototype (built from all 1818). Leakage from held-out is minor (~10% of 45k reference pixels) and acceptable for Phase 1. |
| Co-registration | All channels resampled to 3m PlanetScope grid, EPSG:3857 |
| NaN handling | Drop pixels with NaN in any channel; no interpolation |

### 1.3 Method

- Spearman rank correlation, pairwise across all channels
- Three regimes: all pixels, positive pixels (inside polygons), near-boundary pixels (within N pixels of polygon edge — suggest N=10)
- Secondary: Pearson correlation for reference
- Rankings only — no p-values (spatial autocorrelation makes significance testing unreliable)

### 1.4 Outputs

| Artifact | Path | How to read |
|----------|------|-------------|
| Heatmap — all pixels | `plots/extra_channel_vis/correlation/heatmap_all.png` | 15×15 grid. Cells near ±1 = redundant; near 0 = independent. Inspect the SE-cosine row: which existing channels does it correlate with? |
| Heatmap — positive pixels | `.../heatmap_positive.png` | Same, inside polygons only. A channel can be redundant globally but independent within-class — this regime matters most for segmentation. |
| Heatmap — near-boundary | `.../heatmap_boundary.png` | Same, near polygon edges. This is where the model needs help most. |
| Dendrogram | `.../cluster_dendrogram.png` | Channels joined on low branches are redundant. Find SE-cosine: which cluster does it join? If it joins NIR/NBR/TCB, it's in the brightness cluster (redundant). If it sits alone, it carries distinct information. |
| Distance-from-NDVI | `.../distance_from_ndvi.png` | Bar chart, channels ranked by \|r\| distance from NDVI. Taller bar = more independent of NDVI. |
| Distance-from-NIR | `.../distance_from_nir.png` | Same, against NIR. |
| Pearson heatmap | `.../heatmap_all_pearson.png` | Linear-correlation reference. Use Spearman as primary. |
| Interpretation table | `.../interpretations.md` | For every pair with \|r\| > 0.7, write the physical reason (both measure brightness, both measure vegetation, etc.). Pairs without a physical rationale indicate preprocessing artifacts. |

---

## Phase 2 — Make SE-Cosine Work

### 2.1 Failure Modes Addressed

| Failure | Evidence from original plots | Addressed by |
|---------|------------------------------|--------------|
| Narrow dynamic range | Histograms span [0.85, 0.98] | Per-tile standardization (shared foundation) |
| Prototype too broad | Weak polygon-vs-background separation even when sign is correct | Approach 1 (multi-prototype) |
| Sign inversion | OID 169 polygon mean below background mode | Approach 1 and Approach 2 |

### 2.2 Shared Foundation: Per-Tile Standardization

Also evaluate as a standalone ablation: single-prototype cosine on per-tile-standardized features. This isolates whether feature-space alone explains the current failure. Output to `plots/extra_channel_vis/se_v2/standardized_single/`.

### 2.3 Approach 1 — Multi-Prototype Cosine

| Element | Choice |
|---------|--------|
| Reference pool | 45k SE pixels from 1618 tiles (1818 minus held-out), per-tile standardized before clustering |
| Clustering | k-means |
| k sweep | {3, 5, 8, 12} |
| k selection | Silhouette score + visual inspection of per-cluster mean RGB |
| Output channel | Max-cosine across prototypes: `SE(x) = max_i cos(x, c_i)` |

Inspection step: render each prototype's mean RGB. Check which RTS morphologies each prototype captures. Specifically verify whether any prototype fires inside OID 169's polygon — if not, the reference pool does not contain that slump's morphological subtype.

### 2.4 Approach 2 — Contrastive Cosine

| Element | Choice |
|---------|--------|
| Positive prototype(s) | From Approach 1 (single or multi) |
| Negative pool | All pixels where training label = 0 |
| Negative prototype | k-means over negative pool, same k as positive (or single prototype if positive is single) |
| Output channel | `SE(x) = max_i cos(x, p_pos_i) − max_j cos(x, p_neg_j)` |
| Range | [−2, 2], centered near 0 |

### 2.5 Outputs per Approach

Per approach in `plots/extra_channel_vis/se_v2/{approach1_multiprototype, approach2_contrastive}/`:

| Artifact | How to read |
|----------|-------------|
| Per-tile figures on the 7 test tiles | Same layout as original SE feasibility plots. Look for polygon mean clearly above background mode, consistent direction across all 7 tiles. |
| Sign-consistency table | Count of inversions across the 7 tiles. C2 target: 0–1. |
| Correlation against NIR, NDVI, NBR | Spearman scalars on the held-out set. C1 target: each \|r\| ≤ 0.5. |
| Per-prototype mean RGB (Approach 1 only) | Visual: are prototypes interpretable as distinct RTS types (e.g. bright floor, shadowed headwall, revegetating)? |
| Prototype membership on OID 169 (Approach 1 only) | Which prototype fires in OID 169's polygon? If none, note as reference-pool coverage gap. |

---

## 3. Artifacts

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/channel_correlation.py` | Phase 1: correlation matrices and plots |
| `scripts/se_variants.py` | Phase 2: standardized-single, Approach 1, Approach 2 (selectable via config) |

Both driven by YAML config in `configs/`. No hardcoded paths.

### Output directories

- `plots/extra_channel_vis/correlation/` — Phase 1
- `plots/extra_channel_vis/se_v2/standardized_single/` — Phase 2 ablation
- `plots/extra_channel_vis/se_v2/approach1_multiprototype/` — Phase 2 Approach 1
- `plots/extra_channel_vis/se_v2/approach2_contrastive/` — Phase 2 Approach 2
- `docs/se_investigation_results.md` — findings and group decisions