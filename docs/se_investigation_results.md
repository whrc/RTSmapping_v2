# SE Channel Investigation — Results

Implementation: [scripts/channel_correlation.py](../scripts/channel_correlation.py) (Phase 1), [scripts/se_variants.py](../scripts/se_variants.py) (Phase 2). Config: [configs/se_investigation.yaml](../configs/se_investigation.yaml).

Outputs:
- Phase 1: [plots/extra_channel_vis/correlation/](../plots/extra_channel_vis/correlation/) — 150 held-out tiles
- Phase 2: [plots/extra_channel_vis/se_v2/approach1_multiprototype/](../plots/extra_channel_vis/se_v2/approach1_multiprototype/), [plots/extra_channel_vis/se_v2/approach2_contrastive/](../plots/extra_channel_vis/se_v2/approach2_contrastive/) — 7 test tiles (OID 93, 113, 136, 144, 169, 187, 262)

---

## Phase 1 — Correlation (150 held-out tiles, 2M pixels/regime)

**Original single-prototype SE-cosine vs every Sentinel auxiliary — C1 result by regime:**

| Channel | r (all) | r (positive) | r (boundary) | C1 pass |
|---------|---------|--------------|--------------|---------|
| NIR  | -0.44 | -0.32 | (see heatmap) | pass |
| SWIR | +0.31 | -0.23 | — | pass |
| NBR  | -0.11 | -0.05 | — | pass |
| NDVI | -0.08 | **-0.53** | — | fails in positive regime |
| NDWI | +0.05 | **+0.58** | — | fails in positive regime |
| NDMI | +0.24 | +0.14 | — | pass |
| TCB  | -0.01 | -0.15 | — | pass |
| TCW  | -0.35 | +0.21 | — | pass |

On the diverse all-pixel regime, SE-cosine is at most |r|=0.44 with NIR — **C1 passes globally**. Inside positive polygons the correlation with NDVI (−0.53) and NDWI (+0.58) slightly breaches 0.5, consistent with the fact that SE, NDVI, and NDWI all respond to the exposed-soil / dead-vegetation signature that defines RTS.

**Dendrogram (Spearman, average linkage):** SE-cosine joins NBR + NDMI at distance ≈ 0.60, which sits above the |r|=0.5 redundancy threshold. The two non-SE clusters (RGB; NIR–PC1–NDVI–NDWI; SWIR–TCB–TCW–PC3) each join below 0.5, confirming SE-cosine carries information that is not a linear recombination of existing channels.

**Note on Pearson heatmap:** a zero-variance column (likely a label-255 interaction at some tiles) caused the first run to drop the Pearson matrix. Script is patched ([scripts/channel_correlation.py:405](../scripts/channel_correlation.py#L405)) to mask zero-variance columns; re-run to regenerate `heatmap_all_pearson.png`. Spearman is the primary analysis per spec, so this is low priority.

---

## Phase 2 — SE-cosine variants on 7 test tiles

### Approach 1 — Multi-prototype (k selected by silhouette)

| k | silhouette |
|---|-----------|
| **3** | **0.287** (selected) |
| 5 | 0.201 |
| 8 | 0.159 |
| 12 | 0.164 |

Cluster masses at k=3: 12.7% / 67.8% / 19.6% (one dominant cluster, reflecting the tight concentration of SE embeddings on the unit hypersphere).

**C2 (sign consistency, [approach1_multiprototype/sign_consistency.md](../plots/extra_channel_vis/se_v2/approach1_multiprototype/sign_consistency.md)):** 6/7 correct, 1 inversion (OID 93: polygon 0.9464 vs background 0.9465 — effectively a tie). Target met (≤1 inversion).

**C1 (on the 7 test tiles, not held-out):** NIR −0.26, NDVI −0.29, **NBR −0.79 (fails)**. The max-cosine score correlates strongly with NBR inside positive-heavy tiles because both respond to surface disturbance. Held-out evaluation would be needed to confirm whether this generalizes.

**Dynamic range issue persists:** all cosines still in [0.86, 0.98]. The SE embeddings are so tightly clustered on the unit sphere that max-over-prototypes does not widen the range meaningfully.

**Nearest-pixel prototype samples:** rendering failed because `ee.Image.sampleRegions` does not preserve input ordering and drops points that land outside image extent, breaking the (coord → SE-vector) alignment used to look up source tiles. Silhouette bars and cluster sizes are rendered; nearest-pixel crops require a reorder-preserving sampler.

### Approach 2 — Contrastive (positive prototypes − negative prototypes, k=3 each)

**C2 ([approach2_contrastive/sign_consistency.md](../plots/extra_channel_vis/se_v2/approach2_contrastive/sign_consistency.md)):** **7/7 correct, 0 inversions** — including OID 93 (polygon +0.0014, background −0.0244) and OID 169 (polygon +0.0249, background −0.0033).

**C1 ([approach2_contrastive/correlation_vs_s2.md](../plots/extra_channel_vis/se_v2/approach2_contrastive/correlation_vs_s2.md)):** NIR 0.05, NDVI 0.07, NBR 0.34 — **all pass**.

**Dynamic range:** score centered near 0 with visible spatial structure (see [oid262_se_v2.png](../plots/extra_channel_vis/se_v2/approach2_contrastive/oid262_se_v2.png) for a clean example — strong red inside the polygon, blue elsewhere).

---

## Summary

- **Original SE-cosine** passes C1 on diverse data and joins the NBR–NDMI cluster only weakly (distance 0.60). It is not redundant with existing Sentinel channels globally.
- **Approach 1 (multi-prototype)** fixes the sign-inversion failure on 6/7 tiles but does not widen the dynamic range, and its max-cosine output correlates heavily with NBR on positive tiles.
- **Approach 2 (contrastive)** passes both C1 and C2, visibly separates RTS from background, and is the recommended SE variant to trial as an auxiliary training channel.

## Next steps (pending group decision)

1. Fix nearest-pixel rendering in Approach 1 by tracking per-coord SE vectors through a sampler that preserves ordering (e.g., per-point `ee.Image.sample` in small batches, or `ee.Image.reduceRegions` with an explicit point FeatureCollection).
2. Re-run Phase 1 with the Pearson fix for the linear-correlation reference heatmap.
3. Evaluate Approach 2 correlation on the 150-tile held-out set rather than the 7 test tiles, to confirm C1 generalizes.
