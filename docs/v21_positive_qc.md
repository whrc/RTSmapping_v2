# v2.1 training-data QC (run mid-drop, 2026-06-12 ~16:30 UTC)

Harness: `scripts/validate_v21_positives.py` → `/mnt/outputs/inference/v21_qc/`
(`qc_per_tile.csv`, `qc_gallery.png`, `qc_batch_map.png`, `qc_histograms.png`).
Snapshot state: 11,764 metadata rows (1,757 positive / 10,007 negative), 15,250 RGB files,
1,757 labels — **the negative drop was landing during the run**; counts are a snapshot.

## 🚨 Critical: all 261 batch2 labels are 100% RTS

261 positive tiles have `rts_frac == 1.0` — the *entire* 512×512 label is class 1 (a whole
2.4 km tile of RTS is not physically plausible). Cross-checked against the morning metadata
(which still carried the `Version` column): the 261 tiles are **exactly the batch2 set,
261/261** — batch1 and batch3 have zero such tiles. This is a label-rasterization bug in the
batch2 production path (e.g. polygon burn/fill inverted or nodata written as 1).

**Action for the data team: regenerate batch2 labels.** Until fixed, batch2 must be excluded
from any v2.1 snapshot/training (a model trained on these would learn "everything is RTS" on
261/1757 = 15% of positives).

Healthy distribution for the rest: median RTS fraction 9.4%; 75 tiles < 1%; only 10 tiles in
(0.5, 0.99) — plausible large slumps.

## Other findings

| Check | Result | Detail |
|---|---|---|
| Positives have RGB + label | PASS | 1,757/1,757 both directions |
| Negatives have RGB, no label | PASS | matches the v2.0 convention (synthetic zero labels at load) |
| Raster geometry | PASS | all 512², EPSG:3857, 3-band uint8 |
| Label values ⊆ {0,1,255} | PASS | |
| Zero-RTS positives | PASS | 0 (v2.0 had 1) |
| Degraded bands (>50% zero) | **FAIL, 9 tiles** | `ch42ujzbzfpv ch42ysrbpvpx ckc8ebbj2p24 ckrk11pbzyxz vufc2281bp81 vuuzfwpzzvzb vuzpr8pgpgpv vuzqcpzuxyxc ykum34xvxyxg` — much better than v2.0's 209, but report to data team |
| Centroid in tile bounds | PASS | |
| Objects ↔ metadata | transient | 3,486 RGB not yet in metadata.csv (drop in progress; recheck when done) |

## Watch items (for the final-drop recheck)

1. **Negative latitudes span 44.6–76.0°N** — outside the 60–74°N inference domain on both
   ends. If intentional (hard negatives / domain widening), fine; if not, filter. Ask data team.
2. **Metadata schema drifted again mid-drop**: the `Version` column present this morning is
   gone from the current `metadata.csv`. Batch provenance is now unrecoverable from the bucket
   alone — the morning copy is preserved at
   `/mnt/outputs/inference/validation/metadata_v21.csv` (and is how batch2 was identified).
   Request: keep `Version` in the final metadata.
3. Negative count target unknown (~10k registered so far vs ~13.7k in v2.0) — confirm
   completeness criteria before freezing the snapshot.

## Re-run

```
python scripts/validate_v21_positives.py --out-dir /outputs/inference/v21_qc
```
Re-run after the data team declares the drop complete; all transient findings should clear and
batch2 should show a sane RTS-fraction distribution.
