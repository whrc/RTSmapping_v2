# SE Variant Comparison — Summary

## C1 — |Spearman r| vs Sentinel-2 (pooled 7-tile pixel sample)

| Variant | NIR | NDVI | NBR |
|---------|-----|------|-----|
| original | 0.227 | 0.272 | 0.737 |
| approach1 | 0.262 | 0.300 | 0.787 |
| approach2 | 0.045 | 0.065 | 0.338 |

## C2 — polygon_mean − background_mode (higher = stronger RTS contrast)

| OID | original | approach1 | approach2 |
|-----|-----|-----|-----|
| 93 | +0.0122 | -0.0002 | +0.0258 |
| 113 | +0.0131 | +0.0150 | +0.0108 |
| 136 | +0.1052 | +0.0842 | +0.1201 |
| 144 | +0.0184 | +0.0131 | +0.0222 |
| 169 | +0.0052 | +0.0253 | +0.0283 |
| 187 | +0.0071 | +0.0083 | +0.0137 |
| 262 | +0.0174 | +0.0139 | +0.0213 |