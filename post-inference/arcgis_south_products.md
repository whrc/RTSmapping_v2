# Opening the pan-Arctic South RTS products in ArcGIS Pro

> **Product catalog SSoT: [`south_products.md`](south_products.md)** — the full
> product family (tiered inventory, density grids, likelihood surface),
> provenance, tier definitions, and caveats live there. This page is the
> how-to-open guide.

Deliverables from the 2025 Q3 pan-Arctic South inference run (model v2, 3-seed EffB5
ensemble, threshold 0.65, T 0.512321). All layers are **EPSG:3857 (Web Mercator)**.

## Symbolizing the rasters (which file, which zoom, which stretch)

- **`likelihood_95m.tif`** opens colormapped (embedded white→red table over
  prob 0–1) and its overviews are true block-max — hotspots stay visible at
  every zoom. No styling needed. If your GIS ignores embedded colormaps,
  stretch 0–250 with a white→red ramp.
- **`density_*_browse.tif`** are the *look-at* versions of the density grids:
  RGBA color-relief on log-percentile breaks (hotspots opaque, noise floor and
  gaps transparent). Drop on any basemap, done.
- **`density_*_expected_m2.tif`** are the *compute-with* versions (float m²
  spanning ~7 decades). A linear default stretch renders black — symbolize
  with **classified log breaks** (e.g. 10² / 10³ / 10⁴ / 10⁵ m²) or use the
  browse tif for display.
- **Density GPKGs**: graduated colors on `expected_rts_m2`, *geometric
  interval* or manual log breaks; never equal interval.

**Tiered inventory quick start:** add `south_rts_candidates.gpkg`, then a
definition query `rts_class = 'high_confidence'` gives the zero-decision fact map
(equivalently, add `south_rts_high_confidence.gpkg`); relax to `conf_class` tiers or
filter continuously with `max_prob >= x` and/or `area_m2 >= a` — those two
columns *are* the user-side MMU dial. Symbolize Unique Values on `rts_class`
(high_confidence solid red, candidate orange outline, marginal pale dashed) — see
the catalog for the measured precision behind each class.

**Headline:** **60,167 candidate polygons / 688.2 km²** (MMU≈0, thr 0.30), of
which **19,068 high_confidence / 529.7 km²** (measured precision 0.54–0.90 by size
band), across the pan-Arctic South band (≈50–76°N), from 41,567,572 inferred
tiles. Probability raster: 1,633 super-tile COGs (8.4 M × 1.6 M px canvas,
sparse).

**Two ways to view this in ArcGIS Pro:** the manual steps below (add each layer
yourself, use a generic Imagery basemap for context), or the automated pair of
scripts (`scripts/build_rgb_chips.py` + `scripts/build_arcgis_project.py`) that
also generates real RGB "underlying tile" context chips — the actual Planet
imagery the model saw for each detection, not a generic basemap — and builds
the whole `.aprx` with layers + symbology in one run. Region-generic (works for
Banks and South, `.tif` or sharded `.vrt` alike). See the scripts' own
docstrings for invocation; the manual path below still works standalone if you
just want the polygons/rasters without the RGB chips.

**No ArcGIS license? Use Google Earth Engine instead:** `post-inference/ee_south_viewer.js`
is a ready-to-paste Code Editor script (`south_rts` + `south_mask` are ingested
as real EE assets under `projects/pdg-project-406720/assets/`; the probability
layer is mosaicked live from its 1,633 source COG shards via `loadGeoTIFF` —
EE's ingestion pipeline repeatedly failed on that specific layer, see the
script's header comment for why). No install, no download, opens in a browser.

**No account at all? The public GEE App** (source `post-inference/ee_south_app.js`,
published at `https://abruptthawmapping.projects.earthengine.app/view/south-rts-map`)
shows the high-confidence inventory + 95 m likelihood surface to anyone with the
link — no Earth Engine account, no data access needed. Built from small
ingested assets only (`south_likelihood_95m`, `south_rts_high_confidence`,
`south_rts_centroids`), so it loads fast; full-res probability stays in the
Code Editor script above.

## 1. The products (in GCS)

Bucket prefix: `gs://rts-mapping-v2-usw1/inference/2025q3_south/products/`

| File | What it is |
|---|---|
| `south_rts.gpkg` | **The RTS polygon inventory** — one polygon per detected slump (the primary product). |
| `probability.vrt` + `probability_cog_shards/*.tif` | Probability mosaic. Pixel value = **prob × 250** (`scaled_uint8`); **255 = NoData**. The `.vrt` ties the super-tile COGs together — keep it next to its `probability_cog_shards/` folder. |
| `mask.vrt` + `mask_cog_shards/*.tif` | Binary RTS mask at thr 0.65 (1 = RTS, 0 = background, 255 = NoData). Optional. |
| `region_log.json` | Assembly metadata (canvas, threshold, σ, block/cog sizes, tile counts). |

`south_rts.gpkg` attributes: `rts_id`, `area_m2`, `perimeter_m`, `centroid_lat`,
`centroid_lon`, `mean_prob`, `max_prob`, `detection_scale`, `tile_ids`.

## 2. Download to a local folder (preserve the structure)

The `.vrt` files reference their COGs by **relative path**, so download the whole
`products/` folder as-is. Using the Google Cloud SDK (`gcloud`/`gsutil`):

```bat
gcloud storage cp -r gs://rts-mapping-v2-usw1/inference/2025q3_south/products C:\rts\south_products
```

You should end up with `C:\rts\south_products\` containing `south_rts.gpkg`,
`probability.vrt`, `probability_cog_shards\`, `mask.vrt`, `mask_cog_shards\`,
`region_log.json`.

*(If you only want the polygons, `south_rts.gpkg` alone is self-contained — the VRTs
are only needed for the raster.)*

## 3. Add the layers in ArcGIS Pro

1. **New Map** (any basemap). The Coordinate System can stay default — the layers
   carry EPSG:3857 and will project on the fly.
2. **Polygons:** *Add Data* → browse into `south_rts.gpkg` → add the `south_rts`
   feature class. This is the RTS inventory.
3. **Probability raster (optional):** *Add Data* → `probability.vrt`.
   - It's `scaled_uint8`: displayed pixel values run **0–250 = probability × 250**
     (so 163 ≈ p 0.65, the detection threshold), NoData 255.
   - To view as true probability 0–1, apply a **Raster Function → Arithmetic**
     (Divide by 250), or just read the stretch knowing value/250 = prob.
4. **Mask (optional):** *Add Data* → `mask.vrt` — a 1/0 RTS footprint.

## 4. Suggested symbology & context

- **Polygons by confidence:** symbolize `south_rts` with graduated colors on
  `mean_prob` (or `max_prob`) to triage strong vs marginal detections.
- **Size filter / triage:** `area_m2` is the geodesic slump area. The tiered
  inventory has **no minimum mapping unit** (floor ~2 px ≈ 10–45 m²) — use a
  Definition Query on `area_m2` if you want your own MMU, and `nodata_frac`
  to soft-screen NoData-context false positives.
- **Imagery context:** add an *Imagery* basemap, or the 2025 PlanetScope quads, to
  see each polygon over the terrain it was detected on.
- **Jump to a detection:** `centroid_lat`/`centroid_lon` (WGS84) are handy for
  locating a slump or sharing coordinates.

## 5. Notes & caveats

- **Coverage gaps:** 3 Planet quads were absent from the source bucket
  (`1459-1437`, `1153-1566`, `1189-1531`) and are NoData in those footprints — a
  negligible fraction of the domain, but worth knowing if a specific area reads empty.
- **Threshold / calibration:** polygons are at the deployed operating point
  (thr 0.65 on the temperature-scaled ensemble, `min_blob` 2000). The probability
  raster lets you re-threshold if you want a different precision/recall trade-off.
- **CRS:** everything is EPSG:3857; areas in `area_m2` are geodesic (computed on the
  ellipsoid), not planar Web-Mercator areas, so they are correct despite the
  projection's area distortion at high latitude.
