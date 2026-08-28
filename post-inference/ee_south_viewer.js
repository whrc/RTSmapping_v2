// ============================================================
// South RTS product viewer (pan-Arctic South, 2025 Q3 inference)
// Paste this whole file into the Earth Engine Code Editor and click Run.
//
// Assets:
//   south_rts       - EE Table asset, ingested from south_rts.gpkg (10,984 polygons)
//   south_mask      - EE Image asset, ingested from mask_cog_shards/ (1,633 COGs)
//   probability     - NOT ingested as a persistent asset: EE's ingestion pipeline
//                     repeatedly failed/restarted on this layer (continuous
//                     scaled_uint8 values needing MEAN pyramiding took far longer
//                     to process than mask's binary MODE pyramiding, even after
//                     splitting into two ~817-shard batches). Built live instead by
//                     mosaicking the source COG shards directly via loadGeoTIFF.
//
//                     loadGeoTIFF also only reads from buckets in US-CENTRAL1 (a
//                     hard EE restriction - the source shards live in US-WEST1,
//                     gs://rts-arctic-usw1/...). They're mirrored read-only at
//                     gs://rts-arctic-usc1/ee_mirror/2025q3_south/products/
//                     (US-CENTRAL1) purely so this script can read them; that
//                     mirror is not a product deliverable, just an EE-readable copy.
//
// PlanetScope basemap: not available here. Planet's only free EE integration is
// the NICFI tropical basemap program, which doesn't cover the Arctic - there is
// no public-catalog or licensed path to the actual PlanetScope imagery the model
// was trained/deployed on inside Earth Engine. Sentinel-2 (public EE catalog,
// full Arctic coverage, no ingestion needed) is added below as a free contextual
// basemap instead - not the same imagery, but real satellite context, on for
// visual orientation only.
// ============================================================

// --- RTS polygons ---
var rts = ee.FeatureCollection('projects/abruptthawmapping/assets/south_rts');

// --- Binary mask (thr 0.65) ---
var mask = ee.Image('projects/abruptthawmapping/assets/south_mask');

// --- Probability: live mosaic of the 1,633 source shards ---
// The shard grid is sparse (only cells with real detections/data exist), so
// it can't be generated from a formula - but within each row (the first 4
// digits) the columns run in contiguous stretches. Encoded as
// "row:lo-hi,single,lo-hi,...|row:...|..." and expanded below with regex,
// instead of spelling out all 1,633 ids - same data, ~19x less text.
var probPrefix = 'gs://rts-arctic-usc1/ee_mirror/2025q3_south/products/probability_cog_shards/probability_';
var probEncoded =
  '0000:20-35,40,42,55-57,84-87,92-93,95-104,112-116|' +
  '0001:22-35,43,54-57,83-85,93-104,112-117|' +
  '0002:19-24,28-35,43-44,54-57,70,83-85,93-104,108,112-114|' +
  '0003:19-23,25-36,43-44,53-56,82-84,88-109,113-115|' +
  '0004:19-37,43-45,53-56,82-84,88-110,113-117|' +
  '0005:0,8,19-26,28-38,44-45,53-56,82-83,87-117,127|' +
  '0006:0,6-11,18,20-39,44-46,53-56,60-61,71-75,82-85,87-121,127|' +
  '0007:5-40,44-46,53-56,70-76,81,84-127|' +
  '0008:0,4-40,44-46,52-55,69-77,79-127|' +
  '0009:0-1,4-41,44-46,52-53,69-127|' +
  '0010:0-42,44-46,50-52,55-56,58,68-127|' +
  '0011:0-34,36-41,44-46,49-51,55-59,68-127|' +
  '0012:0-2,4-41,45-46,49,55-59,67-71,75-127|' +
  '0013:2-41,45-46,48-49,55-58,66-69,76-77,80-127|' +
  '0014:4-31,34-41,46-49,65-68,81-127|' +
  '0015:2,4-30,36-41,46-48,65-68,84-125|' +
  '0016:4-10,13-30,35-41,48,65-67,84-123|' +
  '0017:3,7-9,15-32,35-42,66,87-114,119-122|' +
  '0018:3,5-9,15-20,22-33,35-42,90-113,119-122|' +
  '0019:4-7,17-18,24-43,96-114,119-121|' +
  '0020:3-4,26-44,97,99-106,109-114,119-120,125,127|' +
  '0021:28-44,100-106,110-115,119-120,127|' +
  '0022:32-40,43-44,101-104,110-115,118-119|' +
  '0023:102-103,112-113,117-118|' +
  '0024:112,117';
// Expand the compact "row:lo-hi,single,..." encoding back into "xxxx_yyyy"
// shard ids using regex: one match per row ("^(\d{4}):(.+)$"), then one
// match per comma-separated part to tell a range from a single value.
var probShardIds = [];
probEncoded.split('|').forEach(function(row) {
  var rowMatch = row.match(/^(\d{4}):(.+)$/);
  var x = rowMatch[1];
  rowMatch[2].split(',').forEach(function(part) {
    var range = part.match(/^(\d+)-(\d+)$/);
    var lo = range ? parseInt(range[1], 10) : parseInt(part, 10);
    var hi = range ? parseInt(range[2], 10) : lo;
    for (var y = lo; y <= hi; y++) {
      probShardIds.push(x + '_' + ('0000' + y).slice(-4));
    }
  });
});
var probUris = probShardIds.map(function(id) { return probPrefix + id + '.tif'; });
var probRaw = ee.ImageCollection(probUris.map(function(uri) {
  return ee.Image.loadGeoTIFF(uri);
})).mosaic();

// scaled_uint8 decode (writer.py SSoT): value/250 = probability, 255 = NoData
var probability = probRaw.updateMask(probRaw.neq(255)).divide(250);

// --- Visualization ---
// Not Map.centerObject(rts, ...): south_rts is 10,984 polygons and fitting to
// their unioned geometry blows EE's 2M-edge cap (3,354,240 edges). The domain
// is circumpolar (~50-76N, all longitudes) anyway, so a fixed wide view over
// the band is equivalent to "fit to extent" here.
Map.setCenter(0, 65, 3);

// Sentinel-2 context basemap (real satellite imagery, not PlanetScope - see
// header comment). A recent low-cloud composite, off by default so it doesn't
// slow the initial load; toggle on in the layer list for orientation.
var s2Context = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  .filterDate('2025-06-01', '2025-09-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();
Map.addLayer(
  s2Context,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
  'Sentinel-2 context (2025 summer)',
  false
);

Map.addLayer(
  probability,
  {min: 0, max: 1, palette: ['ffffff', 'ffff00', 'ff0000']},
  'RTS probability'
);

Map.addLayer(
  mask.updateMask(mask.eq(1)),
  {palette: ['ff0000']},
  'RTS mask (thr 0.65)',
  false  // off by default - redundant with the polygon layer
);

Map.addLayer(
  rts.style({color: 'ff0000', fillColor: '00000000', width: 1.5}),
  {},
  'RTS polygons (south_rts)'
);

print('South RTS product: 10,984 polygons, 238.08 km2 (mean_prob 0.66-0.98, median 0.84)');
print('Feature count check:', rts.size());
