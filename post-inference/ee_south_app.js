// ============================================================
// Pan-Arctic South RTS map - public Earth Engine App source
//
// Publish (Code Editor UI, one-time): paste this script -> Apps button ->
// NEW APP -> name "south-rts-map", Cloud project pdg-project-406720,
// tick "Anyone can access" -> Publish. URL:
//   https://pdg-project-406720.projects.earthengine.app/view/south-rts-map
//
// Unlike ee_south_viewer.js (the Code Editor power-user script, which
// mosaics the 1,633 full-res probability shards live), this app uses ONLY
// ingested public EE assets - no bucket reads, no auth dependency, fast
// first paint for anonymous viewers:
//   south_likelihood_95m   max-probability overview raster (~95 m)
//   south_rts_high_confidence  19,068 high-confidence RTS polygons (529.7 km2)
//   south_rts_centroids    60,167 candidate-inventory points
// Full-res probability stays out by design (EE ingestion repeatedly failed
// on it; power users -> ee_south_viewer.js or the GCS products archive).
// ============================================================

var likelihood = ee.Image('projects/pdg-project-406720/assets/south_likelihood_95m');
var highConf = ee.FeatureCollection('projects/pdg-project-406720/assets/south_rts_high_confidence');
var centroids = ee.FeatureCollection('projects/pdg-project-406720/assets/south_rts_centroids');

// scaled_uint8 decode (value/250 = probability, 255 = NoData is masked on ingest)
var prob95 = likelihood.divide(250);

Map.setCenter(0, 65, 3);
Map.setOptions('SATELLITE');

Map.addLayer(
  prob95.updateMask(prob95.gte(0.3)),
  {min: 0.3, max: 1, palette: ['fff7bc', 'fe9929', 'd95f0e', '993404']},
  'RTS likelihood (95 m max-prob)'
);

Map.addLayer(
  centroids.style({color: '1a66ff', pointSize: 2, width: 1}),
  {},
  'All candidates (60,167 points)',
  false  // off by default - dense; toggle on to prospect
);

Map.addLayer(
  highConf.style({color: 'ff0000', fillColor: '00000000', width: 1.5}),
  {},
  'High-confidence RTS (19,068 polygons)'
);

// --- Minimal legend / about panel (print() is invisible in Apps) ---
var panel = ui.Panel({style: {width: '260px', padding: '8px'}});
panel.add(ui.Label('Pan-Arctic South RTS map', {fontWeight: 'bold', fontSize: '16px'}));
panel.add(ui.Label(
  'Retrogressive thaw slumps mapped from 2025 Q3 PlanetScope imagery ' +
  '(~50-76N). Red outlines: 19,068 high-confidence RTS (529.7 km2). Orange ' +
  'surface: max detection probability aggregated to 95 m. Blue points ' +
  '(off by default): the full 60,167-polygon candidate inventory.'));
panel.add(ui.Label(
  'High-confidence = model probability tier with QC-measured precision ' +
  '0.54-0.90 by size band - model-derived, not individually human-verified. Zoom in and compare against the satellite basemap.'));
panel.add(ui.Label('Model v2 (3-seed EffB5 ensemble, calibrated). Woodwell Climate.',
                   {fontSize: '11px', color: 'gray'}));
ui.root.add(panel);
