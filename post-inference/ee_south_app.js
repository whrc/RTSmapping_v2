// ============================================================
// Pan-Arctic South RTS map — public Earth Engine App source
//
// Publish (Code Editor UI): paste this script -> Apps button -> the existing
// "south-rts-map" app -> Update. Cloud project abruptthawmapping (the assets
// live in pdg-project-406720 but are public, so any project can host the app).
//   https://abruptthawmapping.projects.earthengine.app/view/south-rts-map
//
// Assets-only (no live COG mosaic) — no bucket reads, no auth dependency, fast
// first paint for anonymous viewers. Power users who need the full-resolution
// probability go to ee_south_viewer.js or the GCS products archive.
//
// WHAT CHANGED (2026-08, replacing the 3-layer version)
//   * The 95 m max-probability layer is GONE. It was ingested as a continuous
//     raster, so EE re-pyramids it with MEAN; zoomed out the mean fell below
//     the layer's own .gte(0.3) mask and the layer erased itself, and zoomed in
//     95 m is coarse against 50-500 m slumps. Re-ingesting with MAX pyramiding
//     would make it visible but not quantitative (max-prob over a 10 km screen
//     pixel saturates to ~1.0 wherever a single detection exists), so the
//     overview is now the threshold-free 10 km expected-area density instead.
//   * Two nested contours: the 0.30 outline (= the candidate inventory's own
//     geometry) and the 0.65 core (south_rts_t65, vectorized 2026-08-03 from
//     the same probability shards at MMU~0).
//   * A minimum-size (MMU) control, because the shipped inventory has NO
//     minimum mapping unit and the question "what does min_blob cost me?" had
//     no answer in the map.
//
// PERFORMANCE CONTRACT (why this file is written the way it is)
//   * No server-side aggregation in the interaction loop. Every count and area
//     in the panel comes from APP_STATS, precomputed by
//     scripts/build_ee_app_stats.py. .evaluate() appears exactly once, in the
//     click inspector, where a single spatially-indexed first() is genuinely
//     needed. Aggregating a filtered 60k collection live costs seconds a tick.
//   * Only the density layer paints at the opening zoom. Building an ee.*
//     object is free — it is only a description — so all layers are created up
//     front and the cost is controlled purely by `shown`. EE requests no tiles
//     for a hidden layer, so updating a hidden layer's eeObject is free too.
//   * Outlines use ee.Image().paint(), not fc.style(): style() rasterizes a
//     transparent fill for every polygon, paint() only strokes the boundary.
//   * The zoom handler returns early when the band is unchanged, so stepping
//     z13 -> z14 -> z15 costs nothing.
// ============================================================

// ---------- assets (all public) ----------
var DENSITY   = ee.Image('projects/pdg-project-406720/assets/south_density_10km');
var CAND      = ee.FeatureCollection('projects/pdg-project-406720/assets/south_rts_candidates');
var T65       = ee.FeatureCollection('projects/pdg-project-406720/assets/south_rts_t65');
var CENTROIDS = ee.FeatureCollection('projects/pdg-project-406720/assets/south_rts_centroids');

// ---------- generated: scripts/build_ee_app_stats.py — do not hand-edit ----------
// Sources: south_rts_attributes.parquet, south_rts_t65.gpkg, region_log.json,
//          configs/deployment.yaml
var APP_STATS = {
  "ladder_m2": [0, 10, 20, 30, 50, 79, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 4532, 5000, 6000, 8000, 10000, 12000, 15000, 20000],
  "series": {
    "high": {
      "n": [19068, 19068, 19068, 19068, 19068, 19068, 19068, 19068, 19068, 19068, 19068, 19068, 19067, 19041, 18947, 18448, 17731, 17320, 16905, 16029, 14446, 12989, 11730, 10172, 8055],
      "km2": [529.703853, 529.703853, 529.703853, 529.703853, 529.703853, 529.703853, 529.703853, 529.703853, 529.703853, 529.703853, 529.703853, 529.703853, 529.702853, 529.66863, 529.501233, 528.22197, 525.692892, 523.939902, 521.96561, 517.159633, 506.093163, 492.993892, 479.195105, 458.210139, 421.382706]
    },
    "medium": {
      "n": [11865, 11865, 11865, 11865, 11865, 11865, 11865, 11865, 11865, 11864, 11840, 11753, 11564, 11045, 10385, 9121, 8023, 7502, 7076, 6253, 4948, 3966, 3205, 2366, 1448],
      "km2": [119.041991, 119.041991, 119.041991, 119.041991, 119.041991, 119.041991, 119.041991, 119.041991, 119.041991, 119.041718, 119.031406, 118.975013, 118.806623, 118.151759, 116.995141, 113.845881, 110.011216, 107.787369, 105.763042, 101.25015, 92.212919, 83.409817, 75.074727, 63.88316, 48.044889]
    },
    "low": {
      "n": [29234, 27169, 25867, 25099, 23871, 22473, 21632, 19929, 18549, 16335, 13345, 10928, 9177, 6734, 5270, 3459, 2465, 2097, 1827, 1412, 868, 547, 365, 217, 98],
      "km2": [39.427449, 39.41503, 39.396295, 39.377381, 39.328694, 39.239082, 39.16392, 38.95229, 38.711282, 38.161365, 36.986248, 35.495698, 33.969674, 30.981342, 28.439055, 23.999115, 20.556148, 18.988957, 17.705319, 15.430707, 11.680597, 8.811355, 6.817245, 4.856015, 2.827798]
    },
    "t65": {
      "n": [23682, 23009, 22655, 22407, 22025, 21604, 21340, 20843, 20451, 19772, 18815, 17852, 17133, 15930, 14931, 13150, 11753, 11072, 10566, 9583, 7951, 6836, 5876, 4724, 3467],
      "km2": [259.90921, 259.905238, 259.900262, 259.894126, 259.878987, 259.852028, 259.828583, 259.766998, 259.699302, 259.53187, 259.156154, 258.559881, 257.931723, 256.438622, 254.693404, 250.257464, 245.387176, 242.488305, 240.076156, 234.688198, 223.371684, 213.362887, 202.857176, 187.415965, 165.607064]
    }
  },
  "min_blob": {
    "px": 2000,
    "resolution_m": 4.777314267158508,
    "by_lat": {"50": 18860, "60": 11411, "70": 5340, "76": 2671},
    "median_lat": 71.63,
    "representative_m2": 4532
  },
  "totals": {"candidates_n": 60167, "candidates_km2": 688.17,
             "t65_n": 23682, "t65_km2": 259.91}
};

// ---------- palette ----------
// Categorical (identity: which contour). Slots 1-2 of the reference data-viz
// palette, unmodified — the documented pair passes the all-pairs CVD and
// normal-vision gates. The same two hues carry the same two inventories in the
// map and in the chart, so identity never depends on where you are looking.
var C30 = '#2a78d6';   // blue   — 0.30 outline (the shipped inventory geometry)
var C65 = '#eb6834';   // orange — 0.65 core

// Sequential (magnitude: expected RTS area). One warm hue family, light to
// dark — ColorBrewer YlOrBr, as the retired 95 m layer used, so returning
// viewers read the overview the same way.
var DENSITY_PALETTE = ['#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014',
                       '#cc4c02', '#8c2d04'];
// Confidence tier is ordered, not nominal, so the points get a sequential ramp
// rather than three unrelated hues.
var TIER_COLOR = {low: '#fed976', medium: '#fd8d3c', high: '#e31a1c'};

// Measured South QC precision, 2026-07 stratified rating (south_products.md).
var TIER_LABEL = {
  high:   'high  ≥ 0.65  — QC precision 0.54–0.90',
  medium: 'medium 0.45–0.65 — QC precision 0.11–0.53',
  low:    'low   0.30–0.45 — QC precision 0.00–0.31'
};

var TIERS = ['high', 'medium', 'low'];
var LADDER = APP_STATS.ladder_m2;

// Zoom bands. A ~170 m slump is ~3 px at z10 and ~13 px at z12, so outlines
// only start meaning something around z12; below that the points carry the
// locations and the 10 km density carries the pattern.
var Z_POINTS = 8;      // >= this zoom: points instead of the density surface
var Z_CONTOUR = 12;    // >= this zoom: the nested contour pair

// ---------- state ----------
var state = {
  mmuIdx: 0,
  tiers: {high: true, medium: true, low: true},
  auto: true,
  band: null
};

function activeTiers() {
  return TIERS.filter(function(t) { return state.tiers[t]; });
}

function mmu() { return LADDER[state.mmuIdx]; }

// ---------- map ----------
Map.setOptions('SATELLITE');
Map.style().set('cursor', 'crosshair');

function sizeFilter() { return ee.Filter.gte('area_m2', mmu()); }

function candFiltered() {
  var fc = CAND.filter(sizeFilter());
  var on = activeTiers();
  if (on.length < TIERS.length) {
    fc = fc.filter(ee.Filter.inList('conf_class', on));
  }
  return fc;
}

// paint() strokes the boundary only; style() would rasterize a transparent
// fill for all 60k polygons to draw the same line.
function outline(fc, color, width) {
  return ee.Image().byte()
           .paint({featureCollection: fc, color: 1, width: width})
           .visualize({min: 0, max: 1, palette: [color]});
}

function pointsImage() {
  var on = activeTiers();
  if (on.length === 0) return ee.Image().byte();       // nothing ticked
  // mosaic() puts the LAST image on top, and TIERS runs high→low, so reverse:
  // otherwise low-confidence points would bury the high-confidence ones.
  var styled = on.slice().reverse().map(function(t) {
    return CENTROIDS.filter(ee.Filter.eq('conf_class', t))
                    .filter(sizeFilter())
                    .style({color: TIER_COLOR[t], pointSize: 3, width: 1});
  });
  return ee.ImageCollection(styled).mosaic();
}

// Expected RTS area per 10 km cell, log-stretched. The field is extremely
// skewed — of the 1,159,928 valid cells, 68% carry some expected area but the
// median is only 43 m² and 66.6% of all the area sits in the 2.4% of cells
// above 10⁴ m². So the floor is 100 m² (≈p60) rather than 0: masking at >0
// would wash two thirds of the Arctic in the palest step and bury the
// hotspots, which is the failure the retired 95 m layer already had. The ramp
// then runs 10² → 10⁵ m² (≈p99.8), so it spans the part of the distribution
// that carries the signal and the top 0.2% saturates, as a browse layer should.
var DENSITY_FLOOR_M2 = 100;
var densityVis = DENSITY.updateMask(DENSITY.gte(DENSITY_FLOOR_M2)).log10()
                        .visualize({min: 2, max: 5, palette: DENSITY_PALETTE,
                                    opacity: 0.75});

var L = {
  density:  ui.Map.Layer({eeObject: densityVis, name: 'Expected RTS area (10 km)', shown: true}),
  points:   ui.Map.Layer({eeObject: pointsImage(), name: 'Detections (points, by tier)', shown: false}),
  c30:      ui.Map.Layer({eeObject: outline(candFiltered(), C30, 1),
                          name: '0.30 outline', shown: false}),
  c65:      ui.Map.Layer({eeObject: outline(T65.filter(sizeFilter()), C65, 2),
                          name: '0.65 core', shown: false}),
  // blank masked image, not an empty FeatureCollection — an empty FC has no
  // geometry for EE to render and errors on the first paint
  pick:     ui.Map.Layer({eeObject: ee.Image().byte(), name: 'selection', shown: true})
};
['density', 'points', 'c30', 'c65', 'pick'].forEach(function(k) {
  Map.layers().add(L[k]);
});

// Updating a hidden layer is free — EE requests no tiles for it — so there is
// no dirty-tracking here; the `shown` flags do all the work.
function refreshLayers() {
  var cand = candFiltered();
  L.c30.setEeObject(outline(cand, C30, 1));
  L.c65.setEeObject(outline(T65.filter(sizeFilter()), C65, 2));
  L.points.setEeObject(pointsImage());
}

function bandFor(z) {
  if (z >= Z_CONTOUR) return 'contour';
  if (z >= Z_POINTS) return 'points';
  return 'density';
}

function applyBand(z) {
  var band = bandFor(z);
  if (band === state.band) return;                     // nothing to redraw
  state.band = band;
  if (state.auto) {
    L.density.setShown(band === 'density');
    L.points.setShown(band === 'points');
    L.c30.setShown(band === 'contour' && showC30.getValue());
    L.c65.setShown(band === 'contour' && showC65.getValue());
  }
  bandHint.setValue(
    band === 'density' ? 'Overview: expected RTS area per 10 km cell (threshold-free). Zoom in for detections.' :
    band === 'points'  ? 'Detections as points. Zoom to ' + Z_CONTOUR + '+ for outlines.' :
                         'Nested contours: 0.30 outline and 0.65 core. Click a detection to inspect it.');
}

// ---------- controls ----------
var bandHint = ui.Label('', {fontSize: '11px', color: '#52514e', margin: '4px 0'});

var readout = ui.Label('', {fontSize: '12px', whiteSpace: 'pre', margin: '4px 0 2px 0'});

// Attributes can arrive null (a polygon whose window held only NoData leaves
// NaN in the source, which serialises to null) — never let that throw inside
// the inspector callback.
function q(v, d) {
  return (v === null || v === undefined || isNaN(v)) ? '—' : Number(v).toFixed(d);
}

function fmt(n) {
  if (n === null || n === undefined || isNaN(n)) return '—';
  // thousands separators without relying on toLocaleString in the EE sandbox
  var s = String(Math.round(n)), out = '', i;
  for (i = 0; i < s.length; i++) {
    out += s.charAt(i);
    if ((s.length - 1 - i) % 3 === 0 && i < s.length - 1) out += ',';
  }
  return out;
}

function seriesAt(keys, idx) {
  var n = 0, km2 = 0;
  keys.forEach(function(k) {
    n += APP_STATS.series[k].n[idx];
    km2 += APP_STATS.series[k].km2[idx];
  });
  return {n: n, km2: km2};
}

function line(label, keys) {
  var now = seriesAt(keys, state.mmuIdx), all = seriesAt(keys, 0);
  var pn = all.n ? 100 * now.n / all.n : 0;
  var pa = all.km2 ? 100 * now.km2 / all.km2 : 0;
  return label + '  ' + fmt(now.n) + ' obj · ' + now.km2.toFixed(1) + ' km²' +
         '   (' + pn.toFixed(0) + '% of objects, ' + pa.toFixed(0) + '% of area)';
}

// Pure client-side: this is the reason the ladders are precomputed.
function updateReadout() {
  var on = activeTiers();
  mmuLabel.setValue('Minimum size: ' + (mmu() === 0 ? 'none (as shipped)'
                                                    : fmt(mmu()) + ' m²'));
  readout.setValue(
    (on.length ? line('0.30 outlines', on) : '0.30 outlines  — no tier selected') +
    '\n' + line('0.65 cores   ', ['t65']));
}

var mmuLabel = ui.Label('', {fontWeight: 'bold', fontSize: '13px', margin: '6px 0 0 0'});

var slider = ui.Slider({
  min: 0, max: LADDER.length - 1, value: 0, step: 1,
  style: {stretch: 'horizontal', margin: '2px 6px'}
});
// The readout tracks the drag (free); the map rebuild waits for a pause.
var rebuildSoon = ui.util.debounce(function() { refreshLayers(); syncUrl(); }, 350);
slider.onChange(function(v) {
  state.mmuIdx = v;
  updateReadout();
  rebuildSoon();
});

function setMmu(valueM2) {
  var idx = 0, i;
  for (i = 0; i < LADDER.length; i++) if (LADDER[i] <= valueM2) idx = i;
  state.mmuIdx = idx;
  slider.setValue(idx, false);
  updateReadout();
  refreshLayers();
  syncUrl();
}

function presetButton(label, valueM2) {
  return ui.Button({
    label: label,
    onClick: function() { setMmu(valueM2); },
    style: {margin: '2px 2px', padding: '0px'}
  });
}

var mb = APP_STATS.min_blob;
var presets = ui.Panel(
  [presetButton('none', 0),
   presetButton('79 m² (ARTS P1)', 79),
   presetButton('min_blob 2000 px', mb.representative_m2)],
  ui.Panel.Layout.flow('horizontal'), {margin: '0px'});

var minBlobNote = ui.Label(
  'The shipped inventory has NO minimum mapping unit (2 px technical floor). ' +
  'The legacy 0.65 product used min_blob = ' + mb.px + ' px — a count of ' +
  'EPSG:3857 pixels, whose ground area is res²·cos²(lat), so that ' +
  'floor slides from ≈' + fmt(mb.by_lat['50']) + ' m² at 50°N to ' +
  '≈' + fmt(mb.by_lat['76']) + ' m² at 76°N. The preset uses ' +
  fmt(mb.representative_m2) + ' m², its value at this inventory’s median ' +
  'latitude (' + mb.median_lat + '°N).',
  {fontSize: '11px', color: '#52514e', margin: '4px 0'});

// Tier checkboxes, not a max_prob slider: object precision PEAKS at 0.65 and
// falls above it (south_products.md caveat 1), so a slider would invite users
// to crank it to 0.9 and get a strictly worse map.
var tierBoxes = TIERS.map(function(t) {
  var cb = ui.Checkbox({
    label: TIER_LABEL[t], value: true,
    style: {fontSize: '11px', margin: '1px 0'},
    onChange: function(v) {
      state.tiers[t] = v;
      updateReadout();
      refreshLayers();
      syncUrl();
    }
  });
  return cb;
});

// With auto off the checkbox is the only authority — otherwise ticking a
// contour at z5 would silently do nothing and look broken.
function contourShown(v) { return v && (!state.auto || state.band === 'contour'); }

var showC30 = ui.Checkbox({
  label: '0.30 outline', value: true, style: {fontSize: '11px', margin: '1px 0'},
  onChange: function(v) { L.c30.setShown(contourShown(v)); syncUrl(); }
});
var showC65 = ui.Checkbox({
  label: '0.65 core', value: true, style: {fontSize: '11px', margin: '1px 0'},
  onChange: function(v) { L.c65.setShown(contourShown(v)); syncUrl(); }
});
var autoBox = ui.Checkbox({
  label: 'Auto-select layers by zoom', value: true,
  style: {fontSize: '11px', margin: '4px 0'},
  onChange: function(v) {
    state.auto = v;
    if (v) { state.band = null; applyBand(Map.getZoom()); }
  }
});

// ---------- retention chart ----------
// Static: no EE compute. Colour carries the inventory (same hues as the map);
// solid vs dashed carries objects vs area, so the measure never becomes a hue.
function retentionChart() {
  var rows = [['Minimum size (m²)',
               '0.30 outlines — objects', '0.30 outlines — area',
               '0.65 cores — objects', '0.65 cores — area']];
  var base30 = seriesAt(TIERS, 0), base65 = seriesAt(['t65'], 0);
  var i;
  for (i = 1; i < LADDER.length; i++) {      // from 10 m²: a log axis excludes 0
    var a = seriesAt(TIERS, i), b = seriesAt(['t65'], i);
    rows.push([LADDER[i],
               100 * a.n / base30.n, 100 * a.km2 / base30.km2,
               100 * b.n / base65.n, 100 * b.km2 / base65.km2]);
  }
  return ui.Chart(rows, 'LineChart', {
    title: 'What a minimum-size filter costs',
    titleTextStyle: {fontSize: 12},
    hAxis: {title: 'Minimum size (m², log)', logScale: true,
            titleTextStyle: {fontSize: 10}, textStyle: {fontSize: 9}},
    vAxis: {title: '% retained', minValue: 0, maxValue: 100,
            titleTextStyle: {fontSize: 10}, textStyle: {fontSize: 9}},
    series: {0: {color: C30, lineDashStyle: [1, 0]},
             1: {color: C30, lineDashStyle: [4, 3]},
             2: {color: C65, lineDashStyle: [1, 0]},
             3: {color: C65, lineDashStyle: [4, 3]}},
    lineWidth: 2, pointSize: 0,
    legend: {position: 'bottom', textStyle: {fontSize: 9}},
    chartArea: {width: '78%', height: '58%'},
    height: 200
  });
}

// ---------- click inspector ----------
var inspector = ui.Label('Click a detection to inspect it.',
                         {fontSize: '11px', whiteSpace: 'pre', margin: '4px 0'});

Map.onClick(function(coords) {
  var pt = ee.Geometry.Point([coords.lon, coords.lat]);
  L.pick.setEeObject(ee.FeatureCollection([ee.Feature(pt)])
                       .style({color: '#ffffff', pointSize: 6, width: 2}));
  inspector.setValue('reading…');
  // The one .evaluate() in this app: a single spatially-indexed lookup.
  CAND.filterBounds(pt.buffer(40)).first().evaluate(function(f) {
    if (!f) {
      inspector.setValue('No detection within 40 m of that point.');
      return;
    }
    var p = f.properties;
    inspector.setValue(
      'rts_id ' + p.rts_id + '   (' + p.conf_class + ' / ' + p.rts_class + ')\n' +
      'probability   max ' + q(p.max_prob, 3) + '  mean ' + q(p.mean_prob, 3) + '\n' +
      'area @0.30    ' + fmt(p.area_m2) + ' m²\n' +
      '  @0.45 ' + fmt(p.a_t45) + '   @0.65 ' + fmt(p.a_t65) + '   @0.80 ' + fmt(p.a_t80) + ' m²\n' +
      'NoData frac   ' + q(p.nodata_f, 3) + '   (soft triage only)\n' +
      'centroid      ' + q(p.clat, 4) + ', ' + q(p.clon, 4));
  });
});

// ---------- URL state (shareable deep links) ----------
var urlReady = false;
var lastCenter = null;

function syncUrl() {
  if (!urlReady) return;
  ui.url.set('mmu', mmu());
  ui.url.set('t', activeTiers().map(function(t) { return t.charAt(0); }).join(''));
  ui.url.set('c', (showC30.getValue() ? '3' : '') + (showC65.getValue() ? '6' : ''));
}

// The centre comes from onChangeCenter's client-side {lon, lat}. Map.getCenter()
// would return an ee.Geometry needing a blocking getInfo() — a server round-trip
// on every pan, which is exactly what this app promises not to do.
var syncView = ui.util.debounce(function() {
  if (!urlReady || !lastCenter) return;
  ui.url.set('lon', Math.round(lastCenter.lon * 1e4) / 1e4);
  ui.url.set('lat', Math.round(lastCenter.lat * 1e4) / 1e4);
  ui.url.set('z', Map.getZoom());
}, 900);

function num(key, dflt) {
  var v = ui.url.get(key, dflt);
  return typeof v === 'number' ? v : parseFloat(v);
}

// ---------- panel ----------
function heading(text) {
  return ui.Label(text, {fontWeight: 'bold', fontSize: '12px',
                         margin: '10px 0 2px 0'});
}

var controls = ui.Panel([
  heading('Minimum size (MMU)'),
  mmuLabel, slider, presets, readout, minBlobNote,
  heading('Confidence tier  — 0.30 outlines and points'),
  ui.Panel(tierBoxes, ui.Panel.Layout.flow('vertical'), {margin: '0px'}),
  ui.Label('The 0.65 core layer is all-high by construction, so tiers do not ' +
           'apply to it. Precision is NOT monotonic in probability: it peaks at ' +
           '0.65 and falls above it, so filtering harder than "high" gives a ' +
           'worse map, not a better one.',
           {fontSize: '11px', color: '#52514e', margin: '2px 0'}),
  heading('Layers'),
  showC30, showC65, autoBox, bandHint,
  heading('Minimum-size impact'),
  retentionChart(),
  heading('Inspector'),
  inspector
], ui.Panel.Layout.flow('vertical'), {padding: '0px 4px'});

var controlsHolder = ui.Panel([controls], ui.Panel.Layout.flow('vertical'),
                              {margin: '0px'});
var toggle = ui.Button({
  label: 'Hide controls ▲',
  style: {margin: '4px 0', stretch: 'horizontal'},
  onClick: function() {
    var open = controlsHolder.widgets().length() > 0;
    if (open) { controlsHolder.clear(); toggle.setLabel('Show controls ▼'); }
    else { controlsHolder.add(controls); toggle.setLabel('Hide controls ▲'); }
  }
});

function colorBar(palette) {
  return ui.Thumbnail({
    image: ee.Image.pixelLonLat().select('longitude'),
    params: {bbox: [0, 0, 1, 0.1], dimensions: '210x14', format: 'png',
             min: 0, max: 1, palette: palette},
    style: {stretch: 'horizontal', margin: '2px 0 0 0'}
  });
}

var legend = ui.Panel([
  ui.Label('Expected RTS area per 10 km cell', {fontSize: '11px', margin: '4px 0 0 0'}),
  colorBar(DENSITY_PALETTE),
  ui.Panel([ui.Label('100 m²', {fontSize: '9px', margin: '0px'}),
            ui.Label('≥ 0.1 km²', {fontSize: '9px', margin: '0px 0px 0px 140px'})],
           ui.Panel.Layout.flow('horizontal'), {margin: '0px'})
], ui.Panel.Layout.flow('vertical'), {margin: '0px'});

var panel = ui.Panel({style: {width: '340px', padding: '8px'}});
panel.add(ui.Label('Pan-Arctic South RTS map',
                   {fontWeight: 'bold', fontSize: '16px', margin: '0px'}));
panel.add(ui.Label(
  'Retrogressive thaw slumps mapped from 2025 Q3 PlanetScope imagery ' +
  '(≈50–76°N). ' + fmt(APP_STATS.totals.candidates_n) +
  ' candidate polygons / ' + APP_STATS.totals.candidates_km2 + ' km² outlined at ' +
  'probability 0.30, of which ' + fmt(APP_STATS.series.high.n[0]) + ' / ' +
  APP_STATS.series.high.km2[0].toFixed(1) + ' km² are high-confidence. The 0.65 ' +
  'core layer re-cuts the same probabilities at 0.65: ' +
  fmt(APP_STATS.totals.t65_n) + ' cores / ' + APP_STATS.totals.t65_km2 + ' km².',
  {fontSize: '11px', margin: '4px 0'}));
panel.add(ui.Label(
  'High-confidence is a model probability tier with QC-measured precision ' +
  '0.54–0.90 by size band — model-derived, not individually human-verified. ' +
  'Compare against the satellite basemap before believing any single polygon.',
  {fontSize: '11px', color: '#52514e', margin: '2px 0'}));
panel.add(legend);
panel.add(toggle);
panel.add(controlsHolder);
panel.add(ui.Label(
  'Model v2 (3-seed EffB5 ensemble, calibrated T=0.512321). Woodwell Climate. ' +
  'Full products: gs://rts-mapping-v2-usw1/inference/2025q3_south/products/',
  {fontSize: '10px', color: '#8a8a85', margin: '8px 0 0 0'}));
ui.root.add(panel);

// ---------- restore + start ----------
setMmu(num('mmu', 0));
var tParam = ui.url.get('t', 'hml');
TIERS.forEach(function(t, i) {
  var on = String(tParam).indexOf(t.charAt(0)) >= 0;
  state.tiers[t] = on;
  tierBoxes[i].setValue(on, false);
});
var cParam = String(ui.url.get('c', '36'));
showC30.setValue(cParam.indexOf('3') >= 0, false);
showC65.setValue(cParam.indexOf('6') >= 0, false);

Map.setCenter(num('lon', 0), num('lat', 65), num('z', 3));
updateReadout();
refreshLayers();
applyBand(Map.getZoom());

Map.onChangeZoom(function(z) { applyBand(z); syncView(); });
Map.onChangeCenter(function(c) { lastCenter = c; syncView(); });
urlReady = true;
