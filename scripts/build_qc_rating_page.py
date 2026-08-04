"""Offline QC rating page: one self-contained HTML, instant navigation.

Replaces the GEE Code Editor rater whose per-polygon `loadGeoTIFF` tile loads
made rating painfully slow (and whose export needed a live session — one page
hang lost a full 280-polygon rating round). Here every polygon's imagery is
pre-rendered from the local chip mosaic into two embedded PNGs (tight ~3× the
feature and wide ~1.5 km context, red outline burned in). The page runs from a
local file: keyboard rating (1=rts 2=false 3=unsure, ←/→ navigate), verdicts
autosaved to localStorage on every keystroke, and EXPORT downloads
qc_ratings.csv directly — no server, no auth, nothing to lose.

Usage:
    python scripts/build_qc_rating_page.py \
        --sample /outputs/.../qc_sample.gpkg \
        --chips-vrt /outputs/.../qc_chips/rgb_chips.vrt \
        --out /outputs/.../qc_rater.html
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from review.crops import crop_bounds, render_crop  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _data_uri(src, geoms, crop, png_px: int) -> str:
    """Rendered crop as an inline data-URI — this page embeds its imagery."""
    return ("data:image/jpeg;base64,"
            + base64.b64encode(render_crop(src, geoms, crop, png_px)).decode())


_HTML = """<!doctype html><meta charset="utf-8"><title>RTS QC rater (offline)</title>
<style>
body{font-family:system-ui,sans-serif;background:#191919;color:#eee;margin:0;
     display:flex;flex-direction:column;align-items:center}
#imgs{display:flex;gap:8px;margin:10px}
#imgs img{width:44vw;max-width:560px;image-rendering:auto;border:1px solid #444}
#bar{display:flex;gap:10px;align-items:center;margin:6px;flex-wrap:wrap}
button{font-size:15px;padding:6px 14px;cursor:pointer}
.v-rts{background:#2a78d6;color:#fff}.v-false{background:#666;color:#fff}
.v-unsure{background:#eda100;color:#000}
#meta{font-size:15px}#tally{color:#aaa;font-size:13px}
kbd{background:#333;border-radius:3px;padding:1px 5px}
</style>
<div id="bar">
  <span id="meta"></span>
  <button class="v-rts" onclick="rate('rts')">RTS <kbd>1</kbd></button>
  <button class="v-false" onclick="rate('false')">false <kbd>2</kbd></button>
  <button class="v-unsure" onclick="rate('unsure')">unsure <kbd>3</kbd></button>
  <button onclick="nav(-1)">◀ <kbd>←</kbd></button>
  <button onclick="nav(1)">▶ <kbd>→</kbd></button>
  <button onclick="exportCsv()" style="background:#0a0">EXPORT qc_ratings.csv</button>
  <span id="tally"></span>
</div>
<div id="imgs"><img id="tight"><img id="wide"></div>
<script>
var ITEMS = __ITEMS__;
var KEY = 'qc_ratings_v2';
var verdicts = JSON.parse(localStorage.getItem(KEY) || '{}');
var idx = 0;
while (idx < ITEMS.length - 1 && verdicts[ITEMS[idx].id]) idx++;  // resume

function show() {
  var f = ITEMS[idx];
  document.getElementById('tight').src = f.t;
  document.getElementById('wide').src  = f.w;
  document.getElementById('meta').textContent =
      (idx + 1) + '/' + ITEMS.length + '  [id ' + f.id + ']  tier=' + f.cls +
      '  area=' + Math.round(f.a) + ' m2  verdict: ' + (verdicts[f.id] || '-');
  var n = Object.keys(verdicts).length;
  document.getElementById('tally').textContent =
      n + ' rated (autosaved locally)';
}
function rate(v) {
  verdicts[ITEMS[idx].id] = v;
  localStorage.setItem(KEY, JSON.stringify(verdicts));
  nav(1);
}
function nav(d) {
  idx = Math.max(0, Math.min(ITEMS.length - 1, idx + d));
  show();
}
function exportCsv() {
  var lines = ['rts_id,qc_verdict'];
  ITEMS.forEach(function (f) {
    if (verdicts[f.id]) lines.push(f.id + ',' + verdicts[f.id]);
  });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\\n')],
                                        {type: 'text/csv'}));
  a.download = 'qc_ratings.csv';
  a.click();
}
document.addEventListener('keydown', function (e) {
  if (e.key === '1') rate('rts');
  else if (e.key === '2') rate('false');
  else if (e.key === '3') rate('unsure');
  else if (e.key === 'ArrowLeft') nav(-1);
  else if (e.key === 'ArrowRight') nav(1);
});
show();
</script>"""


def build_page(sample_gpkg: str, chips_vrt: str, out_html: str,
               png_px: int = 560) -> None:
    """Render every sampled polygon into the self-contained rating page."""
    gdf = gpd.read_file(sample_gpkg)
    items = []
    with rasterio.open(chips_vrt) as src:
        for _, r in gdf.iterrows():
            tight, wide = crop_bounds(r.geometry.bounds)
            items.append({
                "id": int(r["rts_id"]), "cls": str(r["conf_class"]),
                "a": float(r["area_m2"]),
                "t": _data_uri(src, [r.geometry], tight, png_px),
                "w": _data_uri(src, [r.geometry], wide, png_px),
            })
    html = _HTML.replace("__ITEMS__", json.dumps(items))
    Path(out_html).write_text(html)
    logger.info("wrote %s (%d polygons, %.1f MB)", out_html, len(items),
                Path(out_html).stat().st_size / 1e6)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", required=True)
    p.add_argument("--chips-vrt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--png-px", type=int, default=560)
    args = p.parse_args()
    setup_logging()
    build_page(args.sample, args.chips_vrt, args.out, args.png_px)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
