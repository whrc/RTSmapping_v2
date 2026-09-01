"""PROTOTYPE — rebuild an EE cell's NDVI from Earth Search COGs and diff against it.

Not part of the pipeline. This is the evidence behind s2_source_evaluation.md and
qa60_gap.md, kept so the numbers in those docs can be re-derived rather than trusted.

    python prototype_earthsearch_diff.py <cell> <year> <off_x> <off_y> <size>
    NOMASK=1 ...   # composite with no cloud mask, to emulate the 2022/2023 QA60 gap

Needs earthengine-free deps only (rasterio, numpy, requests) — runs in rts-dataprep:v1.


Reads onto the EE product's exact grid (EPSG:3857, 10 m) via WarpedVRT, so the
comparison is pixel-for-pixel with no resampling of the EE side.
"""
import os, sys, json, collections
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN","EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS",".tif,.TIF")
os.environ.setdefault("VSI_CACHE","TRUE")
os.environ.setdefault("GDAL_CACHEMAX","512")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY","5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY","2")
import numpy as np, rasterio, requests
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from concurrent.futures import ThreadPoolExecutor

STAC="https://earth-search.aws.element84.com/v1/search"
# SCL classes treated as INVALID. 0 nodata,1 saturated,3 cloud shadow,
# 8 cloud med-prob, 9 cloud high-prob, 10 thin cirrus.
import os as _os
# NOMASK=1 mimics a composite with NO per-pixel cloud mask — which is exactly what
# EE's mask_clouds() degrades to in 2022/2023, where QA60 is an empty band.
SCL_BAD = {0} if _os.environ.get("NOMASK") else {0,1,3,8,9,10}

def search(bbox, year, max_cloud=20):
    feats, nxt = [], None
    body={"collections":["sentinel-2-l2a"],"bbox":list(bbox),
          "datetime":f"{year}-07-01T00:00:00Z/{year}-09-29T23:59:59Z",
          "query":{"eo:cloud_cover":{"lt":max_cloud}},"limit":100}
    while True:
        r=requests.post(STAC,json=body,timeout=90); r.raise_for_status(); d=r.json()
        feats+=d["features"]
        nxt=next((l for l in d.get("links",[]) if l.get("rel")=="next"),None)
        if not nxt: break
        body=nxt.get("body",body); body["limit"]=100
        if len(feats)>400: break
    return feats

def dedupe(feats):
    """One item per (tile, acquisition). ESA Collection-1 republished the archive,
    so 2019-21 carry both an 02.13 original and an 05.00 reprocessing of the same
    overpass; keeping both double-weights that date in the median."""
    best={}
    for f in feats:
        p=f["properties"]; k=(p.get("grid:code"), p["datetime"])
        b=p.get("s2:processing_baseline","00.00")
        if k not in best or b > best[k]["properties"].get("s2:processing_baseline","00.00"):
            best[k]=f
    return list(best.values())

# MEASURED 2026-08-26: sentinel-cogs DN are ALREADY in the pre-baseline-04.00
# convention — Element 84 removed the +1000 BOA offset before staging. Proof: a
# baseline-05.11 tile has p5 = 44 DN, impossible if +1000 were present, and
# DN/10000 = 0.073 matches EE while (DN-1000)/10000 = -0.027 is negative.
# The STAC raster:bands offset of -0.1 describes the ORIGINAL ESA product, not
# the served COG. Applying it (as the reference script does) subtracts a shift
# that is not there, drives reflectance negative, and makes NDVI explode as the
# denominator crosses zero.
APPLY_BOA_OFFSET = False

def offset_dn(item, key):
    """BOA offset in DN to apply to the served COG — zero, see APPLY_BOA_OFFSET."""
    if not APPLY_BOA_OFFSET:
        return 0
    a=item["assets"][key]
    rb=a.get("raster:bands") or a.get("bands")
    if not rb: raise RuntimeError(f"{item['id']}/{key}: no raster:bands — refusing to assume offset 0")
    b=rb[0]; scale=float(b.get("scale",0.0001)); off=float(b.get("offset",0.0))
    return int(round(off/scale)) if scale else 0

def read_on(href, dst_crs, dst_transform, w, h, resampling):
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=dst_crs, transform=dst_transform,
                       width=w, height=h, resampling=resampling) as vrt:
            return vrt.read(1)

def composite(feats, crs, transform, w, h):
    from rasterio.enums import Resampling
    reds, nirs = [], []
    def one(f):
        try:
            red=read_on(f["assets"]["red"]["href"], crs, transform, w, h, Resampling.nearest)
            nir=read_on(f["assets"]["nir"]["href"], crs, transform, w, h, Resampling.nearest)
            scl=read_on(f["assets"]["scl"]["href"], crs, transform, w, h, Resampling.nearest)
        except Exception as e:
            print("   skip", f["id"], e); return None
        ok = ~np.isin(scl, list(SCL_BAD)) & (red>0) & (nir>0)
        if not ok.any(): return None
        o=offset_dn(f,"red"); o2=offset_dn(f,"nir")
        r=np.where(ok,(red.astype(np.int32)+o),0).astype(np.float32)
        n=np.where(ok,(nir.astype(np.int32)+o2),0).astype(np.float32)
        r[~ok]=np.nan; n[~ok]=np.nan
        return r/10000.0, n/10000.0
    with ThreadPoolExecutor(16) as ex:
        for res in ex.map(one, feats):
            if res: reds.append(res[0]); nirs.append(res[1])
    if not reds: return None, None, 0
    R=np.nanmedian(np.stack(reds),0); N=np.nanmedian(np.stack(nirs),0)
    return R, N, len(reds)

def main(cell, year, off_x, off_y, size):
    gs=f"/vsigs/rts-arctic-usw1/S2_RGB/{year}_south/{cell}.tif"
    with rasterio.open(gs) as src:
        # centre the window if the requested offset would fall outside the raster
        if off_x < 0 or off_y < 0 or off_x+size > src.width or off_y+size > src.height:
            off_x = max(0, (src.width  - size)//2)
            off_y = max(0, (src.height - size)//2)
        size = min(size, src.width-off_x, src.height-off_y)
        win=Window(off_x, off_y, size, size)
        tr=src.window_transform(win); crs=src.crs
        ee_red=src.read(1, window=win).astype(np.float32)
        ee_nir=src.read(4, window=win).astype(np.float32)
        bounds=rasterio.windows.bounds(win, src.transform)
    wgs=transform_bounds(crs,"EPSG:4326",*bounds)
    print(f"[{cell} {year}] grid {size}x{size} @({off_x},{off_y}) bbox {[round(v,3) for v in wgs]}")
    feats=search(wgs, year)
    dd=dedupe(feats)
    bl=collections.Counter(f["properties"].get("s2:processing_baseline") for f in dd)
    print(f"   STAC {len(feats)} items -> {len(dd)} after dedupe; baselines {dict(bl)}")
    R,N,n = composite(dd, crs, tr, size, size)
    if R is None: print("   no valid data"); return
    def ndvi(r,n_): 
        d=n_+r
        return np.where(np.abs(d)>1e-6,(n_-r)/np.where(np.abs(d)>1e-6,d,1),np.nan)
    es=ndvi(R,N)
    ee_=ndvi(ee_red,ee_nir)
    m=np.isfinite(es)&np.isfinite(ee_)
    print(f"   scenes used {n}; comparable px {m.sum()/m.size:.1%}")
    if m.sum()<1000: print("   too few"); return
    d=es[m]-ee_[m]
    print(f"   NDVI  EarthSearch mean {es[m].mean():+.4f}   EE mean {ee_[m].mean():+.4f}")
    print(f"   diff  mean {d.mean():+.4f}  median {np.median(d):+.4f}  MAE {np.abs(d).mean():.4f}  p95|d| {np.percentile(np.abs(d),95):.4f}")
    print(f"   corr  {np.corrcoef(es[m],ee_[m])[0,1]:.4f}")
    print(f"RESULT\t{cell}\t{year}\t{n}\t{m.sum()/m.size:.3f}\t{d.mean():+.4f}\t"
          f"{np.abs(d).mean():.4f}\t{np.percentile(np.abs(d),95):.4f}\t"
          f"{np.corrcoef(es[m],ee_[m])[0,1]:.4f}", flush=True)

if __name__=="__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
