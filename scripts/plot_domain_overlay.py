"""Four-panel comparison: permafrost region, Planet 2025 coverage, inference domain,
tile-count region. Small multiples (not a single overlay) because coverage ≈ domain ≈
tile-region are nearly coincident and hide each other when stacked. Each panel shows its
layer solid + the permafrost outline as a shared reference. EPSG:3413 view; areas in
EPSG:6931 (equal-area)."""
import numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

DOM = "/app/domain"
QUAD = "/outputs/inference/quad_index_2025q3.csv"
TILES = "/outputs/inference/tiles_2025q3_domain_full.csv"
OUT = "/outputs/inference/domain_overlay.png"
T = Transformer.from_crs(3857, 3413, always_xy=True)

perm = gpd.read_file(f"{DOM}/circumpolar_domain.geojson").to_crs(3413)
infd = gpd.read_file(f"{DOM}/circumpolar_south_domain.geojson").to_crs(3413)
perm_km2 = perm.to_crs(6931).area.sum() / 1e6
infd_km2 = infd.to_crs(6931).area.sum() / 1e6

# quad centroids -> 3413
q = pd.read_csv(QUAD, usecols=["minx", "miny", "maxx", "maxy"])
qx, qy = T.transform((q.minx + q.maxx).to_numpy() / 2, (q.miny + q.maxy).to_numpy() / 2)
# tile centroids (subsample 1/30) -> 3413
cx, cy = [], []
for ch in pd.read_csv(TILES, usecols=["minx", "miny", "maxx", "maxy"], chunksize=2_000_000):
    s = ch.iloc[::30]
    x, y = T.transform((s.minx + s.maxx).to_numpy() / 2, (s.miny + s.maxy).to_numpy() / 2)
    cx.append(x); cy.append(y)
cx = np.concatenate(cx); cy = np.concatenate(cy)

# common grid over permafrost extent -> binary coverage footprints
x0, y0, x1, y1 = perm.total_bounds
pad = 1e5
xe = np.linspace(x0 - pad, x1 + pad, 600); ye = np.linspace(y0 - pad, y1 + pad, 600)
from scipy import ndimage
cov, _, _ = np.histogram2d(qx, qy, bins=[xe, ye])
til, _, _ = np.histogram2d(cx, cy, bins=[xe, ye])
extent = [xe[0], xe[-1], ye[0], ye[-1]]
# Close 1-cell moiré holes from reprojecting the regular 3857 quad lattice onto an
# axis-aligned 3413 grid (artifact, not real gaps — verified: real missing quads ~0.05%).
cov_bool = ndimage.binary_closing(cov.T > 0, structure=np.ones((3, 3)))
cov_m = np.ma.masked_where(~cov_bool, cov_bool)
til_m = np.ma.masked_where(~ndimage.binary_dilation(til.T > 0), til.T)  # density

def base(ax, title):
    perm.boundary.plot(ax=ax, edgecolor="#94a3b8", linewidth=0.5, zorder=1)
    for lat in (60, 74):
        lon = np.linspace(-180, 180, 720)
        tx, ty = Transformer.from_crs(4326, 3413, always_xy=True).transform(lon, np.full_like(lon, lat))
        ax.plot(tx, ty, ls=":", lw=0.6, color="#64748b", zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)

fig, axs = plt.subplots(2, 2, figsize=(13, 13))
# (1) permafrost
base(axs[0, 0], f"① Permafrost region\nArctic-boreal ∩ permafrost · {perm_km2/1e6:.2f}M km²")
perm.plot(ax=axs[0, 0], facecolor="#64748b", edgecolor="none", alpha=0.85, zorder=3)
# (2) Planet coverage
base(axs[0, 1], f"② PlanetScope 2025 coverage\n{len(qx):,} quads (binary footprint)")
axs[0, 1].imshow(cov_m, extent=extent, origin="lower", cmap=matplotlib.colors.ListedColormap(["#16a34a"]), zorder=3)
# (3) inference domain
base(axs[1, 0], f"③ Inference domain (circumpolar_south)\npermafrost ∩ Planet · {infd_km2/1e6:.2f}M km²")
infd.plot(ax=axs[1, 0], facecolor="#1d4ed8", edgecolor="none", alpha=0.8, zorder=3)
# (4) tile-count region (density)
base(axs[1, 1], "④ Tile-count region (= domain ∩ coverage)\n41,567,572 tiles · stride 344")
im = axs[1, 1].imshow(til_m, extent=extent, origin="lower", cmap="magma",
                      norm=matplotlib.colors.LogNorm(), zorder=3)
fig.colorbar(im, ax=axs[1, 1], shrink=0.5, pad=0.02, label="tiles/cell (1/30 sample, log)")

fig.suptitle("RTS v2 — inference domain components (EPSG:3413; 60°N/74°N dotted)", fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"perm={perm_km2/1e6:.2f}M km²  domain={infd_km2/1e6:.2f}M km²  quads={len(qx):,}  wrote {OUT}")
