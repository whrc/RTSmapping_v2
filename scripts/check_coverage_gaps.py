"""Are the white gaps real missing quads or a binning artifact? Quads have integer grid
coords (quad_id = 'x-y'), so analyse coverage in NATIVE grid space (no reprojection
aliasing): fill fraction + interior holes (empty cell with >=3 filled orthogonal
neighbours -> a real missing quad surrounded by coverage, vs ocean = large empty)."""
import numpy as np, pandas as pd
from scipy import ndimage

q = pd.read_csv("/outputs/inference/quad_index_2025q3.csv", usecols=["x", "y"])
xs, ys = q.x.to_numpy(), q.y.to_numpy()
x0, y0 = xs.min(), ys.min()
nx, ny = xs.max() - x0 + 1, ys.max() - y0 + 1
grid = np.zeros((nx, ny), dtype=bool)
grid[xs - x0, ys - y0] = True
total_cells = nx * ny
filled = grid.sum()
print(f"quad grid bbox: {nx} x {ny} = {total_cells:,} cells; filled (quads) = {filled:,} "
      f"({100*filled/total_cells:.1f}% of bbox)")

# interior holes: empty cells whose 4-neighbours are filled on >=3 sides
empty = ~grid
nbr = (np.roll(grid, 1, 0).astype(int) + np.roll(grid, -1, 0) +
       np.roll(grid, 1, 1) + np.roll(grid, -1, 1))
interior_holes = int((empty & (nbr >= 3)).sum())
fully_enclosed = int((empty & (nbr == 4)).sum())
print(f"empty cells with >=3 filled neighbours (candidate real missing quads): {interior_holes:,}")
print(f"fully-enclosed single-cell holes (all 4 neighbours filled): {fully_enclosed:,}")
# size of the largest empty blob (ocean) for context
lbl, n = ndimage.label(empty)
sizes = np.bincount(lbl.ravel())[1:]
print(f"empty blobs: {n:,}; largest empty blob = {sizes.max():,} cells (ocean/outside)")
print(f"so 'gaps over land' (enclosed holes) are {fully_enclosed:,} of {filled:,} quads "
      f"= {100*fully_enclosed/filled:.3f}%")
