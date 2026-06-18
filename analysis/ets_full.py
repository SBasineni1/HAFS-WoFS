"""
ETS for HAFS-A QPF (parent domain + moving 2-km nest) verified against
MRMS QPE and NCEP Stage IV QPE over the Hurricane Helene rainfall swath.

Produces one combined ETS-vs-threshold figure (4 curves: parent/nest x
MRMS/StageIV) and one combined CSV. Reuses all GRIB2/MRMS/Stage IV plumbing
from qpf_full_run.py and parent_qpf.py, and the contingency math + MRMS
plumbing from ets_score.py. The existing ets_score.py is left untouched.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/ets_full.py
"""

import sys
import csv
from pathlib import Path

# Make sibling analysis modules importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def regrid_2d_to_fixed(src_lat, src_lon, data, grid_lat, grid_lon):
    """Interpolate a curvilinear/rectilinear source field onto the fixed mesh.

    src_lat/src_lon may be 1-D axes or 2-D meshes; data is shaped like the
    2-D source mesh. Uses linear griddata; points outside the source hull
    come back NaN (no extrapolation).
    """
    src_lat = np.asarray(src_lat, dtype=float)
    src_lon = np.asarray(src_lon, dtype=float)
    if src_lat.ndim == 1 and src_lon.ndim == 1:
        src_lon, src_lat = np.meshgrid(src_lon, src_lat)
    pts = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    vals = np.asarray(data, dtype=float).ravel()
    finite = np.isfinite(vals)
    out = griddata(
        pts[finite], vals[finite],
        (grid_lat, grid_lon), method="linear",
    )
    return out


def score_pair(fcst_grid, obs_grid, swath, thresholds, contingency_fn):
    """Score one forecast/observation pair over the swath's valid points.

    Valid points are swath & finite(obs) & finite(fcst); kept values are
    zero-filled before thresholding. Returns (rows, n_valid).
    """
    valid = swath & np.isfinite(obs_grid) & np.isfinite(fcst_grid)
    n_valid = int(np.sum(valid))
    fcst = np.nan_to_num(fcst_grid[valid], nan=0.0)
    obs = np.nan_to_num(obs_grid[valid], nan=0.0)
    rows = [contingency_fn(fcst, obs, thr) for thr in thresholds]
    return rows, n_valid
