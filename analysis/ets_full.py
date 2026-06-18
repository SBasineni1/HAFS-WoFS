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

from qpf_full_run import (
    HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER, FIXED_DOMAIN, GRID_RES,
    TC_MASK_RADIUS_KM, INIT_DT, OUT_DIR, MRMS_CACHE_DIR,
    discover_files, hafs_event_total,
)
from ets_score import (
    THRESHOLDS_MM, contingency_scores, build_mrms_total, tc_swath_mask,
)
from parent_qpf import (
    default_parent_path, read_hafs_tp_records, pick_cumulative_record,
    stage4_total, STAGE4_CACHE_DIR,
)

OUT_PNG = OUT_DIR.parent / "ets_full_helene.png"
OUT_CSV = OUT_DIR.parent / "ets_full_helene.csv"


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


def build_fixed_grid():
    """Fixed lat/lon verification mesh (same as ets_score.main)."""
    lat_min, lat_max, lon_min, lon_max = FIXED_DOMAIN
    fixed_lons = np.arange(lon_min, lon_max + GRID_RES, GRID_RES)
    fixed_lats = np.arange(lat_min, lat_max + GRID_RES, GRID_RES)
    grid_lon, grid_lat = np.meshgrid(fixed_lons, fixed_lats)
    return grid_lat, grid_lon


def hafs_parent_total(grid_lat, grid_lon):
    """HAFS-A parent cumulative APCP regridded onto the fixed verification mesh.

    Reuses parent_qpf's discovery + cumulative-record selection, then maps the
    parent grid onto the fixed mesh via regrid_2d_to_fixed.
    """
    path = default_parent_path()
    if path is None or not path.exists():
        raise RuntimeError("No parent.atm file found for the configured run.")
    records = read_hafs_tp_records(path)
    if not records:
        raise RuntimeError(f"No 'tp' (APCP) records in parent file {path}.")
    rec = pick_cumulative_record(records)
    print(f"  parent 0->{rec['end_step']}h, grid {rec['lats'].shape}, "
          f"max {np.nanmax(rec['data']):.0f} mm")
    return regrid_2d_to_fixed(rec["lats"], rec["lons"], rec["data"],
                              grid_lat, grid_lon)


def stage4_on_fixed(max_fhour, grid_lat, grid_lon):
    """Stage IV touched-days total (parent_qpf.stage4_total) on the fixed mesh.

    stage4_total masks its output to parent_qpf's 750 km display swath; for
    verification we re-derive the field UNMASKED is not exposed, so we accept
    that mask — it is wider than the 500 km verification swath, so the tighter
    tc_swath_mask applied later still governs the scored footprint. Stage IV is
    CONUS-only, so ocean points regrid to NaN and drop out automatically.
    """
    s4_lat, s4_lon, s4_total, s4_label = stage4_total(
        INIT_DT, max_fhour, STAGE4_CACHE_DIR)
    if s4_total is None:
        return None, "unavailable"
    grid = regrid_2d_to_fixed(s4_lat, s4_lon, s4_total, grid_lat, grid_lon)
    return grid, s4_label
