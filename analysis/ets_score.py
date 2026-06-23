"""
ETS helper functions for HAFS QPF verification against MRMS QPE.

Provides the contingency math and the MRMS/swath plumbing that ets_full.py
composes into the combined parent+nest ETS figure:
  * regrid_mrms_to_fixed — bilinear MRMS → fixed mesh
  * build_mrms_total     — accumulate MRMS 1H QPE over the window, then regrid
  * tc_swath_mask        — boolean mask of points within mask_radius_km of the
                           track at any hour (the verification footprint)
  * contingency_scores   — 2x2 table + ETS / bias / POD / FAR / CSI at one threshold

    ETS = (a - a_ref) / (a + b + c - a_ref),   a_ref = (a+b)(a+c)/n

ETS ranges from -1/3 to 1; 0 = no skill over random, 1 = perfect.

Not a runnable script — use run.py (the full parent+nest ETS lives in ets_full.py).
"""

import sys
from pathlib import Path
from datetime import timedelta

# Make the sibling module importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from hafs_common import haversine_km, load_mrms_hour

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Rainfall thresholds (mm) at which to evaluate skill.
# Kept as a module constant so ets_full.py can import it as a fallback default.
THRESHOLDS_MM = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250]


def regrid_mrms_to_fixed(mlat, mlon, mdata, grid_lat, grid_lon):
    """Bilinear-interpolate native MRMS (regular 1-D lat/lon) onto the fixed mesh."""
    mlat = np.asarray(mlat)
    mlon = np.asarray(mlon, dtype=float)
    # cfgrib returns MRMS longitude in 0–360; the fixed grid (and HAFS) use
    # −180…180, so convert or every query falls outside the interpolator.
    mlon = np.where(mlon > 180, mlon - 360, mlon)
    # RegularGridInterpolator needs strictly ascending axes.
    if mlat[0] > mlat[-1]:
        mlat = mlat[::-1]
        mdata = mdata[::-1, :]
    if mlon[0] > mlon[-1]:
        mlon = mlon[::-1]
        mdata = mdata[:, ::-1]
    interp = RegularGridInterpolator(
        (mlat, mlon), mdata, bounds_error=False, fill_value=np.nan
    )
    pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
    return interp(pts).reshape(grid_lat.shape)


def build_mrms_total(case, max_fhour, grid_lat, grid_lon):
    """Accumulate MRMS 1H QPE over 1..max_fhour on its native grid, then regrid."""
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    mrms_sum = None
    mlat = mlon = None
    for h in range(1, max_fhour + 1):
        t = case.init_dt + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, case.mrms_cache_dir)
            if mrms_sum is None:
                mlat, mlon = lat, lon
                mrms_sum = np.zeros_like(data)
            mrms_sum += data
            if h % 12 == 0 or h == max_fhour:
                print(f"  MRMS h{h:03d}/{max_fhour} ({t.strftime('%Y-%m-%d %HZ')})")
        except Exception as e:
            print(f"  MRMS h{h:03d} unavailable: {e}")
    if mrms_sum is None:
        raise RuntimeError("No MRMS hours could be loaded.")
    return regrid_mrms_to_fixed(mlat, mlon, mrms_sum, grid_lat, grid_lon)


def tc_swath_mask(case, max_fhour, grid_lat, grid_lon):
    """Boolean mask: grid points within case.mask_radius_km of the track at any hour."""
    swath = np.zeros(grid_lat.shape, dtype=bool)
    for h in range(0, max_fhour + 1):
        tlat, tlon = case.position_at(case.init_dt + timedelta(hours=h))
        dist = haversine_km(tlat, tlon, grid_lat, grid_lon)
        swath |= dist <= case.mask_radius_km
    return swath


def contingency_scores(fcst, obs, threshold):
    """2x2 contingency table + skill scores at one threshold over 1-D arrays."""
    fy = fcst >= threshold
    oy = obs >= threshold
    a = int(np.sum(fy & oy))      # hits
    b = int(np.sum(fy & ~oy))     # false alarms
    c = int(np.sum(~fy & oy))     # misses
    d = int(np.sum(~fy & ~oy))    # correct negatives
    n = a + b + c + d
    a_ref = (a + b) * (a + c) / n if n > 0 else 0.0
    denom = (a + b + c) - a_ref
    ets = (a - a_ref) / denom if denom != 0 else np.nan
    bias = (a + b) / (a + c) if (a + c) > 0 else np.nan
    pod = a / (a + c) if (a + c) > 0 else np.nan
    far = b / (a + b) if (a + b) > 0 else np.nan
    csi = a / (a + b + c) if (a + b + c) > 0 else np.nan
    hss_denom = (a + c) * (c + d) + (a + b) * (b + d)
    hss = (2 * (a * d - b * c) / hss_denom) if hss_denom != 0 else np.nan
    return dict(threshold=threshold, a=a, b=b, c=c, d=d,
                ets=ets, bias=bias, pod=pod, far=far, csi=csi, hss=hss)
