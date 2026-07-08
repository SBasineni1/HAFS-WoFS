"""Spatial verification helpers: shared-track swath + Fractions Skill Score."""

import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hafs_common import haversine_km
from hafs_case import position_on_track


def swath_from_track(track, grid_lat, grid_lon, radius_km, init_dt, max_fhour):
    """Boolean mask: union of radius_km circles along the track, hourly over
    [init_dt, init_dt + max_fhour h]. Warns once if the track ends early."""
    swath = np.zeros(grid_lat.shape, dtype=bool)
    last_t = track[-1][0]
    warned = False
    for h in range(0, max_fhour + 1):
        t = init_dt + timedelta(hours=h)
        if t > last_t and not warned:
            print(f"  warning: best track ends {last_t:%Y-%m-%d %HZ} before "
                  f"forecast hour {h} ({t:%Y-%m-%d %HZ}); swath clamps to last fix")
            warned = True
        tlat, tlon = position_on_track(track, t)
        swath |= haversine_km(tlat, tlon, grid_lat, grid_lon) <= radius_km
    return swath


def fractions_skill_score(fcst, obs, threshold, scale, mask):
    """FSS at one threshold and neighborhood size (scale, in grid cells).

    FSS = 1 - MSE(Mf, Of) / (mean(Mf^2) + mean(Of^2)) over mask points, where
    Mf/Of are neighborhood fractions of the binarized fields. NaN if there are
    no events at this threshold anywhere (denominator 0).
    """
    fb = (fcst >= threshold).astype(float)
    ob = (obs >= threshold).astype(float)
    ff = uniform_filter(fb, size=scale, mode="constant", cval=0.0)
    of = uniform_filter(ob, size=scale, mode="constant", cval=0.0)
    if not mask.any():
        return np.nan
    mse = np.mean((ff[mask] - of[mask]) ** 2)
    ref = np.mean(ff[mask] ** 2) + np.mean(of[mask] ** 2)
    return 1.0 - mse / ref if ref > 0 else np.nan


def continuous_scores(fcst, obs):
    """Continuous verification scores over 1-D arrays of valid points.

    The caller selects the points (same footprint as the categorical
    scores: swath & finite obs & finite fcst). bias = mean(fcst - obs),
    so positive means over-forecast. r is Pearson correlation, NaN when
    n < 2 or either field has zero variance; every score is NaN at n = 0.
    """
    fcst = np.asarray(fcst, dtype=float)
    obs = np.asarray(obs, dtype=float)
    n = int(fcst.size)
    if n == 0:
        return dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan, r=np.nan)
    err = fcst - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    if n < 2 or np.std(fcst) == 0.0 or np.std(obs) == 0.0:
        r = np.nan
    else:
        r = float(np.corrcoef(fcst, obs)[0, 1])
    return dict(n=n, rmse=rmse, mae=mae, bias=bias, r=r)
