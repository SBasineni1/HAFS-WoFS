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
