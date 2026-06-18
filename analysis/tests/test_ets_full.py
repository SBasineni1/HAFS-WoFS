"""Local unit tests for ets_full pure helpers (no Hercules data needed).

Run directly:   python3 analysis/tests/test_ets_full.py
Or via pytest:  pytest analysis/tests/test_ets_full.py -v
"""
import sys
from pathlib import Path

import numpy as np

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ets_full import regrid_2d_to_fixed, score_pair


def test_regrid_2d_identity_on_matching_grid():
    # Source is a linear ramp z = lat + lon on a fine 2-D grid.
    s_lat = np.linspace(20.0, 40.0, 41)
    s_lon = np.linspace(-100.0, -70.0, 61)
    slon2d, slat2d = np.meshgrid(s_lon, s_lat)
    data = slat2d + slon2d
    # Fixed grid is a coarser mesh strictly inside the source extent.
    g_lon = np.linspace(-95.0, -75.0, 11)
    g_lat = np.linspace(25.0, 35.0, 9)
    grid_lon, grid_lat = np.meshgrid(g_lon, g_lat)

    out = regrid_2d_to_fixed(slat2d, slon2d, data, grid_lat, grid_lon)

    assert out.shape == grid_lat.shape
    expected = grid_lat + grid_lon
    assert np.allclose(out, expected, atol=1e-6)


def test_regrid_2d_accepts_1d_axes():
    # Passing 1-D lat/lon axes must work the same as 2-D meshes.
    s_lat = np.linspace(20.0, 40.0, 41)
    s_lon = np.linspace(-100.0, -70.0, 61)
    slon2d, slat2d = np.meshgrid(s_lon, s_lat)
    data = slat2d + slon2d
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(-95.0, -75.0, 11), np.linspace(25.0, 35.0, 9)
    )

    out = regrid_2d_to_fixed(s_lat, s_lon, data, grid_lat, grid_lon)
    assert np.allclose(out, grid_lat + grid_lon, atol=1e-6)


def test_regrid_2d_nan_outside_source():
    # Points outside the source hull come back NaN, not extrapolated.
    s_lat = np.linspace(30.0, 35.0, 11)
    s_lon = np.linspace(-90.0, -85.0, 11)
    slon2d, slat2d = np.meshgrid(s_lon, s_lat)
    data = np.ones_like(slat2d)
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(-95.0, -80.0, 7), np.linspace(25.0, 40.0, 7)
    )

    out = regrid_2d_to_fixed(slat2d, slon2d, data, grid_lat, grid_lon)
    # Corner (25, -95) is well outside [30,35]x[-90,-85] -> NaN.
    assert np.isnan(out[0, 0])
    # Center (32.5, -87.5) is inside -> finite ~1.0.
    assert np.isfinite(out[3, 3])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
