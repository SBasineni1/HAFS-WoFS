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


def _toy_contingency(fcst, obs, threshold):
    fy = fcst >= threshold
    oy = obs >= threshold
    a = int(np.sum(fy & oy))
    b = int(np.sum(fy & ~oy))
    c = int(np.sum(~fy & oy))
    d = int(np.sum(~fy & ~oy))
    n = a + b + c + d
    a_ref = (a + b) * (a + c) / n if n else 0.0
    denom = (a + b + c) - a_ref
    ets = (a - a_ref) / denom if denom else float("nan")
    return dict(threshold=threshold, a=a, b=b, c=c, d=d, ets=ets,
                bias=float("nan"), pod=float("nan"),
                far=float("nan"), csi=float("nan"))


def test_score_pair_counts_only_valid_points():
    # 3x3 grids. swath excludes the last column; obs has one NaN inside swath.
    fcst = np.array([[10.0, 0.0, 0.0],
                     [10.0, 10.0, 0.0],
                     [0.0, 0.0, 0.0]])
    obs = np.array([[10.0, 0.0, 99.0],
                    [0.0, np.nan, 99.0],
                    [0.0, 0.0, 99.0]])
    swath = np.array([[True, True, False],
                      [True, True, False],
                      [True, True, False]], dtype=bool)
    # Valid = swath & isfinite(obs) & isfinite(fcst):
    #   row0: (T,T) -> 2 ; row1: (T, NaN->drop) -> 1 ; row2: (T,T) -> 2  => 5
    rows, n_valid = score_pair(fcst, obs, swath, [5.0], _toy_contingency)

    assert n_valid == 5
    r = rows[0]
    # At thr=5 over the 5 valid pts: fcst>=5 at (0,0) and (1,0); obs>=5 at (0,0).
    # hit a=1 (0,0); false alarm b=1 (1,0); miss c=0; correct-neg d=3.
    assert (r["a"], r["b"], r["c"], r["d"]) == (1, 1, 0, 3)


def test_score_pair_one_row_per_threshold():
    fcst = np.zeros((4, 4))
    obs = np.zeros((4, 4))
    swath = np.ones((4, 4), dtype=bool)
    rows, n_valid = score_pair(fcst, obs, swath, [1.0, 5.0, 10.0],
                               _toy_contingency)
    assert len(rows) == 3
    assert [r["threshold"] for r in rows] == [1.0, 5.0, 10.0]
    assert n_valid == 16


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
