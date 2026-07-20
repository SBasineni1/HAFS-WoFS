"""Unit tests for hafs_common grid decoding (no Hercules data needed).

Run directly:   python3 analysis/tests/test_hafs_common.py
Or via pytest:  pytest analysis/tests/test_hafs_common.py -v
"""
import sys
from pathlib import Path

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from hafs_common import grid_latlon


def test_grid_latlon_regional_unchanged():
    """A regional parent grid (lon0 < lon1, no seam) decodes as before."""
    lats, lons = grid_latlon(15.0, 260.0, 45.0, 340.0, nj=601, ni=1601)
    assert lats.shape == lons.shape == (601, 1601)
    assert np.isclose(lats.min(), 15.0) and np.isclose(lats.max(), 45.0)
    # 260 E -> -100, 340 E -> -20, ascending, no NaN.
    assert np.isclose(lons.min(), -100.0)
    assert np.isclose(lons.max(), -20.0)
    assert np.all(np.diff(lons[0]) > 0)


def test_grid_latlon_multistorm_seam_crossing():
    """The HAFS-M global parent runs 210 E -> 10 E across the 0/360 seam.

    linspace(lon0, lon1) would ramp 210 -> 10 (descending, wrong hemisphere),
    dropping every Helene-domain point. The seam-aware decode must keep them.
    """
    lats, lons = grid_latlon(-25.0, 210.0, 65.0, 10.000001, nj=1501, ni=2667)
    box = (lats >= 15) & (lats <= 42) & (lons >= -100) & (lons <= -60)
    assert box.sum() > 100_000
    # 210 E -> -150, wrapping through 0 up to +10, strictly increasing per row.
    assert np.isclose(lons[0, 0], -150.0)
    assert np.isclose(lons[0, -1], 10.0, atol=1e-3)
    assert np.all(np.diff(lons[0]) > 0)


if __name__ == "__main__":
    test_grid_latlon_regional_unchanged()
    test_grid_latlon_multistorm_seam_crossing()
    print("ok")
