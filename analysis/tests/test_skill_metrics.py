import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_metrics import swath_from_track, fractions_skill_score


def test_swath_from_track_marks_points_within_radius():
    # 1-degree grid around a single-fix track at (20, -85).
    lons = np.linspace(-90.0, -80.0, 11)
    lats = np.linspace(15.0, 25.0, 11)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    track = [(datetime(2024, 9, 24, 0), 20.0, -85.0),
             (datetime(2024, 9, 24, 6), 20.0, -85.0)]
    swath = swath_from_track(track, grid_lat, grid_lon, 200.0,
                             datetime(2024, 9, 24, 0), 6)
    # Center cell (20, -85) is inside; a far corner (15, -90) ~ 780 km is outside.
    ci = np.argmin(np.abs(lats - 20.0))
    cj = np.argmin(np.abs(lons - (-85.0)))
    assert swath[ci, cj]
    assert not swath[0, 0]


def test_fss_identical_fields_is_one():
    f = np.zeros((10, 10)); f[3:6, 3:6] = 20.0
    mask = np.ones((10, 10), dtype=bool)
    assert abs(fractions_skill_score(f, f.copy(), 5.0, 1, mask) - 1.0) < 1e-12


def test_fss_disjoint_events_is_zero_at_scale1():
    f = np.zeros((10, 10)); f[1, 1] = 20.0
    o = np.zeros((10, 10)); o[8, 8] = 20.0
    mask = np.ones((10, 10), dtype=bool)
    # No overlap at scale 1 -> MSE == ref -> FSS == 0.
    assert abs(fractions_skill_score(f, o, 5.0, 1, mask) - 0.0) < 1e-12


def test_fss_no_events_is_nan():
    f = np.zeros((10, 10)); o = np.zeros((10, 10))
    mask = np.ones((10, 10), dtype=bool)
    assert np.isnan(fractions_skill_score(f, o, 5.0, 1, mask))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
