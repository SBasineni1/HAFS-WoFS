"""Local unit tests for parent_qpf pure helpers (no Hercules data needed).

Run directly:   python3 analysis/tests/test_parent_qpf.py
Or via pytest:  pytest analysis/tests/test_parent_qpf.py -v
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import parent_qpf
from parent_qpf import swath_masked


class FakeCase:
    """Minimal stand-in for StormCase: a single stationary track point."""
    def __init__(self, lat=30.0, lon=-85.0, radius_km=300.0):
        self.init_dt = datetime(2024, 9, 24, 0)
        self.display_radius_km = radius_km
        self._lat, self._lon = lat, lon

    def position_at(self, _dt):
        return self._lat, self._lon


def _grid():
    lons = np.linspace(-90.0, -80.0, 11)   # ~1 deg spacing
    lats = np.linspace(25.0, 35.0, 11)
    return np.meshgrid(lons, lats)[::-1]    # (lat2d, lon2d)


def test_swath_masked_zeros_outside_radius():
    lat2d, lon2d = _grid()
    field = np.full(lat2d.shape, 50.0)
    case = FakeCase(lat=30.0, lon=-85.0, radius_km=200.0)
    out = swath_masked(field, lat2d, lon2d, case, end_fhour=0)
    # Cell at the track center keeps its value.
    ci = np.argmin(np.abs(lat2d[:, 0] - 30.0))
    cj = np.argmin(np.abs(lon2d[0, :] - (-85.0)))
    assert out[ci, cj] == 50.0
    # A far corner (>200 km away) is zeroed.
    assert out[0, 0] == 0.0
    # Output shape is preserved.
    assert out.shape == field.shape


def test_swath_masked_replaces_nan_with_zero_inside_swath():
    lat2d, lon2d = _grid()
    field = np.full(lat2d.shape, np.nan)
    case = FakeCase(lat=30.0, lon=-85.0, radius_km=2000.0)  # covers whole grid
    out = swath_masked(field, lat2d, lon2d, case, end_fhour=0)
    assert np.all(out == 0.0)          # NaN -> 0 inside the swath
    assert not np.any(np.isnan(out))


def test_compute_nest_field_returns_none_when_no_files(monkeypatch):
    lat2d, lon2d = _grid()

    class C(FakeCase):
        run_dir = Path("/nowhere")
        fhours_filter = None
        def storm_glob(self):
            return "*storm.atm.f*.grb2"

    monkeypatch.setattr(parent_qpf, "discover_files", lambda *a, **k: [])
    out = parent_qpf.compute_nest_field(C(), lat2d, lon2d, end_fhour=0)
    assert out is None


def test_compute_nest_field_masks_total_to_swath(monkeypatch):
    lat2d, lon2d = _grid()

    class C(FakeCase):
        run_dir = Path("/nowhere")
        fhours_filter = None
        def storm_glob(self):
            return "*storm.atm.f*.grb2"

    # Pretend discovery found one file-pair; stub the heavy accumulation.
    monkeypatch.setattr(parent_qpf, "discover_files",
                        lambda *a, **k: [(0, Path("f000"))])
    monkeypatch.setattr(parent_qpf, "hafs_event_total",
                        lambda *a, **k: (np.full(lat2d.shape, 40.0), "amount"))

    case = C(lat=30.0, lon=-85.0, radius_km=200.0)
    out = parent_qpf.compute_nest_field(case, lat2d, lon2d, end_fhour=0)
    assert out is not None
    ci = np.argmin(np.abs(lat2d[:, 0] - 30.0))
    cj = np.argmin(np.abs(lon2d[0, :] - (-85.0)))
    assert out[ci, cj] == 40.0        # kept at center
    assert out[0, 0] == 0.0           # zeroed in far corner


if __name__ == "__main__":
    test_swath_masked_zeros_outside_radius()
    test_swath_masked_replaces_nan_with_zero_inside_swath()
    print("ok (run `pytest` for the monkeypatched nest tests)")
