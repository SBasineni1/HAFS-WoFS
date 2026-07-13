import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper import (
    aggregate_categorical, compute_paper_suite, identify_objects, interpolate_fix,
    shift_to_best, storm_relative_field,
)
from paper_case import PaperSuiteCase, load_paper_config


def test_load_paper_storm_and_suite():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        storm = root / "alpha_paper.yaml"
        storm.write_text("""\
storm_name: Alpha
models:
  HAFS-A: /runs/alpha/HFSA
  HAFS-B: /runs/alpha/HFSB
best_track: /tracks/bal012025.dat
domain: [10, 30, -90, -60]
inits: [2025010100, 2025010112]
lead_hours: [6, 12, 18]
""")
        cfg = load_paper_config(storm)
        assert cfg.storm_name == "Alpha"
        assert [m.name for m in cfg.models] == ["HAFS-A", "HAFS-B"]
        assert cfg.forecast_domain == "parent"
        assert cfg.accumulation_hours == 6
        assert cfg.inits == ["2025010100", "2025010112"]
        suite = root / "suite.yaml"
        suite.write_text(f"storms: [{storm}]\nlabel: Test season\n")
        loaded = load_paper_config(suite)
        assert isinstance(loaded, PaperSuiteCase)
        assert loaded.storm_paths == [storm]


def test_shift_to_best_moves_peak_north_and_east():
    field = np.zeros((7, 7), dtype=float)
    field[2, 2] = 10
    shifted = shift_to_best(field, (0.0, 0.0), (1.0, 2.0), 1.0)
    assert np.nanargmax(shifted) == np.ravel_multi_index((3, 4), shifted.shape)


def test_interpolate_fix_uses_rmw_and_fallback():
    fixes = [(datetime(2025, 1, 1, 0), 10.0, -60.0, 40.0),
             (datetime(2025, 1, 1, 6), 16.0, -66.0, 100.0)]
    lat, lon, rmw, fallback = interpolate_fix(
        fixes, datetime(2025, 1, 1, 3), 50.0)
    assert (lat, lon, rmw, fallback) == (13.0, -63.0, 70.0, False)
    no_rmw = [(t, la, lo, None) for t, la, lo, _ in fixes]
    assert interpolate_fix(no_rmw, datetime(2025, 1, 1, 3), 55.0)[2:] == (55.0, True)


def test_storm_relative_field_samples_center():
    axis = np.arange(-2.0, 2.1, 0.5)
    lon, lat = np.meshgrid(axis, axis)
    field = lat + lon
    x, y, rel = storm_relative_field(field, lat, lon, (0.0, 0.0),
                                     rmw_km=55.5, radius_rmw=2, resolution_rmw=1)
    center = rel.shape[0] // 2
    assert np.isclose(rel[center, center], 0.0)
    assert rel.shape == x.shape == y.shape


def test_aggregate_categorical_pools_counts_not_scores():
    rows = [
        {"model": "A", "shift": "raw", "lead_hour": 6, "threshold": 10.0,
         "a": 4, "b": 1, "c": 2, "d": 3},
        {"model": "A", "shift": "raw", "lead_hour": 6, "threshold": 10.0,
         "a": 2, "b": 3, "c": 1, "d": 4},
    ]
    result = aggregate_categorical(rows, bootstrap_replicates=20, random_seed=1)[0]
    assert (result["a"], result["b"], result["c"], result["d"]) == (6, 4, 3, 7)
    assert result["n_events"] == 2
    assert np.isfinite(result["ets_lo"]) and np.isfinite(result["ets_hi"])


def test_identify_objects_filters_small_components():
    field = np.zeros((10, 10))
    field[1, 1] = 20
    field[5:8, 5:8] = 20
    labels, objects = identify_objects(field, threshold=10, smooth_cells=0,
                                       min_pixels=4)
    assert len(objects) == 1
    assert objects[0]["pixels"] == 9
    assert labels[1, 1] == 0 and labels[6, 6] == 1


def test_suite_rejects_different_model_samples_before_loading_data():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        common = "best_track: /best.dat\ndomain: [0, 1, 0, 1]\nlead_hours: [6]\n"
        one = root / "one.yaml"
        two = root / "two.yaml"
        one.write_text("models: {A: /a, B: /b}\n" + common)
        two.write_text("models: {A: /a}\n" + common)
        suite = PaperSuiteCase(root / "suite.yaml", "Suite", [one, two],
                               root / "out", 0, 42)
        try:
            compute_paper_suite(suite)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "same model names" in str(exc)
