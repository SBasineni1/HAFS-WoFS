"""Local unit tests for track and intensity verification helpers.

Run directly:   python3 analysis/tests/test_track_skill.py
Or via pytest:  pytest analysis/tests/test_track_skill.py -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FIX = Path(__file__).resolve().parent / "fixtures"

import numpy as np
import pytest
from best_track import parse_bdeck_full
from hafs_case import parse_atcfunix, parse_atcfunix_fixes
from track_skill import (
    correlation_annotation, cycle_track_summary, landfall_metrics,
    mean_displacement_deg, score_shifted, shift_field_cells, track_error_rows,
)


def test_parse_atcfunix_full_fixes_and_wrapper_equivalence():
    name, init_dt, fixes = parse_atcfunix_fixes(
        FIX / "helene_realistic.atcfunix")
    assert name == "Helene"
    assert init_dt == datetime(2024, 9, 24, 0)
    assert len(fixes) == 5
    assert fixes[1] == (datetime(2024, 9, 24, 6), 17.8, -83.5,
                        70.0, 980.0)
    assert len([fix for fix in fixes
                if fix[0] == datetime(2024, 9, 24, 0)]) == 1
    old_name, old_init, track = parse_atcfunix(
        FIX / "helene_realistic.atcfunix")
    assert (old_name, old_init) == (name, init_dt)
    assert track == [(t, lat, lon) for t, lat, lon, _, _ in fixes]


def test_parse_bdeck_full_reads_position_and_intensity():
    fixes = parse_bdeck_full(FIX / "bal092024_sample.dat")
    assert len(fixes) == 4
    assert fixes[0] == {
        "t": datetime(2024, 9, 24, 0),
        "lat": 16.8,
        "lon": -83.2,
        "vmax_kt": 35.0,
        "mslp_hpa": 1004.0,
        "rmw_km": None,
    }


def _eastward_truth(start):
    return [
        {"t": start, "lat": 20.0, "lon": 0.0, "vmax_kt": 50.0,
         "mslp_hpa": 990.0, "rmw_km": None},
        {"t": start + timedelta(hours=6), "lat": 20.0, "lon": 1.0,
         "vmax_kt": 60.0, "mslp_hpa": 980.0, "rmw_km": None},
        {"t": start + timedelta(hours=12), "lat": 20.0, "lon": 2.0,
         "vmax_kt": 70.0, "mslp_hpa": 970.0, "rmw_km": None},
    ]


def _forecast(start, lat_offset=0.0, lon_offset=0.0):
    return [
        (start, 20.0 + lat_offset, 0.0 + lon_offset, 50.0, 990.0),
        (start + timedelta(hours=6), 20.0 + lat_offset,
         1.0 + lon_offset, 60.0, 980.0),
        (start + timedelta(hours=12), 20.0 + lat_offset,
         2.0 + lon_offset, 70.0, 970.0),
    ]


def test_track_error_pure_along_due_east():
    start = datetime(2024, 1, 1)
    rows = track_error_rows(_forecast(start, lon_offset=0.25),
                            _eastward_truth(start),
                            start + timedelta(hours=6),
                            start + timedelta(hours=6), 6)
    assert rows[0]["along_km"] > 0
    assert abs(rows[0]["cross_km"]) < 1e-9


def test_track_error_cross_sign_right_of_motion_positive():
    start = datetime(2024, 1, 1)
    truth = _eastward_truth(start)
    valid = start + timedelta(hours=6)
    right = track_error_rows(_forecast(start, lat_offset=-0.25), truth,
                             valid, valid, 6)[0]
    left = track_error_rows(_forecast(start, lat_offset=0.25), truth,
                            valid, valid, 6)[0]
    assert right["cross_km"] > 0
    assert left["cross_km"] < 0
    assert abs(right["along_km"]) < 1e-9
    assert abs(left["along_km"]) < 1e-9


def test_landfall_metrics_known_two_hour_late_offset():
    start = datetime(2024, 1, 1)
    truth = _eastward_truth(start)
    landfall = start + timedelta(hours=6)
    forecast = [
        (start, 20.0, -1.0 / 3.0, 50.0, 990.0),
        (start + timedelta(hours=12), 20.0, 5.0 / 3.0, 70.0, 970.0),
    ]
    metrics = landfall_metrics(forecast, truth, landfall)
    assert metrics["timing_err_h"] == 2.0
    assert metrics["closest_approach_km"] < 1e-9
    assert metrics["pos_err_km"] > 0


def test_cycle_track_summary_ignores_all_none_vmax():
    rows = [{
        "pos_err_km": 10.0, "along_km": 5.0, "cross_km": None,
        "vmax_err_kt": None, "mslp_err_hpa": -2.0,
        "dlat_deg": 0.1, "dlon_deg": -0.1, "_best_lat": 30.0,
    }]
    summary = cycle_track_summary(rows, None)
    assert np.isnan(summary["vmax_bias_kt"])
    assert summary["mslp_bias_hpa"] == -2.0
    assert summary["mean_displacement_km"] > 0


def test_mean_displacement_deg_uses_finite_pairs_by_component():
    rows = [
        {"dlat_deg": 1.0, "dlon_deg": -2.0},
        {"dlat_deg": np.nan, "dlon_deg": -4.0},
        {"dlat_deg": 3.0, "dlon_deg": None},
    ]
    assert mean_displacement_deg(rows) == (2.0, -3.0)
    empty = mean_displacement_deg([])
    assert np.isnan(empty[0]) and np.isnan(empty[1])


def test_shift_field_cells_delta_moves_both_signs():
    grid_res = 0.1
    field = np.zeros((11, 13))
    field[5, 6] = 1.0
    positive = shift_field_cells(
        field, 2 * grid_res, 3 * grid_res, grid_res)
    negative = shift_field_cells(
        field, -2 * grid_res, -3 * grid_res, grid_res)
    assert np.unravel_index(np.argmax(positive), positive.shape) == (7, 9)
    assert np.unravel_index(np.argmax(negative), negative.shape) == (3, 3)
    assert positive[7, 9] == 1.0
    assert negative[3, 3] == 1.0


def test_score_shifted_round_trip_recovers_perfect_ets():
    from ets_score import contingency_scores

    obs = np.zeros((15, 15))
    obs[5:9, 5:9] = 10.0
    grid_res = 0.1
    forecast = shift_field_cells(obs, 2 * grid_res, -3 * grid_res, grid_res)
    swath = np.ones(obs.shape, dtype=bool)
    unshifted = contingency_scores(forecast.ravel(), obs.ravel(), 5.0)["ets"]
    shifted = score_shifted(
        forecast, obs, swath, 5.0, 5.0, 1,
        -2 * grid_res, 3 * grid_res, grid_res)
    assert unshifted < 1.0
    assert shifted["ets_shifted"] == pytest.approx(1.0, abs=1e-10)


def test_score_shifted_nan_displacement_returns_all_nan():
    field = np.ones((3, 3))
    scores = score_shifted(field, field, np.ones_like(field, dtype=bool),
                           1.0, 1.0, 1, np.nan, 0.0, 0.1)
    assert set(scores) == {"ets_shifted", "fss_shifted", "rmse_shifted",
                           "pattern_r_shifted"}
    assert all(np.isnan(value) for value in scores.values())


def test_correlation_annotation_handles_fewer_than_three_pairs():
    assert correlation_annotation([1.0, 2.0], [2.0, np.nan]) == "n<3"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
