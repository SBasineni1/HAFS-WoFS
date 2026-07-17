"""Unit tests for precipitation distributions and object verification."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precip_structure import (
    DIST_FIELDS, OBJECT_FIELDS, distribution_stats, largest_object,
    object_comparison, qq_percentiles,
)


def _grid(n=81, grid_res=0.1, center_lat=30.0):
    offsets = (np.arange(n) - n // 2) * grid_res
    lon, lat = np.meshgrid(-80.0 + offsets, center_lat + offsets)
    return lat, lon


def test_distribution_stats_uniform_field_exact_volume_and_percentiles():
    grid_res = 0.1
    lat = np.zeros((2, 3))
    field = np.full((2, 3), 10.0)
    swath = np.array([[True, True, False], [True, False, False]])
    stats = distribution_stats(field, swath, lat, grid_res)
    cell_area = (grid_res * 111.0) ** 2
    assert stats["p50"] == stats["p90"] == stats["p95"] == 10.0
    assert stats["p99"] == stats["max_mm"] == 10.0
    assert stats["volume_km3"] == 3 * 10.0 * cell_area * 1e-6
    assert stats["wet_frac"] == 1.0


def test_qq_identical_fields_is_identity():
    field = np.arange(25, dtype=float).reshape(5, 5)
    fcst_q, obs_q = qq_percentiles(
        field, field.copy(), np.ones_like(field, dtype=bool))
    np.testing.assert_allclose(fcst_q, obs_q)


def test_rotated_ellipse_centroid_offset_angle_and_area_ratio():
    lat, lon = _grid()
    mean_lat = float(np.mean(lat))
    x = (lon + 80.0) * 111.0 * np.cos(np.radians(mean_lat))
    y = (lat - 30.0) * 111.0

    def gaussian(dx, dy, angle_deg, scale=1.0):
        angle = np.radians(angle_deg)
        xx = (x - dx) * np.cos(angle) + (y - dy) * np.sin(angle)
        yy = -(x - dx) * np.sin(angle) + (y - dy) * np.cos(angle)
        return 100.0 * np.exp(-0.5 * ((xx / (35.0 * scale)) ** 2
                                      + (yy / (12.0 * scale)) ** 2))

    obs = gaussian(0.0, 0.0, 30.0)
    fcst = gaussian(22.0, -11.0, 42.0, scale=1.2)
    metrics = object_comparison(
        fcst, obs, np.ones_like(obs, dtype=bool), lat, lon,
        threshold_mm=20.0, smooth_cells=1, min_area_cells=10,
        motion_unit=(1.0, 0.0))
    cell_km = 0.1 * 111.0
    assert abs(metrics["obj_centroid_along_km"] - 22.0) <= cell_km
    assert abs(metrics["obj_centroid_cross_km"] - 11.0) <= cell_km
    assert abs(metrics["obj_angle_diff_deg"] - 12.0) <= 5.0
    assert abs(metrics["obj_area_ratio"] - 1.2 ** 2) < 0.2


def test_subthreshold_field_returns_all_nan_object_metrics():
    lat, lon = _grid(n=11)
    field = np.full(lat.shape, 2.0)
    metrics = object_comparison(
        field, field, np.ones_like(field, dtype=bool), lat, lon,
        threshold_mm=10.0, smooth_cells=1, min_area_cells=1,
        motion_unit=None)
    assert set(metrics) == set(OBJECT_FIELDS)
    assert all(np.isnan(value) for value in metrics.values())


def test_one_cell_speckle_removed_by_smoothing():
    field = np.zeros((9, 9))
    field[4, 4] = 100.0
    obj = largest_object(field, np.ones_like(field, dtype=bool),
                         threshold_mm=20.0, smooth_cells=3,
                         min_area_cells=1)
    assert obj is None


def test_distribution_stats_all_nan_returns_nan_dict():
    field = np.full((4, 4), np.nan)
    stats = distribution_stats(field, np.ones_like(field, dtype=bool),
                               np.zeros_like(field), 0.1)
    assert set(stats) == set(DIST_FIELDS)
    assert all(np.isnan(value) for value in stats.values())
