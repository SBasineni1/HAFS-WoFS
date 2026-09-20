"""Cached geometry must preserve griddata's values and missing-data support."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ets_full
from ets_full import regrid_2d_to_fixed


@pytest.mark.parametrize("curved", [False, True])
def test_mesh_reuse_preserves_distinct_masks_and_outside_hull(curved):
    lon, lat = np.meshgrid(np.linspace(-90, -75, 12), np.linspace(20, 40, 15))
    if curved:
        lat = lat + 0.1 * np.sin(lon)
        lon = lon + 0.1 * np.cos(lat)
    glon, glat = np.meshgrid(np.linspace(-92, -73, 25), np.linspace(18, 42, 26))
    total = np.random.default_rng(7).uniform(0, 200, lat.shape)
    fields = [np.where(lat < 30, total, 0), np.where(lat > 30, total, 0)]
    cache = {}
    with patch.object(ets_full, "Delaunay", wraps=ets_full.Delaunay) as build:
        results = []
        for field in fields:
            expected = regrid_2d_to_fixed(lat, lon, field, glat, glon)
            actual = regrid_2d_to_fixed(lat, lon, field, glat, glon,
                                       geometry_cache=cache)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12,
                                       equal_nan=True)
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            results.append(actual)
        assert build.call_count == 1
    assert not np.allclose(results[0], results[1], equal_nan=True)
    assert np.isnan(results[0][0, 0])


def test_changed_coordinates_and_missing_support_invalidate_mesh():
    lon, lat = np.meshgrid(np.arange(5.), np.arange(4.))
    glon, glat = np.meshgrid(np.linspace(-1, 5, 12), np.linspace(-1, 4, 11))
    field = np.arange(20.).reshape(4, 5)
    missing = field.copy()
    missing[0, 0] = np.nan
    missing[1, 1] = np.inf
    cache = {}
    with patch.object(ets_full, "Delaunay", wraps=ets_full.Delaunay) as build:
        for slat, slon, values in [(lat, lon, field), (lat, lon, missing),
                                   (lat + 0.25, lon, missing), (lat, lon, field)]:
            expected = regrid_2d_to_fixed(slat, slon, values, glat, glon)
            actual = regrid_2d_to_fixed(slat, slon, values, glat, glon,
                                       geometry_cache=cache)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12,
                                       equal_nan=True)
        assert build.call_count == 4


def test_axes_boundaries_and_new_targets_use_same_mesh():
    lat, lon = np.arange(4.), np.arange(5.)
    field = np.arange(20.).reshape(4, 5)
    cache = {}
    with patch.object(ets_full, "Delaunay", wraps=ets_full.Delaunay) as build:
        # Include vertices/hull edges as well as a second, denser target mesh.
        for size in [5, 17]:
            glon, glat = np.meshgrid(np.linspace(0, 4, size),
                                    np.linspace(0, 3, size))
            expected = regrid_2d_to_fixed(lat, lon, field, glat, glon)
            actual = regrid_2d_to_fixed(lat, lon, field, glat, glon,
                                       geometry_cache=cache)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert build.call_count == 1
