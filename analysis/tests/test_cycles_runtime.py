"""Runtime changes must preserve windows, masks, and offline replot behavior."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cycles
import ets_score
import parent_qpf
import run
from field_cache import parent_cache_path, load_field, save_field
from parallel import ordered_map, positive_workers, worker_count
from test_cycles import _tiny_cycles_case, _tiny_cycle_fields, _stub_case


def _child_info(value):
    return value * value, os.getpid(), os.environ.get("OMP_NUM_THREADS")


def test_process_pool_preserves_order_and_limits_native_threads(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    rows = list(ordered_map(_child_info, [3, 1, 2, 4], workers=2))
    assert [row[0] for row in rows] == [9, 1, 4, 16]
    if worker_count(2, 4) > 1:
        assert all(row[1] != os.getpid() and row[2] == "1" for row in rows)
    assert os.environ["OMP_NUM_THREADS"] == "7"


def test_workers_respect_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert worker_count(20, 10) <= 2
    assert worker_count(20, 1) == 1
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert worker_count(20, 10) == 1


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "auto"])
def test_reject_bad_worker_counts(value):
    with pytest.raises(ValueError):
        positive_workers(value)


def test_mrms_nested_windows_decode_each_hour_once(tmp_path, monkeypatch):
    start = datetime(2024, 9, 26)
    end = start + timedelta(hours=6)
    lat = np.array([1., 0.])
    lon = np.array([280., 281.])
    glon, glat = np.meshgrid([-79.75, -79.25], [0.25, 0.75])
    calls = []

    def load(s3, t, cache):
        calls.append(t)
        h = (t - start).total_seconds() / 3600
        if h == 4:
            raise OSError("missing synthetic hour")
        data = np.array([[h, h * 2], [h / 3, h + 1]])
        if h == 1:
            data[0, 0] = np.nan
        return lat, lon, data

    monkeypatch.setattr(ets_score, "load_mrms_hour", load)
    monkeypatch.setattr(ets_score.boto3, "client", lambda *a, **k: None)
    starts = [start, start + timedelta(hours=2), start + timedelta(hours=5)]
    totals = ets_score.build_mrms_totals_windows(
        starts + [start], end, tmp_path, glat, glon)
    assert len(calls) == 6 and len(set(calls)) == 6
    for window_start in starts:
        expected = ets_score.build_mrms_total_window(
            window_start, end, tmp_path, glat, glon)
        np.testing.assert_allclose(totals[window_start], expected, equal_nan=True)


def test_mrms_empty_short_window_still_fails(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("missing hour")
    monkeypatch.setattr(ets_score, "load_mrms_hour", fail)
    monkeypatch.setattr(ets_score.boto3, "client", lambda *a, **k: None)
    start = datetime(2024, 9, 26)
    with pytest.raises(RuntimeError, match="No MRMS hours"):
        ets_score.build_mrms_totals_windows(
            [start, start + timedelta(hours=1)], start + timedelta(hours=2),
            tmp_path, np.zeros((2, 2)), np.zeros((2, 2)))


def test_stage4_shares_native_total_but_preserves_track_masks(monkeypatch):
    lat = np.array([[0., 0.], [10., 10.]])
    lon = np.array([[0., 10.], [0., 10.]])
    native = np.full((2, 2), 20.)
    start, end = datetime(2024, 9, 26), datetime(2024, 9, 28)
    cache = {}
    with patch.object(parent_qpf, "stage4_sum_days",
                      return_value=(lat, lon, native, ["20240927"])) as summed:
        a = parent_qpf.stage4_total_window(
            Path("unused"), start, end, [(0., 0.)], 50, cache)[2]
        b = parent_qpf.stage4_total_window(
            Path("unused"), start + timedelta(hours=6), end,
            [(10., 10.)], 50, cache)[2]
    assert summed.call_count == 1
    np.testing.assert_array_equal(a, [[20., 0.], [0., 0.]])
    np.testing.assert_array_equal(b, [[0., 0.], [0., 20.]])
    np.testing.assert_array_equal(native, np.full((2, 2), 20.))


def test_forecast_cache_invalidation_and_corrupt_recovery(tmp_path):
    source = tmp_path / "source.grb2"
    source.write_bytes(b"fixture")
    grid = np.zeros((2, 2))
    cache = tmp_path / "cache"
    key = parent_cache_path(cache, [source], 0, 6, grid, grid)
    field = np.array([[1., np.nan], [2., 3.]])
    save_field(key, field)
    np.testing.assert_array_equal(load_field(key, grid.shape), field)
    assert load_field(key, (3, 3)) is None
    assert key != parent_cache_path(cache, [source], 0, 9, grid, grid)
    assert key != parent_cache_path(cache, [source], 0, 6, grid + 1, grid)
    source.write_bytes(b"updated fixture")
    assert key != parent_cache_path(cache, [source], 0, 6, grid, grid)
    key.write_bytes(b"interrupted cache")
    assert load_field(key, grid.shape) is None


def test_parent_worker_reuses_cache_and_refreshes(tmp_path, monkeypatch):
    source = tmp_path / "source.grb2"
    source.write_bytes(b"fixture")
    case = _stub_case(tmp_path)
    grid = np.zeros((2, 2))
    monkeypatch.setattr(cycles, "parent_path_at_fhour", lambda *a: source)
    job = (case, 0, 6, grid, grid, tmp_path / "cache", False)
    with patch.object(cycles, "parent_window_total", return_value=grid + 5) as build:
        first, error = cycles._build_parent_window(job)
        second, error = cycles._build_parent_window(job)
        assert error is None and build.call_count == 1
        np.testing.assert_array_equal(first, second)
        cycles._build_parent_window((*job[:-1], True))
        assert build.call_count == 2


def test_parent_extraction_serial_and_processes_match_real_grib(tmp_path, monkeypatch):
    import eccodes

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    case = _stub_case(tmp_path)
    for fhour, amount in [(3, 2.), (6, 7.)]:
        gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        try:
            for key, value in {
                "Ni": 3, "Nj": 3,
                "latitudeOfFirstGridPointInDegrees": 1.,
                "latitudeOfLastGridPointInDegrees": 0.,
                "longitudeOfFirstGridPointInDegrees": 0.,
                "longitudeOfLastGridPointInDegrees": 1.,
                "iDirectionIncrementInDegrees": 0.5,
                "jDirectionIncrementInDegrees": 0.5,
                "productDefinitionTemplateNumber": 8,
                "shortName": "tp", "startStep": 0, "endStep": fhour,
            }.items():
                eccodes.codes_set(gid, key, value)
            eccodes.codes_set_values(gid, np.full(9, amount))
            path = tmp_path / f"09l.2024092400.hfsa.parent.atm.f{fhour:03d}.grb2"
            with open(path, "wb") as fh:
                eccodes.codes_write(gid, fh)
        finally:
            eccodes.codes_release(gid)
    glat, glon = case.fixed_grid()
    jobs = [(case, 0, 6, glat, glon, None, False),
            (case, 3, 6, glat, glon, None, False),
            (case, 0, 9, glat, glon, None, False)]
    serial = list(ordered_map(cycles._build_parent_window, jobs, workers=1))
    parallel = list(ordered_map(cycles._build_parent_window, jobs, workers=2))
    for expected, actual, value in zip(serial[:2], parallel[:2], [7., 5.]):
        assert expected[1] is None and actual[1] is None
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[0], value)
    assert serial[2][0] is None and parallel[2][0] is None
    assert "no parent.atm file" in parallel[2][1]


def test_identical_mrms_windows_are_built_once(tmp_path, monkeypatch):
    from dataclasses import replace
    ccase = _tiny_cycles_case(tmp_path, tmp_path)
    ccase.inits = ["2024092400", "2024092500"]
    cases = {init: replace(_stub_case(tmp_path), init_str=init,
                          init_dt=datetime.strptime(init, "%Y%m%d%H"))
             for init in ccase.inits}
    monkeypatch.setattr(cycles, "cycle_storm_case", lambda c, init: cases[init])
    monkeypatch.setattr(cycles, "discover_files", lambda *a: [(96, Path("fake"))])
    monkeypatch.setattr(cycles, "parent_window_total", lambda *a: np.ones((3, 3)))
    monkeypatch.setattr(cycles, "stage4_total_window", lambda *a: (None,) * 4)
    with patch.object(cycles, "build_mrms_total_window",
                      return_value=np.ones((3, 3))) as obs:
        fields = cycles.build_cycle_fields(ccase)
    assert obs.call_count == 1
    assert fields["cycles"][0]["mrms_win"] is fields["cycles"][1]["mrms_win"]


def test_cycles_replot_uses_only_saved_tables(tmp_path, monkeypatch):
    ccase = _tiny_cycles_case(tmp_path / "unmounted", tmp_path)
    cycles.compute_cycles(ccase, fields=_tiny_cycle_fields())
    saved_tables = {p: p.read_bytes() for p in tmp_path.glob("*.csv")}
    # Verify offline replot still works even if the best-track archive is gone.
    ccase.best_track = tmp_path / "missing.bdeck"
    with patch.object(cycles, "build_cycle_fields", side_effect=AssertionError), \
         patch.object(cycles, "parse_bdeck_full", side_effect=AssertionError), \
         patch.object(cycles, "cycle_storm_case", side_effect=AssertionError), \
         patch.object(cycles, "plot_metrics", wraps=cycles.plot_metrics) as plot:
        cycles.replot_cycles_from_csv(ccase)
    assert plot.call_count == 1
    assert plot.call_args.args[1][0]["cont"]["rmse"] == 1.0
    for path, content in saved_tables.items():
        assert path.read_bytes() == content


def test_cycles_cli_options_and_scope():
    args = run.parse_options(["case.yaml", "cycles", "--workers", "4",
                              "--no-animation", "--no-ml-features"])
    assert args.workers == 4 and args.no_animation and args.no_ml_features
    assert run.parse_options(["case.yaml", "cycles", "--replot"]).replot
    with pytest.raises(SystemExit):
        run.parse_options(["case.yaml", "compare", "--workers", "2"])
    with pytest.raises(SystemExit):
        run.parse_options(["case.yaml", "cycles", "--workers", "0"])


def test_replot_command_routes_cycles_yaml_without_run_root(tmp_path, monkeypatch):
    config = tmp_path / "cycles.yaml"
    config.write_text("run_root: /unmounted/HFSA\nvalid_start: 2024092600\n"
                      "valid_end: 2024092800\ndomain: [0, 1, 0, 1]\n")
    with patch.object(cycles, "replot_cycles_from_csv") as replot:
        run.main([str(config), "replot"])
        assert replot.call_count == 1
