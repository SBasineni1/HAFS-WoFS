"""Exercise spawned workers with offline observation fixtures and real outputs."""

import csv
import os
from pathlib import Path
import sys
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cycles
import ets_score
import parent_qpf
from parallel import ordered_map, run_stage_job, worker_count
from test_cycles import _tiny_cycles_case, _tiny_cycle_fields


def _two_cpus(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    if worker_count(2, 2) < 2:
        pytest.skip("Two CPUs required to exercise spawned workers")


def _observation_fixture(job):
    label, ccase, payload, barrier = job
    if barrier is not None:
        # Both source jobs must be alive at once; no wall-time speed assertion.
        (barrier / label).touch()
        deadline = monotonic() + 20
        while len(list(barrier.iterdir())) < 2:
            if monotonic() > deadline:
                raise RuntimeError("Observation jobs did not overlap")
            sleep(0.02)
    glat, glon = ccase.fixed_grid()
    calls = []

    def hour(s3, valid, cache):
        calls.append(valid)
        return (np.array([1., .5, 0.]), np.array([0., .5, 1.]),
                np.full((3, 3), float(valid.hour + 1)))

    def days(*args):
        calls.append("days")
        return glat, glon, np.full(glat.shape, 20.), ["20240927"]

    function = (cycles._build_mrms_observations if label == "MRMS"
                else cycles._build_stage4_observations)
    with patch.object(ets_score, "load_mrms_hour", side_effect=hour), \
         patch.object(ets_score.boto3, "client", return_value=None), \
         patch.object(parent_qpf, "stage4_sum_days", side_effect=days):
        result = run_stage_job((label, function, (ccase, payload, glat, glon)))
    return result, os.getpid(), len(calls)


def test_observations_overlap_and_match_serial_without_duplicate_reads(tmp_path, monkeypatch):
    from datetime import timedelta
    _two_cpus(monkeypatch)
    ccase = _tiny_cycles_case(tmp_path, tmp_path)
    ccase.valid_end = ccase.valid_start + timedelta(hours=6)
    ccase.mrms_cache_dir = tmp_path / "mrms"
    starts = [ccase.valid_start, ccase.valid_start + timedelta(hours=2)]
    windows = [(str(i), start, ccase.valid_end, [(i, i)]) for i, start in enumerate(starts)]
    jobs = [("MRMS", ccase, starts, None), ("StageIV", ccase, windows, None)]
    serial = list(ordered_map(_observation_fixture, jobs, 1))
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    parallel = list(ordered_map(_observation_fixture,
                    [(*job[:-1], barrier) for job in jobs], 2))
    assert len({row[1] for row in parallel}) == 2
    assert all(row[1] != os.getpid() for row in parallel)
    assert [row[2] for row in serial] == [6, 1]
    assert [row[2] for row in parallel] == [6, 1]
    for key, value in serial[0][0].items():
        np.testing.assert_array_equal(value, parallel[0][0][key])
    for key, (field, label) in serial[1][0].items():
        np.testing.assert_array_equal(field, parallel[1][0][key][0])
        assert label == parallel[1][0][key][1]


def test_serial_parallel_csv_and_png_outputs_match(tmp_path, monkeypatch):
    _two_cpus(monkeypatch)
    fields = _tiny_cycle_fields()
    fields["mrms_win"][:] = np.arange(16.).reshape(4, 4) + 2
    fields["cycles"][0]["parent_win"][:] = fields["mrms_win"] - 1
    fields["cycles"][1]["parent_win"][:] = fields["mrms_win"] + 1
    fields["cycles"][0]["stage4_win"] = fields["mrms_win"] + 2
    fields["cycles"][0]["parent_win"][0, 0] = np.nan
    fields["cycles"][1]["mrms_win"] = fields["mrms_win"] + 1
    fields["swath"][1, 1] = False
    outputs = []
    for workers in (1, 2):
        path = tmp_path / str(workers)
        ccase = _tiny_cycles_case(tmp_path / "no-model-data", path)
        ccase.workers = workers
        ccase.ml_features = True
        ccase.ml_features_csv = path / "ml_features.csv"
        cycles.compute_cycles(ccase, fields=fields)
        outputs.append({p.name: p.read_bytes() for p in path.glob("*.csv")})
    assert outputs[0] == outputs[1]
    serial_pngs = sorted((tmp_path / "1").glob("*.png"))
    assert len(serial_pngs) >= 7
    for path in serial_pngs:
        with Image.open(path) as a, Image.open(tmp_path / "2" / path.name) as b:
            assert a.size == b.size
            # Font/mathtext caches can shift tight_layout by a pixel in a
            # long-lived parent versus a fresh worker; numerical CSVs above
            # are exact, while rendering permits a small raster difference.
            delta = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
            assert delta.mean() / 255 < 0.01
    with open(tmp_path / "2" / "ml_features.csv") as fh:
        assert len(list(csv.DictReader(fh))) == 2  # appended only once, by parent


def _render_without_map(job):
    # Avoid external Natural Earth assets, but exercise actual GIF encoding.
    with patch.object(cycles, "_map_context", return_value=None):
        return cycles._render_product(job)


def test_parallel_gifs_have_distinct_files_and_frames(tmp_path, monkeypatch):
    _two_cpus(monkeypatch)
    ccase = _tiny_cycles_case(tmp_path, tmp_path)
    fields = _tiny_cycle_fields()
    jobs = [(tmp_path / f"{name}.gif", function,
             (ccase, fields, tmp_path / f"{name}.gif"), True)
            for name, function in [("forecast", cycles.animate_cycle_qpf),
                                   ("difference", cycles.animate_cycle_difference),
                                   ("observed", cycles.animate_cycle_observed)]]
    messages = list(ordered_map(_render_without_map, jobs, 2))
    assert all(message.startswith("Saved movie:") for message in messages)
    for path, *_ in jobs:
        with Image.open(path) as image:
            assert image.format == "GIF" and image.n_frames == 2


def _fail(*args):
    raise RuntimeError("synthetic worker failure")


def test_optional_animation_failure_and_required_task_failure(tmp_path, monkeypatch):
    _two_cpus(monkeypatch)
    optional = (tmp_path / "optional.gif", _fail, (), True)
    assert "Animation unavailable" in cycles._render_product(optional)
    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        list(ordered_map(cycles._render_product,
                         [(tmp_path / "required.png", _fail, (), False)] * 2, 2))


def test_parallel_map_assets_precede_all_render_jobs(tmp_path, monkeypatch):
    _two_cpus(monkeypatch)
    events = []
    ccase = _tiny_cycles_case(tmp_path, tmp_path)
    ccase.workers = 2
    ccase.make_animation = True
    monkeypatch.setattr(cycles, "_prepare_map_features", lambda c: events.append("assets"))

    def map_jobs(function, jobs, workers):
        if function is cycles._render_product:
            assert events == ["assets"]
            assert sum(job[3] for job in jobs) == 3
            paths = [job[0] for job in jobs]
            assert len(paths) == len(set(paths))
            events.append("render")
            yield from ["test render"] * len(jobs)
        else:
            yield from map(function, jobs)

    monkeypatch.setattr(cycles, "ordered_map", map_jobs)
    cycles.compute_cycles(ccase, fields=_tiny_cycle_fields())
    assert events == ["assets", "render"]
