"""Local unit tests for the cycles product (no Hercules data needed).

Run directly:   python3 analysis/tests/test_cycles.py
Or via pytest:  pytest analysis/tests/test_cycles.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hafs_case import (
    CyclesCase, cycles_from_yaml, discover_inits, window_hours,
    cycle_eligibility, cycle_storm_case, from_yaml,
)


# ---------------------------------------------------------------------------
# Window math
# ---------------------------------------------------------------------------

def test_window_hours():
    f1, f2 = window_hours(datetime(2024, 9, 24, 0),
                          datetime(2024, 9, 26, 0),
                          datetime(2024, 9, 28, 0))
    assert (f1, f2) == (48, 96)
    # Init exactly at the window start -> f1 == 0.
    f1, f2 = window_hours(datetime(2024, 9, 26, 0),
                          datetime(2024, 9, 26, 0),
                          datetime(2024, 9, 28, 0))
    assert (f1, f2) == (0, 48)


def test_cycle_eligibility():
    vs, ve = datetime(2024, 9, 26, 0), datetime(2024, 9, 28, 0)
    ok, reason = cycle_eligibility(datetime(2024, 9, 24, 0), 126, vs, ve)
    assert ok and reason == ""
    # Init after window start -> ineligible.
    ok, reason = cycle_eligibility(datetime(2024, 9, 26, 6), 126, vs, ve)
    assert not ok and "after the window start" in reason
    # Run too short to reach window end -> ineligible.
    ok, reason = cycle_eligibility(datetime(2024, 9, 24, 0), 36, vs, ve)
    assert not ok and "before the window end" in reason
    # Boundary cases are eligible: init == valid_start, end == valid_end.
    assert cycle_eligibility(vs, 48, vs, ve)[0]
    assert cycle_eligibility(datetime(2024, 9, 26, 0), 48, vs, ve)[0]


# ---------------------------------------------------------------------------
# Init discovery
# ---------------------------------------------------------------------------

def test_discover_inits():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("2024092412", "2024092400", "notacycle", "20240924"):
            (root / name).mkdir()
        (root / "2024092418").write_text("a file, not a dir")
        assert discover_inits(root) == ["2024092400", "2024092412"]


def test_discover_inits_missing_root():
    try:
        discover_inits(Path("/nonexistent/dir/xyz"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# cycles_from_yaml
# ---------------------------------------------------------------------------

_CYCLES_YAML = """\
run_root: {root}
valid_start: 2024092600
valid_end: 2024092800
storm_name: Hurricane Helene
domain: [15.0, 42.0, -100.0, -60.0]
mask_radius_km: 500
out_dir: {out}
"""


def test_cycles_from_yaml_defaults_and_slug():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "helene_hfsa_cycles.yaml"
        yml.write_text(_CYCLES_YAML.format(root=tmp, out=tmp))
        cc = cycles_from_yaml(yml)
        assert cc.valid_start == datetime(2024, 9, 26, 0)
        assert cc.valid_end == datetime(2024, 9, 28, 0)
        assert cc.inits is None
        assert cc.ets_threshold_mm == 25.0
        assert cc.thresholds_mm[0] == 1
        assert cc.case_slug == "helene_hfsa_cycles"
        assert cc.output_slug == "helene_hfsa_cycles_2024092600_2024092800"
        glat, glon = cc.fixed_grid()
        assert glat.shape == glon.shape and glat.ndim == 2


def test_cycles_from_yaml_rejects_bad_window():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "bad.yaml"
        yml.write_text(_CYCLES_YAML.format(root=tmp, out=tmp).replace(
            "valid_end: 2024092800", "valid_end: 2024092600"))
        try:
            cycles_from_yaml(yml)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "valid_end" in str(e)


def test_cycles_from_yaml_rejects_per_init_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "case.yaml"
        yml.write_text(f"run_dir: {tmp}\ninit: 2024092400\n")
        try:
            cycles_from_yaml(yml)
            assert False, "expected KeyError"
        except KeyError as e:
            assert "run_root" in str(e)


def test_from_yaml_rejects_cycles_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "cyc.yaml"
        yml.write_text(_CYCLES_YAML.format(root=tmp, out=tmp))
        try:
            from_yaml(yml)
            assert False, "expected KeyError"
        except KeyError as e:
            assert "cycles" in str(e)


# ---------------------------------------------------------------------------
# cycle_storm_case
# ---------------------------------------------------------------------------

# Minimal 2-fix atcfunix (cols: basin, cy, init, technum, tech, tau, lat,
# lon, vmax, mslp, ty ... — parse_atcfunix needs >= 8 columns).
_ATCF = (
    "AL, 09, 2024092400, 03, HFSA, 000, 168N, 832W, 45, 1002, TS\n"
    "AL, 09, 2024092400, 03, HFSA, 048, 250N, 840W, 90, 960, HU\n"
)


def _tiny_cycles_case(root, out):
    return CyclesCase(
        run_root=Path(root),
        valid_start=datetime(2024, 9, 26, 0),
        valid_end=datetime(2024, 9, 28, 0),
        storm_name="Testorm", model_label="HAFS-A",
        domain=(0.0, 1.0, 0.0, 1.0), grid_res=0.5,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1], ets_threshold_mm=1.0,
        out_dir=Path(out), mrms_cache_dir=Path("/tmp"),
        stage4_cache_dir=Path("/tmp"), inits=None,
        case_slug="testcycles",
    )


def test_cycle_storm_case_builds_from_run_root():
    with tempfile.TemporaryDirectory() as tmp:
        cyc_dir = Path(tmp) / "2024092400"
        cyc_dir.mkdir()
        (cyc_dir / "storm09l.2024092400.hfsa.trak.atcfunix").write_text(_ATCF)
        cc = _tiny_cycles_case(tmp, tmp)
        case = cycle_storm_case(cc, "2024092400")
        assert case.run_dir == cyc_dir
        assert case.init_dt == datetime(2024, 9, 24, 0)
        assert case.init_str == "2024092400"
        assert case.storm_name == "Testorm"
        assert len(case.track) == 2
        assert case.mask_radius_km == 500.0


# ---------------------------------------------------------------------------
# Windowed observation loaders (Task 2)
# ---------------------------------------------------------------------------

def test_build_mrms_total_window_sums_requested_hours():
    """Patch the hour loader; check which hour-stamps are requested and
    that the window total is the plain sum, regridded."""
    import ets_score

    requested = []

    def fake_load(s3, hour_end_dt, cache_dir):
        requested.append(hour_end_dt)
        lat = np.linspace(0.0, 1.0, 5)
        lon = np.linspace(0.0, 1.0, 5)
        return lat, lon, np.ones((5, 5))

    orig = ets_score.load_mrms_hour
    ets_score.load_mrms_hour = fake_load
    try:
        glat, glon = np.meshgrid(np.linspace(0.2, 0.8, 3),
                                 np.linspace(0.2, 0.8, 3))[::-1]
        total = ets_score.build_mrms_total_window(
            datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 3),
            Path("/tmp"), glat, glon)
    finally:
        ets_score.load_mrms_hour = orig
    # Hour files are stamped by hour END: 01Z, 02Z, 03Z — not 00Z.
    assert requested == [datetime(2024, 9, 26, 1),
                         datetime(2024, 9, 26, 2),
                         datetime(2024, 9, 26, 3)]
    assert np.allclose(total, 3.0)


def test_parent_path_at_fhour():
    from parent_qpf import parent_path_at_fhour
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        names = ["storm09l.2024092400.hfsa.parent.atm.f048.grb2",
                 "storm09l.2024092400.hfsa.parent.atm.f096.grb2"]
        for n in names:
            (run_dir / n).write_text("")
        case = _stub_case(run_dir)
        assert parent_path_at_fhour(case, 48).name == names[0]
        assert parent_path_at_fhour(case, 96).name == names[1]
        assert parent_path_at_fhour(case, 72) is None


def _stub_case(run_dir):
    """Minimal StormCase for glob-based tests."""
    from hafs_case import StormCase
    return StormCase(
        run_dir=Path(run_dir), init_dt=datetime(2024, 9, 24, 0),
        storm_name="Testorm", model_label="HAFS-A",
        domain=(0.0, 1.0, 0.0, 1.0), grid_res=0.5,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1], out_dir=Path(run_dir),
        mrms_cache_dir=Path("/tmp"), stage4_cache_dir=Path("/tmp"),
        fhours_filter=None,
        track=[(datetime(2024, 9, 24, 0), 0.5, 0.5)],
        case_slug="testcase", init_str="2024092400",
    )


def test_stage4_label():
    from parent_qpf import stage4_label
    assert stage4_label(["20240926", "20240927"]) == \
        "~09-25 12Z – 09-27 12Z (2×24h)"


# ---------------------------------------------------------------------------
# Windowed forecast fields + union swath (Task 3)
# ---------------------------------------------------------------------------

def test_filter_window_pairs_boundaries():
    from cycles import filter_window_pairs
    pairs = [(h, Path(f"f{h:03d}")) for h in (42, 48, 51, 90, 96, 99)]
    kept = filter_window_pairs(pairs, 48, 96)
    # f1 exclusive (a bucket ENDING at 48 holds pre-window rain),
    # f2 inclusive (the bucket ending at 96 is the window's last rain).
    assert [h for h, _ in kept] == [51, 90, 96]


def test_nest_window_total_rejects_cumulative_records():
    import cycles

    case = _stub_case(Path("."))
    orig_discover_files = cycles.discover_files
    orig_hafs_event_total = cycles.hafs_event_total
    cycles.discover_files = lambda *args: [(96, Path("f096"))]
    cycles.hafs_event_total = lambda *args: (np.zeros((2, 2)), "cumulative")
    try:
        try:
            cycles.nest_window_total(case, 48, 96,
                                     np.zeros((2, 2)), np.zeros((2, 2)))
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "no per-interval" in str(e)
    finally:
        cycles.discover_files = orig_discover_files
        cycles.hafs_event_total = orig_hafs_event_total


def test_union_swath_is_union_of_single_masks():
    from cycles import union_swath
    glat, glon = np.meshgrid(np.linspace(0, 10, 21),
                             np.linspace(0, 10, 21))[::-1]
    pts_a = [(2.0, 2.0)]
    pts_b = [(8.0, 8.0)]
    m_a = union_swath(pts_a, 150.0, glat, glon)
    m_b = union_swath(pts_b, 150.0, glat, glon)
    m_ab = union_swath(pts_a + pts_b, 150.0, glat, glon)
    assert m_a.any() and m_b.any()
    assert not (m_a & m_b).any()          # disjoint circles at 150 km
    assert np.array_equal(m_ab, m_a | m_b)


def test_window_track_points_hourly_inclusive():
    from cycles import window_track_points
    case = _stub_case(Path("."))
    pts = window_track_points(case, datetime(2024, 9, 26, 0),
                              datetime(2024, 9, 26, 6))
    assert len(pts) == 7                  # 00Z..06Z inclusive, hourly
    # Single-fix track -> position clamps to that fix everywhere.
    assert all(p == (0.5, 0.5) for p in pts)


# ---------------------------------------------------------------------------
# compute_cycles scoring path (Task 4)
# ---------------------------------------------------------------------------

def _tiny_cycle_fields():
    glat, glon = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))[::-1]
    obs = np.full((4, 4), 10.0)
    return dict(
        grid_lat=glat, grid_lon=glon,
        mrms_win=obs,
        stage4_win=None, s4_label="unavailable",
        swath=np.ones((4, 4), dtype=bool),
        cycles=[
            dict(init_str="2024092400", init_dt=datetime(2024, 9, 24, 0),
                 f1=48, f2=96, nest_win=obs + 2.0, parent_win=obs - 1.0),
            dict(init_str="2024092500", init_dt=datetime(2024, 9, 25, 0),
                 f1=24, f2=72, nest_win=obs.copy(), parent_win=obs.copy()),
        ],
    )


def test_compute_cycles_writes_csv_and_pngs():
    import csv as csvmod
    from cycles import compute_cycles
    with tempfile.TemporaryDirectory() as tmp:
        ccase = _tiny_cycles_case(tmp, tmp)
        compute_cycles(ccase, fields=_tiny_cycle_fields())
        slug = "testcycles_2024092600_2024092800"
        csv_path = Path(tmp) / f"cycles_{slug}.csv"
        metrics_png = Path(tmp) / f"cycles_metrics_{slug}.png"
        maps_png = Path(tmp) / f"cycles_maps_{slug}.png"
        assert csv_path.exists(), "CSV not written"
        assert metrics_png.exists(), "metrics PNG not written"
        assert maps_png.exists(), "maps PNG not written"
        with open(csv_path) as fh:
            rows = list(csvmod.DictReader(fh))
        # 2 cycles x 2 forecasts x 1 obs (Stage IV None) x 1 threshold.
        assert len(rows) == 4
        assert rows[0].keys() == {
            "init", "forecast", "observation", "threshold", "n", "rmse",
            "mae", "bias_mm", "r", "a", "b", "c", "d", "ets", "bias",
            "pod", "far", "csi", "hss"}
        by_key = {(r["init"], r["forecast"]): r for r in rows}
        early_nest = by_key[("2024092400", "nest")]
        early_parent = by_key[("2024092400", "parent")]
        late_nest = by_key[("2024092500", "nest")]
        assert all(r["observation"] == "MRMS" for r in rows)
        # Constant offsets: rmse == |bias_mm| (positive bias = over-forecast).
        assert abs(float(early_nest["rmse"]) - 2.0) < 1e-9
        assert abs(float(early_nest["bias_mm"]) - 2.0) < 1e-9
        assert abs(float(early_parent["rmse"]) - 1.0) < 1e-9
        assert abs(float(early_parent["bias_mm"]) - (-1.0)) < 1e-9
        assert abs(float(late_nest["rmse"]) - 0.0) < 1e-9
        assert int(early_nest["n"]) == 16
        # Perfect >= 1mm coverage everywhere -> ETS-relevant counts: all hits.
        assert int(early_nest["a"]) == 16 and int(early_nest["c"]) == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
