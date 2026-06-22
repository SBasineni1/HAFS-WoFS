"""Local unit tests for hafs_case ATCF parsing (no Hercules data needed).

Run directly:   python3 analysis/tests/test_hafs_case.py
Or via pytest:  pytest analysis/tests/test_hafs_case.py -v
"""
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FIX = Path(__file__).resolve().parent / "fixtures"

import numpy as np
from hafs_case import decode_latlon, parse_atcfunix, detect_model, auto_domain, StormCase, from_yaml, find_atcfunix


def test_decode_latlon():
    assert decode_latlon("168N") == 16.8
    assert decode_latlon("832W") == -83.2
    assert decode_latlon("105S") == -10.5
    assert decode_latlon("50E") == 5.0


def test_parse_atcfunix_reproduces_helene_track():
    name, init_dt, track = parse_atcfunix(FIX / "helene.atcfunix")
    assert name == "Helene"
    assert init_dt == datetime(2024, 9, 24, 0)
    # First three known 6-hourly fixes from the current hardcoded TC_TRACK_6H.
    assert track[0] == (datetime(2024, 9, 24, 0), 16.8, -83.2)
    assert track[1] == (datetime(2024, 9, 24, 6), 17.8, -83.5)
    assert track[2] == (datetime(2024, 9, 24, 12), 19.0, -83.8)
    # Sorted ascending by valid time, no duplicate lead hours.
    times = [t for t, _, _ in track]
    assert times == sorted(times)
    assert len(times) == len(set(times))


def test_detect_model():
    assert detect_model("/work2/.../helene/HFSA") == "HAFS-A"
    assert detect_model("/work2/.../helene/HFSB") == "HAFS-B"
    assert detect_model("/data/hfsa_run/lower") == "HAFS-A"   # case-insensitive
    assert detect_model("/work2/.../helene/other") == "HAFS"


def test_auto_domain_pads_track_bbox():
    track = [
        (datetime(2024, 9, 24, 0), 16.8, -83.2),
        (datetime(2024, 9, 26, 0), 28.8, -84.1),
        (datetime(2024, 9, 29, 6), 44.3, -61.5),
    ]
    lat_min, lat_max, lon_min, lon_max = auto_domain(track, pad_deg=2.0)
    assert lat_min == 14.8 and lat_max == 46.3
    assert lon_min == -86.1 and lon_max == -59.5


def _toy_case():
    track = [
        (datetime(2024, 9, 24, 0), 16.8, -83.2),
        (datetime(2024, 9, 24, 6), 17.8, -83.5),
    ]
    return StormCase(
        run_dir=Path("/tmp/HFSA"), init_dt=datetime(2024, 9, 24, 0),
        storm_name="Helene", model_label="HAFS-A",
        domain=(15.0, 20.0, -90.0, -80.0), grid_res=1.0,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1, 5], out_dir=Path("/tmp/out"),
        mrms_cache_dir=Path("/tmp/mrms"), stage4_cache_dir=Path("/tmp/s4"),
        fhours_filter=None, track=track, case_slug="helene_hfsa",
        init_str="2024092400",
    )


def test_position_at_interpolates_and_clamps():
    c = _toy_case()
    # Midpoint between the two 6-hourly fixes.
    lat, lon = c.position_at(datetime(2024, 9, 24, 3))
    assert abs(lat - 17.3) < 1e-9 and abs(lon - (-83.35)) < 1e-9
    # Before/after the track clamps to the endpoints.
    assert c.position_at(datetime(2024, 9, 23, 0)) == (16.8, -83.2)
    assert c.position_at(datetime(2024, 9, 25, 0)) == (17.8, -83.5)


def test_fixed_grid_shape():
    c = _toy_case()
    grid_lat, grid_lon = c.fixed_grid()
    # lon -90..-80 step 1 -> 11 cols; lat 15..20 step 1 -> 6 rows.
    assert grid_lat.shape == (6, 11)
    assert grid_lon.shape == (6, 11)


def test_globs_use_init_str():
    c = _toy_case()
    assert c.parent_glob() == "**/*2024092400*parent.atm.f*.grb2"
    assert c.storm_glob() == "**/*2024092400*storm.atm.f*.grb2"


def test_from_yaml_minimal_autoderives():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        run_dir = tmpdir / "helene" / "HFSA"
        run_dir.mkdir(parents=True)
        shutil.copy(FIX / "helene.atcfunix", run_dir / "helene.atcfunix")
        yaml_path = tmpdir / "helene_hfsa.yaml"
        yaml_path.write_text(f"run_dir: {run_dir}\n")

        case = from_yaml(yaml_path)

        assert case.case_slug == "helene_hfsa"
        assert case.model_label == "HAFS-A"          # auto from path
        assert case.init_dt == datetime(2024, 9, 24, 0)  # from atcfunix
        assert case.storm_name == "Helene"            # from atcfunix name field
        assert case.init_str == "2024092400"
        assert case.grid_res == 0.05                  # default
        assert case.mask_radius_km == 500.0           # default
        assert case.thresholds_mm == [1, 5, 10, 25, 50, 75, 100, 150, 200, 250]
        assert case.out_dir == Path("analysis/output/helene_hfsa")
        # Domain auto-derived from the track bbox (non-empty, sane ordering).
        lat_min, lat_max, lon_min, lon_max = case.domain
        assert lat_min < lat_max and lon_min < lon_max
    finally:
        shutil.rmtree(tmpdir)


def test_from_yaml_overrides_win():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        run_dir = tmpdir / "HFSA"
        run_dir.mkdir(parents=True)
        shutil.copy(FIX / "helene.atcfunix", run_dir / "x.atcfunix")
        yaml_path = tmpdir / "case.yaml"
        yaml_path.write_text(
            f"run_dir: {run_dir}\n"
            "storm_name: Test Storm\n"
            "model_label: HAFS-X\n"
            "domain: [15.0, 42.0, -100.0, -60.0]\n"
            "mask_radius_km: 300\n"
            "out_dir: /tmp/custom_out\n"
        )
        case = from_yaml(yaml_path)
        assert case.storm_name == "Test Storm"
        assert case.model_label == "HAFS-X"
        assert case.domain == (15.0, 42.0, -100.0, -60.0)
        assert case.mask_radius_km == 300.0
        assert case.out_dir == Path("/tmp/custom_out")
    finally:
        shutil.rmtree(tmpdir)


def test_find_atcfunix_missing_raises():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        try:
            find_atcfunix(tmpdir)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert str(tmpdir) in str(e)
    finally:
        shutil.rmtree(tmpdir)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
