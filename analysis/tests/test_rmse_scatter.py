"""Local unit tests for rmse_scatter (no Hercules data needed).

Run directly:   python3 analysis/tests/test_rmse_scatter.py
Or via pytest:  pytest analysis/tests/test_rmse_scatter.py -v
"""
import csv
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmse_scatter import valid_points, compute_rmse
from hafs_case import StormCase


def test_valid_points_mirrors_score_pair_selection():
    # Same fixture as test_ets_full.test_score_pair_counts_only_valid_points:
    # swath excludes the last column; obs has one NaN inside the swath.
    fcst = np.array([[10.0, 0.0, 0.0],
                     [10.0, 10.0, 0.0],
                     [0.0, 0.0, 0.0]])
    obs = np.array([[10.0, 0.0, 99.0],
                    [0.0, np.nan, 99.0],
                    [0.0, 0.0, 99.0]])
    swath = np.array([[True, True, False],
                      [True, True, False],
                      [True, True, False]], dtype=bool)
    f, o = valid_points(fcst, obs, swath)
    assert f.size == o.size == 5
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(o))
    # The NaN-obs point (1,1) is dropped, so fcst keeps one 10 from col 0
    # of row 1 but not the 10 at (1,1).
    assert sorted(f.tolist()) == [0.0, 0.0, 0.0, 10.0, 10.0]


def _tiny_case(out_dir):
    return StormCase(
        run_dir=Path("."), init_dt=datetime(2024, 9, 24, 0),
        storm_name="Testorm", model_label="HAFS-A",
        domain=(0.0, 1.0, 0.0, 1.0), grid_res=0.5,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1], out_dir=Path(out_dir),
        mrms_cache_dir=Path("/tmp"), stage4_cache_dir=Path("/tmp"),
        fhours_filter=None,
        track=[(datetime(2024, 9, 24, 0), 0.5, 0.5)],
        case_slug="testcase", init_str="2024092400",
    )


def _tiny_fields():
    lat, lon = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    obs = np.full((4, 4), 10.0)
    return dict(
        max_fhour=6, grid_lat=lat, grid_lon=lon,
        nest_total=obs + 2.0,          # bias +2, rmse 2 vs mrms
        apcp_mode="incremental",
        parent_total=obs - 1.0,        # bias -1, rmse 1 vs mrms
        mrms_total=obs,
        stage4_grid=None, s4_label="unavailable",
        swath=np.ones((4, 4), dtype=bool),
    )


def test_compute_rmse_writes_csv_and_png():
    with tempfile.TemporaryDirectory() as tmp:
        case = _tiny_case(tmp)
        compute_rmse(case, fields=_tiny_fields())
        csv_path = Path(tmp) / "rmse_testcase_2024092400.csv"
        png_path = Path(tmp) / "rmse_scatter_testcase_2024092400.png"
        assert csv_path.exists(), "CSV not written"
        assert png_path.exists(), "PNG not written"
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        # Stage IV unavailable -> 2 forecasts x 1 observation = 2 rows.
        assert len(rows) == 2
        by_fcst = {r["forecast"]: r for r in rows}
        assert set(by_fcst) == {"parent", "nest"}
        assert all(r["observation"] == "MRMS" for r in rows)
        # Constant offsets: rmse == |bias|, mae == |bias|.
        assert abs(float(by_fcst["parent"]["rmse"]) - 1.0) < 1e-9
        assert abs(float(by_fcst["parent"]["bias"]) - (-1.0)) < 1e-9
        assert abs(float(by_fcst["nest"]["rmse"]) - 2.0) < 1e-9
        assert abs(float(by_fcst["nest"]["bias"]) - 2.0) < 1e-9
        assert int(by_fcst["parent"]["n"]) == 16


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
