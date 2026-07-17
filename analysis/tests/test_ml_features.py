import csv
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_features import (
    TARGET_COLUMNS, append_features, disc_mean, extract_cycle_features,
    translation_speed_kt,
)


def test_append_features_upserts_and_extends_header(tmp_path):
    output = tmp_path / "features.csv"
    append_features([
        {"storm": "alpha", "model": "A", "init": "2024010100",
         "value": 1},
        {"storm": "beta", "model": "B", "init": "2024010106",
         "value": 2},
    ], output)
    with open(output, newline="") as fh:
        fresh = list(csv.DictReader(fh))
    assert len(fresh) == 2

    append_features([
        {"storm": "alpha", "model": "A", "init": "2024010100",
         "value": 9, "new_value": 3},
    ], output)
    with open(output, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = reader.fieldnames
    assert header == ["storm", "model", "init", "value", "new_value"]
    assert len(rows) == 2
    by_storm = {row["storm"]: row for row in rows}
    assert by_storm["alpha"]["value"] == "9"
    assert by_storm["alpha"]["new_value"] == "3"
    assert by_storm["beta"]["new_value"] == ""


def test_translation_speed_centered_twelve_hours():
    init = datetime(2024, 1, 1, 12)
    bdeck = [
        {"t": init - timedelta(hours=6), "lat": 0.0, "lon": -0.5,
         "vmax_kt": 40, "mslp_hpa": 1000, "rmw_km": None},
        {"t": init + timedelta(hours=6), "lat": 0.0, "lon": 0.5,
         "vmax_kt": 50, "mslp_hpa": 995, "rmw_km": None},
    ]
    expected = 111.1949266 / 12.0 / 1.852
    assert np.isclose(translation_speed_kt(bdeck, init), expected,
                      rtol=1e-6)


def test_disc_mean_includes_only_points_in_radius():
    lat, lon = np.meshgrid(np.array([0.0, 1.0]),
                           np.array([0.0, 1.0]), indexing="ij")
    data = np.array([[4.0, 100.0], [100.0, 100.0]])
    assert disc_mean(lat, lon, data, 0.0, 0.0, 80.0) == 4.0
    assert np.isnan(disc_mean(lat, lon, data, 10.0, 10.0, 20.0))


def test_extract_without_bdeck_or_parent_files_degrades_to_nan(tmp_path):
    init = datetime(2024, 9, 24, 0)

    class FakeCase(SimpleNamespace):
        def parent_glob(self):
            return f"**/*{self.init_str}*parent.atm.f*.grb2"

    ccase = SimpleNamespace(storm_name="Hurricane Test Storm",
                            model_label="HAFS-A")
    case = FakeCase(run_dir=tmp_path, init_dt=init,
                    init_str="2024092400", track_fixes=[])
    cycle = {"init_str": "2024092400", "init_dt": init}
    summary = {
        "lead_hours_to_landfall": 27.0,
        "ets_headline": 0.2,
        "fss_headline": 0.5,
        "rmse": 12.0,
        "bias_mm": -2.0,
        "obj_centroid_err_km": 80.0,
        "ets_shifted": 0.3,
    }
    row = extract_cycle_features(ccase, case, cycle, summary, None)
    assert row["storm"] == "hurricane_test_storm"
    assert row["model"] == "HAFS-A"
    assert row["init"] == "2024092400"
    assert row["lead_hours_to_landfall"] == 27.0
    assert all(row[key] == summary[key] for key in TARGET_COLUMNS)
    non_identity = set(row) - set(TARGET_COLUMNS) - {
        "storm", "model", "init", "lead_hours_to_landfall"}
    assert all(np.isnan(row[key]) for key in non_identity)


def test_module_import_does_not_require_eccodes():
    analysis_dir = Path(__file__).resolve().parents[1]
    code = (
        "import builtins,sys; "
        "real=builtins.__import__; "
        "builtins.__import__=lambda name,*a,**k: "
        "(_ for _ in ()).throw(ImportError('blocked eccodes')) "
        "if name in ('eccodes','cartopy') else real(name,*a,**k); "
        f"sys.path.insert(0,{str(analysis_dir)!r}); import ml_features"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
