import sys
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cycles_compare import (
    load_cycles_comparison, load_model_tables, pooled_ets, plot_ets_comparison,
    plot_fss_comparison, plot_rmse_comparison,
    trim_trailing_empty_ets_thresholds,
)


def test_load_cycles_comparison_accepts_three_models_without_data():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = root / "compare.yaml"
        cfg.write_text(
            "label: Test comparison\n"
            "models:\n"
            "  - {name: HAFS-A, cycles_yaml: a.yaml}\n"
            "  - {name: HAFS-B, cycles_yaml: b.yaml}\n"
            "  - {name: HAFS-M, cycles_yaml: m.yaml}\n"
        )
        loaded = load_cycles_comparison(cfg)
        assert [m["name"] for m in loaded["models"]] == [
            "HAFS-A", "HAFS-B", "HAFS-M"]
        assert loaded["ets_thresholds_in"] == list(range(2, 25, 2))
        assert loaded["fss_thresholds_in"] == [1.0, 2.0]


def test_pooled_ets_converts_inches_and_sums_cycles():
    rows = [
        {"init": "2024092400", "forecast": "parent", "observation": "MRMS",
         "threshold": "50.8", "a": "4", "b": "1", "c": "2", "d": "3"},
        {"init": "2024092406", "forecast": "parent", "observation": "MRMS",
         "threshold": "50.8", "a": "2", "b": "2", "c": "1", "d": "5"},
    ]
    result = pooled_ets(rows, [2])[0]
    assert result["n_cycles"] == 2
    assert abs(result["ets"] - 0.24528301886792447) < 1e-12


def test_trim_trailing_empty_ets_thresholds_keeps_last_active_value():
    rows = [
        {"init": "2024092400", "forecast": "parent", "observation": "MRMS",
         "threshold": str(threshold * 25.4),
         "a": "4" if threshold <= 4 else "1", "b": "1",
         "c": "2" if threshold <= 4 else "1",
         "d": "3" if threshold <= 4 else "1"}
        for threshold in (2, 4, 6, 8)
    ]
    models = [{"name": "HAFS-A", "categorical": rows}]
    assert trim_trailing_empty_ets_thresholds(models, [2, 4, 6, 8]) == [2, 4]


def test_comparison_plots_allow_missing_model():
    rows = [
        {"init": "2024092400", "forecast": "parent", "observation": "MRMS",
         "threshold": "50.8", "a": "4", "b": "1", "c": "2", "d": "3"}
    ]
    fss = [
        {"forecast": "parent", "observation": "MRMS", "threshold": "25.4",
         "threshold_in": "1", "scale_km": "5.6", "fss": "0.7"},
        {"forecast": "parent", "observation": "MRMS", "threshold": "25.4",
         "threshold_in": "1", "scale_km": "16.7", "fss": "0.8"},
    ]
    models = [
        {"name": "HAFS-A", "categorical": rows, "fss": fss},
        {"name": "HAFS-B", "categorical": rows, "fss": fss},
        {"name": "HAFS-M", "categorical": [], "fss": []},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        ets = Path(tmp) / "ets.png"
        fss_path = Path(tmp) / "fss.png"
        plot_ets_comparison(models, [2], "Test", ets)
        plot_fss_comparison(models, [1], "Test", fss_path)
        assert ets.exists() and ets.stat().st_size > 0
        assert fss_path.exists() and fss_path.stat().st_size > 0


def test_load_model_tables_parses_summary_numeric_fields(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = SimpleNamespace(out_dir=root, output_slug="sample")
        monkeypatch.setattr("cycles_compare.cycles_from_yaml", lambda _: case)
        path = root / "cycles_summary_sample.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["init", "init_dt", "ets_headline",
                                "mean_track_err_km", "ets_shifted"])
            writer.writeheader()
            writer.writerow({"init": "2024092400",
                             "init_dt": "2024-09-24 00:00:00",
                             "ets_headline": "0.25",
                             "mean_track_err_km": "42.5",
                             "ets_shifted": ""})
        loaded = load_model_tables({"name": "HAFS-A",
                                    "cycles_yaml": root / "case.yaml"})
        assert loaded["summary"][0]["init"] == "2024092400"
        assert loaded["summary"][0]["ets_headline"] == 0.25
        assert loaded["summary"][0]["mean_track_err_km"] == 42.5
        assert np.isnan(loaded["summary"][0]["ets_shifted"])
        path.unlink()
        missing = load_model_tables({"name": "HAFS-A",
                                     "cycles_yaml": root / "case.yaml"})
        assert missing["summary"] == []


def test_plot_rmse_comparison_writes_inches_plot():
    models = [
        {"name": "HAFS-A", "summary": [
            {"lead_hours_to_landfall": 48.0, "rmse": 25.4},
            {"lead_hours_to_landfall": 24.0, "rmse": 50.8},
        ]},
        {"name": "HAFS-B", "summary": [
            {"lead_hours_to_landfall": 48.0, "rmse": 38.1},
            {"lead_hours_to_landfall": 24.0, "rmse": 63.5},
        ]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "rmse.png"
        assert plot_rmse_comparison(models, "Test", output) is True
        assert output.exists() and output.stat().st_size > 0
