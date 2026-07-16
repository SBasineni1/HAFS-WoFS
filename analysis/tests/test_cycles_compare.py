import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cycles_compare import (
    load_cycles_comparison, pooled_ets, plot_ets_comparison,
    plot_fss_comparison,
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
