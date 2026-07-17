import builtins
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_features import FEATURE_COLUMNS, TARGET_COLUMNS
from ml_regime import load_ml_config, run_ml, train_target


def _block_shap(monkeypatch):
    real_import = builtins.__import__

    def import_without_shap(name, *args, **kwargs):
        if name == "shap" or name.startswith("shap."):
            raise ImportError("shap blocked for fallback test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_shap)


def _synthetic_data(n=40):
    rng = np.random.default_rng(4)
    x1 = rng.normal(size=n)
    X = np.column_stack([x1, rng.normal(size=(n, 3))])
    y = 3.0 * x1 + rng.normal(scale=0.25, size=n)
    return X, y


def _write_features(path, n=40):
    X, y = _synthetic_data(n)
    rng = np.random.default_rng(7)
    feature_names = [name for name in FEATURE_COLUMNS
                     if name not in ("storm", "model", "init")]
    fields = FEATURE_COLUMNS + TARGET_COLUMNS
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for index in range(n):
            row = {name: "" for name in fields}
            row.update({"storm": "test", "model": "A" if index % 2 else "B",
                        "init": f"202401{index + 1:02d}00"})
            row[feature_names[0]] = X[index, 0]
            for offset, name in enumerate(feature_names[1:4], start=1):
                row[name] = X[index, offset]
            for name in feature_names[4:]:
                row[name] = rng.normal()
            row["ets_headline"] = y[index]
            writer.writerow(row)


def test_train_target_permutation_importance(tmp_path, monkeypatch):
    _block_shap(monkeypatch)
    X, y = _synthetic_data()
    names = ["x1", "noise1", "noise2", "noise3"]
    summary = train_target(X, y, names, "synthetic", tmp_path, 30)

    assert summary["n"] == 40
    assert summary["cv_r2"] > 0.3
    assert summary["top_features"].split("|")[0] == "x1"
    assert (tmp_path / "ml_importance_synthetic.png").exists()
    assert (tmp_path / "ml_pdp_synthetic.png").exists()
    assert (tmp_path / "ml_pred_vs_actual_synthetic.png").exists()


def test_train_target_insufficient_samples(tmp_path, capsys):
    X, y = _synthetic_data(5)
    result = train_target(X, y, ["x1", "n1", "n2", "n3"],
                          "tiny", tmp_path, 30)
    assert result is None
    assert ("insufficient samples for tiny (n=5); skipping"
            in capsys.readouterr().out)
    assert not list(tmp_path.glob("*tiny*"))


def test_run_ml_writes_summary(tmp_path, monkeypatch):
    _block_shap(monkeypatch)
    features = tmp_path / "features.csv"
    output = tmp_path / "output"
    _write_features(features)

    summaries = run_ml({"features_csv": features, "out_dir": output,
                        "targets": ["ets_headline"],
                        "min_samples_warn": 30})
    assert len(summaries) == 1
    with open(output / "ml_summary.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["target"] == "ets_headline"


def test_load_ml_config_defaults(tmp_path):
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text("{}\n")
    cfg = load_ml_config(config_path)
    assert cfg == {
        "features_csv": Path("analysis/output/ml_features.csv"),
        "out_dir": Path("analysis/output/ml_regime"),
        "targets": ["ets_headline", "fss_headline", "rmse"],
        "min_samples_warn": 30,
    }
