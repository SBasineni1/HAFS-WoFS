"""Pooled machine-learning regime diagnostics for cycle verification."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ml_features import FEATURE_COLUMNS, TARGET_COLUMNS


_IDENTITY_COLUMNS = ("storm", "model", "init")
_DEFAULTS = {
    "features_csv": Path("analysis/output/ml_features.csv"),
    "out_dir": Path("analysis/output/ml_regime"),
    "targets": ["ets_headline", "fss_headline", "rmse"],
    "min_samples_warn": 30,
}


class _FeatureMatrix(np.ndarray):
    """Numeric matrix carrying row identities for diagnostic plot labels."""

    def __new__(cls, values, identities):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.identities = identities
        return obj

    def __array_finalize__(self, obj):
        self.identities = getattr(obj, "identities", None)


def load_ml_config(yaml_path):
    """Load an ML-regime YAML and fill dependency-light defaults."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh) or {}
    return {
        "features_csv": Path(raw.get("features_csv",
                                      _DEFAULTS["features_csv"])),
        "out_dir": Path(raw.get("out_dir", _DEFAULTS["out_dir"])),
        "targets": list(raw.get("targets", _DEFAULTS["targets"])),
        "min_samples_warn": int(raw.get(
            "min_samples_warn", _DEFAULTS["min_samples_warn"])),
    }


def _numeric(value):
    if value is None or str(value).strip() == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_features(csv_path):
    """Read pooled features as dictionaries plus a numeric feature matrix."""
    csv_path = Path(csv_path)
    with open(csv_path, newline="") as fh:
        source_rows = list(csv.DictReader(fh))

    numeric_columns = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS)
    rows = []
    for source in source_rows:
        row = dict(source)
        for name in numeric_columns - set(_IDENTITY_COLUMNS):
            row[name] = _numeric(source.get(name, ""))
        rows.append(row)

    feature_names = [name for name in FEATURE_COLUMNS
                     if name not in _IDENTITY_COLUMNS
                     and name not in TARGET_COLUMNS]
    values = [[row[name] for name in feature_names] for row in rows]
    identities = [{name: row.get(name, "") for name in _IDENTITY_COLUMNS}
                  for row in rows]
    return rows, _FeatureMatrix(values, identities), feature_names


def _sklearn_tools():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import (PartialDependenceDisplay,
                                        permutation_importance)
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import LeaveOneOut
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ML regime diagnostics; "
            "pip install scikit-learn") from exc
    return (HistGradientBoostingRegressor, PartialDependenceDisplay,
            permutation_importance, mean_absolute_error, r2_score,
            LeaveOneOut)


def _model(regressor):
    return regressor(max_iter=200, max_depth=3, l2_regularization=1.0,
                     min_samples_leaf=2, random_state=0)


def _stamp_exploratory(fig, n):
    fig.text(0.5, 0.01,
             f"EXPLORATORY — n={n} cycles; treat rankings qualitatively",
             ha="center", va="bottom", color="red", weight="bold",
             fontsize=10)


def _save_figure(fig, path, exploratory_n=None):
    if exploratory_n is not None:
        _stamp_exploratory(fig, exploratory_n)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def train_target(X, y, feature_names, target, out_dir, min_samples_warn):
    """Fit and diagnose one target using leave-one-out cross-validation."""
    (regressor, partial_dependence, permutation_importance,
     mean_absolute_error, r2_score, leave_one_out) = _sklearn_tools()

    identities = getattr(X, "identities", None)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(y)
    X_fit, y_fit = X[keep], y[keep]
    if identities is not None:
        identities = [identity for identity, wanted in zip(identities, keep)
                      if wanted]
    n = len(y_fit)
    if n < 8:
        print(f"insufficient samples for {target} (n={n}); skipping")
        return None

    exploratory_n = n if n < min_samples_warn else None
    if exploratory_n is not None:
        print("=" * 72)
        print(f"EXPLORATORY — n={n} cycles; treat rankings qualitatively")
        print("=" * 72)

    predictions = np.full(n, np.nan)
    for train_index, test_index in leave_one_out().split(X_fit):
        fold_model = _model(regressor)
        fold_model.fit(X_fit[train_index], y_fit[train_index])
        predictions[test_index] = fold_model.predict(X_fit[test_index])
    cv_r2 = float(r2_score(y_fit, predictions))
    cv_mae = float(mean_absolute_error(y_fit, predictions))

    fitted = _model(regressor)
    fitted.fit(X_fit, y_fit)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    importance_path = out_dir / f"ml_importance_{target}.png"
    try:
        import shap
    except ImportError:
        result = permutation_importance(
            fitted, X_fit, y_fit, n_repeats=30, random_state=0)
        importance = np.asarray(result.importances_mean)
        order = np.argsort(importance)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(np.asarray(feature_names)[order], importance[order],
                color="#2563a6")
        ax.set_xlabel("Decrease in model score")
        ax.set_title(f"{target} (permutation importance — shap not installed)")
        fig.tight_layout()
        _save_figure(fig, importance_path, exploratory_n)
    else:
        explainer = shap.Explainer(fitted, X_fit,
                                   feature_names=feature_names)
        shap_values = explainer(X_fit)
        importance = np.nanmean(np.abs(shap_values.values), axis=0)
        shap.summary_plot(shap_values.values, X_fit,
                          feature_names=feature_names, show=False)
        fig = plt.gcf()
        plt.title(f"SHAP importance — {target}")
        fig.tight_layout()
        _save_figure(fig, importance_path, exploratory_n)

    ranked = np.argsort(np.nan_to_num(importance, nan=-np.inf))[::-1]
    usable = np.any(np.isfinite(X_fit), axis=0)
    ranked = [index for index in ranked if usable[index]]
    top_indices = ranked[:min(3, len(ranked))]
    top_features = [feature_names[index] for index in top_indices]

    if top_indices:
        display = partial_dependence.from_estimator(
            fitted, X_fit, features=list(top_indices),
            feature_names=feature_names)
        display.figure_.suptitle(f"Partial dependence — {target}", y=1.03)
        display.figure_.tight_layout()
        pdp_figure = display.figure_
    else:
        pdp_figure, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No finite feature values available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(f"Partial dependence — {target}")
    _save_figure(pdp_figure, out_dir / f"ml_pdp_{target}.png",
                 exploratory_n)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    models = ([str(identity.get("model", "")) for identity in identities]
              if identities is not None else [])
    named_models = sorted(set(model for model in models if model))
    if named_models:
        for model_name in named_models:
            selected = np.asarray([model == model_name for model in models])
            ax.scatter(y_fit[selected], predictions[selected], s=38, alpha=0.8,
                       label=model_name)
        unnamed = np.asarray([not model for model in models])
        if np.any(unnamed):
            ax.scatter(y_fit[unnamed], predictions[unnamed], s=38, alpha=0.8,
                       label="unlabeled")
        ax.legend(title="Model")
    else:
        ax.scatter(y_fit, predictions, s=38, alpha=0.8, color="#2563a6")
    limits = [float(np.nanmin([y_fit, predictions])),
              float(np.nanmax([y_fit, predictions]))]
    if limits[0] == limits[1]:
        limits = [limits[0] - 0.5, limits[1] + 0.5]
    ax.plot(limits, limits, "k--", linewidth=1.2, label="1:1")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Actual")
    ax.set_ylabel("LOO prediction")
    ax.set_title(f"Predicted vs actual — {target}")
    ax.text(0.04, 0.96, f"CV R² = {cv_r2:.3f}\nCV MAE = {cv_mae:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            bbox={"facecolor": "white", "alpha": 0.85,
                  "edgecolor": "0.7"})
    fig.tight_layout()
    _save_figure(fig, out_dir / f"ml_pred_vs_actual_{target}.png",
                 exploratory_n)

    return {"target": target, "n": n, "cv_r2": cv_r2,
            "cv_mae": cv_mae, "top_features": "|".join(top_features)}


def run_ml(cfg):
    """Run pooled diagnostics for all configured targets."""
    features_csv = Path(cfg["features_csv"])
    if not features_csv.exists():
        print(f"ML features CSV not found: {features_csv}")
        print("Run cycle verification to create ml_features.csv, then retry.")
        return []

    rows, X, feature_names = load_features(features_csv)
    summaries = []
    for target in cfg["targets"]:
        if target not in TARGET_COLUMNS:
            print(f"unknown ML target {target!r}; skipping")
            continue
        y = np.asarray([row.get(target, np.nan) for row in rows], dtype=float)
        summary = train_target(X, y, feature_names, target, cfg["out_dir"],
                               cfg["min_samples_warn"])
        if summary is not None:
            summaries.append(summary)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ml_summary.csv"
    fields = ["target", "n", "cv_r2", "cv_mae", "top_features"]
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"ML regime outputs written to {out_dir}")
    return summaries
