"""Compare multi-cycle verification tables across HAFS configurations.

The comparison is intentionally table-driven: run each available model with
the ``cycles`` command first, then use ``cycles-compare`` to combine its CSVs.
Models without output yet remain visible as "awaiting data" placeholders.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml

from hafs_case import cycles_from_yaml


MODEL_COLORS = {
    "HAFS-A": "#2563a6",
    "HAFS-B": "#dd6b4d",
    "HAFS-M": "#2a9d78",
}


def _resolve_path(value, config_path):
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return config_path.parent / path


def load_cycles_comparison(path):
    """Load model-cycle comparison metadata without requiring model data."""
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    models = raw.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError(f"'models' must contain at least two entries in {path}")
    parsed = []
    for item in models:
        if not isinstance(item, dict) or "name" not in item or "cycles_yaml" not in item:
            raise ValueError("Each model needs 'name' and 'cycles_yaml'.")
        parsed.append({"name": str(item["name"]),
                       "cycles_yaml": _resolve_path(item["cycles_yaml"], path)})
    out_dir = Path(raw.get("out_dir", "analysis/output/helene_cycles_compare"))
    thresholds = [float(v) for v in raw.get(
        "ets_thresholds_in", list(range(2, 25, 2)))]
    fss_thresholds = [float(v) for v in raw.get("fss_thresholds_in", [1, 2])]
    return {"label": raw.get("label", path.stem), "models": parsed,
            "out_dir": out_dir, "ets_thresholds_in": thresholds,
            "fss_thresholds_in": fss_thresholds}


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_model_tables(model):
    """Return categorical/FSS rows and paths for one configured model."""
    try:
        case = cycles_from_yaml(model["cycles_yaml"])
    except (OSError, KeyError, ValueError) as exc:
        return {**model, "case": None, "categorical": [], "fss": [],
                "status": f"configuration unavailable: {exc}"}
    slug = case.output_slug
    cat_path = case.out_dir / f"cycles_{slug}.csv"
    fss_path = case.out_dir / f"cycles_fss_{slug}.csv"
    categorical = _read_csv(cat_path)
    fss = _read_csv(fss_path)
    status = "available" if categorical and fss else "awaiting cycle output"
    return {**model, "case": case, "categorical": categorical, "fss": fss,
            "cat_path": cat_path, "fss_path": fss_path, "status": status}


def pooled_ets(rows, thresholds_in):
    """Pool MRMS parent contingency counts across cycles at inch thresholds."""
    selected = [r for r in rows if r.get("forecast") == "parent"
                and r.get("observation") == "MRMS"]
    output = []
    for threshold_in in thresholds_in:
        threshold_mm = threshold_in * 25.4
        matched = [r for r in selected
                   if np.isclose(float(r["threshold"]), threshold_mm)]
        counts = {key: sum(int(float(r[key])) for r in matched)
                  for key in ("a", "b", "c", "d")}
        n = sum(counts.values())
        random_hits = (((counts["a"] + counts["b"])
                        * (counts["a"] + counts["c"]) / n) if n else 0.0)
        denominator = counts["a"] + counts["b"] + counts["c"] - random_hits
        score = ((counts["a"] - random_hits) / denominator
                 if denominator else np.nan)
        output.append({"threshold_in": threshold_in, "ets": score,
                       "n_cycles": len({r["init"] for r in matched})})
    return output


def plot_ets_comparison(models, thresholds_in, label, out_path):
    """Tasteful grouped ETS bars for all configured models."""
    x = np.arange(len(thresholds_in))
    width = min(0.24, 0.78 / max(len(models), 1))
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    fig.patch.set_facecolor("#f7f9fb")
    ax.set_facecolor("#f7f9fb")
    missing = []
    for index, model in enumerate(models):
        summary = pooled_ets(model["categorical"], thresholds_in)
        values = np.asarray([row["ets"] for row in summary], dtype=float)
        offset = (index - (len(models) - 1) / 2) * width
        color = MODEL_COLORS.get(model["name"], plt.cm.Set2(index))
        if np.isfinite(values).any():
            ax.bar(x + offset, np.nan_to_num(values), width=width * 0.90,
                   color=color, edgecolor="white", linewidth=0.8,
                   label=model["name"], zorder=3)
        else:
            missing.append(model["name"])
    ax.axhline(0, color="#607080", linewidth=0.8)
    ax.set_xticks(x, [f"{value:g}" for value in thresholds_in])
    ax.set_xlabel("Rainfall accumulation threshold (inches)", labelpad=10)
    ax.set_ylabel("Equitable Threat Score (ETS)")
    ax.grid(axis="y", color="#d8e0e8", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=MODEL_COLORS.get(model["name"], "gray"),
                     alpha=1.0 if model["name"] not in missing else 0.35,
                     label=(model["name"] if model["name"] not in missing
                            else f"{model['name']} · awaiting data"))
               for model in models]
    ax.legend(handles=handles, frameon=False, ncols=len(models),
              loc="upper right")
    ax.set_title(f"{label}\nPooled rainfall skill across forecast cycles",
                 loc="left", fontsize=16, fontweight="semibold", pad=16)
    ax.text(1, 1.015, "Higher is better", transform=ax.transAxes,
            ha="right", color="#607080", fontsize=9)
    if missing:
        ax.text(0.01, 0.02, "Awaiting output: " + ", ".join(missing),
                transform=ax.transAxes, fontsize=9, color="#607080")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _fss_series(rows, threshold_in):
    selected = []
    for row in rows:
        if row.get("forecast") != "parent" or row.get("observation") != "MRMS":
            continue
        value_in = float(row.get("threshold_in") or float(row["threshold"]) / 25.4)
        if np.isclose(value_in, threshold_in):
            selected.append(row)
    scales = sorted({float(r["scale_km"]) for r in selected})
    output = []
    for scale in scales:
        values = np.asarray([float(r["fss"]) for r in selected
                             if np.isclose(float(r["scale_km"]), scale)])
        values = values[np.isfinite(values)]
        if values.size:
            output.append((scale, float(values.mean()),
                           float(np.percentile(values, 25)),
                           float(np.percentile(values, 75))))
    return output


def plot_fss_comparison(models, thresholds_in, label, out_path):
    """Mean cycle FSS by scale with an interquartile band per model."""
    fig, axes = plt.subplots(1, len(thresholds_in),
                             figsize=(6.5 * len(thresholds_in), 5.8),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor("#f7f9fb")
    for ax, threshold in zip(axes[0], thresholds_in):
        ax.set_facecolor("#f7f9fb")
        have_data = False
        for index, model in enumerate(models):
            series = _fss_series(model["fss"], threshold)
            if not series:
                continue
            have_data = True
            scale, mean, q25, q75 = map(np.asarray, zip(*series))
            color = MODEL_COLORS.get(model["name"], plt.cm.Set2(index))
            ax.fill_between(scale, q25, q75, color=color, alpha=0.14,
                            linewidth=0)
            ax.plot(scale, mean, color=color, linewidth=2.5, marker="o",
                    markersize=5.5, markeredgecolor="white",
                    label=model["name"])
        unit = "inch" if np.isclose(threshold, 1.0) else "inches"
        ax.set_title(f"Rainfall ≥ {threshold:g} {unit}",
                     fontweight="semibold", fontsize=13)
        ax.set_xlabel("Neighborhood scale (km)")
        ax.set_ylim(0, 1)
        ax.grid(color="#d8e0e8", linewidth=0.8, alpha=0.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if not have_data:
            ax.text(0.5, 0.5, "Awaiting cycle output", transform=ax.transAxes,
                    ha="center", color="#607080")
    axes[0, 0].set_ylabel("Fractions Skill Score (FSS)")
    handles = [plt.Line2D([], [], color=MODEL_COLORS.get(m["name"], "gray"),
                          marker="o", lw=2.5,
                          label=(m["name"] if m["fss"] else
                                 f"{m['name']} · awaiting data"))
               for m in models]
    fig.legend(handles=handles, frameon=False, ncols=len(models),
               loc="upper center", bbox_to_anchor=(0.64, 0.85))
    fig.suptitle(f"{label}\nMean FSS across cycles · shading shows interquartile range",
                 x=0.06, y=0.985, ha="left", fontsize=16,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.74))
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_cycles_comparison(config):
    """Load available cycle tables and write the comparison figures."""
    models = [load_model_tables(model) for model in config["models"]]
    config["out_dir"].mkdir(parents=True, exist_ok=True)
    ets_path = config["out_dir"] / "cycles_compare_ets_bars.png"
    fss_path = config["out_dir"] / "cycles_compare_fss.png"
    plot_ets_comparison(models, config["ets_thresholds_in"],
                        config["label"], ets_path)
    plot_fss_comparison(models, config["fss_thresholds_in"],
                        config["label"], fss_path)
    for model in models:
        print(f"{model['name']}: {model['status']}")
    print(f"Saved plot : {ets_path}")
    print(f"Saved plot : {fss_path}")
    return models
