"""HFSA-vs-HFSB head-to-head comparison over a shared best-track swath.

Loaded via run.py:  python analysis/run.py storms/<name>_compare.yaml compare
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from ets_full import score_pair
from ets_score import contingency_scores
from skill_metrics import fractions_skill_score
from hafs_case import from_yaml
from best_track import parse_bdeck
from skill_metrics import swath_from_track
from hafs_common import discover_files, hafs_event_total
from ets_full import hafs_parent_total, stage4_on_fixed
from ets_score import build_mrms_total

_DEFAULT_FSS_SCALES = [1, 3, 5, 11, 21, 41]
_DEFAULT_FSS_PLOT_THR = [10, 25, 50]


def load_comparison(path):
    """Parse + validate a comparison YAML and fill defaults (no GRIB loading)."""
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    cases = cfg.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError(f"'cases' must list exactly 2 case YAMLs in {path}")
    if "best_track" not in cfg:
        raise KeyError(f"'best_track' is required in {path}")
    return {
        "label": cfg.get("label", path.stem),
        "case_paths": [str(c) for c in cases],
        "best_track": str(cfg["best_track"]),
        "out_dir": Path(cfg["out_dir"]) if cfg.get("out_dir")
                   else Path("analysis/output") / path.stem,
        "thresholds_mm": cfg.get("thresholds_mm"),
        "fss_scales_cells": cfg.get("fss_scales_cells", list(_DEFAULT_FSS_SCALES)),
        "fss_plot_thresholds": cfg.get("fss_plot_thresholds",
                                       list(_DEFAULT_FSS_PLOT_THR)),
    }


def score_matrix(models, swath, thresholds, fss_scales, grid_res):
    """Score every model x forecast x obs over the shared swath.

    Returns (cat_rows, fss_rows). None observations are skipped.
    """
    cat_rows, fss_rows = [], []
    for m in models:
        for fname, fgrid in m["forecasts"].items():
            for oname, ogrid in m["obs"].items():
                if ogrid is None:
                    continue
                rows, _ = score_pair(fgrid, ogrid, swath, thresholds,
                                     contingency_scores)
                for r in rows:
                    cat_rows.append({"model": m["name"], "forecast": fname,
                                     "observation": oname, **r})
                vmask = swath & np.isfinite(fgrid) & np.isfinite(ogrid)
                ff = np.nan_to_num(fgrid, nan=0.0)
                oo = np.nan_to_num(ogrid, nan=0.0)
                for thr in thresholds:
                    for sc in fss_scales:
                        fss_rows.append({
                            "model": m["name"], "forecast": fname,
                            "observation": oname, "threshold": thr,
                            "scale_cells": sc,
                            "scale_km": round(sc * grid_res * 111.0, 1),
                            "fss": fractions_skill_score(ff, oo, thr, sc, vmask),
                        })
    return cat_rows, fss_rows


_MODEL_COLOR = {"HFSA": "#1f77b4", "HFSB": "#d62728"}
_FCST_STYLE = {"parent": dict(ls="-", marker="o"),
               "nest": dict(ls="--", marker="s")}


def plot_categorical_compare(cat_rows, label, out_path, observation="MRMS"):
    """3 panels (ETS, CSI, freq bias) vs threshold; HFSA/HFSB x parent/nest."""
    rows = [r for r in cat_rows if r["observation"] == observation]
    metrics = [("ets", "Equitable Threat Score"),
               ("csi", "Critical Success Index"),
               ("bias", "Frequency bias")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models = sorted({r["model"] for r in rows})
    forecasts = sorted({r["forecast"] for r in rows})
    for ax, (key, title) in zip(axes, metrics):
        for mdl in models:
            for fc in forecasts:
                sub = sorted((r for r in rows
                              if r["model"] == mdl and r["forecast"] == fc),
                             key=lambda r: r["threshold"])
                if not sub:
                    continue
                ax.plot([r["threshold"] for r in sub], [r[key] for r in sub],
                        color=_MODEL_COLOR.get(mdl, "gray"),
                        **_FCST_STYLE.get(fc, dict(ls="-", marker="o")),
                        lw=2, label=f"{mdl} {fc}")
        ax.set_xscale("log")
        ax.set_xlabel("Rainfall threshold (mm)")
        ax.set_ylabel(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if key == "bias":
            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        else:
            ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{label} — HFSA vs HFSB categorical skill (vs {observation})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fss_compare(fss_rows, label, out_path, observation="MRMS",
                     forecast="parent", plot_thresholds=(10, 25, 50)):
    """FSS vs neighborhood scale (km); one line per (model, threshold)."""
    rows = [r for r in fss_rows if r["observation"] == observation
            and r["forecast"] == forecast and r["threshold"] in plot_thresholds]
    fig, ax = plt.subplots(figsize=(9, 6))
    models = sorted({r["model"] for r in rows})
    thrs = sorted({r["threshold"] for r in rows})
    dashes = {t: (None if i == 0 else (4 + 2 * i, 2))
              for i, t in enumerate(thrs)}
    for mdl in models:
        for t in thrs:
            sub = sorted((r for r in rows
                          if r["model"] == mdl and r["threshold"] == t),
                         key=lambda r: r["scale_km"])
            if not sub:
                continue
            line, = ax.plot([r["scale_km"] for r in sub],
                            [r["fss"] for r in sub],
                            color=_MODEL_COLOR.get(mdl, "gray"),
                            lw=2, marker="o",
                            label=f"{mdl}  {int(t)} mm")
            if dashes[t] is not None:
                line.set_dashes(dashes[t])
    ax.set_xlabel("Neighborhood scale (km)")
    ax.set_ylabel("Fractions Skill Score (FSS)")
    ax.set_ylim(0, 1)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{label} — HFSA vs HFSB FSS ({forecast} vs {observation})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _slug(label):
    return label.lower().replace(" ", "_")


def _build_model_fields(case, grid_lat, grid_lon, max_fhour):
    """Parent + nest forecast totals and MRMS + Stage IV obs on the common grid."""
    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    print(f"  {case.model_label}: {len(file_pairs)} storm files, nest total ...")
    nest_total, _ = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  {case.model_label}: parent total ...")
    parent_total = hafs_parent_total(case, grid_lat, grid_lon)
    print(f"  {case.model_label}: MRMS total ...")
    mrms = build_mrms_total(case, max_fhour, grid_lat, grid_lon)
    print(f"  {case.model_label}: Stage IV total ...")
    stage4, _ = stage4_on_fixed(case, max_fhour, grid_lat, grid_lon)
    return {
        "name": case.model_label,
        "forecasts": {"parent": parent_total, "nest": nest_total},
        "obs": {"MRMS": mrms, "Stage IV": stage4},
    }


def generate_comparison(cfg):
    """Score both cases over one best-track swath and write figures + CSVs."""
    cases = [from_yaml(p) for p in cfg["case_paths"]]
    a, b = cases
    if a.domain != b.domain or a.grid_res != b.grid_res:
        raise ValueError(
            f"cases must share domain/grid_res: {cfg['case_paths'][0]} has "
            f"{a.domain}@{a.grid_res}, {cfg['case_paths'][1]} has {b.domain}@{b.grid_res}")
    if a.mask_radius_km != b.mask_radius_km:
        raise ValueError(
            f"cases must share mask_radius_km ({a.mask_radius_km} vs {b.mask_radius_km})")
    thresholds = cfg["thresholds_mm"] or a.thresholds_mm
    grid_lat, grid_lon = a.fixed_grid()

    print(f"Best track: {cfg['best_track']}")
    track = parse_bdeck(cfg["best_track"])

    fa = discover_files(a.run_dir, a.storm_glob(), a.fhours_filter)
    fb = discover_files(b.run_dir, b.storm_glob(), b.fhours_filter)
    if not fa or not fb:
        print("No storm files found for one of the cases; aborting.")
        return
    max_fhour = min(fa[-1][0], fb[-1][0])
    print(f"Shared swath: best track, 0-{max_fhour}h, "
          f"<= {a.mask_radius_km:.0f} km")
    swath = swath_from_track(track, grid_lat, grid_lon, a.mask_radius_km,
                             a.init_dt, max_fhour)

    models = [_build_model_fields(c, grid_lat, grid_lon, max_fhour) for c in cases]
    cat_rows, fss_rows = score_matrix(models, swath, thresholds,
                                      cfg["fss_scales_cells"], a.grid_res)

    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(cfg["label"])

    cat_cols = ["model", "forecast", "observation", "threshold",
                "a", "b", "c", "d", "ets", "csi", "bias", "pod", "far", "hss"]
    cat_csv = out_dir / f"compare_categorical_{slug}.csv"
    with open(cat_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cat_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(cat_rows)

    fss_cols = ["model", "forecast", "observation", "threshold",
                "scale_cells", "scale_km", "fss"]
    fss_csv = out_dir / f"compare_fss_{slug}.csv"
    with open(fss_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fss_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(fss_rows)

    cat_png = out_dir / f"compare_categorical_{slug}.png"
    fss_png = out_dir / f"compare_fss_{slug}.png"
    plot_categorical_compare(cat_rows, cfg["label"], cat_png, observation="MRMS")
    plot_fss_compare(fss_rows, cfg["label"], fss_png, observation="MRMS",
                     forecast="parent",
                     plot_thresholds=tuple(cfg["fss_plot_thresholds"]))

    print(f"\nSaved: {cat_csv}")
    print(f"Saved: {fss_csv}")
    print(f"Saved: {cat_png}")
    print(f"Saved: {fss_png}")
    print("\nETS (parent vs MRMS) at 25 / 50 mm:")
    for mdl in sorted({r["model"] for r in cat_rows}):
        e25 = next((r["ets"] for r in cat_rows if r["model"] == mdl
                    and r["forecast"] == "parent" and r["observation"] == "MRMS"
                    and r["threshold"] == 25), float("nan"))
        e50 = next((r["ets"] for r in cat_rows if r["model"] == mdl
                    and r["forecast"] == "parent" and r["observation"] == "MRMS"
                    and r["threshold"] == 50), float("nan"))
        print(f"  {mdl}: ETS25={e25:.3f}  ETS50={e50:.3f}")
