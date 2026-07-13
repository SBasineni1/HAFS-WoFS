"""Paper-style tropical-cyclone QPF verification.

Implements the reusable figure families in Newman et al. (2024): categorical
skill by lead time with and without track shifting, multi-threshold ETS and
frequency bias, storm-relative composites, radial distributions, connected
precipitation objects, and object intensity distributions.  ``paper`` YAMLs
may describe one storm with several models or a suite of storm YAMLs.

The native observation provider is MRMS because it is already available in
this repository.  This is the land-focused analogue of the paper's
CCPA-over-land verification.  The output metadata states MRMS explicitly; it
does not claim to reproduce the paper's IMERG ocean sample.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, label as nd_label, shift as nd_shift

from best_track import parse_bdeck, parse_bdeck_fixes
from cycles import nest_window_total, parent_window_total
from ets_score import build_mrms_total_window, contingency_scores
from hafs_case import (
    StormCase, decode_latlon, detect_model, discover_inits,
    find_atcfunix, make_fixed_grid, parse_atcfunix, position_on_track,
)
from hafs_common import QPF_COLORS, QPF_LEVELS, haversine_km
from paper_case import PaperStormCase, PaperSuiteCase, load_paper_config, load_paper_storm
from skill_metrics import continuous_scores


MODEL_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]


def _model_colors(names):
    return {name: MODEL_COLORS[i % len(MODEL_COLORS)]
            for i, name in enumerate(sorted(names))}


def _atcf_rmw(cols):
    try:
        value = float(cols[19])
        return value * 1.852 if value > 0 else None
    except (IndexError, TypeError, ValueError):
        return None


def parse_atcfunix_fixes(path):
    """Forecast fixes as (valid time, lat, lon, RMW km or None)."""
    by_tau = {}
    with open(path) as fh:
        for line in fh:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 8:
                continue
            try:
                warn = datetime.strptime(cols[2], "%Y%m%d%H")
                tau = int(cols[5])
                lat = decode_latlon(cols[6])
                lon = decode_latlon(cols[7])
            except (ValueError, IndexError):
                continue
            fix = (warn + timedelta(hours=tau), lat, lon, _atcf_rmw(cols))
            if tau not in by_tau or (by_tau[tau][3] is None and fix[3] is not None):
                by_tau[tau] = fix
    return [by_tau[t] for t in sorted(by_tau)]


def interpolate_fix(fixes, valid_time, fallback_rmw_km):
    """Linearly interpolate position/RMW; fall back only when RMW is absent."""
    basic = [(t, lat, lon) for t, lat, lon, _ in fixes]
    lat, lon = position_on_track(basic, valid_time)
    available = [(t, rmw) for t, _, _, rmw in fixes if rmw is not None]
    if not available:
        return lat, lon, float(fallback_rmw_km), True
    if valid_time <= available[0][0]:
        return lat, lon, available[0][1], False
    if valid_time >= available[-1][0]:
        return lat, lon, available[-1][1], False
    for (t0, r0), (t1, r1) in zip(available, available[1:]):
        if t0 <= valid_time <= t1:
            frac = (valid_time - t0).total_seconds() / (t1 - t0).total_seconds()
            return lat, lon, r0 + frac * (r1 - r0), False
    return lat, lon, available[-1][1], False


def shift_to_best(field, forecast_position, best_position, grid_res):
    """Translate a regular lat/lon field from forecast center to best track."""
    flat, flon = forecast_position
    blat, blon = best_position
    row_shift = (blat - flat) / grid_res
    col_shift = (blon - flon) / grid_res
    return nd_shift(np.asarray(field, float), (row_shift, col_shift), order=1,
                    mode="constant", cval=np.nan, prefilter=False)


def storm_relative_field(field, grid_lat, grid_lon, center, rmw_km,
                         radius_rmw, resolution_rmw):
    """Interpolate a lat/lon field to an RMW-normalized Cartesian grid."""
    axis = np.arange(-radius_rmw, radius_rmw + resolution_rmw / 2,
                     resolution_rmw)
    x, y = np.meshgrid(axis, axis)
    clat, clon = center
    qlat = clat + y * rmw_km / 111.0
    coslat = max(math.cos(math.radians(clat)), 0.1)
    qlon = clon + x * rmw_km / (111.0 * coslat)
    lat_axis = np.asarray(grid_lat[:, 0], float)
    lon_axis = np.asarray(grid_lon[0, :], float)
    values = np.asarray(field, float)
    if lat_axis[0] > lat_axis[-1]:
        lat_axis, values = lat_axis[::-1], values[::-1, :]
    if lon_axis[0] > lon_axis[-1]:
        lon_axis, values = lon_axis[::-1], values[:, ::-1]
    interp = RegularGridInterpolator((lat_axis, lon_axis), values,
                                     bounds_error=False, fill_value=np.nan)
    out = interp(np.column_stack([qlat.ravel(), qlon.ravel()])).reshape(x.shape)
    out[np.hypot(x, y) > radius_rmw] = np.nan
    return x, y, out


def _model_inits(run_root):
    root = Path(run_root)
    try:
        found = discover_inits(root)
    except FileNotFoundError:
        return []
    if found:
        return found
    values = set()
    for path in root.glob("**/*.atcfunix"):
        match = re.search(r"(20\d{8})", path.name)
        if match:
            values.add(match.group(1))
        else:
            _, init, _ = parse_atcfunix(path)
            if init:
                values.add(init.strftime("%Y%m%d%H"))
    return sorted(values)


def common_inits(case: PaperStormCase):
    """Event-equalized initialization intersection across every model."""
    available = {m.name: set(_model_inits(m.run_root)) for m in case.models}
    if case.inits:
        desired = set(case.inits)
        missing = {name: sorted(desired - values) for name, values in available.items()
                   if desired - values}
        if missing:
            detail = "; ".join(f"{name}: {','.join(vals)}" for name, vals in missing.items())
            print(f"  event equalization: dropping unavailable init(s): {detail}")
        common = desired.intersection(*(values for values in available.values()))
    else:
        common = set.intersection(*(values for values in available.values())) if available else set()
    return sorted(common)


def _storm_case(cfg: PaperStormCase, model, init_str):
    candidate = model.run_root / init_str
    run_dir = candidate if candidate.is_dir() else model.run_root
    init_tracks = sorted(run_dir.glob(f"**/*{init_str}*.atcfunix"))
    atcf_path = init_tracks[0] if init_tracks else find_atcfunix(run_dir)
    if init_tracks:
        print(f"Using track file: {atcf_path}")
    _, _, track = parse_atcfunix(atcf_path)
    if not track:
        raise ValueError(f"No forecast track fixes in {atcf_path}")
    init_dt = datetime.strptime(init_str, "%Y%m%d%H")
    return StormCase(
        run_dir=run_dir, init_dt=init_dt, storm_name=cfg.storm_name,
        model_label=model.name or detect_model(run_dir), domain=cfg.domain,
        grid_res=cfg.grid_res, mask_radius_km=cfg.mask_radius_km,
        display_radius_km=cfg.mask_radius_km, thresholds_mm=cfg.thresholds_mm,
        out_dir=cfg.out_dir, mrms_cache_dir=cfg.mrms_cache_dir,
        stage4_cache_dir=Path("/tmp/stage4_cache"), fhours_filter=None,
        track=track, case_slug=cfg.case_slug, init_str=init_str,
    ), atcf_path


def _forecast_window(cfg, storm_case, f1, f2, grid_lat, grid_lon):
    if cfg.forecast_domain == "parent":
        return parent_window_total(storm_case, f1, f2, grid_lat, grid_lon)
    return nest_window_total(storm_case, f1, f2, grid_lat, grid_lon)


def scores_from_counts(a, b, c, d):
    """Categorical metrics from aggregated contingency counts."""
    n = a + b + c + d
    aref = (a + b) * (a + c) / n if n else 0.0
    denom = a + b + c - aref
    ets = (a - aref) / denom if denom else np.nan
    bias = (a + b) / (a + c) if a + c else np.nan
    pod = a / (a + c) if a + c else np.nan
    far = b / (a + b) if a + b else np.nan
    csi = a / (a + b + c) if a + b + c else np.nan
    return dict(a=int(a), b=int(b), c=int(c), d=int(d), ets=float(ets),
                bias=float(bias), pod=float(pod), far=float(far), csi=float(csi))


def aggregate_categorical(sample_rows, bootstrap_replicates=0, random_seed=42):
    """Pool contingency counts and bootstrap forecast events for 95% CIs."""
    keys = ("model", "shift", "lead_hour", "threshold")
    groups = defaultdict(list)
    for row in sample_rows:
        groups[tuple(row[k] for k in keys)].append(row)
    rng = np.random.default_rng(random_seed)
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        counts = [sum(r[name] for r in rows) for name in ("a", "b", "c", "d")]
        result = {k: v for k, v in zip(keys, key)}
        result.update(scores_from_counts(*counts))
        result["n_events"] = len(rows)
        ets_boot, bias_boot = [], []
        if bootstrap_replicates > 0 and rows:
            matrix = np.array([[r[n] for n in ("a", "b", "c", "d")]
                               for r in rows], dtype=float)
            for _ in range(bootstrap_replicates):
                picked = matrix[rng.integers(0, len(rows), len(rows))].sum(axis=0)
                score = scores_from_counts(*picked)
                ets_boot.append(score["ets"])
                bias_boot.append(score["bias"])
        for name, values in (("ets", ets_boot), ("bias", bias_boot)):
            finite = np.asarray(values, float)
            finite = finite[np.isfinite(finite)]
            result[f"{name}_lo"] = float(np.percentile(finite, 2.5)) if finite.size else np.nan
            result[f"{name}_hi"] = float(np.percentile(finite, 97.5)) if finite.size else np.nan
        output.append(result)
    return output


def identify_objects(field, threshold, smooth_cells, min_pixels):
    """MODE-like native connected objects after smoothing and thresholding."""
    data = np.nan_to_num(np.asarray(field, float), nan=0.0)
    smooth = gaussian_filter(data, smooth_cells) if smooth_cells > 0 else data
    labels, count = nd_label(smooth >= threshold,
                             structure=np.ones((3, 3), dtype=int))
    kept = np.zeros_like(labels)
    objects = []
    new_id = 0
    for old_id in range(1, count + 1):
        mask = labels == old_id
        pixels = int(mask.sum())
        if pixels < min_pixels:
            continue
        new_id += 1
        kept[mask] = new_id
        rows, cols = np.where(mask)
        objects.append({"object_id": new_id, "pixels": pixels,
                        "row": float(rows.mean()), "col": float(cols.mean()),
                        "mean_mm": float(data[mask].mean()),
                        "max_mm": float(data[mask].max())})
    return kept, objects


def _object_records(field, cfg, init_str, lead, source, grid_lat, grid_lon):
    north_south_km = cfg.grid_res * 111.0
    east_west_km = north_south_km * max(
        math.cos(math.radians(float(np.nanmean(grid_lat)))), 0.1)
    cell_area_km2 = north_south_km * east_west_km
    min_pixels = max(1, int(math.ceil(cfg.object_min_area_km2 / cell_area_km2)))
    labels, objects = identify_objects(
        field, cfg.object_threshold_mm, cfg.object_smooth_cells, min_pixels)
    records = []
    for obj in objects:
        row = int(round(obj["row"]))
        col = int(round(obj["col"]))
        records.append({
            "init": init_str, "lead_hour": lead, "source": source,
            "object_id": obj["object_id"],
            "area_km2": obj["pixels"] * cell_area_km2,
            "centroid_lat": float(grid_lat[row, col]),
            "centroid_lon": float(grid_lon[row, col]),
            "mean_mm": obj["mean_mm"], "max_mm": obj["max_mm"],
        })
    return labels, records


def build_paper_samples(cfg: PaperStormCase):
    """Build event-equalized 6-h samples and lightweight plot payloads."""
    init_strs = common_inits(cfg)
    if not init_strs:
        raise RuntimeError(f"No common initializations across models for {cfg.storm_name}")
    print(f"Paper sample: {cfg.storm_name} | models: "
          f"{', '.join(m.name for m in cfg.models)} | common inits: "
          f"{', '.join(init_strs)}")
    grid_lat, grid_lon = make_fixed_grid(cfg.domain, cfg.grid_res)
    best_track = parse_bdeck(cfg.best_track)
    best_fixes = parse_bdeck_fixes(cfg.best_track)
    rows = []
    relative = {"MRMS": []}
    object_values = {"MRMS": []}
    for model in cfg.models:
        relative[model.name] = []
        object_values[model.name] = []
    object_rows = []
    selected_object = None
    successful_events = set()

    for init_str in init_strs:
        model_cases = {}
        forecast_fixes = {}
        for model in cfg.models:
            try:
                storm_case, atcf_path = _storm_case(cfg, model, init_str)
                model_cases[model.name] = storm_case
                forecast_fixes[model.name] = parse_atcfunix_fixes(atcf_path)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  skip init {init_str}: {model.name}: {exc}")
                model_cases = {}
                break
        if len(model_cases) != len(cfg.models):
            continue
        init_dt = next(iter(model_cases.values())).init_dt

        for lead in cfg.lead_hours:
            f1, f2 = lead - cfg.accumulation_hours, lead
            valid_start = init_dt + timedelta(hours=f1)
            valid_end = init_dt + timedelta(hours=f2)
            forecasts = {}
            failed = None
            for model in cfg.models:
                try:
                    forecasts[model.name] = _forecast_window(
                        cfg, model_cases[model.name], f1, f2, grid_lat, grid_lon)
                except RuntimeError as exc:
                    failed = f"{model.name}: {exc}"
                    break
            if failed:
                print(f"  skip {init_str} F{lead:03d}: {failed}")
                continue
            try:
                obs = build_mrms_total_window(valid_start, valid_end,
                                              cfg.mrms_cache_dir,
                                              grid_lat, grid_lon)
            except RuntimeError as exc:
                print(f"  skip {init_str} F{lead:03d}: MRMS: {exc}")
                continue

            best_lat, best_lon, best_rmw, best_fallback = interpolate_fix(
                best_fixes, valid_end, cfg.rmw_fallback_km)
            forecast_state, shifted = {}, {}
            for model in cfg.models:
                flat, flon, frmw, ffallback = interpolate_fix(
                    forecast_fixes[model.name], valid_end, cfg.rmw_fallback_km)
                forecast_state[model.name] = (flat, flon, frmw, ffallback)
                shifted[model.name] = shift_to_best(
                    forecasts[model.name], (flat, flon), (best_lat, best_lon),
                    cfg.grid_res)

            mask = haversine_km(best_lat, best_lon, grid_lat, grid_lon) <= cfg.mask_radius_km
            common = mask & np.isfinite(obs)
            for model in cfg.models:
                common &= np.isfinite(forecasts[model.name])
                common &= np.isfinite(shifted[model.name])
            if not common.any():
                print(f"  skip {init_str} F{lead:03d}: no common valid grid points")
                continue
            event_id = f"{cfg.case_slug}:{init_str}:F{lead:03d}"
            successful_events.add(event_id)
            obs_values = obs[common]
            for model in cfg.models:
                flat, flon, frmw, ffallback = forecast_state[model.name]
                track_error_km = float(haversine_km(
                    flat, flon, np.asarray(best_lat), np.asarray(best_lon)))
                for shift_name, field in (("raw", forecasts[model.name]),
                                          ("shifted", shifted[model.name])):
                    fcst_values = field[common]
                    cont = continuous_scores(fcst_values, obs_values)
                    for threshold in cfg.thresholds_mm:
                        score = contingency_scores(fcst_values, obs_values, threshold)
                        rows.append({
                            "storm": cfg.storm_name, "event": event_id,
                            "init": init_str, "valid": valid_end.strftime("%Y%m%d%H"),
                            "lead_hour": lead, "model": model.name,
                            "forecast_domain": cfg.forecast_domain,
                            "observation": "MRMS", "shift": shift_name,
                            "track_error_km": track_error_km,
                            "forecast_rmw_km": frmw,
                            "forecast_rmw_fallback": ffallback,
                            "best_rmw_km": best_rmw,
                            "best_rmw_fallback": best_fallback,
                            "n": cont["n"], "rmse": cont["rmse"],
                            "mae": cont["mae"], "bias_mm": cont["bias"],
                            "r": cont["r"], **score,
                        })

            if lead == cfg.composite_lead_hour:
                _, _, rel_obs = storm_relative_field(
                    obs, grid_lat, grid_lon, (best_lat, best_lon), best_rmw,
                    cfg.storm_relative_radius_rmw, cfg.storm_relative_res_rmw)
                relative["MRMS"].append(rel_obs)
                for model in cfg.models:
                    flat, flon, frmw, ffallback = forecast_state[model.name]
                    _, _, rel_fcst = storm_relative_field(
                        forecasts[model.name], grid_lat, grid_lon, (flat, flon), frmw,
                        cfg.storm_relative_radius_rmw, cfg.storm_relative_res_rmw)
                    relative[model.name].append(rel_fcst)

            if lead == cfg.object_lead_hour:
                obs_labels, obs_records = _object_records(
                    obs, cfg, init_str, lead, "MRMS", grid_lat, grid_lon)
                object_rows.extend(obs_records)
                object_values["MRMS"].append(obs[obs_labels > 0])
                model_labels = {}
                for model in cfg.models:
                    labels, records = _object_records(
                        forecasts[model.name], cfg, init_str, lead, model.name,
                        grid_lat, grid_lon)
                    object_rows.extend(records)
                    object_values[model.name].append(
                        forecasts[model.name][labels > 0])
                    model_labels[model.name] = labels
                if cfg.object_init is None or cfg.object_init == init_str:
                    selected_object = {
                        "init": init_str, "lead_hour": lead,
                        "grid_lat": grid_lat, "grid_lon": grid_lon,
                        "obs": obs, "obs_labels": obs_labels,
                        "forecasts": dict(forecasts), "model_labels": model_labels,
                    }
    if not rows:
        raise RuntimeError(f"No complete paper-style samples for {cfg.storm_name}")
    return {
        "sample_rows": rows, "relative": relative,
        "object_values": object_values, "object_rows": object_rows,
        "selected_object": selected_object, "grid_lat": grid_lat,
        "grid_lon": grid_lon, "n_events": len(successful_events),
    }


def _write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows:
        return
    fields = fields or list(rows[0])
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _matching(rows, **conditions):
    for row in rows:
        good = True
        for key, wanted in conditions.items():
            actual = row.get(key)
            if isinstance(wanted, float):
                good &= bool(np.isclose(float(actual), wanted))
            else:
                good &= actual == wanted
        if good:
            yield row


def plot_track_shift(rows, title, headline_thresholds, out_path):
    """Paper Figure 2 analogue: ETS/bias, low/high, raw/track-shifted."""
    thresholds = list(headline_thresholds[:2])
    while len(thresholds) < 2:
        thresholds.append(thresholds[0])
    models = sorted({r["model"] for r in rows})
    colors = _model_colors(models)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for col, threshold in enumerate(thresholds):
        for row_idx, metric in enumerate(("ets", "bias")):
            ax = axes[row_idx, col]
            for model in models:
                for shift_name, ls in (("raw", "-"), ("shifted", "--")):
                    subset = sorted(_matching(rows, model=model, shift=shift_name,
                                              threshold=float(threshold)),
                                    key=lambda r: r["lead_hour"])
                    if not subset:
                        continue
                    x = [r["lead_hour"] for r in subset]
                    y = [r[metric] for r in subset]
                    ax.plot(x, y, color=colors[model], ls=ls, marker="o",
                            ms=3, lw=1.8, label=f"{model} {shift_name}")
                    lo, hi = f"{metric}_lo", f"{metric}_hi"
                    if all(np.isfinite(r.get(lo, np.nan)) for r in subset):
                        ax.fill_between(x, [r[lo] for r in subset],
                                        [r[hi] for r in subset],
                                        color=colors[model], alpha=0.10)
            ax.grid(True, ls=":", alpha=0.4)
            ax.set_title(f"{metric.upper() if metric == 'ets' else 'Frequency bias'} "
                         f"at {threshold:g} mm")
            ax.set_ylabel("ETS" if metric == "ets" else "Frequency bias")
            if metric == "bias":
                ax.axhline(1, color="gray", ls=":", lw=1)
            else:
                ax.axhline(0, color="gray", ls=":", lw=1)
            ax.set_xlabel("Forecast lead time (h)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{title}\n6-h QPF vs MRMS - raw and best-track shifted")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_threshold_leads(rows, title, metric, out_path):
    """Paper Figures 3/4/9/10 analogue, one panel per model."""
    models = sorted({r["model"] for r in rows})
    thresholds = sorted({float(r["threshold"]) for r in rows})
    cmap = plt.get_cmap("viridis")
    colors = {thr: cmap(i / max(len(thresholds) - 1, 1))
              for i, thr in enumerate(thresholds)}
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5),
                             squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models):
        for threshold in thresholds:
            subset = sorted(_matching(rows, model=model, shift="shifted",
                                      threshold=threshold),
                            key=lambda r: r["lead_hour"])
            if subset:
                ax.plot([r["lead_hour"] for r in subset],
                        [r[metric] for r in subset], marker="o", ms=3,
                        color=colors[threshold], lw=1.7,
                        label=f"{threshold:g} mm")
        ax.set_title(model)
        ax.set_xlabel("Forecast lead time (h)")
        ax.grid(True, ls=":", alpha=0.4)
        if metric == "bias":
            ax.axhline(1, color="gray", ls=":", lw=1)
        else:
            ax.axhline(0, color="gray", ls=":", lw=1)
    axes[0, 0].set_ylabel("Equitable Threat Score (ETS)" if metric == "ets"
                          else "Frequency bias")
    axes[0, -1].legend(fontsize=8, loc="best")
    label = "ETS" if metric == "ets" else "frequency bias"
    fig.suptitle(f"{title}\nTrack-shifted 6-h QPF {label} vs MRMS")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _relative_axis(cfg):
    axis = np.arange(-cfg.storm_relative_radius_rmw,
                     cfg.storm_relative_radius_rmw + cfg.storm_relative_res_rmw / 2,
                     cfg.storm_relative_res_rmw)
    return np.meshgrid(axis, axis)


def plot_storm_relative(cfg, relative, out_path):
    sources = ["MRMS"] + [m.name for m in cfg.models]
    available = [s for s in sources if relative.get(s)]
    if not available:
        return False
    x, y = _relative_axis(cfg)
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), squeeze=False,
                             sharex=True, sharey=True)
    cmap = ListedColormap(QPF_COLORS)
    norm = BoundaryNorm(QPF_LEVELS, cmap.N)
    mesh = None
    for ax, source in zip(axes[0], available):
        mean = np.nanmean(np.stack(relative[source]), axis=0)
        mesh = ax.pcolormesh(x, y, mean, cmap=cmap, norm=norm, shading="auto")
        for radius in range(1, int(cfg.storm_relative_radius_rmw) + 1):
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="#555",
                                    lw=0.6, alpha=0.7))
        ax.axhline(0, color="#777", lw=0.5)
        ax.axvline(0, color="#777", lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{source} (n={len(relative[source])})")
        ax.set_xlabel("x / RMW")
    axes[0, 0].set_ylabel("y / RMW")
    fig.colorbar(mesh, ax=axes, label="Mean 6-h precipitation (mm)",
                 shrink=0.75)
    fig.suptitle(f"{cfg.storm_name} - storm-relative composite at "
                 f"F{cfg.composite_lead_hour:03d}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def radial_statistics(cfg, relative):
    x, y = _relative_axis(cfg)
    radius = np.hypot(x, y)
    edges = np.arange(0, cfg.storm_relative_radius_rmw + cfg.radial_bin_rmw,
                      cfg.radial_bin_rmw)
    rows = []
    for source, arrays in relative.items():
        if not arrays:
            continue
        stack = np.stack(arrays)
        for lo, hi in zip(edges[:-1], edges[1:]):
            values = stack[:, (radius >= lo) & (radius < hi)].ravel()
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            q05, q25, q50, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
            rows.append({"source": source, "radius_lo_rmw": lo,
                         "radius_hi_rmw": hi, "radius_mid_rmw": (lo + hi) / 2,
                         "n": int(values.size), "p05_mm": q05, "p25_mm": q25,
                         "median_mm": q50, "mean_mm": float(values.mean()),
                         "p75_mm": q75, "p95_mm": q95})
    return rows


def _bxp_stats(rows):
    return [{"med": r["median_mm"], "q1": r["p25_mm"], "q3": r["p75_mm"],
             "whislo": r["p05_mm"], "whishi": r["p95_mm"], "fliers": []}
            for r in rows]


def plot_radial(cfg, radial_rows, out_path):
    models = [m.name for m in cfg.models]
    if not radial_rows:
        return False
    colors = _model_colors(models)
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5),
                             squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models):
        obs = sorted(_matching(radial_rows, source="MRMS"),
                     key=lambda r: r["radius_mid_rmw"])
        fcst = sorted(_matching(radial_rows, source=model),
                      key=lambda r: r["radius_mid_rmw"])
        if obs:
            positions = [r["radius_mid_rmw"] - cfg.radial_bin_rmw * 0.12 for r in obs]
            artists = ax.bxp(_bxp_stats(obs), positions=positions,
                             widths=cfg.radial_bin_rmw * 0.20,
                             patch_artist=True, showfliers=False,
                             manage_ticks=False)
            for box in artists["boxes"]:
                box.set(facecolor="#555", alpha=0.45)
        if fcst:
            positions = [r["radius_mid_rmw"] + cfg.radial_bin_rmw * 0.12 for r in fcst]
            artists = ax.bxp(_bxp_stats(fcst), positions=positions,
                             widths=cfg.radial_bin_rmw * 0.20,
                             patch_artist=True, showfliers=False,
                             manage_ticks=False)
            for box in artists["boxes"]:
                box.set(facecolor=colors[model], alpha=0.55)
        ax.plot([], [], color="#555", lw=8, alpha=0.45, label="MRMS")
        ax.plot([], [], color=colors[model], lw=8, alpha=0.55, label=model)
        ax.set_title(model)
        ax.set_xlabel("Distance from center (RMW)")
        ax.set_xlim(0, cfg.storm_relative_radius_rmw)
        ax.set_xticks(np.arange(0, cfg.storm_relative_radius_rmw + 0.1, 1.0))
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("6-h precipitation (mm)")
    fig.suptitle(f"{cfg.storm_name} - F{cfg.composite_lead_hour:03d} radial "
                 f"distributions ({cfg.radial_bin_rmw:g}-RMW bins)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_objects(cfg, selected, out_path):
    if selected is None:
        return False
    models = [m.name for m in cfg.models]
    nrows = len(models)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 4.5 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    lat, lon = selected["grid_lat"], selected["grid_lon"]
    cmap = ListedColormap(QPF_COLORS)
    norm = BoundaryNorm(QPF_LEVELS, cmap.N)
    mesh = None
    for row, model in enumerate(models):
        forecast = selected["forecasts"][model]
        mesh = axes[row, 0].pcolormesh(lon, lat, forecast, cmap=cmap, norm=norm,
                                       shading="auto")
        axes[row, 0].set_title(f"{model} forecast")
        axes[row, 1].pcolormesh(lon, lat, selected["obs"], cmap=cmap, norm=norm,
                                shading="auto")
        axes[row, 1].set_title("MRMS observation")
        axes[row, 2].contourf(lon, lat,
                              (selected["model_labels"][model] > 0).astype(int),
                              levels=[0.5, 1.5], colors=["red"], alpha=0.65)
        if np.any(selected["obs_labels"] > 0):
            axes[row, 2].contour(lon, lat,
                                 (selected["obs_labels"] > 0).astype(int),
                                 levels=[0.5], colors=["blue"], linewidths=1.5)
        axes[row, 2].set_title(f"{model} objects (red), MRMS outlines (blue)")
        for ax in axes[row]:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, ls=":", alpha=0.25)
    fig.colorbar(mesh, ax=axes[:, :2], label="6-h precipitation (mm)", shrink=0.75)
    fig.suptitle(f"{cfg.storm_name} - object identification, init "
                 f"{selected['init']} F{selected['lead_hour']:03d}, "
                 f"threshold {cfg.object_threshold_mm:g} mm")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_object_frequency(cfg, values, out_path):
    models = [m.name for m in cfg.models]
    if not values.get("MRMS"):
        return False
    obs_arrays = [v for v in values["MRMS"] if v.size]
    if not obs_arrays:
        return False
    obs = np.concatenate(obs_arrays)
    upper = max(150.0, float(np.nanpercentile(obs, 99.5))) if obs.size else 150.0
    bins = np.linspace(0, upper, 61)
    colors = _model_colors(models)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.8),
                             squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models):
        model_arrays = [v for v in values.get(model, []) if v.size]
        fcst = np.concatenate(model_arrays) if model_arrays else np.array([])
        for data, color, label in ((obs, "black", "MRMS"),
                                   (fcst, colors[model], model)):
            if not data.size:
                continue
            counts, edges = np.histogram(data, bins=bins)
            frequency = counts / counts.sum() if counts.sum() else counts
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, np.where(frequency > 0, frequency, np.nan),
                    color=color, lw=2, label=label)
        ax.set_yscale("log")
        ax.set_title(model)
        ax.set_xlabel("6-h precipitation within objects (mm)")
        ax.grid(True, ls=":", alpha=0.35)
        ax.legend()
    axes[0, 0].set_ylabel("Relative frequency")
    fig.suptitle(f"{cfg.storm_name} - F{cfg.object_lead_hour:03d} object "
                 "precipitation distributions")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def compute_paper_storm(cfg: PaperStormCase):
    """Run every paper figure family for one storm and return sample rows."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_paper_samples(cfg)
    samples = payload["sample_rows"]
    aggregate = aggregate_categorical(samples, cfg.bootstrap_replicates,
                                      cfg.random_seed)
    radial = radial_statistics(cfg, payload["relative"])
    slug = cfg.case_slug

    sample_path = cfg.out_dir / f"paper_samples_{slug}.csv"
    aggregate_path = cfg.out_dir / f"paper_categorical_{slug}.csv"
    radial_path = cfg.out_dir / f"paper_radial_{slug}.csv"
    object_path = cfg.out_dir / f"paper_objects_{slug}.csv"
    _write_csv(sample_path, samples)
    _write_csv(aggregate_path, aggregate)
    _write_csv(radial_path, radial)
    _write_csv(object_path, payload["object_rows"])

    outputs = [
        ("track shift", cfg.out_dir / f"paper_track_shift_{slug}.png"),
        ("ETS by lead", cfg.out_dir / f"paper_ets_lead_{slug}.png"),
        ("frequency bias by lead", cfg.out_dir / f"paper_frequency_bias_lead_{slug}.png"),
    ]
    plot_track_shift(aggregate, cfg.storm_name, cfg.headline_thresholds_mm,
                     outputs[0][1])
    plot_threshold_leads(aggregate, cfg.storm_name, "ets", outputs[1][1])
    plot_threshold_leads(aggregate, cfg.storm_name, "bias", outputs[2][1])

    relative_path = cfg.out_dir / f"paper_storm_relative_{slug}.png"
    radial_plot_path = cfg.out_dir / f"paper_radial_{slug}.png"
    objects_plot_path = cfg.out_dir / f"paper_object_identification_{slug}.png"
    frequency_path = cfg.out_dir / f"paper_object_frequency_{slug}.png"
    if plot_storm_relative(cfg, payload["relative"], relative_path):
        outputs.append(("storm-relative composite", relative_path))
    if plot_radial(cfg, radial, radial_plot_path):
        outputs.append(("radial distributions", radial_plot_path))
    if plot_objects(cfg, payload["selected_object"], objects_plot_path):
        outputs.append(("object identification", objects_plot_path))
    if plot_object_frequency(cfg, payload["object_values"], frequency_path):
        outputs.append(("object frequency", frequency_path))

    print(f"\nPaper-style outputs for {cfg.storm_name} ({payload['n_events']} events):")
    for path in (sample_path, aggregate_path, radial_path, object_path):
        if path.exists():
            print(f"  CSV  {path}")
    for label, path in outputs:
        print(f"  PNG  {label}: {path}")
    return {"config": cfg, "samples": samples, "aggregate": aggregate,
            "payload": payload}


def compute_paper_suite(cfg: PaperSuiteCase):
    """Run each storm, then pool event-equalized counts across storms."""
    all_samples = []
    storm_results = []
    storm_configs = [load_paper_storm(path) for path in cfg.storm_paths]
    reference = storm_configs[0]
    ref_models = {m.name for m in reference.models}
    for storm_cfg in storm_configs[1:]:
        if {m.name for m in storm_cfg.models} != ref_models:
            raise ValueError("Every paper-suite storm must contain the same model names "
                             f"({reference.case_slug}: {sorted(ref_models)}; "
                             f"{storm_cfg.case_slug}: "
                             f"{sorted(m.name for m in storm_cfg.models)})")
        if storm_cfg.thresholds_mm != reference.thresholds_mm:
            raise ValueError("Every paper-suite storm must use identical thresholds_mm")
        if storm_cfg.accumulation_hours != reference.accumulation_hours:
            raise ValueError("Every paper-suite storm must use the same accumulation_hours")
        if storm_cfg.forecast_domain != reference.forecast_domain:
            raise ValueError("Every paper-suite storm must use the same forecast_domain")
    for storm_cfg in storm_configs:
        result = compute_paper_storm(storm_cfg)
        storm_results.append(result)
        all_samples.extend(result["samples"])
    if not all_samples:
        raise RuntimeError(f"No samples produced for suite {cfg.label}")
    aggregate = aggregate_categorical(all_samples, cfg.bootstrap_replicates,
                                      cfg.random_seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    slug = cfg.case_slug
    samples_path = cfg.out_dir / f"paper_multistorm_samples_{slug}.csv"
    aggregate_path = cfg.out_dir / f"paper_multistorm_categorical_{slug}.csv"
    _write_csv(samples_path, all_samples)
    _write_csv(aggregate_path, aggregate)
    thresholds = sorted({float(r["threshold"]) for r in aggregate})
    headline = [thresholds[0], thresholds[min(4, len(thresholds) - 1)]]
    shift_path = cfg.out_dir / f"paper_multistorm_track_shift_{slug}.png"
    ets_path = cfg.out_dir / f"paper_multistorm_ets_lead_{slug}.png"
    bias_path = cfg.out_dir / f"paper_multistorm_frequency_bias_lead_{slug}.png"
    plot_track_shift(aggregate, cfg.label, headline, shift_path)
    plot_threshold_leads(aggregate, cfg.label, "ets", ets_path)
    plot_threshold_leads(aggregate, cfg.label, "bias", bias_path)
    print(f"\nMulti-storm paper outputs ({len(cfg.storm_paths)} storms):")
    for path in (samples_path, aggregate_path, shift_path, ets_path, bias_path):
        print(f"  {path}")
    return {"samples": all_samples, "aggregate": aggregate,
            "storms": storm_results}


def compute_paper(config):
    if isinstance(config, PaperSuiteCase):
        return compute_paper_suite(config)
    return compute_paper_storm(config)


if __name__ == "__main__":
    compute_paper(load_paper_config(sys.argv[1]))
