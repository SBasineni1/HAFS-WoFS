"""
Cycle comparison for HAFS QPF: score every initialization of one storm on a
common valid window against the same observations over a shared
union-of-tracks footprint, so the only thing changing between cycles is
lead time.

Uses the fixed parent domain exclusively. Produces landfall-relative metric
curves, QPF small multiples, ETS/FSS heatmaps, representative spatial-error
maps, an animated QPF sequence, and long-format categorical/continuous and
FSS CSVs.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/run.py storms/helene_hfsa_cycles.yaml cycles
"""

import sys
import csv
from pathlib import Path
from datetime import timedelta

# Make sibling analysis modules importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import seaborn as sns
except ImportError:  # Optional at runtime; matplotlib remains a supported fallback.
    sns = None

from hafs_common import (
    QPF_LEVELS, discover_files, haversine_km, read_hafs_tp_records,
)
from hafs_case import (
    cycles_from_yaml, cycle_storm_case, discover_inits, window_hours,
    cycle_eligibility,
)
from ets_score import contingency_scores, build_mrms_total_window
from ets_full import regrid_2d_to_fixed, _OBS_COLOR, _FCST_STYLE
from parent_qpf import (
    parent_path_at_fhour, pick_cumulative_record, stage4_total_window,
    qpf_cmap,
)
from skill_metrics import continuous_scores, fractions_skill_score
from rmse_scatter import valid_points


# =============================================================================
# Union footprint
# =============================================================================

def window_track_points(case, valid_start, valid_end):
    """Hourly (lat, lon) track positions of one cycle inside the window,
    endpoints inclusive."""
    pts = []
    t = valid_start
    while t <= valid_end:
        pts.append(case.position_at(t))
        t += timedelta(hours=1)
    return pts


def union_swath(track_points, radius_km, grid_lat, grid_lon):
    """Boolean mask: grid points within radius_km of ANY (lat, lon) point."""
    swath = np.zeros(grid_lat.shape, dtype=bool)
    for tlat, tlon in track_points:
        swath |= haversine_km(tlat, tlon, grid_lat, grid_lon) <= radius_km
    return swath


# =============================================================================
# Windowed parent forecast fields
# =============================================================================


def parent_window_total(case, f1, f2, grid_lat, grid_lon):
    """Parent QPF for the window: cumulative APCP at f2 minus at f1.

    The 6-km parent domain is fixed, so its 0->fhour cumulative record is
    geographically valid and the window total is a clean difference.
    Interpolation noise can leave tiny negatives; those are floored at 0
    (NaN outside the parent hull is preserved). Raises RuntimeError when a
    needed parent file or record is missing.

    Sweeping consecutive windows (…f6, f6->f12, f12->f18…) asks for each
    cumulative field twice: once as a window's f2, then as the next window's
    f1. The regridded cumulative field is memoized on the case (keyed by
    fhour + grid shape) so that repeated decode+regrid work is done once per
    fhour per case. The fixed mesh is constant for a case's lifetime, so the
    grid-shape key only guards against accidental cross-grid reuse.
    """
    cache = getattr(case, "_parent_cumulative_cache", None)
    if cache is None:
        cache = {}
        try:
            case._parent_cumulative_cache = cache
        except AttributeError:  # slotted/frozen case: run without memoization
            cache = None

    def cumulative_at(fh):
        key = (fh, grid_lat.shape)
        if cache is not None and key in cache:
            return cache[key]
        path = parent_path_at_fhour(case, fh)
        if path is None:
            raise RuntimeError(
                f"no parent.atm file at f{fh:03d} for init {case.init_str}")
        rec = pick_cumulative_record(read_hafs_tp_records(path))
        if rec is None:
            raise RuntimeError(
                f"no APCP record in parent f{fh:03d} for init {case.init_str}")
        field = regrid_2d_to_fixed(rec["lats"], rec["lons"], rec["data"],
                                   grid_lat, grid_lon)
        if cache is not None:
            cache[key] = field
        return field

    # cumulative_at returns the cached array by reference; the arithmetic
    # below (subtraction, np.where) always allocates new arrays, so callers
    # never receive a handle that could mutate a cached field in place.
    total = cumulative_at(f2)
    if f1 > 0:
        total = total - cumulative_at(f1)
    return np.where(total < 0, 0.0, total)


# =============================================================================
# Field building (once per cycles case)
# =============================================================================

def build_cycle_fields(ccase):
    """Build everything the cycles product scores and plots.

    Returns a dict: grid_lat, grid_lon, mrms_win, stage4_win (None when
    Stage IV is unavailable), s4_label, swath (shared union mask), and
    cycles — a list of dicts {init_str, init_dt, f1, f2, parent_win}, one per
    surviving cycle. Raises RuntimeError when no
    cycle is eligible or every eligible cycle fails field extraction.
    """
    init_strs = ccase.inits or discover_inits(ccase.run_root)
    if not init_strs:
        raise RuntimeError(
            f"No YYYYMMDDHH cycle directories under {ccase.run_root} "
            f"and no 'inits' list given.")
    print(f"Window {ccase.valid_start:%Y-%m-%d %HZ} -> "
          f"{ccase.valid_end:%Y-%m-%d %HZ} | candidate inits: "
          f"{', '.join(init_strs)}")

    grid_lat, grid_lon = ccase.fixed_grid()
    print(f"Fixed grid: {grid_lat.shape[0]}x{grid_lat.shape[1]} "
          f"@ {ccase.grid_res}deg")

    # Pass 1: load cases; keep only cycles that fully cover the window.
    cases = []
    for init_str in init_strs:
        try:
            case = cycle_storm_case(ccase, init_str)
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip {init_str}: {e}")
            continue
        file_pairs = discover_files(case.run_dir, case.parent_glob(), None)
        if not file_pairs:
            print(f"  skip {init_str}: no parent files matching "
                  f"{case.parent_glob()}")
            continue
        max_fhour = file_pairs[-1][0]
        ok, reason = cycle_eligibility(case.init_dt, max_fhour,
                                       ccase.valid_start, ccase.valid_end)
        if not ok:
            print(f"  skip {init_str}: {reason}")
            continue
        cases.append(case)
    if not cases:
        raise RuntimeError(
            f"No eligible cycles for window "
            f"{ccase.valid_start:%Y%m%d%H}->{ccase.valid_end:%Y%m%d%H} "
            f"(inspected: {', '.join(init_strs)})")

    # Per-cycle forecast windows.
    cycles = []
    survivors = []
    for case in cases:
        f1, f2 = window_hours(case.init_dt, ccase.valid_start,
                              ccase.valid_end)
        print(f"\nCycle {case.init_str} (window f{f1:03d}-f{f2:03d})")
        try:
            parent_win = parent_window_total(case, f1, f2,
                                             grid_lat, grid_lon)
        except RuntimeError as e:
            print(f"  skip {case.init_str}: {e}")
            continue
        cycles.append(dict(init_str=case.init_str, init_dt=case.init_dt,
                           f1=f1, f2=f2, parent_win=parent_win))
        survivors.append(case)
    if not cycles:
        raise RuntimeError("Every eligible cycle failed field extraction.")

    # Shared footprint: union of every surviving cycle's in-window track.
    print("Union verification swath ...")
    all_pts = []
    for case in survivors:
        all_pts.extend(window_track_points(case, ccase.valid_start,
                                           ccase.valid_end))
    swath = union_swath(all_pts, ccase.mask_radius_km, grid_lat, grid_lon)
    print(f"  swath: {int(swath.sum()):,} grid points from "
          f"{len(survivors)} track(s)")

    # Shared observations (absolute-time, computed once for all cycles).
    print("MRMS window total ...")
    mrms_win = build_mrms_total_window(ccase.valid_start, ccase.valid_end,
                                       ccase.mrms_cache_dir,
                                       grid_lat, grid_lon)
    print("Stage IV window total ...")
    s4_lat, s4_lon, s4_native, s4_label = stage4_total_window(
        ccase.stage4_cache_dir, ccase.valid_start, ccase.valid_end,
        all_pts, ccase.display_radius_km)
    if s4_native is None:
        stage4_win, s4_label = None, "unavailable"
        print("  Stage IV unavailable — scoring MRMS only.")
    else:
        stage4_win = regrid_2d_to_fixed(s4_lat, s4_lon, s4_native,
                                        grid_lat, grid_lon)

    return dict(grid_lat=grid_lat, grid_lon=grid_lon, mrms_win=mrms_win,
                stage4_win=stage4_win, s4_label=s4_label, swath=swath,
                cycles=cycles)


# =============================================================================
# Scoring + outputs
# =============================================================================

def _plot_theme():
    """Use seaborn when available while retaining a dependency-light fallback."""
    if sns is not None:
        sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def hours_before_landfall(ccase, init_dt):
    """Hours from initialization to landfall, or None when not configured."""
    if ccase.landfall_time is None:
        return None
    return (ccase.landfall_time - init_dt).total_seconds() / 3600.0


def cycle_label(ccase, init_dt):
    """Compact cycle label with optional landfall-relative lead time."""
    lead = hours_before_landfall(ccase, init_dt)
    if lead is None:
        return init_dt.strftime("%m-%d %HZ")
    return f"{init_dt:%m-%d %HZ}\n{lead:.0f} h pre-LF"


def _draw_heatmap(ax, values, xlabels, ylabels, title, cbar_label,
                  vmin, vmax, cmap, fmt=".2f"):
    """Seaborn heatmap with an annotated matplotlib fallback."""
    values = np.asarray(values, dtype=float)
    if sns is not None:
        sns.heatmap(values, ax=ax, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
                    xticklabels=xlabels, yticklabels=ylabels,
                    cbar_kws={"label": cbar_label})
    else:
        im = ax.imshow(values, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.figure.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
        ax.set_xticks(np.arange(len(xlabels)), labels=xlabels)
        ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)
        midpoint = (vmin + vmax) / 2.0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(col, row, format(value, fmt), ha="center",
                            va="center",
                            color="white" if value > midpoint else "black",
                            fontsize=8)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

def cycles_caveat(fields, ccase):
    """Figure-footer caveat describing the Stage IV window (or its absence)."""
    if fields["stage4_win"] is None:
        return "Stage IV unavailable — not scored."
    return (f"Stage IV: CONUS-only, 24h 12Z–12Z files summed over touched "
            f"days ({fields['s4_label']}) — window approximates "
            f"{ccase.valid_start:%m-%d %HZ}–{ccase.valid_end:%m-%d %HZ}.")


def plot_metrics(ccase, results, out_path, caveat=""):
    """Continuous skill and headline ETS versus initialization/landfall lead.

    results: list of dicts {init_str, init_dt, forecast, observation,
    cont, rows} — one per cycle x pair.
    """
    _plot_theme()
    inits = sorted({r["init_dt"] for r in results})
    pairs = sorted({(r["forecast"], r["observation"]) for r in results},
                   key=lambda p: (p[1], p[0]))
    thr = ccase.ets_threshold_mm
    use_lead = ccase.landfall_time is not None
    x = ([hours_before_landfall(ccase, init_dt) for init_dt in inits]
         if use_lead else inits)
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 12), sharex=True)

    def series(fname, oname, getter):
        by_init = {r["init_dt"]: r for r in results
                   if r["forecast"] == fname and r["observation"] == oname}
        return [getter(by_init[i]) if i in by_init else np.nan
                for i in inits]

    def ets_at(res):
        for row in res["rows"]:
            if row["threshold"] == thr:
                return row["ets"]
        return np.nan

    panels = [
        (axes[0], lambda res: res["cont"]["rmse"], "RMSE (mm)"),
        (axes[1], lambda res: res["cont"]["bias"], "bias (mm)"),
        (axes[2], lambda res: res["cont"]["r"], "Pearson correlation"),
        (axes[3], ets_at, f"ETS @ {thr:g} mm"),
    ]
    for ax, getter, label in panels:
        for fname, oname in pairs:
            style = _FCST_STYLE.get(fname, dict(ls="-", marker="o"))
            ax.plot(x, series(fname, oname, getter),
                    color=_OBS_COLOR.get(oname, "gray"), lw=2, **style,
                    label=f"{fname} vs {oname}")
        ax.set_ylabel(label)
        ax.grid(True, ls=":", alpha=0.4)
    axes[1].axhline(0, color="gray", ls=":", lw=0.8)
    axes[0].legend(loc="best", fontsize=9)
    if use_lead:
        axes[-1].set_xlabel("Hours before landfall (forecast initialization)")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([f"{value:.0f}" for value in x])
        axes[-1].invert_xaxis()
        landfall_text = f" | landfall {ccase.landfall_time:%Y-%m-%d %H%MZ}"
    else:
        axes[-1].set_xlabel("initialization")
        axes[-1].set_xticks(inits)
        axes[-1].set_xticklabels([i.strftime("%m-%d %HZ") for i in inits],
                                 rotation=45, ha="right")
        landfall_text = ""
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} QPF by initialization\n"
        f"valid {ccase.valid_start:%Y-%m-%d %HZ} – "
        f"{ccase.valid_end:%Y-%m-%d %HZ} | union TC swath "
        f"≤{ccase.mask_radius_km:.0f} km{landfall_text}")
    if caveat:
        fig.text(0.5, -0.01, caveat, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_maps(ccase, fields, out_path):
    """Small-multiple parent-QPF maps per cycle + observed MRMS panel.

    Shared color scale and extent; the union verification swath is
    outlined on every panel. Wraps at 4 columns.
    """
    panels = [(cycle_label(ccase, c["init_dt"]), c["parent_win"])
              for c in fields["cycles"]]
    panels.append(("MRMS observed", fields["mrms_win"]))
    n = len(panels)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    lat_min, lat_max, lon_min, lon_max = ccase.domain
    cmap, norm = qpf_cmap()
    grid_lat, grid_lon = fields["grid_lat"], fields["grid_lon"]

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.6 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    flat = axes.ravel()
    cf = None
    for ax, (title, data) in zip(flat, panels):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        cf = ax.contourf(grid_lon, grid_lat,
                         np.nan_to_num(data, nan=0.0),
                         levels=QPF_LEVELS, cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), extend="max")
        ax.contour(grid_lon, grid_lat, fields["swath"].astype(float),
                   levels=[0.5], colors="k", linewidths=1.0,
                   transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=10)
    for ax in flat[n:]:
        ax.set_visible(False)
    if cf is not None:
        fig.colorbar(cf, ax=axes, label="Accumulated Precipitation (mm)",
                     ticks=QPF_LEVELS, shrink=0.7, fraction=0.02)
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} parent QPF by "
        f"initialization\nvalid {ccase.valid_start:%Y-%m-%d %HZ} – "
        f"{ccase.valid_end:%Y-%m-%d %HZ} | swath outline "
        f"≤{ccase.mask_radius_km:.0f} km", y=1.02)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ets_heatmaps(ccase, results, out_path):
    """Parent-domain ETS across all thresholds and cycles."""
    _plot_theme()
    rows = [r for r in results if r["observation"] == "MRMS"]
    inits = sorted({r["init_dt"] for r in rows})
    thresholds = sorted({item["threshold"] for r in rows for item in r["rows"]})
    forecasts = [name for name in ("parent",)
                 if any(r["forecast"] == name for r in rows)]
    matrices = []
    for forecast in forecasts:
        by_init = {r["init_dt"]: r for r in rows if r["forecast"] == forecast}
        matrix = []
        for init_dt in inits:
            by_threshold = {item["threshold"]: item["ets"]
                            for item in by_init[init_dt]["rows"]}
            matrix.append([by_threshold.get(threshold, np.nan)
                           for threshold in thresholds])
        matrices.append(np.asarray(matrix, dtype=float))
    finite_parts = [m[np.isfinite(m)] for m in matrices
                    if np.isfinite(m).any()]
    finite = np.concatenate(finite_parts) if finite_parts else np.array([])
    vmax = (min(1.0, max(0.35, float(np.nanpercentile(finite, 98))))
            if finite.size else 0.35)
    fig, axes = plt.subplots(1, len(forecasts), figsize=(7 * len(forecasts),
                                                        0.65 * len(inits) + 4),
                             squeeze=False)
    labels = [cycle_label(ccase, init_dt) for init_dt in inits]
    for ax, forecast, matrix in zip(axes[0], forecasts, matrices):
        _draw_heatmap(ax, matrix, [f"{t:g}" for t in thresholds], labels,
                      forecast.title(), "ETS", 0.0, vmax,
                      "mako" if sns is not None else "YlGnBu")
        ax.set_xlabel("Rainfall threshold (mm)")
        ax.set_ylabel("Initialization")
    fig.suptitle(f"{ccase.storm_name} — {ccase.model_label} ETS by cycle "
                 "(vs MRMS)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fss_heatmaps(ccase, fss_rows, out_path):
    """FSS across cycle, neighborhood scale, threshold, and forecast grid."""
    _plot_theme()
    inits = sorted({r["init_dt"] for r in fss_rows})
    thresholds = sorted({r["threshold"] for r in fss_rows})
    forecasts = [name for name in ("parent",)
                 if any(r["forecast"] == name for r in fss_rows)]
    scales = sorted({r["scale_km"] for r in fss_rows})
    fig, axes = plt.subplots(len(forecasts), len(thresholds),
                             figsize=(6.5 * len(thresholds),
                                     4.2 * len(forecasts) + 1.2),
                             squeeze=False)
    labels = [cycle_label(ccase, init_dt) for init_dt in inits]
    for row_index, forecast in enumerate(forecasts):
        for col_index, threshold in enumerate(thresholds):
            lookup = {(r["init_dt"], r["scale_km"]): r["fss"]
                      for r in fss_rows
                      if r["forecast"] == forecast
                      and r["threshold"] == threshold}
            matrix = [[lookup.get((init_dt, scale), np.nan) for scale in scales]
                      for init_dt in inits]
            ax = axes[row_index, col_index]
            _draw_heatmap(
                ax, matrix, [f"{scale:g}" for scale in scales], labels,
                f"{forecast.title()} · {threshold:g} mm", "FSS",
                0.0, 1.0, "rocket" if sns is not None else "YlOrRd")
            ax.set_xlabel("Neighborhood scale (km)")
            ax.set_ylabel("Initialization")
    fig.suptitle(f"{ccase.storm_name} — {ccase.model_label} FSS by cycle "
                 "(vs MRMS)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _map_context(ax, ccase):
    lat_min, lat_max, lon_min, lon_max = ccase.domain
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")


def plot_error_maps(ccase, fields, out_path):
    """Parent minus MRMS for representative early, middle, late cycles."""
    cycles_data = fields["cycles"]
    indices = sorted(set((0, len(cycles_data) // 2, len(cycles_data) - 1)))
    selected = [cycles_data[index] for index in indices]
    errors = [cycle["parent_win"] - fields["mrms_win"]
              for cycle in selected]
    finite = np.concatenate([field[np.isfinite(field)] for field in errors
                             if np.isfinite(field).any()])
    limit = max(25.0, float(np.nanpercentile(np.abs(finite), 98)))
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(
        1, len(selected), figsize=(5.3 * len(selected), 4.8), squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()})
    mesh = None
    for col, cycle in enumerate(selected):
        ax = axes[0, col]
        _map_context(ax, ccase)
        error = np.where(fields["swath"], errors[col], np.nan)
        mesh = ax.pcolormesh(fields["grid_lon"], fields["grid_lat"], error,
                             cmap="RdBu_r", norm=norm, shading="auto",
                             transform=ccrs.PlateCarree())
        ax.contour(fields["grid_lon"], fields["grid_lat"],
                   fields["swath"].astype(float), levels=[0.5],
                   colors="#333333", linewidths=0.7,
                   transform=ccrs.PlateCarree())
        ax.set_title(cycle_label(ccase, cycle["init_dt"]))
    fig.colorbar(mesh, ax=axes, shrink=0.82, pad=0.02,
                 label="Forecast − MRMS (mm)")
    fig.suptitle(f"{ccase.storm_name} — {ccase.model_label} representative "
                 "cycle errors", fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def animate_cycle_qpf(ccase, fields, out_path):
    """Animate parent QPF, MRMS, and their difference across initializations."""
    cycles_data = fields["cycles"]
    errors = [cycle["parent_win"] - fields["mrms_win"]
              for cycle in cycles_data]
    finite = np.concatenate([field[np.isfinite(field)] for field in errors
                             if np.isfinite(field).any()])
    limit = max(25.0, float(np.nanpercentile(np.abs(finite), 98)))
    error_norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    qpf_cmap_obj, qpf_norm = qpf_cmap()
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2),
                             subplot_kw={"projection": projection})
    qpf_map = plt.cm.ScalarMappable(norm=qpf_norm, cmap=qpf_cmap_obj)
    err_map = plt.cm.ScalarMappable(norm=error_norm, cmap="RdBu_r")
    fig.colorbar(qpf_map, ax=axes[:2], shrink=0.72, pad=0.02,
                 label="Accumulated precipitation (mm)")
    fig.colorbar(err_map, ax=axes[2], shrink=0.72, pad=0.02,
                 label="Forecast − MRMS (mm)")

    def update(frame):
        cycle = cycles_data[frame]
        panels = ((cycle["parent_win"], qpf_cmap_obj, qpf_norm, "Parent forecast"),
                  (fields["mrms_win"], qpf_cmap_obj, qpf_norm, "MRMS observed"),
                  (errors[frame], "RdBu_r", error_norm, "Parent − MRMS"))
        for ax, (data, cmap, norm, title) in zip(axes, panels):
            ax.clear()
            _map_context(ax, ccase)
            masked = np.where(fields["swath"], data, np.nan)
            ax.pcolormesh(fields["grid_lon"], fields["grid_lat"], masked,
                          cmap=cmap, norm=norm, shading="auto",
                          transform=projection)
            ax.contour(fields["grid_lon"], fields["grid_lat"],
                       fields["swath"].astype(float), levels=[0.5],
                       colors="#333333", linewidths=0.7,
                       transform=projection)
            ax.set_title(title)
        fig.suptitle(f"{ccase.storm_name} — {ccase.model_label} · "
                     f"{cycle_label(ccase, cycle['init_dt']).replace(chr(10), ' · ')}")
        return axes

    movie = animation.FuncAnimation(fig, update, frames=len(cycles_data),
                                    interval=1200, repeat=True, blit=False)
    movie.save(out_path, writer=animation.PillowWriter(fps=1), dpi=110)
    plt.close(fig)


def compute_cycles(ccase, fields=None):
    if fields is None:
        fields = build_cycle_fields(ccase)
    swath = fields["swath"]
    observations = [("MRMS", fields["mrms_win"])]
    if fields["stage4_win"] is not None:
        observations.append(("Stage IV", fields["stage4_win"]))

    # Make sure the headline ETS threshold is actually scored.
    thresholds = list(ccase.thresholds_mm)
    if ccase.ets_threshold_mm not in thresholds:
        thresholds = sorted(set(thresholds) | {ccase.ets_threshold_mm})

    results = []
    fss_rows = []
    print("\n" + "=" * 84)
    for cyc in fields["cycles"]:
        for fname, fgrid in (("parent", cyc["parent_win"]),):
            for oname, ogrid in observations:
                fcst, obs = valid_points(fgrid, ogrid, swath)
                cont = continuous_scores(fcst, obs)
                rows = [contingency_scores(fcst, obs, thr)
                        for thr in thresholds]
                results.append(dict(init_str=cyc["init_str"],
                                    init_dt=cyc["init_dt"],
                                    forecast=fname, observation=oname,
                                    cont=cont, rows=rows))
                print(f"{cyc['init_str']} {fname:>7} vs {oname:<9} "
                      f"n={cont['n']:>9,} RMSE={cont['rmse']:>7.2f} "
                      f"MAE={cont['mae']:>7.2f} bias={cont['bias']:>+7.2f} "
                      f"r={cont['r']:>5.2f}")
                if oname == "MRMS":
                    valid = (swath & np.isfinite(fgrid)
                             & np.isfinite(ogrid))
                    clean_fcst = np.nan_to_num(fgrid, nan=0.0)
                    clean_obs = np.nan_to_num(ogrid, nan=0.0)
                    for threshold in ccase.fss_thresholds_mm:
                        for scale in ccase.fss_scales_cells:
                            fss_rows.append({
                                "init": cyc["init_str"],
                                "init_dt": cyc["init_dt"],
                                "lead_hours_to_landfall":
                                    hours_before_landfall(ccase, cyc["init_dt"]),
                                "forecast": fname,
                                "observation": oname,
                                "threshold": threshold,
                                "scale_cells": scale,
                                "scale_km": round(
                                    scale * ccase.grid_res * 111.0, 1),
                                "fss": fractions_skill_score(
                                    clean_fcst, clean_obs, threshold,
                                    scale, valid),
                            })
    print("=" * 84)

    ccase.out_dir.mkdir(parents=True, exist_ok=True)
    slug = ccase.output_slug
    out_csv = ccase.out_dir / f"cycles_{slug}.csv"

    fieldnames = ["init", "lead_hours_to_landfall", "forecast",
                  "observation", "threshold", "n",
                  "rmse", "mae", "bias_mm", "r", "a", "b", "c", "d",
                  "ets", "bias", "pod", "far", "csi", "hss"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            cont = res["cont"]
            for r in res["rows"]:
                w.writerow({"init": res["init_str"],
                            "lead_hours_to_landfall": hours_before_landfall(
                                ccase, res["init_dt"]),
                            "forecast": res["forecast"],
                            "observation": res["observation"],
                            "n": cont["n"], "rmse": cont["rmse"],
                            "mae": cont["mae"], "bias_mm": cont["bias"],
                            "r": cont["r"], **r})
    print(f"\nSaved table: {out_csv}")

    out_fss_csv = ccase.out_dir / f"cycles_fss_{slug}.csv"
    with open(out_fss_csv, "w", newline="") as fh:
        fieldnames = ["init", "lead_hours_to_landfall", "forecast",
                      "observation", "threshold", "scale_cells",
                      "scale_km", "fss"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fss_rows)
    print(f"Saved table: {out_fss_csv}")

    caveat = cycles_caveat(fields, ccase)
    print(caveat)
    out_metrics = ccase.out_dir / f"cycles_metrics_{slug}.png"
    out_maps = ccase.out_dir / f"cycles_maps_{slug}.png"
    out_ets_heatmap = ccase.out_dir / f"cycles_ets_heatmap_{slug}.png"
    out_fss_heatmap = ccase.out_dir / f"cycles_fss_heatmap_{slug}.png"
    out_errors = ccase.out_dir / f"cycles_errors_{slug}.png"
    plot_metrics(ccase, results, out_metrics, caveat=caveat)
    print(f"Saved plot : {out_metrics}")
    plot_maps(ccase, fields, out_maps)
    print(f"Saved plot : {out_maps}")
    plot_ets_heatmaps(ccase, results, out_ets_heatmap)
    print(f"Saved plot : {out_ets_heatmap}")
    plot_fss_heatmaps(ccase, fss_rows, out_fss_heatmap)
    print(f"Saved plot : {out_fss_heatmap}")
    plot_error_maps(ccase, fields, out_errors)
    print(f"Saved plot : {out_errors}")
    if ccase.make_animation:
        out_animation = ccase.out_dir / f"cycles_qpf_{slug}.gif"
        try:
            animate_cycle_qpf(ccase, fields, out_animation)
        except (ImportError, RuntimeError) as exc:
            print(f"Animation unavailable: {exc}")
        else:
            print(f"Saved movie: {out_animation}")


if __name__ == "__main__":
    compute_cycles(cycles_from_yaml(sys.argv[1]))
