"""
Cycle comparison for HAFS QPF: score every initialization of one storm on a
common valid window against the same observations over a shared
union-of-tracks footprint, so the only thing changing between cycles is
lead time.

Produces a metrics-vs-init figure (RMSE / bias / ETS at one threshold),
nest-QPF map small-multiples with the observed MRMS panel, and one
long-format CSV.

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from hafs_common import (
    QPF_LEVELS, discover_files, hafs_event_total, haversine_km,
    read_hafs_tp_records,
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
from skill_metrics import continuous_scores
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
# Windowed forecast fields
# =============================================================================

def filter_window_pairs(file_pairs, f1, f2):
    """Nest (fhour, path) pairs whose bucket falls inside (f1, f2].

    Each storm.atm file's per-interval bucket ENDS at its fhour, so files
    with f1 < fhour <= f2 together hold exactly the window's rain.
    """
    return [(h, p) for h, p in file_pairs if f1 < h <= f2]


def nest_window_total(case, f1, f2, grid_lat, grid_lon):
    """Nest QPF accumulated over window hours (f1, f2] on the fixed mesh.

    Sums the per-interval buckets exactly as hafs_event_total does for the
    full event (its incremental mode). Raises RuntimeError when the window
    has no nest files or per-interval buckets. Accumulated-mode differencing
    is intentionally not implemented because nest cumulative records are a
    geographic trap on the moving nest (see pick_total_record, which never
    selects them); these windows raise RuntimeError so build_cycle_fields
    skips the cycle with a printed reason rather than silently scoring zeros.
    """
    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    window_pairs = filter_window_pairs(file_pairs, f1, f2)
    if not window_pairs:
        raise RuntimeError(
            f"no nest files in f{f1:03d}-f{f2:03d} for init {case.init_str}")
    total, mode = hafs_event_total(window_pairs, grid_lat, grid_lon)
    if mode != "incremental":
        raise RuntimeError(
            f"no per-interval APCP buckets in f{f1:03d}-f{f2:03d} for init "
            f"{case.init_str} (mode={mode!r}) — nest cumulative records are "
            "not usable on the moving nest")
    return total


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
    cycles — a list of dicts {init_str, init_dt, f1, f2, nest_win,
    parent_win}, one per surviving cycle. Raises RuntimeError when no
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
        file_pairs = discover_files(case.run_dir, case.storm_glob(), None)
        if not file_pairs:
            print(f"  skip {init_str}: no nest files matching "
                  f"{case.storm_glob()}")
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
            nest_win = nest_window_total(case, f1, f2, grid_lat, grid_lon)
            parent_win = parent_window_total(case, f1, f2,
                                             grid_lat, grid_lon)
        except RuntimeError as e:
            print(f"  skip {case.init_str}: {e}")
            continue
        cycles.append(dict(init_str=case.init_str, init_dt=case.init_dt,
                           f1=f1, f2=f2, nest_win=nest_win,
                           parent_win=parent_win))
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

def cycles_caveat(fields, ccase):
    """Figure-footer caveat describing the Stage IV window (or its absence)."""
    if fields["stage4_win"] is None:
        return "Stage IV unavailable — not scored."
    return (f"Stage IV: CONUS-only, 24h 12Z–12Z files summed over touched "
            f"days ({fields['s4_label']}) — window approximates "
            f"{ccase.valid_start:%m-%d %HZ}–{ccase.valid_end:%m-%d %HZ}.")


def plot_metrics(ccase, results, out_path, caveat=""):
    """Metrics vs init time: RMSE, bias, and ETS@headline-threshold panels.

    results: list of dicts {init_str, init_dt, forecast, observation,
    cont, rows} — one per cycle x pair.
    """
    inits = sorted({r["init_dt"] for r in results})
    pairs = sorted({(r["forecast"], r["observation"]) for r in results},
                   key=lambda p: (p[1], p[0]))
    thr = ccase.ets_threshold_mm
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10), sharex=True)

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
        (axes[2], ets_at, f"ETS @ {thr:g} mm"),
    ]
    for ax, getter, label in panels:
        for fname, oname in pairs:
            style = _FCST_STYLE.get(fname, dict(ls="-", marker="o"))
            ax.plot(inits, series(fname, oname, getter),
                    color=_OBS_COLOR.get(oname, "gray"), lw=2, **style,
                    label=f"{fname} vs {oname}")
        ax.set_ylabel(label)
        ax.grid(True, ls=":", alpha=0.4)
    axes[1].axhline(0, color="gray", ls=":", lw=0.8)
    axes[0].legend(loc="best", fontsize=9)
    axes[2].set_xlabel("initialization")
    axes[2].set_xticks(inits)
    axes[2].set_xticklabels([i.strftime("%m-%d %HZ") for i in inits],
                            rotation=45, ha="right")
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} QPF by initialization\n"
        f"valid {ccase.valid_start:%Y-%m-%d %HZ} – "
        f"{ccase.valid_end:%Y-%m-%d %HZ} | union TC swath "
        f"≤{ccase.mask_radius_km:.0f} km")
    if caveat:
        fig.text(0.5, -0.01, caveat, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_maps(ccase, fields, out_path):
    """Small-multiple nest-QPF maps per cycle + observed MRMS panel.

    Shared color scale and extent; the union verification swath is
    outlined on every panel. Wraps at 4 columns.
    """
    panels = [(f"init {c['init_dt']:%m-%d %HZ}", c["nest_win"])
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
        f"{ccase.storm_name} — {ccase.model_label} nest QPF by "
        f"initialization\nvalid {ccase.valid_start:%Y-%m-%d %HZ} – "
        f"{ccase.valid_end:%Y-%m-%d %HZ} | swath outline "
        f"≤{ccase.mask_radius_km:.0f} km", y=1.02)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
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
    print("\n" + "=" * 84)
    for cyc in fields["cycles"]:
        for fname, fgrid in (("parent", cyc["parent_win"]),
                             ("nest", cyc["nest_win"])):
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
    print("=" * 84)

    ccase.out_dir.mkdir(parents=True, exist_ok=True)
    slug = ccase.output_slug
    out_csv = ccase.out_dir / f"cycles_{slug}.csv"

    fieldnames = ["init", "forecast", "observation", "threshold", "n",
                  "rmse", "mae", "bias_mm", "r", "a", "b", "c", "d",
                  "ets", "bias", "pod", "far", "csi", "hss"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            cont = res["cont"]
            for r in res["rows"]:
                w.writerow({"init": res["init_str"],
                            "forecast": res["forecast"],
                            "observation": res["observation"],
                            "n": cont["n"], "rmse": cont["rmse"],
                            "mae": cont["mae"], "bias_mm": cont["bias"],
                            "r": cont["r"], **r})
    print(f"\nSaved table: {out_csv}")

    caveat = cycles_caveat(fields, ccase)
    print(caveat)
    out_metrics = ccase.out_dir / f"cycles_metrics_{slug}.png"
    out_maps = ccase.out_dir / f"cycles_maps_{slug}.png"
    plot_metrics(ccase, results, out_metrics, caveat=caveat)
    print(f"Saved plot : {out_metrics}")
    plot_maps(ccase, fields, out_maps)
    print(f"Saved plot : {out_maps}")


if __name__ == "__main__":
    compute_cycles(cycles_from_yaml(sys.argv[1]))
