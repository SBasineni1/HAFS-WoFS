"""
ETS for HAFS-A QPF (parent domain + moving 2-km nest) verified against
MRMS QPE and NCEP Stage IV QPE over the Hurricane Helene rainfall swath.

Produces one combined ETS-vs-threshold figure (4 curves: parent/nest x
MRMS/StageIV) and one combined CSV. Reuses all GRIB2/MRMS/Stage IV plumbing
from qpf_full_run.py and parent_qpf.py, and the contingency math + MRMS
plumbing from ets_score.py. The existing ets_score.py is left untouched.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/ets_full.py
"""

import sys
import csv
from pathlib import Path

# Make sibling analysis modules importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpf_full_run import (
    HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER, FIXED_DOMAIN, GRID_RES,
    TC_MASK_RADIUS_KM, INIT_DT, OUT_DIR,
    discover_files, hafs_event_total,
)
from ets_score import (
    THRESHOLDS_MM, contingency_scores, build_mrms_total, tc_swath_mask,
)
from parent_qpf import (
    default_parent_path, read_hafs_tp_records, pick_cumulative_record,
    stage4_total, STAGE4_CACHE_DIR,
)

OUT_PNG = OUT_DIR.parent / "ets_full_helene.png"
OUT_CSV = OUT_DIR.parent / "ets_full_helene.csv"


def regrid_2d_to_fixed(src_lat, src_lon, data, grid_lat, grid_lon):
    """Interpolate a curvilinear/rectilinear source field onto the fixed mesh.

    src_lat/src_lon may be 1-D axes or 2-D meshes; data is shaped like the
    2-D source mesh. Uses linear griddata; points outside the source hull
    come back NaN (no extrapolation).
    """
    src_lat = np.asarray(src_lat, dtype=float)
    src_lon = np.asarray(src_lon, dtype=float)
    if src_lat.ndim == 1 and src_lon.ndim == 1:
        src_lon, src_lat = np.meshgrid(src_lon, src_lat)
    pts = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    vals = np.asarray(data, dtype=float).ravel()
    finite = np.isfinite(vals)
    out = griddata(
        pts[finite], vals[finite],
        (grid_lat, grid_lon), method="linear",
    )
    return out


def score_pair(fcst_grid, obs_grid, swath, thresholds, contingency_fn):
    """Score one forecast/observation pair over the swath's valid points.

    Valid points are swath & finite(obs) & finite(fcst); kept values are
    zero-filled before thresholding. Returns (rows, n_valid).
    """
    valid = swath & np.isfinite(obs_grid) & np.isfinite(fcst_grid)
    n_valid = int(np.sum(valid))
    fcst = np.nan_to_num(fcst_grid[valid], nan=0.0)
    obs = np.nan_to_num(obs_grid[valid], nan=0.0)
    rows = [contingency_fn(fcst, obs, thr) for thr in thresholds]
    return rows, n_valid


def build_fixed_grid():
    """Fixed lat/lon verification mesh (same as ets_score.main)."""
    lat_min, lat_max, lon_min, lon_max = FIXED_DOMAIN
    fixed_lons = np.arange(lon_min, lon_max + GRID_RES, GRID_RES)
    fixed_lats = np.arange(lat_min, lat_max + GRID_RES, GRID_RES)
    grid_lon, grid_lat = np.meshgrid(fixed_lons, fixed_lats)
    return grid_lat, grid_lon


def hafs_parent_total(grid_lat, grid_lon):
    """HAFS-A parent cumulative APCP regridded onto the fixed verification mesh.

    Reuses parent_qpf's discovery + cumulative-record selection, then maps the
    parent grid onto the fixed mesh via regrid_2d_to_fixed.
    """
    path = default_parent_path()
    if path is None or not path.exists():
        raise RuntimeError("No parent.atm file found for the configured run.")
    records = read_hafs_tp_records(path)
    if not records:
        raise RuntimeError(f"No 'tp' (APCP) records in parent file {path}.")
    rec = pick_cumulative_record(records)
    print(f"  parent 0->{rec['end_step']}h, grid {rec['lats'].shape}, "
          f"max {np.nanmax(rec['data']):.0f} mm")
    return regrid_2d_to_fixed(rec["lats"], rec["lons"], rec["data"],
                              grid_lat, grid_lon)


def stage4_on_fixed(max_fhour, grid_lat, grid_lon):
    """Stage IV touched-days total (parent_qpf.stage4_total) on the fixed mesh.

    stage4_total returns its field already masked to parent_qpf's 750 km
    display swath; an unmasked variant is not exposed. We accept that mask
    because it is wider than the 500 km verification swath, so the tighter
    tc_swath_mask applied later governs the scored footprint. Stage IV is
    CONUS-only, so ocean points regrid to NaN and drop out automatically.
    """
    s4_lat, s4_lon, s4_total, s4_label = stage4_total(
        INIT_DT, max_fhour, STAGE4_CACHE_DIR)
    if s4_total is None:
        return None, "unavailable"
    grid = regrid_2d_to_fixed(s4_lat, s4_lon, s4_total, grid_lat, grid_lon)
    return grid, s4_label


# obs -> color, forecast -> linestyle/marker, so 4 curves stay legible.
_OBS_COLOR = {"MRMS": "#1f77b4", "Stage IV": "#2ca02c"}
_FCST_STYLE = {"parent": dict(ls="-", marker="o"),
               "nest": dict(ls="--", marker="s")}


def plot_curves(results, max_fhour, out_path, caveat=""):
    """results: list of dicts {forecast, observation, rows, n_valid}."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for res in results:
        rows = res["rows"]
        if not rows:
            continue
        thr = [r["threshold"] for r in rows]
        ets = [r["ets"] for r in rows]
        style = _FCST_STYLE.get(res["forecast"], dict(ls="-", marker="o"))
        ax.plot(thr, ets, color=_OBS_COLOR.get(res["observation"], "gray"),
                lw=2, **style,
                label=f"{res['forecast']} vs {res['observation']} "
                      f"(n={res['n_valid']:,})")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(THRESHOLDS_MM)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Rainfall threshold (mm)")
    ax.set_ylabel("Equitable Threat Score (ETS)")
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"Hurricane Helene — HAFS-A QPF ETS vs MRMS & Stage IV\n"
        f"0–{max_fhour}h | init {INIT_DT:%Y-%m-%d %HZ} | "
        f"TC swath ≤{TC_MASK_RADIUS_KM:.0f} km"
    )
    if caveat:
        fig.text(0.5, -0.02, caveat, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    file_pairs = discover_files(HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER)
    if not file_pairs:
        print(f"No files matching {FILE_GLOB} in {HAFS_RUN_DIR}")
        return
    max_fhour = file_pairs[-1][0]
    print(f"Init {INIT_DT:%Y-%m-%d %HZ} | accumulation 0–{max_fhour}h")

    grid_lat, grid_lon = build_fixed_grid()
    print(f"Fixed grid: {grid_lat.shape[0]}x{grid_lat.shape[1]} @ {GRID_RES}deg")

    print("\nHAFS nest total ...")
    nest_total, apcp_mode = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  nest APCP mode: {apcp_mode}, max {np.nanmax(nest_total):.0f} mm")

    print("HAFS parent total ...")
    parent_total = hafs_parent_total(grid_lat, grid_lon)

    print("MRMS total ...")
    mrms_total = build_mrms_total(max_fhour, grid_lat, grid_lon)

    print("Stage IV total ...")
    stage4_grid, s4_label = stage4_on_fixed(max_fhour, grid_lat, grid_lon)

    print("TC verification swath ...")
    swath = tc_swath_mask(max_fhour, grid_lat, grid_lon)

    forecasts = [("parent", parent_total), ("nest", nest_total)]
    observations = [("MRMS", mrms_total)]
    if stage4_grid is not None:
        observations.append(("Stage IV", stage4_grid))
    else:
        print("  Stage IV unavailable — scoring MRMS only.")

    results = []
    print("\n" + "=" * 84)
    for fname, fgrid in forecasts:
        for oname, ogrid in observations:
            rows, n_valid = score_pair(fgrid, ogrid, swath,
                                       THRESHOLDS_MM, contingency_scores)
            results.append(dict(forecast=fname, observation=oname,
                                rows=rows, n_valid=n_valid))
            print(f"\n{fname} vs {oname}  (n_valid={n_valid:,})")
            print(f"{'thr':>5} {'a':>7} {'b':>7} {'c':>7} {'d':>7} {'ETS':>7} "
                  f"{'bias':>6} {'POD':>6} {'FAR':>6} {'CSI':>6}")
            for r in rows:
                print(f"{r['threshold']:>5} {r['a']:>7} {r['b']:>7} {r['c']:>7} "
                      f"{r['d']:>7} {r['ets']:>7.3f} {r['bias']:>6.2f} {r['pod']:>6.2f} "
                      f"{r['far']:>6.2f} {r['csi']:>6.2f}")
    print("=" * 84)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["forecast", "observation", "threshold", "a", "b", "c", "d",
                  "ets", "bias", "pod", "far", "csi"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            for r in res["rows"]:
                w.writerow({"forecast": res["forecast"],
                            "observation": res["observation"], **r})
    print(f"\nSaved table: {OUT_CSV}")

    if stage4_grid is None:
        caveat = "Stage IV unavailable — not scored."
    else:
        caveat = (f"Stage IV: CONUS-only, 24h 12Z–12Z files summed over touched "
                  f"days ({s4_label}) — window approximates the 0–{max_fhour}h "
                  f"forecast accumulation.")
    print(caveat)
    plot_curves(results, max_fhour, OUT_PNG, caveat=caveat)
    print(f"Saved plot : {OUT_PNG}")


if __name__ == "__main__":
    main()
