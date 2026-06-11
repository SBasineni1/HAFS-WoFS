"""
Equitable Threat Score (ETS / Gilbert Skill Score) for HAFS-A QPF vs MRMS QPE.

For the full Helene event (0–Nh accumulation) this script:
  1. Builds the HAFS-A storm-total precip on a fixed lat/lon grid by taking the
     running max of every forecast-hour's storm-nest tp field (same approach as
     the animation, so accumulation sticks to geography as the nest moves).
  2. Accumulates the matching MRMS 1H QPE over the same window and regrids it
     onto the identical fixed grid.
  3. Restricts verification to the TC rainfall swath — grid points within
     TC_MASK_RADIUS_KM of Helene's best track at any hour — so non-Helene
     synoptic rain doesn't pollute the contingency tables.
  4. For a set of rainfall thresholds, builds the 2x2 contingency table
     (a=hits, b=false alarms, c=misses, d=correct negatives) and computes

        ETS = (a - a_ref) / (a + b + c - a_ref),   a_ref = (a+b)(a+c)/n

     along with frequency bias, POD, FAR, and CSI for context.

ETS ranges from -1/3 to 1; 0 = no skill over random, 1 = perfect.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/ets_score.py

Reuses the GRIB2 / MRMS plumbing from qpf_full_run.py.
"""

import sys
import csv
from pathlib import Path
from datetime import timedelta

# Make the sibling module importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpf_full_run import (
    HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER, FIXED_DOMAIN, GRID_RES,
    TC_MASK_RADIUS_KM, INIT_DT, OUT_DIR, MRMS_CACHE_DIR,
    MRMS_BUCKET, MRMS_PRODUCT,
    discover_files, hafs_event_total,
    tc_position_at, haversine_km, load_mrms_hour,
)

# Rainfall thresholds (mm) at which to evaluate skill.
THRESHOLDS_MM = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250]

ETS_PNG = OUT_DIR.parent / "ets_helene_hfsa.png"
ETS_CSV = OUT_DIR.parent / "ets_helene_hfsa.csv"


def regrid_mrms_to_fixed(mlat, mlon, mdata, grid_lat, grid_lon):
    """Bilinear-interpolate native MRMS (regular 1-D lat/lon) onto the fixed mesh."""
    mlat = np.asarray(mlat)
    mlon = np.asarray(mlon, dtype=float)
    # cfgrib returns MRMS longitude in 0–360; the fixed grid (and HAFS) use
    # −180…180, so convert or every query falls outside the interpolator.
    mlon = np.where(mlon > 180, mlon - 360, mlon)
    # RegularGridInterpolator needs strictly ascending axes.
    if mlat[0] > mlat[-1]:
        mlat = mlat[::-1]
        mdata = mdata[::-1, :]
    if mlon[0] > mlon[-1]:
        mlon = mlon[::-1]
        mdata = mdata[:, ::-1]
    interp = RegularGridInterpolator(
        (mlat, mlon), mdata, bounds_error=False, fill_value=np.nan
    )
    pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
    return interp(pts).reshape(grid_lat.shape)


def build_mrms_total(max_fhour, grid_lat, grid_lon):
    """Accumulate MRMS 1H QPE over 1..max_fhour on its native grid, then regrid."""
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    mrms_sum = None
    mlat = mlon = None
    for h in range(1, max_fhour + 1):
        t = INIT_DT + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, MRMS_CACHE_DIR)
            if mrms_sum is None:
                mlat, mlon = lat, lon
                mrms_sum = np.zeros_like(data)
            mrms_sum += data
            if h % 12 == 0 or h == max_fhour:
                print(f"  MRMS h{h:03d}/{max_fhour} ({t.strftime('%Y-%m-%d %HZ')})")
        except Exception as e:
            print(f"  MRMS h{h:03d} unavailable: {e}")
    if mrms_sum is None:
        raise RuntimeError("No MRMS hours could be loaded.")
    return regrid_mrms_to_fixed(mlat, mlon, mrms_sum, grid_lat, grid_lon)


def tc_swath_mask(max_fhour, grid_lat, grid_lon):
    """Boolean mask: grid points within TC_MASK_RADIUS_KM of the track at any hour."""
    swath = np.zeros(grid_lat.shape, dtype=bool)
    for h in range(0, max_fhour + 1):
        tlat, tlon = tc_position_at(INIT_DT + timedelta(hours=h))
        dist = haversine_km(tlat, tlon, grid_lat, grid_lon)
        swath |= dist <= TC_MASK_RADIUS_KM
    return swath


def contingency_scores(fcst, obs, threshold):
    """2x2 contingency table + skill scores at one threshold over 1-D arrays."""
    fy = fcst >= threshold
    oy = obs >= threshold
    a = int(np.sum(fy & oy))      # hits
    b = int(np.sum(fy & ~oy))     # false alarms
    c = int(np.sum(~fy & oy))     # misses
    d = int(np.sum(~fy & ~oy))    # correct negatives
    n = a + b + c + d
    a_ref = (a + b) * (a + c) / n if n > 0 else 0.0
    denom = (a + b + c) - a_ref
    ets = (a - a_ref) / denom if denom != 0 else np.nan
    bias = (a + b) / (a + c) if (a + c) > 0 else np.nan
    pod = a / (a + c) if (a + c) > 0 else np.nan
    far = b / (a + b) if (a + b) > 0 else np.nan
    csi = a / (a + b + c) if (a + b + c) > 0 else np.nan
    return dict(threshold=threshold, a=a, b=b, c=c, d=d,
                ets=ets, bias=bias, pod=pod, far=far, csi=csi)


def main():
    file_pairs = discover_files(HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER)
    if not file_pairs:
        print(f"No files matching {FILE_GLOB} in {HAFS_RUN_DIR}")
        return
    max_fhour = file_pairs[-1][0]
    print(f"HAFS files   : {len(file_pairs)} (through F{max_fhour:03d})")
    print(f"Init time    : {INIT_DT.strftime('%Y-%m-%d %HZ')}")
    print(f"Accumulation : 0–{max_fhour}h")

    lat_min, lat_max, lon_min, lon_max = FIXED_DOMAIN
    fixed_lons = np.arange(lon_min, lon_max + GRID_RES, GRID_RES)
    fixed_lats = np.arange(lat_min, lat_max + GRID_RES, GRID_RES)
    grid_lon, grid_lat = np.meshgrid(fixed_lons, fixed_lats)
    print(f"Fixed grid   : {grid_lat.shape[0]}x{grid_lat.shape[1]} @ {GRID_RES}deg")

    print("\nBuilding HAFS storm-total (accumulation-aware) ...")
    hafs_total, apcp_mode = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  HAFS APCP mode: {apcp_mode}")

    print("\nAccumulating + regridding MRMS QPE ...")
    mrms_total = build_mrms_total(max_fhour, grid_lat, grid_lon)

    print("\nBuilding TC verification swath ...")
    swath = tc_swath_mask(max_fhour, grid_lat, grid_lon)

    valid = swath & np.isfinite(mrms_total)
    n_valid = int(np.sum(valid))
    print(f"  Verification points: {n_valid:,} "
          f"({100*n_valid/swath.size:.1f}% of grid)")
    if n_valid == 0:
        print("  Swath points:", int(np.sum(swath)),
              "| finite MRMS points:", int(np.sum(np.isfinite(mrms_total))))
        print("  No overlapping valid points — check MRMS lon/lat alignment.")
        return

    fcst = np.nan_to_num(hafs_total[valid], nan=0.0)
    obs = np.nan_to_num(mrms_total[valid], nan=0.0)
    print(f"  HAFS max {fcst.max():.0f} mm | MRMS max {obs.max():.0f} mm "
          f"(within swath)")

    print("\n" + "=" * 78)
    print(f"{'thr(mm)':>7} {'hits':>8} {'FA':>8} {'miss':>8} {'corrNeg':>10} "
          f"{'ETS':>7} {'bias':>6} {'POD':>6} {'FAR':>6} {'CSI':>6}")
    print("-" * 78)
    rows = []
    for thr in THRESHOLDS_MM:
        s = contingency_scores(fcst, obs, thr)
        rows.append(s)
        print(f"{s['threshold']:>7} {s['a']:>8} {s['b']:>8} {s['c']:>8} "
              f"{s['d']:>10} {s['ets']:>7.3f} {s['bias']:>6.2f} "
              f"{s['pod']:>6.2f} {s['far']:>6.2f} {s['csi']:>6.2f}")
    print("=" * 78)

    # CSV
    ETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(ETS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved table: {ETS_CSV}")

    # Plot ETS (and frequency bias) vs threshold
    thr = [r["threshold"] for r in rows]
    ets = [r["ets"] for r in rows]
    bias = [r["bias"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(thr, ets, "o-", color="#1f77b4", lw=2, label="ETS")
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xticks(thr)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("Rainfall threshold (mm)")
    ax1.set_ylabel("Equitable Threat Score (ETS)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(-0.05, 1.0)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(thr, bias, "s--", color="#d62728", lw=1.5, alpha=0.8,
             label="Frequency bias")
    ax2.axhline(1.0, color="#d62728", ls=":", lw=0.8, alpha=0.6)
    ax2.set_ylabel("Frequency bias", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(
        f"Hurricane Helene — HAFS-A vs MRMS QPE\n"
        f"ETS by threshold | 0–{max_fhour}h accumulation | "
        f"init {INIT_DT.strftime('%Y-%m-%d %HZ')} | "
        f"TC swath ≤{TC_MASK_RADIUS_KM:.0f} km"
    )
    fig.tight_layout()
    fig.savefig(ETS_PNG, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot : {ETS_PNG}")


if __name__ == "__main__":
    main()
