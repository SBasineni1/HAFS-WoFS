"""
HAFS-A PARENT-domain QPF vs MRMS QPE — single static comparison.

This is the field the operational viewers (e.g. Tropical Tidbits' "HAFS-A
Parent Model — Total Accumulated Precip") actually draw.  The 6-km parent
domain is FIXED, so its 0->fhour cumulative APCP (tp) accumulates geographically
correctly — no moving-nest storm-relative inflation, no bucket-summing, no
eyewall scallops, and no dependence on storm translation speed.  The whole-run
storm total is just the cumulative tp in the last forecast-hour file.

Contrast:
  * qpf_full_run.py  -> 2-km moving NEST (storm.atm): high-res but its
    cumulative tp is storm-relative garbage, so we sum 3-h buckets and still
    get speed-dependent eyewall arcs.
  * parent_qpf.py    -> 6-km PARENT (parent.atm): coarser, lower local peaks,
    but geographically honest and viewer-consistent.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/parent_qpf.py [/path/to/parent.atm.fXXX.grb2]

If no path is given, the highest forecast-hour parent.atm file for the
configured run is found automatically.  Reuses the GRIB2 / MRMS plumbing from
qpf_full_run.py.
"""

import re
import sys
from pathlib import Path
from datetime import timedelta

# Make the sibling module importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from qpf_full_run import (
    HAFS_RUN_DIR, INIT_STR, INIT_DT, FIXED_DOMAIN,
    TC_MASK_RADIUS_KM, OUT_DIR, MRMS_CACHE_DIR,
    QPF_LEVELS, QPF_COLORS,
    read_hafs_tp_records,
    tc_position_at, haversine_km, load_mrms_hour, crop_to_domain,
)

PARENT_PNG = OUT_DIR.parent / "parent_qpf_helene_hfsa.png"


# =============================================================================
# Parent file discovery + cumulative-APCP selection
# =============================================================================

def default_parent_path():
    """Highest forecast-hour parent.atm file for the configured run."""
    hits = sorted(HAFS_RUN_DIR.glob(f"**/*{INIT_STR}*parent.atm.f*.grb2"))
    if not hits:
        return None

    def fhour(p):
        m = re.search(r"\.f(\d{3})\.grb2$", p.name)
        return int(m.group(1)) if m else -1

    return max(hits, key=fhour)


def pick_cumulative_record(records):
    """Pick the 0->fhour cumulative APCP record on the (fixed) parent grid.

    On a fixed grid the cumulative total IS the geographic storm total, so we
    take the longest 0-> window directly — no fmax, no bucket summing.
    """
    if not records:
        return None
    finest = max(r["npoints"] for r in records)
    cand = [r for r in records if r["npoints"] == finest]
    cumulative = [r for r in cand if r["start_step"] in (0, None)]
    pool = cumulative or cand
    return max(pool, key=lambda r: (r["end_step"] or 0))


# =============================================================================
# Plotting
# =============================================================================

def qpf_cmap():
    cmap = mcolors.ListedColormap(QPF_COLORS)
    norm = mcolors.BoundaryNorm(QPF_LEVELS, cmap.N)
    return cmap, norm


def plot_parent(end_fhour, hafs_lons, hafs_lats, hafs_mm,
                mrms_lons, mrms_lats, mrms_mm, domain, out_path):
    valid_dt = INIT_DT + timedelta(hours=end_fhour)
    lat_min, lat_max, lon_min, lon_max = domain
    cmap, norm = qpf_cmap()

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for ax in axes:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                          linestyle="--", alpha=0.5)
        gl.top_labels = gl.right_labels = False

    cf = axes[0].contourf(
        hafs_lons, hafs_lats, hafs_mm,
        levels=QPF_LEVELS, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), extend="max",
    )
    axes[0].set_title(
        f"HAFS-A Parent-Domain Accumulated Precip (APCP)\n"
        f"Init {INIT_DT.strftime('%Y-%m-%d %HZ')} | "
        f"0–{end_fhour}h (valid {valid_dt.strftime('%Y-%m-%d %HZ')})"
    )

    if mrms_mm is not None:
        axes[1].contourf(
            mrms_lons, mrms_lats, mrms_mm,
            levels=QPF_LEVELS, cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(), extend="max",
        )
        axes[1].set_title(
            f"MRMS MultiSensor QPE (Pass2)\n"
            f"{end_fhour}h accumulation "
            f"({INIT_DT.strftime('%Y-%m-%d %HZ')} – {valid_dt.strftime('%Y-%m-%d %HZ')})"
        )
    else:
        axes[1].text(0.5, 0.5, "MRMS data unavailable",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("MRMS MultiSensor QPE — unavailable")

    plt.colorbar(cf, ax=axes, label="Accumulated Precipitation (mm)",
                 ticks=QPF_LEVELS, shrink=0.7, fraction=0.02)
    fig.suptitle(
        f"Hurricane Helene — HAFS-A parent-domain QPF vs MRMS QPE "
        f"(0–{end_fhour}h)",
        fontsize=13, y=1.01,
    )
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MRMS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_parent_path()
    if path is None or not path.exists():
        print(f"No parent.atm file found under {HAFS_RUN_DIR} "
              f"(matching *{INIT_STR}*parent.atm.f*.grb2)")
        return
    print(f"Parent file: {path}")

    # ------------------------------------------------------------------
    # HAFS parent-domain cumulative APCP (fixed grid -> geographic total).
    # ------------------------------------------------------------------
    records = read_hafs_tp_records(path)
    if not records:
        print("\nNo 'tp' (APCP) records in the parent file.")
        print(f"Run  python analysis/inspect_grib.py {path}  to list its fields.")
        return
    print(f"\nFound {len(records)} tp record(s):")
    for i, r in enumerate(records):
        print(f"  [{i}] window {r['start_step']}->{r['end_step']}h "
              f"stepType={r['step_type']} npoints={r['npoints']:,} "
              f"max={np.nanmax(r['data']):.0f} mm")

    rec = pick_cumulative_record(records)
    end_fhour = rec["end_step"] or 0
    hafs_lats, hafs_lons, hafs_mm = rec["lats"], rec["lons"], rec["data"]
    print(f"\nUsing 0->{end_fhour}h cumulative record, grid {hafs_lats.shape}, "
          f"max {np.nanmax(hafs_mm):.0f} mm ({np.nanmax(hafs_mm)/25.4:.1f} in)")

    # TC rainfall swath mask on the parent grid (union of 500 km circles along
    # the best track for hours 0..end_fhour) — same footprint as MRMS below.
    hafs_swath = np.zeros(hafs_lats.shape, dtype=bool)
    for h in range(0, end_fhour + 1):
        tlat, tlon = tc_position_at(INIT_DT + timedelta(hours=h))
        hafs_swath |= haversine_km(tlat, tlon, hafs_lats, hafs_lons) <= TC_MASK_RADIUS_KM
    hafs_display = np.where(hafs_swath, np.nan_to_num(hafs_mm, nan=0.0), 0.0)

    # ------------------------------------------------------------------
    # MRMS total over the same 0->end_fhour window, masked to the swath.
    # ------------------------------------------------------------------
    lat_min, lat_max, lon_min, lon_max = FIXED_DOMAIN
    print(f"\nAccumulating MRMS 1H QPE over hours 1–{end_fhour} ...")
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    mrms_lats = mrms_lons = None
    mrms_total = None
    for h in range(1, end_fhour + 1):
        t = INIT_DT + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, MRMS_CACHE_DIR)
            clat, clon, cdata = crop_to_domain(lat, lon, data,
                                               lat_min, lat_max, lon_min, lon_max)
        except Exception as e:
            print(f"  h{h:03d} unavailable: {e}")
            continue
        if mrms_total is None:
            mrms_lats, mrms_lons = clat, clon
            mrms_total = np.zeros((clat.size, clon.size))
        tlat, tlon = tc_position_at(t)
        mlon2d, mlat2d = np.meshgrid(mrms_lons, mrms_lats)
        dist = haversine_km(tlat, tlon, mlat2d, mlon2d)
        mrms_total += np.where(dist <= TC_MASK_RADIUS_KM, cdata, 0.0)
        if h % 12 == 0 or h == end_fhour:
            print(f"  cached h{h:03d}/{end_fhour} ({t.strftime('%Y-%m-%d %HZ')})")

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------
    plot_parent(end_fhour, hafs_lons, hafs_lats, hafs_display,
                mrms_lons, mrms_lats, mrms_total, FIXED_DOMAIN, PARENT_PNG)
    hafs_max = float(np.nanmax(hafs_display))
    mrms_max = float(np.nanmax(mrms_total)) if mrms_total is not None else float("nan")
    print(f"\nSaved {PARENT_PNG}")
    print(f"  HAFS parent max {hafs_max:.0f} mm | MRMS max {mrms_max:.0f} mm")


if __name__ == "__main__":
    main()
