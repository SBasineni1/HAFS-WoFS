"""
HAFS-A native rainfall SWATH vs MRMS QPE — single static comparison.

Instead of reconstructing the storm-total precip by running-max'ing the moving
storm-nest tp (qpf_full_run.py / ets_score.py), this reads HAFS's own swath
product:

    09l.<init>.hfsa.parent.swath.grb2

The swath file lives on the FIXED parent grid and already carries the
whole-run accumulated precipitation, so there's no reprojection and no fmax —
which removes the west-side overestimation artifacts that the moving-nest
running-max introduces (the wide nest sweeps in non-Helene/frontal rain and
fmax stamps the highest cumulative value at every point it ever covered).

MRMS QPE is accumulated over the same forecast window (0 -> end of the swath
accumulation) and both panels are masked to the same TC rainfall swath so only
Helene's rain is compared.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/swath_qpf.py [/path/to/parent.swath.grb2]

If no path is given, the parent.swath file for the configured run is found
automatically.  Reuses the GRIB2 / MRMS plumbing from qpf_full_run.py.
"""

import sys
from pathlib import Path
from datetime import timedelta

# Make the sibling module importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import eccodes
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
    tc_position_at, haversine_km, load_mrms_hour, crop_to_domain,
)

SWATH_PNG = OUT_DIR.parent / "swath_helene_hfsa.png"


# =============================================================================
# Swath file discovery + precip record selection
# =============================================================================

def default_swath_path():
    hits = sorted(HAFS_RUN_DIR.glob(f"**/*{INIT_STR}*parent.swath*.grb2"))
    return hits[0] if hits else None


def read_swath_prate_records(filepath):
    """Read every 'prate' (precipitation rate) record from a HAFS swath file.

    The swath product has NO accumulated tp/APCP field — it stores the
    time-MEAN precipitation rate over each 0->Nh window (stepType 'avg', units
    kg m**-2 s**-1).  Total accumulated rainfall over the window is then

        total_mm = avg_rate * window_seconds      (1 kg m**-2 == 1 mm)

    Returns a list of dicts: npoints, lats, lons (-180..180), rate (kg/m2/s,
    fill->NaN), start_step, end_step, step_type.  Same eccodes plumbing as
    qpf_full_run.read_hafs_tp_records, just filtered on shortName 'prate'.
    """
    records = []
    with open(str(filepath), "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                try:
                    sn = eccodes.codes_get(gid, "shortName", ktype=str)
                except Exception:
                    continue
                if sn != "prate":
                    continue
                nj = eccodes.codes_get(gid, "Nj")
                ni = eccodes.codes_get(gid, "Ni")
                lat0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                lon0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
                lat1 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
                lon1 = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")

                def _opt(key, cast=int):
                    try:
                        return cast(eccodes.codes_get(gid, key))
                    except Exception:
                        return None

                start_step = _opt("startStep")
                end_step = _opt("endStep")
                step_type = _opt("stepType", cast=str)

                vals = eccodes.codes_get_values(gid)
                lats_1d = np.linspace(lat0, lat1, nj)
                lons_1d = np.linspace(lon0, lon1, ni)
                lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)
                lons_180 = np.where(lons_2d > 180, lons_2d - 360, lons_2d)
                rate = vals.reshape(nj, ni)
                missing = eccodes.codes_get(gid, "missingValue")
                rate = np.where(np.abs(rate - missing) < 1.0, np.nan, rate)
                rate = np.where(rate < 0, np.nan, rate)
                records.append({
                    "npoints": ni * nj,
                    "lats": lats_2d,
                    "lons": lons_180,
                    "rate": rate,
                    "start_step": start_step,
                    "end_step": end_step,
                    "step_type": step_type,
                })
            except Exception:
                pass
            finally:
                eccodes.codes_release(gid)
    return records


def pick_swath_precip(records):
    """Pick the whole-run prate record (longest 0->Nh averaging window)."""
    if not records:
        return None

    def window(r):
        return (r["end_step"] or 0) - (r["start_step"] or 0)

    def from_init(r):
        return 1 if r["start_step"] in (0, None) else 0

    return max(records, key=lambda r: (window(r), from_init(r), r["end_step"] or 0))


# =============================================================================
# Plotting
# =============================================================================

def qpf_cmap():
    cmap = mcolors.ListedColormap(QPF_COLORS)
    norm = mcolors.BoundaryNorm(QPF_LEVELS, cmap.N)
    return cmap, norm


def plot_swath(end_fhour, hafs_lons, hafs_lats, hafs_mm,
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
        f"HAFS-A Rainfall Swath (native parent product)\n"
        f"Init {INIT_DT.strftime('%Y-%m-%d %HZ')} | "
        f"0–{end_fhour}h accumulation (valid {valid_dt.strftime('%Y-%m-%d %HZ')})"
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
        f"Hurricane Helene — HAFS-A native rainfall swath vs MRMS QPE "
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

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_swath_path()
    if path is None or not path.exists():
        print(f"No parent.swath file found under {HAFS_RUN_DIR} "
              f"(matching *{INIT_STR}*parent.swath*.grb2)")
        return
    print(f"Swath file : {path}")

    # ------------------------------------------------------------------
    # HAFS native swath precip on the fixed parent grid.
    # The swath stores time-MEAN precip rate (prate, kg/m2/s) per 0->Nh window,
    # not accumulated tp.  Total rainfall = avg_rate * window_seconds.
    # ------------------------------------------------------------------
    records = read_swath_prate_records(path)
    if not records:
        print("\nNo 'prate' records in the swath file.")
        print("Run  python analysis/inspect_grib.py "
              f"{path}  to list its fields.")
        return
    print(f"\nFound {len(records)} prate record(s):")
    for i, r in enumerate(records):
        win_s = ((r["end_step"] or 0) - (r["start_step"] or 0)) * 3600
        peak_mm = np.nanmax(r["rate"]) * win_s
        print(f"  [{i}] window {r['start_step']}->{r['end_step']}h "
              f"stepType={r['step_type']} npoints={r['npoints']:,} "
              f"max_rate={np.nanmax(r['rate']):.2e} kg/m2/s "
              f"(~{peak_mm:.0f} mm total)")

    rec = pick_swath_precip(records)
    end_fhour = rec["end_step"] or 0
    window_s = ((rec["end_step"] or 0) - (rec["start_step"] or 0)) * 3600
    hafs_lats, hafs_lons = rec["lats"], rec["lons"]
    # Convert mean rate over the window to total accumulated rainfall (mm).
    hafs_mm = rec["rate"] * window_s
    print(f"\nUsing 0->{end_fhour}h prate record, grid {hafs_lats.shape}, "
          f"converted to total rainfall max {np.nanmax(hafs_mm):.0f} mm "
          f"({np.nanmax(hafs_mm)/25.4:.1f} in)")

    # TC rainfall swath mask on the HAFS grid (union of 500 km circles along
    # the best track for hours 0..end_fhour) — same idea as the other scripts.
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
        # Mask this hour's QPE to within 500 km of the TC center.
        tlat, tlon = tc_position_at(t)
        mlon2d, mlat2d = np.meshgrid(mrms_lons, mrms_lats)
        dist = haversine_km(tlat, tlon, mlat2d, mlon2d)
        mrms_total += np.where(dist <= TC_MASK_RADIUS_KM, cdata, 0.0)
        if h % 12 == 0 or h == end_fhour:
            print(f"  cached h{h:03d}/{end_fhour} ({t.strftime('%Y-%m-%d %HZ')})")

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------
    plot_swath(end_fhour, hafs_lons, hafs_lats, hafs_display,
               mrms_lons, mrms_lats, mrms_total, FIXED_DOMAIN, SWATH_PNG)
    hafs_max = float(np.nanmax(hafs_display))
    mrms_max = float(np.nanmax(mrms_total)) if mrms_total is not None else float("nan")
    print(f"\nSaved {SWATH_PNG}")
    print(f"  HAFS swath max {hafs_max:.0f} mm | MRMS max {mrms_max:.0f} mm")


if __name__ == "__main__":
    main()
