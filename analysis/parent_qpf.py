"""
HAFS-A PARENT-domain QPF vs MRMS QPE vs Stage IV QPE — static 3-panel.

The parent panel is the field operational viewers (e.g. Tropical Tidbits'
"HAFS-A Parent Model — Total Accumulated Precip") draw.  The 6-km parent domain
is FIXED, so its 0->fhour cumulative APCP (tp) accumulates geographically
correctly — no moving-nest storm-relative inflation, no bucket-summing, no
eyewall scallops.  The whole-run storm total is just the cumulative tp in the
last forecast-hour file.

Two observed-QPE references are shown for comparison of rainfall amounts:
  * MRMS MultiSensor QPE (Pass2), 1-hourly, from the noaa-mrms-pds S3 bucket.
  * NCEP Stage IV QPE, 6-hourly, from the NOAA water.noaa.gov daily tarballs.
    Stage IV is gauge+radar and is the classic verification benchmark; it is
    CONUS-only, so Helene's Gulf/Caribbean rain won't appear in that panel.

Contrast with qpf_full_run.py (2-km moving NEST: high-res but its cumulative
tp is storm-relative, so it needs 3-h bucket summing and still scallops with
storm speed).  The parent domain is coarser (lower peaks) but geographically
honest and viewer-consistent.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/parent_qpf.py [/path/to/parent.atm.fXXX.grb2]

Reuses the GRIB2 / MRMS plumbing from qpf_full_run.py.
"""

import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta

# Make the sibling module importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cfgrib
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

# Stage IV QPE — NOAA water.noaa.gov daily source tarballs (GRIB2 since 2020).
# Each daily tar is valid 12Z->12Z and holds 1h/6h/24h accumulation files.
STAGE4_BASE = "https://water.noaa.gov/resources/downloads/precip/stageIV"
STAGE4_CACHE_DIR = Path("/tmp/stage4_cache")


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
# Stage IV QPE downloader / accumulator
# =============================================================================

def stage4_tar_url(day):
    return (f"{STAGE4_BASE}/{day:%Y}/{day:%m}/{day:%d}/"
            f"ncep_stage_iv_source_files_{day:%Y%m%d}.tar")


def ensure_stage4_files(start_dt, end_dt, cache_dir):
    """Download + extract the daily Stage IV tarballs covering [start, end].

    Tarballs are valid 12Z->12Z, so we fetch from the day BEFORE start through
    end to be safe.  A per-day marker file makes this idempotent (cached).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    day = (start_dt - timedelta(days=1)).date()
    last = end_dt.date()
    while day <= last:
        marker = cache_dir / f".{day:%Y%m%d}.done"
        if not marker.exists():
            url = stage4_tar_url(day)
            tar_path = cache_dir / f"st4_{day:%Y%m%d}.tar"
            try:
                urllib.request.urlretrieve(url, tar_path)
                with tarfile.open(tar_path) as tf:
                    tf.extractall(cache_dir)
                marker.write_text("ok")
            except Exception as e:
                print(f"  Stage IV tar {day} failed: {e}")
            finally:
                tar_path.unlink(missing_ok=True)
        day += timedelta(days=1)


def index_stage4_6h(cache_dir):
    """Map valid-datetime -> 6h Stage IV grib path among extracted files."""
    idx = {}
    for p in cache_dir.rglob("*.grb2"):
        m = re.search(r"(\d{10})\.06h", p.name)
        if m:
            idx[datetime.strptime(m.group(1), "%Y%m%d%H")] = p
    return idx


def read_stage4(path):
    """Return (lat2d, lon2d, data_mm) for a Stage IV 6h file (fill/neg -> 0)."""
    for ds in cfgrib.open_datasets(str(path)):
        for var in ds.data_vars:
            da = ds[var]
            data = np.where(da.values < 0, 0.0, da.values)
            data = np.nan_to_num(data, nan=0.0)
            lat = da.latitude.values
            lon = da.longitude.values
            lon = np.where(lon > 180, lon - 360, lon)
            return lat, lon, data
    raise RuntimeError(f"no variable in {path}")


def stage4_total(start_dt, end_dt, cache_dir):
    """Sum 6-hourly Stage IV over (start_dt, end_dt], masked to the TC swath.

    Returns (lat2d, lon2d, total_mm) or (None, None, None) if unavailable.
    """
    ensure_stage4_files(start_dt, end_dt, cache_dir)
    idx = index_stage4_6h(cache_dir)
    if not idx:
        return None, None, None

    total = lat2d = lon2d = None
    t = start_dt + timedelta(hours=6)
    while t <= end_dt:
        path = idx.get(t)
        if path is None:
            print(f"  Stage IV 6h missing for {t:%Y-%m-%d %HZ}")
            t += timedelta(hours=6)
            continue
        try:
            lat, lon, data = read_stage4(path)
        except Exception as e:
            print(f"  Stage IV read failed {t:%Y-%m-%d %HZ}: {e}")
            t += timedelta(hours=6)
            continue
        if total is None:
            lat2d, lon2d = lat, lon
            total = np.zeros_like(data)
        # Mask this 6h field to within TC_MASK_RADIUS_KM of the TC center.
        tlat, tlon = tc_position_at(t)
        dist = haversine_km(tlat, tlon, lat2d, lon2d)
        total += np.where(dist <= TC_MASK_RADIUS_KM, data, 0.0)
        t += timedelta(hours=6)
    return lat2d, lon2d, total


# =============================================================================
# Plotting
# =============================================================================

def qpf_cmap():
    cmap = mcolors.ListedColormap(QPF_COLORS)
    norm = mcolors.BoundaryNorm(QPF_LEVELS, cmap.N)
    return cmap, norm


def plot_compare(panels, end_fhour, domain, out_path):
    """panels: list of (lons, lats, data_mm, title); data may be None."""
    lat_min, lat_max, lon_min, lon_max = domain
    cmap, norm = qpf_cmap()

    fig, axes = plt.subplots(
        1, len(panels), figsize=(8 * len(panels), 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if len(panels) == 1:
        axes = [axes]

    cf = None
    for ax, (lons, lats, data, title) in zip(axes, panels):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                          linestyle="--", alpha=0.5)
        gl.top_labels = gl.right_labels = False
        if data is not None:
            cf = ax.contourf(
                lons, lats, data, levels=QPF_LEVELS, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(), extend="max",
            )
        else:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(title)

    if cf is not None:
        plt.colorbar(cf, ax=axes, label="Accumulated Precipitation (mm)",
                     ticks=QPF_LEVELS, shrink=0.7, fraction=0.02)
    valid_dt = INIT_DT + timedelta(hours=end_fhour)
    fig.suptitle(
        f"Hurricane Helene — HAFS-A parent QPF vs MRMS vs Stage IV "
        f"(0–{end_fhour}h, valid {valid_dt:%Y-%m-%d %HZ})",
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

    valid_end = INIT_DT + timedelta(hours=end_fhour)

    # TC rainfall swath mask on the parent grid (union of 500 km circles along
    # the best track for hours 0..end_fhour) — same footprint as the QPE below.
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
    mrms_lats = mrms_lons = mrms_total = None
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
            print(f"  cached h{h:03d}/{end_fhour} ({t:%Y-%m-%d %HZ})")

    # ------------------------------------------------------------------
    # Stage IV total over the same window (6-hourly), masked to the swath.
    # ------------------------------------------------------------------
    print(f"\nAccumulating Stage IV 6H QPE over {INIT_DT:%Y-%m-%d %HZ} "
          f"-> {valid_end:%Y-%m-%d %HZ} ...")
    s4_lats, s4_lons, s4_total = stage4_total(INIT_DT, valid_end, STAGE4_CACHE_DIR)
    if s4_total is None:
        print("  Stage IV unavailable (download/extract failed).")

    # ------------------------------------------------------------------
    # Plot 3-panel.
    # ------------------------------------------------------------------
    panels = [
        (hafs_lons, hafs_lats, hafs_display,
         f"HAFS-A Parent APCP\n0–{end_fhour}h (valid {valid_end:%Y-%m-%d %HZ})"),
        (mrms_lons, mrms_lats, mrms_total,
         f"MRMS MultiSensor QPE (Pass2)\n{end_fhour}h accumulation"),
        (s4_lons, s4_lats, s4_total,
         f"NCEP Stage IV QPE (CONUS)\n{end_fhour}h accumulation"),
    ]
    plot_compare(panels, end_fhour, FIXED_DOMAIN, PARENT_PNG)

    def _mx(a):
        return float(np.nanmax(a)) if a is not None else float("nan")
    print(f"\nSaved {PARENT_PNG}")
    print(f"  HAFS parent max {_mx(hafs_display):.0f} mm | "
          f"MRMS max {_mx(mrms_total):.0f} mm | "
          f"Stage IV max {_mx(s4_total):.0f} mm")


if __name__ == "__main__":
    main()
