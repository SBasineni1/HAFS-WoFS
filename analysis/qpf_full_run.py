"""
HAFS-A QPF vs MRMS QPE — full-run animation

Runs on Hercules HPC. For each HAFS forecast hour found in the run directory,
plots accumulated precipitation side-by-side with the matching MRMS MultiSensor
QPE accumulation, then stitches all frames into an animation.

Usage (on Hercules login node):
    module load python/3.10  (or whatever your env is)
    conda activate your_env
    python analysis/qpf_full_run.py

To stitch frames into an MP4 after the script finishes:
    ffmpeg -r 4 -pattern_type glob -i 'output/frames/qpf_frame_*.png' \
           -vcodec libx264 -crf 22 -pix_fmt yuv420p output/qpf_animation.mp4

Config:
    Edit the CONFIG block below to match your run.
"""

import re
import gzip
import io
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cfgrib
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG — edit these for your run
# =============================================================================

# Root directory of your HAFS run on Hercules
HAFS_RUN_DIR = Path(
    "/work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/HFSA"
)

# Init time of the run you want to process
INIT_STR = "2024092400"   # YYYYMMDDCC

# File pattern to match within the run directory.
# HAFS naming: {storm}.{init}.hfsa.storm.atm.f{FHR}.grb2
# Adjust the glob if your files use a different naming convention.
FILE_GLOB = f"**/*{INIT_STR}*storm.atm.f*.grb2"

# Output directory (create it before running, or let the script create it)
OUT_DIR = Path(
    "/work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/qpf_frames"
)

# Only process these forecast hours (set to None to process all found)
# e.g. FHOURS_FILTER = list(range(6, 127, 6))  to do every 6 hours
FHOURS_FILTER = None

# Local cache directory for downloaded MRMS files (avoids re-downloading)
MRMS_CACHE_DIR = Path("/tmp/mrms_cache")

# =============================================================================

INIT_DT = datetime.strptime(INIT_STR, "%Y%m%d%H")
MRMS_BUCKET = "noaa-mrms-pds"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"

QPF_LEVELS = [0, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
QPF_COLORS = [
    "#ffffff", "#c8f0f0", "#64d2ff", "#3296ff",
    "#02fd02", "#01c501", "#008e00", "#fdf802",
    "#e5bc00", "#fd9500", "#fd0000", "#d40000",
]


# =============================================================================
# File discovery
# =============================================================================

def parse_fhour(filepath):
    """Extract integer forecast hour from filename like ...f036.grb2"""
    m = re.search(r"\.f(\d{3})\.grb2$", filepath.name)
    return int(m.group(1)) if m else None


def discover_files(run_dir, glob, fhours_filter=None):
    files = sorted(run_dir.glob(glob))
    pairs = [(parse_fhour(f), f) for f in files]
    pairs = [(h, f) for h, f in pairs if h is not None]
    if fhours_filter:
        pairs = [(h, f) for h, f in pairs if h in fhours_filter]
    pairs.sort()
    return pairs


# =============================================================================
# HAFS loader
# =============================================================================

def load_hafs_precip(filepath):
    """Return (lats, lons_180, precip_mm) from a HAFS GRIB2 file."""
    datasets = cfgrib.open_datasets(str(filepath))
    for ds in datasets:
        if "tp" in ds.data_vars:
            da = ds["tp"]
            lats = da.latitude.values
            lons = np.where(da.longitude.values > 180,
                            da.longitude.values - 360,
                            da.longitude.values)
            # units are kg m-2 = mm
            return lats, lons, da.values
    raise RuntimeError(f"tp not found in {filepath}")


# =============================================================================
# MRMS downloader / cache
# =============================================================================

def mrms_s3_key(hour_end_dt):
    date_str = hour_end_dt.strftime("%Y%m%d")
    time_str = hour_end_dt.strftime("%Y%m%d-%H%M%S")
    fname = f"MRMS_{MRMS_PRODUCT}_{time_str}.grib2.gz"
    return f"CONUS/{MRMS_PRODUCT}/{date_str}/{fname}", fname


def load_mrms_hour(s3, hour_end_dt, cache_dir):
    """
    Return precipitation (mm) for the 1-hour window ending at hour_end_dt.
    Downloads from S3 on first call; subsequent calls use the local cache.
    Returns (lats, lons, data) or raises on failure.
    """
    key, fname = mrms_s3_key(hour_end_dt)
    cache_path = cache_dir / fname.replace(".gz", "")

    if not cache_path.exists():
        gz_buf = io.BytesIO()
        s3.download_fileobj(MRMS_BUCKET, key, gz_buf)
        gz_buf.seek(0)
        raw = gzip.decompress(gz_buf.read())
        cache_path.write_bytes(raw)

    datasets = cfgrib.open_datasets(str(cache_path))
    for ds in datasets:
        for var in ds.data_vars:
            da = ds[var]
            data = da.values
            data = np.where(data < 0, 0.0, data)   # replace fill values
            return da.latitude.values, da.longitude.values, data

    raise RuntimeError(f"No variable in {cache_path}")


def accumulate_mrms(s3, init_dt, n_hours, cache_dir):
    """
    Sum n_hours of MRMS 1H QPE (hours 1..n_hours after init_dt).
    Returns (lats, lons, total_mm).  Missing hours are skipped (treated as 0).
    """
    total = lats = lons = None
    for h in range(1, n_hours + 1):
        t = init_dt + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, cache_dir)
            if total is None:
                lats, lons, total = lat, lon, np.zeros_like(data)
            total += data
        except Exception as e:
            print(f"    MRMS h{h:03d} ({t.strftime('%Y-%m-%d %HZ')}) skipped: {e}")
    return lats, lons, total


def crop_to_domain(lats, lons, data, lat_min, lat_max, lon_min, lon_max):
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    ri = np.where(lat_mask)[0]
    ci = np.where(lon_mask)[0]
    if ri.size == 0 or ci.size == 0:
        return lats, lons, data
    return (lats[ri[0]:ri[-1]+1],
            lons[ci[0]:ci[-1]+1],
            data[ri[0]:ri[-1]+1, ci[0]:ci[-1]+1])


# =============================================================================
# Plotting
# =============================================================================

def qpf_cmap():
    cmap = mcolors.ListedColormap(QPF_COLORS)
    norm = mcolors.BoundaryNorm(QPF_LEVELS, cmap.N)
    return cmap, norm


def plot_frame(fhour, hafs_lats, hafs_lons, hafs_mm,
               mrms_lats, mrms_lons, mrms_mm, out_path):
    valid_dt = INIT_DT + timedelta(hours=fhour)
    lat_min = np.nanmin(hafs_lats) - 0.5
    lat_max = np.nanmax(hafs_lats) + 0.5
    lon_min = np.nanmin(hafs_lons) - 0.5
    lon_max = np.nanmax(hafs_lons) + 0.5

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

    # HAFS panel
    cf = axes[0].contourf(
        hafs_lons, hafs_lats, hafs_mm,
        levels=QPF_LEVELS, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), extend="max",
    )
    axes[0].set_title(
        f"HAFS-A Accumulated Precip\n"
        f"Init {INIT_DT.strftime('%Y-%m-%d %HZ')} | "
        f"F{fhour:03d} (0–{fhour}h, valid {valid_dt.strftime('%Y-%m-%d %HZ')})"
    )

    # MRMS panel
    if mrms_mm is not None:
        axes[1].contourf(
            mrms_lons, mrms_lats, mrms_mm,
            levels=QPF_LEVELS, cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(), extend="max",
        )
        axes[1].set_title(
            f"MRMS MultiSensor QPE (Pass2)\n"
            f"{fhour}h accumulation "
            f"({INIT_DT.strftime('%Y-%m-%d %HZ')} – {valid_dt.strftime('%Y-%m-%d %HZ')})"
        )
    else:
        axes[1].text(0.5, 0.5, "MRMS data unavailable",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("MRMS MultiSensor QPE — unavailable")

    plt.colorbar(cf, ax=axes, label="Accumulated Precipitation (mm)",
                 ticks=QPF_LEVELS, shrink=0.7, fraction=0.02)
    fig.suptitle(
        f"Hurricane Helene — HAFS-A QPF vs MRMS QPE | "
        f"F{fhour:03d} ending {valid_dt.strftime('%Y-%m-%d %HZ')}",
        fontsize=13, y=1.01,
    )

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MRMS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"HAFS run dir : {HAFS_RUN_DIR}")
    print(f"Init time    : {INIT_DT.strftime('%Y-%m-%d %HZ')}")
    print(f"Output dir   : {OUT_DIR}")

    # --- discover HAFS files ---
    file_pairs = discover_files(HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER)
    if not file_pairs:
        print(f"\nNo files found matching {FILE_GLOB} in {HAFS_RUN_DIR}")
        print("Check FILE_GLOB or HAFS_RUN_DIR in the CONFIG block.")
        return
    print(f"\nFound {len(file_pairs)} HAFS files:")
    for h, f in file_pairs:
        print(f"  F{h:03d}  {f.name}")

    max_fhour = file_pairs[-1][0]
    valid_end = INIT_DT + timedelta(hours=max_fhour)

    # --- pre-download all needed MRMS 1H files up front ---
    print(f"\nPre-caching MRMS 1H QPE: hours 1–{max_fhour} "
          f"(up to {valid_end.strftime('%Y-%m-%d %HZ')}) ...")
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    mrms_hourly = {}   # {hour_int: (lats, lons, data)}  — in-memory cache
    for h in range(1, max_fhour + 1):
        t = INIT_DT + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, MRMS_CACHE_DIR)
            mrms_hourly[h] = (lat, lon, data)
            if h % 12 == 0 or h == max_fhour:
                print(f"  cached h{h:03d}/{max_fhour} ({t.strftime('%Y-%m-%d %HZ')})")
        except Exception as e:
            print(f"  h{h:03d} unavailable: {e}")

    # --- process each forecast hour ---
    print(f"\nGenerating {len(file_pairs)} frames ...")
    for fhour, filepath in file_pairs:
        out_path = OUT_DIR / f"qpf_frame_{fhour:03d}.png"
        if out_path.exists():
            print(f"  F{fhour:03d} — already exists, skipping.")
            continue

        print(f"  F{fhour:03d} ({filepath.name}) ...", end=" ", flush=True)

        # Load HAFS
        try:
            hafs_lats, hafs_lons, hafs_mm = load_hafs_precip(filepath)
        except Exception as e:
            print(f"HAFS load failed: {e}")
            continue

        domain_pad = 0.5
        lat_min = np.nanmin(hafs_lats) - domain_pad
        lat_max = np.nanmax(hafs_lats) + domain_pad
        lon_min = np.nanmin(hafs_lons) - domain_pad
        lon_max = np.nanmax(hafs_lons) + domain_pad

        # Accumulate MRMS up to this forecast hour using in-memory cache
        mrms_lats = mrms_lons = mrms_total = None
        for h in range(1, fhour + 1):
            if h not in mrms_hourly:
                continue
            lat, lon, data = mrms_hourly[h]
            if mrms_total is None:
                mrms_lats, mrms_lons = lat, lon
                mrms_total = np.zeros_like(data)
            mrms_total += data

        # Crop MRMS to HAFS domain
        if mrms_total is not None:
            mrms_lats, mrms_lons, mrms_cropped = crop_to_domain(
                mrms_lats, mrms_lons, mrms_total,
                lat_min, lat_max, lon_min, lon_max,
            )
        else:
            mrms_cropped = None

        plot_frame(fhour, hafs_lats, hafs_lons, hafs_mm,
                   mrms_lats, mrms_lons, mrms_cropped, out_path)
        hafs_max = np.nanmax(hafs_mm)
        mrms_max = np.nanmax(mrms_cropped) if mrms_cropped is not None else float("nan")
        print(f"saved  (HAFS max {hafs_max:.0f} mm | MRMS max {mrms_max:.0f} mm)")

    print(f"\nAll frames written to {OUT_DIR}")
    print("\nTo make an MP4:")
    print(f"  ffmpeg -r 4 -pattern_type glob -i '{OUT_DIR}/qpf_frame_*.png' \\")
    print(f"         -vcodec libx264 -crf 22 -pix_fmt yuv420p {OUT_DIR}/../qpf_animation.mp4")


if __name__ == "__main__":
    main()
