"""
HAFS-A QPF vs MRMS QPE — full-run animation

HAFS uses a storm-following grid that moves with the TC, so each frame's
domain shifts.  This script reprojects every HAFS tp field onto a single
fixed lat/lon grid and keeps a running maximum (np.fmax) so accumulated
precipitation sticks to the geography as the storm tracks inland.

MRMS QPE is accumulated on the same fixed domain, adding one hour at a
time in sync with the forecast hours being plotted.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/qpf_full_run.py

To stitch frames into an MP4:
    ffmpeg -r 4 -pattern_type glob -i '<OUT_DIR>/qpf_frame_*.png' \
           -vf "format=rgb24" -vcodec mpeg4 -q:v 3 -pix_fmt yuv420p \
           <OUT_DIR>/../qpf_animation.mp4

Config:
    Edit the CONFIG block below to match your run.
"""

import logging
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
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")
for _log in ["cfgrib", "cfgrib.messages", "cfgrib.xarray_store", "cfgrib.dataset"]:
    logging.getLogger(_log).setLevel(logging.CRITICAL)

# =============================================================================
# CONFIG — edit these for your run
# =============================================================================

HAFS_RUN_DIR = Path(
    "/work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/HFSA"
)
INIT_STR = "2024092400"
FILE_GLOB = f"**/*{INIT_STR}*storm.atm.f*.grb2"
OUT_DIR = Path("/work2/noaa/aoml-hafs1/suchit/qpf_frames")
FHOURS_FILTER = None
MRMS_CACHE_DIR = Path("/tmp/mrms_cache")
CFGRIB_IDX_DIR = Path("/work2/noaa/aoml-hafs1/suchit/.cfgrib_idx")

# Fixed grid resolution in degrees (~5 km)
GRID_RES = 0.05

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
# HAFS loader + fixed-grid reprojection
# =============================================================================

def load_hafs_precip(filepath):
    """Return (lats, lons_180, precip_mm) on the native storm-following grid.

    cfgrib.open_datasets splits a multi-grid GRIB2 into separate datasets,
    avoiding the shape-mismatch error that xr.open_dataset causes when it
    tries to concatenate tp fields from both the parent domain and the
    storm-following nest.  We take the first dataset that contains tp.
    """
    datasets = cfgrib.open_datasets(str(filepath))
    for ds in datasets:
        if "tp" in ds.data_vars:
            da = ds["tp"]
            lats = da.latitude.values
            lons = np.where(da.longitude.values > 180,
                            da.longitude.values - 360,
                            da.longitude.values)
            return lats, lons, da.values
    raise RuntimeError(f"tp not found in {filepath}")


def regrid_hafs(src_lats, src_lons, src_data, grid_lat, grid_lon):
    """
    Reproject HAFS tp from the moving storm grid onto the fixed lat/lon mesh.
    Uses nearest-neighbour — fast and sufficient for QPF accumulation plots.
    Returns NaN outside the storm grid footprint for this frame.
    """
    pts = np.column_stack([src_lons.ravel(), src_lats.ravel()])
    vals = src_data.ravel()
    valid = np.isfinite(vals) & np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if valid.sum() < 4:
        return np.full(grid_lat.shape, np.nan)
    return griddata(
        pts[valid], vals[valid],
        (grid_lon, grid_lat),
        method="nearest",
        fill_value=np.nan,
    )


# =============================================================================
# MRMS downloader / cache
# =============================================================================

def mrms_s3_key(hour_end_dt):
    date_str = hour_end_dt.strftime("%Y%m%d")
    time_str = hour_end_dt.strftime("%Y%m%d-%H%M%S")
    fname = f"MRMS_{MRMS_PRODUCT}_{time_str}.grib2.gz"
    return f"CONUS/{MRMS_PRODUCT}/{date_str}/{fname}", fname


def load_mrms_hour(s3, hour_end_dt, cache_dir):
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
            data = np.where(da.values < 0, 0.0, da.values)
            return da.latitude.values, da.longitude.values, data
    raise RuntimeError(f"No variable in {cache_path}")


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


def plot_frame(fhour, fixed_lons, fixed_lats, hafs_mm,
               mrms_lons, mrms_lats, mrms_mm,
               full_domain, out_path):
    valid_dt = INIT_DT + timedelta(hours=fhour)
    lat_min, lat_max, lon_min, lon_max = full_domain
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

    # HAFS — running accumulated total on fixed grid
    cf = axes[0].contourf(
        fixed_lons, fixed_lats, hafs_mm,
        levels=QPF_LEVELS, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), extend="max",
    )
    axes[0].set_title(
        f"HAFS-A Accumulated Precip\n"
        f"Init {INIT_DT.strftime('%Y-%m-%d %HZ')} | "
        f"F{fhour:03d} (0–{fhour}h, valid {valid_dt.strftime('%Y-%m-%d %HZ')})"
    )

    # MRMS
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
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
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

    file_pairs = discover_files(HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER)
    if not file_pairs:
        print(f"\nNo files found matching {FILE_GLOB} in {HAFS_RUN_DIR}")
        return
    print(f"\nFound {len(file_pairs)} HAFS files")
    max_fhour = file_pairs[-1][0]

    # ------------------------------------------------------------------
    # First pass: scan all files to find the union lat/lon extent so we
    # can build one fixed grid that covers the entire TC track.
    # ------------------------------------------------------------------
    print("\nFirst pass: scanning all HAFS files for full domain extent ...")
    lat_min_all, lat_max_all = 90.0, -90.0
    lon_min_all, lon_max_all = 180.0, -180.0
    for fhour, filepath in file_pairs:
        try:
            lats, lons, _ = load_hafs_precip(filepath)
            lat_min_all = min(lat_min_all, float(np.nanmin(lats)))
            lat_max_all = max(lat_max_all, float(np.nanmax(lats)))
            lon_min_all = min(lon_min_all, float(np.nanmin(lons)))
            lon_max_all = max(lon_max_all, float(np.nanmax(lons)))
        except Exception as e:
            print(f"  F{fhour:03d} scan failed: {e}")
    lat_min_all -= 0.5
    lat_max_all += 0.5
    lon_min_all -= 0.5
    lon_max_all += 0.5
    full_domain = (lat_min_all, lat_max_all, lon_min_all, lon_max_all)
    print(f"  Full domain : lat [{lat_min_all:.1f}, {lat_max_all:.1f}]  "
          f"lon [{lon_min_all:.1f}, {lon_max_all:.1f}]")

    fixed_lons = np.arange(lon_min_all, lon_max_all + GRID_RES, GRID_RES)
    fixed_lats = np.arange(lat_min_all, lat_max_all + GRID_RES, GRID_RES)
    grid_lon, grid_lat = np.meshgrid(fixed_lons, fixed_lats)
    print(f"  Fixed grid  : {grid_lat.shape[0]}×{grid_lat.shape[1]} "
          f"at {GRID_RES}° resolution")

    # ------------------------------------------------------------------
    # Pre-download all MRMS 1H files and crop to the fixed domain.
    # ------------------------------------------------------------------
    valid_end = INIT_DT + timedelta(hours=max_fhour)
    print(f"\nPre-caching MRMS 1H QPE: hours 1–{max_fhour} "
          f"(up to {valid_end.strftime('%Y-%m-%d %HZ')}) ...")
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    mrms_hourly = {}
    for h in range(1, max_fhour + 1):
        t = INIT_DT + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, MRMS_CACHE_DIR)
            clat, clon, cdata = crop_to_domain(lat, lon, data,
                                               lat_min_all, lat_max_all,
                                               lon_min_all, lon_max_all)
            mrms_hourly[h] = (clat, clon, cdata)
            if h % 12 == 0 or h == max_fhour:
                print(f"  cached h{h:03d}/{max_fhour} ({t.strftime('%Y-%m-%d %HZ')})")
        except Exception as e:
            print(f"  h{h:03d} unavailable: {e}")

    mrms_lats = mrms_lons = None
    for h in sorted(mrms_hourly):
        mrms_lats, mrms_lons, _ = mrms_hourly[h]
        break

    # ------------------------------------------------------------------
    # Second pass: generate frames.
    # Running state is always updated (even for frames that already exist)
    # so that later frames have the correct accumulated totals.
    # ------------------------------------------------------------------
    print(f"\nGenerating {len(file_pairs)} frames ...")
    hafs_running_max = np.zeros(grid_lat.shape)
    mrms_running_total = None
    last_mrms_h = 0

    for fhour, filepath in file_pairs:
        out_path = OUT_DIR / f"qpf_frame_{fhour:03d}.png"

        # Update HAFS running max regardless of whether we skip the PNG
        try:
            hafs_lats, hafs_lons, hafs_mm = load_hafs_precip(filepath)
            hafs_interp = regrid_hafs(hafs_lats, hafs_lons, hafs_mm,
                                      grid_lat, grid_lon)
            # tp is cumulative from init; fmax stamps the latest value at
            # each fixed grid point as the storm grid sweeps over it.
            hafs_running_max = np.fmax(hafs_running_max,
                                       np.nan_to_num(hafs_interp, nan=0.0))
        except Exception as e:
            print(f"  F{fhour:03d} HAFS load failed: {e}")
            continue

        # Add MRMS hours from where we left off up to this forecast hour
        for h in range(last_mrms_h + 1, fhour + 1):
            if h in mrms_hourly:
                _, _, data = mrms_hourly[h]
                if mrms_running_total is None:
                    mrms_running_total = np.zeros_like(data)
                mrms_running_total += data
        last_mrms_h = fhour

        if out_path.exists():
            print(f"  F{fhour:03d} — already exists, skipping.")
            continue

        print(f"  F{fhour:03d} ({filepath.name}) ...", end=" ", flush=True)
        plot_frame(
            fhour,
            fixed_lons, fixed_lats, hafs_running_max,
            mrms_lons, mrms_lats, mrms_running_total,
            full_domain, out_path,
        )
        hafs_max = float(np.nanmax(hafs_running_max))
        mrms_max = (float(np.nanmax(mrms_running_total))
                    if mrms_running_total is not None else float("nan"))
        print(f"saved  (HAFS max {hafs_max:.0f} mm | MRMS max {mrms_max:.0f} mm)")

    print(f"\nAll frames written to {OUT_DIR}")
    print("\nTo make an MP4:")
    print(f"  ffmpeg -r 4 -pattern_type glob -i '{OUT_DIR}/qpf_frame_*.png' \\")
    print(f'         -vf "format=rgb24" -vcodec mpeg4 -q:v 3 -pix_fmt yuv420p '
          f"{OUT_DIR}/../qpf_animation.mp4")


if __name__ == "__main__":
    main()
