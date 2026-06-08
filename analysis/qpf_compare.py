"""
Compare HAFS-A QPF (accumulated total precip) against MRMS MultiSensor QPE
for the same accumulation window.

HAFS run : init 2024-09-24 00Z, f036 → valid 2024-09-25 12Z (36-hour accumulation)
MRMS QPE : 36 × 1-hour Pass2 files summed over the identical window

Data source: s3://noaa-mrms-pds (public, no credentials needed)

Dependencies: cfgrib, xarray, numpy, matplotlib, cartopy, boto3
  pip install boto3

Run from repo root: python analysis/qpf_compare.py
"""

from pathlib import Path
from datetime import datetime, timedelta
import gzip
import io
import warnings
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cfgrib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

HAFS_FILE = Path("helene_sample/HFSA/2024092400/09l.2024092400.hfsa.storm.nhc.f036.grb2")
OUT = Path("analysis/output")
OUT.mkdir(exist_ok=True)

INIT_DT = datetime(2024, 9, 24, 0)
FHOUR = 36
VALID_DT = INIT_DT + timedelta(hours=FHOUR)   # 2024-09-25 12Z
MRMS_BUCKET = "noaa-mrms-pds"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"


# ---------------------------------------------------------------------------
# HAFS
# ---------------------------------------------------------------------------

def load_hafs_precip():
    """Return (lats, lons_180, precip_mm) for HAFS total accumulated precip."""
    datasets = cfgrib.open_datasets(str(HAFS_FILE))
    for ds in datasets:
        if "tp" in ds.data_vars:
            da = ds["tp"]
            lats = da.latitude.values
            lons = da.longitude.values
            lons = np.where(lons > 180, lons - 360, lons)
            # GRIB2 tp from HAFS NHC grid is in kg m-2 (= mm); no conversion needed
            return lats, lons, da.values
    raise RuntimeError("tp not found in HAFS GRIB2 file")


# ---------------------------------------------------------------------------
# MRMS
# ---------------------------------------------------------------------------

def mrms_s3_key(valid_hour_dt):
    """S3 key for the 1H QPE file whose accumulation ends at valid_hour_dt."""
    date_str = valid_hour_dt.strftime("%Y%m%d")
    time_str = valid_hour_dt.strftime("%Y%m%d-%H%M%S")
    fname = f"MRMS_{MRMS_PRODUCT}_{time_str}.grib2.gz"
    return f"CONUS/{MRMS_PRODUCT}/{date_str}/{fname}"


def download_mrms_hour(s3, valid_hour_dt):
    """Download one 1H QPE grib2.gz and return (lats, lons, data_mm)."""
    key = mrms_s3_key(valid_hour_dt)
    buf = io.BytesIO()
    s3.download_fileobj(MRMS_BUCKET, key, buf)
    buf.seek(0)
    raw = gzip.decompress(buf.read())

    # cfgrib needs a real file path; write to a temp buffer via BytesIO
    tmp = Path("/tmp") / key.split("/")[-1].replace(".gz", "")
    tmp.write_bytes(raw)
    try:
        datasets = cfgrib.open_datasets(str(tmp))
        for ds in datasets:
            for var in ds.data_vars:
                da = ds[var]
                return da.latitude.values, da.longitude.values, da.values
        raise RuntimeError(f"No variable found in {key}")
    finally:
        tmp.unlink(missing_ok=True)


def accumulate_mrms_qpe(s3, init_dt, n_hours):
    """Accumulate n_hours of 1H QPE starting one hour after init_dt."""
    print(f"  Downloading {n_hours} MRMS 1H QPE files ...")
    total = None
    lats = lons = None
    hours_done = 0

    for h in range(1, n_hours + 1):
        t = init_dt + timedelta(hours=h)
        try:
            lat, lon, data = download_mrms_hour(s3, t)
            if total is None:
                lats, lons, total = lat, lon, np.zeros_like(data)
            # Replace fill values (−999, −3e38, etc.) with 0
            data = np.where(data < 0, 0.0, data)
            total += data
            hours_done += 1
            if h % 6 == 0 or h == n_hours:
                print(f"    hour {h:3d}/{n_hours}  ({t.strftime('%Y-%m-%d %HZ')})")
        except Exception as e:
            print(f"    hour {h:3d} FAILED ({t.strftime('%Y-%m-%d %HZ')}): {e}")

    print(f"  Accumulated {hours_done}/{n_hours} hours of MRMS QPE.")
    return lats, lons, total


def crop_mrms(mrms_lats, mrms_lons, mrms_data, lat_min, lat_max, lon_min, lon_max):
    """Crop MRMS arrays to the HAFS domain."""
    lat_mask = (mrms_lats >= lat_min) & (mrms_lats <= lat_max)
    lon_mask = (mrms_lons >= lon_min) & (mrms_lons <= lon_max)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        return mrms_lats, mrms_lons, mrms_data
    r0, r1 = lat_idx[0], lat_idx[-1] + 1
    c0, c1 = lon_idx[0], lon_idx[-1] + 1
    return mrms_lats[r0:r1], mrms_lons[c0:c1], mrms_data[r0:r1, c0:c1]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

QPF_LEVELS = [0, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
QPF_COLORS = [
    "#ffffff", "#c8f0f0", "#64d2ff", "#3296ff",
    "#00c800", "#ffff00", "#ffc800", "#ff6400",
    "#ff0000", "#c80000", "#960000", "#640064",
]


def qpf_cmap():
    cmap = mcolors.ListedColormap(QPF_COLORS)
    norm = mcolors.BoundaryNorm(QPF_LEVELS, cmap.N)
    return cmap, norm


def make_axes(n_cols, lon_min, lon_max, lat_min, lat_max):
    fig, axes = plt.subplots(
        1, n_cols, figsize=(8 * n_cols, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if n_cols == 1:
        axes = [axes]
    for ax in axes:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)
        gl.top_labels = gl.right_labels = False
    return fig, axes


def plot_qpf_comparison(hafs_lats, hafs_lons, hafs_mm,
                        mrms_lats, mrms_lons, mrms_mm,
                        domain):
    lat_min, lat_max, lon_min, lon_max = domain
    cmap, norm = qpf_cmap()

    fig, axes = make_axes(2, lon_min, lon_max, lat_min, lat_max)

    # HAFS panel
    ax = axes[0]
    cf = ax.contourf(hafs_lons, hafs_lats, hafs_mm,
                     levels=QPF_LEVELS, cmap=cmap, norm=norm,
                     transform=ccrs.PlateCarree(), extend="max")
    ax.set_title(
        f"HAFS-A Total Accumulated Precip\n"
        f"Init {INIT_DT.strftime('%Y-%m-%d %HZ')} | F{FHOUR:03d} "
        f"(0–{FHOUR}h, valid {VALID_DT.strftime('%Y-%m-%d %HZ')})"
    )

    # MRMS panel
    ax = axes[1]
    if mrms_mm is not None:
        ax.contourf(mrms_lons, mrms_lats, mrms_mm,
                    levels=QPF_LEVELS, cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree(), extend="max")
        ax.set_title(
            f"MRMS MultiSensor QPE (Pass2)\n"
            f"{FHOUR}h accumulation "
            f"({INIT_DT.strftime('%Y-%m-%d %HZ')} – {VALID_DT.strftime('%Y-%m-%d %HZ')})"
        )
    else:
        ax.text(0.5, 0.5, "MRMS data unavailable", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("MRMS MultiSensor QPE — unavailable")

    plt.colorbar(cf, ax=axes, label="Accumulated Precipitation (mm)",
                 ticks=QPF_LEVELS, shrink=0.7, fraction=0.02)
    fig.suptitle(
        f"Hurricane Helene — HAFS-A QPF vs MRMS QPE | {FHOUR}h ending {VALID_DT.strftime('%Y-%m-%d %HZ')}",
        fontsize=13, y=1.01,
    )

    out = OUT / f"qpf_compare_{VALID_DT.strftime('%Y%m%d_%HZ')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"HAFS accumulation window: {INIT_DT.strftime('%Y-%m-%d %HZ')} → "
          f"{VALID_DT.strftime('%Y-%m-%d %HZ')} ({FHOUR}h)")

    print("\nLoading HAFS total precipitation ...")
    hafs_lats, hafs_lons, hafs_mm = load_hafs_precip()
    lat_min = hafs_lats.min() - 0.5
    lat_max = hafs_lats.max() + 0.5
    lon_min = hafs_lons.min() - 0.5
    lon_max = hafs_lons.max() + 0.5
    print(f"  Domain: lat [{lat_min:.1f}, {lat_max:.1f}]  "
          f"lon [{lon_min:.1f}, {lon_max:.1f}]")
    print(f"  HAFS precip range: {np.nanmin(hafs_mm):.1f} – {np.nanmax(hafs_mm):.1f} mm")

    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))

    print("\nDownloading MRMS QPE ...")
    mrms_lats_full, mrms_lons_full, mrms_total = accumulate_mrms_qpe(s3, INIT_DT, FHOUR)

    print("\nCropping MRMS to HAFS domain ...")
    mrms_lats, mrms_lons, mrms_mm = crop_mrms(
        mrms_lats_full, mrms_lons_full, mrms_total,
        lat_min, lat_max, lon_min, lon_max,
    )
    print(f"  MRMS precip range: {mrms_mm.min():.1f} – {mrms_mm.max():.1f} mm")

    print("\nGenerating QPF comparison plot ...")
    domain = (lat_min, lat_max, lon_min, lon_max)
    plot_qpf_comparison(hafs_lats, hafs_lons, hafs_mm,
                        mrms_lats, mrms_lons, mrms_mm,
                        domain)
    print("Done.")


if __name__ == "__main__":
    main()
