"""
Download NEXRAD Level-2 data and compare with HAFS composite reflectivity.

Dependencies: cfgrib, xarray, numpy, matplotlib, cartopy, nexradaws, arm-pyart
  pip install nexradaws arm-pyart

Run from repo root: python analysis/nexrad_compare.py
"""

from pathlib import Path
from datetime import datetime, timedelta
import warnings
import tempfile
import cfgrib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import nexradaws
import pyart

warnings.filterwarnings("ignore")

SAMPLE = Path("helene_sample/HFSA/2024092400/09l.2024092400.hfsa.storm.nhc.f036.grb2")
OUT = Path("analysis/output")
OUT.mkdir(exist_ok=True)

INIT_DT = datetime.strptime("2024092400", "%Y%m%d%H")
FHOUR = 36
VALID_DT = INIT_DT + timedelta(hours=FHOUR)  # 2024-09-25 12Z

# NEXRAD stations (lat, lon) — southeastern US covering Helene's track
NEXRAD_STATIONS = {
    "KAMX": (25.6111, -80.4128),
    "KTBW": (27.7056, -82.4019),
    "KEVX": (30.5644, -85.9219),
    "KTLH": (30.3975, -84.3289),
    "KBYX": (24.5975, -81.7031),
    "KJAX": (30.4846, -81.7018),
    "KLTX": (33.9891, -78.4292),
    "KGSP": (34.8833, -82.2203),
}


def to_180(lons):
    return np.where(lons > 180, lons - 360, lons)


def load_hafs_refc():
    datasets = cfgrib.open_datasets(str(SAMPLE))
    for ds in datasets:
        if "refc" in ds.data_vars:
            da = ds["refc"]
            lats = da.latitude.values
            lons = to_180(da.longitude.values)
            return lats, lons, da.values
    raise RuntimeError("refc not found in HAFS file")


def get_domain(lats, lons, pad=0.5):
    return (lats.min() - pad, lats.max() + pad,
            lons.min() - pad, lons.max() + pad)


def stations_in_domain(lat_min, lat_max, lon_min, lon_max):
    return {sid: coords for sid, coords in NEXRAD_STATIONS.items()
            if lat_min <= coords[0] <= lat_max and lon_min <= coords[1] <= lon_max}


def find_closest_scan(conn, station, valid_dt, window_minutes=30):
    """Return the LocalNexradFile for the scan closest to valid_dt, or None."""
    scans = conn.get_avail_scans(
        str(valid_dt.year), f"{valid_dt.month:02d}",
        f"{valid_dt.day:02d}", station,
    )
    if not scans:
        return None
    closest = min(
        scans,
        key=lambda s: abs((s.scan_time.replace(tzinfo=None) - valid_dt).total_seconds()),
    )
    delta_min = abs((closest.scan_time.replace(tzinfo=None) - valid_dt).total_seconds()) / 60
    if delta_min > window_minutes:
        return None
    return closest, delta_min


def reflectivity_cmap():
    colors_hex = [
        "#646464", "#04e9e7", "#019ff4", "#0300f4",
        "#02fd02", "#01c501", "#008e00", "#fdf802",
        "#e5bc00", "#fd9500", "#fd0000", "#d40000",
        "#bc0000", "#f800fd", "#9854c6",
    ]
    bounds = [-10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75]
    cmap = mcolors.ListedColormap(colors_hex)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def plot_comparison(hafs_lats, hafs_lons, hafs_refc, nexrad_files, domain):
    lat_min, lat_max, lon_min, lon_max = domain
    cmap, norm, bounds = reflectivity_cmap()

    n_cols = 1 + len(nexrad_files)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    if n_cols == 1:
        axes = [axes]

    def setup_ax(ax):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)
        gl.top_labels = gl.right_labels = False

    # HAFS panel
    ax = axes[0]
    setup_ax(ax)
    cf = ax.contourf(hafs_lons, hafs_lats, hafs_refc, levels=bounds,
                     cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend="both")
    ax.set_title(
        f"HAFS-A Composite Reflectivity\n"
        f"Init: {INIT_DT.strftime('%Y-%m-%d %HZ')} | F{FHOUR:03d} "
        f"(Valid: {VALID_DT.strftime('%Y-%m-%d %HZ')})"
    )

    # NEXRAD panels
    for i, (sid, (fpath, delta_min)) in enumerate(nexrad_files.items()):
        ax = axes[i + 1]
        setup_ax(ax)
        try:
            radar = pyart.io.read_nexrad_archive(str(fpath))
            display = pyart.graph.RadarMapDisplay(radar)
            display.plot_ppi_map(
                "reflectivity", ax=ax, sweep=0,
                vmin=-10, vmax=75, cmap=cmap, norm=norm,
                colorbar_flag=False, title_flag=False,
                lat_lines=[], lon_lines=[],
            )
            scan_time = pyart.util.datetime_from_radar(radar).strftime("%H:%M UTC")
            ax.set_title(
                f"NEXRAD {sid} Reflectivity\n"
                f"Scan: {scan_time}  (Δ{delta_min:.0f} min from valid)"
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not render {sid}\n{e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(f"NEXRAD {sid} — render error")

    plt.colorbar(cf, ax=axes, label="Reflectivity (dBZ)", shrink=0.6,
                 ticks=bounds[::2], fraction=0.02)
    fig.suptitle(
        f"Hurricane Helene — HAFS-A vs NEXRAD | Valid: {VALID_DT.strftime('%Y-%m-%d %HZ')}",
        fontsize=13, y=1.01,
    )

    out = OUT / f"nexrad_compare_{VALID_DT.strftime('%Y%m%d_%HZ')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    print(f"HAFS valid time: {VALID_DT.strftime('%Y-%m-%d %HZ')}")

    print("Loading HAFS composite reflectivity...")
    hafs_lats, hafs_lons, hafs_refc = load_hafs_refc()
    domain = get_domain(hafs_lats, hafs_lons)
    lat_min, lat_max, lon_min, lon_max = domain
    print(f"  Domain: lat [{lat_min:.1f}, {lat_max:.1f}]  lon [{lon_min:.1f}, {lon_max:.1f}]")

    stations = stations_in_domain(lat_min, lat_max, lon_min, lon_max)
    if not stations:
        print("  No pre-listed stations in domain — using all southeastern stations.")
        stations = NEXRAD_STATIONS
    print(f"  Stations to try: {list(stations.keys())}")

    conn = nexradaws.NexradAwsInterface()
    tmp_dir = Path(tempfile.mkdtemp(prefix="nexrad_"))
    print(f"  Temp directory: {tmp_dir}")

    nexrad_files = {}
    for sid in stations:
        result = find_closest_scan(conn, sid, VALID_DT)
        if result is None:
            print(f"  {sid}: no scan within 30 min of {VALID_DT.strftime('%H:%M UTC')}")
            continue
        scan_obj, delta_min = result
        print(f"  {sid}: downloading {scan_obj.filename} (Δ{delta_min:.1f} min) ...")
        try:
            dl = conn.download(scan_obj, str(tmp_dir))
            if dl.success:
                nexrad_files[sid] = (Path(dl.success[0].filepath), delta_min)
        except Exception as e:
            print(f"  {sid}: download failed — {e}")

    if not nexrad_files:
        print("\nNo NEXRAD files downloaded — plotting HAFS only.")

    print("\nGenerating comparison plot...")
    plot_comparison(hafs_lats, hafs_lons, hafs_refc, nexrad_files, domain)
    print("Done.")


if __name__ == "__main__":
    main()
