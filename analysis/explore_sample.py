"""
Explore and visualize the HAFS Helene sample GRIB2 file.
Run from repo root: python analysis/explore_sample.py
"""

from pathlib import Path
import warnings
import cfgrib
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

warnings.filterwarnings("ignore")

SAMPLE = Path("helene_sample/HFSA/2024092400/09l.2024092400.hfsa.storm.nhc.f036.grb2")
OUT = Path("analysis/output")
OUT.mkdir(exist_ok=True)


def load_datasets():
    datasets = cfgrib.open_datasets(str(SAMPLE))
    print(f"\nLoaded {len(datasets)} GRIB2 message groups:\n")
    for i, ds in enumerate(datasets):
        coords = {k: float(v) for k, v in ds.coords.items() if v.ndim == 0}
        print(f"  [{i}] vars={list(ds.data_vars)}  coords={coords}")
    return datasets


def find_var(datasets, name):
    for ds in datasets:
        if name in ds.data_vars:
            return ds[name]
    return None


def plot_surface_winds(datasets):
    u = find_var(datasets, "u10")
    v = find_var(datasets, "v10")
    if u is None or v is None:
        print("  10m winds not found, skipping.")
        return

    speed = np.sqrt(u**2 + v**2) * 1.94384  # m/s -> knots
    lats = u.latitude.values
    lons = u.longitude.values

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="gray")

    cf = ax.contourf(lons, lats, speed, levels=np.arange(0, 130, 10),
                     cmap="RdYlGn_r", transform=ccrs.PlateCarree())
    skip = max(1, len(lats) // 20)
    ax.quiver(lons[::skip], lats[::skip], u.values[::skip, ::skip], v.values[::skip, ::skip],
              scale=300, width=0.002, transform=ccrs.PlateCarree(), alpha=0.6)

    plt.colorbar(cf, ax=ax, label="Wind Speed (knots)", shrink=0.7)
    ax.set_title("HFSA | Hurricane Helene | 10m Wind Speed\nInit: 2024-09-24 00Z | F036 (Valid: 2024-09-25 12Z)")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
    gl.top_labels = gl.right_labels = False

    out = OUT / "surface_winds.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_mslp(datasets):
    mslp = find_var(datasets, "mslet")
    if mslp is None:
        print("  MSLP not found, skipping.")
        return

    data = mslp.values / 100  # Pa -> hPa
    lats = mslp.latitude.values
    lons = mslp.longitude.values

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="gray")

    levels = np.arange(900, 1025, 4)
    cf = ax.contourf(lons, lats, data, levels=levels, cmap="RdBu_r", transform=ccrs.PlateCarree())
    cs = ax.contour(lons, lats, data, levels=levels, colors="black", linewidths=0.5, transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt="%d", fontsize=7)

    plt.colorbar(cf, ax=ax, label="MSLP (hPa)", shrink=0.7)
    ax.set_title("HFSA | Hurricane Helene | Mean Sea Level Pressure\nInit: 2024-09-24 00Z | F036 (Valid: 2024-09-25 12Z)")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
    gl.top_labels = gl.right_labels = False

    out = OUT / "mslp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_precip(datasets):
    apcp = find_var(datasets, "tp")
    if apcp is None:
        print("  Total precip not found, skipping.")
        return

    data = apcp.values * 1000 if apcp.values.max() < 10 else apcp.values  # m -> mm if needed
    lats = apcp.latitude.values
    lons = apcp.longitude.values

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="gray")

    levels = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
    cf = ax.contourf(lons, lats, data, levels=levels, cmap="Blues", transform=ccrs.PlateCarree())

    plt.colorbar(cf, ax=ax, label="Accumulated Precipitation (mm)", shrink=0.7)
    ax.set_title("HFSA | Hurricane Helene | Total Accumulated Precip (0–36h)\nInit: 2024-09-24 00Z | F036 (Valid: 2024-09-25 12Z)")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
    gl.top_labels = gl.right_labels = False

    out = OUT / "precip.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_reflectivity(datasets):
    refc = find_var(datasets, "refc")
    if refc is None:
        print("  Composite reflectivity not found, skipping.")
        return

    lats = refc.latitude.values
    lons = refc.longitude.values

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="gray")

    levels = np.arange(-10, 75, 5)
    cf = ax.contourf(lons, lats, refc.values, levels=levels, cmap="gist_ncar", transform=ccrs.PlateCarree())

    plt.colorbar(cf, ax=ax, label="Reflectivity (dBZ)", shrink=0.7)
    ax.set_title("HFSA | Hurricane Helene | Composite Reflectivity\nInit: 2024-09-24 00Z | F036 (Valid: 2024-09-25 12Z)")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
    gl.top_labels = gl.right_labels = False

    out = OUT / "reflectivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("HAFS Helene Sample Explorer")
    print(f"File: {SAMPLE}")

    datasets = load_datasets()

    print("\nGenerating plots...")
    plot_surface_winds(datasets)
    plot_mslp(datasets)
    plot_precip(datasets)
    plot_reflectivity(datasets)

    print(f"\nDone. Check {OUT}/ for output images.")
