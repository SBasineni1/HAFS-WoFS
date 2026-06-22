"""
HAFS QPF vs MRMS QPE — full-run animation

HAFS uses a storm-following grid that moves with the TC, so each frame's
domain shifts.  The nest's 0->fhour "cumulative" tp is accumulated per grid
CELL in storm-relative space (a cell under the eyewall racks up rain all run
long -> impossible ~2400 mm totals), so it is NOT usable for a geographic map.
Instead this script takes each frame's short per-interval precip BUCKET,
reprojects it onto a single fixed lat/lon grid at that frame's true position,
and SUMS the buckets so accumulation sticks to the geography as the storm
tracks inland.

MRMS QPE is accumulated on the same fixed domain, adding one hour at a
time in sync with the forecast hours being plotted.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/qpf_full_run.py storms/helene_hfsa.yaml

To stitch frames into an MP4:
    ffmpeg -r 4 -pattern_type glob -i '<OUT_DIR>/qpf_frame_*.png' \
           -vf "format=rgb24" -vcodec mpeg4 -q:v 3 -pix_fmt yuv420p \
           <OUT_DIR>/../qpf_animation.mp4
"""

import logging
import re
import sys
import gzip
import io
import warnings
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hafs_case import StormCase  # noqa: F401

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import eccodes
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

def read_hafs_tp_records(filepath):
    """Return every APCP/tp record in a HAFS GRIB2 file with grid + accumulation
    metadata, as a list of dicts.

    Reads directly via eccodes rather than cfgrib so we can pick messages one
    at a time.  cfgrib fails when the file has tp on two different grids (parent
    + storm nest) because it tries to concatenate them without an index file.

    Each dict: npoints, lats, lons (−180…180), data (mm, fill→NaN),
    start_step, end_step, step_type.
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
                if sn != "tp":
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
                data = vals.reshape(nj, ni)
                missing = eccodes.codes_get(gid, "missingValue")
                data = np.where(np.abs(data - missing) < 1.0, np.nan, data)
                data = np.where(data < 0, np.nan, data)
                records.append({
                    "npoints": ni * nj,
                    "lats": lats_2d,
                    "lons": lons_180,
                    "data": data,
                    "start_step": start_step,
                    "end_step": end_step,
                    "step_type": step_type,
                })
            except Exception:
                pass
            finally:
                eccodes.codes_release(gid)
    return records


def pick_total_record(records):
    """Pick the per-interval precip BUCKET from a moving-nest file.

    HAFS storm.atm files carry two tp records on the nest grid: a per-interval
    bucket (e.g. 60->63h) and a 0->fhour "cumulative" total.  The cumulative
    record is a TRAP on a storm-following nest: precip accumulates per grid
    CELL, and a cell that stays under the eyewall racks up rain for the whole
    run, giving physically impossible 0->126h totals (~2400 mm) in storm-
    relative space rather than at any fixed point on the ground.

    The short bucket is geographically valid — over one output step the nest
    barely moves — so we take the SHORTEST positive window and SUM the buckets
    (each reprojected to its own lat/lon) across frames to build the true
    geographic storm total.

    Returns (record, "incremental"), or (None, None) for a zero-length window
    (e.g. F000) which contributes nothing.
    """
    finest = max(r["npoints"] for r in records)
    cand = [r for r in records if r["npoints"] == finest]
    buckets = [r for r in cand
               if (r["end_step"] or 0) > (r["start_step"] or 0)]
    if not buckets:
        return None, None
    window = lambda r: (r["end_step"] or 0) - (r["start_step"] or 0)
    return min(buckets, key=window), "incremental"


def load_hafs_precip(filepath):
    """Back-compat: (lats_2d, lons_2d_180, precip_mm) for the cumulative tp record."""
    records = read_hafs_tp_records(filepath)
    if not records:
        raise RuntimeError(f"tp not found in {filepath}")
    r, _ = pick_total_record(records)
    return r["lats"], r["lons"], r["data"]


def regrid_hafs(src_lats, src_lons, src_data, grid_lat, grid_lon):
    """
    Reproject HAFS tp from the moving storm grid onto the fixed lat/lon mesh.
    Uses linear interpolation which only fills points inside the storm grid
    convex hull — NaN outside, so no rectangular-edge stripe artifacts when
    the running max accumulates across frames.
    """
    pts = np.column_stack([src_lons.ravel(), src_lats.ravel()])
    vals = src_data.ravel()
    valid = np.isfinite(vals) & np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if valid.sum() < 4:
        return np.full(grid_lat.shape, np.nan)
    return griddata(
        pts[valid], vals[valid],
        (grid_lon, grid_lat),
        method="linear",
        fill_value=np.nan,
    )


def accumulate_hafs_step(running, interp, mode):
    """Fold one frame's regridded tp into the running event total on the fixed grid.

    - cumulative: tp is monotonic since init, so fmax stamps the latest total
      at each fixed point as the storm nest sweeps over it.
    - incremental: each file holds only its own interval's precip, so the
      increments are summed across forecast hours.
    """
    interp = np.nan_to_num(interp, nan=0.0)
    if mode == "incremental":
        return running + interp
    return np.fmax(running, interp)


def hafs_event_total(file_pairs, grid_lat, grid_lon, verbose=True):
    """Full-event accumulated HAFS precip (mm) on the fixed grid.

    Selects the cumulative tp record per frame and folds it in with the right
    reducer.  Returns (total, mode).
    """
    total = np.zeros(grid_lat.shape)
    mode_seen = None
    for fhour, fp in file_pairs:
        try:
            recs = read_hafs_tp_records(fp)
            if not recs:
                continue
            r, mode = pick_total_record(recs)
            if r is None:
                continue
        except Exception as e:
            if verbose:
                print(f"  F{fhour:03d} HAFS read failed: {e}")
            continue
        if mode_seen is None:
            mode_seen = mode
            if verbose:
                print(f"  APCP record: {mode} "
                      f"(window {r['start_step']}->{r['end_step']}h)")
        interp = regrid_hafs(r["lats"], r["lons"], r["data"], grid_lat, grid_lon)
        total = accumulate_hafs_step(total, interp, mode)
    return total, (mode_seen or "cumulative")


# =============================================================================
# TC track interpolation + MRMS mask
# =============================================================================

def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """Great-circle distance (km) from (lat1,lon1) to each point in arrays."""
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * \
        np.cos(np.radians(lat2_arr)) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def apply_tc_mask(lats, lons, data, valid_dt, case):
    """Zero out QPE beyond case.mask_radius_km from the TC center at valid_dt."""
    tc_lat, tc_lon = case.position_at(valid_dt)
    if lats.ndim == 1 and lons.ndim == 1:
        lons_2d, lats_2d = np.meshgrid(lons, lats)
    else:
        lats_2d, lons_2d = lats, lons
    dist = haversine_km(tc_lat, tc_lon, lats_2d, lons_2d)
    return np.where(dist <= case.mask_radius_km, data, 0.0)


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


def plot_frame(case, fhour, fixed_lons, fixed_lats, hafs_mm,
               mrms_lons, mrms_lats, mrms_mm,
               full_domain, out_path):
    valid_dt = case.init_dt + timedelta(hours=fhour)
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
        f"{case.model_label} Accumulated Precip\n"
        f"Init {case.init_dt:%Y-%m-%d %HZ} | "
        f"F{fhour:03d} (0–{fhour}h, valid {valid_dt:%Y-%m-%d %HZ})"
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
            f"({case.init_dt:%Y-%m-%d %HZ} – {valid_dt:%Y-%m-%d %HZ})"
        )
    else:
        axes[1].text(0.5, 0.5, "MRMS data unavailable",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("MRMS MultiSensor QPE — unavailable")

    plt.colorbar(cf, ax=axes, label="Accumulated Precipitation (mm)",
                 ticks=QPF_LEVELS, shrink=0.7, fraction=0.02)
    fig.suptitle(
        f"{case.storm_name} — {case.model_label} QPF vs MRMS QPE | "
        f"F{fhour:03d} ending {valid_dt:%Y-%m-%d %HZ}",
        fontsize=13, y=1.01,
    )
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def generate_animation(case):
    case.out_dir.mkdir(parents=True, exist_ok=True)
    case.mrms_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"HAFS run dir : {case.run_dir}")
    print(f"Init time    : {case.init_dt:%Y-%m-%d %HZ}")
    print(f"Output dir   : {case.out_dir}")

    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    if not file_pairs:
        print(f"\nNo files found matching {case.storm_glob()} in {case.run_dir}")
        return
    print(f"\nFound {len(file_pairs)} HAFS files")
    max_fhour = file_pairs[-1][0]

    # ------------------------------------------------------------------
    # Fixed domain from case (auto-domain resolved in from_yaml).
    # ------------------------------------------------------------------
    full_domain = case.domain
    lat_min_all, lat_max_all, lon_min_all, lon_max_all = full_domain
    print(f"\nUsing domain: lat [{lat_min_all:.1f}, {lat_max_all:.1f}]  "
          f"lon [{lon_min_all:.1f}, {lon_max_all:.1f}]")

    fixed_lons = np.arange(lon_min_all, lon_max_all + case.grid_res,
                           case.grid_res)
    fixed_lats = np.arange(lat_min_all, lat_max_all + case.grid_res,
                           case.grid_res)
    grid_lon, grid_lat = np.meshgrid(fixed_lons, fixed_lats)
    print(f"  Fixed grid  : {grid_lat.shape[0]}x{grid_lat.shape[1]} "
          f"at {case.grid_res} deg resolution")

    # ------------------------------------------------------------------
    # Pre-download all MRMS 1H files and crop to the fixed domain.
    # ------------------------------------------------------------------
    valid_end = case.init_dt + timedelta(hours=max_fhour)
    print(f"\nPre-caching MRMS 1H QPE: hours 1-{max_fhour} "
          f"(up to {valid_end:%Y-%m-%d %HZ}) ...")
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    mrms_hourly = {}
    for h in range(1, max_fhour + 1):
        t = case.init_dt + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, case.mrms_cache_dir)
            clat, clon, cdata = crop_to_domain(lat, lon, data,
                                               lat_min_all, lat_max_all,
                                               lon_min_all, lon_max_all)
            mrms_hourly[h] = (clat, clon, cdata)
            if h % 12 == 0 or h == max_fhour:
                print(f"  cached h{h:03d}/{max_fhour} ({t:%Y-%m-%d %HZ})")
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
    hafs_total = np.zeros(grid_lat.shape)
    mode_seen = None
    mrms_running_total = None
    last_mrms_h = 0

    # Growing TC swath on the fixed grid (union of mask_radius_km circles
    # along the track so far).  HAFS is accumulated unmasked but DISPLAYED
    # masked to this swath, so both panels show only the storm's rain — not
    # predecessor/frontal precip the large moving nest also contains.
    # Seed with the init position.
    hafs_swath = np.zeros(grid_lat.shape, dtype=bool)
    t0lat, t0lon = case.position_at(case.init_dt)
    hafs_swath |= (haversine_km(t0lat, t0lon, grid_lat, grid_lon)
                   <= case.mask_radius_km)

    for fhour, filepath in file_pairs:
        out_path = case.out_dir / f"qpf_frame_{fhour:03d}.png"

        # Update the HAFS event total regardless of whether we skip the PNG.
        # We select the cumulative 0->fhour tp record; fmax stamps the latest
        # total at each fixed point as the storm nest sweeps past.
        try:
            recs = read_hafs_tp_records(filepath)
            if not recs:
                raise RuntimeError("no tp records")
            r, mode = pick_total_record(recs)
            if r is None:
                # zero-length window (e.g. F000) — nothing to add.
                continue
            if mode_seen is None:
                mode_seen = mode
                print(f"  APCP record: {mode} "
                      f"(window {r['start_step']}->{r['end_step']}h)")
            hafs_interp = regrid_hafs(r["lats"], r["lons"], r["data"],
                                      grid_lat, grid_lon)
            hafs_total = accumulate_hafs_step(hafs_total, hafs_interp, mode)
        except Exception as e:
            print(f"  F{fhour:03d} HAFS load failed: {e}")
            continue

        # Advance the TC swath and accumulate MRMS for the new hours.  The MRMS
        # mask and the HAFS swath both use the interpolated track position, so
        # the two panels share an identical storm-only footprint.
        for h in range(last_mrms_h + 1, fhour + 1):
            valid_dt_h = case.init_dt + timedelta(hours=h)
            tlat, tlon = case.position_at(valid_dt_h)
            hafs_swath |= (haversine_km(tlat, tlon, grid_lat, grid_lon)
                           <= case.mask_radius_km)
            if h in mrms_hourly:
                clat, clon, data = mrms_hourly[h]
                data = apply_tc_mask(clat, clon, data, valid_dt_h, case)
                if mrms_running_total is None:
                    mrms_running_total = np.zeros_like(data)
                mrms_running_total += data
        last_mrms_h = fhour

        if out_path.exists():
            print(f"  F{fhour:03d} — already exists, skipping.")
            continue

        # Mask HAFS to the swath-so-far for display (accumulation stays unmasked).
        hafs_display = np.where(hafs_swath, hafs_total, 0.0)

        print(f"  F{fhour:03d} ({filepath.name}) ...", end=" ", flush=True)
        plot_frame(
            case, fhour,
            fixed_lons, fixed_lats, hafs_display,
            mrms_lons, mrms_lats, mrms_running_total,
            full_domain, out_path,
        )
        hafs_max = float(np.nanmax(hafs_display))
        mrms_max = (float(np.nanmax(mrms_running_total))
                    if mrms_running_total is not None else float("nan"))
        print(f"saved  (HAFS max {hafs_max:.0f} mm | MRMS max {mrms_max:.0f} mm)")

    print(f"\nAll frames written to {case.out_dir}")
    print("\nTo make an MP4:")
    print(f"  ffmpeg -r 4 -pattern_type glob -i '{case.out_dir}/qpf_frame_*.png' \\")
    print(f'         -vf "format=rgb24" -vcodec mpeg4 -q:v 3 -pix_fmt yuv420p '
          f"{case.out_dir}/../qpf_animation.mp4")


if __name__ == "__main__":
    from hafs_case import from_yaml
    generate_animation(from_yaml(sys.argv[1]))
