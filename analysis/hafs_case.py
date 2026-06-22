"""StormCase config + ATCF track parsing for the HAFS QPF/ETS framework.

Dependency-light on purpose (stdlib + numpy + yaml only) so it imports and
tests off-Hercules, away from cfgrib/boto3/eccodes/cartopy.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def decode_latlon(token):
    """ATCF tenths-of-degree + hemisphere token -> signed float degrees.

    '168N' -> 16.8, '832W' -> -83.2, '105S' -> -10.5, '50E' -> 5.0.
    """
    token = token.strip()
    hemi = token[-1].upper()
    value = int(token[:-1]) / 10.0
    if hemi in ("S", "W"):
        value = -value
    return value


def parse_atcfunix(path):
    """Parse a HAFS .atcfunix track file.

    Returns (storm_name_or_None, init_dt, track) where track is a list of
    (valid_dt, lat, lon) deduped by lead hour (TAU) and sorted ascending.
    """
    init_dt = None
    name = None
    by_tau = {}
    with open(path) as fh:
        for line in fh:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 8:
                continue
            try:
                warn = datetime.strptime(cols[2], "%Y%m%d%H")
                tau = int(cols[5])
                lat = decode_latlon(cols[6])
                lon = decode_latlon(cols[7])
            except (ValueError, IndexError):
                continue
            if init_dt is None:
                init_dt = warn
            if tau not in by_tau:
                by_tau[tau] = (warn + timedelta(hours=tau), lat, lon)
            # Storm name: trailing alpha field (index 27 in standard atcfunix).
            if name is None and len(cols) > 27 and re.fullmatch(r"[A-Za-z]+", cols[27]):
                name = cols[27].title()
    track = [by_tau[t] for t in sorted(by_tau)]
    return name, init_dt, track


def detect_model(run_dir):
    """'HAFS-A'/'HAFS-B'/'HAFS' from HFSA/HFSB in the run-dir path."""
    s = str(run_dir).upper()
    if "HFSA" in s:
        return "HAFS-A"
    if "HFSB" in s:
        return "HAFS-B"
    return "HAFS"


def auto_domain(track, pad_deg=2.0):
    """(lat_min, lat_max, lon_min, lon_max) = padded bbox of the track."""
    lats = [la for _, la, _ in track]
    lons = [lo for _, _, lo in track]
    return (min(lats) - pad_deg, max(lats) + pad_deg,
            min(lons) - pad_deg, max(lons) + pad_deg)


@dataclass
class StormCase:
    run_dir: Path
    init_dt: datetime
    storm_name: str
    model_label: str
    domain: tuple          # (lat_min, lat_max, lon_min, lon_max)
    grid_res: float
    mask_radius_km: float
    display_radius_km: float
    thresholds_mm: list
    out_dir: Path
    mrms_cache_dir: Path
    stage4_cache_dir: Path
    fhours_filter: list
    track: list            # [(valid_dt, lat, lon), ...]
    case_slug: str
    init_str: str

    def position_at(self, valid_dt):
        """Linear interpolation of the track to any time; clamps to endpoints."""
        times = [t for t, _, _ in self.track]
        lats = [la for _, la, _ in self.track]
        lons = [lo for _, _, lo in self.track]
        if valid_dt <= times[0]:
            return lats[0], lons[0]
        if valid_dt >= times[-1]:
            return lats[-1], lons[-1]
        for i in range(len(times) - 1):
            if times[i] <= valid_dt <= times[i + 1]:
                frac = ((valid_dt - times[i]).total_seconds()
                        / (times[i + 1] - times[i]).total_seconds())
                return (lats[i] + frac * (lats[i + 1] - lats[i]),
                        lons[i] + frac * (lons[i + 1] - lons[i]))
        return lats[-1], lons[-1]

    def fixed_grid(self):
        """Fixed lat/lon verification/plot mesh from domain + grid_res."""
        lat_min, lat_max, lon_min, lon_max = self.domain
        fixed_lons = np.arange(lon_min, lon_max + self.grid_res, self.grid_res)
        fixed_lats = np.arange(lat_min, lat_max + self.grid_res, self.grid_res)
        return np.meshgrid(fixed_lons, fixed_lats)[::-1]  # (grid_lat, grid_lon)

    def parent_glob(self):
        return f"**/*{self.init_str}*parent.atm.f*.grb2"

    def storm_glob(self):
        return f"**/*{self.init_str}*storm.atm.f*.grb2"
