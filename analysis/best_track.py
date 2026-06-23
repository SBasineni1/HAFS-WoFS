"""Parse NHC ATCF best-track (b-deck) files into a track for verification.

A b-deck holds 'BEST' fix lines; unlike the HAFS .atcfunix (init + TAU), a
b-deck line's valid time is column 3 (YYYYMMDDHH) directly and its TAU is 0.
The same time repeats across 34/50/64-kt wind-radii lines, so dedupe by time.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hafs_case import decode_latlon


def parse_bdeck(path):
    """Return [(valid_dt, lat, lon), ...] from a b-deck, deduped + sorted."""
    by_time = {}
    with open(path) as fh:
        for line in fh:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 8 or cols[4] != "BEST":
                continue
            try:
                t = datetime.strptime(cols[2], "%Y%m%d%H")
                lat = decode_latlon(cols[6])
                lon = decode_latlon(cols[7])
            except (ValueError, IndexError):
                continue
            if t not in by_time:
                by_time[t] = (t, lat, lon)
    track = [by_time[t] for t in sorted(by_time)]
    if not track:
        raise ValueError(f"No BEST fixes parsed from {path}")
    return track
