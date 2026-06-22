"""Local unit tests for hafs_case ATCF parsing (no Hercules data needed).

Run directly:   python3 analysis/tests/test_hafs_case.py
Or via pytest:  pytest analysis/tests/test_hafs_case.py -v
"""
import sys
from datetime import datetime
from pathlib import Path

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FIX = Path(__file__).resolve().parent / "fixtures"

from hafs_case import decode_latlon, parse_atcfunix


def test_decode_latlon():
    assert decode_latlon("168N") == 16.8
    assert decode_latlon("832W") == -83.2
    assert decode_latlon("105S") == -10.5
    assert decode_latlon("50E") == 5.0


def test_parse_atcfunix_reproduces_helene_track():
    name, init_dt, track = parse_atcfunix(FIX / "helene.atcfunix")
    assert name == "Helene"
    assert init_dt == datetime(2024, 9, 24, 0)
    # First three known 6-hourly fixes from the current hardcoded TC_TRACK_6H.
    assert track[0] == (datetime(2024, 9, 24, 0), 16.8, -83.2)
    assert track[1] == (datetime(2024, 9, 24, 6), 17.8, -83.5)
    assert track[2] == (datetime(2024, 9, 24, 12), 19.0, -83.8)
    # Sorted ascending by valid time, no duplicate lead hours.
    times = [t for t, _, _ in track]
    assert times == sorted(times)
    assert len(times) == len(set(times))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
