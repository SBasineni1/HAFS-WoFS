import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FIX = Path(__file__).resolve().parent / "fixtures"

from best_track import parse_bdeck


def test_parse_bdeck_times_from_column_and_dedup():
    track = parse_bdeck(FIX / "bal092024_sample.dat")
    # 5 lines but the first two share 2024092400 -> 4 unique fixes.
    assert len(track) == 4
    assert track[0] == (datetime(2024, 9, 24, 0), 16.8, -83.2)
    assert track[1] == (datetime(2024, 9, 24, 6), 17.8, -83.5)
    assert track[2] == (datetime(2024, 9, 24, 12), 19.0, -83.8)
    times = [t for t, _, _ in track]
    assert times == sorted(times)


def test_parse_bdeck_no_fixes_raises(tmp_path=None):
    import tempfile
    p = Path(tempfile.mkdtemp()) / "empty.dat"
    p.write_text("AL, 09, 2024092400,   , CARQ,   0, 168N,  832W\n")  # not BEST
    try:
        parse_bdeck(p)
        assert False, "expected ValueError"
    except ValueError as e:
        assert str(p) in str(e)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
