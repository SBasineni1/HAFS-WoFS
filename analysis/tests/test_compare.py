import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare import load_comparison


def _write(cfg_text):
    p = Path(tempfile.mkdtemp()) / "helene_compare.yaml"
    p.write_text(cfg_text)
    return p


def test_load_comparison_defaults():
    p = _write(
        "label: Hurricane Helene\n"
        "cases: [storms/helene_hfsa.yaml, storms/helene_hfsb.yaml]\n"
        "best_track: /data/bal092024.dat\n"
    )
    cfg = load_comparison(p)
    assert cfg["label"] == "Hurricane Helene"
    assert cfg["case_paths"] == ["storms/helene_hfsa.yaml", "storms/helene_hfsb.yaml"]
    assert cfg["best_track"] == "/data/bal092024.dat"
    assert cfg["out_dir"] == Path("analysis/output/helene_compare")
    assert cfg["fss_scales_cells"] == [1, 3, 5, 11, 21, 41]
    assert cfg["fss_plot_thresholds"] == [10, 25, 50]
    assert cfg["thresholds_mm"] is None


def test_load_comparison_requires_two_cases():
    p = _write("cases: [a.yaml]\nbest_track: /x.dat\n")
    try:
        load_comparison(p); assert False
    except ValueError:
        pass


def test_load_comparison_requires_best_track():
    p = _write("cases: [a.yaml, b.yaml]\n")
    try:
        load_comparison(p); assert False
    except KeyError:
        pass


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
