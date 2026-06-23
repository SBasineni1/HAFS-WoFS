import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from compare import load_comparison, score_matrix


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


def test_score_matrix_shapes_and_perfect_fss():
    g = np.zeros((12, 12)); g[4:8, 4:8] = 20.0
    swath = np.ones((12, 12), dtype=bool)
    models = [
        {"name": "HFSA", "forecasts": {"parent": g.copy()},
         "obs": {"MRMS": g.copy(), "Stage IV": None}},
        {"name": "HFSB", "forecasts": {"parent": g.copy()},
         "obs": {"MRMS": g.copy()}},
    ]
    thresholds = [5.0, 50.0]
    scales = [1, 3]
    cat, fss = score_matrix(models, swath, thresholds, scales, 0.05)
    # 2 models x 1 forecast x 1 obs (None skipped) x 2 thr = 4 categorical rows.
    assert len(cat) == 4
    # FSS rows: 2 x 1 x 1 x 2 thr x 2 scales = 8.
    assert len(fss) == 8
    # Forecast == obs -> FSS == 1.0 at thr=5 where events exist.
    f5 = [r for r in fss if r["threshold"] == 5.0]
    assert all(abs(r["fss"] - 1.0) < 1e-9 for r in f5)
    # Categorical rows carry the hss key and the model/forecast/observation tags.
    assert "hss" in cat[0] and cat[0]["model"] in ("HFSA", "HFSB")
    assert cat[0]["observation"] == "MRMS"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
