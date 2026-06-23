import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import types
from datetime import datetime
import numpy as np
from compare import (load_comparison, score_matrix, plot_categorical_compare,
                     plot_fss_compare, _model_colors, replot_from_csv,
                     _check_same_init, _init_tag, _plot_comparison)


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


def _toy_rows():
    g = np.zeros((12, 12)); g[4:8, 4:8] = 20.0
    swath = np.ones((12, 12), dtype=bool)
    # Use the real model labels ("HAFS-A"/"HAFS-B") so the plots exercise the
    # same names production uses — a regression guard for the color lookup.
    models = [
        {"name": "HAFS-A", "forecasts": {"parent": g.copy(), "nest": g.copy()},
         "obs": {"MRMS": g.copy()}},
        {"name": "HAFS-B", "forecasts": {"parent": g.copy(), "nest": g.copy()},
         "obs": {"MRMS": g.copy()}},
    ]
    return score_matrix(models, swath, [5.0, 25.0, 50.0], [1, 3], 0.05)


def test_model_colors_distinct_and_not_gray():
    # Real production labels must each get a distinct, non-gray color.
    colors = _model_colors(["HAFS-A", "HAFS-B"])
    assert colors["HAFS-A"] != colors["HAFS-B"]
    assert "gray" not in (colors["HAFS-A"], colors["HAFS-B"])


def test_plot_comparison_writes_init_tagged_pngs():
    cat, fss = _toy_rows()
    d = Path(tempfile.mkdtemp())
    slug = "hurricane_helene_2024092400"
    cat_cols = ["init", "model", "forecast", "observation", "threshold",
                "a", "b", "c", "d", "ets", "csi", "bias", "pod", "far", "hss"]
    fss_cols = ["init", "model", "forecast", "observation", "threshold",
                "scale_cells", "scale_km", "fss"]
    for r in cat:
        r["init"] = "2024092400"
    for r in fss:
        r["init"] = "2024092400"
    with open(d / f"compare_categorical_{slug}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cat_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(cat)
    with open(d / f"compare_fss_{slug}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fss_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(fss)
    from compare import _plot_comparison
    _plot_comparison(d, slug, "Hurricane Helene (init 2024-09-24 00Z)", [25, 50])
    assert (d / f"compare_categorical_{slug}.png").stat().st_size > 0
    assert (d / f"compare_fss_{slug}.png").stat().st_size > 0


def test_score_matrix_common_n_across_models_when_footprints_differ():
    obs = np.full((6, 6), 10.0)
    a_grid = np.full((6, 6), 20.0); a_grid[4:, :] = np.nan   # finite rows 0-3 (24)
    b_grid = np.full((6, 6), 20.0); b_grid[5:, :] = np.nan   # finite rows 0-4 (30)
    swath = np.ones((6, 6), dtype=bool)
    models = [
        {"name": "HFSA", "forecasts": {"nest": a_grid}, "obs": {"MRMS": obs}},
        {"name": "HFSB", "forecasts": {"nest": b_grid}, "obs": {"MRMS": obs}},
    ]
    cat, _ = score_matrix(models, swath, [5.0], [1], 0.05)
    by_model = {r["model"]: r for r in cat if r["threshold"] == 5.0}
    na = sum(by_model["HFSA"][k] for k in "abcd")
    nb = sum(by_model["HFSB"][k] for k in "abcd")
    assert na == nb == 24   # intersection = rows 0-3 = 24 cells


def test_plots_write_png_files():
    cat, fss = _toy_rows()
    d = Path(tempfile.mkdtemp())
    cat_png = d / "cat.png"
    fss_png = d / "fss.png"
    plot_categorical_compare(cat, "Test", cat_png, observation="MRMS")
    plot_fss_compare(fss, "Test", fss_png, observation="MRMS",
                     forecast="parent", plot_thresholds=(5.0, 25.0))
    assert cat_png.exists() and cat_png.stat().st_size > 0
    assert fss_png.exists() and fss_png.stat().st_size > 0


def test_init_tag_appends_and_formats():
    slug, title = _init_tag("Hurricane Helene", datetime(2024, 9, 24, 0))
    assert slug == "hurricane_helene_2024092400"
    assert title == "Hurricane Helene (init 2024-09-24 00Z)"


def test_init_tag_dedups_when_label_has_init():
    slug, _ = _init_tag("helene 2024092400", datetime(2024, 9, 24, 0))
    assert slug == "helene_2024092400"   # not ..._2024092400_2024092400


def test_check_same_init_passes_when_equal():
    c = types.SimpleNamespace(init_dt=datetime(2024, 9, 24, 0),
                              init_str="2024092400")
    _check_same_init([c, c], ["a.yaml", "b.yaml"])   # no raise


def test_check_same_init_raises_when_differ():
    a = types.SimpleNamespace(init_dt=datetime(2024, 9, 24, 0),
                              init_str="2024092400")
    b = types.SimpleNamespace(init_dt=datetime(2024, 9, 24, 12),
                              init_str="2024092412")
    try:
        _check_same_init([a, b], ["a.yaml", "b.yaml"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "2024092400" in str(e) and "2024092412" in str(e)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
