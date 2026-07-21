import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import types
from datetime import datetime
import numpy as np
from compare import (continuous_matrix, load_comparison, score_matrix,
                     plot_categorical_compare, plot_fss_compare,
                     plot_rmse_compare, _model_colors, replot_from_csv,
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
    continuous = [
        {"init": "2024092400", "model": "HAFS-A", "forecast": "parent",
         "observation": "MRMS", "n": 100, "rmse": 25.4, "mae": 20.0,
         "bias": 2.0, "r": 0.8},
        {"init": "2024092400", "model": "HAFS-B", "forecast": "parent",
         "observation": "MRMS", "n": 100, "rmse": 50.8, "mae": 40.0,
         "bias": 4.0, "r": 0.7},
    ]
    continuous_cols = ["init", "model", "forecast", "observation", "n",
                       "rmse", "mae", "bias", "r"]
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
    with open(d / f"compare_continuous_{slug}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=continuous_cols)
        w.writeheader(); w.writerows(continuous)
    from compare import _plot_comparison
    _plot_comparison(d, slug, "Hurricane Helene (init 2024-09-24 00Z)", [25, 50])
    assert (d / f"compare_categorical_{slug}.png").stat().st_size > 0
    assert (d / f"compare_fss_{slug}.png").stat().st_size > 0
    assert (d / f"compare_performance_{slug}.png").stat().st_size > 0
    assert (d / f"compare_rmse_{slug}.png").stat().st_size > 0


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


def test_continuous_matrix_uses_common_footprint_and_computes_rmse():
    obs = np.full((3, 3), 10.0)
    a_grid = np.full((3, 3), 12.0)
    b_grid = np.full((3, 3), 14.0)
    a_grid[2, :] = np.nan
    models = [
        {"name": "HAFS-A", "forecasts": {"parent": a_grid},
         "obs": {"MRMS": obs}},
        {"name": "HAFS-B", "forecasts": {"parent": b_grid},
         "obs": {"MRMS": obs}},
    ]
    rows = continuous_matrix(models, np.ones((3, 3), dtype=bool))
    by_model = {row["model"]: row for row in rows}
    assert by_model["HAFS-A"]["n"] == by_model["HAFS-B"]["n"] == 6
    assert by_model["HAFS-A"]["rmse"] == 2.0
    assert by_model["HAFS-B"]["rmse"] == 4.0


def test_plots_write_png_files():
    cat, fss = _toy_rows()
    d = Path(tempfile.mkdtemp())
    cat_png = d / "cat.png"
    fss_png = d / "fss.png"
    rmse_png = d / "rmse.png"
    plot_categorical_compare(cat, "Test", cat_png, observation="MRMS")
    plot_fss_compare(fss, "Test", fss_png, observation="MRMS",
                     forecast="parent", plot_thresholds=(5.0, 25.0))
    plot_rmse_compare([
        {"model": "HAFS-A", "forecast": "parent", "observation": "MRMS",
         "n": 20, "rmse": 25.4, "mae": 20.0, "bias": 2.0, "r": 0.8},
        {"model": "HAFS-B", "forecast": "parent", "observation": "MRMS",
         "n": 20, "rmse": 50.8, "mae": 40.0, "bias": 3.0, "r": 0.7},
    ], "Test", rmse_png)
    assert cat_png.exists() and cat_png.stat().st_size > 0
    assert fss_png.exists() and fss_png.stat().st_size > 0
    assert rmse_png.exists() and rmse_png.stat().st_size > 0


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


def test_csi_from_pod_sr_known_values():
    """CSI reconstructed from POD and success ratio: 1/CSI = 1/SR + 1/POD - 1."""
    from compare import csi_from_pod_sr
    assert abs(csi_from_pod_sr(1.0, 1.0) - 1.0) < 1e-12       # perfect
    assert abs(csi_from_pod_sr(0.5, 0.5) - 1.0 / 3.0) < 1e-12  # 1/(2+2-1)


def test_csi_from_pod_sr_vectorizes():
    """The contour grid needs CSI over POD/SR meshes, elementwise."""
    from compare import csi_from_pod_sr
    pod = np.array([[1.0, 0.5]])
    sr = np.array([[1.0, 0.5]])
    out = csi_from_pod_sr(pod, sr)
    assert out.shape == (1, 2)
    assert abs(out[0, 0] - 1.0) < 1e-12
    assert abs(out[0, 1] - 1.0 / 3.0) < 1e-12


def test_performance_points_extracts_sr_pod_filtered():
    """Success ratio = 1 - FAR; only the requested forecast/observation kept."""
    from compare import performance_points
    rows = [
        {"model": "HAFS-A", "forecast": "parent", "observation": "MRMS",
         "threshold": 25.0, "pod": 0.8, "far": 0.4, "csi": 0.5, "bias": 1.2},
        {"model": "HAFS-A", "forecast": "nest", "observation": "MRMS",
         "threshold": 25.0, "pod": 0.1, "far": 0.9, "csi": 0.05, "bias": 0.3},
        {"model": "HAFS-B", "forecast": "parent", "observation": "Stage IV",
         "threshold": 25.0, "pod": 0.7, "far": 0.5, "csi": 0.4, "bias": 1.0},
        {"model": "HAFS-B", "forecast": "parent", "observation": "MRMS",
         "threshold": 50.0, "pod": 0.6, "far": 0.5, "csi": 0.4, "bias": 0.9},
    ]
    pts = performance_points(rows, observation="MRMS", forecast="parent")
    assert len(pts) == 2  # nest row and Stage IV row filtered out
    a = next(p for p in pts if p["model"] == "HAFS-A")
    assert abs(a["success_ratio"] - 0.6) < 1e-12
    assert abs(a["pod"] - 0.8) < 1e-12
    assert a["threshold"] == 25.0


def test_plot_performance_diagram_writes_png():
    from compare import plot_performance_diagram
    cat, _ = _toy_rows()
    d = Path(tempfile.mkdtemp())
    png = d / "perf.png"
    plot_performance_diagram(cat, "Test", png, observation="MRMS",
                             forecast="parent")
    assert png.exists() and png.stat().st_size > 0


def test_cell_area_km2_scales_with_latitude():
    """Grid-cell area shrinks with cos(latitude); N-S extent is constant."""
    from compare import cell_area_km2
    lat = np.array([[0.0, 60.0]])
    a = cell_area_km2(lat, 0.1)
    assert abs(a[0, 0] - (0.1 * 111.0) ** 2) < 1e-6
    assert abs(a[0, 1] - (0.1 * 111.0) ** 2 * np.cos(np.radians(60.0))) < 1e-6


def test_accumulation_stats_area_and_max():
    """Max over the swath, plus km^2 of cells at/above each threshold."""
    from compare import accumulation_stats
    lat = np.zeros((3, 3))            # cos(0)=1 -> every cell (0.1*111)^2 km^2
    field = np.zeros((3, 3))
    field[0, 0] = 30.0
    field[0, 1] = 30.0
    field[1, 1] = 10.0
    swath = np.ones((3, 3), dtype=bool)
    st = accumulation_stats(field, swath, lat, 0.1, [25.0, 5.0])
    cell = (0.1 * 111.0) ** 2
    assert abs(st["max_mm"] - 30.0) < 1e-9
    assert abs(st["area_km2"][25.0] - 2 * cell) < 1e-6   # two cells >= 25
    assert abs(st["area_km2"][5.0] - 3 * cell) < 1e-6    # three cells >= 5


def test_accumulation_stats_respects_swath():
    """Points outside the swath are excluded from max and area."""
    from compare import accumulation_stats
    lat = np.zeros((3, 3))
    field = np.zeros((3, 3))
    field[0, 0] = 99.0                # this cell is outside the swath
    field[2, 2] = 20.0
    swath = np.ones((3, 3), dtype=bool)
    swath[0, 0] = False
    st = accumulation_stats(field, swath, lat, 0.1, [5.0])
    cell = (0.1 * 111.0) ** 2
    assert abs(st["max_mm"] - 20.0) < 1e-9                # 99 excluded
    assert abs(st["area_km2"][5.0] - 1 * cell) < 1e-6     # only the (2,2) cell


def test_plot_storm_total_writes_png():
    from compare import plot_storm_total
    lat1 = np.linspace(20, 30, 12)
    lon1 = np.linspace(-90, -80, 12)
    grid_lon, grid_lat = np.meshgrid(lon1, lat1)
    g = np.zeros((12, 12))
    g[4:8, 4:8] = 120.0
    swath = np.ones((12, 12), dtype=bool)
    sources = [("MRMS", g.copy()), ("HAFS-A", g * 0.8), ("HAFS-B", g * 1.1)]
    d = Path(tempfile.mkdtemp())
    png = d / "storm_total.png"
    plot_storm_total(sources, swath, grid_lat, grid_lon, 0.1, "Test", png)
    assert png.exists() and png.stat().st_size > 0


def test_interp_center_rmw_interpolates_and_falls_back():
    """Center + RMW linearly interpolated between best-track fixes; the fallback
    RMW is used (and flagged) only when no fix carries an RMW."""
    from compare import _interp_center_rmw
    fixes = [(datetime(2024, 9, 26, 0), 20.0, -85.0, 40.0),
             (datetime(2024, 9, 26, 12), 22.0, -85.0, 60.0)]
    lat, lon, rmw, fb = _interp_center_rmw(fixes, datetime(2024, 9, 26, 6), 50.0)
    assert fb is False
    assert abs(lat - 21.0) < 1e-6          # midpoint latitude
    assert abs(rmw - 50.0) < 1e-9          # midpoint of 40 and 60 km

    no_rmw = [(datetime(2024, 9, 26, 0), 20.0, -85.0, None),
              (datetime(2024, 9, 26, 12), 22.0, -85.0, None)]
    lat, lon, rmw, fb = _interp_center_rmw(no_rmw, datetime(2024, 9, 26, 6), 50.0)
    assert fb is True and rmw == 50.0


def test_storm_relative_field_uniform_inside_nan_outside():
    """A uniform field maps to that constant within radius, NaN beyond it."""
    from compare import storm_relative_field
    lat1 = np.linspace(20, 40, 61)
    lon1 = np.linspace(-100, -70, 91)
    grid_lon, grid_lat = np.meshgrid(lon1, lat1)
    field = np.full(grid_lat.shape, 5.0)
    x, y, out = storm_relative_field(field, grid_lat, grid_lon, (30.0, -85.0),
                                     rmw_km=50.0, radius_rmw=4.0, res_rmw=0.5)
    r = np.hypot(x, y)
    assert np.allclose(out[r <= 1.0], 5.0)          # interior preserved
    assert np.all(np.isnan(out[r > 4.0 + 1e-9]))    # masked past the radius


def test_radial_profile_uniform_median():
    """Radial bins of a uniform disk all report that constant as the median."""
    from compare import radial_profile
    axis = np.arange(-6, 6 + 0.1, 0.2)
    x, y = np.meshgrid(axis, axis)
    r = np.hypot(x, y)
    field = np.where(r <= 6, 5.0, np.nan)
    prof = radial_profile([field], radius_rmw=6.0, res_rmw=0.2, bin_rmw=0.4)
    assert len(prof) > 0
    assert all(abs(b["median"] - 5.0) < 1e-9 for b in prof)
    assert prof[0]["r_lo"] == 0.0            # bins start at the center


def test_storm_relative_composite_pools_windows():
    """The compositing loop pools every lead window per source and normalizes
    each model on its own track. Patches the GRIB/MRMS loaders so the loop's
    bookkeeping is exercised without Hercules data."""
    import compare
    import cycles
    import ets_score
    lat1 = np.linspace(24, 38, 71)
    lon1 = np.linspace(-92, -76, 81)
    grid_lon, grid_lat = np.meshgrid(lon1, lat1)
    track = [(datetime(2024, 9, 26, 0), 31.0, -84.0),
             (datetime(2024, 9, 27, 0), 32.0, -83.0)]
    fixes = [(datetime(2024, 9, 26, 0), 31.0, -84.0, 40.0),
             (datetime(2024, 9, 27, 0), 32.0, -83.0, 45.0)]

    def fake_parent(case, f1, f2, gla, glo):
        return np.full(gla.shape, 10.0)

    def fake_mrms(vs, ve, cache, gla, glo):
        return np.full(gla.shape, 8.0)

    saved = (cycles.parent_window_total, ets_score.build_mrms_total_window,
             compare.parse_bdeck_fixes)
    cycles.parent_window_total = fake_parent
    ets_score.build_mrms_total_window = fake_mrms
    compare.parse_bdeck_fixes = lambda path: fixes
    try:
        cases = [types.SimpleNamespace(init_dt=datetime(2024, 9, 26, 0),
                                       mrms_cache_dir=Path("/tmp"),
                                       model_label=name, track=track)
                 for name in ("HAFS-A", "HAFS-B")]
        comps, radial, fb = compare.storm_relative_composite(
            cases, "ignored", grid_lat, grid_lon, max_fhour=12,
            accumulation_hours=6)
    finally:
        (cycles.parent_window_total, ets_score.build_mrms_total_window,
         compare.parse_bdeck_fixes) = saved

    assert [name for name, _ in comps] == ["MRMS", "HAFS-A", "HAFS-B"]
    assert fb is False                       # fixes carried an RMW
    # Uniform inputs -> composite radial medians equal the input constants.
    assert all(abs(b["median"] - 8.0) < 1e-9 for b in radial["MRMS"])
    assert all(abs(b["median"] - 10.0) < 1e-9 for b in radial["HAFS-A"])


def test_plot_storm_relative_writes_png():
    from compare import plot_storm_relative, radial_profile
    axis = np.arange(-6, 6 + 0.1, 0.2)
    x, y = np.meshgrid(axis, axis)
    r = np.hypot(x, y)
    obs = np.where(r <= 6, 20.0 * np.exp(-r), np.nan)
    a = np.where(r <= 6, 15.0 * np.exp(-r / 1.3), np.nan)
    b = np.where(r <= 6, 18.0 * np.exp(-r / 1.1), np.nan)
    composites = [("MRMS", obs), ("HAFS-A", a), ("HAFS-B", b)]
    radial = {name: radial_profile([f]) for name, f in composites}
    d = Path(tempfile.mkdtemp())
    png = d / "storm_rel.png"
    plot_storm_relative(composites, radial, "Test", png)
    assert png.exists() and png.stat().st_size > 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
