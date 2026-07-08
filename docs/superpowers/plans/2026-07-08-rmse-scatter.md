# RMSE Scatter Product Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add storm-total continuous verification (RMSE, MAE, bias, Pearson r) of HAFS QPF vs MRMS/Stage IV as a new `rmse` product: hexbin scatter figure + CSV, sharing one field-build pass with the ETS product.

**Architecture:** A `continuous_scores` metric function joins `skill_metrics.py`. The expensive field-building block of `ets_full.compute_ets` is extracted into `build_verification_fields(case)` (stays in `ets_full.py` — it is already the shared hub `compare.py` imports from). A new `rmse_scatter.py` product consumes those fields and renders forecast-vs-observed hexbin panels. `run.py` gains an `rmse` command; `all` builds fields once and feeds both ETS and RMSE.

**Tech Stack:** Python 3, numpy, matplotlib (Agg). Tests are stdlib+numpy standalone scripts (`python3 analysis/tests/<file>.py`), also pytest-compatible.

**Spec:** `docs/superpowers/specs/2026-07-08-rmse-scatter-design.md`

## Global Constraints

- Repo root: `/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS` — all paths below are relative to it; run everything from the repo root.
- **Do NOT `git commit` or `git push`.** Leave all changes in the working tree; Suchit reviews and commits himself. (Where a task template would say "commit", instead report the task done.)
- Never commit/add data files (`.nc`, `.grb2`, `.grb`).
- New figure/CSV outputs go to `case.out_dir` and are init-tagged via `case.output_slug` (already handled by the case object — just use it).
- `bias` sign convention everywhere: `mean(forecast − observed)`; positive = over-forecast.
- Tests must pass BOTH ways: `python3 analysis/tests/<file>.py` and `pytest analysis/tests/<file>.py -v`. The full local suite is: `for f in analysis/tests/test_*.py; do python3 "$f" || break; done`
- Console/plot text style matches existing products (see `ets_full.py` for reference).

---

### Task 1: `continuous_scores` in `skill_metrics.py`

**Files:**
- Modify: `analysis/skill_metrics.py` (append at end of file)
- Test: `analysis/tests/test_skill_metrics.py` (append tests + they run via the existing `_run_all`)

**Interfaces:**
- Consumes: nothing new (numpy already imported in both files).
- Produces: `continuous_scores(fcst, obs) -> dict` with keys `n` (int), `rmse`, `mae`, `bias`, `r` (floats, NaN where undefined). Task 3 imports it as `from skill_metrics import continuous_scores`.

- [ ] **Step 1: Write the failing tests**

Append to `analysis/tests/test_skill_metrics.py`, immediately BEFORE the `_run_all()` definition (so discovery picks them up):

```python
from skill_metrics import continuous_scores


def test_continuous_scores_hand_computed():
    # fcst=[1,2,3], obs=[0,2,5] -> err=[1,0,-2]
    # rmse=sqrt(5/3), mae=1, bias=-1/3, r=15/sqrt(228)
    s = continuous_scores(np.array([1.0, 2.0, 3.0]),
                          np.array([0.0, 2.0, 5.0]))
    assert s["n"] == 3
    assert abs(s["rmse"] - np.sqrt(5.0 / 3.0)) < 1e-12
    assert abs(s["mae"] - 1.0) < 1e-12
    assert abs(s["bias"] - (-1.0 / 3.0)) < 1e-12
    assert abs(s["r"] - 15.0 / np.sqrt(228.0)) < 1e-12


def test_continuous_scores_perfect_forecast():
    f = np.array([0.0, 5.0, 20.0, 100.0])
    s = continuous_scores(f, f.copy())
    assert s["rmse"] == 0.0 and s["mae"] == 0.0 and s["bias"] == 0.0
    assert abs(s["r"] - 1.0) < 1e-12


def test_continuous_scores_empty_is_nan():
    s = continuous_scores(np.array([]), np.array([]))
    assert s["n"] == 0
    assert all(np.isnan(s[k]) for k in ("rmse", "mae", "bias", "r"))


def test_continuous_scores_constant_field_r_is_nan():
    # Zero variance in obs -> correlation undefined -> NaN (not a warning/crash).
    s = continuous_scores(np.array([1.0, 2.0, 3.0]),
                          np.array([4.0, 4.0, 4.0]))
    assert np.isnan(s["r"])
    assert abs(s["bias"] - (-2.0)) < 1e-12


def test_continuous_scores_single_point_r_is_nan():
    s = continuous_scores(np.array([3.0]), np.array([1.0]))
    assert s["n"] == 1
    assert abs(s["rmse"] - 2.0) < 1e-12
    assert np.isnan(s["r"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS" && python3 analysis/tests/test_skill_metrics.py`
Expected: `ImportError: cannot import name 'continuous_scores' from 'skill_metrics'`

- [ ] **Step 3: Implement `continuous_scores`**

Append to `analysis/skill_metrics.py`:

```python
def continuous_scores(fcst, obs):
    """Continuous verification scores over 1-D arrays of valid points.

    The caller selects the points (same footprint as the categorical
    scores: swath & finite obs & finite fcst). bias = mean(fcst - obs),
    so positive means over-forecast. r is Pearson correlation, NaN when
    n < 2 or either field has zero variance; every score is NaN at n = 0.
    """
    fcst = np.asarray(fcst, dtype=float)
    obs = np.asarray(obs, dtype=float)
    n = int(fcst.size)
    if n == 0:
        return dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan, r=np.nan)
    err = fcst - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    if n < 2 or np.std(fcst) == 0.0 or np.std(obs) == 0.0:
        r = np.nan
    else:
        r = float(np.corrcoef(fcst, obs)[0, 1])
    return dict(n=n, rmse=rmse, mae=mae, bias=bias, r=r)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS" && python3 analysis/tests/test_skill_metrics.py`
Expected: all tests print `PASS`, including the 5 new ones (9 passed total).

---

### Task 2: Extract `build_verification_fields` in `ets_full.py`

**Files:**
- Modify: `analysis/ets_full.py:156-233` (the `compute_ets` function)
- Test: existing `analysis/tests/test_ets_full.py` (must still pass unchanged)

**Interfaces:**
- Consumes: everything `compute_ets` already imports (no new imports).
- Produces (Tasks 3 & 4 rely on these exact names):
  - `build_verification_fields(case) -> dict` with keys:
    `max_fhour` (int), `grid_lat`, `grid_lon`, `nest_total`, `apcp_mode` (str),
    `parent_total`, `mrms_total`, `stage4_grid` (2-D array or **None**),
    `s4_label` (str), `swath` (bool array).
    Raises `RuntimeError` when no nest files match or no MRMS hour loads.
  - `field_pairs(fields) -> (forecasts, observations)` — lists of
    `(name, grid)` tuples: `[("parent", ...), ("nest", ...)]` and
    `[("MRMS", ...)]` + `("Stage IV", ...)` only when available (prints
    `"  Stage IV unavailable — scoring MRMS only."` when not).
  - `stage4_caveat(fields) -> str` — the figure-footer caveat text.
  - `compute_ets(case, fields=None)` — unchanged behavior, but accepts
    precomputed fields.
- Everything `compare.py` and tests import (`score_pair`, `hafs_parent_total`, `stage4_on_fixed`, `regrid_2d_to_fixed`) stays where it is, signatures unchanged.

**Behavior change (intentional, per spec):** "no files matching" was a print-and-return; it becomes `raise RuntimeError(...)` from the builder. Loud beats silent no-op.

- [ ] **Step 1: Replace `compute_ets` with the builder + helpers + slim `compute_ets`**

In `analysis/ets_full.py`, replace the entire `compute_ets` function (currently lines 156–233) with:

```python
def build_verification_fields(case):
    """Build every field the verification products need, once per case.

    Returns a dict: max_fhour, grid_lat, grid_lon, nest_total, apcp_mode,
    parent_total, mrms_total, stage4_grid, s4_label, swath. stage4_grid is
    None when Stage IV is unavailable. Raises RuntimeError when no nest
    files match the case glob or no MRMS hour can be loaded.
    """
    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    if not file_pairs:
        raise RuntimeError(
            f"No files matching {case.storm_glob()} in {case.run_dir}")
    max_fhour = file_pairs[-1][0]
    print(f"Init {case.init_dt:%Y-%m-%d %HZ} | accumulation 0–{max_fhour}h")

    grid_lat, grid_lon = build_fixed_grid(case)
    print(f"Fixed grid: {grid_lat.shape[0]}x{grid_lat.shape[1]} "
          f"@ {case.grid_res}deg")

    print("\nHAFS nest total ...")
    nest_total, apcp_mode = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  nest APCP mode: {apcp_mode}, max {np.nanmax(nest_total):.0f} mm")

    print("HAFS parent total ...")
    parent_total = hafs_parent_total(case, grid_lat, grid_lon)

    print("MRMS total ...")
    mrms_total = build_mrms_total(case, max_fhour, grid_lat, grid_lon)

    print("Stage IV total ...")
    stage4_grid, s4_label = stage4_on_fixed(case, max_fhour, grid_lat,
                                            grid_lon)

    print("TC verification swath ...")
    swath = tc_swath_mask(case, max_fhour, grid_lat, grid_lon)

    return dict(max_fhour=max_fhour, grid_lat=grid_lat, grid_lon=grid_lon,
                nest_total=nest_total, apcp_mode=apcp_mode,
                parent_total=parent_total, mrms_total=mrms_total,
                stage4_grid=stage4_grid, s4_label=s4_label, swath=swath)


def field_pairs(fields):
    """(forecasts, observations) as (name, grid) lists from a fields dict.

    Stage IV joins the observations only when it was available.
    """
    forecasts = [("parent", fields["parent_total"]),
                 ("nest", fields["nest_total"])]
    observations = [("MRMS", fields["mrms_total"])]
    if fields["stage4_grid"] is not None:
        observations.append(("Stage IV", fields["stage4_grid"]))
    else:
        print("  Stage IV unavailable — scoring MRMS only.")
    return forecasts, observations


def stage4_caveat(fields):
    """Figure-footer caveat describing the Stage IV accumulation window."""
    if fields["stage4_grid"] is None:
        return "Stage IV unavailable — not scored."
    return (f"Stage IV: CONUS-only, 24h 12Z–12Z files summed over touched "
            f"days ({fields['s4_label']}) — window approximates the "
            f"0–{fields['max_fhour']}h forecast accumulation.")


def compute_ets(case, fields=None):
    if fields is None:
        fields = build_verification_fields(case)
    max_fhour = fields["max_fhour"]
    swath = fields["swath"]
    forecasts, observations = field_pairs(fields)

    results = []
    print("\n" + "=" * 84)
    for fname, fgrid in forecasts:
        for oname, ogrid in observations:
            rows, n_valid = score_pair(fgrid, ogrid, swath,
                                       case.thresholds_mm, contingency_scores)
            results.append(dict(forecast=fname, observation=oname,
                                rows=rows, n_valid=n_valid))
            print(f"\n{fname} vs {oname}  (n_valid={n_valid:,})")
            print(f"{'thr':>5} {'a':>7} {'b':>7} {'c':>7} {'d':>7} {'ETS':>7} "
                  f"{'bias':>6} {'POD':>6} {'FAR':>6} {'CSI':>6}")
            for r in rows:
                print(f"{r['threshold']:>5} {r['a']:>7} {r['b']:>7} {r['c']:>7} "
                      f"{r['d']:>7} {r['ets']:>7.3f} {r['bias']:>6.2f} {r['pod']:>6.2f} "
                      f"{r['far']:>6.2f} {r['csi']:>6.2f}")
    print("=" * 84)

    case.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = case.out_dir / f"ets_full_{case.output_slug}.csv"
    out_png = case.out_dir / f"ets_full_{case.output_slug}.png"

    fieldnames = ["forecast", "observation", "threshold", "a", "b", "c", "d",
                  "ets", "bias", "pod", "far", "csi", "hss"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            for r in res["rows"]:
                w.writerow({"forecast": res["forecast"],
                            "observation": res["observation"], **r})
    print(f"\nSaved table: {out_csv}")

    caveat = stage4_caveat(fields)
    print(caveat)
    plot_curves(case, results, max_fhour, out_png, caveat=caveat)
    print(f"Saved plot : {out_png}")
```

Notes for the implementer:
- `build_fixed_grid` (defined earlier in the file) stays and is used by the builder.
- The old ad-hoc caveat construction and the `forecasts`/`observations` assembly inside `compute_ets` are gone — they now live in `stage4_caveat` / `field_pairs`.
- The `if __name__ == "__main__":` block at the bottom of the file is unchanged.

- [ ] **Step 2: Verify the module still imports and existing tests pass**

Run: `cd "/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS" && python3 analysis/tests/test_ets_full.py && python3 analysis/tests/test_compare.py`
Expected: all PASS, no import errors. (`compare.py` imports `score_pair`, `hafs_parent_total`, `stage4_on_fixed` from `ets_full` — untouched.)

- [ ] **Step 3: Syntax/immediate-smoke check of the new functions**

Run: `cd "/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS" && python3 -c "
import sys; sys.path.insert(0, 'analysis')
from ets_full import build_verification_fields, field_pairs, stage4_caveat, compute_ets
import numpy as np
f = dict(max_fhour=126, grid_lat=None, grid_lon=None, nest_total='N',
         apcp_mode='incremental', parent_total='P', mrms_total='M',
         stage4_grid=None, s4_label='unavailable', swath=None)
fc, ob = field_pairs(f)
assert [n for n, _ in fc] == ['parent', 'nest']
assert [n for n, _ in ob] == ['MRMS']
assert stage4_caveat(f) == 'Stage IV unavailable — not scored.'
f['stage4_grid'] = 'S4'; f['s4_label'] = '3 files'
fc, ob = field_pairs(f)
assert [n for n, _ in ob] == ['MRMS', 'Stage IV']
assert '0–126h' in stage4_caveat(f)
print('OK')
"`
Expected: `OK`

---

### Task 3: New product `analysis/rmse_scatter.py`

**Files:**
- Create: `analysis/rmse_scatter.py`
- Test: `analysis/tests/test_rmse_scatter.py` (create)

**Interfaces:**
- Consumes: `build_verification_fields`, `field_pairs`, `stage4_caveat` from `ets_full` (Task 2); `continuous_scores` from `skill_metrics` (Task 1); `from_yaml` from `hafs_case`.
- Produces: `compute_rmse(case, fields=None)` — Task 4's `run.py` imports it as `from rmse_scatter import compute_rmse`. Writes `rmse_scatter_<case.output_slug>.png` and `rmse_<case.output_slug>.csv` into `case.out_dir`.

- [ ] **Step 1: Write the failing tests**

Create `analysis/tests/test_rmse_scatter.py`:

```python
"""Local unit tests for rmse_scatter (no Hercules data needed).

Run directly:   python3 analysis/tests/test_rmse_scatter.py
Or via pytest:  pytest analysis/tests/test_rmse_scatter.py -v
"""
import csv
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmse_scatter import valid_points, compute_rmse
from hafs_case import StormCase


def test_valid_points_mirrors_score_pair_selection():
    # Same fixture as test_ets_full.test_score_pair_counts_only_valid_points:
    # swath excludes the last column; obs has one NaN inside the swath.
    fcst = np.array([[10.0, 0.0, 0.0],
                     [10.0, 10.0, 0.0],
                     [0.0, 0.0, 0.0]])
    obs = np.array([[10.0, 0.0, 99.0],
                    [0.0, np.nan, 99.0],
                    [0.0, 0.0, 99.0]])
    swath = np.array([[True, True, False],
                      [True, True, False],
                      [True, True, False]], dtype=bool)
    f, o = valid_points(fcst, obs, swath)
    assert f.size == o.size == 5
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(o))
    # The NaN-obs point (1,1) is dropped, so fcst keeps one 10 from col 0
    # of row 1 but not the 10 at (1,1).
    assert sorted(f.tolist()) == [0.0, 0.0, 0.0, 10.0, 10.0]


def _tiny_case(out_dir):
    return StormCase(
        run_dir=Path("."), init_dt=datetime(2024, 9, 24, 0),
        storm_name="Testorm", model_label="HAFS-A",
        domain=(0.0, 1.0, 0.0, 1.0), grid_res=0.5,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1], out_dir=Path(out_dir),
        mrms_cache_dir=Path("/tmp"), stage4_cache_dir=Path("/tmp"),
        fhours_filter=None,
        track=[(datetime(2024, 9, 24, 0), 0.5, 0.5)],
        case_slug="testcase", init_str="2024092400",
    )


def _tiny_fields():
    lat, lon = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    obs = np.full((4, 4), 10.0)
    return dict(
        max_fhour=6, grid_lat=lat, grid_lon=lon,
        nest_total=obs + 2.0,          # bias +2, rmse 2 vs mrms
        apcp_mode="incremental",
        parent_total=obs - 1.0,        # bias -1, rmse 1 vs mrms
        mrms_total=obs,
        stage4_grid=None, s4_label="unavailable",
        swath=np.ones((4, 4), dtype=bool),
    )


def test_compute_rmse_writes_csv_and_png():
    with tempfile.TemporaryDirectory() as tmp:
        case = _tiny_case(tmp)
        compute_rmse(case, fields=_tiny_fields())
        csv_path = Path(tmp) / "rmse_testcase_2024092400.csv"
        png_path = Path(tmp) / "rmse_scatter_testcase_2024092400.png"
        assert csv_path.exists(), "CSV not written"
        assert png_path.exists(), "PNG not written"
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        # Stage IV unavailable -> 2 forecasts x 1 observation = 2 rows.
        assert len(rows) == 2
        by_fcst = {r["forecast"]: r for r in rows}
        assert set(by_fcst) == {"parent", "nest"}
        assert all(r["observation"] == "MRMS" for r in rows)
        # Constant offsets: rmse == |bias|, mae == |bias|.
        assert abs(float(by_fcst["parent"]["rmse"]) - 1.0) < 1e-9
        assert abs(float(by_fcst["parent"]["bias"]) - (-1.0)) < 1e-9
        assert abs(float(by_fcst["nest"]["rmse"]) - 2.0) < 1e-9
        assert abs(float(by_fcst["nest"]["bias"]) - 2.0) < 1e-9
        assert int(by_fcst["parent"]["n"]) == 16


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS" && python3 analysis/tests/test_rmse_scatter.py`
Expected: `ModuleNotFoundError: No module named 'rmse_scatter'`

- [ ] **Step 3: Implement `analysis/rmse_scatter.py`**

Create `analysis/rmse_scatter.py`:

```python
"""
Storm-total continuous verification for HAFS QPF (parent domain + 2-km
nest) against MRMS QPE and NCEP Stage IV QPE over the TC rainfall swath.

Computes RMSE / MAE / bias / Pearson r on the event-total rainfall over
the same valid-point footprint the ETS uses, and renders one figure of
forecast-vs-observed hexbin scatter panels (rows = parent/nest, cols =
MRMS/Stage IV) with a 1:1 line and the scores annotated per panel.

Usage (on Hercules):
    python analysis/run.py storms/<case>.yaml rmse
"""

import sys
import csv
from pathlib import Path

# Make sibling analysis modules importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ets_full import build_verification_fields, field_pairs, stage4_caveat
from skill_metrics import continuous_scores
from hafs_case import from_yaml


def valid_points(fcst_grid, obs_grid, swath):
    """1-D (fcst, obs) arrays over the swath's valid points, zero-filled.

    Mirrors ets_full.score_pair's selection (swath & finite obs & finite
    fcst) so the continuous scores describe the identical footprint as
    the categorical ones.
    """
    valid = swath & np.isfinite(obs_grid) & np.isfinite(fcst_grid)
    fcst = np.nan_to_num(fcst_grid[valid], nan=0.0)
    obs = np.nan_to_num(obs_grid[valid], nan=0.0)
    return fcst, obs


def plot_scatter(case, results, max_fhour, out_path, caveat=""):
    """results: list of dicts {forecast, observation, fcst, obs, scores}.

    One hexbin panel per pair; panel grid is forecasts x observations in
    the order the results were computed.
    """
    fcst_names = list(dict.fromkeys(r["forecast"] for r in results))
    obs_names = list(dict.fromkeys(r["observation"] for r in results))
    nrows, ncols = len(fcst_names), len(obs_names)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.6 * ncols, 4.8 * nrows),
                             squeeze=False)

    # One shared, equal axis range (mm) across all panels.
    lim = 1.0
    for r in results:
        if r["fcst"].size:
            lim = max(lim, float(r["fcst"].max()), float(r["obs"].max()))
    lim = float(np.ceil(lim / 25.0) * 25.0)

    by_pair = {(r["forecast"], r["observation"]): r for r in results}
    for i, fname in enumerate(fcst_names):
        for j, oname in enumerate(obs_names):
            ax = axes[i][j]
            res = by_pair[(fname, oname)]
            s = res["scores"]
            if s["n"] == 0:
                ax.text(0.5, 0.5, "no valid points", ha="center",
                        va="center", transform=ax.transAxes, color="#777")
            else:
                hb = ax.hexbin(res["obs"], res["fcst"], gridsize=60,
                               extent=(0, lim, 0, lim), norm=LogNorm(),
                               cmap="viridis", mincnt=1)
                fig.colorbar(hb, ax=ax, label="grid points")
                box = (f"RMSE {s['rmse']:.1f} mm\n"
                       f"MAE  {s['mae']:.1f} mm\n"
                       f"bias {s['bias']:+.1f} mm\n"
                       f"r    {s['r']:.2f}\n"
                       f"n    {s['n']:,}")
                ax.text(0.03, 0.97, box, transform=ax.transAxes, va="top",
                        fontsize=8.5, family="monospace",
                        bbox=dict(boxstyle="round", fc="white",
                                  ec="#999", alpha=0.85))
            ax.plot([0, lim], [0, lim], color="gray", ls=":", lw=1)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{oname} observed (mm)")
            ax.set_ylabel(f"{fname} forecast (mm)")
            ax.set_title(f"{fname} vs {oname}", fontsize=10)

    fig.suptitle(
        f"{case.storm_name} — {case.model_label} storm-total QPF vs observed\n"
        f"0–{max_fhour}h | init {case.init_dt:%Y-%m-%d %HZ} | "
        f"TC swath ≤{case.mask_radius_km:.0f} km", fontsize=12)
    if caveat:
        fig.text(0.5, -0.01, caveat, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_rmse(case, fields=None):
    if fields is None:
        fields = build_verification_fields(case)
    max_fhour = fields["max_fhour"]
    swath = fields["swath"]
    forecasts, observations = field_pairs(fields)

    results = []
    print("\n" + "=" * 64)
    print(f"{'forecast':>8} {'obs':>9} {'n':>9} {'RMSE':>8} {'MAE':>8} "
          f"{'bias':>8} {'r':>6}")
    for fname, fgrid in forecasts:
        for oname, ogrid in observations:
            fcst, obs = valid_points(fgrid, ogrid, swath)
            s = continuous_scores(fcst, obs)
            results.append(dict(forecast=fname, observation=oname,
                                fcst=fcst, obs=obs, scores=s))
            print(f"{fname:>8} {oname:>9} {s['n']:>9,} {s['rmse']:>8.2f} "
                  f"{s['mae']:>8.2f} {s['bias']:>+8.2f} {s['r']:>6.2f}")
    print("=" * 64)

    case.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = case.out_dir / f"rmse_{case.output_slug}.csv"
    out_png = case.out_dir / f"rmse_scatter_{case.output_slug}.png"

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["forecast", "observation", "n",
                                           "rmse", "mae", "bias", "r"])
        w.writeheader()
        for res in results:
            w.writerow({"forecast": res["forecast"],
                        "observation": res["observation"], **res["scores"]})
    print(f"\nSaved table: {out_csv}")

    caveat = stage4_caveat(fields)
    print(caveat)
    plot_scatter(case, results, max_fhour, out_png, caveat=caveat)
    print(f"Saved plot : {out_png}")


if __name__ == "__main__":
    compute_rmse(from_yaml(sys.argv[1]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS" && python3 analysis/tests/test_rmse_scatter.py`
Expected: `PASS test_compute_rmse_writes_csv_and_png`, `PASS test_valid_points_mirrors_score_pair_selection`, `2 passed`

---

### Task 4: `rmse` command in `run.py` + shared fields in `all`

**Files:**
- Modify: `analysis/run.py` (docstring, `COMMANDS`, `parse_args` usage line, `dispatch`)
- Test: `analysis/tests/test_run.py` (add one test)

**Interfaces:**
- Consumes: `compute_rmse` (Task 3), `build_verification_fields` + `compute_ets(case, fields=None)` (Task 2).
- Produces: CLI `python analysis/run.py <case.yaml> [parent|ets|rmse|all|compare|replot]`.

- [ ] **Step 1: Write the failing test**

Append to `analysis/tests/test_run.py`, before `_run_all`:

```python
def test_parse_args_accepts_rmse():
    yaml_path, command = run.parse_args(["case.yaml", "rmse"])
    assert command == "rmse"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS" && python3 analysis/tests/test_run.py`
Expected: `✗ test_parse_args_accepts_rmse` with SystemExit (unknown command), then the raise.

- [ ] **Step 3: Implement**

In `analysis/run.py`:

1. Docstring — replace the usage line and the product list:

```python
"""Single entry point for the HAFS QPF/ETS framework.

    python analysis/run.py <case.yaml> [parent|ets|rmse|all|compare|replot]

Loads a StormCase from the YAML case file and runs the requested product(s):
  parent  the parent-domain QPF vs MRMS vs Stage IV 3-panel figure
  ets     the combined parent+nest ETS-vs-threshold figure + CSV
  rmse    storm-total RMSE/MAE/bias/r scatter panels + CSV
  all     parent + ets + rmse (fields built once; default)
  compare HFSA-vs-HFSB rainfall comparison (takes a comparison YAML)
  replot  redraw the comparison figures from existing CSVs (no recompute)
"""
```

2. Commands tuple:

```python
COMMANDS = ("parent", "ets", "rmse", "all", "compare", "replot")
```

3. The `parse_args` usage message:

```python
        print("usage: run.py <case.yaml> [parent|ets|rmse|all|compare|replot]")
```

4. Replace `dispatch`:

```python
def dispatch(case, command):
    """Run the requested product(s) for a loaded StormCase."""
    from parent_qpf import generate_parent_figure
    from ets_full import compute_ets, build_verification_fields
    from rmse_scatter import compute_rmse
    if command in ("parent", "all"):
        generate_parent_figure(case)
    if command == "ets":
        compute_ets(case)
    if command == "rmse":
        compute_rmse(case)
    if command == "all":
        # Build the expensive verification fields once, share across products.
        fields = build_verification_fields(case)
        compute_ets(case, fields=fields)
        compute_rmse(case, fields=fields)
```

- [ ] **Step 4: Run the full local suite**

Run: `cd "/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS" && for f in analysis/tests/test_*.py; do echo "== $f"; python3 "$f" || break; done`
Expected: every file ends with `N passed` / all `✓`, no failures.

---

### Task 5: Documentation (README + CLAUDE.md)

**Files:**
- Modify: `README.md` (sections 3 "Running" and its Outputs list; add a short "How to read the RMSE scatter" note after section 6)
- Modify: `CLAUDE.md` (run-modes line)

**Interfaces:** none (docs only).

- [ ] **Step 1: README running section**

In `README.md` section 3, extend the command list:

```bash
python analysis/run.py storms/<case>.yaml all      # parent figure + ETS + RMSE (default)
python analysis/run.py storms/<case>.yaml parent   # 3-panel QPF figure only
python analysis/run.py storms/<case>.yaml ets      # ETS plot + CSV only
python analysis/run.py storms/<case>.yaml rmse     # RMSE scatter + CSV only
```

And extend the Outputs block:

```
parent_qpf_<case>.png     # HAFS parent vs MRMS vs Stage IV, 3-panel
ets_full_<case>.png       # ETS vs threshold: parent/nest × MRMS/Stage IV
ets_full_<case>.csv       # the same scores as a table (a/b/c/d, ETS, bias, POD, FAR, CSI)
rmse_scatter_<case>.png   # forecast-vs-observed hexbin panels: parent/nest × MRMS/Stage IV
rmse_<case>.csv           # storm-total continuous scores (n, RMSE, MAE, bias, r)
```

- [ ] **Step 2: README metrics note**

After section 6 ("How to read the ETS plot"), add:

```markdown
## 7. How to read the RMSE scatter

Each `rmse_scatter_<case>.png` panel plots every swath grid point's
storm-total rainfall: observed (x) vs forecast (y), with a dotted 1:1
line. Points above the line are over-forecasts. The annotation box gives:

- **RMSE** — root-mean-square error of the totals (mm); penalizes big misses.
- **MAE** — mean absolute error (mm).
- **bias** — mean(forecast − observed): **positive = over-forecast**.
- **r** — Pearson correlation of the point totals.
- **n** — valid swath points scored (same footprint as the ETS).

RMSE/MAE/bias are continuous companions to the categorical ETS curves:
ETS asks "did we put rain ≥ X in the right places", the scatter asks
"how far off were the amounts".
```

- [ ] **Step 3: CLAUDE.md run-modes line**

In `CLAUDE.md`, change:

```
- Run analyses via `python analysis/run.py storms/<case>.yaml [parent|ets|all]`
```

to:

```
- Run analyses via `python analysis/run.py storms/<case>.yaml [parent|ets|rmse|all]`
```

(If the existing line's wording differs slightly, keep its wording and just add `rmse` to the bracket list.)

- [ ] **Step 4: Verify docs render / final suite**

Run: `cd "/Users/suchitbasineni/Documents/GitHub/HAFS&WoFS" && for f in analysis/tests/test_*.py; do echo "== $f"; python3 "$f" || break; done`
Expected: all pass. Report the working tree ready for Suchit's review (no commits).

---

## Verification on Hercules (manual, by Suchit)

Not a plan task — the real-data smoke test once the code is synced to the HPC:

```bash
conda activate hafs
python analysis/run.py storms/helene_hfsa.yaml rmse   # RMSE only
python analysis/run.py storms/helene_hfsa.yaml all    # confirm fields built once (one "HAFS nest total ..." pass)
```

Expect `rmse_scatter_helene_hfsa_2024092400.png` + `rmse_helene_hfsa_2024092400.csv` in `analysis/output/helene_hfsa/`, and `all` printing the nest/parent/MRMS build sequence a single time.
