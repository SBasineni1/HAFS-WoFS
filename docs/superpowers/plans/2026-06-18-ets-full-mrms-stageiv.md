# ETS Full (HAFS QPF vs MRMS & Stage IV) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `analysis/ets_full.py` that computes ETS (+ bias/POD/FAR/CSI) for two HAFS-A forecast fields (fixed parent domain, moving 2-km nest) against two observed QPE references (MRMS, NCEP Stage IV) over Helene's TC swath, producing one combined ETS-vs-threshold figure and one combined CSV.

**Architecture:** A new standalone script reuses all GRIB2/MRMS/Stage IV plumbing from `qpf_full_run.py` and `parent_qpf.py` — nothing is duplicated. Two new *pure* helpers are added and unit-tested locally with synthetic arrays: a general 2-D curvilinear→fixed-grid regridder (`regrid_2d_to_fixed`, used for both the parent field and Stage IV) and a forecast/observation scoring helper (`score_pair`). The existing `ets_score.py` is left untouched; `ets_full.py` imports `contingency_scores`, `regrid_mrms_to_fixed`, `build_mrms_total`, and `tc_swath_mask` from it.

**Tech Stack:** Python 3, numpy, scipy (`griddata`, `RegularGridInterpolator`), matplotlib (Agg), cfgrib, boto3 — all already used in the repo.

## Global Constraints

- Scripts live in `analysis/`; run from repo root as `python analysis/ets_full.py`. (CLAUDE.md)
- Never commit large data files (`.nc`, `.grb2`, `.grb`, etc.); plot output goes in `analysis/output/` (gitignored). (CLAUDE.md)
- Git commits: **no** `Co-Authored-By` lines. (CLAUDE.md)
- Verification radius for ETS is `TC_MASK_RADIUS_KM` (500 km, from `qpf_full_run`) — NOT `parent_qpf.MASK_RADIUS_KM` (750 km).
- Local dev env has scipy/numpy/matplotlib but **no pytest** — tests must run via `python3 <file>` directly (plain `assert` + `__main__` runner), and also pass under pytest if present.
- Full GRIB/MRMS/Stage IV data is only on Hercules HPC; the data-driven `main()` is verified by a manual Hercules run, not a local unit test.
- ETS math is `ETS = (a - a_ref)/(a + b + c - a_ref)`, `a_ref = (a+b)(a+c)/n`; reuse `ets_score.contingency_scores` rather than reimplementing.

---

### Task 1: Pure helper — `regrid_2d_to_fixed` (TDD)

Add a general regridder that maps a source field on a curvilinear (2-D) or
rectilinear (1-D) lat/lon grid onto the fixed verification mesh, using
`scipy.interpolate.griddata` (linear, NaN outside the source hull). Used for
both the HAFS parent field and Stage IV.

**Files:**
- Create: `analysis/ets_full.py`
- Test: `analysis/tests/test_ets_full.py`

**Interfaces:**
- Consumes: nothing (pure numpy/scipy).
- Produces: `regrid_2d_to_fixed(src_lat, src_lon, data, grid_lat, grid_lon) -> np.ndarray`
  - `src_lat`, `src_lon`: 1-D (axes) or 2-D arrays of source coordinates; if 1-D they are meshgridded internally.
  - `data`: 2-D array matching the meshgridded source shape.
  - `grid_lat`, `grid_lon`: 2-D fixed-grid meshes (from `np.meshgrid`).
  - Returns: 2-D array shaped like `grid_lat`, NaN where outside the source coverage.

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/test_ets_full.py`:

```python
"""Local unit tests for ets_full pure helpers (no Hercules data needed).

Run directly:   python3 analysis/tests/test_ets_full.py
Or via pytest:  pytest analysis/tests/test_ets_full.py -v
"""
import sys
from pathlib import Path

import numpy as np

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ets_full import regrid_2d_to_fixed, score_pair


def test_regrid_2d_identity_on_matching_grid():
    # Source is a linear ramp z = lat + lon on a fine 2-D grid.
    s_lat = np.linspace(20.0, 40.0, 41)
    s_lon = np.linspace(-100.0, -70.0, 61)
    slon2d, slat2d = np.meshgrid(s_lon, s_lat)
    data = slat2d + slon2d
    # Fixed grid is a coarser mesh strictly inside the source extent.
    g_lon = np.linspace(-95.0, -75.0, 11)
    g_lat = np.linspace(25.0, 35.0, 9)
    grid_lon, grid_lat = np.meshgrid(g_lon, g_lat)

    out = regrid_2d_to_fixed(slat2d, slon2d, data, grid_lat, grid_lon)

    assert out.shape == grid_lat.shape
    expected = grid_lat + grid_lon
    assert np.allclose(out, expected, atol=1e-6)


def test_regrid_2d_accepts_1d_axes():
    # Passing 1-D lat/lon axes must work the same as 2-D meshes.
    s_lat = np.linspace(20.0, 40.0, 41)
    s_lon = np.linspace(-100.0, -70.0, 61)
    slon2d, slat2d = np.meshgrid(s_lon, s_lat)
    data = slat2d + slon2d
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(-95.0, -75.0, 11), np.linspace(25.0, 35.0, 9)
    )

    out = regrid_2d_to_fixed(s_lat, s_lon, data, grid_lat, grid_lon)
    assert np.allclose(out, grid_lat + grid_lon, atol=1e-6)


def test_regrid_2d_nan_outside_source():
    # Points outside the source hull come back NaN, not extrapolated.
    s_lat = np.linspace(30.0, 35.0, 11)
    s_lon = np.linspace(-90.0, -85.0, 11)
    slon2d, slat2d = np.meshgrid(s_lon, s_lat)
    data = np.ones_like(slat2d)
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(-95.0, -80.0, 7), np.linspace(25.0, 40.0, 7)
    )

    out = regrid_2d_to_fixed(slat2d, slon2d, data, grid_lat, grid_lon)
    # Corner (25, -95) is well outside [30,35]x[-90,-85] -> NaN.
    assert np.isnan(out[0, 0])
    # Center (32.5, -87.5) is inside -> finite ~1.0.
    assert np.isfinite(out[3, 3])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_ets_full.py`
Expected: FAIL — `ImportError: cannot import name 'regrid_2d_to_fixed' from 'ets_full'` (module/function not yet defined).

- [ ] **Step 3: Write minimal implementation**

Create `analysis/ets_full.py` with the module docstring, imports, and the two
pure helpers (`score_pair` is filled in Task 2 but stubbed here so the import
succeeds):

```python
"""
ETS for HAFS-A QPF (parent domain + moving 2-km nest) verified against
MRMS QPE and NCEP Stage IV QPE over the Hurricane Helene rainfall swath.

Produces one combined ETS-vs-threshold figure (4 curves: parent/nest x
MRMS/StageIV) and one combined CSV. Reuses all GRIB2/MRMS/Stage IV plumbing
from qpf_full_run.py and parent_qpf.py, and the contingency math + MRMS
plumbing from ets_score.py. The existing ets_score.py is left untouched.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/ets_full.py
"""

import sys
import csv
from pathlib import Path

# Make sibling analysis modules importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def regrid_2d_to_fixed(src_lat, src_lon, data, grid_lat, grid_lon):
    """Interpolate a curvilinear/rectilinear source field onto the fixed mesh.

    src_lat/src_lon may be 1-D axes or 2-D meshes; data is shaped like the
    2-D source mesh. Uses linear griddata; points outside the source hull
    come back NaN (no extrapolation).
    """
    src_lat = np.asarray(src_lat, dtype=float)
    src_lon = np.asarray(src_lon, dtype=float)
    if src_lat.ndim == 1 and src_lon.ndim == 1:
        src_lon, src_lat = np.meshgrid(src_lon, src_lat)
    pts = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    vals = np.asarray(data, dtype=float).ravel()
    finite = np.isfinite(vals)
    out = griddata(
        pts[finite], vals[finite],
        (grid_lat, grid_lon), method="linear",
    )
    return out


def score_pair(fcst_grid, obs_grid, swath, thresholds, contingency_fn):
    """Placeholder — implemented in Task 2."""
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 analysis/tests/test_ets_full.py`
Expected: the regrid tests PASS. (`test_score_pair_*` from Task 2 don't exist yet, so only the three regrid tests run here.)

- [ ] **Step 5: Commit**

```bash
git add analysis/ets_full.py analysis/tests/test_ets_full.py
git commit -m "Add ets_full regrid_2d_to_fixed helper + tests"
```

---

### Task 2: Pure helper — `score_pair` (TDD)

Given a forecast grid, an observation grid, and the swath mask, restrict to
valid points (`swath & isfinite(obs) & isfinite(fcst)`), zero-fill the kept
values, and return one `contingency_scores` dict per threshold plus the valid
point count.

**Files:**
- Modify: `analysis/ets_full.py`
- Test: `analysis/tests/test_ets_full.py`

**Interfaces:**
- Consumes: `regrid_2d_to_fixed` (Task 1); `ets_score.contingency_scores` (existing) passed in as `contingency_fn` so the helper stays pure/testable.
- Produces: `score_pair(fcst_grid, obs_grid, swath, thresholds, contingency_fn) -> (rows, n_valid)`
  - `fcst_grid`, `obs_grid`, `swath`: 2-D arrays of identical shape (`swath` is boolean).
  - `thresholds`: iterable of floats (mm).
  - `contingency_fn(fcst_1d, obs_1d, threshold) -> dict` with keys `threshold,a,b,c,d,ets,bias,pod,far,csi`.
  - Returns: `(rows, n_valid)` where `rows` is a list of those dicts (one per threshold) and `n_valid` is `int` count of scored points.

- [ ] **Step 1: Write the failing test**

Append to `analysis/tests/test_ets_full.py` (above the `_run_all` block):

```python
def _toy_contingency(fcst, obs, threshold):
    fy = fcst >= threshold
    oy = obs >= threshold
    a = int(np.sum(fy & oy))
    b = int(np.sum(fy & ~oy))
    c = int(np.sum(~fy & oy))
    d = int(np.sum(~fy & ~oy))
    n = a + b + c + d
    a_ref = (a + b) * (a + c) / n if n else 0.0
    denom = (a + b + c) - a_ref
    ets = (a - a_ref) / denom if denom else float("nan")
    return dict(threshold=threshold, a=a, b=b, c=c, d=d, ets=ets,
                bias=float("nan"), pod=float("nan"),
                far=float("nan"), csi=float("nan"))


def test_score_pair_counts_only_valid_points():
    # 3x3 grids. swath excludes the last column; obs has one NaN inside swath.
    fcst = np.array([[10.0, 0.0, 0.0],
                     [10.0, 10.0, 0.0],
                     [0.0, 0.0, 0.0]])
    obs = np.array([[10.0, 0.0, 99.0],
                    [0.0, np.nan, 99.0],
                    [0.0, 0.0, 99.0]])
    swath = np.array([[True, True, False],
                      [True, True, False],
                      [True, True, False]], dtype=bool)
    # Valid = swath & isfinite(obs) & isfinite(fcst):
    #   row0: (T,T) -> 2 ; row1: (T, NaN->drop) -> 1 ; row2: (T,T) -> 2  => 5
    rows, n_valid = score_pair(fcst, obs, swath, [5.0], _toy_contingency)

    assert n_valid == 5
    r = rows[0]
    # At thr=5 over the 5 valid pts: fcst>=5 at (0,0) and (1,0); obs>=5 at (0,0).
    # hit a=1 (0,0); false alarm b=1 (1,0); miss c=0; correct-neg d=3.
    assert (r["a"], r["b"], r["c"], r["d"]) == (1, 1, 0, 3)


def test_score_pair_one_row_per_threshold():
    fcst = np.zeros((4, 4))
    obs = np.zeros((4, 4))
    swath = np.ones((4, 4), dtype=bool)
    rows, n_valid = score_pair(fcst, obs, swath, [1.0, 5.0, 10.0],
                               _toy_contingency)
    assert len(rows) == 3
    assert [r["threshold"] for r in rows] == [1.0, 5.0, 10.0]
    assert n_valid == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_ets_full.py`
Expected: FAIL — `NotImplementedError` raised from the `score_pair` stub.

- [ ] **Step 3: Write minimal implementation**

Replace the `score_pair` stub in `analysis/ets_full.py` with:

```python
def score_pair(fcst_grid, obs_grid, swath, thresholds, contingency_fn):
    """Score one forecast/observation pair over the swath's valid points.

    Valid points are swath & finite(obs) & finite(fcst); kept values are
    zero-filled before thresholding. Returns (rows, n_valid).
    """
    valid = swath & np.isfinite(obs_grid) & np.isfinite(fcst_grid)
    n_valid = int(np.sum(valid))
    fcst = np.nan_to_num(fcst_grid[valid], nan=0.0)
    obs = np.nan_to_num(obs_grid[valid], nan=0.0)
    rows = [contingency_fn(fcst, obs, thr) for thr in thresholds]
    return rows, n_valid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 analysis/tests/test_ets_full.py`
Expected: PASS — all five tests (3 regrid + 2 score_pair) report `PASS`, ending `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add analysis/ets_full.py analysis/tests/test_ets_full.py
git commit -m "Add ets_full score_pair helper + tests"
```

---

### Task 3: Field assembly — gather the 2 forecasts and 2 observations on the fixed grid

Wire the imported plumbing into functions that return each field on the fixed
verification mesh. No new math; this is orchestration of existing helpers.
Verified by the Hercules run in Task 5 (needs real data), so no local unit test.

**Files:**
- Modify: `analysis/ets_full.py`

**Interfaces:**
- Consumes (imports, all existing):
  - From `qpf_full_run`: `HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER, FIXED_DOMAIN, GRID_RES, TC_MASK_RADIUS_KM, INIT_DT, OUT_DIR, MRMS_CACHE_DIR, discover_files, hafs_event_total`.
  - From `ets_score`: `THRESHOLDS_MM, contingency_scores, build_mrms_total, tc_swath_mask`.
  - From `parent_qpf`: `default_parent_path, read_hafs_tp_records, pick_cumulative_record, stage4_total, STAGE4_CACHE_DIR`.
- Produces:
  - `build_fixed_grid() -> (grid_lat, grid_lon)` — 2-D meshes from `FIXED_DOMAIN`/`GRID_RES` (identical construction to `ets_score.main`).
  - `hafs_parent_total(grid_lat, grid_lon) -> np.ndarray` — parent cumulative APCP regridded onto the fixed mesh (NaN outside parent coverage), or raises if no parent file/records.
  - `stage4_on_fixed(max_fhour, grid_lat, grid_lon) -> (np.ndarray, str)` — Stage IV touched-days total regridded onto the fixed mesh (NaN outside CONUS), plus the `s4_label` window string; returns `(None, "unavailable")` if Stage IV download/read fails.

- [ ] **Step 1: Add the imports and `build_fixed_grid`**

Add below the existing imports in `analysis/ets_full.py`:

```python
from qpf_full_run import (
    HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER, FIXED_DOMAIN, GRID_RES,
    TC_MASK_RADIUS_KM, INIT_DT, OUT_DIR, MRMS_CACHE_DIR,
    discover_files, hafs_event_total,
)
from ets_score import (
    THRESHOLDS_MM, contingency_scores, build_mrms_total, tc_swath_mask,
)
from parent_qpf import (
    default_parent_path, read_hafs_tp_records, pick_cumulative_record,
    stage4_total, STAGE4_CACHE_DIR,
)

OUT_PNG = OUT_DIR.parent / "ets_full_helene.png"
OUT_CSV = OUT_DIR.parent / "ets_full_helene.csv"


def build_fixed_grid():
    """Fixed lat/lon verification mesh (same as ets_score.main)."""
    lat_min, lat_max, lon_min, lon_max = FIXED_DOMAIN
    fixed_lons = np.arange(lon_min, lon_max + GRID_RES, GRID_RES)
    fixed_lats = np.arange(lat_min, lat_max + GRID_RES, GRID_RES)
    grid_lon, grid_lat = np.meshgrid(fixed_lons, fixed_lats)
    return grid_lat, grid_lon
```

- [ ] **Step 2: Add `hafs_parent_total`**

```python
def hafs_parent_total(grid_lat, grid_lon):
    """HAFS-A parent cumulative APCP regridded onto the fixed verification mesh.

    Reuses parent_qpf's discovery + cumulative-record selection, then maps the
    parent grid onto the fixed mesh via regrid_2d_to_fixed.
    """
    path = default_parent_path()
    if path is None or not path.exists():
        raise RuntimeError("No parent.atm file found for the configured run.")
    records = read_hafs_tp_records(path)
    if not records:
        raise RuntimeError(f"No 'tp' (APCP) records in parent file {path}.")
    rec = pick_cumulative_record(records)
    print(f"  parent 0->{rec['end_step']}h, grid {rec['lats'].shape}, "
          f"max {np.nanmax(rec['data']):.0f} mm")
    return regrid_2d_to_fixed(rec["lats"], rec["lons"], rec["data"],
                              grid_lat, grid_lon)
```

- [ ] **Step 3: Add `stage4_on_fixed`**

```python
def stage4_on_fixed(max_fhour, grid_lat, grid_lon):
    """Stage IV touched-days total (parent_qpf.stage4_total) on the fixed mesh.

    stage4_total masks its output to parent_qpf's 750 km display swath; for
    verification we re-derive the field UNMASKED is not exposed, so we accept
    that mask — it is wider than the 500 km verification swath, so the tighter
    tc_swath_mask applied later still governs the scored footprint. Stage IV is
    CONUS-only, so ocean points regrid to NaN and drop out automatically.
    """
    s4_lat, s4_lon, s4_total, s4_label = stage4_total(
        INIT_DT, max_fhour, STAGE4_CACHE_DIR)
    if s4_total is None:
        return None, "unavailable"
    grid = regrid_2d_to_fixed(s4_lat, s4_lon, s4_total, grid_lat, grid_lon)
    return grid, s4_label
```

- [ ] **Step 4: Smoke-check the module imports cleanly**

Run: `python3 -c "import sys; sys.path.insert(0,'analysis'); import ets_full; print('ok')"`
Expected: prints `ok` (imports resolve; this does NOT touch Hercules data — the import of `qpf_full_run`/`parent_qpf` must succeed locally). If `qpf_full_run`/`parent_qpf` import-time code needs network/data, note it and defer this check to Hercules.

- [ ] **Step 5: Commit**

```bash
git add analysis/ets_full.py
git commit -m "Add ets_full field-assembly functions (parent + Stage IV on fixed grid)"
```

---

### Task 4: Driver `main()` + combined plot/CSV

Assemble all fields, build the 500 km swath, score the 4 forecast/observation
pairs, write the combined CSV, and render the 4-curve figure (color = obs,
linestyle = forecast). Data-driven, so verified on Hercules in Task 5.

**Files:**
- Modify: `analysis/ets_full.py`

**Interfaces:**
- Consumes: `build_fixed_grid, hafs_parent_total, stage4_on_fixed, score_pair` (this module); `discover_files, hafs_event_total, build_mrms_total, tc_swath_mask, contingency_scores, THRESHOLDS_MM, OUT_PNG, OUT_CSV` (imported/defined above).
- Produces: `main()` writing `OUT_CSV` and `OUT_PNG`; `plot_curves(results, max_fhour, out_path)`.

- [ ] **Step 1: Add `plot_curves`**

```python
# obs -> color, forecast -> linestyle/marker, so 4 curves stay legible.
_OBS_COLOR = {"MRMS": "#1f77b4", "Stage IV": "#2ca02c"}
_FCST_STYLE = {"parent": dict(ls="-", marker="o"),
               "nest": dict(ls="--", marker="s")}


def plot_curves(results, max_fhour, out_path, caveat=""):
    """results: list of dicts {forecast, observation, rows, n_valid}."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for res in results:
        rows = res["rows"]
        if not rows:
            continue
        thr = [r["threshold"] for r in rows]
        ets = [r["ets"] for r in rows]
        style = _FCST_STYLE.get(res["forecast"], dict(ls="-", marker="o"))
        ax.plot(thr, ets, color=_OBS_COLOR.get(res["observation"], "gray"),
                lw=2, **style,
                label=f"{res['forecast']} vs {res['observation']} "
                      f"(n={res['n_valid']:,})")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(THRESHOLDS_MM)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Rainfall threshold (mm)")
    ax.set_ylabel("Equitable Threat Score (ETS)")
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"Hurricane Helene — HAFS-A QPF ETS vs MRMS & Stage IV\n"
        f"0–{max_fhour}h | init {INIT_DT:%Y-%m-%d %HZ} | "
        f"TC swath ≤{TC_MASK_RADIUS_KM:.0f} km"
    )
    if caveat:
        fig.text(0.5, -0.02, caveat, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
```

- [ ] **Step 2: Add `main()`**

```python
def main():
    file_pairs = discover_files(HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER)
    if not file_pairs:
        print(f"No files matching {FILE_GLOB} in {HAFS_RUN_DIR}")
        return
    max_fhour = file_pairs[-1][0]
    print(f"Init {INIT_DT:%Y-%m-%d %HZ} | accumulation 0–{max_fhour}h")

    grid_lat, grid_lon = build_fixed_grid()
    print(f"Fixed grid: {grid_lat.shape[0]}x{grid_lat.shape[1]} @ {GRID_RES}deg")

    print("\nHAFS nest total ...")
    nest_total, apcp_mode = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  nest APCP mode: {apcp_mode}, max {np.nanmax(nest_total):.0f} mm")

    print("HAFS parent total ...")
    parent_total = hafs_parent_total(grid_lat, grid_lon)

    print("MRMS total ...")
    mrms_total = build_mrms_total(max_fhour, grid_lat, grid_lon)

    print("Stage IV total ...")
    stage4_grid, s4_label = stage4_on_fixed(max_fhour, grid_lat, grid_lon)

    print("TC verification swath ...")
    swath = tc_swath_mask(max_fhour, grid_lat, grid_lon)

    forecasts = [("parent", parent_total), ("nest", nest_total)]
    observations = [("MRMS", mrms_total)]
    if stage4_grid is not None:
        observations.append(("Stage IV", stage4_grid))
    else:
        print("  Stage IV unavailable — scoring MRMS only.")

    results = []
    print("\n" + "=" * 84)
    for fname, fgrid in forecasts:
        for oname, ogrid in observations:
            rows, n_valid = score_pair(fgrid, ogrid, swath,
                                       THRESHOLDS_MM, contingency_scores)
            results.append(dict(forecast=fname, observation=oname,
                                rows=rows, n_valid=n_valid))
            print(f"\n{fname} vs {oname}  (n_valid={n_valid:,})")
            print(f"{'thr':>5} {'a':>7} {'b':>7} {'c':>7} {'ETS':>7} "
                  f"{'bias':>6} {'POD':>6} {'FAR':>6} {'CSI':>6}")
            for r in rows:
                print(f"{r['threshold']:>5} {r['a']:>7} {r['b']:>7} {r['c']:>7} "
                      f"{r['ets']:>7.3f} {r['bias']:>6.2f} {r['pod']:>6.2f} "
                      f"{r['far']:>6.2f} {r['csi']:>6.2f}")
    print("=" * 84)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["forecast", "observation", "threshold", "a", "b", "c", "d",
                  "ets", "bias", "pod", "far", "csi"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            for r in res["rows"]:
                w.writerow({"forecast": res["forecast"],
                            "observation": res["observation"], **r})
    print(f"\nSaved table: {OUT_CSV}")

    caveat = (f"Stage IV: CONUS-only, 24h 12Z–12Z files summed over touched "
              f"days ({s4_label}) — window approximates the 0–{max_fhour}h "
              f"forecast accumulation.")
    plot_curves(results, max_fhour, OUT_PNG, caveat=caveat)
    print(f"Saved plot : {OUT_PNG}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-check the module still imports**

Run: `python3 -c "import sys; sys.path.insert(0,'analysis'); import ets_full; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Re-run the unit tests (regression — helpers untouched)**

Run: `python3 analysis/tests/test_ets_full.py`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add analysis/ets_full.py
git commit -m "Add ets_full main() driver + combined ETS plot/CSV"
```

---

### Task 5: Hercules integration verification

The only place the full GRIB/MRMS/Stage IV data exists. Run end-to-end and
sanity-check the outputs against the spec's verification criteria.

**Files:** none (run + observe).

**Interfaces:** none.

- [ ] **Step 1: Run end-to-end on Hercules**

```bash
module load miniconda3
conda activate hafs
python analysis/ets_full.py
```
Expected: prints per-pair contingency tables, writes `analysis/ets_full_helene.csv` and `analysis/ets_full_helene.png` (under `OUT_DIR.parent`).

- [ ] **Step 2: Regression check vs existing ets_score.py**

Run: `python analysis/ets_score.py`
Compare its nest-vs-MRMS ETS values to the `nest vs MRMS` rows in
`ets_full_helene.csv`.
Expected: identical numbers (same grid, same swath, same plumbing).

- [ ] **Step 3: Sanity checks (spec verification criteria)**

Confirm from stdout / the CSV:
- Stage IV `n_valid` is meaningfully **smaller** than MRMS `n_valid` (Gulf/ocean clipped to CONUS).
- All `ets` values lie in `[-1/3, 1]`; all `bias` values are ≥ 0.

Expected: all three hold. If the nest-vs-MRMS regression differs, stop and
diagnose grid/swath construction before trusting the new curves.

- [ ] **Step 4: Commit any fixes**

If Steps 2–3 surfaced bugs, fix in `analysis/ets_full.py`, re-run Step 1, then:

```bash
git add analysis/ets_full.py
git commit -m "Fix ets_full issues found in Hercules verification"
```

---

## Self-Review

**Spec coverage:**
- 2 forecasts × 2 obs → Tasks 3 (assembly) + 4 (4-pair scoring loop). ✓
- Parent field = parent_qpf cumulative record regridded → `hafs_parent_total`. ✓
- Nest field = `hafs_event_total` → Task 4. ✓
- MRMS via `build_mrms_total`/`regrid_mrms_to_fixed` → imported, Task 4. ✓
- Stage IV reuse `stage4_total` (touched-days) + 2-D regrid → `stage4_on_fixed`, Task 3. ✓
- 500 km verification swath (not 750) → `tc_swath_mask` import, Global Constraints. ✓
- Per-obs validity = swath & finite(obs) (Stage IV CONUS NaN auto-dropped) → `score_pair`, Task 2 + `regrid_2d_to_fixed` NaN fill, Task 1. ✓
- Combined figure (color=obs, linestyle=forecast) + combined CSV → Task 4. ✓
- `contingency_scores` reused, not reimplemented → imported, Task 4. ✓
- Stage IV caveat printed + figure footnote → Task 4 `main()`/`plot_curves`. ✓
- `ets_score.py` untouched → only imported from. ✓
- Tests runnable without pytest → `__main__` runner, Global Constraints. ✓

**Note (deviation from spec):** the spec said to regrid the **unmasked** Stage IV total, but `parent_qpf.stage4_total` only returns the field already masked to its 750 km display swath. `stage4_on_fixed` documents that the wider 750 km Stage IV mask is harmless because the tighter 500 km `tc_swath_mask` governs the scored footprint, and genuine zero-rain CONUS points inside the swath are preserved. This keeps us DRY (no fork of `stage4_total`) at no cost to the score. If exact unmasked behavior is later required, refactor `stage4_total` to take an optional `mask_radius_km=None`.

**Placeholder scan:** none — every code step contains complete code; `score_pair` stub in Task 1 is intentional and replaced in Task 2.

**Type consistency:** `regrid_2d_to_fixed`, `score_pair`, `contingency_scores` (dict keys `threshold,a,b,c,d,ets,bias,pod,far,csi`), and the `results` dict shape (`forecast, observation, rows, n_valid`) are used consistently across Tasks 1–4. ✓
