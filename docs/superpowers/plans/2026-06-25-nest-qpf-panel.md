# Nest QPF Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a moving-nest precipitation panel to the QPF comparison figure so each single-storm case produces a 4-panel map: Nest · Parent · MRMS · Stage IV.

**Architecture:** Extend `analysis/parent_qpf.py` only. Extract the existing inline swath-mask logic into a shared `swath_masked()` helper (used by both the parent and the new nest panel). Add a `compute_nest_field()` helper that discovers the `storm.atm` files, accumulates the moving-nest running-max via `hafs_event_total` on the case fixed grid, and masks it to the display swath — returning `None` when no nest files exist so the figure degrades gracefully. Prepend the nest panel to the existing `plot_compare` panel list; no plotting-layer changes needed.

**Tech Stack:** Python 3.11, numpy, xarray/cfgrib, matplotlib (Agg), cartopy, boto3. Tests via pytest (also runnable as plain scripts), no Hercules data required.

## Global Constraints

- Single-storm product only — produced by the `parent`/`all` commands; do NOT touch `ets_full.py`, `compare.py`, `run.py`, or the YAML/`StormCase` layer.
- Output filename unchanged: `parent_qpf_<case.output_slug>.png`.
- Do NOT import `build_fixed_grid` from `ets_full` (it imports `parent_qpf` — circular). Use `case.fixed_grid()`, which returns `(grid_lat, grid_lon)` 2-D meshes and is what `build_fixed_grid` wraps.
- Reuse the existing `QPF_LEVELS` / `qpf_cmap()` color scale so all four panels share one colorbar.
- Nest field = `hafs_event_total(...)` (moving-nest running-max) — the same field `ets_full.py` scores. Panel title must carry the inflation caveat.
- Never name anything "multistorm".
- No large data files committed.

---

### Task 1: Extract `swath_masked()` helper and reuse it for the parent panel

Pull the parent panel's inline swath-mask loop into a reusable pure function, prove it with unit tests, then refactor the parent panel to call it. This is a behavior-preserving refactor that creates the helper the nest panel needs.

**Files:**
- Create: `analysis/tests/test_parent_qpf.py`
- Modify: `analysis/parent_qpf.py` (add helper near the plotting section ~line 226; refactor parent swath block at lines 319–323)

**Interfaces:**
- Consumes: `haversine_km` (already imported in `parent_qpf.py`), `numpy as np`, `timedelta` (already imported).
- Produces: `swath_masked(field, lats, lons, case, end_fhour) -> np.ndarray` — builds a boolean mask of all grid cells within `case.display_radius_km` of the best-track position at any hour `0..end_fhour`, then returns `np.where(mask, np.nan_to_num(field, nan=0.0), 0.0)`. `lats`/`lons` are 2-D arrays of identical shape; `field` matches that shape. `case` must expose `.init_dt`, `.display_radius_km`, and `.position_at(dt) -> (lat, lon)`.

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/test_parent_qpf.py`:

```python
"""Local unit tests for parent_qpf pure helpers (no Hercules data needed).

Run directly:   python3 analysis/tests/test_parent_qpf.py
Or via pytest:  pytest analysis/tests/test_parent_qpf.py -v
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import parent_qpf
from parent_qpf import swath_masked


class FakeCase:
    """Minimal stand-in for StormCase: a single stationary track point."""
    def __init__(self, lat=30.0, lon=-85.0, radius_km=300.0):
        self.init_dt = datetime(2024, 9, 24, 0)
        self.display_radius_km = radius_km
        self._lat, self._lon = lat, lon

    def position_at(self, _dt):
        return self._lat, self._lon


def _grid():
    lons = np.linspace(-90.0, -80.0, 11)   # ~1 deg spacing
    lats = np.linspace(25.0, 35.0, 11)
    return np.meshgrid(lons, lats)[::-1]    # (lat2d, lon2d)


def test_swath_masked_zeros_outside_radius():
    lat2d, lon2d = _grid()
    field = np.full(lat2d.shape, 50.0)
    case = FakeCase(lat=30.0, lon=-85.0, radius_km=200.0)
    out = swath_masked(field, lat2d, lon2d, case, end_fhour=0)
    # Cell at the track center keeps its value.
    ci = np.argmin(np.abs(lat2d[:, 0] - 30.0))
    cj = np.argmin(np.abs(lon2d[0, :] - (-85.0)))
    assert out[ci, cj] == 50.0
    # A far corner (>200 km away) is zeroed.
    assert out[0, 0] == 0.0
    # Output shape is preserved.
    assert out.shape == field.shape


def test_swath_masked_replaces_nan_with_zero_inside_swath():
    lat2d, lon2d = _grid()
    field = np.full(lat2d.shape, np.nan)
    case = FakeCase(lat=30.0, lon=-85.0, radius_km=2000.0)  # covers whole grid
    out = swath_masked(field, lat2d, lon2d, case, end_fhour=0)
    assert np.all(out == 0.0)          # NaN -> 0 inside the swath
    assert not np.any(np.isnan(out))


if __name__ == "__main__":
    test_swath_masked_zeros_outside_radius()
    test_swath_masked_replaces_nan_with_zero_inside_swath()
    print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest analysis/tests/test_parent_qpf.py -v`
Expected: FAIL with `ImportError: cannot import name 'swath_masked' from 'parent_qpf'`.

- [ ] **Step 3: Add the helper**

In `analysis/parent_qpf.py`, add this function just above `def plot_compare` (around line 233, after `qpf_cmap`):

```python
def swath_masked(field, lats, lons, case, end_fhour):
    """Zero `field` outside the TC display swath; NaN->0 inside it.

    The swath is the union of circles of `case.display_radius_km` around the
    best-track position at each hour 0..end_fhour.  `lats`/`lons` are 2-D
    arrays of the same shape as `field` (any grid: HAFS-native or fixed mesh).
    """
    mask = np.zeros(np.shape(lats), dtype=bool)
    for h in range(0, end_fhour + 1):
        tlat, tlon = case.position_at(case.init_dt + timedelta(hours=h))
        mask |= haversine_km(tlat, tlon, lats, lons) <= case.display_radius_km
    return np.where(mask, np.nan_to_num(field, nan=0.0), 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest analysis/tests/test_parent_qpf.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Refactor the parent panel to use the helper**

In `analysis/parent_qpf.py`, replace the inline parent swath block (currently lines 319–323):

```python
    hafs_swath = np.zeros(hafs_lats.shape, dtype=bool)
    for h in range(0, end_fhour + 1):
        tlat, tlon = case.position_at(case.init_dt + timedelta(hours=h))
        hafs_swath |= haversine_km(tlat, tlon, hafs_lats, hafs_lons) <= case.display_radius_km
    hafs_display = np.where(hafs_swath, np.nan_to_num(hafs_mm, nan=0.0), 0.0)
```

with:

```python
    hafs_display = swath_masked(hafs_mm, hafs_lats, hafs_lons, case, end_fhour)
```

- [ ] **Step 6: Re-run the suite to confirm nothing broke**

Run: `pytest analysis/tests/ -v`
Expected: PASS (all existing tests + the 2 new ones).

- [ ] **Step 7: Commit**

```bash
git add analysis/parent_qpf.py analysis/tests/test_parent_qpf.py
git commit -m "Extract swath_masked helper; reuse for parent panel"
```

---

### Task 2: Add the nest panel to the QPF figure

Add a `compute_nest_field()` helper (discover + accumulate + mask, with a graceful `None` when no nest files exist), wire it into `generate_parent_figure`, prepend the nest panel, and update the title and summary print.

**Files:**
- Modify: `analysis/parent_qpf.py` (imports ~line 53; new helper; `generate_parent_figure` panels block at lines 367–376 and summary print 380–383; suptitle at lines 270–275)
- Modify: `analysis/tests/test_parent_qpf.py` (add tests)

**Interfaces:**
- Consumes: `swath_masked` (Task 1); `case.fixed_grid() -> (grid_lat, grid_lon)`; `case.storm_glob()`, `case.run_dir`, `case.fhours_filter`.
- Produces: `compute_nest_field(case, grid_lat, grid_lon, end_fhour) -> np.ndarray | None` — returns the swath-masked moving-nest total on the fixed grid, or `None` if `discover_files` finds no `storm.atm` files.

- [ ] **Step 1: Write the failing tests**

Append to `analysis/tests/test_parent_qpf.py` (before the `if __name__` block):

```python
def test_compute_nest_field_returns_none_when_no_files(monkeypatch):
    lat2d, lon2d = _grid()

    class C(FakeCase):
        run_dir = Path("/nowhere")
        fhours_filter = None
        def storm_glob(self):
            return "*storm.atm.f*.grb2"

    monkeypatch.setattr(parent_qpf, "discover_files", lambda *a, **k: [])
    out = parent_qpf.compute_nest_field(C(), lat2d, lon2d, end_fhour=0)
    assert out is None


def test_compute_nest_field_masks_total_to_swath(monkeypatch):
    lat2d, lon2d = _grid()

    class C(FakeCase):
        run_dir = Path("/nowhere")
        fhours_filter = None
        def storm_glob(self):
            return "*storm.atm.f*.grb2"

    # Pretend discovery found one file-pair; stub the heavy accumulation.
    monkeypatch.setattr(parent_qpf, "discover_files",
                        lambda *a, **k: [(0, Path("f000"))])
    monkeypatch.setattr(parent_qpf, "hafs_event_total",
                        lambda *a, **k: (np.full(lat2d.shape, 40.0), "amount"))

    case = C(lat=30.0, lon=-85.0, radius_km=200.0)
    out = parent_qpf.compute_nest_field(case, lat2d, lon2d, end_fhour=0)
    assert out is not None
    ci = np.argmin(np.abs(lat2d[:, 0] - 30.0))
    cj = np.argmin(np.abs(lon2d[0, :] - (-85.0)))
    assert out[ci, cj] == 40.0        # kept at center
    assert out[0, 0] == 0.0           # zeroed in far corner
```

Update the `if __name__ == "__main__"` block to keep the script-run path working without pytest fixtures (the monkeypatch tests are pytest-only):

```python
if __name__ == "__main__":
    test_swath_masked_zeros_outside_radius()
    test_swath_masked_replaces_nan_with_zero_inside_swath()
    print("ok (run `pytest` for the monkeypatched nest tests)")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest analysis/tests/test_parent_qpf.py -v`
Expected: FAIL with `AttributeError: module 'parent_qpf' has no attribute 'compute_nest_field'`.

- [ ] **Step 3: Add the imports and the helper**

In `analysis/parent_qpf.py`, extend the `from hafs_common import (...)` block (starts line 53) to also import `discover_files` and `hafs_event_total`. The block currently reads:

```python
from hafs_common import (
    QPF_LEVELS, QPF_COLORS,
    read_hafs_tp_records, haversine_km, load_mrms_hour, crop_to_domain,
```

Add the two names to it, e.g. append a line inside the parentheses:

```python
    discover_files, hafs_event_total,
```

Then add the helper just above `def generate_parent_figure` (around line 284):

```python
def compute_nest_field(case, grid_lat, grid_lon, end_fhour):
    """Moving-nest running-max APCP on the fixed grid, masked to the swath.

    Same field ets_full.py scores (hafs_event_total over the storm.atm files).
    Returns None when no nest files are found so the figure can still render
    its other panels.
    """
    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    if not file_pairs:
        print("  No storm.atm (nest) files found — nest panel unavailable.")
        return None
    nest_total, mode = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  nest APCP mode: {mode}, max {np.nanmax(nest_total):.0f} mm")
    return swath_masked(nest_total, grid_lat, grid_lon, case, end_fhour)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest analysis/tests/test_parent_qpf.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Wire the nest panel into `generate_parent_figure`**

In `analysis/parent_qpf.py`, immediately before the `# Plot 3-panel.` comment (around line 364, after the Stage IV block), compute the nest field:

```python
    # ------------------------------------------------------------------
    # Moving-nest total on the fixed verification grid (same field ETS
    # scores), masked to the display swath.
    # ------------------------------------------------------------------
    print("\nAccumulating HAFS nest (storm.atm) running-max APCP ...")
    grid_lat, grid_lon = case.fixed_grid()
    nest_display = compute_nest_field(case, grid_lat, grid_lon, end_fhour)
```

Then change the panels list (lines 367–376) to prepend the nest panel and update the comment:

```python
    # ------------------------------------------------------------------
    # Plot 4-panel.
    # ------------------------------------------------------------------
    out_png = case.out_dir / f"parent_qpf_{case.output_slug}.png"
    panels = [
        (grid_lon, grid_lat, nest_display,
         f"{case.model_label} Nest APCP (moving 2-km, running-max)\n"
         f"0–{end_fhour}h — can inflate vs swept frontal rain"),
        (hafs_lons, hafs_lats, hafs_display,
         f"{case.model_label} Parent APCP\n0–{end_fhour}h (valid {valid_end:%Y-%m-%d %HZ})"),
        (mrms_lons, mrms_lats, mrms_total,
         f"MRMS MultiSensor QPE (Pass2)\n{end_fhour}h accumulation"),
        (s4_lons, s4_lats, s4_total,
         f"NCEP Stage IV QPE (CONUS)\n{s4_label}"),
    ]
    plot_compare(case, panels, end_fhour, out_png)
```

- [ ] **Step 6: Update the suptitle and summary print**

In `plot_compare`, change the suptitle (lines 270–275) from:

```python
        f"{case.storm_name} — {case.model_label} parent QPF vs MRMS vs Stage IV "
```

to:

```python
        f"{case.storm_name} — {case.model_label} QPF: nest vs parent vs MRMS vs Stage IV "
```

In `generate_parent_figure`, extend the closing summary print (lines 381–383) to include the nest max:

```python
    print(f"\nSaved {out_png}")
    print(f"  HAFS nest max {_mx(nest_display):.0f} mm | "
          f"HAFS parent max {_mx(hafs_display):.0f} mm | "
          f"MRMS max {_mx(mrms_total):.0f} mm | "
          f"Stage IV max {_mx(s4_total):.0f} mm")
```

- [ ] **Step 7: Run the full suite**

Run: `pytest analysis/tests/ -v`
Expected: PASS (all tests, including the 4 in `test_parent_qpf.py`).

- [ ] **Step 8: Commit**

```bash
git add analysis/parent_qpf.py analysis/tests/test_parent_qpf.py
git commit -m "Add nest QPF panel to parent_qpf figure"
```

- [ ] **Step 9: Integration check on Hercules (manual)**

After `git pull` on Hercules:

```bash
python analysis/run.py storms/helene_hfsa.yaml parent
```

Expected: console prints a nest APCP max, and `analysis/output/helene_hfsa/parent_qpf_helene_hfsa_2024092400.png` now has **4** panels in order Nest · Parent · MRMS · Stage IV, one shared colorbar, and the updated suptitle. The printed nest max should match the nest max reported by `... ets`. Pull the PNG down with `scp` to eyeball it.

---

## Self-Review

**Spec coverage:**
- 4-panel Nest·Parent·MRMS·Stage IV, order Nest first → Task 2 Step 5. ✓
- Nest = `hafs_event_total` moving-nest running-max → Task 2 Step 3. ✓
- Masked to same display swath, same colorbar → Task 1 helper reused (Task 2 Step 3); `QPF_LEVELS` unchanged in `plot_compare`. ✓
- Caveat in panel title → Task 2 Step 5. ✓
- Graceful "unavailable" when no nest files → `compute_nest_field` returns `None`; `plot_compare` already renders "unavailable" for `None` data. ✓
- Suptitle updated; nest max printed → Task 2 Step 6. ✓
- Filename unchanged; no other modules touched; no circular import (uses `case.fixed_grid()`) → Global Constraints + Task 2. ✓
- Runtime note (now reads all storm.atm files) → inherent to `compute_nest_field`; flagged in spec, no code consequence.

**Placeholder scan:** No TBD/TODO; all code blocks complete. ✓

**Type consistency:** `swath_masked(field, lats, lons, case, end_fhour)` defined Task 1, called identically in `compute_nest_field` and the parent refactor. `compute_nest_field(case, grid_lat, grid_lon, end_fhour) -> ndarray|None` defined and called consistently. `case.fixed_grid()` returns `(grid_lat, grid_lon)` matching usage. ✓
