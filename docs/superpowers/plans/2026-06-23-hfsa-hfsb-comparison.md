# HFSA-vs-HFSB Head-to-Head Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fair HFSA-vs-HFSB comparison driven by a comparison YAML — both models verified over one shared NHC-best-track swath, with categorical skill (ETS/CSI/bias/POD/FAR/HSS) and FSS, output as two comparison figures + two full-matrix CSVs.

**Architecture:** New `best_track.py` (b-deck parser), `skill_metrics.py` (FSS + shared-swath builder), and `compare.py` (config loader + pure scoring + plotting + heavy driver). Tiny edits to `ets_score.py` (add HSS), `hafs_case.py` (extract track interpolation), and `run.py` (add a `compare` command). The per-case `parent|ets|all` path is untouched.

**Tech Stack:** Python 3, numpy, scipy (`ndimage.uniform_filter`), pyyaml, matplotlib; heavy accumulation reuses cfgrib/eccodes/boto3/cartopy on Hercules. Tests are standalone-runnable (`python3 analysis/tests/<file>.py`).

## Global Constraints

- This repo uses `python3` (NOT `python`). `pytest` is NOT installed; every test file must run standalone via `python3 analysis/tests/<file>.py` using a `_run_all()` block that discovers `test_*` functions (pattern below, copy verbatim into each new test file):
  ```python
  def _run_all():
      fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
      for fn in fns:
          fn(); print(f"PASS {fn.__name__}")
      print(f"\n{len(fns)} passed")

  if __name__ == "__main__":
      _run_all()
  ```
- `hafs_case.py` MUST stay import-light: stdlib + numpy + yaml only (no cfgrib/boto3/eccodes/cartopy). `best_track.py` likewise imports only stdlib + `hafs_case`.
- The full scientific dependency stack IS installed locally, so all new pure/array functions (parsers, FSS, swath, scoring, plotting, config) are unit-tested locally. Only `compare.generate_comparison` (GRIB/MRMS/Stage IV accumulation) is integration-only (Hercules).
- HSS formula (verbatim): `hss = 2*(a*d - b*c) / ((a+c)*(c+d) + (a+b)*(b+d))`, NaN if denom == 0.
- FSS formula (verbatim): binarize at threshold, fractional fields via `scipy.ndimage.uniform_filter(size=scale, mode="constant", cval=0.0)`, then `FSS = 1 - MSE(Mf,Of)/(mean(Mf^2)+mean(Of^2))` over `mask` points only; NaN if denom == 0.
- Default `fss_scales_cells = [1, 3, 5, 11, 21, 41]`; default `fss_plot_thresholds = [10, 25, 50]`; `scale_km = round(scale_cells * grid_res * 111.0, 1)`.
- Categorical CSV columns (exact order): `model, forecast, observation, threshold, a, b, c, d, ets, csi, bias, pod, far, hss`.
- FSS CSV columns (exact order): `model, forecast, observation, threshold, scale_cells, scale_km, fss`.
- Comparison config requires `cases` (exactly 2 case-YAML paths) and `best_track`; `out_dir` defaults to `analysis/output/<config-stem>`.
- Commit messages: NO `Co-Authored-By` lines. Run scripts from repo root.
- The per-case path and all existing tests (`test_hafs_case.py` 14, `test_ets_full.py` 5, `test_run.py` 3) must stay green.

---

### Task 1: Add HSS to `contingency_scores`

**Files:**
- Modify: `analysis/ets_score.py` (the `contingency_scores` function)
- Test: `analysis/tests/test_ets_score.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `contingency_scores(fcst, obs, threshold)` now also returns key `"hss"` in its dict (alongside existing `threshold,a,b,c,d,ets,bias,pod,far,csi`).

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/test_ets_score.py`:

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ets_score import contingency_scores


def test_hss_perfect_forecast_is_one():
    # Forecast exactly equals obs -> b=c=0 -> HSS = 1.0
    fcst = np.array([10.0, 0.0, 20.0, 0.0])
    obs = np.array([10.0, 0.0, 20.0, 0.0])
    s = contingency_scores(fcst, obs, 5.0)
    assert s["b"] == 0 and s["c"] == 0
    assert abs(s["hss"] - 1.0) < 1e-12


def test_hss_known_table():
    # Build a case with a=2,b=1,c=1,d=2 at threshold 5.
    #   fcst>=5 at idx 0,1,2 ; obs>=5 at idx 0,2,4 (over 6 points)
    fcst = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    obs = np.array([10.0, 0.0, 10.0, 0.0, 10.0, 0.0])
    s = contingency_scores(fcst, obs, 5.0)
    assert (s["a"], s["b"], s["c"], s["d"]) == (2, 1, 1, 2)
    # HSS = 2(ad-bc)/((a+c)(c+d)+(a+b)(b+d))
    #     = 2(4-1)/((3)(3)+(3)(3)) = 6/18 = 0.3333...
    assert abs(s["hss"] - (6.0 / 18.0)) < 1e-12


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_ets_score.py`
Expected: FAIL — `KeyError: 'hss'`.

- [ ] **Step 3: Implement HSS**

In `analysis/ets_score.py`, inside `contingency_scores`, after the `csi = ...`
line and before the `return dict(...)`, add:

```python
    hss_denom = (a + c) * (c + d) + (a + b) * (b + d)
    hss = (2 * (a * d - b * c) / hss_denom) if hss_denom != 0 else np.nan
```

Then add `hss=hss,` to the returned dict (place it after `csi=csi`):

```python
    return dict(threshold=threshold, a=a, b=b, c=c, d=d,
                ets=ets, bias=bias, pod=pod, far=far, csi=csi, hss=hss)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 analysis/tests/test_ets_score.py`
Expected: `2 passed`.
Also confirm the per-case ETS test is unaffected: `python3 analysis/tests/test_ets_full.py`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add analysis/ets_score.py analysis/tests/test_ets_score.py
git commit -m "Add HSS (Heidke Skill Score) to contingency_scores"
```

---

### Task 2: Extract `position_on_track` from `StormCase.position_at`

**Files:**
- Modify: `analysis/hafs_case.py`
- Test: `analysis/tests/test_hafs_case.py` (existing — add one test)

**Interfaces:**
- Consumes: nothing new.
- Produces: module-level `position_on_track(track, valid_dt) -> (lat, lon)` where
  `track` is `[(datetime, lat, lon), ...]`; linear interp, clamps to endpoints.
  `StormCase.position_at` now delegates to it (unchanged behavior).

- [ ] **Step 1: Write the failing test**

Add to `analysis/tests/test_hafs_case.py` (it already imports datetime, Path, sys):

```python
from hafs_case import position_on_track


def test_position_on_track_interpolates_and_clamps():
    track = [
        (datetime(2024, 9, 24, 0), 16.8, -83.2),
        (datetime(2024, 9, 24, 6), 17.8, -83.5),
    ]
    lat, lon = position_on_track(track, datetime(2024, 9, 24, 3))
    assert abs(lat - 17.3) < 1e-9 and abs(lon - (-83.35)) < 1e-9
    assert position_on_track(track, datetime(2024, 9, 23, 0)) == (16.8, -83.2)
    assert position_on_track(track, datetime(2024, 9, 25, 0)) == (17.8, -83.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: FAIL — `ImportError: cannot import name 'position_on_track'`.

- [ ] **Step 3: Implement the free function and delegate**

In `analysis/hafs_case.py`, add this module-level function (place it just above
the `@dataclass class StormCase` line):

```python
def position_on_track(track, valid_dt):
    """Linear interpolation of a [(dt, lat, lon), ...] track to any time.

    Clamps to the endpoints outside the track's time span.
    """
    times = [t for t, _, _ in track]
    lats = [la for _, la, _ in track]
    lons = [lo for _, _, lo in track]
    if valid_dt <= times[0]:
        return lats[0], lons[0]
    if valid_dt >= times[-1]:
        return lats[-1], lons[-1]
    for i in range(len(times) - 1):
        if times[i] <= valid_dt <= times[i + 1]:
            frac = ((valid_dt - times[i]).total_seconds()
                    / (times[i + 1] - times[i]).total_seconds())
            return (lats[i] + frac * (lats[i + 1] - lats[i]),
                    lons[i] + frac * (lons[i + 1] - lons[i]))
    return lats[-1], lons[-1]
```

Then replace the body of `StormCase.position_at` with a one-line delegate:

```python
    def position_at(self, valid_dt):
        """Linear interpolation of the track to any time; clamps to endpoints."""
        return position_on_track(self.track, valid_dt)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: `15 passed` (14 existing + 1 new; the existing
`test_position_at_interpolates_and_clamps` still passes — behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add analysis/hafs_case.py analysis/tests/test_hafs_case.py
git commit -m "Extract position_on_track free function from StormCase.position_at"
```

---

### Task 3: NHC b-deck parser (`best_track.py`)

**Files:**
- Create: `analysis/best_track.py`
- Create (fixture): `analysis/tests/fixtures/bal092024_sample.dat`
- Test: `analysis/tests/test_best_track.py` (new)

**Interfaces:**
- Consumes: `hafs_case.decode_latlon`.
- Produces: `parse_bdeck(path) -> list[(datetime, lat, lon)]` — `BEST` lines only,
  time from column index 2 (`YYYYMMDDHH`), deduped by time, sorted ascending;
  raises `ValueError` if no BEST fixes parse.

- [ ] **Step 1: Create the b-deck fixture**

Create `analysis/tests/fixtures/bal092024_sample.dat` (real b-deck layout — note
the same time repeats across 34/50/64-kt wind-radii lines, which the parser must
dedupe; `BEST` is column 5, time is column 3, `TAU` column 6 is 0):

```
AL, 09, 2024092400,   , BEST,   0, 168N,  832W,  35, 1004, TS,  34, NEQ, 0060, 0000, 0000, 0060
AL, 09, 2024092400,   , BEST,   0, 168N,  832W,  35, 1004, TS,  50, NEQ, 0000, 0000, 0000, 0000
AL, 09, 2024092406,   , BEST,   0, 178N,  835W,  45, 1000, TS,  34, NEQ, 0090, 0060, 0030, 0090
AL, 09, 2024092412,   , BEST,   0, 190N,  838W,  55,  994, TS,  34, NEQ, 0120, 0090, 0060, 0120
AL, 09, 2024092418,   , BEST,   0, 204N,  841W,  65,  984, HU,  64, NEQ, 0020, 0020, 0000, 0020
```

- [ ] **Step 2: Write the failing test**

Create `analysis/tests/test_best_track.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 analysis/tests/test_best_track.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'best_track'`.

- [ ] **Step 4: Implement `best_track.py`**

Create `analysis/best_track.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 analysis/tests/test_best_track.py`
Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add analysis/best_track.py analysis/tests/test_best_track.py analysis/tests/fixtures/bal092024_sample.dat
git commit -m "Add NHC b-deck best-track parser"
```

---

### Task 4: Shared-swath builder (`skill_metrics.py`)

**Files:**
- Create: `analysis/skill_metrics.py`
- Test: `analysis/tests/test_skill_metrics.py` (new)

**Interfaces:**
- Consumes: `hafs_common.haversine_km`, `hafs_case.position_on_track` (Task 2).
- Produces: `swath_from_track(track, grid_lat, grid_lon, radius_km, init_dt,
  max_fhour) -> bool ndarray` — union of `radius_km` circles along the track,
  interpolated hourly over `[init_dt, init_dt + max_fhour h]`; prints a warning
  (once) if the track ends before the window.

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/test_skill_metrics.py`:

```python
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_metrics import swath_from_track


def test_swath_from_track_marks_points_within_radius():
    # 1-degree grid around a single-fix track at (20, -85).
    lons = np.linspace(-90.0, -80.0, 11)
    lats = np.linspace(15.0, 25.0, 11)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    track = [(datetime(2024, 9, 24, 0), 20.0, -85.0),
             (datetime(2024, 9, 24, 6), 20.0, -85.0)]
    swath = swath_from_track(track, grid_lat, grid_lon, 200.0,
                             datetime(2024, 9, 24, 0), 6)
    # Center cell (20, -85) is inside; a far corner (15, -90) ~ 780 km is outside.
    ci = np.argmin(np.abs(lats - 20.0))
    cj = np.argmin(np.abs(lons - (-85.0)))
    assert swath[ci, cj]
    assert not swath[0, 0]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_skill_metrics.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'skill_metrics'`.

- [ ] **Step 3: Implement `skill_metrics.py` (swath part)**

Create `analysis/skill_metrics.py`:

```python
"""Spatial verification helpers: shared-track swath + Fractions Skill Score."""

import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hafs_common import haversine_km
from hafs_case import position_on_track


def swath_from_track(track, grid_lat, grid_lon, radius_km, init_dt, max_fhour):
    """Boolean mask: union of radius_km circles along the track, hourly over
    [init_dt, init_dt + max_fhour h]. Warns once if the track ends early."""
    swath = np.zeros(grid_lat.shape, dtype=bool)
    last_t = track[-1][0]
    warned = False
    for h in range(0, max_fhour + 1):
        t = init_dt + timedelta(hours=h)
        if t > last_t and not warned:
            print(f"  warning: best track ends {last_t:%Y-%m-%d %HZ} before "
                  f"forecast hour {h} ({t:%Y-%m-%d %HZ}); swath clamps to last fix")
            warned = True
        tlat, tlon = position_on_track(track, t)
        swath |= haversine_km(tlat, tlon, grid_lat, grid_lon) <= radius_km
    return swath
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 analysis/tests/test_skill_metrics.py`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add analysis/skill_metrics.py analysis/tests/test_skill_metrics.py
git commit -m "Add swath_from_track shared-swath builder"
```

---

### Task 5: Fractions Skill Score (`skill_metrics.py`)

**Files:**
- Modify: `analysis/skill_metrics.py`
- Test: `analysis/tests/test_skill_metrics.py` (existing — add tests)

**Interfaces:**
- Consumes: `scipy.ndimage.uniform_filter` (already imported in Task 4).
- Produces: `fractions_skill_score(fcst, obs, threshold, scale, mask) -> float` —
  FSS at one (threshold, neighborhood size in cells) over `mask` points; NaN if
  no events at that threshold within the mask.

- [ ] **Step 1: Write the failing test**

Add to `analysis/tests/test_skill_metrics.py`:

```python
from skill_metrics import fractions_skill_score


def test_fss_identical_fields_is_one():
    f = np.zeros((10, 10)); f[3:6, 3:6] = 20.0
    mask = np.ones((10, 10), dtype=bool)
    assert abs(fractions_skill_score(f, f.copy(), 5.0, 1, mask) - 1.0) < 1e-12


def test_fss_disjoint_events_is_zero_at_scale1():
    f = np.zeros((10, 10)); f[1, 1] = 20.0
    o = np.zeros((10, 10)); o[8, 8] = 20.0
    mask = np.ones((10, 10), dtype=bool)
    # No overlap at scale 1 -> MSE == ref -> FSS == 0.
    assert abs(fractions_skill_score(f, o, 5.0, 1, mask) - 0.0) < 1e-12


def test_fss_no_events_is_nan():
    f = np.zeros((10, 10)); o = np.zeros((10, 10))
    mask = np.ones((10, 10), dtype=bool)
    assert np.isnan(fractions_skill_score(f, o, 5.0, 1, mask))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 analysis/tests/test_skill_metrics.py`
Expected: FAIL — `ImportError: cannot import name 'fractions_skill_score'`.

- [ ] **Step 3: Implement `fractions_skill_score`**

Append to `analysis/skill_metrics.py`:

```python
def fractions_skill_score(fcst, obs, threshold, scale, mask):
    """FSS at one threshold and neighborhood size (scale, in grid cells).

    FSS = 1 - MSE(Mf, Of) / (mean(Mf^2) + mean(Of^2)) over mask points, where
    Mf/Of are neighborhood fractions of the binarized fields. NaN if there are
    no events at this threshold anywhere (denominator 0).
    """
    fb = (fcst >= threshold).astype(float)
    ob = (obs >= threshold).astype(float)
    ff = uniform_filter(fb, size=scale, mode="constant", cval=0.0)
    of = uniform_filter(ob, size=scale, mode="constant", cval=0.0)
    mse = np.mean((ff[mask] - of[mask]) ** 2)
    ref = np.mean(ff[mask] ** 2) + np.mean(of[mask] ** 2)
    return 1.0 - mse / ref if ref > 0 else np.nan
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 analysis/tests/test_skill_metrics.py`
Expected: `4 passed` (1 swath + 3 FSS).

- [ ] **Step 5: Commit**

```bash
git add analysis/skill_metrics.py analysis/tests/test_skill_metrics.py
git commit -m "Add fractions_skill_score (FSS)"
```

---

### Task 6: Comparison config loader (`compare.py`)

**Files:**
- Create: `analysis/compare.py`
- Test: `analysis/tests/test_compare.py` (new)

**Interfaces:**
- Consumes: nothing new (stdlib + yaml).
- Produces: `load_comparison(path) -> dict` with keys `label`, `case_paths`
  (list of 2 strings), `best_track` (str), `out_dir` (Path), `thresholds_mm`
  (list or None), `fss_scales_cells` (list), `fss_plot_thresholds` (list).
  Raises `ValueError` if `cases` is missing/not length 2, `KeyError` if
  `best_track` missing.

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/test_compare.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 analysis/tests/test_compare.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'compare'`.

- [ ] **Step 3: Implement `compare.py` (config loader + imports)**

Create `analysis/compare.py`:

```python
"""HFSA-vs-HFSB head-to-head comparison over a shared best-track swath.

Loaded via run.py:  python analysis/run.py storms/<name>_compare.yaml compare
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_DEFAULT_FSS_SCALES = [1, 3, 5, 11, 21, 41]
_DEFAULT_FSS_PLOT_THR = [10, 25, 50]


def load_comparison(path):
    """Parse + validate a comparison YAML and fill defaults (no GRIB loading)."""
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    cases = cfg.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError(f"'cases' must list exactly 2 case YAMLs in {path}")
    if "best_track" not in cfg:
        raise KeyError(f"'best_track' is required in {path}")
    return {
        "label": cfg.get("label", path.stem),
        "case_paths": [str(c) for c in cases],
        "best_track": str(cfg["best_track"]),
        "out_dir": Path(cfg["out_dir"]) if cfg.get("out_dir")
                   else Path("analysis/output") / path.stem,
        "thresholds_mm": cfg.get("thresholds_mm"),
        "fss_scales_cells": cfg.get("fss_scales_cells", list(_DEFAULT_FSS_SCALES)),
        "fss_plot_thresholds": cfg.get("fss_plot_thresholds",
                                       list(_DEFAULT_FSS_PLOT_THR)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 analysis/tests/test_compare.py`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add analysis/compare.py analysis/tests/test_compare.py
git commit -m "Add comparison config loader (compare.load_comparison)"
```

---

### Task 7: Scoring matrix (`compare.py`)

**Files:**
- Modify: `analysis/compare.py`
- Test: `analysis/tests/test_compare.py` (existing — add tests)

**Interfaces:**
- Consumes: `ets_full.score_pair`, `ets_score.contingency_scores` (Task 1, now
  includes `hss`), `skill_metrics.fractions_skill_score` (Task 5).
- Produces: `score_matrix(models, swath, thresholds, fss_scales, grid_res) ->
  (cat_rows, fss_rows)` where `models` is a list of dicts
  `{"name": str, "forecasts": {fname: 2d-array}, "obs": {oname: 2d-array|None}}`.
  `cat_rows` items: `{model, forecast, observation, threshold, a,b,c,d, ets,
  bias, pod, far, csi, hss}`. `fss_rows` items: `{model, forecast, observation,
  threshold, scale_cells, scale_km, fss}`. None obs are skipped.

- [ ] **Step 1: Write the failing test**

Add to `analysis/tests/test_compare.py`:

```python
import numpy as np
from compare import score_matrix


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_compare.py`
Expected: FAIL — `ImportError: cannot import name 'score_matrix'`.

- [ ] **Step 3: Implement `score_matrix`**

Add these imports near the top of `analysis/compare.py` (after the existing
imports):

```python
from ets_full import score_pair
from ets_score import contingency_scores
from skill_metrics import fractions_skill_score
```

Append the function:

```python
def score_matrix(models, swath, thresholds, fss_scales, grid_res):
    """Score every model x forecast x obs over the shared swath.

    Returns (cat_rows, fss_rows). None observations are skipped.
    """
    cat_rows, fss_rows = [], []
    for m in models:
        for fname, fgrid in m["forecasts"].items():
            for oname, ogrid in m["obs"].items():
                if ogrid is None:
                    continue
                rows, _ = score_pair(fgrid, ogrid, swath, thresholds,
                                     contingency_scores)
                for r in rows:
                    cat_rows.append({"model": m["name"], "forecast": fname,
                                     "observation": oname, **r})
                vmask = swath & np.isfinite(fgrid) & np.isfinite(ogrid)
                ff = np.nan_to_num(fgrid, nan=0.0)
                oo = np.nan_to_num(ogrid, nan=0.0)
                for thr in thresholds:
                    for sc in fss_scales:
                        fss_rows.append({
                            "model": m["name"], "forecast": fname,
                            "observation": oname, "threshold": thr,
                            "scale_cells": sc,
                            "scale_km": round(sc * grid_res * 111.0, 1),
                            "fss": fractions_skill_score(ff, oo, thr, sc, vmask),
                        })
    return cat_rows, fss_rows
```

Note: `score_pair` already restricts to `swath & finite & finite` and zero-fills,
and `contingency_scores` (Task 1) now returns `hss`, so each categorical row
carries the full column set. The `cat_rows` dict order (`model, forecast,
observation`, then the contingency keys `threshold, a, b, c, d, ets, bias, pod,
far, csi, hss`) is reordered for the CSV in Task 9.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 analysis/tests/test_compare.py`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add analysis/compare.py analysis/tests/test_compare.py
git commit -m "Add score_matrix: categorical + FSS over the shared swath"
```

---

### Task 8: Comparison plots (`compare.py`)

**Files:**
- Modify: `analysis/compare.py`
- Test: `analysis/tests/test_compare.py` (existing — add tests)

**Interfaces:**
- Consumes: `cat_rows` / `fss_rows` from `score_matrix` (Task 7); matplotlib
  (already imported).
- Produces:
  - `plot_categorical_compare(cat_rows, label, out_path, observation="MRMS")` —
    3 panels (ETS, CSI, frequency bias) vs threshold; HFSA vs HFSB by color,
    parent/nest by solid/dashed; rows filtered to `observation`. Saves PNG.
  - `plot_fss_compare(fss_rows, label, out_path, observation="MRMS",
    forecast="parent", plot_thresholds=(10, 25, 50))` — FSS vs scale_km, one
    line per (model, threshold); rows filtered to `observation` + `forecast`.
    Saves PNG.

- [ ] **Step 1: Write the failing test**

Add to `analysis/tests/test_compare.py`:

```python
from compare import plot_categorical_compare, plot_fss_compare


def _toy_rows():
    g = np.zeros((12, 12)); g[4:8, 4:8] = 20.0
    swath = np.ones((12, 12), dtype=bool)
    models = [
        {"name": "HFSA", "forecasts": {"parent": g.copy(), "nest": g.copy()},
         "obs": {"MRMS": g.copy()}},
        {"name": "HFSB", "forecasts": {"parent": g.copy(), "nest": g.copy()},
         "obs": {"MRMS": g.copy()}},
    ]
    return score_matrix(models, swath, [5.0, 25.0, 50.0], [1, 3], 0.05)


def test_plots_write_png_files():
    import tempfile
    cat, fss = _toy_rows()
    d = Path(tempfile.mkdtemp())
    cat_png = d / "cat.png"
    fss_png = d / "fss.png"
    plot_categorical_compare(cat, "Test", cat_png, observation="MRMS")
    plot_fss_compare(fss, "Test", fss_png, observation="MRMS",
                     forecast="parent", plot_thresholds=(5.0, 25.0))
    assert cat_png.exists() and cat_png.stat().st_size > 0
    assert fss_png.exists() and fss_png.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_compare.py`
Expected: FAIL — `ImportError: cannot import name 'plot_categorical_compare'`.

- [ ] **Step 3: Implement the two plot functions**

Append to `analysis/compare.py`:

```python
_MODEL_COLOR = {"HFSA": "#1f77b4", "HFSB": "#d62728"}
_FCST_STYLE = {"parent": dict(ls="-", marker="o"),
               "nest": dict(ls="--", marker="s")}


def plot_categorical_compare(cat_rows, label, out_path, observation="MRMS"):
    """3 panels (ETS, CSI, freq bias) vs threshold; HFSA/HFSB x parent/nest."""
    rows = [r for r in cat_rows if r["observation"] == observation]
    metrics = [("ets", "Equitable Threat Score"),
               ("csi", "Critical Success Index"),
               ("bias", "Frequency bias")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models = sorted({r["model"] for r in rows})
    forecasts = sorted({r["forecast"] for r in rows})
    for ax, (key, title) in zip(axes, metrics):
        for mdl in models:
            for fc in forecasts:
                sub = sorted((r for r in rows
                              if r["model"] == mdl and r["forecast"] == fc),
                             key=lambda r: r["threshold"])
                if not sub:
                    continue
                ax.plot([r["threshold"] for r in sub], [r[key] for r in sub],
                        color=_MODEL_COLOR.get(mdl, "gray"),
                        **_FCST_STYLE.get(fc, dict(ls="-", marker="o")),
                        lw=2, label=f"{mdl} {fc}")
        ax.set_xscale("log")
        ax.set_xlabel("Rainfall threshold (mm)")
        ax.set_ylabel(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if key == "bias":
            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        else:
            ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{label} — HFSA vs HFSB categorical skill (vs {observation})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fss_compare(fss_rows, label, out_path, observation="MRMS",
                     forecast="parent", plot_thresholds=(10, 25, 50)):
    """FSS vs neighborhood scale (km); one line per (model, threshold)."""
    rows = [r for r in fss_rows if r["observation"] == observation
            and r["forecast"] == forecast and r["threshold"] in plot_thresholds]
    fig, ax = plt.subplots(figsize=(9, 6))
    models = sorted({r["model"] for r in rows})
    thrs = sorted({r["threshold"] for r in rows})
    dashes = {t: (None if i == 0 else (4 + 2 * i, 2))
              for i, t in enumerate(thrs)}
    for mdl in models:
        for t in thrs:
            sub = sorted((r for r in rows
                          if r["model"] == mdl and r["threshold"] == t),
                         key=lambda r: r["scale_km"])
            if not sub:
                continue
            line, = ax.plot([r["scale_km"] for r in sub],
                            [r["fss"] for r in sub],
                            color=_MODEL_COLOR.get(mdl, "gray"),
                            lw=2, marker="o",
                            label=f"{mdl}  {int(t)} mm")
            if dashes[t] is not None:
                line.set_dashes(dashes[t])
    ax.set_xlabel("Neighborhood scale (km)")
    ax.set_ylabel("Fractions Skill Score (FSS)")
    ax.set_ylim(0, 1)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{label} — HFSA vs HFSB FSS ({forecast} vs {observation})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 analysis/tests/test_compare.py`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add analysis/compare.py analysis/tests/test_compare.py
git commit -m "Add HFSA-vs-HFSB comparison plots (categorical + FSS)"
```

---

### Task 9: Comparison driver (`compare.generate_comparison`)

**Files:**
- Modify: `analysis/compare.py`

**Interfaces:**
- Consumes: `hafs_case.from_yaml`, `best_track.parse_bdeck` (Task 3),
  `skill_metrics.swath_from_track` (Task 4), `hafs_common.discover_files` +
  `hafs_common.hafs_event_total`, `ets_full.hafs_parent_total` +
  `ets_full.stage4_on_fixed`, `ets_score.build_mrms_total`, `score_matrix`
  (Task 7), the plot functions (Task 8).
- Produces: `generate_comparison(cfg) -> None` — full driver. Writes
  `compare_categorical_<slug>.csv`, `compare_fss_<slug>.csv`,
  `compare_categorical_<slug>.png`, `compare_fss_<slug>.png` under `cfg["out_dir"]`,
  and prints a summary. (`<slug>` = `cfg["label"]` lowercased, spaces→`_`.)

- [ ] **Step 1: Add the driver imports + helper**

Add to the imports block of `analysis/compare.py`:

```python
from hafs_case import from_yaml
from best_track import parse_bdeck
from skill_metrics import swath_from_track
from hafs_common import discover_files, hafs_event_total
from ets_full import hafs_parent_total, stage4_on_fixed
from ets_score import build_mrms_total
```

- [ ] **Step 2: Implement `generate_comparison`**

Append to `analysis/compare.py`:

```python
def _slug(label):
    return label.lower().replace(" ", "_")


def _build_model_fields(case, grid_lat, grid_lon, max_fhour):
    """Parent + nest forecast totals and MRMS + Stage IV obs on the common grid."""
    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    print(f"  {case.model_label}: {len(file_pairs)} storm files, nest total ...")
    nest_total, _ = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  {case.model_label}: parent total ...")
    parent_total = hafs_parent_total(case, grid_lat, grid_lon)
    print(f"  {case.model_label}: MRMS total ...")
    mrms = build_mrms_total(case, max_fhour, grid_lat, grid_lon)
    print(f"  {case.model_label}: Stage IV total ...")
    stage4, _ = stage4_on_fixed(case, max_fhour, grid_lat, grid_lon)
    return {
        "name": case.model_label,
        "forecasts": {"parent": parent_total, "nest": nest_total},
        "obs": {"MRMS": mrms, "Stage IV": stage4},
    }


def generate_comparison(cfg):
    """Score both cases over one best-track swath and write figures + CSVs."""
    cases = [from_yaml(p) for p in cfg["case_paths"]]
    a, b = cases
    if a.domain != b.domain or a.grid_res != b.grid_res:
        raise ValueError(
            f"cases must share domain/grid_res: {cfg['case_paths'][0]} has "
            f"{a.domain}@{a.grid_res}, {cfg['case_paths'][1]} has {b.domain}@{b.grid_res}")
    if a.mask_radius_km != b.mask_radius_km:
        raise ValueError(
            f"cases must share mask_radius_km ({a.mask_radius_km} vs {b.mask_radius_km})")
    thresholds = cfg["thresholds_mm"] or a.thresholds_mm
    grid_lat, grid_lon = a.fixed_grid()

    print(f"Best track: {cfg['best_track']}")
    track = parse_bdeck(cfg["best_track"])

    fa = discover_files(a.run_dir, a.storm_glob(), a.fhours_filter)
    fb = discover_files(b.run_dir, b.storm_glob(), b.fhours_filter)
    if not fa or not fb:
        print("No storm files found for one of the cases; aborting.")
        return
    max_fhour = min(fa[-1][0], fb[-1][0])
    print(f"Shared swath: best track, 0-{max_fhour}h, "
          f"<= {a.mask_radius_km:.0f} km")
    swath = swath_from_track(track, grid_lat, grid_lon, a.mask_radius_km,
                             a.init_dt, max_fhour)

    models = [_build_model_fields(c, grid_lat, grid_lon, max_fhour) for c in cases]
    cat_rows, fss_rows = score_matrix(models, swath, thresholds,
                                      cfg["fss_scales_cells"], a.grid_res)

    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(cfg["label"])

    cat_cols = ["model", "forecast", "observation", "threshold",
                "a", "b", "c", "d", "ets", "csi", "bias", "pod", "far", "hss"]
    cat_csv = out_dir / f"compare_categorical_{slug}.csv"
    with open(cat_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cat_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(cat_rows)

    fss_cols = ["model", "forecast", "observation", "threshold",
                "scale_cells", "scale_km", "fss"]
    fss_csv = out_dir / f"compare_fss_{slug}.csv"
    with open(fss_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fss_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(fss_rows)

    cat_png = out_dir / f"compare_categorical_{slug}.png"
    fss_png = out_dir / f"compare_fss_{slug}.png"
    plot_categorical_compare(cat_rows, cfg["label"], cat_png, observation="MRMS")
    plot_fss_compare(fss_rows, cfg["label"], fss_png, observation="MRMS",
                     forecast="parent",
                     plot_thresholds=tuple(cfg["fss_plot_thresholds"]))

    print(f"\nSaved: {cat_csv}")
    print(f"Saved: {fss_csv}")
    print(f"Saved: {cat_png}")
    print(f"Saved: {fss_png}")
    print("\nETS (parent vs MRMS) at 25 / 50 mm:")
    for mdl in sorted({r["model"] for r in cat_rows}):
        e25 = next((r["ets"] for r in cat_rows if r["model"] == mdl
                    and r["forecast"] == "parent" and r["observation"] == "MRMS"
                    and r["threshold"] == 25), float("nan"))
        e50 = next((r["ets"] for r in cat_rows if r["model"] == mdl
                    and r["forecast"] == "parent" and r["observation"] == "MRMS"
                    and r["threshold"] == 50), float("nan"))
        print(f"  {mdl}: ETS25={e25:.3f}  ETS50={e50:.3f}")
```

- [ ] **Step 3: Verify it compiles and imports**

Run: `python3 -m py_compile analysis/compare.py`
Expected: no output (exit 0).

Run: `python3 -c "import sys; sys.path.insert(0,'analysis'); import compare; print('compare imports OK')"`
Expected: `compare imports OK`.

Run the existing compare tests still pass (driver not exercised):
`python3 analysis/tests/test_compare.py`
Expected: `5 passed`.

- [ ] **Step 4: Commit**

```bash
git add analysis/compare.py
git commit -m "Add generate_comparison driver (heavy, integration)"
```

---

### Task 10: Wire the `compare` command into `run.py`

**Files:**
- Modify: `analysis/run.py`
- Test: `analysis/tests/test_run.py` (existing — add one test)

**Interfaces:**
- Consumes: `compare.load_comparison`, `compare.generate_comparison` (lazy import).
- Produces: `run.py` `COMMANDS` includes `"compare"`; `main` routes
  `command == "compare"` to the comparison driver instead of `from_yaml`+`dispatch`.

- [ ] **Step 1: Write the failing test**

Add to `analysis/tests/test_run.py`:

```python
def test_parse_args_accepts_compare():
    yaml_path, command = run.parse_args(["storms/helene_compare.yaml", "compare"])
    assert command == "compare"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_run.py`
Expected: FAIL — `SystemExit` (compare not yet in `COMMANDS`).

- [ ] **Step 3: Add `compare` to commands and route it**

In `analysis/run.py`, change the `COMMANDS` tuple and the usage strings to
include `compare`:

```python
COMMANDS = ("parent", "ets", "all", "compare")
```

Update the two usage strings (module docstring line and the `parse_args` print)
to read `[parent|ets|all|compare]`.

Then route it in `main` — replace the current `main` body with:

```python
def main(argv):
    yaml_path, command = parse_args(argv)
    if command == "compare":
        from compare import load_comparison, generate_comparison
        generate_comparison(load_comparison(yaml_path))
        return
    from hafs_case import from_yaml
    case = from_yaml(yaml_path)
    print(f"Case   : {case.storm_name} ({case.model_label})")
    print(f"Init   : {case.init_dt:%Y-%m-%d %HZ}  | run_dir: {case.run_dir}")
    print(f"Domain : {case.domain}  | track points: {len(case.track)}")
    print(f"Output : {case.out_dir}  | command: {command}")
    dispatch(case, command)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 analysis/tests/test_run.py`
Expected: all pass including `test_parse_args_accepts_compare` (the existing
default-all, explicit-command, and reject-unknown tests still pass).

- [ ] **Step 5: Commit**

```bash
git add analysis/run.py analysis/tests/test_run.py
git commit -m "Wire compare command into run.py"
```

---

### Task 11: Example comparison YAML + README docs

**Files:**
- Create: `storms/helene_compare.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: the comparison-config schema (Task 6) and the `compare` command (Task 10).
- Produces: a runnable example case + user docs.

- [ ] **Step 1: Create `storms/helene_compare.yaml`**

```yaml
# HFSA vs HFSB head-to-head for Hurricane Helene, scored over the NHC best-track
# swath. Get the best track once with:
#   wget https://ftp.nhc.noaa.gov/atcf/archive/2024/bal092024.dat.gz
#   gunzip bal092024.dat.gz
label: Hurricane Helene
cases:
  - storms/helene_hfsa.yaml
  - storms/helene_hfsb.yaml
best_track: /work2/noaa/aoml-hafs1/suchit/bal092024.dat
out_dir: analysis/output/helene_compare
# optional overrides:
# thresholds_mm: [1, 5, 10, 25, 50, 75, 100, 150, 200, 250]
# fss_scales_cells: [1, 3, 5, 11, 21, 41]
# fss_plot_thresholds: [10, 25, 50]
```

- [ ] **Step 2: Add a "Comparing HFSA vs HFSB" section to `README.md`**

Insert after the "Running a new storm" section:

```markdown
## Comparing HFSA vs HFSB (head-to-head)

To score two configs of the same storm against each other over one **shared,
fair verification swath** (the NHC best track), use a comparison config.

1. Fetch the storm's NHC best track (ATCF b-deck) once:

       wget https://ftp.nhc.noaa.gov/atcf/archive/<year>/b<basin><cy><year>.dat.gz
       gunzip b<basin><cy><year>.dat.gz
       # Helene 2024 -> bal092024.dat

2. Edit `storms/helene_compare.yaml` (or copy it) to point `cases:` at the two
   case YAMLs and `best_track:` at the file you just downloaded.

3. Run:

       python analysis/run.py storms/helene_compare.yaml compare

Outputs in `out_dir`:

- `compare_categorical_<label>.png` — ETS / CSI / frequency bias vs threshold,
  HFSA vs HFSB (parent solid, nest dashed), vs MRMS
- `compare_fss_<label>.png` — Fractions Skill Score vs neighborhood scale
- `compare_categorical_<label>.csv`, `compare_fss_<label>.csv` — the full matrix
  (both forecasts × MRMS & Stage IV, all thresholds/scales)

Both models are scored over the identical best-track swath, so their point
counts (`n`) match and the comparison is apples-to-apples. This step runs ~2× a
single `ets` run (it builds both models' nest totals).
```

- [ ] **Step 3: Verify the example YAML parses**

Run: `python3 -c "import sys; sys.path.insert(0,'analysis'); from compare import load_comparison; print(load_comparison('storms/helene_compare.yaml')['label'])"`
Expected: `Hurricane Helene`.

- [ ] **Step 4: Commit**

```bash
git add storms/helene_compare.yaml README.md
git commit -m "Add example comparison YAML and HFSA-vs-HFSB README docs"
```

---

## Self-Review

**Spec coverage:**
- best_track.py / `parse_bdeck` → Task 3. ✓
- skill_metrics.py `swath_from_track` → Task 4; `fractions_skill_score` → Task 5. ✓
- compare.py `load_comparison` → Task 6; scoring → Task 7; plots → Task 8; driver → Task 9. ✓
- `hss` in contingency_scores → Task 1. ✓
- `position_on_track` refactor → Task 2. ✓
- run.py `compare` command → Task 10. ✓
- storms/helene_compare.yaml + README → Task 11. ✓
- Fair best-track swath, identical n → Tasks 4 + 9 (swath built once, used for both). ✓
- CSV column orders, FSS/HSS formulas, default scales → Global Constraints + Tasks 1/5/9. ✓
- Error handling (missing best track, mismatched cases, Stage IV None) → Tasks 3/9 + score_matrix None-skip. ✓
- Testing (local unit tests; driver integration-only) → each task + Task 9 Step 3. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/bare-prose steps — every code
step shows the code; every test step shows the test.

**Type consistency:** `cat_rows`/`fss_rows` dict keys defined in Task 7 match the
CSV column lists and plot accessors in Tasks 8–9 (`model, forecast, observation,
threshold, a,b,c,d, ets, csi, bias, pod, far, hss` and `…, scale_cells, scale_km,
fss`). `score_matrix`, `score_pair`, `contingency_scores` (now with `hss`),
`fractions_skill_score`, `swath_from_track`, `position_on_track`, `parse_bdeck`,
`load_comparison`, `generate_comparison` signatures are consistent across tasks.
`models` dict shape (`name`/`forecasts`/`obs`) is identical in Tasks 7, 8, 9.

## Notes on local vs Hercules testing

- Tasks 1–8, 10, 11 are fully unit-tested locally with `python3` (the full dep
  stack — numpy, scipy, matplotlib, yaml, cfgrib/eccodes/boto3/cartopy — is
  installed on the dev machine; no GRIB *data* needed for these).
- Task 9 (`generate_comparison`) touches real GRIB/MRMS/Stage IV and is verified
  locally only by `py_compile` + import; full integration is the Hercules run:
  `python analysis/run.py storms/helene_compare.yaml compare`, which must produce
  the two PNGs + two CSVs with identical `n` for HFSA and HFSB.
