# Cycle Comparison Product Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `cycles` run mode that scores every initialization of one storm on a common valid window and a shared union-track footprint, producing a metrics-vs-init figure, QPF map small-multiples, and one long-format CSV.

**Architecture:** A thin orchestrator (`analysis/cycles.py`) composes small *windowed* variants of the existing loaders. Observations are absolute-time, so the windowed MRMS/Stage IV totals and the union swath are computed once and shared by every cycle; only forecast window extraction runs per cycle. Config parsing and pure window math live in the dependency-light `analysis/hafs_case.py`.

**Tech Stack:** Python, numpy, matplotlib (Agg), cartopy, eccodes/cfgrib, boto3 (all already in the `hafs` conda env — no new dependencies).

**Spec:** `docs/superpowers/specs/2026-07-09-cycles-comparison-design.md`

## Global Constraints

- **Do NOT run `git commit` or `git push` — ever.** Leave all changes in the working tree. Where this plan's steps would normally commit, instead re-run the tests and report the changed file list.
- Every test file must pass BOTH ways: `python3 analysis/tests/<file>.py` (standalone) and `pytest analysis/tests/<file>.py -v`.
- Continuous bias = `mean(forecast − observed)`; **positive = over-forecast**. In the cycles CSV this column is named `bias_mm` (the categorical frequency bias keeps the name `bias`).
- CSV columns, exactly this order: `init, forecast, observation, threshold, n, rmse, mae, bias_mm, r, a, b, c, d, ets, bias, pod, far, csi, hss`.
- Output filenames, exactly: `cycles_<slug>.csv`, `cycles_metrics_<slug>.png`, `cycles_maps_<slug>.png`, where `<slug>` = `<yaml stem>_<valid_start:%Y%m%d%H>_<valid_end:%Y%m%d%H>`.
- Cycle eligibility: included only if `init ≤ valid_start` AND `init + max forecast hour ≥ valid_end`. Skipped cycles are printed with the reason, never silently dropped.
- The valid-point selection for scoring is identical to `ets_full.score_pair` / `rmse_scatter.valid_points`: `swath & isfinite(obs) & isfinite(fcst)`, kept values zero-filled.
- Do not change the behavior of any existing command (`parent`, `ets`, `rmse`, `all`, `compare`, `replot`); refactors of `build_mrms_total` and `stage4_total` must be behavior-preserving delegations.
- No `Co-Authored-By` lines if the human later commits. Never touch `.nc`/`.grb2`/`.grb` files.

## File Structure

| File | Responsibility |
|---|---|
| `analysis/hafs_case.py` (modify) | `CyclesCase` dataclass, `cycles_from_yaml`, `discover_inits`, `window_hours`, `cycle_eligibility`, `cycle_storm_case`, shared `make_fixed_grid`, cycles-YAML guard in `from_yaml` |
| `analysis/ets_score.py` (modify) | `build_mrms_total_window` (absolute-time MRMS sum); `build_mrms_total` becomes a delegation |
| `analysis/parent_qpf.py` (modify) | `parent_path_at_fhour`, `stage4_sum_days` (extracted core), `stage4_label`, `stage4_total_window`; `stage4_total` becomes a delegation |
| `analysis/cycles.py` (create) | Orchestrator: union swath, windowed nest/parent totals, `build_cycle_fields`, `compute_cycles`, both figures, CSV |
| `analysis/run.py` (modify) | `cycles` command dispatch |
| `analysis/tests/test_cycles.py` (create) | All new unit tests (synthetic data only) |
| `analysis/tests/test_run.py` (modify) | `cycles` accepted by `parse_args` |
| `storms/helene_hfsa_cycles.yaml` (create) | Example cycles case |
| `README.md`, `CLAUDE.md` (modify) | Docs (note: CLAUDE.md is git-ignored; edit it anyway) |

---

### Task 1: Cycles config + window math (`hafs_case.py`)

**Files:**
- Modify: `analysis/hafs_case.py`
- Create: `analysis/tests/test_cycles.py`

**Interfaces:**
- Consumes: existing `StormCase`, `detect_model`, `find_atcfunix`, `parse_atcfunix`, `_DEFAULT_THRESHOLDS` in `hafs_case.py`.
- Produces (later tasks rely on these exact signatures):
  - `make_fixed_grid(domain, grid_res) -> (grid_lat, grid_lon)` 2-D meshes
  - `@dataclass CyclesCase` with fields `run_root: Path, valid_start: datetime, valid_end: datetime, storm_name: str, model_label: str, domain: tuple, grid_res: float, mask_radius_km: float, display_radius_km: float, thresholds_mm: list, ets_threshold_mm: float, out_dir: Path, mrms_cache_dir: Path, stage4_cache_dir: Path, inits: list, case_slug: str`, method `fixed_grid()`, property `output_slug`
  - `cycles_from_yaml(yaml_path) -> CyclesCase`
  - `discover_inits(run_root) -> list[str]` (sorted YYYYMMDDHH dir names)
  - `window_hours(init_dt, valid_start, valid_end) -> (f1: int, f2: int)`
  - `cycle_eligibility(init_dt, max_fhour, valid_start, valid_end) -> (bool, str)`
  - `cycle_storm_case(ccase, init_str) -> StormCase`

- [ ] **Step 1: Write the failing tests**

Create `analysis/tests/test_cycles.py`:

```python
"""Local unit tests for the cycles product (no Hercules data needed).

Run directly:   python3 analysis/tests/test_cycles.py
Or via pytest:  pytest analysis/tests/test_cycles.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hafs_case import (
    CyclesCase, cycles_from_yaml, discover_inits, window_hours,
    cycle_eligibility, cycle_storm_case, from_yaml,
)


# ---------------------------------------------------------------------------
# Window math
# ---------------------------------------------------------------------------

def test_window_hours():
    f1, f2 = window_hours(datetime(2024, 9, 24, 0),
                          datetime(2024, 9, 26, 0),
                          datetime(2024, 9, 28, 0))
    assert (f1, f2) == (48, 96)
    # Init exactly at the window start -> f1 == 0.
    f1, f2 = window_hours(datetime(2024, 9, 26, 0),
                          datetime(2024, 9, 26, 0),
                          datetime(2024, 9, 28, 0))
    assert (f1, f2) == (0, 48)


def test_cycle_eligibility():
    vs, ve = datetime(2024, 9, 26, 0), datetime(2024, 9, 28, 0)
    ok, reason = cycle_eligibility(datetime(2024, 9, 24, 0), 126, vs, ve)
    assert ok and reason == ""
    # Init after window start -> ineligible.
    ok, reason = cycle_eligibility(datetime(2024, 9, 26, 6), 126, vs, ve)
    assert not ok and "after the window start" in reason
    # Run too short to reach window end -> ineligible.
    ok, reason = cycle_eligibility(datetime(2024, 9, 24, 0), 36, vs, ve)
    assert not ok and "before the window end" in reason
    # Boundary cases are eligible: init == valid_start, end == valid_end.
    assert cycle_eligibility(vs, 48, vs, ve)[0]
    assert cycle_eligibility(datetime(2024, 9, 26, 0), 48, vs, ve)[0]


# ---------------------------------------------------------------------------
# Init discovery
# ---------------------------------------------------------------------------

def test_discover_inits():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("2024092412", "2024092400", "notacycle", "20240924"):
            (root / name).mkdir()
        (root / "2024092418").write_text("a file, not a dir")
        assert discover_inits(root) == ["2024092400", "2024092412"]


def test_discover_inits_missing_root():
    try:
        discover_inits(Path("/nonexistent/dir/xyz"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# cycles_from_yaml
# ---------------------------------------------------------------------------

_CYCLES_YAML = """\
run_root: {root}
valid_start: 2024092600
valid_end: 2024092800
storm_name: Hurricane Helene
domain: [15.0, 42.0, -100.0, -60.0]
mask_radius_km: 500
out_dir: {out}
"""


def test_cycles_from_yaml_defaults_and_slug():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "helene_hfsa_cycles.yaml"
        yml.write_text(_CYCLES_YAML.format(root=tmp, out=tmp))
        cc = cycles_from_yaml(yml)
        assert cc.valid_start == datetime(2024, 9, 26, 0)
        assert cc.valid_end == datetime(2024, 9, 28, 0)
        assert cc.inits is None
        assert cc.ets_threshold_mm == 25.0
        assert cc.thresholds_mm[0] == 1
        assert cc.case_slug == "helene_hfsa_cycles"
        assert cc.output_slug == "helene_hfsa_cycles_2024092600_2024092800"
        glat, glon = cc.fixed_grid()
        assert glat.shape == glon.shape and glat.ndim == 2


def test_cycles_from_yaml_rejects_bad_window():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "bad.yaml"
        yml.write_text(_CYCLES_YAML.format(root=tmp, out=tmp).replace(
            "valid_end: 2024092800", "valid_end: 2024092600"))
        try:
            cycles_from_yaml(yml)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "valid_end" in str(e)


def test_cycles_from_yaml_rejects_per_init_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "case.yaml"
        yml.write_text(f"run_dir: {tmp}\ninit: 2024092400\n")
        try:
            cycles_from_yaml(yml)
            assert False, "expected KeyError"
        except KeyError as e:
            assert "run_root" in str(e)


def test_from_yaml_rejects_cycles_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "cyc.yaml"
        yml.write_text(_CYCLES_YAML.format(root=tmp, out=tmp))
        try:
            from_yaml(yml)
            assert False, "expected KeyError"
        except KeyError as e:
            assert "cycles" in str(e)


# ---------------------------------------------------------------------------
# cycle_storm_case
# ---------------------------------------------------------------------------

# Minimal 2-fix atcfunix (cols: basin, cy, init, technum, tech, tau, lat,
# lon, vmax, mslp, ty ... — parse_atcfunix needs >= 8 columns).
_ATCF = (
    "AL, 09, 2024092400, 03, HFSA, 000, 168N, 832W, 45, 1002, TS\n"
    "AL, 09, 2024092400, 03, HFSA, 048, 250N, 840W, 90, 960, HU\n"
)


def _tiny_cycles_case(root, out):
    return CyclesCase(
        run_root=Path(root),
        valid_start=datetime(2024, 9, 26, 0),
        valid_end=datetime(2024, 9, 28, 0),
        storm_name="Testorm", model_label="HAFS-A",
        domain=(0.0, 1.0, 0.0, 1.0), grid_res=0.5,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1], ets_threshold_mm=1.0,
        out_dir=Path(out), mrms_cache_dir=Path("/tmp"),
        stage4_cache_dir=Path("/tmp"), inits=None,
        case_slug="testcycles",
    )


def test_cycle_storm_case_builds_from_run_root():
    with tempfile.TemporaryDirectory() as tmp:
        cyc_dir = Path(tmp) / "2024092400"
        cyc_dir.mkdir()
        (cyc_dir / "storm09l.2024092400.hfsa.trak.atcfunix").write_text(_ATCF)
        cc = _tiny_cycles_case(tmp, tmp)
        case = cycle_storm_case(cc, "2024092400")
        assert case.run_dir == cyc_dir
        assert case.init_dt == datetime(2024, 9, 24, 0)
        assert case.init_str == "2024092400"
        assert case.storm_name == "Testorm"
        assert len(case.track) == 2
        assert case.mask_radius_km == 500.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 analysis/tests/test_cycles.py`
Expected: FAIL with `ImportError: cannot import name 'CyclesCase' from 'hafs_case'`

- [ ] **Step 3: Implement in `analysis/hafs_case.py`**

3a. Add `make_fixed_grid` as a module-level function (place it right above the `StormCase` dataclass) and make `StormCase.fixed_grid` delegate to it:

```python
def make_fixed_grid(domain, grid_res):
    """Fixed lat/lon verification/plot mesh from a domain + resolution."""
    lat_min, lat_max, lon_min, lon_max = domain
    fixed_lons = np.arange(lon_min, lon_max + grid_res, grid_res)
    fixed_lats = np.arange(lat_min, lat_max + grid_res, grid_res)
    return np.meshgrid(fixed_lons, fixed_lats)[::-1]  # (grid_lat, grid_lon)
```

Replace the body of `StormCase.fixed_grid` with:

```python
    def fixed_grid(self):
        """Fixed lat/lon verification/plot mesh from domain + grid_res."""
        return make_fixed_grid(self.domain, self.grid_res)
```

3b. In `from_yaml`, replace the `run_dir` requirement check:

```python
    if "run_dir" not in cfg:
        if "run_root" in cfg:
            raise KeyError(
                f"{yaml_path} looks like a cycles YAML (has 'run_root') — "
                f"run it with the 'cycles' command."
            )
        raise KeyError(f"'run_dir' is required in {yaml_path}")
```

3c. Append at the end of the file (after `from_yaml`):

```python
# =============================================================================
# Cycle comparison: config + window math
# =============================================================================

_INIT_DIR_RE = re.compile(r"^\d{10}$")


@dataclass
class CyclesCase:
    """One storm x model across initializations, scored on a common window."""
    run_root: Path
    valid_start: datetime
    valid_end: datetime
    storm_name: str
    model_label: str
    domain: tuple          # (lat_min, lat_max, lon_min, lon_max)
    grid_res: float
    mask_radius_km: float
    display_radius_km: float
    thresholds_mm: list
    ets_threshold_mm: float
    out_dir: Path
    mrms_cache_dir: Path
    stage4_cache_dir: Path
    inits: list            # explicit YYYYMMDDHH strings, or None to discover
    case_slug: str

    def fixed_grid(self):
        """Fixed lat/lon verification/plot mesh from domain + grid_res."""
        return make_fixed_grid(self.domain, self.grid_res)

    @property
    def output_slug(self):
        """YAML stem + window, so different windows never overwrite output."""
        return (f"{self.case_slug}_{self.valid_start:%Y%m%d%H}_"
                f"{self.valid_end:%Y%m%d%H}")


def discover_inits(run_root):
    """Sorted YYYYMMDDHH-named subdirectories of run_root."""
    root = Path(run_root)
    if not root.is_dir():
        raise FileNotFoundError(f"run_root does not exist: {root}")
    inits = []
    for p in root.iterdir():
        if not (p.is_dir() and _INIT_DIR_RE.fullmatch(p.name)):
            continue
        try:
            datetime.strptime(p.name, "%Y%m%d%H")
        except ValueError:
            continue
        inits.append(p.name)
    return sorted(inits)


def window_hours(init_dt, valid_start, valid_end):
    """(f1, f2): the common valid window as forecast hours of one cycle."""
    f1 = (valid_start - init_dt).total_seconds() / 3600.0
    f2 = (valid_end - init_dt).total_seconds() / 3600.0
    return int(round(f1)), int(round(f2))


def cycle_eligibility(init_dt, max_fhour, valid_start, valid_end):
    """(eligible, reason). Eligible iff the run fully covers the window."""
    if init_dt > valid_start:
        return False, (f"init {init_dt:%Y-%m-%d %HZ} is after the window "
                       f"start {valid_start:%Y-%m-%d %HZ}")
    run_end = init_dt + timedelta(hours=max_fhour)
    if run_end < valid_end:
        return False, (f"run ends {run_end:%Y-%m-%d %HZ}, before the "
                       f"window end {valid_end:%Y-%m-%d %HZ}")
    return True, ""


def cycles_from_yaml(yaml_path):
    """Load a CyclesCase from a cycles YAML (run_root/valid_start/valid_end
    and domain required)."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if "run_root" not in cfg:
        hint = (" This looks like a per-init case YAML (has 'run_dir'); "
                "the 'cycles' command needs a cycles YAML with 'run_root'."
                if "run_dir" in cfg else "")
        raise KeyError(
            f"'run_root' is required in a cycles YAML ({yaml_path}).{hint}")
    for key in ("valid_start", "valid_end", "domain"):
        if key not in cfg:
            raise KeyError(f"'{key}' is required in a cycles YAML "
                           f"({yaml_path})")
    valid_start = datetime.strptime(str(cfg["valid_start"]), "%Y%m%d%H")
    valid_end = datetime.strptime(str(cfg["valid_end"]), "%Y%m%d%H")
    if valid_end <= valid_start:
        raise ValueError(
            f"valid_end must be after valid_start in {yaml_path}")
    run_root = Path(cfg["run_root"])
    out_dir = (Path(cfg["out_dir"]) if cfg.get("out_dir")
               else Path("analysis/output") / yaml_path.stem)
    return CyclesCase(
        run_root=run_root,
        valid_start=valid_start,
        valid_end=valid_end,
        storm_name=cfg.get("storm_name", "Storm"),
        model_label=(cfg["model_label"] if "model_label" in cfg
                     else detect_model(run_root)),
        domain=tuple(cfg["domain"]),
        grid_res=float(cfg.get("grid_res", 0.05)),
        mask_radius_km=float(cfg.get("mask_radius_km", 500.0)),
        display_radius_km=float(cfg.get("display_radius_km", 750.0)),
        thresholds_mm=cfg.get("thresholds_mm", list(_DEFAULT_THRESHOLDS)),
        ets_threshold_mm=float(cfg.get("ets_threshold_mm", 25.0)),
        out_dir=out_dir,
        mrms_cache_dir=Path(cfg.get("mrms_cache_dir", "/tmp/mrms_cache")),
        stage4_cache_dir=Path(cfg.get("stage4_cache_dir",
                                      "/tmp/stage4_cache")),
        inits=[str(i) for i in cfg["inits"]] if cfg.get("inits") else None,
        case_slug=yaml_path.stem,
    )


def cycle_storm_case(ccase, init_str):
    """StormCase for one cycle of a CyclesCase (run_dir = run_root/<init>).

    Track comes from the cycle's own .atcfunix; grid/mask/output settings
    are inherited from the CyclesCase so every cycle verifies identically.
    """
    run_dir = ccase.run_root / init_str
    atcf_path = find_atcfunix(run_dir)
    _, _, track = parse_atcfunix(atcf_path)
    if not track:
        raise ValueError(f"Parsed 0 track fixes from {atcf_path}")
    return StormCase(
        run_dir=run_dir,
        init_dt=datetime.strptime(init_str, "%Y%m%d%H"),
        storm_name=ccase.storm_name,
        model_label=ccase.model_label,
        domain=ccase.domain,
        grid_res=ccase.grid_res,
        mask_radius_km=ccase.mask_radius_km,
        display_radius_km=ccase.display_radius_km,
        thresholds_mm=ccase.thresholds_mm,
        out_dir=ccase.out_dir,
        mrms_cache_dir=ccase.mrms_cache_dir,
        stage4_cache_dir=ccase.stage4_cache_dir,
        fhours_filter=None,
        track=track,
        case_slug=ccase.case_slug,
        init_str=init_str,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 analysis/tests/test_cycles.py` — Expected: all PASS.
Run: `pytest analysis/tests/test_cycles.py -v` — Expected: all PASS.
Run: `pytest analysis/tests/ -v` — Expected: no existing test broken (especially `test_ets_full.py`, which exercises `StormCase.fixed_grid` indirectly).

- [ ] **Step 5: Report (no commit)**

Do NOT commit (project rule). Re-state the changed/created files and the test summary in your report.

---

### Task 2: Windowed observation loaders (`ets_score.py`, `parent_qpf.py`)

**Files:**
- Modify: `analysis/ets_score.py` (around `build_mrms_total`, ets_score.py:61-81)
- Modify: `analysis/parent_qpf.py` (around `default_parent_path` and `stage4_total`, parent_qpf.py:70-223)
- Modify: `analysis/tests/test_cycles.py` (append tests)

**Interfaces:**
- Consumes: `load_mrms_hour`, `regrid_mrms_to_fixed`, `ensure_stage4_files`, `index_stage4_24h_conus`, `read_stage4`, `haversine_km` (all existing).
- Produces:
  - `ets_score.build_mrms_total_window(valid_start, valid_end, mrms_cache_dir, grid_lat, grid_lon) -> 2-D array`
  - `parent_qpf.parent_path_at_fhour(case, fhour) -> Path | None`
  - `parent_qpf.stage4_sum_days(cache_dir, start_dt, end_dt) -> (lat2d, lon2d, total, used_keys)`
  - `parent_qpf.stage4_label(used_keys) -> str`
  - `parent_qpf.stage4_total_window(cache_dir, valid_start, valid_end, track_points, radius_km) -> (lat2d, lon2d, total, label)` — `track_points` is a list of `(lat, lon)` tuples; returns `(None, None, None, None)` when unavailable.
- Behavior-preserving: `build_mrms_total(case, max_fhour, grid_lat, grid_lon)` and `stage4_total(case, end_fhour)` keep their exact signatures and outputs.

- [ ] **Step 1: Write the failing tests**

Append to `analysis/tests/test_cycles.py` (before `_run_all`):

```python
# ---------------------------------------------------------------------------
# Windowed observation loaders (Task 2)
# ---------------------------------------------------------------------------

def test_build_mrms_total_window_sums_requested_hours():
    """Patch the hour loader; check which hour-stamps are requested and
    that the window total is the plain sum, regridded."""
    import ets_score

    requested = []

    def fake_load(s3, hour_end_dt, cache_dir):
        requested.append(hour_end_dt)
        lat = np.linspace(0.0, 1.0, 5)
        lon = np.linspace(0.0, 1.0, 5)
        return lat, lon, np.ones((5, 5))

    orig = ets_score.load_mrms_hour
    ets_score.load_mrms_hour = fake_load
    try:
        glat, glon = np.meshgrid(np.linspace(0.2, 0.8, 3),
                                 np.linspace(0.2, 0.8, 3))[::-1]
        total = ets_score.build_mrms_total_window(
            datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 3),
            Path("/tmp"), glat, glon)
    finally:
        ets_score.load_mrms_hour = orig
    # Hour files are stamped by hour END: 01Z, 02Z, 03Z — not 00Z.
    assert requested == [datetime(2024, 9, 26, 1),
                         datetime(2024, 9, 26, 2),
                         datetime(2024, 9, 26, 3)]
    assert np.allclose(total, 3.0)


def test_parent_path_at_fhour():
    from parent_qpf import parent_path_at_fhour
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        names = ["storm09l.2024092400.hfsa.parent.atm.f048.grb2",
                 "storm09l.2024092400.hfsa.parent.atm.f096.grb2"]
        for n in names:
            (run_dir / n).write_text("")
        case = _stub_case(run_dir)
        assert parent_path_at_fhour(case, 48).name == names[0]
        assert parent_path_at_fhour(case, 96).name == names[1]
        assert parent_path_at_fhour(case, 72) is None


def _stub_case(run_dir):
    """Minimal StormCase for glob-based tests."""
    from hafs_case import StormCase
    return StormCase(
        run_dir=Path(run_dir), init_dt=datetime(2024, 9, 24, 0),
        storm_name="Testorm", model_label="HAFS-A",
        domain=(0.0, 1.0, 0.0, 1.0), grid_res=0.5,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1], out_dir=Path(run_dir),
        mrms_cache_dir=Path("/tmp"), stage4_cache_dir=Path("/tmp"),
        fhours_filter=None,
        track=[(datetime(2024, 9, 24, 0), 0.5, 0.5)],
        case_slug="testcase", init_str="2024092400",
    )


def test_stage4_label():
    from parent_qpf import stage4_label
    assert stage4_label(["20240926", "20240927"]) == \
        "~09-25 12Z – 09-27 12Z (2×24h)"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 analysis/tests/test_cycles.py`
Expected: FAIL with `AttributeError: module 'ets_score' has no attribute 'build_mrms_total_window'` (the first new test).

- [ ] **Step 3: Implement**

3a. In `analysis/ets_score.py`, replace `build_mrms_total` (ets_score.py:61-81) with the windowed function plus a delegating wrapper:

```python
def build_mrms_total_window(valid_start, valid_end, mrms_cache_dir,
                            grid_lat, grid_lon):
    """MRMS 1H QPE summed over (valid_start, valid_end], then regridded.

    Absolute-time core of build_mrms_total: MRMS hour files are stamped by
    hour END, so the file at valid_start + 1h is the first inside the
    window. Raises RuntimeError when no hour can be loaded.
    """
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    n_hours = int(round((valid_end - valid_start).total_seconds() / 3600))
    mrms_sum = None
    mlat = mlon = None
    for h in range(1, n_hours + 1):
        t = valid_start + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, mrms_cache_dir)
            if mrms_sum is None:
                mlat, mlon = lat, lon
                mrms_sum = np.zeros_like(data)
            mrms_sum += data
            if h % 12 == 0 or h == n_hours:
                print(f"  MRMS h{h:03d}/{n_hours} ({t:%Y-%m-%d %HZ})")
        except Exception as e:
            print(f"  MRMS h{h:03d} unavailable: {e}")
    if mrms_sum is None:
        raise RuntimeError("No MRMS hours could be loaded.")
    return regrid_mrms_to_fixed(mlat, mlon, mrms_sum, grid_lat, grid_lon)


def build_mrms_total(case, max_fhour, grid_lat, grid_lon):
    """Accumulate MRMS 1H QPE over 1..max_fhour on its native grid, then regrid."""
    return build_mrms_total_window(
        case.init_dt, case.init_dt + timedelta(hours=max_fhour),
        case.mrms_cache_dir, grid_lat, grid_lon)
```

3b. In `analysis/parent_qpf.py`, add after `default_parent_path` (parent_qpf.py:80):

```python
def parent_path_at_fhour(case, fhour):
    """The parent.atm file for one exact forecast hour, or None if absent."""
    suffix = f".f{fhour:03d}.grb2"
    hits = [p for p in sorted(case.run_dir.glob(case.parent_glob()))
            if p.name.endswith(suffix)]
    return hits[0] if hits else None
```

3c. In `analysis/parent_qpf.py`, refactor `stage4_total` (parent_qpf.py:163-223). Extract the day-summing core and the label into their own functions, keep `stage4_total`'s masking/diagnostics behavior identical, and add the windowed variant:

```python
def stage4_sum_days(cache_dir, start_dt, end_dt):
    """Sum the daily CONUS 24h Stage IV files whose date the window touches.

    Returns (lat2d, lon2d, total_mm, used_keys) with the total UNMASKED;
    (None, None, None, []) when nothing is available.
    """
    ensure_stage4_files(start_dt, end_dt, cache_dir)
    idx = index_stage4_24h_conus(cache_dir)
    if not idx:
        return None, None, None, []
    total = lat2d = lon2d = None
    used = []
    day = start_dt.date()
    while day <= end_dt.date():
        key = day.strftime("%Y%m%d")
        path = idx.get(key)
        if path is not None:
            try:
                lat, lon, data = read_stage4(path)
            except Exception as e:
                print(f"  Stage IV read failed {key}: {e}")
                day += timedelta(days=1)
                continue
            if total is None:
                lat2d, lon2d = lat, lon
                total = np.zeros_like(data)
            total += data
            used.append(key)
            # Per-file diagnostic: where is the daily max, how many extreme px?
            j, i = np.unravel_index(np.argmax(data), data.shape)
            print(f"  {key} 24h: max {data[j, i]:6.0f} mm at "
                  f"({lat[j, i]:.2f}, {lon[j, i]:.2f})  "
                  f">300mm:{int((data > 300).sum())}  "
                  f">600mm:{int((data > 600).sum())}")
        day += timedelta(days=1)
    if total is None:
        return None, None, None, []
    return lat2d, lon2d, total, used


def stage4_label(used):
    """Window label from the used day keys (24h file D covers 12Z(D-1)->12Z(D))."""
    d0 = datetime.strptime(used[0], "%Y%m%d") - timedelta(hours=12)
    d1 = datetime.strptime(used[-1], "%Y%m%d")
    return f"~{d0:%m-%d %HZ} – {d1:%m-%d 12Z} ({len(used)}×24h)"


def stage4_total(case, end_fhour):
    """Sum the daily CONUS 24h Stage IV files spanning the forecast window,
    then mask the total to the full-track TC swath (same footprint as HAFS).

    24h files are valid 12Z->12Z so they don't align exactly with the 00Z-init
    0->end_fhour window; we sum every day the event touches.  Returns
    (lat2d, lon2d, total_mm, label) or (None, None, None, None) if unavailable.
    """
    valid_end = case.init_dt + timedelta(hours=end_fhour)
    lat2d, lon2d, total, used = stage4_sum_days(
        case.stage4_cache_dir, case.init_dt, valid_end)
    if total is None:
        return None, None, None, None

    # Mask the summed total to the union of circles along the track.
    swath = np.zeros(lat2d.shape, dtype=bool)
    for h in range(0, end_fhour + 1):
        tlat, tlon = case.position_at(case.init_dt + timedelta(hours=h))
        swath |= haversine_km(tlat, tlon, lat2d, lon2d) <= case.display_radius_km
    total = np.where(swath, total, 0.0)

    # Total diagnostic: locate the event-total max so you can judge if it's a
    # real accumulation or an artifact (e.g. a stuck pixel across days).
    j, i = np.unravel_index(np.argmax(total), total.shape)
    print(f"  Stage IV total: max {total[j, i]:.0f} mm at "
          f"({lat2d[j, i]:.2f}, {lon2d[j, i]:.2f})  "
          f">500mm:{int((total > 500).sum())}  >800mm:{int((total > 800).sum())}")

    return lat2d, lon2d, total, stage4_label(used)


def stage4_total_window(cache_dir, valid_start, valid_end, track_points,
                        radius_km):
    """Stage IV touched-days total for an absolute window, masked to the
    union of circles of radius_km around track_points [(lat, lon), ...].

    Returns (lat2d, lon2d, total_mm, label) or (None, None, None, None).
    """
    lat2d, lon2d, total, used = stage4_sum_days(cache_dir, valid_start,
                                                valid_end)
    if total is None:
        return None, None, None, None
    swath = np.zeros(lat2d.shape, dtype=bool)
    for tlat, tlon in track_points:
        swath |= haversine_km(tlat, tlon, lat2d, lon2d) <= radius_km
    total = np.where(swath, total, 0.0)
    return lat2d, lon2d, total, stage4_label(used)
```

(The old inline day-summing loop, per-file diagnostic prints, and label construction inside `stage4_total` are deleted — they now live in `stage4_sum_days` / `stage4_label`. `datetime` is already imported in parent_qpf.py.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 analysis/tests/test_cycles.py` — Expected: all PASS.
Run: `pytest analysis/tests/ -v` — Expected: all pass; `test_ets_full.py` and `test_rmse_scatter.py` confirm the `build_mrms_total` / `stage4_total` refactors broke nothing.

- [ ] **Step 5: Report (no commit)**

Do NOT commit. Report changed files + test summary.

---

### Task 3: Windowed forecast fields + union swath (`cycles.py`)

**Files:**
- Create: `analysis/cycles.py`
- Modify: `analysis/tests/test_cycles.py` (append tests)

**Interfaces:**
- Consumes: Task 1's `cycles_from_yaml`, `cycle_storm_case`, `discover_inits`, `window_hours`, `cycle_eligibility`; Task 2's `build_mrms_total_window`, `stage4_total_window`, `parent_path_at_fhour`; existing `discover_files`, `hafs_event_total`, `haversine_km`, `read_hafs_tp_records` (hafs_common), `pick_cumulative_record` (parent_qpf), `regrid_2d_to_fixed` (ets_full).
- Produces (Task 4 relies on):
  - `window_track_points(case, valid_start, valid_end) -> list[(lat, lon)]`
  - `union_swath(track_points, radius_km, grid_lat, grid_lon) -> bool 2-D`
  - `filter_window_pairs(file_pairs, f1, f2) -> list[(fhour, path)]`
  - `nest_window_total(case, f1, f2, grid_lat, grid_lon) -> 2-D array`
  - `parent_window_total(case, f1, f2, grid_lat, grid_lon) -> 2-D array`
  - `build_cycle_fields(ccase) -> dict` with keys `grid_lat, grid_lon, mrms_win, stage4_win (None when unavailable), s4_label, swath, cycles` where `cycles` is a list of dicts `{init_str, init_dt, f1, f2, nest_win, parent_win}`

- [ ] **Step 1: Write the failing tests**

Append to `analysis/tests/test_cycles.py` (before `_run_all`):

```python
# ---------------------------------------------------------------------------
# Windowed forecast fields + union swath (Task 3)
# ---------------------------------------------------------------------------

def test_filter_window_pairs_boundaries():
    from cycles import filter_window_pairs
    pairs = [(h, Path(f"f{h:03d}")) for h in (42, 48, 51, 90, 96, 99)]
    kept = filter_window_pairs(pairs, 48, 96)
    # f1 exclusive (a bucket ENDING at 48 holds pre-window rain),
    # f2 inclusive (the bucket ending at 96 is the window's last rain).
    assert [h for h, _ in kept] == [51, 90, 96]


def test_union_swath_is_union_of_single_masks():
    from cycles import union_swath
    glat, glon = np.meshgrid(np.linspace(0, 10, 21),
                             np.linspace(0, 10, 21))[::-1]
    pts_a = [(2.0, 2.0)]
    pts_b = [(8.0, 8.0)]
    m_a = union_swath(pts_a, 150.0, glat, glon)
    m_b = union_swath(pts_b, 150.0, glat, glon)
    m_ab = union_swath(pts_a + pts_b, 150.0, glat, glon)
    assert m_a.any() and m_b.any()
    assert not (m_a & m_b).any()          # disjoint circles at 150 km
    assert np.array_equal(m_ab, m_a | m_b)


def test_window_track_points_hourly_inclusive():
    from cycles import window_track_points
    case = _stub_case(Path("."))
    pts = window_track_points(case, datetime(2024, 9, 26, 0),
                              datetime(2024, 9, 26, 6))
    assert len(pts) == 7                  # 00Z..06Z inclusive, hourly
    # Single-fix track -> position clamps to that fix everywhere.
    assert all(p == (0.5, 0.5) for p in pts)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 analysis/tests/test_cycles.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'cycles'`

- [ ] **Step 3: Create `analysis/cycles.py`**

```python
"""
Cycle comparison for HAFS QPF: score every initialization of one storm on a
common valid window against the same observations over a shared
union-of-tracks footprint, so the only thing changing between cycles is
lead time.

Produces a metrics-vs-init figure (RMSE / bias / ETS at one threshold),
nest-QPF map small-multiples with the observed MRMS panel, and one
long-format CSV.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/run.py storms/helene_hfsa_cycles.yaml cycles
"""

import sys
import csv
from pathlib import Path
from datetime import timedelta

# Make sibling analysis modules importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from hafs_common import (
    QPF_LEVELS, discover_files, hafs_event_total, haversine_km,
    read_hafs_tp_records,
)
from hafs_case import (
    cycles_from_yaml, cycle_storm_case, discover_inits, window_hours,
    cycle_eligibility,
)
from ets_score import contingency_scores, build_mrms_total_window
from ets_full import regrid_2d_to_fixed, _OBS_COLOR, _FCST_STYLE
from parent_qpf import (
    parent_path_at_fhour, pick_cumulative_record, stage4_total_window,
    qpf_cmap,
)
from skill_metrics import continuous_scores
from rmse_scatter import valid_points


# =============================================================================
# Union footprint
# =============================================================================

def window_track_points(case, valid_start, valid_end):
    """Hourly (lat, lon) track positions of one cycle inside the window,
    endpoints inclusive."""
    pts = []
    t = valid_start
    while t <= valid_end:
        pts.append(case.position_at(t))
        t += timedelta(hours=1)
    return pts


def union_swath(track_points, radius_km, grid_lat, grid_lon):
    """Boolean mask: grid points within radius_km of ANY (lat, lon) point."""
    swath = np.zeros(grid_lat.shape, dtype=bool)
    for tlat, tlon in track_points:
        swath |= haversine_km(tlat, tlon, grid_lat, grid_lon) <= radius_km
    return swath


# =============================================================================
# Windowed forecast fields
# =============================================================================

def filter_window_pairs(file_pairs, f1, f2):
    """Nest (fhour, path) pairs whose bucket falls inside (f1, f2].

    Each storm.atm file's per-interval bucket ENDS at its fhour, so files
    with f1 < fhour <= f2 together hold exactly the window's rain.
    """
    return [(h, p) for h, p in file_pairs if f1 < h <= f2]


def nest_window_total(case, f1, f2, grid_lat, grid_lon):
    """Nest QPF accumulated over window hours (f1, f2] on the fixed mesh.

    Sums the per-interval buckets exactly as hafs_event_total does for the
    full event (its incremental mode). Raises RuntimeError when the window
    has no nest files.
    """
    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    window_pairs = filter_window_pairs(file_pairs, f1, f2)
    if not window_pairs:
        raise RuntimeError(
            f"no nest files in f{f1:03d}-f{f2:03d} for init {case.init_str}")
    total, _ = hafs_event_total(window_pairs, grid_lat, grid_lon)
    return total


def parent_window_total(case, f1, f2, grid_lat, grid_lon):
    """Parent QPF for the window: cumulative APCP at f2 minus at f1.

    The 6-km parent domain is fixed, so its 0->fhour cumulative record is
    geographically valid and the window total is a clean difference.
    Interpolation noise can leave tiny negatives; those are floored at 0
    (NaN outside the parent hull is preserved). Raises RuntimeError when a
    needed parent file or record is missing.
    """
    def cumulative_at(fh):
        path = parent_path_at_fhour(case, fh)
        if path is None:
            raise RuntimeError(
                f"no parent.atm file at f{fh:03d} for init {case.init_str}")
        rec = pick_cumulative_record(read_hafs_tp_records(path))
        if rec is None:
            raise RuntimeError(
                f"no APCP record in parent f{fh:03d} for init {case.init_str}")
        return regrid_2d_to_fixed(rec["lats"], rec["lons"], rec["data"],
                                  grid_lat, grid_lon)

    total = cumulative_at(f2)
    if f1 > 0:
        total = total - cumulative_at(f1)
    return np.where(total < 0, 0.0, total)


# =============================================================================
# Field building (once per cycles case)
# =============================================================================

def build_cycle_fields(ccase):
    """Build everything the cycles product scores and plots.

    Returns a dict: grid_lat, grid_lon, mrms_win, stage4_win (None when
    Stage IV is unavailable), s4_label, swath (shared union mask), and
    cycles — a list of dicts {init_str, init_dt, f1, f2, nest_win,
    parent_win}, one per surviving cycle. Raises RuntimeError when no
    cycle is eligible or every eligible cycle fails field extraction.
    """
    init_strs = ccase.inits or discover_inits(ccase.run_root)
    if not init_strs:
        raise RuntimeError(
            f"No YYYYMMDDHH cycle directories under {ccase.run_root} "
            f"and no 'inits' list given.")
    print(f"Window {ccase.valid_start:%Y-%m-%d %HZ} -> "
          f"{ccase.valid_end:%Y-%m-%d %HZ} | candidate inits: "
          f"{', '.join(init_strs)}")

    grid_lat, grid_lon = ccase.fixed_grid()
    print(f"Fixed grid: {grid_lat.shape[0]}x{grid_lat.shape[1]} "
          f"@ {ccase.grid_res}deg")

    # Pass 1: load cases; keep only cycles that fully cover the window.
    cases = []
    for init_str in init_strs:
        try:
            case = cycle_storm_case(ccase, init_str)
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip {init_str}: {e}")
            continue
        file_pairs = discover_files(case.run_dir, case.storm_glob(), None)
        if not file_pairs:
            print(f"  skip {init_str}: no nest files matching "
                  f"{case.storm_glob()}")
            continue
        max_fhour = file_pairs[-1][0]
        ok, reason = cycle_eligibility(case.init_dt, max_fhour,
                                       ccase.valid_start, ccase.valid_end)
        if not ok:
            print(f"  skip {init_str}: {reason}")
            continue
        cases.append(case)
    if not cases:
        raise RuntimeError(
            f"No eligible cycles for window "
            f"{ccase.valid_start:%Y%m%d%H}->{ccase.valid_end:%Y%m%d%H} "
            f"(inspected: {', '.join(init_strs)})")

    # Shared footprint: union of every eligible cycle's in-window track.
    print("Union verification swath ...")
    all_pts = []
    for case in cases:
        all_pts.extend(window_track_points(case, ccase.valid_start,
                                           ccase.valid_end))
    swath = union_swath(all_pts, ccase.mask_radius_km, grid_lat, grid_lon)
    print(f"  swath: {int(swath.sum()):,} grid points from "
          f"{len(cases)} track(s)")

    # Shared observations (absolute-time, computed once for all cycles).
    print("MRMS window total ...")
    mrms_win = build_mrms_total_window(ccase.valid_start, ccase.valid_end,
                                       ccase.mrms_cache_dir,
                                       grid_lat, grid_lon)
    print("Stage IV window total ...")
    s4_lat, s4_lon, s4_native, s4_label = stage4_total_window(
        ccase.stage4_cache_dir, ccase.valid_start, ccase.valid_end,
        all_pts, ccase.display_radius_km)
    if s4_native is None:
        stage4_win, s4_label = None, "unavailable"
        print("  Stage IV unavailable — scoring MRMS only.")
    else:
        stage4_win = regrid_2d_to_fixed(s4_lat, s4_lon, s4_native,
                                        grid_lat, grid_lon)

    # Per-cycle forecast windows.
    cycles = []
    for case in cases:
        f1, f2 = window_hours(case.init_dt, ccase.valid_start,
                              ccase.valid_end)
        print(f"\nCycle {case.init_str} (window f{f1:03d}-f{f2:03d})")
        try:
            nest_win = nest_window_total(case, f1, f2, grid_lat, grid_lon)
            parent_win = parent_window_total(case, f1, f2,
                                             grid_lat, grid_lon)
        except RuntimeError as e:
            print(f"  skip {case.init_str}: {e}")
            continue
        cycles.append(dict(init_str=case.init_str, init_dt=case.init_dt,
                           f1=f1, f2=f2, nest_win=nest_win,
                           parent_win=parent_win))
    if not cycles:
        raise RuntimeError("Every eligible cycle failed field extraction.")

    return dict(grid_lat=grid_lat, grid_lon=grid_lon, mrms_win=mrms_win,
                stage4_win=stage4_win, s4_label=s4_label, swath=swath,
                cycles=cycles)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 analysis/tests/test_cycles.py` — Expected: all PASS.
Run: `pytest analysis/tests/ -v` — Expected: all pass.

- [ ] **Step 5: Report (no commit)**

Do NOT commit. Report changed files + test summary.

---

### Task 4: Scoring, CSV, figures, and the `cycles` command

**Files:**
- Modify: `analysis/cycles.py` (append)
- Modify: `analysis/run.py`
- Modify: `analysis/tests/test_cycles.py` (append tests)
- Modify: `analysis/tests/test_run.py` (one test)

**Interfaces:**
- Consumes: Task 3's `build_cycle_fields` dict shape; `continuous_scores` (skill_metrics), `contingency_scores` (ets_score), `valid_points` (rmse_scatter), `qpf_cmap` (parent_qpf), `_OBS_COLOR`/`_FCST_STYLE` (ets_full) — all already imported in cycles.py.
- Produces: `compute_cycles(ccase, fields=None)` — the public entry point `run.py` calls.

- [ ] **Step 1: Write the failing tests**

Append to `analysis/tests/test_cycles.py` (before `_run_all`):

```python
# ---------------------------------------------------------------------------
# compute_cycles scoring path (Task 4)
# ---------------------------------------------------------------------------

def _tiny_cycle_fields():
    glat, glon = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))[::-1]
    obs = np.full((4, 4), 10.0)
    return dict(
        grid_lat=glat, grid_lon=glon,
        mrms_win=obs,
        stage4_win=None, s4_label="unavailable",
        swath=np.ones((4, 4), dtype=bool),
        cycles=[
            dict(init_str="2024092400", init_dt=datetime(2024, 9, 24, 0),
                 f1=48, f2=96, nest_win=obs + 2.0, parent_win=obs - 1.0),
            dict(init_str="2024092500", init_dt=datetime(2024, 9, 25, 0),
                 f1=24, f2=72, nest_win=obs.copy(), parent_win=obs.copy()),
        ],
    )


def test_compute_cycles_writes_csv_and_pngs():
    import csv as csvmod
    from cycles import compute_cycles
    with tempfile.TemporaryDirectory() as tmp:
        ccase = _tiny_cycles_case(tmp, tmp)
        compute_cycles(ccase, fields=_tiny_cycle_fields())
        slug = "testcycles_2024092600_2024092800"
        csv_path = Path(tmp) / f"cycles_{slug}.csv"
        metrics_png = Path(tmp) / f"cycles_metrics_{slug}.png"
        maps_png = Path(tmp) / f"cycles_maps_{slug}.png"
        assert csv_path.exists(), "CSV not written"
        assert metrics_png.exists(), "metrics PNG not written"
        assert maps_png.exists(), "maps PNG not written"
        with open(csv_path) as fh:
            rows = list(csvmod.DictReader(fh))
        # 2 cycles x 2 forecasts x 1 obs (Stage IV None) x 1 threshold.
        assert len(rows) == 4
        assert rows[0].keys() == {
            "init", "forecast", "observation", "threshold", "n", "rmse",
            "mae", "bias_mm", "r", "a", "b", "c", "d", "ets", "bias",
            "pod", "far", "csi", "hss"}
        by_key = {(r["init"], r["forecast"]): r for r in rows}
        early_nest = by_key[("2024092400", "nest")]
        early_parent = by_key[("2024092400", "parent")]
        late_nest = by_key[("2024092500", "nest")]
        assert all(r["observation"] == "MRMS" for r in rows)
        # Constant offsets: rmse == |bias_mm| (positive bias = over-forecast).
        assert abs(float(early_nest["rmse"]) - 2.0) < 1e-9
        assert abs(float(early_nest["bias_mm"]) - 2.0) < 1e-9
        assert abs(float(early_parent["rmse"]) - 1.0) < 1e-9
        assert abs(float(early_parent["bias_mm"]) - (-1.0)) < 1e-9
        assert abs(float(late_nest["rmse"]) - 0.0) < 1e-9
        assert int(early_nest["n"]) == 16
        # Perfect >= 1mm coverage everywhere -> ETS-relevant counts: all hits.
        assert int(early_nest["a"]) == 16 and int(early_nest["c"]) == 0
```

Append to `analysis/tests/test_run.py`, next to the existing `test_parse_args_accepts_rmse`:

```python
def test_parse_args_accepts_cycles():
    yaml_path, command = parse_args(["case.yaml", "cycles"])
    assert yaml_path == "case.yaml"
    assert command == "cycles"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 analysis/tests/test_cycles.py`
Expected: FAIL with `ImportError: cannot import name 'compute_cycles' from 'cycles'`
Run: `python3 analysis/tests/test_run.py`
Expected: FAIL (`cycles` rejected by `parse_args` — note this test raises SystemExit, which the standalone runner reports as an error; that is a valid failing state).

- [ ] **Step 3: Implement**

3a. Append to `analysis/cycles.py`:

```python
# =============================================================================
# Scoring + outputs
# =============================================================================

def cycles_caveat(fields, ccase):
    """Figure-footer caveat describing the Stage IV window (or its absence)."""
    if fields["stage4_win"] is None:
        return "Stage IV unavailable — not scored."
    return (f"Stage IV: CONUS-only, 24h 12Z–12Z files summed over touched "
            f"days ({fields['s4_label']}) — window approximates "
            f"{ccase.valid_start:%m-%d %HZ}–{ccase.valid_end:%m-%d %HZ}.")


def plot_metrics(ccase, results, out_path, caveat=""):
    """Metrics vs init time: RMSE, bias, and ETS@headline-threshold panels.

    results: list of dicts {init_str, init_dt, forecast, observation,
    cont, rows} — one per cycle x pair.
    """
    inits = sorted({r["init_dt"] for r in results})
    pairs = sorted({(r["forecast"], r["observation"]) for r in results},
                   key=lambda p: (p[1], p[0]))
    thr = ccase.ets_threshold_mm
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10), sharex=True)

    def series(fname, oname, getter):
        by_init = {r["init_dt"]: r for r in results
                   if r["forecast"] == fname and r["observation"] == oname}
        return [getter(by_init[i]) if i in by_init else np.nan
                for i in inits]

    def ets_at(res):
        for row in res["rows"]:
            if row["threshold"] == thr:
                return row["ets"]
        return np.nan

    panels = [
        (axes[0], lambda res: res["cont"]["rmse"], "RMSE (mm)"),
        (axes[1], lambda res: res["cont"]["bias"], "bias (mm)"),
        (axes[2], ets_at, f"ETS @ {thr:g} mm"),
    ]
    for ax, getter, label in panels:
        for fname, oname in pairs:
            style = _FCST_STYLE.get(fname, dict(ls="-", marker="o"))
            ax.plot(inits, series(fname, oname, getter),
                    color=_OBS_COLOR.get(oname, "gray"), lw=2, **style,
                    label=f"{fname} vs {oname}")
        ax.set_ylabel(label)
        ax.grid(True, ls=":", alpha=0.4)
    axes[1].axhline(0, color="gray", ls=":", lw=0.8)
    axes[0].legend(loc="best", fontsize=9)
    axes[2].set_xlabel("initialization")
    axes[2].set_xticks(inits)
    axes[2].set_xticklabels([i.strftime("%m-%d %HZ") for i in inits],
                            rotation=45, ha="right")
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} QPF by initialization\n"
        f"valid {ccase.valid_start:%Y-%m-%d %HZ} – "
        f"{ccase.valid_end:%Y-%m-%d %HZ} | union TC swath "
        f"≤{ccase.mask_radius_km:.0f} km")
    if caveat:
        fig.text(0.5, -0.01, caveat, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_maps(ccase, fields, out_path):
    """Small-multiple nest-QPF maps per cycle + observed MRMS panel.

    Shared color scale and extent; the union verification swath is
    outlined on every panel. Wraps at 4 columns.
    """
    panels = [(f"init {c['init_dt']:%m-%d %HZ}", c["nest_win"])
              for c in fields["cycles"]]
    panels.append(("MRMS observed", fields["mrms_win"]))
    n = len(panels)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    lat_min, lat_max, lon_min, lon_max = ccase.domain
    cmap, norm = qpf_cmap()
    grid_lat, grid_lon = fields["grid_lat"], fields["grid_lon"]

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.6 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    flat = axes.ravel()
    cf = None
    for ax, (title, data) in zip(flat, panels):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        cf = ax.contourf(grid_lon, grid_lat,
                         np.nan_to_num(data, nan=0.0),
                         levels=QPF_LEVELS, cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), extend="max")
        ax.contour(grid_lon, grid_lat, fields["swath"].astype(float),
                   levels=[0.5], colors="k", linewidths=1.0,
                   transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=10)
    for ax in flat[n:]:
        ax.set_visible(False)
    if cf is not None:
        fig.colorbar(cf, ax=axes, label="Accumulated Precipitation (mm)",
                     ticks=QPF_LEVELS, shrink=0.7, fraction=0.02)
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} nest QPF by "
        f"initialization\nvalid {ccase.valid_start:%Y-%m-%d %HZ} – "
        f"{ccase.valid_end:%Y-%m-%d %HZ} | swath outline "
        f"≤{ccase.mask_radius_km:.0f} km", y=1.02)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_cycles(ccase, fields=None):
    if fields is None:
        fields = build_cycle_fields(ccase)
    swath = fields["swath"]
    observations = [("MRMS", fields["mrms_win"])]
    if fields["stage4_win"] is not None:
        observations.append(("Stage IV", fields["stage4_win"]))

    # Make sure the headline ETS threshold is actually scored.
    thresholds = list(ccase.thresholds_mm)
    if ccase.ets_threshold_mm not in thresholds:
        thresholds = sorted(set(thresholds) | {ccase.ets_threshold_mm})

    results = []
    print("\n" + "=" * 84)
    for cyc in fields["cycles"]:
        for fname, fgrid in (("parent", cyc["parent_win"]),
                             ("nest", cyc["nest_win"])):
            for oname, ogrid in observations:
                fcst, obs = valid_points(fgrid, ogrid, swath)
                cont = continuous_scores(fcst, obs)
                rows = [contingency_scores(fcst, obs, thr)
                        for thr in thresholds]
                results.append(dict(init_str=cyc["init_str"],
                                    init_dt=cyc["init_dt"],
                                    forecast=fname, observation=oname,
                                    cont=cont, rows=rows))
                print(f"{cyc['init_str']} {fname:>7} vs {oname:<9} "
                      f"n={cont['n']:>9,} RMSE={cont['rmse']:>7.2f} "
                      f"MAE={cont['mae']:>7.2f} bias={cont['bias']:>+7.2f} "
                      f"r={cont['r']:>5.2f}")
    print("=" * 84)

    ccase.out_dir.mkdir(parents=True, exist_ok=True)
    slug = ccase.output_slug
    out_csv = ccase.out_dir / f"cycles_{slug}.csv"

    fieldnames = ["init", "forecast", "observation", "threshold", "n",
                  "rmse", "mae", "bias_mm", "r", "a", "b", "c", "d",
                  "ets", "bias", "pod", "far", "csi", "hss"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            cont = res["cont"]
            for r in res["rows"]:
                w.writerow({"init": res["init_str"],
                            "forecast": res["forecast"],
                            "observation": res["observation"],
                            "n": cont["n"], "rmse": cont["rmse"],
                            "mae": cont["mae"], "bias_mm": cont["bias"],
                            "r": cont["r"], **r})
    print(f"\nSaved table: {out_csv}")

    caveat = cycles_caveat(fields, ccase)
    print(caveat)
    out_metrics = ccase.out_dir / f"cycles_metrics_{slug}.png"
    out_maps = ccase.out_dir / f"cycles_maps_{slug}.png"
    plot_metrics(ccase, results, out_metrics, caveat=caveat)
    print(f"Saved plot : {out_metrics}")
    plot_maps(ccase, fields, out_maps)
    print(f"Saved plot : {out_maps}")


if __name__ == "__main__":
    compute_cycles(cycles_from_yaml(sys.argv[1]))
```

3b. In `analysis/run.py`:

- Docstring usage line becomes `python analysis/run.py <case.yaml> [parent|ets|rmse|cycles|all|compare|replot]` and the product list gains: `  cycles  per-initialization comparison on a common valid window (takes a cycles YAML)`
- `COMMANDS = ("parent", "ets", "rmse", "cycles", "all", "compare", "replot")`
- The `parse_args` usage message becomes `usage: run.py <case.yaml> [parent|ets|rmse|cycles|all|compare|replot]`
- In `main`, add a branch BEFORE the `from_yaml` load (mirroring the compare/replot branch):

```python
    if command == "cycles":
        from hafs_case import cycles_from_yaml
        from cycles import compute_cycles
        ccase = cycles_from_yaml(yaml_path)
        print(f"Case   : {ccase.storm_name} ({ccase.model_label})")
        print(f"Window : {ccase.valid_start:%Y-%m-%d %HZ} -> "
              f"{ccase.valid_end:%Y-%m-%d %HZ}  | run_root: {ccase.run_root}")
        print(f"Output : {ccase.out_dir}  | command: {command}")
        compute_cycles(ccase)
        return
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 analysis/tests/test_cycles.py` — Expected: all PASS.
Run: `python3 analysis/tests/test_run.py` — Expected: all PASS.
Run: `pytest analysis/tests/ -v` — Expected: entire suite passes.

- [ ] **Step 5: Report (no commit)**

Do NOT commit. Report changed files + test summary.

---

### Task 5: Docs + example cycles YAML

**Files:**
- Create: `storms/helene_hfsa_cycles.yaml`
- Modify: `README.md`
- Modify: `CLAUDE.md` (git-ignored — edit it anyway)

**Interfaces:**
- Consumes: the YAML schema from Task 1 (`cycles_from_yaml`) and output names from Task 4.

- [ ] **Step 1: Create `storms/helene_hfsa_cycles.yaml`**

```yaml
# HAFS-A — Hurricane Helene, all initializations vs the landfall rainfall.
# run_root holds one YYYYMMDDHH subdirectory per cycle (auto-discovered);
# every eligible cycle is scored on the same valid window over a shared
# union-of-tracks footprint, so only lead time changes between cycles.
run_root: /work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/HFSA
valid_start: 2024092600        # common window (UTC, YYYYMMDDHH)
valid_end:   2024092800
storm_name: Hurricane Helene
domain: [15.0, 42.0, -100.0, -60.0]   # lat_min, lat_max, lon_min, lon_max
mask_radius_km: 500
out_dir: analysis/output/helene_hfsa_cycles
# inits: [2024092400, 2024092412]    # optional; overrides auto-discovery
# ets_threshold_mm: 25               # headline threshold for the ETS panel
```

- [ ] **Step 2: Update `README.md`**

In the running/commands section (where `rmse` was added), add:

```
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles   # cross-init comparison
```

In the outputs list, add:

```
- `cycles_<slug>.csv` — per-init continuous + categorical scores on the common valid window (long format; `bias_mm` is the mean error in mm, `bias` the categorical frequency bias)
- `cycles_metrics_<slug>.png` — RMSE / bias / ETS vs initialization time
- `cycles_maps_<slug>.png` — nest window-QPF small-multiples per init + observed MRMS panel
```

Add a new section "How to read the cycle comparison" (after the RMSE section), covering, in a few sentences each:
- **Common valid window:** every cycle is scored on its rainfall accumulated over the same absolute `valid_start`–`valid_end` window against the same observed total, so metric changes across inits reflect lead time, not window length.
- **Union footprint:** one shared swath mask from the union of every included cycle's in-window forecast track (≤ `mask_radius_km`); a run with a bad track is penalized for raining in the wrong place — that is part of what is measured.
- **Eligibility:** a cycle is included only if it fully covers the window (`init ≤ valid_start` and run end `≥ valid_end`); skipped cycles are printed with the reason.
- **Cycles YAML:** `run_root` with YYYYMMDDHH subdirectories (auto-discovered, `inits:` overrides), required `valid_start`/`valid_end`/`domain`.

- [ ] **Step 3: Update `CLAUDE.md`**

Change the run-modes line from `[parent|ets|rmse|all]` to `[parent|ets|rmse|cycles|all]`, and add one line to the products/outputs description: `cycles — per-initialization comparison on a common valid window (cycles YAML with run_root; outputs cycles_*.csv/png)`.

- [ ] **Step 4: Verify**

Run: `pytest analysis/tests/ -v` — Expected: entire suite still passes (docs-only task; this is a regression guard).
Run: `python3 -c "import yaml; yaml.safe_load(open('storms/helene_hfsa_cycles.yaml'))"` — Expected: no error.

- [ ] **Step 5: Report (no commit)**

Do NOT commit. Report changed files.
