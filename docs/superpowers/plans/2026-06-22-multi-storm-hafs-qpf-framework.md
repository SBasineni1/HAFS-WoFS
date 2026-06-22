# Multi-storm HAFS QPF/ETS Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Helene/HFSA-specific QPF and ETS scripts run any storm and any HAFS (HFSA/HFSB) output by dropping in a small YAML case file, driven by a `StormCase` config object that auto-parses the run's `.atcfunix` track.

**Architecture:** A new dependency-light `hafs_case.py` defines a `StormCase` dataclass loaded from YAML; it parses the `.atcfunix` track, auto-detects HAFS-A/B from the path, and auto-derives init/domain when omitted. The four existing scripts drop their module-level config blocks and `main()`s, gaining functions that take `case`. A new `run.py` dispatches `parent`/`animation`/`ets`/`all`.

**Tech Stack:** Python 3, numpy, PyYAML, (heavy paths: cfgrib/eccodes/boto3/cartopy/scipy on Hercules). Tests: standalone-runnable (`python3 path/test.py`), also pytest-compatible.

## Global Constraints

- `hafs_case.py` MUST import only stdlib + numpy + yaml (NO cfgrib/boto3/eccodes/cartopy) so it is testable off-Hercules. Verbatim from spec: heavy GRIB/MRMS/Stage IV paths remain integration-only.
- Track lat/lon decode: tenths-of-degree + hemisphere suffix; `168N`→`16.8`, `832W`→`-83.2`. Must reproduce the current `TC_TRACK_6H` values (within rounding).
- Model label: `HFSA`→`"HAFS-A"`, `HFSB`→`"HAFS-B"`, neither→`"HAFS"`.
- Defaults when YAML omits a field: `grid_res=0.05`, `mask_radius_km=500.0`, `display_radius_km=750.0`, `thresholds_mm=[1,5,10,25,50,75,100,150,200,250]`, `out_dir=analysis/output/<case_slug>`, `mrms_cache_dir=/tmp/mrms_cache`, `stage4_cache_dir=/tmp/stage4_cache`, `fhours_filter=None`. `case_slug` = YAML filename stem.
- Single entry point is `analysis/run.py`; per-script `main()` config blocks are removed.
- Output filenames gain a `<case_slug>` suffix and live under `case.out_dir`.
- The science (accumulation logic, contingency math, color scale, thresholds) is UNCHANGED.
- Commit style: NO `Co-Authored-By` lines (per CLAUDE.md). Run scripts from repo root.
- Tests live in `analysis/tests/`; new modules importable via `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`.

---

### Task 1: ATCF track parser

**Files:**
- Create: `analysis/hafs_case.py`
- Test: `analysis/tests/test_hafs_case.py`
- Create (fixture): `analysis/tests/fixtures/helene.atcfunix`

**Interfaces:**
- Consumes: nothing (new module).
- Produces:
  - `decode_latlon(token: str) -> float` — `"168N"`→`16.8`, `"832W"`→`-83.2`, `"105S"`→`-10.5`, `"50E"`→`5.0`.
  - `parse_atcfunix(path) -> tuple[str | None, datetime, list[tuple[datetime, float, float]]]` — returns `(storm_name_or_None, init_dt, track)` where `track` is `[(valid_dt, lat, lon), ...]` deduped by lead hour (TAU), sorted ascending.

- [ ] **Step 1: Create the fixture atcfunix file**

Create `analysis/tests/fixtures/helene.atcfunix` with format-correct ATCF
lines encoding Helene's known 6-hourly track (init `2024092400`). One line per
lead hour; storm name in the trailing name field. Columns:
`BASIN, CY, YYYYMMDDHH, TECHNUM, TECH, TAU, LATd, LONd, VMAX, MSLP, ...`

```
AL, 09, 2024092400, 03, HFSA, 000, 168N, 832W, 65, 985, XX, 34, NEQ, 0000, 0000, 0000, 0000, 1010, 180, 30, 65, 0, L, 0, , 0, 0, HELENE, D
AL, 09, 2024092400, 03, HFSA, 006, 178N, 835W, 70, 980, XX, 34, NEQ, 0000, 0000, 0000, 0000, 1010, 180, 30, 70, 0, L, 0, , 0, 0, HELENE, D
AL, 09, 2024092400, 03, HFSA, 012, 190N, 838W, 75, 975, XX, 34, NEQ, 0000, 0000, 0000, 0000, 1010, 180, 30, 75, 0, L, 0, , 0, 0, HELENE, D
AL, 09, 2024092400, 03, HFSA, 018, 204N, 841W, 80, 970, XX, 34, NEQ, 0000, 0000, 0000, 0000, 1010, 180, 30, 80, 0, L, 0, , 0, 0, HELENE, D
AL, 09, 2024092400, 03, HFSA, 024, 218N, 843W, 90, 960, XX, 34, NEQ, 0000, 0000, 0000, 0000, 1010, 180, 30, 90, 0, L, 0, , 0, 0, HELENE, D
```

(Five lines is enough to assert decoding + ordering; more may be added.)

- [ ] **Step 2: Write the failing test**

Add to `analysis/tests/test_hafs_case.py`:

```python
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FIX = Path(__file__).resolve().parent / "fixtures"

from hafs_case import decode_latlon, parse_atcfunix


def test_decode_latlon():
    assert decode_latlon("168N") == 16.8
    assert decode_latlon("832W") == -83.2
    assert decode_latlon("105S") == -10.5
    assert decode_latlon("50E") == 5.0


def test_parse_atcfunix_reproduces_helene_track():
    name, init_dt, track = parse_atcfunix(FIX / "helene.atcfunix")
    assert name == "Helene"
    assert init_dt == datetime(2024, 9, 24, 0)
    # First three known 6-hourly fixes from the current hardcoded TC_TRACK_6H.
    assert track[0] == (datetime(2024, 9, 24, 0), 16.8, -83.2)
    assert track[1] == (datetime(2024, 9, 24, 6), 17.8, -83.5)
    assert track[2] == (datetime(2024, 9, 24, 12), 19.0, -83.8)
    # Sorted ascending by valid time, no duplicate lead hours.
    times = [t for t, _, _ in track]
    assert times == sorted(times)
    assert len(times) == len(set(times))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'hafs_case'` (or ImportError on the two names).

- [ ] **Step 4: Write minimal implementation**

Create `analysis/hafs_case.py`:

```python
"""StormCase config + ATCF track parsing for the HAFS QPF/ETS framework.

Dependency-light on purpose (stdlib + numpy + yaml only) so it imports and
tests off-Hercules, away from cfgrib/boto3/eccodes/cartopy.
"""

import re
from datetime import datetime, timedelta


def decode_latlon(token):
    """ATCF tenths-of-degree + hemisphere token -> signed float degrees.

    '168N' -> 16.8, '832W' -> -83.2, '105S' -> -10.5, '50E' -> 5.0.
    """
    token = token.strip()
    hemi = token[-1].upper()
    value = int(token[:-1]) / 10.0
    if hemi in ("S", "W"):
        value = -value
    return value


def parse_atcfunix(path):
    """Parse a HAFS .atcfunix track file.

    Returns (storm_name_or_None, init_dt, track) where track is a list of
    (valid_dt, lat, lon) deduped by lead hour (TAU) and sorted ascending.
    """
    init_dt = None
    name = None
    by_tau = {}
    with open(path) as fh:
        for line in fh:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 8:
                continue
            try:
                warn = datetime.strptime(cols[2], "%Y%m%d%H")
                tau = int(cols[5])
                lat = decode_latlon(cols[6])
                lon = decode_latlon(cols[7])
            except (ValueError, IndexError):
                continue
            if init_dt is None:
                init_dt = warn
            if tau not in by_tau:
                by_tau[tau] = (warn + timedelta(hours=tau), lat, lon)
            # Storm name: trailing alpha field (index 27 in standard atcfunix).
            if name is None and len(cols) > 27 and re.fullmatch(r"[A-Za-z]+", cols[27]):
                name = cols[27].title()
    track = [by_tau[t] for t in sorted(by_tau)]
    return name, init_dt, track
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: PASS for both tests (printed `PASS ...`).

- [ ] **Step 6: Commit**

```bash
git add analysis/hafs_case.py analysis/tests/test_hafs_case.py analysis/tests/fixtures/helene.atcfunix
git commit -m "Add ATCF atcfunix track parser for HAFS case framework"
```

---

### Task 2: Model detection + auto domain

**Files:**
- Modify: `analysis/hafs_case.py`
- Test: `analysis/tests/test_hafs_case.py`

**Interfaces:**
- Consumes: `parse_atcfunix` (Task 1).
- Produces:
  - `detect_model(run_dir) -> str` — `"HAFS-A"`/`"HAFS-B"`/`"HAFS"` from the path string (case-insensitive `HFSA`/`HFSB`).
  - `auto_domain(track, pad_deg=2.0) -> tuple[float, float, float, float]` — `(lat_min, lat_max, lon_min, lon_max)` = track bbox padded by `pad_deg`.

- [ ] **Step 1: Write the failing test**

Add to `analysis/tests/test_hafs_case.py`:

```python
from hafs_case import detect_model, auto_domain
from datetime import datetime


def test_detect_model():
    assert detect_model("/work2/.../helene/HFSA") == "HAFS-A"
    assert detect_model("/work2/.../helene/HFSB") == "HAFS-B"
    assert detect_model("/data/hfsa_run/lower") == "HAFS-A"   # case-insensitive
    assert detect_model("/work2/.../helene/other") == "HAFS"


def test_auto_domain_pads_track_bbox():
    track = [
        (datetime(2024, 9, 24, 0), 16.8, -83.2),
        (datetime(2024, 9, 26, 0), 28.8, -84.1),
        (datetime(2024, 9, 29, 6), 44.3, -61.5),
    ]
    lat_min, lat_max, lon_min, lon_max = auto_domain(track, pad_deg=2.0)
    assert lat_min == 14.8 and lat_max == 46.3
    assert lon_min == -86.1 and lon_max == -59.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: FAIL — ImportError on `detect_model` / `auto_domain`.

- [ ] **Step 3: Write minimal implementation**

Append to `analysis/hafs_case.py`:

```python
def detect_model(run_dir):
    """'HAFS-A'/'HAFS-B'/'HAFS' from HFSA/HFSB in the run-dir path."""
    s = str(run_dir).upper()
    if "HFSA" in s:
        return "HAFS-A"
    if "HFSB" in s:
        return "HAFS-B"
    return "HAFS"


def auto_domain(track, pad_deg=2.0):
    """(lat_min, lat_max, lon_min, lon_max) = padded bbox of the track."""
    lats = [la for _, la, _ in track]
    lons = [lo for _, _, lo in track]
    return (min(lats) - pad_deg, max(lats) + pad_deg,
            min(lons) - pad_deg, max(lons) + pad_deg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add analysis/hafs_case.py analysis/tests/test_hafs_case.py
git commit -m "Add HAFS model detection and auto domain helpers"
```

---

### Task 3: StormCase dataclass (position_at, fixed_grid, globs)

**Files:**
- Modify: `analysis/hafs_case.py`
- Test: `analysis/tests/test_hafs_case.py`

**Interfaces:**
- Consumes: `auto_domain` (Task 2).
- Produces: `StormCase` dataclass with fields (`run_dir: Path`, `init_dt: datetime`, `storm_name: str`, `model_label: str`, `domain: tuple`, `grid_res: float`, `mask_radius_km: float`, `display_radius_km: float`, `thresholds_mm: list[int]`, `out_dir: Path`, `mrms_cache_dir: Path`, `stage4_cache_dir: Path`, `fhours_filter: list[int] | None`, `track: list[tuple]`, `case_slug: str`, `init_str: str`) and methods:
  - `position_at(valid_dt) -> (lat, lon)` (linear track interpolation; clamps to endpoints).
  - `fixed_grid() -> (grid_lat, grid_lon)` (numpy meshgrid from domain+grid_res).
  - `parent_glob() -> str` / `storm_glob() -> str`.

- [ ] **Step 1: Write the failing test**

Add to `analysis/tests/test_hafs_case.py`:

```python
import numpy as np
from hafs_case import StormCase


def _toy_case():
    track = [
        (datetime(2024, 9, 24, 0), 16.8, -83.2),
        (datetime(2024, 9, 24, 6), 17.8, -83.5),
    ]
    return StormCase(
        run_dir=Path("/tmp/HFSA"), init_dt=datetime(2024, 9, 24, 0),
        storm_name="Helene", model_label="HAFS-A",
        domain=(15.0, 20.0, -90.0, -80.0), grid_res=1.0,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1, 5], out_dir=Path("/tmp/out"),
        mrms_cache_dir=Path("/tmp/mrms"), stage4_cache_dir=Path("/tmp/s4"),
        fhours_filter=None, track=track, case_slug="helene_hfsa",
        init_str="2024092400",
    )


def test_position_at_interpolates_and_clamps():
    c = _toy_case()
    # Midpoint between the two 6-hourly fixes.
    lat, lon = c.position_at(datetime(2024, 9, 24, 3))
    assert abs(lat - 17.3) < 1e-9 and abs(lon - (-83.35)) < 1e-9
    # Before/after the track clamps to the endpoints.
    assert c.position_at(datetime(2024, 9, 23, 0)) == (16.8, -83.2)
    assert c.position_at(datetime(2024, 9, 25, 0)) == (17.8, -83.5)


def test_fixed_grid_shape():
    c = _toy_case()
    grid_lat, grid_lon = c.fixed_grid()
    # lon -90..-80 step 1 -> 11 cols; lat 15..20 step 1 -> 6 rows.
    assert grid_lat.shape == (6, 11)
    assert grid_lon.shape == (6, 11)


def test_globs_use_init_str():
    c = _toy_case()
    assert c.parent_glob() == "**/*2024092400*parent.atm.f*.grb2"
    assert c.storm_glob() == "**/*2024092400*storm.atm.f*.grb2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: FAIL — ImportError on `StormCase`.

- [ ] **Step 3: Write minimal implementation**

Add to the top imports of `analysis/hafs_case.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
```

Append to `analysis/hafs_case.py`:

```python
@dataclass
class StormCase:
    run_dir: Path
    init_dt: datetime
    storm_name: str
    model_label: str
    domain: tuple          # (lat_min, lat_max, lon_min, lon_max)
    grid_res: float
    mask_radius_km: float
    display_radius_km: float
    thresholds_mm: list
    out_dir: Path
    mrms_cache_dir: Path
    stage4_cache_dir: Path
    fhours_filter: list
    track: list            # [(valid_dt, lat, lon), ...]
    case_slug: str
    init_str: str

    def position_at(self, valid_dt):
        """Linear interpolation of the track to any time; clamps to endpoints."""
        times = [t for t, _, _ in self.track]
        lats = [la for _, la, _ in self.track]
        lons = [lo for _, _, lo in self.track]
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

    def fixed_grid(self):
        """Fixed lat/lon verification/plot mesh from domain + grid_res."""
        lat_min, lat_max, lon_min, lon_max = self.domain
        fixed_lons = np.arange(lon_min, lon_max + self.grid_res, self.grid_res)
        fixed_lats = np.arange(lat_min, lat_max + self.grid_res, self.grid_res)
        return np.meshgrid(fixed_lons, fixed_lats)[::-1]  # (grid_lat, grid_lon)

    def parent_glob(self):
        return f"**/*{self.init_str}*parent.atm.f*.grb2"

    def storm_glob(self):
        return f"**/*{self.init_str}*storm.atm.f*.grb2"
```

Note: `np.meshgrid(lons, lats)` returns `(lon2d, lat2d)`; `[::-1]` swaps to
`(grid_lat, grid_lon)` to match the order used everywhere downstream.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add analysis/hafs_case.py analysis/tests/test_hafs_case.py
git commit -m "Add StormCase dataclass with track interp and fixed grid"
```

---

### Task 4: from_yaml loader (find atcfunix, defaults, auto-derivation)

**Files:**
- Modify: `analysis/hafs_case.py`
- Test: `analysis/tests/test_hafs_case.py`

**Interfaces:**
- Consumes: `parse_atcfunix`, `detect_model`, `auto_domain`, `StormCase`.
- Produces:
  - `find_atcfunix(run_dir) -> Path` — first match of `*.atcfunix` under run dir; raises `FileNotFoundError` naming the dir + glob if none.
  - `from_yaml(yaml_path) -> StormCase` — loads YAML, parses track, fills defaults + auto-derivation. `run_dir` required. Optional keys: `storm_name`, `init`, `domain`, `grid_res`, `mask_radius_km`, `display_radius_km`, `thresholds_mm`, `out_dir`, `mrms_cache_dir`, `stage4_cache_dir`, `fhours`, `model_label`, `track` (explicit fallback). Raises `KeyError` if `run_dir` missing.

- [ ] **Step 1: Write the failing test**

Add to `analysis/tests/test_hafs_case.py`. The test builds a temp run dir
containing a copy of the fixture atcfunix, plus a minimal YAML:

```python
import shutil
import tempfile
from hafs_case import from_yaml, find_atcfunix


def test_from_yaml_minimal_autoderives(tmp_path_factory=None):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        run_dir = tmpdir / "helene" / "HFSA"
        run_dir.mkdir(parents=True)
        shutil.copy(FIX / "helene.atcfunix", run_dir / "helene.atcfunix")
        yaml_path = tmpdir / "helene_hfsa.yaml"
        yaml_path.write_text(f"run_dir: {run_dir}\n")

        case = from_yaml(yaml_path)

        assert case.case_slug == "helene_hfsa"
        assert case.model_label == "HAFS-A"          # auto from path
        assert case.init_dt == datetime(2024, 9, 24, 0)  # from atcfunix
        assert case.storm_name == "Helene"            # from atcfunix name field
        assert case.init_str == "2024092400"
        assert case.grid_res == 0.05                  # default
        assert case.mask_radius_km == 500.0           # default
        assert case.thresholds_mm == [1, 5, 10, 25, 50, 75, 100, 150, 200, 250]
        assert case.out_dir == Path("analysis/output/helene_hfsa")
        # Domain auto-derived from the track bbox (non-empty, sane ordering).
        lat_min, lat_max, lon_min, lon_max = case.domain
        assert lat_min < lat_max and lon_min < lon_max
    finally:
        shutil.rmtree(tmpdir)


def test_from_yaml_overrides_win():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        run_dir = tmpdir / "HFSA"
        run_dir.mkdir(parents=True)
        shutil.copy(FIX / "helene.atcfunix", run_dir / "x.atcfunix")
        yaml_path = tmpdir / "case.yaml"
        yaml_path.write_text(
            f"run_dir: {run_dir}\n"
            "storm_name: Test Storm\n"
            "domain: [15.0, 42.0, -100.0, -60.0]\n"
            "mask_radius_km: 300\n"
            "out_dir: /tmp/custom_out\n"
        )
        case = from_yaml(yaml_path)
        assert case.storm_name == "Test Storm"
        assert case.domain == (15.0, 42.0, -100.0, -60.0)
        assert case.mask_radius_km == 300.0
        assert case.out_dir == Path("/tmp/custom_out")
    finally:
        shutil.rmtree(tmpdir)


def test_find_atcfunix_missing_raises():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        try:
            find_atcfunix(tmpdir)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert str(tmpdir) in str(e)
    finally:
        shutil.rmtree(tmpdir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: FAIL — ImportError on `from_yaml` / `find_atcfunix`.

- [ ] **Step 3: Write minimal implementation**

Add `import yaml` to the imports of `analysis/hafs_case.py`. Append:

```python
_DEFAULT_THRESHOLDS = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250]


def find_atcfunix(run_dir):
    """First *.atcfunix under run_dir, else FileNotFoundError naming the dir."""
    hits = sorted(Path(run_dir).glob("**/*.atcfunix"))
    if not hits:
        raise FileNotFoundError(
            f"No .atcfunix track file under {run_dir} (glob '**/*.atcfunix'). "
            f"Add an explicit 'track', 'init', and 'domain' to the YAML to run "
            f"without one."
        )
    return hits[0]


def from_yaml(yaml_path):
    """Load a StormCase from a YAML case file (run_dir required)."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if "run_dir" not in cfg:
        raise KeyError(f"'run_dir' is required in {yaml_path}")
    run_dir = Path(cfg["run_dir"])

    atcf_path = find_atcfunix(run_dir)
    name, init_from_atcf, track = parse_atcfunix(atcf_path)

    # init: YAML override (YYYYMMDDHH) else from atcfunix.
    if cfg.get("init"):
        init_dt = datetime.strptime(str(cfg["init"]), "%Y%m%d%H")
    else:
        init_dt = init_from_atcf
    init_str = init_dt.strftime("%Y%m%d%H")

    domain = tuple(cfg["domain"]) if cfg.get("domain") else auto_domain(track)
    out_dir = (Path(cfg["out_dir"]) if cfg.get("out_dir")
               else Path("analysis/output") / yaml_path.stem)

    return StormCase(
        run_dir=run_dir,
        init_dt=init_dt,
        storm_name=cfg.get("storm_name") or name or "Storm",
        model_label=cfg.get("model_label") or detect_model(run_dir),
        domain=domain,
        grid_res=float(cfg.get("grid_res", 0.05)),
        mask_radius_km=float(cfg.get("mask_radius_km", 500.0)),
        display_radius_km=float(cfg.get("display_radius_km", 750.0)),
        thresholds_mm=cfg.get("thresholds_mm", list(_DEFAULT_THRESHOLDS)),
        out_dir=out_dir,
        mrms_cache_dir=Path(cfg.get("mrms_cache_dir", "/tmp/mrms_cache")),
        stage4_cache_dir=Path(cfg.get("stage4_cache_dir", "/tmp/stage4_cache")),
        fhours_filter=cfg.get("fhours"),
        track=track,
        case_slug=yaml_path.stem,
        init_str=init_str,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add analysis/hafs_case.py analysis/tests/test_hafs_case.py
git commit -m "Add from_yaml loader with defaults and auto-derivation"
```

---

### Task 5: Refactor qpf_full_run.py to be case-driven

**Files:**
- Modify: `analysis/qpf_full_run.py`

**Interfaces:**
- Consumes: `StormCase` (Task 3) via function args.
- Produces:
  - Removes module-level config block (`HAFS_RUN_DIR`, `INIT_STR`, `INIT_DT`, `FILE_GLOB`, `FHOURS_FILTER`, `OUT_DIR`, `MRMS_CACHE_DIR`, `CFGRIB_IDX_DIR`, `GRID_RES`, `FIXED_DOMAIN`, `TC_TRACK_6H`, `TC_MASK_RADIUS_KM`). KEEPS module constants `MRMS_BUCKET`, `MRMS_PRODUCT`, `QPF_LEVELS`, `QPF_COLORS`.
  - Removes `tc_position_at` (now `case.position_at`).
  - `apply_tc_mask(lats, lons, data, valid_dt, case)` — uses `case.position_at`, `case.mask_radius_km`.
  - `plot_frame(case, fhour, fixed_lons, fixed_lats, hafs_mm, mrms_lons, mrms_lats, mrms_mm, full_domain, out_path)` — titles from `case`.
  - `generate_animation(case)` — replaces `main()`.
  - Unchanged free helpers: `parse_fhour`, `discover_files`, `read_hafs_tp_records`, `pick_total_record`, `load_hafs_precip`, `regrid_hafs`, `accumulate_hafs_step`, `hafs_event_total`, `haversine_km`, `crop_to_domain`, MRMS loaders, `qpf_cmap`.

- [ ] **Step 1: Replace the CONFIG block (lines ~58-120)**

Delete the entire `# CONFIG` block through `TC_TRACK_6H`, `TC_MASK_RADIUS_KM`,
and the `INIT_DT = datetime.strptime(...)` line. KEEP these module constants
(move them just below the imports):

```python
MRMS_BUCKET = "noaa-mrms-pds"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"

QPF_LEVELS = [0, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
QPF_COLORS = [
    "#ffffff", "#c8f0f0", "#64d2ff", "#3296ff",
    "#02fd02", "#01c501", "#008e00", "#fdf802",
    "#e5bc00", "#fd9500", "#fd0000", "#d40000",
]
```

Add to imports near the top (after `from datetime import ...`):

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))  # already implied
from hafs_case import StormCase  # noqa: F401  (type reference / import side)
```

(Use `import sys` if not already imported.)

- [ ] **Step 2: Delete `tc_position_at` and update `apply_tc_mask`**

Remove the `tc_position_at` function entirely. Replace `apply_tc_mask`:

```python
def apply_tc_mask(lats, lons, data, valid_dt, case):
    """Zero out QPE beyond case.mask_radius_km from the TC center at valid_dt."""
    tc_lat, tc_lon = case.position_at(valid_dt)
    if lats.ndim == 1 and lons.ndim == 1:
        lons_2d, lats_2d = np.meshgrid(lons, lats)
    else:
        lats_2d, lons_2d = lats, lons
    dist = haversine_km(tc_lat, tc_lon, lats_2d, lons_2d)
    return np.where(dist <= case.mask_radius_km, data, 0.0)
```

- [ ] **Step 3: Update `plot_frame` signature + title text**

Change the signature to take `case` first and replace every `INIT_DT` with
`case.init_dt` and the hardcoded `"Hurricane Helene — HAFS-A"` /
`"HAFS-A Accumulated Precip"` strings with `case.storm_name` /
`f"{case.model_label} Accumulated Precip"` and
`f"{case.storm_name} — {case.model_label} QPF vs MRMS QPE"`.

```python
def plot_frame(case, fhour, fixed_lons, fixed_lats, hafs_mm,
               mrms_lons, mrms_lats, mrms_mm, full_domain, out_path):
    valid_dt = case.init_dt + timedelta(hours=fhour)
    ...
    axes[0].set_title(
        f"{case.model_label} Accumulated Precip\n"
        f"Init {case.init_dt:%Y-%m-%d %HZ} | "
        f"F{fhour:03d} (0–{fhour}h, valid {valid_dt:%Y-%m-%d %HZ})"
    )
    ...
    fig.suptitle(
        f"{case.storm_name} — {case.model_label} QPF vs MRMS QPE | "
        f"F{fhour:03d} ending {valid_dt:%Y-%m-%d %HZ}",
        fontsize=13, y=1.01,
    )
```

- [ ] **Step 4: Convert `main()` to `generate_animation(case)`**

Replace `def main():` with `def generate_animation(case):` and inside, replace:
`HAFS_RUN_DIR`→`case.run_dir`, `FILE_GLOB`→`case.storm_glob()`,
`FHOURS_FILTER`→`case.fhours_filter`, `INIT_DT`→`case.init_dt`,
`OUT_DIR`→`case.out_dir`, `MRMS_CACHE_DIR`→`case.mrms_cache_dir`,
`GRID_RES`→`case.grid_res`, `FIXED_DOMAIN`→`case.domain`,
`TC_MASK_RADIUS_KM`→`case.mask_radius_km`, `tc_position_at(x)`→`case.position_at(x)`,
each `apply_tc_mask(...)` call gets a trailing `, case`, and `plot_frame(...)`
gets `case` as its first arg. Frame filenames stay `qpf_frame_{fhour:03d}.png`
under `case.out_dir`. Delete the `if FIXED_DOMAIN is not None` auto-scan branch's
reliance on the global — `case.domain` is always set, so keep only the
hardcoded-domain branch (auto-domain now happens in `from_yaml`).

- [ ] **Step 5: Replace the `__main__` guard**

```python
if __name__ == "__main__":
    import sys
    from hafs_case import from_yaml
    generate_animation(from_yaml(sys.argv[1]))
```

- [ ] **Step 6: Verify it compiles**

Run: `python3 -m py_compile analysis/qpf_full_run.py`
Expected: no output (exit 0). (Full run requires Hercules; this only checks syntax/names.)

- [ ] **Step 7: Commit**

```bash
git add analysis/qpf_full_run.py
git commit -m "Refactor qpf_full_run to case-driven generate_animation"
```

---

### Task 6: Refactor parent_qpf.py to be case-driven

**Files:**
- Modify: `analysis/parent_qpf.py`

**Interfaces:**
- Consumes: `StormCase`, refactored `qpf_full_run` (no `INIT_DT`/`FIXED_DOMAIN`/`OUT_DIR`/`MRMS_CACHE_DIR`/`tc_position_at`/`haversine_km` globals — import the surviving helpers only).
- Produces:
  - `default_parent_path(case)` — uses `case.run_dir`, `case.parent_glob()`.
  - `stage4_total(case)` — uses `case.init_dt`, end_fhour, `case.stage4_cache_dir`, `case.display_radius_km`, `case.position_at`.
  - `plot_compare(case, panels, end_fhour, out_path)` — uses `case.domain`, `case.storm_name`, `case.model_label`, `case.init_dt`.
  - `generate_parent_figure(case)` — replaces `main()`; writes `case.out_dir / f"parent_qpf_{case.case_slug}.png"`.

- [ ] **Step 1: Fix the imports**

Replace the `from qpf_full_run import (...)` block with only the surviving names:

```python
from qpf_full_run import (
    QPF_LEVELS, QPF_COLORS,
    read_hafs_tp_records, haversine_km, load_mrms_hour, crop_to_domain,
)
from hafs_case import from_yaml  # used in __main__
```

Remove `HAFS_RUN_DIR, INIT_STR, INIT_DT, FIXED_DOMAIN, OUT_DIR, MRMS_CACHE_DIR, tc_position_at` from the import. Delete the module-level
`MASK_RADIUS_KM = 750.0`, `PARENT_PNG = ...` lines, and the `STAGE4_*` constants
stay EXCEPT move `STAGE4_CACHE_DIR` usage to `case.stage4_cache_dir`
(keep `STAGE4_BASE`).

- [ ] **Step 2: Update `default_parent_path` and `stage4_total`**

```python
def default_parent_path(case):
    hits = sorted(case.run_dir.glob(case.parent_glob()))
    if not hits:
        return None
    def fhour(p):
        m = re.search(r"\.f(\d{3})\.grb2$", p.name)
        return int(m.group(1)) if m else -1
    return max(hits, key=fhour)
```

In `stage4_total`, change the signature to `stage4_total(case, end_fhour)` and
replace `init_dt`→`case.init_dt`, `cache_dir`→`case.stage4_cache_dir`,
`MASK_RADIUS_KM`→`case.display_radius_km`, `tc_position_at(x)`→`case.position_at(x)`.
(`ensure_stage4_files`, `index_stage4_24h_conus`, `read_stage4`, `stage4_tar_url`
are unchanged pure helpers.)

- [ ] **Step 3: Update `plot_compare` title + domain**

```python
def plot_compare(case, panels, end_fhour, out_path):
    lat_min, lat_max, lon_min, lon_max = case.domain
    ...
    valid_dt = case.init_dt + timedelta(hours=end_fhour)
    fig.suptitle(
        f"{case.storm_name} — {case.model_label} parent QPF vs MRMS vs Stage IV "
        f"(0–{end_fhour}h, valid {valid_dt:%Y-%m-%d %HZ})",
        fontsize=13, y=1.01,
    )
```

- [ ] **Step 4: Convert `main()` to `generate_parent_figure(case)`**

Rename and replace globals: `OUT_DIR`→`case.out_dir`,
`MRMS_CACHE_DIR`→`case.mrms_cache_dir`, `default_parent_path()`→`default_parent_path(case)`,
`HAFS_RUN_DIR`→`case.run_dir`, `INIT_STR`→`case.init_str`,
`INIT_DT`→`case.init_dt`, `FIXED_DOMAIN`→`case.domain`,
`MASK_RADIUS_KM`→`case.display_radius_km`, `tc_position_at(x)`→`case.position_at(x)`,
`load_mrms_hour(s3, t, MRMS_CACHE_DIR)`→`load_mrms_hour(s3, t, case.mrms_cache_dir)`,
`stage4_total(INIT_DT, end_fhour, STAGE4_CACHE_DIR)`→`stage4_total(case, end_fhour)`,
`plot_compare(panels, end_fhour, FIXED_DOMAIN, PARENT_PNG)`→
`plot_compare(case, panels, end_fhour, out_png)` where
`out_png = case.out_dir / f"parent_qpf_{case.case_slug}.png"` and
`case.out_dir.mkdir(parents=True, exist_ok=True)` is called first. The HAFS panel
title uses `f"{case.model_label} Parent APCP\n..."`.

- [ ] **Step 5: Replace the `__main__` guard**

```python
if __name__ == "__main__":
    generate_parent_figure(from_yaml(sys.argv[1]))
```

- [ ] **Step 6: Verify it compiles**

Run: `python3 -m py_compile analysis/parent_qpf.py`
Expected: no output (exit 0).

- [ ] **Step 7: Commit**

```bash
git add analysis/parent_qpf.py
git commit -m "Refactor parent_qpf to case-driven generate_parent_figure"
```

---

### Task 7: Refactor ets_score.py to be case-driven

**Files:**
- Modify: `analysis/ets_score.py`

**Interfaces:**
- Consumes: `StormCase`, refactored `qpf_full_run` helpers.
- Produces:
  - `build_mrms_total(case, max_fhour, grid_lat, grid_lon)` — uses `case.init_dt`, `case.mrms_cache_dir`.
  - `tc_swath_mask(case, max_fhour, grid_lat, grid_lon)` — uses `case.position_at`, `case.mask_radius_km`.
  - Unchanged: `regrid_mrms_to_fixed`, `contingency_scores`, `THRESHOLDS_MM` (kept as fallback default but ETS now reads `case.thresholds_mm`).
  - `compute_ets_single(case)` — replaces `main()` (MRMS-only path); writes `case.out_dir / f"ets_{case.case_slug}.{png,csv}"`.

- [ ] **Step 1: Fix imports**

Replace the `from qpf_full_run import (...)` block:

```python
from qpf_full_run import (
    MRMS_BUCKET, MRMS_PRODUCT,
    discover_files, hafs_event_total,
    haversine_km, load_mrms_hour,
)
from hafs_case import from_yaml
```

Remove `HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER, FIXED_DOMAIN, GRID_RES, TC_MASK_RADIUS_KM, INIT_DT, OUT_DIR, MRMS_CACHE_DIR, tc_position_at` from the import. Delete module-level `ETS_PNG`/`ETS_CSV`.

- [ ] **Step 2: Parameterize `build_mrms_total` and `tc_swath_mask`**

```python
def build_mrms_total(case, max_fhour, grid_lat, grid_lon):
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    mrms_sum = None
    mlat = mlon = None
    for h in range(1, max_fhour + 1):
        t = case.init_dt + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, case.mrms_cache_dir)
            ...
```

```python
def tc_swath_mask(case, max_fhour, grid_lat, grid_lon):
    swath = np.zeros(grid_lat.shape, dtype=bool)
    for h in range(0, max_fhour + 1):
        tlat, tlon = case.position_at(case.init_dt + timedelta(hours=h))
        swath |= haversine_km(tlat, tlon, grid_lat, grid_lon) <= case.mask_radius_km
    return swath
```

- [ ] **Step 3: Convert `main()` to `compute_ets_single(case)`**

Replace globals: `discover_files(HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER)`→
`discover_files(case.run_dir, case.storm_glob(), case.fhours_filter)`,
`INIT_DT`→`case.init_dt`, `FIXED_DOMAIN`/`GRID_RES`→`case.fixed_grid()`,
`build_mrms_total(max_fhour, ...)`→`build_mrms_total(case, max_fhour, ...)`,
`tc_swath_mask(max_fhour, ...)`→`tc_swath_mask(case, max_fhour, ...)`,
`THRESHOLDS_MM`→`case.thresholds_mm`, `TC_MASK_RADIUS_KM`→`case.mask_radius_km`,
output paths →`case.out_dir / f"ets_{case.case_slug}.csv"` and `.png`
(mkdir `case.out_dir` first). Title uses `case.storm_name`/`case.model_label`.

- [ ] **Step 4: Replace `__main__` guard**

```python
if __name__ == "__main__":
    import sys
    compute_ets_single(from_yaml(sys.argv[1]))
```

- [ ] **Step 5: Verify it compiles**

Run: `python3 -m py_compile analysis/ets_score.py`
Expected: no output (exit 0).

- [ ] **Step 6: Commit**

```bash
git add analysis/ets_score.py
git commit -m "Refactor ets_score to case-driven compute_ets_single"
```

---

### Task 8: Refactor ets_full.py to be case-driven + fix its test

**Files:**
- Modify: `analysis/ets_full.py`
- Modify: `analysis/tests/test_ets_full.py` (import path only if needed)

**Interfaces:**
- Consumes: `StormCase`, refactored `qpf_full_run`/`ets_score`/`parent_qpf`.
- Produces:
  - `build_fixed_grid(case)` → `case.fixed_grid()`.
  - `hafs_parent_total(case, grid_lat, grid_lon)`.
  - `stage4_on_fixed(case, max_fhour, grid_lat, grid_lon)`.
  - `plot_curves(case, results, max_fhour, out_path, caveat="")`.
  - `compute_ets(case)` — replaces `main()`; writes `case.out_dir / f"ets_full_{case.case_slug}.{png,csv}"`.
  - Unchanged pure helpers `regrid_2d_to_fixed`, `score_pair` (keeps `test_ets_full.py` green).

- [ ] **Step 1: Fix imports**

```python
from qpf_full_run import discover_files, hafs_event_total
from ets_score import (
    contingency_scores, build_mrms_total, tc_swath_mask,
)
from parent_qpf import (
    default_parent_path, read_hafs_tp_records, pick_cumulative_record,
    stage4_total,
)
from hafs_case import from_yaml
```

Remove `HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER, FIXED_DOMAIN, GRID_RES, TC_MASK_RADIUS_KM, INIT_DT, OUT_DIR` and `THRESHOLDS_MM`, `STAGE4_CACHE_DIR` from imports. Delete module-level `OUT_PNG`/`OUT_CSV`.

- [ ] **Step 2: Parameterize the helpers**

```python
def build_fixed_grid(case):
    return case.fixed_grid()


def hafs_parent_total(case, grid_lat, grid_lon):
    path = default_parent_path(case)
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


def stage4_on_fixed(case, max_fhour, grid_lat, grid_lon):
    s4_lat, s4_lon, s4_total, s4_label = stage4_total(case, max_fhour)
    if s4_total is None:
        return None, "unavailable"
    grid = regrid_2d_to_fixed(s4_lat, s4_lon, s4_total, grid_lat, grid_lon)
    return grid, s4_label
```

- [ ] **Step 3: Update `plot_curves` title**

Add `case` as first arg; replace the hardcoded title with:

```python
    ax.set_title(
        f"{case.storm_name} — {case.model_label} QPF ETS vs MRMS & Stage IV\n"
        f"0–{max_fhour}h | init {case.init_dt:%Y-%m-%d %HZ} | "
        f"TC swath ≤{case.mask_radius_km:.0f} km"
    )
```

Also replace `ax.set_xticks(THRESHOLDS_MM)` with `ax.set_xticks(case.thresholds_mm)`.

- [ ] **Step 4: Convert `main()` to `compute_ets(case)`**

Replace globals: `discover_files(HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER)`→
`discover_files(case.run_dir, case.storm_glob(), case.fhours_filter)`,
`INIT_DT`→`case.init_dt`, `build_fixed_grid()`→`build_fixed_grid(case)`,
`hafs_parent_total(...)`→`hafs_parent_total(case, ...)`,
`build_mrms_total(max_fhour, ...)`→`build_mrms_total(case, max_fhour, ...)`,
`stage4_on_fixed(max_fhour, ...)`→`stage4_on_fixed(case, max_fhour, ...)`,
`tc_swath_mask(max_fhour, ...)`→`tc_swath_mask(case, max_fhour, ...)`,
`THRESHOLDS_MM`→`case.thresholds_mm`,
`score_pair(fgrid, ogrid, swath, THRESHOLDS_MM, ...)`→
`score_pair(fgrid, ogrid, swath, case.thresholds_mm, ...)`,
outputs →`out_csv = case.out_dir / f"ets_full_{case.case_slug}.csv"` and
`out_png = case.out_dir / f"ets_full_{case.case_slug}.png"`
(`case.out_dir.mkdir(parents=True, exist_ok=True)` first),
`plot_curves(results, max_fhour, OUT_PNG, caveat=...)`→
`plot_curves(case, results, max_fhour, out_png, caveat=...)`.

- [ ] **Step 5: Replace `__main__` guard**

```python
if __name__ == "__main__":
    import sys
    compute_ets(from_yaml(sys.argv[1]))
```

- [ ] **Step 6: Verify compile + existing test still passes**

Run: `python3 -m py_compile analysis/ets_full.py`
Expected: no output.

Run: `python3 analysis/tests/test_ets_full.py`
Expected: `5 passed` (the pure-helper tests are unaffected — `regrid_2d_to_fixed`
and `score_pair` signatures unchanged). If import of `ets_full` now fails locally
due to heavy deps, change the test import to pull from the module functions
without importing heavy submodules — but since `ets_full` imports `qpf_full_run`
(needs cfgrib/boto3), the test may only run on Hercules. If so, note it and
proceed; the helpers are also covered there.

- [ ] **Step 7: Commit**

```bash
git add analysis/ets_full.py analysis/tests/test_ets_full.py
git commit -m "Refactor ets_full to case-driven compute_ets"
```

---

### Task 9: run.py entry point

**Files:**
- Create: `analysis/run.py`
- Test: `analysis/tests/test_run.py`

**Interfaces:**
- Consumes: `from_yaml`, `generate_parent_figure`, `generate_animation`, `compute_ets`.
- Produces: `dispatch(case, command) -> None` and a `main(argv)` CLI. Heavy modules are imported lazily inside `dispatch` so arg parsing is testable off-Hercules.

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/test_run.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run


def test_parse_args_defaults_to_all():
    yaml_path, command = run.parse_args(["case.yaml"])
    assert yaml_path == "case.yaml" and command == "all"


def test_parse_args_explicit_command():
    yaml_path, command = run.parse_args(["case.yaml", "ets"])
    assert command == "ets"


def test_parse_args_rejects_unknown_command():
    try:
        run.parse_args(["case.yaml", "bogus"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_run.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'run'`.

- [ ] **Step 3: Write minimal implementation**

Create `analysis/run.py`:

```python
"""Single entry point for the HAFS QPF/ETS framework.

    python analysis/run.py <case.yaml> [parent|animation|ets|all]

Loads a StormCase from the YAML case file and runs the requested product(s).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = ("parent", "animation", "ets", "all")


def parse_args(argv):
    """(yaml_path, command) from argv; command defaults to 'all'."""
    if not argv:
        print("usage: run.py <case.yaml> [parent|animation|ets|all]")
        raise SystemExit(2)
    yaml_path = argv[0]
    command = argv[1] if len(argv) > 1 else "all"
    if command not in COMMANDS:
        print(f"unknown command '{command}'; choose from {COMMANDS}")
        raise SystemExit(2)
    return yaml_path, command


def dispatch(case, command):
    """Run the requested product(s) for a loaded StormCase."""
    from parent_qpf import generate_parent_figure
    from qpf_full_run import generate_animation
    from ets_full import compute_ets
    if command in ("parent", "all"):
        generate_parent_figure(case)
    if command in ("animation", "all"):
        generate_animation(case)
    if command in ("ets", "all"):
        compute_ets(case)


def main(argv):
    from hafs_case import from_yaml
    yaml_path, command = parse_args(argv)
    case = from_yaml(yaml_path)
    print(f"Case   : {case.storm_name} ({case.model_label})")
    print(f"Init   : {case.init_dt:%Y-%m-%d %HZ}  | run_dir: {case.run_dir}")
    print(f"Domain : {case.domain}  | track points: {len(case.track)}")
    print(f"Output : {case.out_dir}  | command: {command}")
    dispatch(case, command)


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 analysis/tests/test_run.py`
Expected: PASS (3 tests). (`parse_args` works without heavy deps because the
`generate_*` imports are inside `dispatch`/`main`.)

- [ ] **Step 5: Commit**

```bash
git add analysis/run.py analysis/tests/test_run.py
git commit -m "Add run.py single entry point with subcommands"
```

---

### Task 10: Example YAML cases + docs

**Files:**
- Create: `storms/helene_hfsa.yaml`
- Create: `storms/helene_hfsb.yaml`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the YAML schema from Task 4.
- Produces: runnable example cases + usage docs.

- [ ] **Step 1: Create `storms/helene_hfsa.yaml`**

```yaml
# HAFS-A — Hurricane Helene (2024-09-24 00Z init).
# Only run_dir is required; everything below is optional and auto-derived
# from the run's .atcfunix track + path when omitted.
run_dir: /work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/HFSA
storm_name: Hurricane Helene
init: 2024092400
domain: [15.0, 42.0, -100.0, -60.0]   # lat_min, lat_max, lon_min, lon_max
mask_radius_km: 500
out_dir: analysis/output/helene_hfsa
```

- [ ] **Step 2: Create `storms/helene_hfsb.yaml`**

Identical except the path and output dir — demonstrates the A/B switch:

```yaml
# HAFS-B — Hurricane Helene. Model label auto-detected as HAFS-B from HFSB path.
run_dir: /work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/HFSB
storm_name: Hurricane Helene
init: 2024092400
domain: [15.0, 42.0, -100.0, -60.0]
mask_radius_km: 500
out_dir: analysis/output/helene_hfsb
```

- [ ] **Step 3: Add a "Running a new storm" section to `README.md`**

Add (adapt to existing README structure):

```markdown
## Running a new storm

The QPF/ETS pipeline is case-driven. To analyze a new storm or a different
HAFS run, copy a YAML case file and change `run_dir`:

    cp storms/helene_hfsa.yaml storms/<storm>_<model>.yaml
    # edit run_dir (and optionally domain)

Then run from the repo root:

    python analysis/run.py storms/<storm>_<model>.yaml all      # parent + animation + ets
    python analysis/run.py storms/<storm>_<model>.yaml parent   # 3-panel QPF figure only
    python analysis/run.py storms/<storm>_<model>.yaml ets      # ETS curves + CSV only

The storm track, init time, and HAFS-A/B label are read automatically from the
run's `.atcfunix` file and path. Only `run_dir` is required in the YAML; all
other keys are optional overrides. Outputs land in `out_dir`
(default `analysis/output/<case_name>/`).
```

- [ ] **Step 4: Update `CLAUDE.md` Code Conventions**

Add under Code Conventions:

```markdown
- Run analyses via `python analysis/run.py storms/<case>.yaml [parent|animation|ets|all]`
- Per-storm config lives in `storms/*.yaml`; only `run_dir` is required
- Storm track / init / HAFS-A vs HAFS-B are auto-derived from the run's `.atcfunix` + path
```

- [ ] **Step 5: Verify the example YAML parses (track parser only, no GRIB)**

This requires a real run dir, so it is a Hercules check:
Run (on Hercules): `python analysis/run.py storms/helene_hfsa.yaml parent`
Expected: prints the case summary, regenerates the 3-panel figure under
`analysis/output/helene_hfsa/parent_qpf_helene_hfsa.png` matching the current
Helene figure.

- [ ] **Step 6: Commit**

```bash
git add storms/helene_hfsa.yaml storms/helene_hfsb.yaml README.md CLAUDE.md
git commit -m "Add example storm YAML cases and usage docs"
```

---

## Self-Review

**Spec coverage:**
- `hafs_case.py` / `StormCase` / `from_yaml` / atcfunix parse / model detect / auto domain → Tasks 1-4. ✓
- `run.py` entry point + subcommands → Task 9. ✓
- Refactor qpf_full_run / parent_qpf / ets_score / ets_full → Tasks 5-8. ✓
- Example YAML + A/B demonstration + docs → Task 10. ✓
- Output filenames gain `<case_slug>` under `out_dir` → Tasks 6,7,8. ✓
- Behavior-preserving keystone test (reproduces TC_TRACK_6H) → Task 1. ✓
- Error handling (missing atcfunix / run_dir) → Task 4 tests. ✓
- Keep test_ets_full.py green → Task 8 Step 6. ✓

**Placeholder scan:** Task 6 Step 1 had a throwaway `OUT_DIR_UNUSED := None`
illustration immediately corrected by the real import block — the real block is
the one to apply. No "TBD"/"handle edge cases" placeholders remain.

**Type consistency:** `case.position_at`, `case.fixed_grid`, `case.storm_glob`,
`case.parent_glob`, `case.mask_radius_km`, `case.display_radius_km`,
`case.thresholds_mm`, `case.init_str`, `case.case_slug`, `case.out_dir` are used
consistently across Tasks 5-9 exactly as defined in Tasks 3-4. `stage4_total`
becomes `(case, end_fhour)` in Task 6 and is called that way in Task 8. ✓

## Notes on local vs Hercules testing

- Tasks 1-4 (hafs_case) and Task 9 (run arg parsing) are fully unit-tested
  locally with `python3` (numpy + pyyaml only; no pytest required).
- Tasks 5-8 refactor modules that import cfgrib/boto3/eccodes/cartopy, available
  only on Hercules. Locally they are verified with `python3 -m py_compile`;
  full integration is the Hercules run in Task 10 Step 5, which must reproduce
  the current Helene HFSA figures.
