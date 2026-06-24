# Init-tagged Output Filenames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-stamp the run's initialization time into every product's output filenames (per-case parent + ETS, and the HFSA-vs-HFSB comparison) so runs from different inits never collide and each file says which init it came from.

**Architecture:** A single `StormCase.output_slug` property (`case_slug` + init, de-duplicated) drives per-case output names. The comparison derives its init from the two cases (which must share one) via small `_check_same_init` / `_init_tag` helpers and an `init`-tagged slug; a pure `_plot_comparison` helper keeps `replot` testable. Labeling only — no science change.

**Tech Stack:** Python 3, numpy, pyyaml, matplotlib; heavy paths (cfgrib/boto3/cartopy) on Hercules. Tests are standalone-runnable (`python3 analysis/tests/<file>.py`).

## Global Constraints

- `python3` (NOT `python`); `pytest` is NOT installed. Test files run standalone via `python3 analysis/tests/<file>.py` using the existing `_run_all()` harness (already present in each test file).
- `hafs_case.py` stays import-light: stdlib + numpy + yaml only.
- `StormCase.output_slug` = `case_slug` if `init_str in case_slug` else `f"{case_slug}_{init_str}"` (de-duplicating substring check). `init_str` format is `%Y%m%d%H`.
- Only OUTPUT FILENAME slugs change. Do NOT change the `out_dir` default in `hafs_case.from_yaml` (it uses `case_slug` for the directory and must stay).
- Comparison: the two cases MUST share `init_dt` → `ValueError` naming both paths + inits if they differ. Init tag is de-duplicated the same way.
- `_init_tag(label, init_dt)` returns `(slug, title)`: `slug = _slug(label)` with `init_str` appended unless already present; `title = f"{label} (init {init_dt:%Y-%m-%d %HZ})"`.
- Comparison CSV column orders become — categorical: `init, model, forecast, observation, threshold, a, b, c, d, ets, csi, bias, pod, far, hss`; FSS: `init, model, forecast, observation, threshold, scale_cells, scale_km, fss`.
- Commit messages: NO `Co-Authored-By` lines. Run scripts from repo root.
- All existing tests stay green: hafs_case 15, ets_full 5, ets_score 2, best_track 2, skill_metrics 4, compare 8, run 4.

---

### Task 1: `StormCase.output_slug` property

**Files:**
- Modify: `analysis/hafs_case.py` (the `StormCase` dataclass)
- Test: `analysis/tests/test_hafs_case.py` (add tests)

**Interfaces:**
- Consumes: existing `StormCase.case_slug`, `StormCase.init_str`.
- Produces: `StormCase.output_slug` (property, `str`).

- [ ] **Step 1: Write the failing tests**

Add to `analysis/tests/test_hafs_case.py`. It already has a `_toy_case()`
helper that builds a `StormCase` with `case_slug="helene_hfsa"` and
`init_str="2024092400"`; reuse it.

```python
def test_output_slug_appends_init():
    c = _toy_case()
    assert c.case_slug == "helene_hfsa"
    assert c.output_slug == "helene_hfsa_2024092400"


def test_output_slug_dedups_when_init_already_in_slug():
    import dataclasses
    c = dataclasses.replace(_toy_case(), case_slug="helene_hfsa_2024092400")
    assert c.output_slug == "helene_hfsa_2024092400"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: FAIL — `AttributeError: 'StormCase' object has no attribute 'output_slug'`.

- [ ] **Step 3: Add the property**

In `analysis/hafs_case.py`, inside the `StormCase` dataclass (place it just
after the `parent_glob`/`storm_glob` methods), add:

```python
    @property
    def output_slug(self):
        """case_slug with the init appended for output filenames, de-duplicated.

        'helene_hfsa' -> 'helene_hfsa_2024092400'; returned unchanged if the
        slug already contains the init string (YAML was named with it).
        """
        if self.init_str in self.case_slug:
            return self.case_slug
        return f"{self.case_slug}_{self.init_str}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 analysis/tests/test_hafs_case.py`
Expected: `17 passed` (15 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add analysis/hafs_case.py analysis/tests/test_hafs_case.py
git commit -m "Add StormCase.output_slug (init-tagged, de-duplicated)"
```

---

### Task 2: Per-case outputs use `output_slug` (+ init in parent title)

**Files:**
- Modify: `analysis/parent_qpf.py`
- Modify: `analysis/ets_full.py`
- Modify: `analysis/ets_score.py`

**Interfaces:**
- Consumes: `StormCase.output_slug` (Task 1).
- Produces: per-case output filenames carry the init; parent figure title shows init.

- [ ] **Step 1: Swap the output-filename slug in all three files**

In each file, the OUTPUT FILENAME f-strings use `case.case_slug`; change those to
`case.output_slug`. `grep -n "case_slug" analysis/parent_qpf.py analysis/ets_full.py analysis/ets_score.py`
to find them. The exact lines to change:

- `analysis/parent_qpf.py` (in `generate_parent_figure`):
  `out_png = case.out_dir / f"parent_qpf_{case.case_slug}.png"`
  → `out_png = case.out_dir / f"parent_qpf_{case.output_slug}.png"`
- `analysis/ets_full.py` (in `compute_ets`):
  `out_csv = case.out_dir / f"ets_full_{case.case_slug}.csv"`
  → `... f"ets_full_{case.output_slug}.csv"`
  and `out_png = case.out_dir / f"ets_full_{case.case_slug}.png"`
  → `... f"ets_full_{case.output_slug}.png"`
- `analysis/ets_score.py` (in `compute_ets_single`):
  `ets_csv = case.out_dir / f"ets_{case.case_slug}.csv"`
  → `... f"ets_{case.output_slug}.csv"`
  and `ets_png = case.out_dir / f"ets_{case.case_slug}.png"`
  → `... f"ets_{case.output_slug}.png"`

Do NOT touch any other `case_slug` use (there are none in these three files
besides the filename strings; `out_dir` default lives in `hafs_case.py` and must
stay as `case_slug`).

- [ ] **Step 2: Add the init to the parent figure title**

In `analysis/parent_qpf.py`, the `plot_compare` function builds the figure
`suptitle`. It currently reads (text similar to):

```python
    fig.suptitle(
        f"{case.storm_name} — {case.model_label} parent QPF vs MRMS vs Stage IV "
        f"(0–{end_fhour}h, valid {valid_dt:%Y-%m-%d %HZ})",
        fontsize=13, y=1.01,
    )
```

Change the title text to include the init:

```python
    fig.suptitle(
        f"{case.storm_name} — {case.model_label} parent QPF vs MRMS vs Stage IV "
        f"(init {case.init_dt:%Y-%m-%d %HZ}, 0–{end_fhour}h, "
        f"valid {valid_dt:%Y-%m-%d %HZ})",
        fontsize=13, y=1.01,
    )
```

(If the exact existing wording differs, keep the existing wording and just insert
`init {case.init_dt:%Y-%m-%d %HZ}, ` before the `0–{end_fhour}h` part.)

- [ ] **Step 3: Verify compile + no stray filename slug remains**

Run: `python3 -m py_compile analysis/parent_qpf.py analysis/ets_full.py analysis/ets_score.py`
Expected: no output (exit 0).

Run: `grep -n "case_slug" analysis/parent_qpf.py analysis/ets_full.py analysis/ets_score.py`
Expected: NO matches (every output filename now uses `output_slug`; these files
have no other `case_slug` use).

- [ ] **Step 4: Confirm existing per-case tests still pass**

Run: `python3 analysis/tests/test_ets_full.py`
Expected: `5 passed` (pure helpers unaffected).

- [ ] **Step 5: Commit**

```bash
git add analysis/parent_qpf.py analysis/ets_full.py analysis/ets_score.py
git commit -m "Init-tag per-case output filenames; show init in parent title"
```

---

### Task 3: Comparison helpers `_check_same_init` and `_init_tag`

**Files:**
- Modify: `analysis/compare.py`
- Test: `analysis/tests/test_compare.py`

**Interfaces:**
- Consumes: existing `compare._slug`.
- Produces:
  - `_check_same_init(cases, case_paths)` — raises `ValueError` naming both paths + inits if `cases[0].init_dt != cases[1].init_dt`; returns None otherwise.
  - `_init_tag(label, init_dt) -> (slug, title)` — `slug = _slug(label)` with `init_dt.strftime('%Y%m%d%H')` appended unless already present; `title = f"{label} (init {init_dt:%Y-%m-%d %HZ})"`.

- [ ] **Step 1: Write the failing tests**

Add to `analysis/tests/test_compare.py` (it already imports `from compare import ...`;
extend that import to include `_check_same_init, _init_tag`):

```python
import types
from datetime import datetime
from compare import _check_same_init, _init_tag


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 analysis/tests/test_compare.py`
Expected: FAIL — `ImportError: cannot import name '_check_same_init'`.

- [ ] **Step 3: Implement the helpers**

In `analysis/compare.py`, add (place just above `_slug` or just below it):

```python
def _check_same_init(cases, case_paths):
    """Raise ValueError naming both if the two comparison cases' inits differ."""
    a, b = cases
    if a.init_dt != b.init_dt:
        raise ValueError(
            f"comparison cases must share an init: {case_paths[0]} is "
            f"{a.init_str}, {case_paths[1]} is {b.init_str}")


def _init_tag(label, init_dt):
    """(slug, title) tagged by init. slug = '<label-slug>_<YYYYMMDDHH>'
    (de-duplicated); title = '<label> (init YYYY-MM-DD HHZ)'."""
    init_str = init_dt.strftime("%Y%m%d%H")
    base = _slug(label)
    slug = base if init_str in base else f"{base}_{init_str}"
    title = f"{label} (init {init_dt:%Y-%m-%d %HZ})"
    return slug, title
```

(`_slug` already exists in compare.py as `label.lower().replace(" ", "_")`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 analysis/tests/test_compare.py`
Expected: `12 passed` (8 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add analysis/compare.py analysis/tests/test_compare.py
git commit -m "Add comparison init helpers (_check_same_init, _init_tag)"
```

---

### Task 4: Wire init-tagging into the comparison driver + replot

**Files:**
- Modify: `analysis/compare.py`
- Test: `analysis/tests/test_compare.py`

**Interfaces:**
- Consumes: `_check_same_init`, `_init_tag` (Task 3), existing `_read_rows`, `plot_categorical_compare`, `plot_fss_compare`, `from_yaml`, `score_matrix`.
- Produces:
  - `_plot_comparison(out_dir, slug, title, fss_plot_thresholds)` — reads the init-tagged CSVs and writes the two init-tagged PNGs (pure file-I/O + plotting, no case loading).
  - `generate_comparison` and `replot_from_csv` produce init-tagged filenames/titles and (generate) a leading `init` CSV column.

- [ ] **Step 1: Write the failing test (the pure plot helper)**

Replace the existing `test_replot_from_csv_regenerates_pngs` in
`analysis/tests/test_compare.py` with a test of the new `_plot_comparison`
helper (and extend the compare import to include `_plot_comparison`):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 analysis/tests/test_compare.py`
Expected: FAIL — `ImportError: cannot import name '_plot_comparison'`.

- [ ] **Step 3: Add `_plot_comparison` and rewire generate/replot**

In `analysis/compare.py`, add the pure plot helper (place near `replot_from_csv`):

```python
def _plot_comparison(out_dir, slug, title, fss_plot_thresholds):
    """Read the init-tagged CSVs and (re)draw the two PNGs. No case loading."""
    cat_rows = _read_rows(out_dir / f"compare_categorical_{slug}.csv", _CAT_NUM)
    fss_rows = _read_rows(out_dir / f"compare_fss_{slug}.csv", _FSS_NUM)
    plot_categorical_compare(cat_rows, title,
                             out_dir / f"compare_categorical_{slug}.png",
                             observation="MRMS")
    plot_fss_compare(fss_rows, title,
                     out_dir / f"compare_fss_{slug}.png",
                     observation="MRMS", forecast="parent",
                     plot_thresholds=tuple(float(t) for t in fss_plot_thresholds))
```

Replace the body of `replot_from_csv` with:

```python
def replot_from_csv(cfg):
    """Regenerate the comparison figures from the existing CSVs — no GRIB work.

    Loads the two case YAMLs (cheap; only parses the .atcfunix) to learn the
    shared init, then redraws the init-tagged figures.
    """
    cases = [from_yaml(p) for p in cfg["case_paths"]]
    _check_same_init(cases, cfg["case_paths"])
    slug, title = _init_tag(cfg["label"], cases[0].init_dt)
    _plot_comparison(cfg["out_dir"], slug, title, cfg["fss_plot_thresholds"])
    print(f"Replotted: {cfg['out_dir']}/compare_categorical_{slug}.png")
    print(f"Replotted: {cfg['out_dir']}/compare_fss_{slug}.png")
```

In `generate_comparison`, after the existing `_check_same_init` /
domain/grid/radius validation and after `cat_rows, fss_rows = score_matrix(...)`,
make these changes:

1. Right after loading `cases` (`cases = [from_yaml(p) for p in cfg["case_paths"]]`),
   add the init check and derive the tagged slug/title:
   ```python
   _check_same_init(cases, cfg["case_paths"])
   slug, title = _init_tag(cfg["label"], cases[0].init_dt)
   init_str = cases[0].init_str
   ```
   and DELETE the old `slug = _slug(cfg["label"])` line.
2. Stamp the init onto every row before writing:
   ```python
   for r in cat_rows:
       r["init"] = init_str
   for r in fss_rows:
       r["init"] = init_str
   ```
3. Add `"init"` as the FIRST entry of both column lists:
   ```python
   cat_cols = ["init", "model", "forecast", "observation", "threshold",
               "a", "b", "c", "d", "ets", "csi", "bias", "pod", "far", "hss"]
   fss_cols = ["init", "model", "forecast", "observation", "threshold",
               "scale_cells", "scale_km", "fss"]
   ```
4. The CSV/PNG filenames already use `slug` (now init-tagged), so
   `compare_categorical_{slug}.csv` etc. are correct unchanged.
5. Pass `title` (not `cfg["label"]`) to the two plot calls:
   ```python
   plot_categorical_compare(cat_rows, title, cat_png, observation="MRMS")
   plot_fss_compare(fss_rows, title, fss_png, observation="MRMS",
                    forecast="parent",
                    plot_thresholds=tuple(cfg["fss_plot_thresholds"]))
   ```

(If `_check_same_init` is now called twice — once in your new block and once if
it already existed — keep only one call. The init check must run before the
domain/grid/radius checks or right alongside them.)

- [ ] **Step 4: Run tests + compile**

Run: `python3 analysis/tests/test_compare.py`
Expected: `12 passed` (8 existing − 1 replaced + 1 new + 4 from Task 3 = 12).

Run: `python3 -m py_compile analysis/compare.py`
Expected: no output.

Run: `python3 -c "import sys; sys.path.insert(0,'analysis'); import compare; print('import OK')"`
Expected: `import OK`.

- [ ] **Step 5: Commit**

```bash
git add analysis/compare.py analysis/tests/test_compare.py
git commit -m "Init-tag comparison outputs (filenames, titles, init CSV column)"
```

---

### Task 5: README docs

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the init-tagging behavior from Tasks 1–4.
- Produces: user docs.

- [ ] **Step 1: Document init-tagging**

In `README.md`, add a short note. In the "Running a new storm" section (after the
outputs description), add:

```markdown
> **Init tagging:** every output filename is automatically stamped with the run's
> initialization time (e.g. `parent_qpf_helene_hfsa_2024092400.png`,
> `ets_full_helene_hfsa_2024092400.csv`), so a storm's many init times never
> overwrite each other. You don't need the init in the YAML name — it's added
> automatically (and de-duplicated if you do include it).
```

In the "Comparing HFSA vs HFSB" section, add:

```markdown
The two cases must share an initialization time (the comparison errors if they
don't). The comparison outputs are init-tagged too
(`compare_categorical_hurricane_helene_2024092400.png`), and each CSV has a
leading `init` column.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document init-tagged output filenames"
```

---

## Self-Review

**Spec coverage:**
- `StormCase.output_slug` (append + de-dup) → Task 1. ✓
- Per-case parent/ets/ets_score filenames use output_slug + parent title init → Task 2. ✓
- Comparison `_check_same_init` + `_init_tag` → Task 3. ✓
- Comparison wiring: init-tagged slug, init CSV column, titles, `_plot_comparison`, replot → Task 4. ✓
- README docs → Task 5. ✓
- CSV column orders with leading `init` → Task 4 (verbatim from Global Constraints). ✓
- `out_dir` default unchanged → Task 2 Step 1 explicit note. ✓

**Placeholder scan:** No "TBD"/vague steps; every code step shows the code, every
test step shows the test. Task 2 Step 2 hedges on exact title wording but gives a
concrete insertion rule and the full replacement.

**Type consistency:** `output_slug` (property), `_check_same_init(cases, case_paths)`,
`_init_tag(label, init_dt) -> (slug, title)`, `_plot_comparison(out_dir, slug,
title, fss_plot_thresholds)` are used identically across Tasks 3–4. CSV column
lists match between Task 4 wiring and the `_plot_comparison`/`_read_rows` readers
(init is a string column, not in `_CAT_NUM`/`_FSS_NUM`, so it's read back as-is).

## Notes on local vs Hercules testing

- Tasks 1, 3, 4 are fully unit-tested locally with `python3` (`output_slug`,
  `_init_tag`, `_check_same_init`, `_plot_comparison`).
- Task 2 edits heavy modules; verified locally by `py_compile` + the `grep`
  check (no stray `case_slug` in output names) + the unaffected `ets_full`
  helper tests. Full integration is the Hercules run: per-case and comparison
  runs produce `*_2024092400.*` files, and a second init produces `*_2024092412.*`
  without overwriting.
