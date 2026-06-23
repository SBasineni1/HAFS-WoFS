# Task 9 Report: Comparison driver (`compare.generate_comparison`)

## Implementation Status
✅ COMPLETE

All three functions added to `analysis/compare.py`:
- `_slug(label)` — converts label to CSV filename slug (lowercase, spaces→underscore)
- `_build_model_fields(case, grid_lat, grid_lon, max_fhour)` — loads parent+nest forecasts and MRMS+Stage IV observations onto common grid
- `generate_comparison(cfg)` — main driver that validates cases, builds shared swath once, scores both models, writes CSVs and plots

Imports added (6 new imports from reused modules):
- `from hafs_case import from_yaml`
- `from best_track import parse_bdeck`
- `from skill_metrics import swath_from_track`
- `from hafs_common import discover_files, hafs_event_total`
- `from ets_full import hafs_parent_total, stage4_on_fixed`
- `from ets_score import build_mrms_total`

## Verification Results

### 1. Compile check
```bash
python3 -m py_compile analysis/compare.py
```
✅ Exit 0 (no errors)

### 2. Import check
```bash
python3 -c "import sys; sys.path.insert(0,'analysis'); import compare; print('compare imports OK')"
```
✅ Output: `compare imports OK`

### 3. Existing tests
```bash
python3 analysis/tests/test_compare.py
```
✅ Output: `5 passed` (all existing tests still pass; driver not exercised by tests)

## Key Implementation Details

### CSV Column Orders
**Categorical** (compare_categorical_<slug>.csv):
`model, forecast, observation, threshold, a, b, c, d, ets, csi, bias, pod, far, hss`

**FSS** (compare_fss_<slug>.csv):
`model, forecast, observation, threshold, scale_cells, scale_km, fss`

Both use `csv.DictWriter(..., extrasaction="ignore")` as specified.

### Validation
- Cases must share `domain`, `grid_res`, and `mask_radius_km` (raises ValueError with both case paths if not)
- Aborts gracefully if either case has no storm files
- Swath built once from best track, passed to `score_matrix` for both models

### Output
Writes to `cfg["out_dir"]` (defaults to `analysis/output/<config_stem>`):
- `compare_categorical_<slug>.csv`
- `compare_fss_<slug>.csv`
- `compare_categorical_<slug>.png` (3-panel plot: ETS, CSI, bias vs threshold)
- `compare_fss_<slug>.png` (FSS vs neighborhood scale)

Prints summary including ETS at 25 mm and 50 mm thresholds for each model.

## Files Changed
- `analysis/compare.py` — +103 lines (6 imports + 3 functions)

## Concerns
None. Module imports successfully; all existing tests pass; implementation follows the brief exactly. CSV column orders verified. Function signatures match interfaces. Ready for integration into `run.py` (Task 10).

## Next Tasks
- Task 10: Wire `generate_comparison` into `run.py`
- Task 11+: Example comparison YAML and execution on Hercules

---

## Final-review fixes (2026-06-23)

Three fixes from the whole-branch code review, applied in a single commit.

### Fix 1 (IMPORTANT): Common `n` across models for every forecast type

**Problem:** `score_matrix` scored each model independently via `swath & isfinite(fcst) & isfinite(obs)`, so HFSA-nest and HFSB-nest could have different `n` because their moving-nest footprints differ (NaN outside each model's footprint).

**Fix:** Rewrote `score_matrix` to compute a COMMON mask per (forecast, observation) pair: `swath & intersection-over-all-models(isfinite(forecast)) & isfinite(obs)`. Every model is scored over this identical mask, guaranteeing equal `n`. Loop order changed to forecast -> obs -> model (row ORDER changes, but plots/CSVs filter by column so no impact).

**File:** `analysis/compare.py` lines 54-95

### Fix 2 (MINOR): FSS empty-mask guard

**Problem:** `fractions_skill_score` would raise a numpy RuntimeWarning ("mean of empty slice") if `mask` was all-False.

**Fix:** Added `if not mask.any(): return np.nan` after binarizing, before computing MSE.

**File:** `analysis/skill_metrics.py` line 42

### Fix 3 (MINOR): Clarify threshold fallback

**Problem:** The line `thresholds = cfg["thresholds_mm"] or a.thresholds_mm` had no comment explaining the fallback logic.

**Fix:** Added comment: `# None or empty list -> fall back to the cases' default thresholds.`

**File:** `analysis/compare.py` line 197

### New test

Added `test_score_matrix_common_n_across_models_when_footprints_differ` to `analysis/tests/test_compare.py`. Two models with different NaN footprints verified to produce identical `n == 24` (the intersection).

### Documentation updates

- `docs/superpowers/specs/2026-06-23-hfsa-hfsb-comparison-design.md` — added that both models are scored over COMMON coverage so `n` is identical for every forecast type including the nest.
- `README.md` — section 5 now notes the nest comparison uses common coverage of both nests.

### Verification commands and output

```
$ python3 analysis/tests/test_compare.py
PASS test_load_comparison_defaults
PASS test_load_comparison_requires_best_track
PASS test_load_comparison_requires_two_cases
PASS test_plots_write_png_files
PASS test_score_matrix_common_n_across_models_when_footprints_differ
PASS test_score_matrix_shapes_and_perfect_fss

6 passed

$ python3 analysis/tests/test_skill_metrics.py
PASS test_fss_disjoint_events_is_zero_at_scale1
PASS test_fss_identical_fields_is_one
PASS test_fss_no_events_is_nan
PASS test_swath_from_track_marks_points_within_radius

4 passed

$ python3 -m py_compile analysis/compare.py analysis/skill_metrics.py
(exit 0)

$ python3 -c "import sys; sys.path.insert(0,'analysis'); import compare; print('import OK')"
import OK
```

### Commit

`71398c0` — Fix score_matrix common-n fairness for nest, FSS empty-mask guard, threshold comment
