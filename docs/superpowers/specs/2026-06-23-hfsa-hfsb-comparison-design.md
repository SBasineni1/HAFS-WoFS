# HFSA-vs-HFSB head-to-head comparison — design

**Date:** 2026-06-23
**Status:** Approved (design)

## Goal

Add a fair, head-to-head comparison of two HAFS configurations (HFSA vs HFSB)
for the same storm, using additional statistical methods beyond the single ETS
curve the per-case pipeline already produces. The two models must be scored over
an identical verification footprint so "which model did better" is not confounded
by each model being graded on a different region.

The new statistics, chosen with the user:
- **Categorical** companions from the contingency table (ETS already; add CSI,
  frequency bias, POD, FAR, and HSS).
- **FSS** (Fractions Skill Score) — neighborhood/spatial skill vs scale.

The fair-comparison footprint, chosen with the user: a single shared swath built
from the **NHC best track** (observed), so both models are verified over where
the storm actually went.

This is additive: the existing per-case path (`run.py <case.yaml> parent|ets|all`)
is untouched.

## Background / problem

Today each model's ETS is scored over its OWN forecast-track swath (≤500 km of
where that model placed the storm), so HFSA and HFSB cover different regions and
different point counts (observed: n=129,803 vs 117,953). That is a confound for a
direct A-vs-B comparison. The fix is one shared swath, from the best track, used
for both models.

## Architecture

### New files

- **`analysis/best_track.py`**
  - `parse_bdeck(path) -> list[(datetime, lat, lon)]` — parse an NHC ATCF
    **b-deck** file (the `BEST` lines). Column layout matches the existing
    `.atcfunix` parser EXCEPT the fix time is column index 2 (`YYYYMMDDHH`)
    directly, because a b-deck's `TAU` (col 5) is always 0. Decode lat/lon with
    `hafs_case.decode_latlon`. Dedupe by time, sort ascending. Raise a clear
    error if no `BEST` fixes parse.

- **`analysis/skill_metrics.py`**
  - `swath_from_track(track, grid_lat, grid_lon, radius_km, init_dt, max_fhour)
    -> bool array` — union of `radius_km` circles along the track, interpolated
    hourly across `[init_dt, init_dt + max_fhour h]`. Reuses
    `hafs_common.haversine_km` and `hafs_case.position_on_track`. If the track
    ends before `init_dt + max_fhour`, the interpolation clamps to the last fix
    (i.e. circles stop advancing) and a warning is printed.
  - `fractions_skill_score(fcst, obs, threshold, scale, mask) -> float` — FSS at
    one (threshold, neighborhood size in cells):
    1. Binarize `fcst >= threshold` and `obs >= threshold`.
    2. Fractional fields via `scipy.ndimage.uniform_filter(size=scale)` on the
       binary fields.
    3. `FSS = 1 - MSE(Mf, Of) / (mean(Mf^2) + mean(Of^2))`, with all means taken
       over `mask` (the shared swath) points only. Returns NaN if the
       denominator is 0 (no events anywhere at that threshold).

- **`analysis/compare.py`**
  - `generate_comparison(cfg)` — the driver (see Data flow). `cfg` is the loaded
    comparison config dict.
  - plotting helpers `plot_categorical_compare(...)`, `plot_fss_compare(...)`.
  - `__main__`: `generate_comparison(load_comparison(sys.argv[1]))`.

- **`storms/helene_compare.yaml`** — example comparison config.

### Touches to existing files

- **`analysis/ets_score.py`** — add `hss` to `contingency_scores`:
  `hss = 2*(a*d - b*c) / ((a+c)*(c+d) + (a+b)*(b+d))` (NaN if denom 0). This is
  the only categorical companion not already returned; the rest (csi, bias, pod,
  far) are already there. Benefits the per-case path too.

- **`analysis/hafs_case.py`** — extract the track interpolation from
  `StormCase.position_at` into a module-level free function
  `position_on_track(track, valid_dt) -> (lat, lon)` (linear interp, clamps to
  endpoints). `StormCase.position_at` becomes `return position_on_track(self.track, valid_dt)`.
  No behavior change; lets the best track reuse the same interpolation.

- **`analysis/run.py`** — add `compare` to `COMMANDS`. When `command == "compare"`,
  `main` treats `argv[0]` as a comparison-config YAML (not a single case): it
  calls `compare.load_comparison(path)` then `compare.generate_comparison(cfg)`,
  instead of `from_yaml` + `dispatch`. The existing `parent|ets|all` path is
  unchanged.

### Comparison config (`storms/helene_compare.yaml`)

```yaml
label: Hurricane Helene
cases:
  - storms/helene_hfsa.yaml
  - storms/helene_hfsb.yaml
best_track: /work2/.../bal092024.dat       # NHC ATCF b-deck
out_dir: analysis/output/helene_compare      # optional; default analysis/output/<config-stem>
# optional overrides:
# thresholds_mm: [1,5,10,25,50,75,100,150,200,250]
# fss_scales_cells: [1,3,5,11,21,41]
# fss_plot_thresholds: [10,25,50]
```

`load_comparison(path)` reads this YAML, requires `cases` (exactly 2) and
`best_track`, fills defaults (`out_dir = analysis/output/<stem>`,
`thresholds_mm` from the first case, `fss_scales_cells = [1,3,5,11,21,41]`,
`fss_plot_thresholds = [10,25,50]`), and returns a dict including the two loaded
`StormCase` objects.

One-time best-track fetch (documented in README):
```
wget https://ftp.nhc.noaa.gov/atcf/archive/2024/bal092024.dat.gz && gunzip bal092024.dat.gz
```

## Data flow (`generate_comparison`)

1. Load both `StormCase`s from `cfg["cases"]`. Require matching `domain` and
   `grid_res`; error naming both if they differ. Common grid = first case's
   `fixed_grid()`.
2. Parse best track (`parse_bdeck`). `max_fhour` = min of the two cases' last
   forecast hour (both 126 for Helene).
3. Build the shared swath: `swath_from_track(best_track, grid_lat, grid_lon,
   mask_radius_km, init_dt, max_fhour)`. `init_dt` / `mask_radius_km` taken from
   the cases (identical); error if the two cases disagree on `mask_radius_km`.
4. For each case (model): build onto the common grid
   - parent total — `parent_qpf` parent record → `regrid_2d_to_fixed`
     (reuse `ets_full.hafs_parent_total`)
   - nest total — `hafs_common.hafs_event_total` (the slow griddata step; runs
     once per model, so the driver is ~2× a single `ets` run)
   - MRMS total — `ets_score.build_mrms_total`
   - Stage IV — `parent_qpf.stage4_total` → `regrid_2d_to_fixed`
     (None if unavailable)
5. Score the full matrix over the shared swath:
   model {HFSA,HFSB} × forecast {parent,nest} × obs {MRMS, Stage IV present?}:
   - categorical: `contingency_scores` at each `thresholds_mm`
   - FSS: `fractions_skill_score` at each threshold × `fss_scales_cells`
6. Write CSVs + plots, print a compact summary.

## Outputs (in `out_dir`)

- **`compare_categorical_<label>.png`** — 3 panels (ETS, CSI, frequency bias) vs
  threshold (log x). Lines: HFSA vs HFSB by color, parent vs nest by
  solid/dashed; obs = MRMS (headline). 4 lines/panel.
- **`compare_fss_<label>.png`** — FSS vs neighborhood scale (km) at
  `fss_plot_thresholds`; HFSA vs HFSB by color; parent forecast vs MRMS.
- **`compare_categorical_<label>.csv`** — full matrix:
  `model, forecast, observation, threshold, a, b, c, d, ets, csi, bias, pod,
  far, hss`.
- **`compare_fss_<label>.csv`** — full matrix:
  `model, forecast, observation, threshold, scale_cells, scale_km, fss`.
- **Console** — compact HFSA-vs-HFSB summary: ETS at 25 & 50 mm and FSS at
  25 mm / ~100 km, per model, for parent vs MRMS.

`<label>` is `cfg["label"]` slugified. Headline plots use parent vs MRMS for
legibility; the full matrix (Stage IV, nest) lives in the CSVs.

## Error handling

- best_track path missing / no BEST fixes parse → clear error naming the file.
- best track shorter than the forecast window → swath interpolation clamps to
  the last fix; print a warning with the gap.
- cases mismatched on `domain` / `grid_res` / `mask_radius_km` → error naming
  both cases and the field.
- `cases` not exactly 2, or `best_track` missing in config → validation error.
- Stage IV unavailable for a model → that model's Stage IV columns are dropped
  (matches existing behavior); comparison proceeds on MRMS.

## Testing

Local unit tests (numpy/scipy/stdlib only, no GRIB):
- `parse_bdeck` vs a small b-deck fixture (`analysis/tests/fixtures/bal092024_sample.dat`):
  times come from column 3 (not init+tau), lat/lon decode, sorted/deduped.
- `fractions_skill_score`: identical fields → 1.0; disjoint events → ~0; a small
  hand-computed case at one scale.
- `hss` in `contingency_scores`: perfect forecast (b=c=0) → 1.0; a known 2×2.
- `swath_from_track`: a tiny grid + 2-point track → points within radius True,
  far points False; clamp behavior when track ends early.
- `position_on_track` (refactored): midpoint interpolation + endpoint clamp
  (the existing `StormCase.position_at` tests still pass unchanged).

The `compare.py` driver (GRIB/MRMS/Stage IV) is integration-only, verified on
Hercules: `python analysis/run.py storms/helene_compare.yaml compare` produces
the two PNGs + two CSVs with identical `n` for HFSA and HFSB.

## Out of scope (YAGNI)

- Auto-downloading the best track (user supplies the b-deck path; documented wget).
- Continuous-amount metrics (RMSE/MAE/correlation) — not selected.
- More than two models in one comparison.
- Statistical significance testing of the A-vs-B difference (could be a later add).
- Object-based metrics (SAL/MODE).

## Compatibility

Purely additive. New entry is `run.py … compare <comparison.yaml>`. The only
edits to existing modules are: one `hss` line in `ets_score`, a no-op refactor of
`position_at` in `hafs_case`, and a new `compare` branch in `run.py`. The
per-case `parent|ets|all` path and all existing tests are unaffected.
