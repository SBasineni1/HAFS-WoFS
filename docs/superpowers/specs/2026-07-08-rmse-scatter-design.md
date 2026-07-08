# RMSE Scatter Product — Design

**Date:** 2026-07-08
**Status:** Approved

## Goal

Add storm-total RMSE (and companion continuous metrics) to the case-driven
verification framework, comparing HAFS forecast rainfall against observed
rainfall (MRMS QPE and NCEP Stage IV QPE). Like every other product, it must
run from a storm YAML with one command and produce a graphic output:

```bash
python analysis/run.py storms/<case>.yaml rmse   # RMSE product only
python analysis/run.py storms/<case>.yaml all    # parent map + ETS + RMSE
```

The existing framework scores only categorical metrics (ETS, bias, POD, FAR,
CSI, HSS, FSS), which threshold rainfall into yes/no events. RMSE is the
missing continuous metric: how far off the forecast rainfall *amounts* are,
in mm, over the TC swath.

## Scope

- Storm-total RMSE (one number per forecast × observation pair), computed on
  the event-total rainfall fields — the same fields and swath the ETS uses.
- Forecast fields: HAFS parent (6-km) and 2-km nest event totals.
- Observation fields: MRMS QPE and Stage IV QPE event totals.
- Graphic: forecast-vs-observed scatter panels (hexbin), one per pair.
- Out of scope (possible follow-up): RMSE in the HFSA-vs-HFSB `compare`
  mode; RMSE as a function of lead time.

## Components

### 1. Continuous metrics — `analysis/skill_metrics.py`

New function:

```python
def continuous_scores(fcst, obs) -> dict
# returns {n, rmse, mae, bias, r}
```

- Operates on 1-D arrays of already-selected valid points (the caller applies
  the same selection `ets_full.score_pair` uses: `swath & finite(obs) &
  finite(fcst)`, values zero-filled). Categorical and continuous scores
  therefore describe the identical footprint.
- `rmse = sqrt(mean((f - o)^2))`, `mae = mean(|f - o|)`,
  `bias = mean(f - o)` (positive = over-forecast),
  `r` = Pearson correlation, NaN when either field has zero variance or
  `n < 2`. All metrics NaN when `n == 0`.

### 2. Shared field builder — refactor inside `analysis/ets_full.py`

Extract the expensive field-building block of `compute_ets` into:

```python
def build_verification_fields(case) -> dict
# keys: max_fhour, grid_lat, grid_lon, nest_total, apcp_mode,
#        parent_total, mrms_total, stage4_grid, s4_label, swath
```

`compute_ets(case, fields=None)` builds fields only when not handed in.
The builder stays in `ets_full.py` because `compare.py` and the tests already
import shared helpers from there (`score_pair`, `hafs_parent_total`,
`stage4_on_fixed`, `regrid_2d_to_fixed`) — nothing they import moves, so no
import churn and no circular imports.

### 3. New product — `analysis/rmse_scatter.py`

```python
def compute_rmse(case, fields=None)
```

- Builds (or receives) the verification fields.
- For each forecast × observation pair — (parent, nest) × (MRMS, Stage IV) —
  selects valid swath points and computes `continuous_scores`.
- Prints a console table (forecast, observation, n, RMSE, MAE, bias, r) in
  the same style as the ETS run.
- Renders one figure: 2×2 hexbin scatter panels (rows = parent/nest,
  cols = MRMS/Stage IV); 2×1 when Stage IV is unavailable.
  - Hexbin with log-scaled counts (a raw scatter is a blob at ~10⁵ points).
  - 1:1 diagonal line; equal x/y limits shared across panels, 0 to the max
    of all fields, axes labeled in mm.
  - Annotation box per panel: RMSE, MAE, bias, r, n.
  - Title matches the ETS figure style: storm, model, 0–max_fhour window,
    init, swath radius. Caveat footer for Stage IV (same text as ETS).
- Outputs to `case.out_dir`, init-tagged via `case.output_slug`:
  - `rmse_scatter_<slug>.png`
  - `rmse_<slug>.csv` — columns: forecast, observation, n, rmse, mae, bias, r

### 4. Entry point — `analysis/run.py`

- `COMMANDS` gains `"rmse"`; usage/docstring updated.
- Dispatch:
  - `rmse` → `compute_rmse(case)`
  - `all` → parent figure, then `fields = build_verification_fields(case)`
    once, passed to both `compute_ets(case, fields)` and
    `compute_rmse(case, fields)`. Parent QPF keeps its own display pipeline,
    unchanged.

## Error handling

- Stage IV unavailable → MRMS-only panels, caveat footer, "Stage IV
  unavailable — not scored" printed (mirrors ETS behavior).
- No MRMS hours loadable → hard error from `build_mrms_total` (existing
  behavior, unchanged).
- A pair with zero valid points → NaN scores in CSV, panel annotated
  "no valid points".

## Testing

New `analysis/tests/test_rmse_scatter.py`, in the existing standalone style
(`python3 analysis/tests/test_rmse_scatter.py`):

- `continuous_scores` on hand-computed arrays → exact RMSE/MAE/bias/r.
- Edge cases: empty arrays (all NaN), constant fields (r is NaN), perfect
  forecast (rmse = mae = bias = 0, r = 1 for varying fields).
- Scoring path of `compute_rmse` driven by a small synthetic fields dict
  (tiny grids, known values); figure rendered to a temp dir.
- Existing `test_ets_full.py` still passes after the `compute_ets` refactor
  (its imports are untouched).

## Documentation

- README: add `rmse` to the running section and the outputs list; note the
  metrics definitions (esp. bias sign convention).
- CLAUDE.md run-modes line updated (`[parent|ets|rmse|all]`).
