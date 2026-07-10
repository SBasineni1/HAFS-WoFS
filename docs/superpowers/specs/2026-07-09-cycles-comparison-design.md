# Cycle Comparison Product — Design

**Date:** 2026-07-09
**Status:** Approved

## Goal

Compare HAFS model runs across initializations (00z/06z/12z/18z over the
days before landfall) for a single storm, quantitatively and qualitatively.
Like every other product, it runs from one YAML with one command and
produces graphic output:

```bash
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles
```

The existing framework scores one initialization per case. This product
answers: *how did the forecast of the landfall rainfall change as
initializations approached landfall?*

## Scientific decisions (settled during brainstorming)

1. **Common valid window.** Every cycle is scored on its QPF accumulated
   over the same absolute time window (e.g., 2024-09-26 00Z → 2024-09-28
   00Z), verified against the same observed total. The only thing changing
   between cycles is lead time.
2. **Union footprint.** One shared swath mask, built from the union of
   every included cycle's forecast track positions inside the valid window
   (within `mask_radius_km`). Every cycle is scored over the identical set
   of grid points; a run with a bad track is penalized for raining in the
   wrong place — that is part of what is being measured.
3. **Full-coverage eligibility.** A cycle is included only if
   `init ≤ valid_start` and `init + max forecast hour ≥ valid_end`.
   Skipped cycles are printed with the reason, never silently dropped.

## Configuration

New storm-level YAML (one per storm × model):

```yaml
run_root: /work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/HFSA
valid_start: 2024092600        # common window start (UTC, YYYYMMDDHH)
valid_end:   2024092800        # common window end
storm_name: Hurricane Helene
domain: [15.0, 42.0, -100.0, -60.0]
mask_radius_km: 500
out_dir: analysis/output/helene_hfsa_cycles
# inits: [2024092400, 2024092412]   # optional; overrides auto-discovery
# ets_threshold_mm: 25              # optional; headline threshold, default 25
```

- **Auto-discovery:** inits are the `YYYYMMDDHH`-named subdirectories of
  `run_root`, sorted. An explicit `inits:` list overrides discovery.
- Each init becomes a per-cycle `StormCase` via the existing constructor
  path with `run_dir = run_root/<init>` and `init` set explicitly, so
  track parsing and model-label auto-detection work unchanged.
- A cycles YAML is distinguished by the presence of `run_root`. Running
  `cycles` on a per-init YAML, or another command on a cycles YAML, fails
  with a clear message.

## Architecture (Approach A)

Thin orchestrator in a new `analysis/cycles.py`; small *windowed* variants
of the existing loaders live next to the originals. Observations are
absolute-time, so the windowed MRMS/Stage IV totals and the union swath are
computed **once** and shared by every cycle; only the forecast window
extraction runs per cycle. Scoring reuses `score_pair`,
`contingency_scores`, and `continuous_scores` unchanged.

### Windowed fields

Per cycle: `f1 = valid_start − init`, `f2 = valid_end − init` (hours).

- **Nest:** windowed variant of `hafs_common.hafs_event_total`,
  per-interval APCP mode only → sum of regridded intervals with fhour in
  `(f1, f2]`. Accumulated-mode differencing is **intentionally not
  implemented**: nest cumulative records are a geographic trap on the
  moving nest (see `pick_total_record`), which never selects them — so a
  window whose files yield no per-interval buckets raises `RuntimeError`
  and the cycle is skipped with a printed reason rather than silently
  scoring an all-zeros forecast. (Decision recorded at final review,
  2026-07-09.)
- **Parent:** difference of the cumulative `tp` records ending at `f2`
  and `f1` (from `parent_qpf.read_hafs_tp_records`), regridded to the
  fixed mesh; `f1 == 0` → just the `f2` record.
- **MRMS:** windowed total over `[valid_start, valid_end]` on the fixed
  mesh — computed once (a windowed variant of `build_mrms_total` that
  takes absolute start/end datetimes).
- **Stage IV:** touched-days total for the window (windowed variant of
  `stage4_total`); keeps the CONUS-only, 12Z–12Z caveat text. `None`
  when unavailable → MRMS-only scoring, caveat footer (existing pattern).
- **Swath:** union over included cycles of each run's track positions
  with valid time inside the window, radius `mask_radius_km` — one shared
  boolean mask on the fixed mesh.

### Scoring

Per cycle × forecast (parent, nest) × observation (MRMS, Stage IV if
available), over the shared swath with the same valid-point selection as
`score_pair` (`swath & finite(obs) & finite(fcst)`, zero-filled):

- continuous: n, RMSE, MAE, bias (positive = over-forecast), Pearson r
- categorical: ETS/bias/POD/FAR/CSI/HSS at the case thresholds

Output `<slug>` for a cycles case is the YAML stem plus the window
(e.g., `helene_hfsa_cycles_2024092600_2024092800`), so re-running with a
different window never overwrites earlier output.

One CSV, `cycles_<slug>.csv`, long format: columns
`init, forecast, observation, threshold, n, rmse, mae, bias_mm, r, a, b,
c, d, ets, bias, pod, far, csi, hss` — continuous metrics repeated on each
threshold row of that (init, forecast, observation) group (matching the
ETS CSV's long-format style; `bias_mm` is the continuous mean error, to
avoid colliding with the categorical frequency `bias`).

### Graphics

1. **`cycles_metrics_<slug>.png`** — init time on the x-axis, three
   stacked panels: RMSE (mm), bias (mm, zero reference line), ETS at the
   headline threshold (`ets_threshold_mm`, default 25). Line conventions
   reused from the ETS figure: observation → color, forecast →
   linestyle/marker. Title: storm, model, valid window, mask radius.
   Stage IV caveat footer when applicable.
2. **`cycles_maps_<slug>.png`** — small-multiple maps: each cycle's
   **nest** windowed QPF plus a final MRMS panel, shared colorbar and
   domain extent, union-swath outline on every panel, panel titles = init
   time (MRMS panel titled "MRMS observed"). Grid layout wraps at 4
   columns.

## Entry point

`run.py <cycles.yaml> cycles` → `cycles.compute_cycles(cycles_case)`.
`COMMANDS` gains `"cycles"`; usage/docstring updated. `cycles` is not part
of `all` (it is a storm-level product, not a per-init one).

## Error handling

- No eligible cycles (none discovered, or none covering the window) →
  `RuntimeError` naming the window and the inits inspected.
- A cycle whose GRIB files are missing mid-window → skipped with a
  printed reason; remaining cycles proceed.
- Stage IV unavailable → MRMS-only, caveat footer, "Stage IV unavailable —
  not scored" printed.
- A (cycle, pair) with zero valid points → NaN scores in the CSV; the
  metrics plot simply has no marker at that init.

## Testing

New `analysis/tests/test_cycles.py`, existing standalone style
(`python3 analysis/tests/test_cycles.py`, also pytest-compatible):

- window arithmetic: f1/f2 from init and window; eligibility filtering
  (too-late init, too-short run) with hand-picked datetimes
- init auto-discovery from a temp directory tree (valid + junk names);
  explicit `inits:` override
- union swath on a tiny mesh: two synthetic tracks → mask equals the
  union of the individual masks
- windowed nest accumulation, both APCP modes, on tiny synthetic grids
  with known values (including the NaN-at-f1 rule)
- `compute_cycles` scoring path driven by synthetic fields: CSV rows and
  exact init-tagged output filenames in a temp dir; both figures render

No HPC data needed by any test.

## Documentation

- README: `cycles` command, the cycles YAML schema, outputs, and a "How to
  read the cycle comparison" section (common window, union footprint,
  eligibility rule).
- CLAUDE.md run-modes line gains `cycles`.

## Out of scope (possible follow-ups)

- HFSA-vs-HFSB cycle comparison (two run roots on one figure)
- FSS or track error as a function of lead time
- Parent maps in the small-multiples
