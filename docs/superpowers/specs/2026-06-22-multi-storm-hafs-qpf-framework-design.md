# Multi-storm HAFS QPF/ETS framework — design

**Date:** 2026-06-22
**Status:** Approved (design)

## Goal

Turn the Helene/HFSA-specific QPF and ETS scripts into a reusable framework so
that running **any storm** and **any HAFS output** (HFSA or HFSB) requires only
dropping in a small YAML case file — primarily changing the run directory path.
The motivation is to streamline new-storm analysis and let other workers use the
code without editing Python.

Two products must be reproducible per case, identical in look to the current
Helene outputs:

1. **3-panel QPF figure** — HAFS parent APCP vs MRMS QPE vs NCEP Stage IV QPE
   (`parent_qpf.py`).
2. **ETS-vs-threshold figure + CSV** — parent/nest forecasts vs MRMS & Stage IV
   over the TC swath (`ets_full.py`).

The full-run animation (`qpf_full_run.py`) and single-obs ETS (`ets_score.py`)
are refactored the same way so the whole toolkit is case-driven.

## Problem with the current code

Every script hangs off module-level constants defined in `qpf_full_run.py` and
imported across the others:

- `HAFS_RUN_DIR`, `INIT_STR`, `INIT_DT`, `FILE_GLOB`, `FHOURS_FILTER`
- `FIXED_DOMAIN`, `GRID_RES`, `TC_MASK_RADIUS_KM`
- `OUT_DIR`, `MRMS_CACHE_DIR`, `CFGRIB_IDX_DIR`
- `TC_TRACK_6H` (22 hardcoded NHC best-track points) — read by
  `tc_position_at()`, which the swath mask in every script depends on
- Storm name / "HAFS-A" labels are hardcoded in titles and output filenames

So a new storm currently means hand-editing constants in multiple files,
including transcribing a best track. The framework removes all of that.

## Key decisions (from brainstorming)

- **Track source:** parse the HAFS `.atcfunix` track file that already ships in
  the run directory. No manual track entry, no network needed. The atcfunix
  fix records decode to exactly the values currently hardcoded for Helene
  (`168N`/`832W` at lead hour `TAU` → `16.8, -83.2`), so behavior is preserved.
- **Config style:** one YAML case file per storm. `run_dir` is the only required
  field; everything else is optional and auto-derived if omitted.
- **HFSA vs HFSB:** auto-detected from the run-dir path / filenames
  (`HFSA` → "HAFS-A", `HFSB` → "HAFS-B").
- **Refactor approach:** introduce a `StormCase` config object threaded through
  the functions (no runtime global mutation, no hidden state). Refactor the four
  existing scripts **in place**.

## Architecture

### New module: `analysis/hafs_case.py`

The framework core. Holds all per-storm state and the track logic that used to
live in `qpf_full_run.py`.

`StormCase` dataclass fields:

| field | type | source if omitted in YAML |
|-------|------|---------------------------|
| `run_dir` | `Path` | **required** |
| `init_dt` | `datetime` | from atcfunix warning time (col 3) |
| `storm_name` | `str` | from atcfunix storm name field, title-cased |
| `model_label` | `str` ("HAFS-A"/"HAFS-B") | auto-detected from `run_dir` path |
| `domain` | `(lat_min, lat_max, lon_min, lon_max)` | auto from track bbox + pad |
| `grid_res` | `float` | `0.05` |
| `mask_radius_km` | `float` | `500.0` |
| `display_radius_km` | `float` | `750.0` (parent figure display only) |
| `thresholds_mm` | `list[int]` | `[1,5,10,25,50,75,100,150,200,250]` |
| `out_dir` | `Path` | `analysis/output/<case_slug>` |
| `mrms_cache_dir` | `Path` | `/tmp/mrms_cache` |
| `stage4_cache_dir` | `Path` | `/tmp/stage4_cache` |
| `fhours_filter` | `list[int] | None` | `None` |
| `track` | `list[(datetime, lat, lon)]` | parsed from atcfunix |
| `case_slug` | `str` | YAML filename stem (used to namespace output) |

Methods:

- `position_at(valid_dt) -> (lat, lon)` — linear track interpolation (moved
  verbatim from `qpf_full_run.tc_position_at`, now reading `self.track`).
- `fixed_grid() -> (grid_lat, grid_lon)` — builds the fixed mesh from `domain`
  and `grid_res` (logic currently duplicated in `ets_score`/`ets_full`).
- `parent_glob()` / `storm_glob()` — file globs built from the init string
  (`**/*{init}*parent.atm.f*.grb2`, `**/*{init}*storm.atm.f*.grb2`).

Module functions:

- `from_yaml(path) -> StormCase` — load YAML, locate + parse the atcfunix track,
  auto-detect model, auto-derive domain/init/name where absent, apply defaults,
  set `case_slug` from the YAML stem.
- `parse_atcfunix(path) -> (storm_name, init_dt, track)` — parse the
  comma-separated ATCF fixes: basin/cy/warn-time/`TAU`/lat/lon. Lat/lon are
  tenths-of-degree with N/S/E/W suffix. Build one `(init_dt + TAU, lat, lon)`
  point per lead time; dedupe to 6-hourly to match current behavior.
- `find_atcfunix(run_dir) -> Path` — locate the track file under the run dir
  (glob `*.atcfunix` / `*.trak*.atcfunix`).
- `detect_model(run_dir) -> str` — "HAFS-A"/"HAFS-B" from path substring;
  default "HAFS" if neither matches.
- `auto_domain(track, pad_deg=2.0) -> tuple` — track lat/lon bounding box padded
  by `pad_deg`, used only when YAML omits `domain`.

### New entry point: `analysis/run.py`

```
python analysis/run.py <case.yaml> [parent|animation|ets|all]
```

- Loads `StormCase.from_yaml(case_yaml)`.
- Dispatches to `generate_parent_figure(case)`, `generate_animation(case)`,
  and/or `compute_ets(case)`.
- Default subcommand: `all`.
- Prints a one-line case summary (storm, model, init, domain, track length)
  before running so the user can sanity-check auto-detection.

### Refactors (in place)

All four scripts lose their module-level config block and `main()`. Each gains a
function taking `case`:

- **`qpf_full_run.py`**
  - Remove the CONFIG block + `TC_TRACK_6H` (moved to `hafs_case`).
  - `tc_position_at` → use `case.position_at`; `apply_tc_mask(...)` gains a
    `case` argument for radius + position.
  - `plot_frame(...)` reads `case.init_dt`, `case.storm_name`, `case.model_label`
    for titles.
  - `main()` → `generate_animation(case)`.
  - Keep pure helpers (`discover_files`, `read_hafs_tp_records`,
    `pick_total_record`, `regrid_hafs`, `accumulate_hafs_step`,
    `hafs_event_total`, `haversine_km`, MRMS loaders, `qpf_cmap`, `QPF_LEVELS`,
    `QPF_COLORS`) — these are storm-agnostic and stay as free functions.
- **`parent_qpf.py`**
  - `default_parent_path()` takes `case` (uses `case.run_dir`, `case.parent_glob`).
  - `stage4_total(...)` and `plot_compare(...)` take `case` (replace
    `MASK_RADIUS_KM`, `INIT_DT`, `FIXED_DOMAIN`, titles, `OUT_DIR`).
  - `main()` → `generate_parent_figure(case)`; output →
    `case.out_dir / "parent_qpf_<case_slug>.png"`.
- **`ets_score.py`**
  - `build_mrms_total(...)`, `tc_swath_mask(...)` take `case`.
  - `contingency_scores`, `regrid_mrms_to_fixed` unchanged (pure math).
  - `main()` → `compute_ets_single(case)` (kept for the MRMS-only path).
- **`ets_full.py`**
  - `build_fixed_grid`, `hafs_parent_total`, `stage4_on_fixed`, `plot_curves`
    take `case`.
  - `main()` → `compute_ets(case)`; outputs →
    `case.out_dir / "ets_full_<case_slug>.{png,csv}"`.

### New: `storms/helene_hfsa.yaml`

The example case and the copy-me template:

```yaml
run_dir: /work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/HFSA
# Everything below is optional — auto-derived from the ATCF track + path:
storm_name: Hurricane Helene
init: 2024092400
domain: [15.0, 42.0, -100.0, -60.0]   # lat_min, lat_max, lon_min, lon_max
mask_radius_km: 500
out_dir: analysis/output/helene_hfsa
```

A second `storms/helene_hfsb.yaml` (same, `HFSB` path) demonstrates the A/B
switch with no other change.

## Data flow (unchanged per product, now case-parameterized)

1. `run.py` loads YAML → `StormCase` (parses atcfunix, fills defaults).
2. `parent`: discover highest-f parent file → cumulative APCP → swath-mask via
   `case.position_at` → MRMS + Stage IV totals → 3-panel PNG.
3. `ets`: nest event total + parent total + MRMS total + Stage IV on fixed grid
   → `tc_swath_mask(case)` → contingency scores per threshold → curves PNG + CSV.

## Error handling

- **Missing atcfunix:** `from_yaml` raises a clear error naming the run dir and
  the glob it tried; suggests adding an explicit `track`/`domain`/`init` to the
  YAML as a fallback (so a run without a track file can still be used manually).
- **Missing `run_dir` in YAML:** explicit validation error.
- **No GRIB files match the glob:** existing per-script "no files found"
  messages, now naming the case.
- **Stage IV / MRMS download failures:** unchanged — already degrade gracefully
  ("unavailable" panel / scored MRMS-only).
- **Auto-detected model is "HAFS" (neither A nor B in path):** allowed; user can
  set `model_label` in YAML to override.

## Testing

- Unit-test `parse_atcfunix` against a small fixture excerpt of Helene's track
  and assert it reproduces the current hardcoded `TC_TRACK_6H` values (within
  rounding). This is the correctness keystone — it proves the refactor preserves
  behavior.
- Unit-test `detect_model` ("…/HFSA" → "HAFS-A", "…/HFSB" → "HAFS-B", other →
  "HAFS") and `auto_domain` (bbox + pad).
- Unit-test `StormCase.from_yaml` with a minimal YAML (run_dir only) using a
  fixture run dir, asserting defaults + auto-derivation populate correctly.
- Keep `analysis/tests/test_ets_full.py` passing (adapt imports to the new
  signatures).
- Heavy GRIB/MRMS/Stage IV paths remain integration-only (run on Hercules);
  not unit-tested.

## Out of scope (YAGNI)

- NHC-by-storm-ID download and min-MSLP center-finding (atcfunix covers it).
- Multi-init / multi-cycle batch comparison.
- Any change to the science (accumulation logic, contingency math, color scale).
- A GUI or notebook front end.

## Migration / compatibility

The refactor changes function signatures and removes the per-script `main()`
config blocks. The single supported entry point becomes `run.py`. Output
filenames gain a `<case_slug>` suffix and move under `case.out_dir`. Helene/HFSA
results must match the current figures (same domain, swath, thresholds).
