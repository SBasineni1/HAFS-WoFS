# Nest QPF panel — design

**Date:** 2026-06-25
**Status:** Approved (design)

## Goal

Add a **Nest** precipitation panel to the existing QPF comparison figure so each
case produces a single side-by-side graphic of **four** accumulated-precip
fields over the TC rainfall swath:

```
[ Nest ]   [ Parent ]   [ MRMS ]   [ Stage IV ]
```

Today `parent_qpf.py` draws only three panels (Parent · MRMS · Stage IV). The
moving 2-km nest total is computed elsewhere (`ets_full.py`) but only ever shown
as ETS-vs-threshold curves — never as a precip map. This change surfaces the
nest field visually, next to the parent and the two observation benchmarks, so
the parent-vs-nest difference is easy to see at a glance.

This is a framework feature: the panel is fully case-driven and produced for any
storm/init via the existing run flow — no per-storm code edits.

## Scope

**In scope:** `analysis/parent_qpf.py` only.

**Not touched:** `ets_full.py`, `compare.py` (HFSA-vs-HFSB FSS/ETS comparison),
the YAML/`StormCase` layer, `run.py`. The output filename is unchanged.

## Key decisions (from brainstorming)

- **Layout:** one 4-panel figure per storm/init (not a multi-storm grid).
- **Nest field source:** the moving-nest running-max from
  `hafs_event_total(...)` on the case fixed grid — the *same field*
  `ets_full.py` scores. There is no separate "clean" nest product; the
  `storm.atm` files are the nest.
- **Honest about artifacts:** the running-max sweeps in non-TC / frontal rain
  and inflates totals (this is exactly why the viewer-consistent product uses
  the parent domain — see the framework design spec). The nest panel is drawn
  as-is and its title carries the caveat, so the map and the ETS curves agree.
- **Masking:** the nest panel is masked to the same display swath
  (`display_radius_km` circles along the track) as the parent panel, on the
  fixed grid.
- **Shared color scale:** reuse the existing `QPF_LEVELS` / `qpf_cmap()` so all
  four panels share one colorbar and are directly comparable.
- **Filename unchanged:** stays `parent_qpf_<output_slug>.png` to avoid breaking
  README/docs references. (Rename to drop "parent" is a possible follow-up, not
  part of this change.)

## Approach

Extend `parent_qpf.py` in place. Rejected alternatives:

- *New standalone script* — would duplicate the MRMS download + Stage IV
  tarball plumbing already in `parent_qpf.py`.
- *Shared nest cache between `parent_qpf` and `ets_full`* — over-engineering for
  two commands that are run separately; YAGNI.

The plotting layer needs no change: `plot_compare(case, panels, …)` already
loops over an N-length `panels` list, sizes the figure to `8 × n` inches, draws
each field with the shared `QPF_LEVELS` BoundaryNorm, and renders one shared
colorbar. Adding a panel = prepending one tuple to the `panels` list.

## Changes to `analysis/parent_qpf.py`

1. **Imports.** Add `discover_files`, `hafs_event_total`, and `build_fixed_grid`
   from `hafs_common` (mirroring `ets_full.py`). Swath/track helpers reused as
   already imported.

2. **Compute the nest total** in `generate_parent_figure`, after the parent
   record is read:
   - `grid_lat, grid_lon = build_fixed_grid(case)` (the proven path
     `ets_full.compute_ets` uses; equivalent to `case.fixed_grid()`).
   - `file_pairs = discover_files(case.run_dir, case.storm_glob(), case.fhours_filter)`
     (same discovery `ets_full.compute_ets` uses).
   - If no `storm.atm` files are found, skip the nest panel gracefully (pass
     `None` data so `plot_compare` renders "unavailable" — matching how MRMS /
     Stage IV gaps are handled) and print a notice. The figure still produces
     the other three panels.
   - `nest_total, _mode = hafs_event_total(file_pairs, grid_lat, grid_lon)`.
   - Build the display swath mask on the fixed grid (circles of
     `display_radius_km` along the best track, hours `0..end_fhour` — same loop
     already used for the parent swath) and zero the nest outside it.

3. **Prepend the nest panel** to the `panels` list:
   ```
   (grid_lon, grid_lat, nest_display,
    f"{case.model_label} Nest APCP (moving 2-km, running-max)\n"
    f"0–{end_fhour}h — can inflate vs swept frontal rain")
   ```
   so panel order is Nest · Parent · MRMS · Stage IV.

4. **Update the suptitle** text from
   `"… parent QPF vs MRMS vs Stage IV …"` to
   `"… QPF: nest vs parent vs MRMS vs Stage IV …"`.

5. **Print the nest max** in the closing summary alongside the existing
   parent / MRMS / Stage IV maxes.

## Data flow

```
build_fixed_grid ──┐
storm.atm files  ──┴─ hafs_event_total ─→ nest_total ─→ swath-mask ─→ nest panel
parent.atm        ── read_hafs_tp_records ─→ parent panel        ┐
MRMS S3 1H        ── load/accumulate ─────→ MRMS panel           ├─ plot_compare ─→ parent_qpf_<slug>.png
Stage IV tarballs ── stage4_total ────────→ Stage IV panel       ┘
```

All four panels share `QPF_LEVELS` and one colorbar.

## Framework / interchangeability

No new wiring. `generate_parent_figure(case)` is already invoked by the `parent`
and `all` commands in `run.py`, so the 4-panel figure is produced automatically
alongside the ETS figure/CSV (the `ets`/`all` path) and remains separate from
the HFSA-vs-HFSB FSS/ETS comparison (`compare`). Every input is derived from
`case` (globs, fixed grid, track, radii), so dropping in a new storm YAML
produces the nest panel with no code changes.

## Tradeoffs / risks

- **Runtime.** `parent_qpf` currently reads one `parent.atm` file; it will now
  also read every `storm.atm` forecast hour — the same heavy I/O the `ets`
  command already does. The `parent`/`all` run will be noticeably slower.
  Accepted: correctness/visualization over speed; the per-file progress logging
  in `hafs_event_total` keeps it observable.
- **Visible inflation.** The nest panel shows the running-max artifact by
  design; the panel title states it so readers aren't misled.

## Testing

- Unit/light: with a stubbed/empty `storm.atm` discovery, `generate_parent_figure`
  still renders the other three panels and marks the nest "unavailable" (no
  crash). Follow existing test patterns under `analysis/tests`.
- Integration (manual, Hercules): run
  `python analysis/run.py storms/helene_hfsa.yaml parent` and confirm the output
  PNG has four panels in order Nest · Parent · MRMS · Stage IV, a single shared
  colorbar, the updated suptitle, and a printed nest max consistent with the
  `ets` run's reported nest max.
