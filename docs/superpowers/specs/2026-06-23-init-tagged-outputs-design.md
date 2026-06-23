# Init-tagged output filenames — design

**Date:** 2026-06-23
**Status:** Approved (design)

## Goal

A storm can have many model runs at different initialization times. Make every
product's output filenames automatically carry the run's init time, so outputs
from different inits never collide and each file says which init it came from —
without the user having to manage that by hand.

Applies to all products: the per-case parent QPF figure, the per-case ETS
figure/CSV, and the HFSA-vs-HFSB comparison figures/CSVs.

This is labeling only — no change to the science, the accumulation, or the
fair-comparison logic.

## Decisions (from brainstorming)

- **Scope:** all products (per-case parent + ETS, and the comparison).
- **Source:** the init is auto-derived from the case(s) (`StormCase.init_str`),
  not a manual field.
- **De-duplication:** if the user already named a YAML with the init, the tag is
  not added twice.
- **Comparison init:** derived from the two cases, which must share an init
  (enforced — error if they differ, since one init must be stamped).

## Architecture

### `StormCase.output_slug` — single source of truth (`analysis/hafs_case.py`)

Add a property to the `StormCase` dataclass:

```python
@property
def output_slug(self):
    """case_slug with the init appended for output filenames, de-duplicated:
    'helene_hfsa' -> 'helene_hfsa_2024092400'; if the slug already contains the
    init (the YAML was named with it), it is returned unchanged."""
    if self.init_str in self.case_slug:
        return self.case_slug
    return f"{self.case_slug}_{self.init_str}"
```

`case_slug` (YAML stem) and `init_str` (`%Y%m%d%H`) already exist on `StormCase`.

### Per-case outputs

Swap the output-filename slug from `case.case_slug` to `case.output_slug`:

- **`analysis/parent_qpf.py`** (`generate_parent_figure`):
  `case.out_dir / f"parent_qpf_{case.output_slug}.png"`.
  Also add the init to the figure `suptitle` (currently shows valid-end only):
  include `init {case.init_dt:%Y-%m-%d %HZ}` in the title text.
- **`analysis/ets_full.py`** (`compute_ets`):
  `f"ets_full_{case.output_slug}.csv"` and `.png`. (Its title already shows the
  init — no title change.)
- **`analysis/ets_score.py`** (`compute_ets_single`, the standalone MRMS-only
  path): `f"ets_{case.output_slug}.csv"` and `.png`, for consistency.

`out_dir` is unchanged (default `analysis/output/<case_slug>`); multiple inits
coexist in one directory, distinguished by the init in the filename.

### Comparison outputs (`analysis/compare.py`)

Two small helpers, used by both `generate_comparison` and `replot_from_csv` so
the names stay consistent:

```python
def _check_same_init(cases, case_paths):
    """Raise ValueError naming both if the two cases' init_dt differ."""
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

- `generate_comparison`: after loading the cases, call `_check_same_init`; derive
  `slug, title = _init_tag(cfg["label"], cases[0].init_dt)`. Output files use the
  init-tagged `slug` (`compare_categorical_<slug>.{png,csv}`,
  `compare_fss_<slug>.{png,csv}`). Plot calls pass `title` as the label.
  Inject `init = cases[0].init_str` into every `cat_row`/`fss_row`, and add a
  leading `init` column to both CSV column lists.
- `replot_from_csv`: load the two cases (cheap — `from_yaml` only parses the
  `.atcfunix`, no GRIB), call `_check_same_init`, derive the same `slug, title`,
  read the init-tagged CSVs, and redraw with `title`.

To keep `replot`'s file-I/O + plotting unit-testable without run dirs, factor it
into a pure inner helper:

```python
def _plot_comparison(out_dir, slug, title, fss_plot_thresholds):
    cat_rows = _read_rows(out_dir / f"compare_categorical_{slug}.csv", _CAT_NUM)
    fss_rows = _read_rows(out_dir / f"compare_fss_{slug}.csv", _FSS_NUM)
    plot_categorical_compare(cat_rows, title,
                             out_dir / f"compare_categorical_{slug}.png", "MRMS")
    plot_fss_compare(fss_rows, title,
                     out_dir / f"compare_fss_{slug}.png", "MRMS", "parent",
                     tuple(float(t) for t in fss_plot_thresholds))
```

`replot_from_csv` then loads cases → `_check_same_init` → `_init_tag` →
`_plot_comparison(out_dir, slug, title, cfg["fss_plot_thresholds"])`.

CSV column orders become (categorical):
`init, model, forecast, observation, threshold, a, b, c, d, ets, csi, bias, pod,
far, hss`; (FSS): `init, model, forecast, observation, threshold, scale_cells,
scale_km, fss`.

## Data flow (unchanged except naming)

Per-case and comparison computation is identical; only the output filename slug
(now `output_slug` / init-tagged comparison slug), the parent title, the
comparison titles, and the new CSV `init` column change.

## Error handling

- Comparison cases with differing `init_dt` → `ValueError` naming both paths and
  their inits (new, alongside the existing domain/grid/radius checks).
- Existing behavior otherwise unchanged.

## Testing

Local unit tests (no GRIB):
- `StormCase.output_slug`: appends init when absent
  (`helene_hfsa` + `2024092400` → `helene_hfsa_2024092400`); returns unchanged
  when the slug already contains the init
  (`helene_hfsa_2024092400` → unchanged). Built from a directly-constructed
  `StormCase`.
- `_init_tag(label, init_dt)`: slug ends with the init string; de-dups when the
  label slug already has it; title contains the formatted init.
- `_check_same_init`: two directly-constructed `StormCase`s with equal init pass;
  differing init raises `ValueError`.
- `_plot_comparison`: write toy CSVs at an init-tagged slug, assert the two
  init-tagged PNGs are produced (replaces/extends the existing replot test).
- Existing per-case and comparison tests stay green.

Per-case filename wiring (`parent_qpf`/`ets_full`/`ets_score`) and the full
`generate_comparison`/`replot` drivers are integration-verified on Hercules:
a run produces `*_2024092400.*` files, and a second init's run produces
`*_2024092412.*` without overwriting.

## Out of scope (YAGNI)

- Comparing different init times against each other (separate future feature).
- Putting the init in `out_dir` (filenames carry it; directory stays per-config).
- Renaming the existing example YAMLs (the auto-tag makes that unnecessary).

## Docs

README: note that all output filenames are auto-tagged with the run's init
(so you don't need the init in YAML names; it's de-duplicated if you add it),
and that a comparison's two cases must share an init.

## Compatibility

Additive/relabeling only. Old output files (without the init tag) are left in
place; new runs write init-tagged names. The per-case `parent|ets|all` path and
the `compare`/`replot` commands are otherwise unchanged.
