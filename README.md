# HAFS & WoFS rainfall verification

Case-driven tools for evaluating tropical-cyclone quantitative precipitation
forecasts. The current implementation covers HAFS-A and HAFS-B; WoFS and
multi-storm aggregation are planned extensions.

The framework supports three analysis levels:

1. One model initialization: QPF maps, categorical skill, and continuous
   errors.
2. Every eligible initialization of one model: a fixed-window comparison that
   isolates lead-time differences.
3. HAFS-A versus HAFS-B for one initialization: a head-to-head comparison over
   a shared NHC best-track footprint.

## Setup

On Orion or Hercules:

```bash
module load miniconda3
conda env create -f environment.yml
conda activate hafs
```

The main dependencies are NumPy, SciPy, PyYAML, boto3, cfgrib, ecCodes,
xarray, Matplotlib, Cartopy, Seaborn, and Pillow. `environment.yml` is preferred because it
installs the required native GRIB and mapping libraries.

MRMS observations are downloaded anonymously from NOAA's public S3 bucket.
Stage IV files are downloaded from `water.noaa.gov`. Downloads are cached in
`/tmp/mrms_cache` and `/tmp/stage4_cache` unless a YAML file overrides those
locations.

## Expected HAFS layout

A model root should contain one `YYYYMMDDHH` directory per initialization:

```text
/work2/.../helene/HFSA/
  2024092400/
    09l.2024092400.hfsa.parent.trak.atcfunix
    09l.2024092400.hfsa.parent.atm.f000.grb2
    09l.2024092400.hfsa.parent.atm.f003.grb2
    ...
    09l.2024092400.hfsa.storm.atm.f000.grb2
    09l.2024092400.hfsa.storm.atm.f003.grb2
    ...
  2024092412/
  ...
```

Large model data and generated products are intentionally excluded from Git.

## Run one initialization

A per-initialization YAML points to the model root or initialization directory:

```yaml
run_dir: /work2/.../helene/HFSA
storm_name: Hurricane Helene
init: 2024092400
domain: [15.0, 42.0, -100.0, -60.0]
mask_radius_km: 500
out_dir: analysis/output/helene_hfsa
```

Run all products or select one:

```bash
python analysis/run.py storms/helene_hfsa.yaml all
python analysis/run.py storms/helene_hfsa.yaml parent
python analysis/run.py storms/helene_hfsa.yaml ets
python analysis/run.py storms/helene_hfsa.yaml rmse
```

`all` builds the expensive verification fields once and shares them between
the ETS and continuous-error products.

Outputs are initialization-tagged and written below the configured `out_dir`:

```text
parent_qpf_<case>_<init>.png    nest + parent + MRMS + Stage IV QPF
ets_full_<case>_<init>.png     ETS versus rainfall threshold
ets_full_<case>_<init>.csv     contingency counts and categorical scores
rmse_scatter_<case>_<init>.png forecast-versus-observed scatter panels
rmse_<case>_<init>.csv         RMSE, MAE, bias, and correlation
```

## Compare every initialization of one model

This is the scalable workflow for roughly ten pre-landfall runs. A single YAML
describes a storm, model, and absolute verification window; initialization
directories are discovered automatically.

```yaml
run_root: /work2/.../helene/HFSA
valid_start: 2024092600
valid_end: 2024092800
landfall_time: 202409270310
storm_name: Hurricane Helene
domain: [15.0, 42.0, -100.0, -60.0]
mask_radius_km: 500
out_dir: analysis/output/helene_hfsa_cycles
```

Run either model with its corresponding config:

```bash
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles
python analysis/run.py storms/helene_hfsb_cycles.yaml cycles
```

Cycles initialized before `valid_start` accumulate from `valid_start`; later
cycles accumulate from their initialization. Every cycle must extend through
the common `valid_end` and is verified against MRMS/Stage IV accumulated over
its matching interval. The output CSV records each cycle's effective start and
end. Use an optional `inits:` list only when automatic discovery should be
restricted. `landfall_time` enables a common "hours before landfall" axis; it
accepts `YYYYMMDDHH` or `YYYYMMDDHHMM` UTC.

Scores use one shared spatial swath: the union of all surviving forecast-track
positions within their effective windows. Because later initializations use
shorter accumulation periods, interpret cycle-to-cycle changes together with
the recorded valid window.

Cycle outputs are:

```text
cycles_<case>_<start>_<end>.csv
cycles_fss_<case>_<start>_<end>.csv
cycles_metrics_<case>_<start>_<end>.png
cycles_ets_heatmap_<case>_<start>_<end>.png
cycles_ets_bars_<case>_<start>_<end>.png
cycles_fss_heatmap_<case>_<start>_<end>.png
cycles_qpf_<case>_<start>_<end>.gif
cycles_difference_<case>_<start>_<end>.gif
cycles_observed_<case>_<start>_<end>.gif
```

The multi-cycle products use only the fixed parent domain. ETS is shown as a
Seaborn heatmap of rainfall threshold by initialization; FSS uses one Seaborn
heatmap per rainfall threshold, with neighborhood scale by initialization.
The model-level ETS bar chart follows the paper-style 2–24 inch threshold axis
and pools contingency counts across cycles before calculating ETS. The suite
also includes separate parent-forecast, parent-minus-MRMS, and observed-MRMS
animations. The
observed animation remains visually static while its accumulation window is
unchanged. Set `make_animation: false` to skip all three GIFs.
Optional `ets_bar_thresholds_in`,
`fss_thresholds_in` and `fss_scales_cells` lists control the bar-chart and FSS
thresholds/scales.

### Faster cycle runs and reruns

Use multiple CPU processes for parent GRIB extraction and interpolation:

```bash
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles --workers 4
```

The default is one worker. You can persist the setting with `workers: 4` in
the cycles YAML; the CLI overrides it. Workers are capped by the number of
cycles, CPU affinity where available, and `SLURM_CPUS_PER_TASK` in a Slurm
allocation. A Slurm job without an explicit CPU-per-task allocation uses one
worker. This is a single-node process pool; requesting more nodes does not
speed it up. Child processes use one native numerical-library thread each.

On Orion/Hercules, request the CPUs for **one task** in your batch script
(also supply your usual account, partition, time, and memory settings):

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module load miniconda3
conda activate hafs
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles --workers 4
```

See the [Slurm CPU allocation documentation](https://slurm.schedmd.com/sbatch.html#OPT_cpus-per-task).

The repository includes a launcher that allocates one task with four CPUs,
binds the task to its allocated cores, and matches `--workers` to the allocation.
From the repository root, activate `hafs`, then submit (replace `YOUR_ACCOUNT`
and add your usual partition/QoS options if required):

```bash
conda activate hafs
sbatch --account=YOUR_ACCOUNT analysis/cycles.sbatch storms/helene_hfsa_cycles.yaml
```

The defaults are 32 GB memory and two hours; these are starting allocations,
not measured requirements. Override them, or the CPU count, before the script
name, for example `sbatch --account=YOUR_ACCOUNT --cpus-per-task=8 --mem=64G
--time=04:00:00 analysis/cycles.sbatch storms/helene_hfsa_cycles.yaml`.
Analysis options such as `--no-animation` go after the YAML path. This launcher
sets `--workers` from the allocation, overriding any YAML or argument value.
It inherits the active conda environment; it does not load a different Python.

`sbatch` prints the real Slurm job ID. Use that ID with `scontrol show job JOBID`
and follow `cycles-JOBID.log`. A command launched with `python ... &` is only a
background shell process: the number from `jobs -l` is a PID, not a Slurm job
ID, and running it on `hercules-login-*` does not allocate a compute node.

The log now identifies the host, PID, Slurm job, CPU affinity, requested and
effective worker counts, and the start of each phase. Parent tasks print their
PIDs. The pool exits before observations/scoring/plotting, so seeing no child
processes during those phases is expected. Redirected CLI logs are line-buffered;
`tail -f` shows progress without waiting for the output buffer to fill.

Start with 2–4 workers on an allocated compute node and check peak memory:
each worker decodes native forecast grids and builds interpolation structures.
Large HAFS-M grids may need fewer workers or more allocated memory. Plotting,
scoring, and observation processing remain sequential, so total speedup will
depend on which phase dominates. No new Python dependencies are required.

Several optimizations apply even with one worker:

- MRMS totals are shared across cycles with identical windows. Overlapping
  windows with different starts decode each hourly GRIB only once, accumulating
  on the native grid before interpolation. For ten cycles sharing a 48-hour
  window, this reduces hourly decode calls from 480 to 48. Different starts
  are accumulated backwards using float64; tiny floating-point differences
  from independent forward sums are possible.
- Stage IV reuses the unmasked native total when the touched dates match,
  while keeping each cycle's distinct track mask.
- Derived parent-window grids are saved under `out_dir/.field_cache` and
  reused on subsequent `cycles` runs. The key includes source paths, file
  sizes and modification times, forecast window, and grid coordinates.
  `--refresh-cache` rebuilds those grids; `cache_fields: false` in YAML disables
  this cache. Refresh after replacing source data while preserving both size
  and timestamps. Older cache entries remain on disk. This cache covers parent
  fields only; a full rerun still builds observations and recomputes scores.

Keep `mrms_cache_dir` and `stage4_cache_dir` on persistent scratch storage if
jobs can move between nodes; `/tmp` caches are node-local and may be cleared.
That avoids downloading the same observation files on subsequent jobs.

To shorten a run further when you only need verification tables and PNGs:

```bash
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles --workers 4 \
  --no-animation --no-ml-features
```

These switches skip GIF rendering and optional environmental ML feature
extraction. They do not change verification scores. Their YAML equivalents
are `make_animation: false` and `ml_features: false`. Runs print elapsed times
for parent extraction, MRMS, Stage IV, scoring/tables, plots/animations, and
the entire command so you can identify the next bottleneck.

For plot styling changes, redraw directly from existing cycle CSVs:

```bash
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles --replot
# Equivalent; automatically recognizes a cycles YAML:
python analysis/run.py storms/helene_hfsa_cycles.yaml replot
```

This regenerates metric, ETS, FSS, percentile, pattern-correlation, and saved
track/shifted-skill figures when their tables are available. It works without
the model or best-track archive mounted and leaves CSVs/ML features untouched.
Distribution curves and GIFs require gridded fields, so use a normal `cycles`
rerun for those. New verification thresholds, windows, masks, or grid settings
also require `cycles` computation; replot can only display already saved scores.

To compare the cycle tables from HAFS-A, HAFS-B, and HAFS-M, run:

```bash
python3 analysis/run.py storms/helene_cycles_compare.yaml cycles-compare
```

This creates grouped ETS bars and a scale-dependent FSS comparison in
`analysis/output/helene_cycles_compare`. A configured model without cycle
CSVs—currently HAFS-M—is retained in the legend as “awaiting data.” After its
files arrive, update `storms/helene_hfsm_cycles.yaml`, run that model's
`cycles` command, and rerun `cycles-compare`.

## Compare HAFS-A and HAFS-B

The two case YAMLs must describe the same storm and initialization. Download
the storm's NHC ATCF b-deck and reference it from a comparison YAML:

```yaml
label: Hurricane Helene
cases:
  - storms/helene_hfsa.yaml
  - storms/helene_hfsb.yaml
best_track: /work2/.../bal092024.dat
out_dir: analysis/output/helene_compare
```

```bash
python analysis/run.py storms/helene_compare.yaml compare
```

Both configurations are evaluated on the same best-track swath and common
finite-data coverage. Products include categorical curves, FSS by
neighborhood scale, a performance diagram, storm-total maps and exceedance
areas, and RMW-normalized storm-relative composites. CSVs contain the complete
categorical and FSS matrices.

Existing comparison CSVs can be replotted without reopening GRIB files:

```bash
python analysis/run.py storms/helene_compare.yaml replot
```

## Analysis viewer

Browse generated cases locally:

```bash
python analysis/viewer.py
```

Generate missing products first:

```bash
python analysis/viewer.py --generate missing
```

On a remote cluster, forward the printed port through SSH. If port forwarding
is unavailable, create a self-contained gallery:

```bash
python analysis/viewer.py --export
```

## Verification details

- The fixed HAFS parent grid uses its cumulative `0 -> forecast hour` APCP
  record.
- The moving nest cannot use its storm-relative cumulative APCP as a
  geographic storm total. The code regrids and sums short, geographically
  valid incremental buckets instead.
- MRMS uses hourly gauge-corrected multisensor QPE accumulated over the exact
  requested window.
- Stage IV is CONUS-only and consists of 12Z-to-12Z daily products. Summing
  touched days approximates windows that do not align to those boundaries;
  figures and CSV workflows retain that caveat.
- Categorical outputs include ETS, CSI, frequency bias, POD, FAR, and HSS.
- Continuous outputs include RMSE, MAE, mean bias, and Pearson correlation.
- FSS evaluates spatial displacement tolerance across neighborhood sizes.

## Tests

```bash
python -m pytest analysis/tests -q
```

Tests use synthetic fields and small track fixtures; they do not require the
HPC model archive or observation downloads.

## Repository layout

```text
analysis/
  run.py             command dispatcher
  hafs_case.py       YAML loading, track parsing, and case models
  hafs_common.py     GRIB loading, nest accumulation, and MRMS access
  parent_qpf.py      QPF maps and Stage IV access
  ets_full.py        per-run categorical verification
  rmse_scatter.py    per-run continuous verification
  cycles.py          fixed-window, multi-initialization analysis
  compare.py         HAFS-A versus HAFS-B analysis
  skill_metrics.py   shared continuous and neighborhood metrics
  best_track.py      NHC b-deck parsing
  viewer.py          local/offline results gallery
  tests/             unit and plotting tests
storms/              active case and cycle configurations
```
