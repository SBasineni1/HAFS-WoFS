# HAFS & WoFS

A Comparative Evaluation of Rainfall Forecast Skill for Landfalling Tropical
Cyclones from HAFS and WoFS.

This repo holds a **case-driven framework** that, for any storm and any HAFS
configuration (HFSA or HFSB), produces:

1. **A parent-domain QPF comparison** — HAFS parent total rainfall vs MRMS QPE
   vs NCEP Stage IV QPE, as a 3-panel map (`parent_qpf_<case>.png`).
2. **An ETS skill plot + table** — Equitable Threat Score vs rainfall threshold
   for the HAFS parent *and* 2-km nest, each verified against MRMS and Stage IV
   over the TC rainfall swath (`ets_full_<case>.png` + `.csv`).
3. **A cycle-comparison product** — for one storm and model configuration,
   scores every eligible initialization over the same rainfall window and
   shared verification footprint (`cycles` mode).
4. **Paper-style TC QPF verification** — 6-h lead-time statistics, best-track
   shifting, RMW-normalized composites and radial distributions, rainfall
   objects, and homogeneous multi-storm aggregation (`paper` mode).

Running a new storm only requires a small YAML file pointing at the run
directory — the storm track, init time, and HAFS-A/B label are read
automatically from the run's `.atcfunix` track and path.

---

## Quick start

One-time setup, then one command per product — details in the numbered
sections below.

```bash
# Setup (once, on Orion/Hercules — §1)
module load miniconda3
conda env create -f environment.yml
conda activate hafs

# Per-run products for one initialization (§3): qualitative parent-domain
# rainfall map + ETS curves + RMSE scatter, all in one go
python analysis/run.py storms/helene_hfsa.yaml all

# ...or individually:
python analysis/run.py storms/helene_hfsa.yaml parent   # qualitative 3-panel rainfall map
python analysis/run.py storms/helene_hfsa.yaml ets      # ETS vs threshold + CSV
python analysis/run.py storms/helene_hfsa.yaml rmse     # RMSE/MAE/bias/r scatter + CSV

# Another initialization of the same storm: copy the YAML, change `init:` (§4)
python analysis/run.py storms/helene_hfsa_2024092412.yaml all

# Cross-initialization comparison (§5): every eligible cycle scored over one
# common valid window — metrics-vs-init figure + QPF map small-multiples
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles

# Newman et al. (2024) figure families for one storm / multiple models
python analysis/run.py storms/helene_paper.yaml paper

# Run each listed storm and pool the homogeneous multi-storm sample
python analysis/run.py storms/paper_multistorm.yaml paper

# Browse every storm/init and its QPF, ETS, and RMSE figures in one place
python analysis/viewer.py

# Generate any missing products first, then open the same live viewer
python analysis/viewer.py --generate missing

# Pull the figures to your laptop (run this FROM the laptop — §3)
scp -r <user>@<cluster>:/path/to/HAFS-WoFS/analysis/output/<case> ~/Downloads/
```

---

## 1. Setup (conda on Orion / Hercules)

The framework needs a Python 3 environment with the scientific GRIB stack. The
fastest path is the conda env file in this repo.

```bash
# 1. Make conda available (adjust to your cluster — on Orion/Hercules this is
#    usually a module load, or sourcing your own miniconda install):
module load miniconda3          # or: source ~/miniconda3/etc/profile.d/conda.sh

# 2. Create the environment from the repo (run from the repo root):
conda env create -f environment.yml      # creates an env named "hafs"

# 3. Activate it:
conda activate hafs
```

**Already have a working `hafs` env?** You likely only need PyYAML, which isn't
always installed:

```bash
conda activate hafs
pip install pyyaml          # or: conda install -c conda-forge pyyaml
```

The full dependency list (also in `environment.yml` / `requirements.txt`):
`numpy`, `scipy`, `pyyaml`, `boto3`, `cfgrib`, `eccodes`, `xarray`,
`matplotlib`, `cartopy`.

> **Network note:** the verification step downloads MRMS QPE (AWS S3, anonymous)
> and Stage IV QPE (water.noaa.gov). The login nodes on Orion/Hercules have
> outbound internet, so running there works; first runs are slower while these
> cache to `/tmp` (override with `mrms_cache_dir` / `stage4_cache_dir` in the
> YAML).

---

## 2. What the framework expects

Point a case at the **run directory** for one storm + one HAFS config. The
framework globs recursively under it for:

- `*.atcfunix`           — the forecast track (storm name, init, and track fixes)
- `*parent.atm.f*.grb2`  — parent-domain output (for the parent QPF + parent ETS)
- `*storm.atm.f*.grb2`   — 2-km nest output (for the nest ETS curve)

Example (Helene HFSA on Hercules):

```
/work2/.../helene/HFSA/2024092400/
    09l.2024092400.hfsa.parent.trak.atcfunix
    09l.2024092400.hfsa.parent.atm.f000.grb2 … f126.grb2
    09l.2024092400.hfsa.storm.atm.f000.grb2  … f126.grb2
```

HAFS-A vs HAFS-B is auto-detected from `HFSA` / `HFSB` in the path.

---

## 3. Running

From the repo root, with the env activated:

```bash
python analysis/run.py storms/<case>.yaml all      # parent map + ETS + RMSE (default)
python analysis/run.py storms/<case>.yaml parent   # qualitative 3-panel rainfall map only
python analysis/run.py storms/<case>.yaml ets      # ETS plot + CSV only
python analysis/run.py storms/<case>.yaml rmse     # RMSE scatter + CSV only
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles  # cross-init comparison
python analysis/run.py storms/helene_paper.yaml paper          # paper-style verification
```

The two shipped examples:

```bash
python analysis/run.py storms/helene_hfsa.yaml all
python analysis/run.py storms/helene_hfsb.yaml all
```

### One-file analysis viewer

Run `python analysis/viewer.py` from the repository root and open the printed
address (normally `http://127.0.0.1:8765`). The storm, model, and initialization
selectors put the side-by-side QPF map, ETS curve, RMSE scatter, and their CSVs
in one gallery. The page rescans `analysis/output/` every five seconds, so new
graphics appear automatically while it remains open.

When the viewer runs on Hercules, `127.0.0.1` refers to the remote login node.
Keep it running and create an SSH tunnel from a second terminal on your laptop:

```bash
ssh -N -L 8765:127.0.0.1:8765 <your-normal-Hercules-SSH-host>
```

Then open `http://127.0.0.1:8765` on the laptop. The viewer detects an SSH
session and prints these instructions automatically. Pass `--ssh-host hercules`
if you want it to print your exact SSH alias.

If Hercules rejects the tunnel with `administratively prohibited`, export a
single offline gallery instead:

```bash
python analysis/viewer.py --export
```

Download `analysis/output/hafs-viewer.html` through the
[Hercules Open OnDemand portal](https://hercules-ood.hpc.msstate.edu), then open
that file on the laptop. Its plots and CSV links are embedded, so it needs no
server or network connection. Re-export after generating new graphics.

To make plots and view them with one command:

```bash
python analysis/viewer.py --generate missing              # all storm YAMLs
python analysis/viewer.py --generate always --case helene_hfsa
python analysis/viewer.py --generate missing --no-serve   # batch/HPC only
```

`missing` leaves complete cases alone; `always` recomputes the selected cases.
The viewer recognizes normal per-init, cycles, and HFSA-vs-HFSB comparison
YAMLs and sends each to the existing `analysis/run.py` workflow. A failed case
does not hide plots that were already generated for the other cases.

Each run prints a one-line case summary (storm, model, init, domain, track
file) so you can sanity-check the auto-detection before it grinds through the
GRIB files.

### Outputs

Land in `analysis/output/<case>/` (or wherever `out_dir` points):

```
parent_qpf_<case>.png     # HAFS parent vs MRMS vs Stage IV, 3-panel
ets_full_<case>.png       # ETS vs threshold: parent/nest × MRMS/Stage IV
ets_full_<case>.csv       # the same scores as a table (a/b/c/d, ETS, bias, POD, FAR, CSI)
rmse_scatter_<case>.png   # forecast-vs-observed hexbin panels: parent/nest × MRMS/Stage IV
rmse_<case>.csv           # storm-total continuous scores (n, RMSE, MAE, bias, r)
cycles_<slug>.csv         # per-init continuous + categorical scores for a common window
cycles_metrics_<slug>.png # RMSE, bias, and ETS vs initialization time
cycles_maps_<slug>.png    # nest window-QPF panels per init + an observed-MRMS panel
```

For `cycles` outputs, `<slug>` is `<cycles-yaml-stem>_<valid-start>_<valid-end>`
(for example, `helene_hfsa_cycles_2024092600_2024092800`). The cycles CSV is
long format: one row per initialization, forecast, observation, and rainfall
threshold. `bias_mm` is continuous mean forecast-minus-observation error in
millimetres; `bias` is categorical frequency bias.

> **Init tagging:** every output filename is automatically stamped with the run's
> initialization time (e.g. `parent_qpf_helene_hfsa_2024092400.png`,
> `ets_full_helene_hfsa_2024092400.csv`), so a storm's many init times never
> overwrite each other. You don't need the init in the YAML name — it's added
> automatically (and de-duplicated if you do include it).

### Pull results to your laptop

```bash
scp -r <user>@<cluster>:/path/to/HAFS-WoFS/analysis/output/<case> ~/Downloads/
open ~/Downloads/<case>/*.png      # macOS
```

---

## 4. Adding a new storm

Copy a YAML and change `run_dir` — that's the minimum:

```bash
cp storms/helene_hfsa.yaml storms/<storm>_<model>.yaml
```

```yaml
# storms/<storm>_<model>.yaml
run_dir: /work2/.../<storm>/HFSA          # REQUIRED — the only thing you must set

# Everything below is OPTIONAL — auto-derived from the .atcfunix + path if omitted:
storm_name: Hurricane <Name>              # else from the atcfunix
init: 2024092400                          # YYYYMMDDHH; else from the atcfunix
domain: [15.0, 42.0, -100.0, -60.0]       # lat_min, lat_max, lon_min, lon_max; else from track bbox
mask_radius_km: 500                       # TC verification swath radius
out_dir: analysis/output/<storm>_<model>
```

Then `python analysis/run.py storms/<storm>_<model>.yaml all`.

For **another initialization of the same storm**, copy the YAML and change
only `init:` (see `storms/helene_hfsa_2024092412.yaml`) — `run_dir` stays the
storm/model root, and the init-tagged output filenames keep runs from
overwriting each other even in a shared `out_dir`.

---

## 5. Comparing initialization cycles

Use `cycles` to compare a single model configuration's forecasts as their
initializations approach landfall. It takes a storm-level cycles YAML rather
than a normal per-init case YAML:

```bash
python analysis/run.py storms/helene_hfsa_cycles.yaml cycles
```

```yaml
# storms/<storm>_<model>_cycles.yaml
run_root: /work2/.../<storm>/HFSA  # YYYYMMDDHH subdirectories, one per cycle
valid_start: 2024092600            # REQUIRED: common UTC window start
valid_end: 2024092800              # REQUIRED: common UTC window end
domain: [15.0, 42.0, -100.0, -60.0] # REQUIRED: lat_min, lat_max, lon_min, lon_max

# Optional:
storm_name: Hurricane <Name>
mask_radius_km: 500
out_dir: analysis/output/<storm>_<model>_cycles
inits: [2024092400, 2024092412]    # otherwise discover YYYYMMDDHH directories
ets_threshold_mm: 25               # headline threshold in the metrics figure
```

`run_root` subdirectories named `YYYYMMDDHH` are discovered and sorted unless
you set `inits:` explicitly. A cycle is included only when
`init ≤ valid_start` and `init + maximum forecast hour ≥ valid_end`.
Skipped cycles are printed with their reason; a cycle with missing GRIB data
inside the window is also skipped so the remaining eligible cycles can run.

---

## 6. Comparing HFSA vs HFSB (head-to-head)

To score two configs of the same storm against each other over one **shared,
fair verification swath** (the NHC best track), use a comparison config.

1. Fetch the storm's NHC best track (ATCF b-deck) once:

       wget https://ftp.nhc.noaa.gov/atcf/archive/<year>/b<basin><cy><year>.dat.gz
       gunzip b<basin><cy><year>.dat.gz
       # Helene 2024 -> bal092024.dat

2. Edit `storms/helene_compare.yaml` (or copy it) to point `cases:` at the two
   case YAMLs and `best_track:` at the file you just downloaded.

3. Run:

       python analysis/run.py storms/helene_compare.yaml compare

Outputs in `out_dir`:

- `compare_categorical_<label>.png` — ETS / CSI / frequency bias vs threshold,
  HFSA vs HFSB (parent solid, nest dashed), vs MRMS
- `compare_fss_<label>.png` — Fractions Skill Score vs neighborhood scale
- `compare_categorical_<label>.csv`, `compare_fss_<label>.csv` — the full matrix
  (both forecasts × MRMS & Stage IV, all thresholds/scales)

The two cases must share an initialization time (the comparison errors if they
don't). The comparison outputs are init-tagged too
(`compare_categorical_hurricane_helene_2024092400.png`), and each CSV has a
leading `init` column.

Both models are scored over the identical best-track swath, so their point
counts (`n`) match and the comparison is apples-to-apples. For the nest
comparison, both models are scored over the common coverage of both nests
(points where only one nest has valid data are excluded), so nest `n` also
matches. This step runs ~2× a single `ets` run (it builds both models' nest
totals).

---

## 7. How to read the ETS plot

Each `ets_full_<case>.png` has four curves for one model run:

- **parent vs MRMS / parent vs Stage IV** — the 6-km parent domain
- **nest vs MRMS / nest vs Stage IV** — the 2-km moving nest

Within a single run, parent and nest are scored over the **same** swath, so
comparing them ("does the high-res nest verify better than the parent?") is a
fair, apples-to-apples read.

> **Comparing HFSA to HFSB directly:** each model is scored over *its own*
> forecast-track swath, so the point counts (`n`) differ between A and B. That
> makes a direct A-vs-B ETS comparison a confound. If you need a clean
> head-to-head, both should be verified over a common swath (e.g. the NHC best
> track) — not yet implemented; ask if you want it.

---

## 8. How to read the RMSE scatter

Each `rmse_scatter_<case>.png` panel plots every swath grid point's
storm-total rainfall: observed (x) vs forecast (y), with a dotted 1:1
line. Points above the line are over-forecasts. The annotation box gives:

- **RMSE** — root-mean-square error of the totals (mm); penalizes big misses.
- **MAE** — mean absolute error (mm).
- **bias** — mean(forecast − observed): **positive = over-forecast**.
- **r** — Pearson correlation of the point totals.
- **n** — valid swath points scored (same footprint as the ETS).

RMSE/MAE/bias are continuous companions to the categorical ETS curves:
ETS asks "did we put rain ≥ X in the right places", the scatter asks
"how far off were the amounts".

---

## 9. How to read the cycle comparison

Every included cycle is accumulated over the same absolute
`valid_start`–`valid_end` window and verified against the same observed total.
Therefore, changes in RMSE, bias, or ETS across initialization times describe
lead-time differences rather than changing accumulation lengths.

All cycles are scored on one shared swath: the union of each included cycle's
forecast-track positions within the valid window, expanded by
`mask_radius_km`. A run with an inaccurate track is thus penalized for placing
rain in the wrong location; that is intentionally part of the comparison.

The small-multiple map shows each cycle's **nest** window-total QPF plus the
MRMS observed total; it does not include parent-domain map panels. Metrics and
CSV scores include both parent and nest forecasts. Stage IV is scored when it
is available, but remains CONUS-only and is based on 12Z–12Z daily products
summed over the days touched by the requested window, so it approximates an
arbitrary UTC window. If Stage IV is unavailable, the product continues with
MRMS-only scoring and marks that caveat in its output.

---

## 10. Newman et al. paper-style verification

The `paper` command implements the reusable analysis figure families from
Newman et al. (2024), *Multi-season evaluation of Hurricane Analysis and
Forecast System (HAFS) quantitative precipitation forecasts*:

```bash
python analysis/run.py storms/helene_paper.yaml paper
```

A paper-storm YAML describes one storm and one or more model run roots. The
initialization sample is the intersection available across every model, so no
model receives extra cases. Each forecast is evaluated on the same 0.1-degree
grid and best-track-centered 600-km mask. The defaults reproduce the paper's
6-h accumulation period and 0.1, 0.5, 1.0, 1.5, 2.5, 3.5, and 5.0-inch
thresholds (stored in millimetres).

```yaml
storm_name: Hurricane Helene
models:
  HAFS-A: /work2/.../helene/HFSA
  HAFS-B: /work2/.../helene/HFSB
best_track: /work2/.../bal092024.dat
domain: [15.0, 42.0, -100.0, -60.0]

# Optional: otherwise use the intersection of cycle directories.
inits: [2024092400, 2024092412]
lead_hours: [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72,
             78, 84, 90, 96, 102, 108, 114, 120, 126]
forecast_domain: parent       # paper default; "nest" is also supported
composite_lead_hour: 12
object_lead_hour: 12
out_dir: analysis/output/helene_paper
```

The command writes:

```text
paper_track_shift_<case>.png           # Fig. 2 family: raw/shifted ETS + frequency bias
paper_ets_lead_<case>.png               # Figs. 3/9 family: threshold ETS by lead
paper_frequency_bias_lead_<case>.png    # Figs. 4/10 family: frequency bias by lead
paper_storm_relative_<case>.png         # Figs. 5/11/13 family: mean RMW composite
paper_radial_<case>.png                 # Figs. 6/12/14 family: 0.4-RMW boxplots
paper_object_identification_<case>.png  # Fig. 7 family: forecast/obs rainfall objects
paper_object_frequency_<case>.png       # Fig. 8 family: log intensity frequencies
paper_samples_<case>.csv                # every init/lead/model/threshold contingency table
paper_categorical_<case>.csv            # pooled scores and bootstrap 95% intervals
paper_radial_<case>.csv                 # radial means and distribution percentiles
paper_objects_<case>.csv                # object area, centroid, mean, and maximum rain
```

Track shifting translates each 6-h forecast by its forecast-track minus
best-track displacement at the valid time. RMW is read from ATCF column 20 and
converted from nautical miles to kilometres; `rmw_fallback_km` (50 km by
default) is used and documented by the configuration when RMW is absent.

The framework uses MRMS because it is the hourly land QPE source already
available in this project. It is therefore a land-focused analogue of the
paper's CCPA verification, not a claim that MRMS equals CCPA. Ocean verification
with IMERG is not fabricated when IMERG is unavailable. Object identification
uses a transparent smooth-threshold-connected-component method and reports its
threshold, smoothing, and minimum area in the YAML. It reproduces the paper's
object figure family but is not the MET MODE fuzzy-logic implementation.

### Multiple storms

List paper-storm YAMLs in a suite:

```yaml
label: HAFS multi-storm QPF verification
storms:
  - storms/helene_paper.yaml
  - storms/ian_paper.yaml
  - storms/idalia_paper.yaml
out_dir: analysis/output/paper_multistorm
```

Then run:

```bash
python analysis/run.py storms/paper_multistorm.yaml paper
```

Every storm first receives all storm-centric products. Contingency counts are
then pooled across storms by model, lead time, track-shift state, and threshold;
ETS is recomputed from the pooled counts rather than averaged. Initializations
are resampled as whole forecast events for the plotted 95% bootstrap intervals.

---

## Project layout

```
analysis/
  run.py          # entry point — loads a YAML case, runs the products
  hafs_case.py    # StormCase config: YAML loader + .atcfunix track parser
  hafs_common.py  # shared plumbing: HAFS GRIB readers, nest accumulation, MRMS
  parent_qpf.py   # parent QPF vs MRMS vs Stage IV 3-panel + Stage IV downloader
  ets_full.py     # combined parent+nest ETS figure + CSV
  ets_score.py    # ETS contingency math + MRMS/swath helpers
  cycles.py       # cross-initialization comparison on a common valid window
  paper_case.py    # per-storm and multi-storm paper YAML configuration
  paper.py         # lead-time, track-shift, RMW, radial, and object products
  tests/          # standalone unit tests (run: python3 analysis/tests/<file>.py)
storms/           # per-storm YAML case files
analysis/output/  # generated figures (gitignored)
```

## Data

Large model output lives on the HPC and is never committed. See the HFSA/HFSB
paths in the example `storms/*.yaml`.
