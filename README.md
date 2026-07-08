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

Running a new storm only requires a small YAML file pointing at the run
directory — the storm track, init time, and HAFS-A/B label are read
automatically from the run's `.atcfunix` track and path.

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
python analysis/run.py storms/<case>.yaml all      # parent figure + ETS + RMSE (default)
python analysis/run.py storms/<case>.yaml parent   # 3-panel QPF figure only
python analysis/run.py storms/<case>.yaml ets      # ETS plot + CSV only
python analysis/run.py storms/<case>.yaml rmse     # RMSE scatter + CSV only
```

The two shipped examples:

```bash
python analysis/run.py storms/helene_hfsa.yaml all
python analysis/run.py storms/helene_hfsb.yaml all
```

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
```

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

---

## 5. Comparing HFSA vs HFSB (head-to-head)

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

## 6. How to read the ETS plot

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

## 7. How to read the RMSE scatter

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

## Project layout

```
analysis/
  run.py          # entry point — loads a YAML case, runs the products
  hafs_case.py    # StormCase config: YAML loader + .atcfunix track parser
  hafs_common.py  # shared plumbing: HAFS GRIB readers, nest accumulation, MRMS
  parent_qpf.py   # parent QPF vs MRMS vs Stage IV 3-panel + Stage IV downloader
  ets_full.py     # combined parent+nest ETS figure + CSV
  ets_score.py    # ETS contingency math + MRMS/swath helpers
  tests/          # standalone unit tests (run: python3 analysis/tests/<file>.py)
storms/           # per-storm YAML case files
analysis/output/  # generated figures (gitignored)
```

## Data

Large model output lives on the HPC and is never committed. See the HFSA/HFSB
paths in the example `storms/*.yaml`.
