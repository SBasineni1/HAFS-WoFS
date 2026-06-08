# HAFS & WoFS Project

## Project Overview
Analysis of HAFS (Hurricane Analysis and Forecast System) model output for Hurricane Helene. Comparing HFSA and HFSB configurations.

## Data
- All large model output lives on Hercules HPC — never commit it to GitHub
- HFSA: `/work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/HFSA`
- HFSB: `/work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene/HFSB`
- Sample data (local): `helene_sample/HFSA/2024092400/`

## File Format
- GRIB2 (`.grb2`) files, read with `cfgrib` + `xarray`
- Use `cfgrib.open_datasets()` to load — returns multiple message groups per file
- Key variable names: `u10`, `v10`, `t2m`, `mslet`, `tp`, `refc`, `pwat`, `t`, `u`, `v`, `gh`, `r`

## Code Conventions
- Scripts go in `analysis/`
- Plot output goes in `analysis/output/` (gitignored)
- Run scripts from repo root: `python analysis/script.py`

## Git
- No `Co-Authored-By` lines in commits
- Don't commit large data files (`.nc`, `.grb2`, `.grb`, etc.)
