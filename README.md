# HAFS&WoFS
A Comparative Evaluation of Rainfall Forecast Skill for Landfalling Tropical Cyclones from HAFS and WoFS

## Running a new storm

The QPF/ETS pipeline is case-driven. To analyze a new storm or a different
HAFS run, copy a YAML case file and change `run_dir`:

    cp storms/helene_hfsa.yaml storms/<storm>_<model>.yaml
    # edit run_dir (and optionally domain)

Then run from the repo root:

    python analysis/run.py storms/<storm>_<model>.yaml all      # parent + ets
    python analysis/run.py storms/<storm>_<model>.yaml parent   # 3-panel QPF figure only
    python analysis/run.py storms/<storm>_<model>.yaml ets      # ETS curves + CSV only

The storm track, init time, and HAFS-A/B label are read automatically from the
run's `.atcfunix` file and path. Only `run_dir` is required in the YAML; all
other keys are optional overrides. Outputs land in `out_dir`
(default `analysis/output/<case_name>/`).
