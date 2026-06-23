"""HFSA-vs-HFSB head-to-head comparison over a shared best-track swath.

Loaded via run.py:  python analysis/run.py storms/<name>_compare.yaml compare
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_DEFAULT_FSS_SCALES = [1, 3, 5, 11, 21, 41]
_DEFAULT_FSS_PLOT_THR = [10, 25, 50]


def load_comparison(path):
    """Parse + validate a comparison YAML and fill defaults (no GRIB loading)."""
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    cases = cfg.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError(f"'cases' must list exactly 2 case YAMLs in {path}")
    if "best_track" not in cfg:
        raise KeyError(f"'best_track' is required in {path}")
    return {
        "label": cfg.get("label", path.stem),
        "case_paths": [str(c) for c in cases],
        "best_track": str(cfg["best_track"]),
        "out_dir": Path(cfg["out_dir"]) if cfg.get("out_dir")
                   else Path("analysis/output") / path.stem,
        "thresholds_mm": cfg.get("thresholds_mm"),
        "fss_scales_cells": cfg.get("fss_scales_cells", list(_DEFAULT_FSS_SCALES)),
        "fss_plot_thresholds": cfg.get("fss_plot_thresholds",
                                       list(_DEFAULT_FSS_PLOT_THR)),
    }
