"""
Inspect the APCP / total-precip (tp) records inside HAFS GRIB2 files.

Prints, for a few forecast hours (first / middle / last), every tp record
with its grid size and accumulation window (startStep -> endStep) plus data
stats.  This tells us whether HAFS APCP is:

  * cumulative  — window 0 -> fhour, totals grow each hour   (use np.fmax)
  * incremental — window (fhour-Δ) -> fhour, per-interval rain (must SUM)

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/inspect_hafs_apcp.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from qpf_full_run import (
    HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER,
    discover_files, read_hafs_tp_records,
)


def main():
    file_pairs = discover_files(HAFS_RUN_DIR, FILE_GLOB, FHOURS_FILTER)
    if not file_pairs:
        print(f"No files matching {FILE_GLOB} in {HAFS_RUN_DIR}")
        return

    # Sample first, middle, and last forecast hours.
    idxs = sorted(set([0, len(file_pairs) // 2, len(file_pairs) - 1]))
    sample = [file_pairs[i] for i in idxs]

    print(f"Total HAFS files: {len(file_pairs)} "
          f"(F{file_pairs[0][0]:03d} -> F{file_pairs[-1][0]:03d})\n")

    for fhour, fp in sample:
        print("=" * 74)
        print(f"F{fhour:03d}  {fp.name}")
        print("-" * 74)
        try:
            recs = read_hafs_tp_records(fp)
        except Exception as e:
            print(f"  read failed: {e}")
            continue
        if not recs:
            print("  no tp records")
            continue
        print(f"  {'grid (ni*nj)':>14} {'window':>12} {'stepType':>10} "
              f"{'min':>7} {'max':>8} {'mean':>7}")
        for r in sorted(recs, key=lambda x: -x["npoints"]):
            d = r["data"]
            with np.errstate(all="ignore"):
                dmin = np.nanmin(d)
                dmax = np.nanmax(d)
                dmean = np.nanmean(d)
            window = f"{r['start_step']}->{r['end_step']}h"
            print(f"  {r['npoints']:>14,} {window:>12} {str(r['step_type']):>10} "
                  f"{dmin:>7.1f} {dmax:>8.1f} {dmean:>7.2f}")
    print("=" * 74)
    print("\nInterpretation:")
    print("  - If the largest-grid window stays 0->fhour and max grows with "
          "fhour -> CUMULATIVE.")
    print("  - If the window is a short interval like 123->126h -> INCREMENTAL "
          "(buckets must be summed).")


if __name__ == "__main__":
    main()
