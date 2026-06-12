"""
Dump every GRIB2 record in a file: shortName, name, units, level, step
window, grid size, and data stats.  Works on any HAFS/MRMS GRIB2 file.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/inspect_grib.py /path/to/09l.2024092400.hfsa.parent.swath.grb2

If no path is given, defaults to the parent.swath file for the configured run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import eccodes
from qpf_full_run import HAFS_RUN_DIR, INIT_STR


def default_swath_path():
    hits = sorted(HAFS_RUN_DIR.glob(f"**/*{INIT_STR}*parent.swath*.grb2"))
    return hits[0] if hits else None


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = default_swath_path()
        if path is None:
            print(f"No path given and no parent.swath file found under {HAFS_RUN_DIR}")
            return
    print(f"File: {path}\n")

    hdr = (f"{'#':>3} {'shortName':>10} {'level':>16} {'window':>12} "
           f"{'stepType':>9} {'grid(ni*nj)':>14} {'min':>8} {'max':>9} "
           f"{'mean':>8}  name [units]")
    print(hdr)
    print("-" * len(hdr))

    n = 0
    with open(str(path), "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            n += 1
            try:
                def g(key, cast=str, default="?"):
                    try:
                        return cast(eccodes.codes_get(gid, key))
                    except Exception:
                        return default

                sn = g("shortName")
                name = g("name")
                units = g("units")
                ltype = g("typeOfLevel")
                lev = g("level")
                start = g("startStep")
                end = g("endStep")
                step_type = g("stepType")
                ni = g("Ni", int, 0)
                nj = g("Nj", int, 0)
                try:
                    vals = eccodes.codes_get_values(gid)
                    miss = eccodes.codes_get(gid, "missingValue")
                    vals = np.where(np.abs(vals - miss) < 1.0, np.nan, vals)
                    with np.errstate(all="ignore"):
                        vmin, vmax, vmean = (np.nanmin(vals), np.nanmax(vals),
                                             np.nanmean(vals))
                except Exception:
                    vmin = vmax = vmean = float("nan")

                level = f"{ltype}:{lev}"
                window = f"{start}->{end}h"
                print(f"{n:>3} {sn:>10} {level:>16} {window:>12} {step_type:>9} "
                      f"{ni*nj:>14,} {vmin:>8.1f} {vmax:>9.1f} {vmean:>8.2f}  "
                      f"{name} [{units}]")
            finally:
                eccodes.codes_release(gid)

    print(f"\n{n} records total.")


if __name__ == "__main__":
    main()
