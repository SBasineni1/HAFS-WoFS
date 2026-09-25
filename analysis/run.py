"""Single entry point for the HAFS QPF/ETS framework.

    python analysis/run.py <case.yaml> [parent|ets|rmse|cycles|cycles-compare|all|compare|replot|ml]

Loads a StormCase from the YAML case file and runs the requested product(s):
  parent  nest + parent QPF vs MRMS + Stage IV 4-panel figure
  ets     the combined parent+nest ETS-vs-threshold figure + CSV
  rmse    storm-total RMSE/MAE/bias/r scatter panels + CSV
  cycles  per-initialization comparison on a common valid window (takes a cycles YAML)
  cycles-compare compare HAFS-A/B/M cycle CSVs; missing models are allowed
  all     parent + ets + rmse (fields built once; default)
  compare HFSA-vs-HFSB rainfall comparison (takes a comparison YAML)
  replot  redraw comparison or cycles figures from existing CSVs (no recompute)
  ml      pooled ML regime diagnostics over a feature CSV
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = ("parent", "ets", "rmse", "cycles", "cycles-compare", "all",
            "compare", "replot", "ml")


def parse_options(argv):
    from parallel import positive_workers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_path")
    parser.add_argument("command", nargs="?", choices=COMMANDS, default="all")
    parser.add_argument("--workers", type=positive_workers,
                        help="cycles: processes for extraction, observations, scoring, "
                             "rendering and features (overrides YAML)")
    parser.add_argument("--replot", action="store_true",
                        help="cycles: redraw table-based plots from existing CSVs")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="cycles: rebuild cached parent-window grids")
    parser.add_argument("--no-animation", action="store_true",
                        help="cycles: skip GIF rendering")
    parser.add_argument("--no-ml-features", action="store_true",
                        help="cycles: skip optional environmental feature extraction")
    args = parser.parse_args(argv)
    if args.command != "cycles" and any((args.workers is not None, args.replot,
            args.refresh_cache, args.no_animation, args.no_ml_features)):
        parser.error("runtime options apply to the cycles command only")
    if args.replot and (args.workers is not None or args.refresh_cache):
        parser.error("--replot uses CSVs; --workers/--refresh-cache require computation")
    return args


def parse_args(argv):
    """Keep the established (yaml_path, command) parser interface."""
    args = parse_options(argv)
    return args.yaml_path, args.command


def dispatch(case, command):
    """Run the requested product(s) for a loaded StormCase."""
    from parent_qpf import generate_parent_figure
    from ets_full import compute_ets, build_verification_fields
    from rmse_scatter import compute_rmse
    if command in ("parent", "all"):
        generate_parent_figure(case)
    if command == "ets":
        compute_ets(case)
    if command == "rmse":
        compute_rmse(case)
    if command == "all":
        # Build the expensive verification fields once, share across products.
        fields = build_verification_fields(case)
        compute_ets(case, fields=fields)
        compute_rmse(case, fields=fields)


def main(argv):
    # Keep progress visible in redirected run.log and Slurm output files.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    args = parse_options(argv)
    yaml_path, command = args.yaml_path, args.command
    if command == "replot":
        import yaml
        with open(yaml_path) as fh:
            is_cycles = "run_root" in (yaml.safe_load(fh) or {})
        if is_cycles:
            from hafs_case import cycles_from_yaml
            from cycles import replot_cycles_from_csv
            replot_cycles_from_csv(cycles_from_yaml(yaml_path))
            return
    if command == "ml":
        from ml_regime import load_ml_config, run_ml
        run_ml(load_ml_config(yaml_path))
        return
    if command == "cycles-compare":
        from cycles_compare import (load_cycles_comparison,
                                    generate_cycles_comparison)
        generate_cycles_comparison(load_cycles_comparison(yaml_path))
        return
    if command in ("compare", "replot"):
        from compare import (load_comparison, generate_comparison,
                             replot_from_csv)
        cfg = load_comparison(yaml_path)
        (generate_comparison if command == "compare" else replot_from_csv)(cfg)
        return
    if command == "cycles":
        from hafs_case import cycles_from_yaml
        from cycles import compute_cycles, replot_cycles_from_csv
        ccase = cycles_from_yaml(yaml_path)
        if args.workers is not None:
            ccase.workers = args.workers
        if args.no_animation:
            ccase.make_animation = False
        if args.no_ml_features:
            ccase.ml_features = False
        print(f"Case   : {ccase.storm_name} ({ccase.model_label})")
        print(f"Window : {ccase.valid_start:%Y-%m-%d %HZ} -> "
              f"{ccase.valid_end:%Y-%m-%d %HZ}  | run_root: {ccase.run_root}")
        print(f"Output : {ccase.out_dir}  | command: {command}")
        if args.replot:
            replot_cycles_from_csv(ccase)
        else:
            compute_cycles(ccase, refresh_cache=args.refresh_cache)
        return
    from hafs_case import from_yaml
    case = from_yaml(yaml_path)
    print(f"Case   : {case.storm_name} ({case.model_label})")
    print(f"Init   : {case.init_dt:%Y-%m-%d %HZ}  | run_dir: {case.run_dir}")
    print(f"Domain : {case.domain}  | track points: {len(case.track)}")
    print(f"Output : {case.out_dir}  | command: {command}")
    dispatch(case, command)


if __name__ == "__main__":
    main(sys.argv[1:])
