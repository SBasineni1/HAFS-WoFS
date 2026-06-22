"""Single entry point for the HAFS QPF/ETS framework.

    python analysis/run.py <case.yaml> [parent|ets|all]

Loads a StormCase from the YAML case file and runs the requested product(s):
  parent  the parent-domain QPF vs MRMS vs Stage IV 3-panel figure
  ets     the combined parent+nest ETS-vs-threshold figure + CSV
  all     both (default)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = ("parent", "ets", "all")


def parse_args(argv):
    """(yaml_path, command) from argv; command defaults to 'all'."""
    if not argv:
        print("usage: run.py <case.yaml> [parent|ets|all]")
        raise SystemExit(2)
    yaml_path = argv[0]
    command = argv[1] if len(argv) > 1 else "all"
    if command not in COMMANDS:
        print(f"unknown command '{command}'; choose from {COMMANDS}")
        raise SystemExit(2)
    return yaml_path, command


def dispatch(case, command):
    """Run the requested product(s) for a loaded StormCase."""
    from parent_qpf import generate_parent_figure
    from ets_full import compute_ets
    if command in ("parent", "all"):
        generate_parent_figure(case)
    if command in ("ets", "all"):
        compute_ets(case)


def main(argv):
    from hafs_case import from_yaml
    yaml_path, command = parse_args(argv)
    case = from_yaml(yaml_path)
    print(f"Case   : {case.storm_name} ({case.model_label})")
    print(f"Init   : {case.init_dt:%Y-%m-%d %HZ}  | run_dir: {case.run_dir}")
    print(f"Domain : {case.domain}  | track points: {len(case.track)}")
    print(f"Output : {case.out_dir}  | command: {command}")
    dispatch(case, command)


if __name__ == "__main__":
    main(sys.argv[1:])
