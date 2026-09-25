"""Compare stage timings between two cycles Slurm logs.

    python analysis/compare_timing.py cycles-OLD.log cycles-NEW.log

Reads the ``Timing:`` lines that ``run.py <case.yaml> cycles`` prints (the
cycles-<jobid>.log files from analysis/cycles.sbatch). If a log holds several
runs, only the last one is used. Older logs time MRMS and Stage IV one after
the other, so their sum stands in for the newer "observations wall time".
Worker time is the summed duration of per-job lines; with a process pool it can
exceed the wall time of its stage.
"""

import argparse
import re
from pathlib import Path

TIMING = re.compile(r"^Timing: (?P<label>.+) (?P<secs>\d+(?:\.\d+)?)s\s*$")
RUNTIME = re.compile(r"CPU affinity=(?P<cpus>\d+) requested workers=(?P<workers>\S+)")
STAGES = ["parent extraction", "observations", "fields", "scoring/tables",
          "plots/animations", "ML features", "total"]
JOB_GROUPS = ["MRMS", "Stage IV", "Score", "Render", "Features"]


def parse_log(path):
    """Return ({label: seconds}, {"cpus", "workers"}) for the last run in a log."""
    timings, runtime = {}, {}
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        match = RUNTIME.search(line)
        if match:  # Each run starts with a Runtime line; keep the latest run.
            timings, runtime = {}, match.groupdict()
            continue
        match = TIMING.match(line)
        if match:
            label = match["label"]
            timings[label] = timings.get(label, 0.0) + float(match["secs"])
    return timings, runtime


def stage_times(timings):
    """Top-level wall times, with the old sequential observations summed."""
    stages = {name: timings[name] for name in STAGES if name in timings}
    if "observations wall time" in timings:
        stages["observations"] = timings["observations wall time"]
    elif "MRMS" in timings or "Stage IV" in timings:
        stages["observations"] = timings.get("MRMS", 0.0) + timings.get("Stage IV", 0.0)
    return stages


def job_times(timings):
    """Summed worker seconds and job count per job group (Score, Render, ...)."""
    groups = {}
    for label, secs in timings.items():
        for group in JOB_GROUPS:
            if label == group or label.startswith(group + " "):
                total, count = groups.get(group, (0.0, 0))
                groups[group] = (total + secs, count + 1)
    return groups


def _cell(value):
    return "—" if value is None else f"{value:.1f}s"


def _change(old, new):
    if old is None or new is None:
        return ""
    if new == 0:
        return "n/a" if old == 0 else "inf faster"
    delta = new - old
    return f"{delta:+7.1f}s  {old / new:5.2f}x"


def compare(old_path, new_path):
    old, old_runtime = parse_log(old_path)
    new, new_runtime = parse_log(new_path)
    if not old or not new:
        empty = old_path if not old else new_path
        raise SystemExit(f"No 'Timing:' lines found in {empty}")

    lines = [f"old: {old_path}", f"new: {new_path}"]
    for name, runtime in (("old", old_runtime), ("new", new_runtime)):
        if runtime:
            lines.append(f"{name} runtime: CPU affinity={runtime['cpus']} "
                         f"requested workers={runtime['workers']}")
    if old_runtime and new_runtime and old_runtime != new_runtime:
        lines.append("WARNING: CPU/worker settings differ; speedups are not "
                     "a like-for-like code comparison.")

    rows = [("stage (wall)", "old", "new", "change  speedup")]
    old_stages, new_stages = stage_times(old), stage_times(new)
    for name in STAGES:
        if name in old_stages or name in new_stages:
            o, n = old_stages.get(name), new_stages.get(name)
            rows.append((name, _cell(o), _cell(n), _change(o, n)))

    old_jobs, new_jobs = job_times(old), job_times(new)
    if old_jobs or new_jobs:
        rows.append(("", "", "", ""))
        rows.append(("worker time (sum)", "old", "new", "change  speedup"))
        for group in JOB_GROUPS:
            if group in old_jobs or group in new_jobs:
                o, oc = old_jobs.get(group, (None, 0))
                n, nc = new_jobs.get(group, (None, 0))
                label = f"{group} (jobs {oc}/{nc})" if oc > 1 or nc > 1 else group
                rows.append((label, _cell(o), _cell(n), _change(o, n)))

    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    for row in rows:
        lines.append(f"{row[0]:<{widths[0]}}  {row[1]:>{widths[1]}}  "
                     f"{row[2]:>{widths[2]}}  {row[3]}".rstrip())
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old_log", help="log from the baseline run")
    parser.add_argument("new_log", help="log from the optimized run")
    args = parser.parse_args(argv)
    print(compare(args.old_log, args.new_log))


if __name__ == "__main__":
    main()
