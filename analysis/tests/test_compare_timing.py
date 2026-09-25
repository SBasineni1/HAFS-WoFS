"""Timing comparison must pair old sequential logs with new pooled logs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_timing import compare, job_times, parse_log, stage_times

OLD_LOG = """\
Runtime: host=hercules-01-1 pid=10 Slurm job=1 CPU affinity=4 requested workers=4
Timing: parent extraction 100.0s
Timing: MRMS 30.0s
Timing: Stage IV 50.0s
Timing: fields 185.0s
Timing: scoring/tables 60.0s
Timing: plots/animations 90.0s
Timing: total 340.0s
"""

NEW_LOG = """\
Runtime: host=hercules-01-2 pid=20 Slurm job=2 CPU affinity=4 requested workers=4
Timing: parent extraction 80.0s
  MRMS: started pid=21
Timing: MRMS 30.0s
Timing: Stage IV 45.0s
Timing: observations wall time 46.0s
Timing: fields 130.0s
Timing: Score 2024092400 20.0s
Timing: Score 2024092406 22.0s
Timing: scoring/tables 25.0s
Timing: Render fss.png 12.0s
Timing: plots/animations 30.0s
Timing: total 190.0s
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_old_observations_are_sequential_sum_and_new_use_wall_time(tmp_path):
    old, _ = parse_log(_write(tmp_path, "old.log", OLD_LOG))
    new, runtime = parse_log(_write(tmp_path, "new.log", NEW_LOG))
    assert stage_times(old)["observations"] == 80.0
    assert stage_times(new)["observations"] == 46.0
    assert runtime == {"cpus": "4", "workers": "4"}
    assert job_times(new)["Score"] == (42.0, 2)


def test_only_last_run_in_log_is_used(tmp_path):
    log = _write(tmp_path, "two.log", OLD_LOG + NEW_LOG)
    timings, _ = parse_log(log)
    assert timings["total"] == 190.0
    assert "Score 2024092400" in timings


def test_compare_reports_speedup_and_flags_setting_mismatch(tmp_path):
    old = _write(tmp_path, "old.log", OLD_LOG)
    new = _write(tmp_path, "new.log",
                 NEW_LOG.replace("CPU affinity=4", "CPU affinity=8"))
    report = compare(old, new)
    total = next(line for line in report.splitlines() if line.startswith("total"))
    assert "340.0s" in total and "190.0s" in total and "1.79x" in total
    assert "WARNING" in report


def test_log_without_timing_lines_is_rejected(tmp_path):
    old = _write(tmp_path, "old.log", OLD_LOG)
    empty = _write(tmp_path, "empty.log", "Phase: discovery\n")
    with pytest.raises(SystemExit, match="empty.log"):
        compare(old, empty)
