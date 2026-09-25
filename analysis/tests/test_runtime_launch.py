"""Check allocation diagnostics and the batch launch without an HPC cluster."""

import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parallel import report_runtime, report_workers

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "analysis" / "cycles.sbatch"


def test_runtime_distinguishes_pid_from_slurm_job(monkeypatch, capsys):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    report_runtime(4)
    output = capsys.readouterr().out
    assert f"pid={os.getpid()} Slurm job=none" in output
    assert "outside a Slurm allocation" in output
    assert "independent observation sources" in output


def test_slurm_default_cap_is_explained(monkeypatch, capsys):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    report_runtime(4)
    report_workers(4, 8)
    output = capsys.readouterr().out
    assert "Slurm job=123" in output
    assert "unset (worker cap: 1)" in output
    assert "Parent extraction: 1 worker(s)" in output
    assert "Worker count reduced" in output


def test_batch_script_rejects_plain_shell_launch():
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    result = subprocess.run(["bash", str(SCRIPT), "case.yaml"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "Submit this script with sbatch" in result.stderr


def test_batch_launch_binds_one_task_and_preserves_arguments(tmp_path):
    # Stand in for srun: inspect exactly what the script would launch.
    conda = tmp_path / "conda env"
    (conda / "bin").mkdir(parents=True)
    (conda / "bin/python").symlink_to(sys.executable)
    srun = tmp_path / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], "
        "'omp': os.environ['OMP_NUM_THREADS']}))\n")
    srun.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="4",
               SLURM_SUBMIT_DIR=str(ROOT), CONDA_PREFIX=str(conda),
               OMP_NUM_THREADS="8")
    result = subprocess.run(
        ["bash", str(SCRIPT), "case with spaces.yaml", "--no-animation"],
        env=env, capture_output=True, text=True, check=True)
    launch = json.loads(result.stdout)
    assert launch["args"] == [
        "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--cpu-bind=cores",
        str(conda / "bin/python"), "-u", "analysis/run.py",
        "case with spaces.yaml", "cycles", "--no-animation", "--workers", "4"]
    assert launch["omp"] == "1"
