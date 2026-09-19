#!/bin/bash
#SBATCH --job-name=hafs-cycles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cycles-%j.log

# Submit from the repository root with the hafs conda environment activated.
# Supply your site's account/partition/QoS via sbatch command-line options.
# Override memory, time, and CPUs on the sbatch command line as needed.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo "Submit this script with sbatch; running it with bash does not allocate CPUs." >&2
    exit 2
fi
if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch [Slurm options] analysis/cycles.sbatch <case.yaml> [cycles options]" >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ ! -f analysis/run.py ]]; then
    echo "Submit from the repository root (the directory containing analysis/run.py)." >&2
    exit 2
fi
if [[ -z "${CONDA_PREFIX:-}" || ! -x "$CONDA_PREFIX/bin/python" ]]; then
    echo "Activate the hafs conda environment before submitting this job." >&2
    exit 2
fi

case_yaml=$1
shift
# Prevent nested numerical-library threads from multiplying the worker count.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

# One Python parent and its pool share this task's allocated CPU set.
# The final --workers keeps the worker request aligned with the allocation.
exec srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores "$CONDA_PREFIX/bin/python" -u analysis/run.py \
    "$case_yaml" cycles "$@" --workers "$SLURM_CPUS_PER_TASK"
