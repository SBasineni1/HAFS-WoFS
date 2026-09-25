"""Small, allocation-aware process pools for independent analysis jobs."""

import os
import socket
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context
from time import perf_counter


def positive_workers(value):
    """Validate CLI/YAML worker counts without silently truncating floats."""
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("workers must be a positive integer") from exc
    if isinstance(value, bool) or str(value) != str(count) or count < 1:
        raise ValueError("workers must be a positive integer")
    return count


def worker_count(requested, jobs):
    """Respect the Slurm allocation and, where exposed, CPU affinity."""
    limits = [positive_workers(requested), max(1, jobs)]
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        limits.append(positive_workers(os.environ["SLURM_CPUS_PER_TASK"]))
    elif os.environ.get("SLURM_JOB_ID"):
        limits.append(1)  # Slurm's default is one CPU per task.
    if hasattr(os, "sched_getaffinity"):
        limits.append(len(os.sched_getaffinity(0)))
    else:
        limits.append(os.cpu_count() or 1)
    return max(1, min(limits))


def report_runtime(requested):
    """Identify the actual execution host/allocation before expensive work."""
    affinity = (len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity") else os.cpu_count())
    job = os.environ.get("SLURM_JOB_ID")
    print(f"Runtime: host={socket.gethostname()} pid={os.getpid()} "
          f"Slurm job={job or 'none'} CPU affinity={affinity} "
          f"requested workers={requested}", flush=True)
    if job:
        print("Runtime: SLURM_CPUS_PER_TASK="
              f"{os.environ.get('SLURM_CPUS_PER_TASK', 'unset (worker cap: 1)')}",
              flush=True)
    else:
        print("Runtime: running outside a Slurm allocation. --workers does not "
              "request compute-node CPUs. On Hercules, submit with "
              "sbatch analysis/cycles.sbatch <case.yaml>.", flush=True)
    print("Runtime: process pools handle parent extraction, independent "
          "observation sources, per-cycle scoring, plots/GIFs and ML features. "
          "Discovery and shared table writes run in the parent.",
          flush=True)


def report_workers(requested, jobs, stage="Parent extraction"):
    count = worker_count(requested, jobs)
    print(f"{stage}: {count} worker(s) "
          f"(requested={requested}, jobs={jobs})", flush=True)
    if count < requested:
        print("Worker count reduced by available jobs, CPU affinity, or the "
              "Slurm CPUs-per-task limit. Check the Runtime lines above.",
              flush=True)


def report_phase(name):
    """Flush before work starts, including when stdout is redirected."""
    print(f"Phase: {name}", flush=True)


def run_stage_job(job):
    """Run a picklable stage with an explicit PID and elapsed time."""
    label, function, args = job
    print(f"  {label}: started pid={os.getpid()}", flush=True)
    started = perf_counter()
    result = function(*args)
    print(f"Timing: {label} {perf_counter() - started:.1f}s", flush=True)
    return result


def _configure_child():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


@contextmanager
def _single_threaded_children():
    # Set before spawn imports NumPy/SciPy. Each process owns one CPU; nested
    # BLAS/OpenMP pools would oversubscribe the allocation and multiply memory.
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def ordered_map(function, jobs, workers=1):
    """Yield results in input order; workers must be module-level functions.

    Spawn avoids inheriting native GRIB/plotting state. Serial execution stays
    in-process for debugging and small jobs. Exceptions propagate to callers.
    """
    jobs = list(jobs)
    count = worker_count(workers, len(jobs))
    if count == 1:
        yield from map(function, jobs)
        return
    with _single_threaded_children():
        with ProcessPoolExecutor(max_workers=count,
                                 mp_context=get_context("spawn"),
                                 initializer=_configure_child) as pool:
            # Keep at most count large array jobs/results in flight. This also
            # works on Python versions before Executor.map gained buffersize.
            pending = deque()
            remaining = iter(jobs)
            for _ in range(count):
                pending.append(pool.submit(function, next(remaining)))
            while pending:
                result = pending.popleft().result()
                yield result
                del result
                try:
                    job = next(remaining)
                except StopIteration:
                    continue
                pending.append(pool.submit(function, job))
