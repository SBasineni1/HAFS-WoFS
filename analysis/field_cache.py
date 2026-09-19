"""Content-addressed cache of derived parent-window grids (no pickle)."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def parent_cache_path(cache_dir, paths, f1, f2, grid_lat, grid_lon):
    """Invalidate on source replacement, window/grid changes or version bump.

    Size + nanosecond mtime avoid reading huge GRIBs just to check the cache.
    Use --refresh-cache if an archive was altered while preserving both.
    """
    sources = []
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        sources.append((str(path), stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256(json.dumps(
        {"version": 1, "sources": sources, "f1": f1, "f2": f2},
        sort_keys=True).encode())
    for grid in (grid_lat, grid_lon):
        array = np.ascontiguousarray(grid, dtype=np.float64)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return Path(cache_dir) / f"parent_{digest.hexdigest()}.npy"


def load_field(path, shape):
    try:
        field = np.load(path, allow_pickle=False)
        if field.shape == shape and field.dtype == np.float64:
            return field
    except (OSError, ValueError, EOFError):
        pass
    return None


def save_field(path, field):
    """Atomic replace keeps interrupted/concurrent runs from exposing partials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp",
                                         delete=False) as fh:
            temp_path = Path(fh.name)
            np.save(fh, np.asarray(field, dtype=np.float64), allow_pickle=False)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
