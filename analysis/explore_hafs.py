"""
Explore HFSA and HFSB output file structure on Hercules HPC.
Run this script on Hercules: python explore_hafs.py
"""

from pathlib import Path
import sys

BASE = Path("/work2/noaa/aoml-hafs1/ahazelto/student_data/suchit_data/helene")
HFSA = BASE / "HFSA"
HFSB = BASE / "HFSB"


def summarize_directory(path: Path, label: str, max_files: int = 50):
    print(f"\n{'='*60}")
    print(f"{label}: {path}")
    print('='*60)

    if not path.exists():
        print(f"  ERROR: Path does not exist.")
        return

    all_files = list(path.rglob("*"))
    files = [f for f in all_files if f.is_file()]
    dirs = [f for f in all_files if f.is_dir()]

    print(f"  Directories: {len(dirs)}")
    print(f"  Files:       {len(files)}")

    # Show top-level subdirectories
    top_dirs = sorted({f.relative_to(path).parts[0] for f in all_files if f.relative_to(path).parts})
    if top_dirs:
        print(f"\n  Top-level subdirs:")
        for d in top_dirs:
            print(f"    {d}/")

    # File extensions summary
    exts = {}
    for f in files:
        ext = f.suffix.lower() or "(no ext)"
        exts[ext] = exts.get(ext, 0) + 1
    if exts:
        print(f"\n  File types:")
        for ext, count in sorted(exts.items(), key=lambda x: -x[1]):
            print(f"    {ext:15s}  {count}")

    # Sample file listing
    print(f"\n  Sample files (up to {max_files}):")
    for f in sorted(files)[:max_files]:
        rel = f.relative_to(path)
        size_mb = f.stat().st_size / 1e6
        print(f"    {str(rel):<60s}  {size_mb:8.2f} MB")

    if len(files) > max_files:
        print(f"    ... and {len(files) - max_files} more files")


if __name__ == "__main__":
    print("HAFS Helene Output Explorer")
    print(f"Base path: {BASE}\n")

    if not BASE.exists():
        print("ERROR: Base path not found. Are you running this on Hercules?")
        sys.exit(1)

    summarize_directory(HFSA, "HFSA")
    summarize_directory(HFSB, "HFSB")
