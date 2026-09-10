#!/usr/bin/env bash
# Verify that every source file on disk is actually tracked by git.
#
# Why this exists: an unanchored "data/" entry in .gitignore also matched
# "src/data/", which silently excluded the entire data package from the first
# commit. Nothing failed locally -- pytest ran, the smoke pipeline ran, the
# push succeeded. It only surfaced when a clean clone tried to import
# src.data and could not. Run this before every push.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import pathlib, subprocess, sys

def tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout
    return {line for line in out.splitlines() if line}

# Whole trees that are generated, not source. Listed as path PREFIXES rather
# than bare directory names so the intent is legible and so a name like
# "output" cannot accidentally swallow an unrelated directory of the same name.
#
# "kaggle/output" is what `kaggle kernels output` downloads: a nested clone of
# this repo plus its result artifacts. Every file under it is legitimately
# untracked, and listing it here is why this check does not report ~25 false
# positives from that clone.
EXCLUDE_PREFIXES = (
    "data/",
    "outputs/",
    "runs/",
    "kaggle/output/",
    "__pycache__/",
    ".pytest_cache/",
    ".git/",
)

def _excluded(path: pathlib.Path) -> bool:
    rel = path.as_posix()
    return any(
        rel.startswith(prefix) or f"/{prefix}" in f"/{rel}"
        for prefix in EXCLUDE_PREFIXES
    )

def on_disk(suffixes: tuple[str, ...]) -> set[str]:
    found = set()
    for path in pathlib.Path(".").rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if _excluded(path):
            continue
        found.add(str(path))
    return found

disk = on_disk((".py", ".yaml", ".yml", ".sh", ".md", ".txt", ".ipynb"))
tracked_files = tracked()

missing = sorted(disk - tracked_files)
print(f"files on disk : {len(disk)}")
print(f"files tracked : {len(tracked_files)}")

if missing:
    print("\n*** NOT TRACKED -- .gitignore is probably swallowing them ***")
    for path in missing:
        rule = subprocess.run(
            ["git", "check-ignore", "-v", path], capture_output=True, text=True
        ).stdout.strip()
        print(f"  {path}")
        if rule:
            print(f"      ignored by: {rule}")
    sys.exit(1)

print("\nOK - every source file on disk is tracked")
PY