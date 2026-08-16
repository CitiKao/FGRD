#!/usr/bin/env python3
"""Fail if a public release contains datasets, weights, caches, or oversized files."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    "data",
    "datasets",
    "artifacts",
    "paper_checkpoints",
    "paper_cache",
    "paper_weights",
    "checkpoints",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".pyc"}
MAX_BYTES = 10 * 1024 * 1024


def main() -> int:
    violations: list[str] = []
    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    for path in files:
        relative = path.relative_to(ROOT)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & FORBIDDEN_PARTS:
            violations.append(f"forbidden directory: {relative.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden binary artifact: {relative.as_posix()}")
        if path.stat().st_size > MAX_BYTES:
            violations.append(f"file exceeds 10 MiB: {relative.as_posix()}")

    result_path = ROOT / "results" / "controlled_ablation" / "controlled_ablation_results.csv"
    with result_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    variants = {row.get("variant") for row in rows}
    if len(rows) != 24 or len(variants) != 8:
        violations.append(
            f"controlled ablation must contain 24 rows / 8 variants; found {len(rows)} / {len(variants)}"
        )

    if violations:
        print("Release audit: FAIL")
        for violation in violations:
            print(f"- {violation}")
        return 1
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Release audit: PASS ({len(files)} files, {total_bytes} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
