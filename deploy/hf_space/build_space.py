"""Assemble the Hugging Face Space directory from this repository.

The Space needs to be a self-contained tree: Hugging Face clones it and runs
`app.py`, so the `forgelm` package and the trained adapter have to travel with
it. Rather than duplicating 45 MB of weights inside the GitHub repository, this
script stages a copy on demand.

    python deploy/hf_space/build_space.py --out build/space

The layout it produces mirrors the source repository closely enough that
`forgelm.ledger.REPO_ROOT` (which is `parents[2]` of the package file) still
resolves to the Space root:

    <out>/app.py
    <out>/requirements.txt
    <out>/README.md
    <out>/.gitattributes
    <out>/src/forgelm/*.py
    <out>/artifacts/lora_adapter/*
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPACE_DIR.parents[1]

# Copied verbatim from this directory into the Space root.
SPACE_FILES = ["app.py", "requirements.txt", "README.md", ".gitattributes"]

# Copied from the repository. The adapter is the trained artifact; src is the
# package app.py imports.
TREES = [
    (REPO_ROOT / "src" / "forgelm", "src/forgelm"),
    (REPO_ROOT / "artifacts" / "lora_adapter", "artifacts/lora_adapter"),
]


def build(out: Path, clean: bool = True) -> int:
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for name in SPACE_FILES:
        source = SPACE_DIR / name
        if not source.exists():
            print(f"missing Space file: {source}", file=sys.stderr)
            return 1
        shutil.copy2(source, out / name)
        print(f"  {name}")

    for source, relative in TREES:
        if not source.exists():
            print(f"missing tree: {source}", file=sys.stderr)
            return 1
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        n = sum(1 for _ in target.rglob("*") if _.is_file())
        print(f"  {relative}/  ({n} files)")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nStaged {out}  ({total / 1e6:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "build" / "space"),
                    help="directory to stage the Space into")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete an existing output directory first")
    args = ap.parse_args()
    return build(Path(args.out).resolve(), clean=not args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
