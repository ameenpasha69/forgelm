"""Assemble a deployable tree from this repository.

Both deployment targets need the same three things -- the Gradio app, the
`forgelm` package, and the trained adapter -- plus a few files that only make
sense for one of them. Rather than duplicating 45 MB of weights inside the
repository once per target, this stages a copy on demand.

    python deploy/build.py --target cloudrun --out build/cloudrun
    python deploy/build.py --target hfspace  --out build/space

The layout mirrors the source repository closely enough that
`forgelm.ledger.REPO_ROOT` (which is `parents[2]` of the package file) still
resolves to the tree root:

    <out>/app.py
    <out>/requirements.txt
    <out>/src/forgelm/*.py
    <out>/artifacts/lora_adapter/*
    <out>/...                     target-specific files
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_DIR.parent

# Shared by every target.
COMMON_FILES = ["app.py", "requirements.txt"]

TREES = [
    (REPO_ROOT / "src" / "forgelm", "src/forgelm"),
    (REPO_ROOT / "artifacts" / "lora_adapter", "artifacts/lora_adapter"),
]

# Target-specific files, given as (source relative to deploy/, name in tree).
# Hugging Face reads Space configuration out of README.md front matter and
# needs .gitattributes to route the weights through LFS. Cloud Run reads a
# Dockerfile and needs .dockerignore to keep the build context small.
TARGETS = {
    "hfspace": [
        ("hf_space/README.md", "README.md"),
        ("hf_space/.gitattributes", ".gitattributes"),
    ],
    "cloudrun": [
        ("cloudrun/Dockerfile", "Dockerfile"),
        ("cloudrun/.dockerignore", ".dockerignore"),
    ],
}


def build(target: str, out: Path, clean: bool = True) -> int:
    files = [(DEPLOY_DIR / name, name) for name in COMMON_FILES]
    files += [(DEPLOY_DIR / src, dst) for src, dst in TARGETS[target]]

    for source, _ in files:
        if not source.exists():
            print(f"missing file: {source}", file=sys.stderr)
            return 1
    for source, _ in TREES:
        if not source.exists():
            print(f"missing tree: {source}", file=sys.stderr)
            return 1

    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for source, name in files:
        shutil.copy2(source, out / name)
        print(f"  {name}")

    for source, relative in TREES:
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        n = sum(1 for p in destination.rglob("*") if p.is_file())
        print(f"  {relative}/  ({n} files)")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nStaged {target} tree at {out}  ({total / 1e6:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--out", default=None,
                    help="directory to stage into "
                         "(default: build/<target>)")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete an existing output directory first")
    args = ap.parse_args()

    out = Path(args.out) if args.out else REPO_ROOT / "build" / args.target
    return build(args.target, out.resolve(), clean=not args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
