"""Shared fixtures.

Fixtures that touch the network or the GPU are session-scoped and marked, so
`pytest -m "not slow"` gives a fast, offline suite that still covers every piece
of logic that does not need a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forgelm import dataio  # noqa: E402
from forgelm.datagen import generate_dataset  # noqa: E402
from forgelm.splits import build_manifest  # noqa: E402


@pytest.fixture(scope="session")
def records():
    """Freshly generated records -- not read from disk.

    Generating rather than loading means the tests validate the *generator*,
    so a test cannot pass just because a stale file on disk happens to be fine.
    """
    return generate_dataset()


@pytest.fixture(scope="session")
def manifest(records):
    return build_manifest(records)


@pytest.fixture(scope="session")
def on_disk_records():
    """The committed dataset, if it has been built."""
    if not dataio.PROCESSED_DATASET.exists():
        pytest.skip("dataset not built; run scripts/00_build_dataset.py")
    return dataio.read_jsonl(dataio.PROCESSED_DATASET)


@pytest.fixture(scope="session")
def tokenizer():
    pytest.importorskip("transformers")
    from forgelm.modeling import load_tokenizer

    try:
        return load_tokenizer()
    except Exception as exc:  # network or cache failure
        pytest.skip(f"tokenizer unavailable: {exc}")


@pytest.fixture(scope="session")
def cuda_available():
    torch = pytest.importorskip("torch")
    return torch.cuda.is_available()
