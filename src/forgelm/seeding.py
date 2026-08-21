"""Centralised seed control.

Every random decision in ForgeLM flows through a named seed so that a run can
be described completely by the seeds it used. Named seeds (rather than one
global seed) mean that changing, say, the few-shot demonstration choice does
not silently reshuffle the dataset split.
"""

from __future__ import annotations

import hashlib
import os
import random

# Canonical seeds. Changing any value here changes the experiment identity and
# must be recorded in DECISIONS.md.
SEEDS: dict[str, int] = {
    "dataset_generation": 20240517,
    "split_assignment": 913,
    "fewshot_selection": 41,
    "training": 1337,
    "bootstrap": 7,
}


def derive(name: str, *parts: object) -> int:
    """Derive a stable 32-bit sub-seed from a named seed plus context parts.

    Deterministic across processes and Python versions (hashlib, not hash()).
    Used for per-example randomness so that regenerating example 42 gives the
    same text regardless of how many examples came before it.
    """
    base = SEEDS[name]
    payload = "|".join([str(base), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def rng(name: str, *parts: object) -> random.Random:
    """A fresh, independently-seeded Random instance."""
    return random.Random(derive(name, *parts))


def seed_everything(seed: int, deterministic_torch: bool = True) -> dict[str, object]:
    """Seed python / numpy / torch and return what was actually applied.

    Returns a dict describing which libraries were seeded and whether full
    determinism was achievable, so the caller can write it into the run ledger
    instead of assuming.
    """
    applied: dict[str, object] = {"seed": seed, "python_random": True}
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        applied["numpy"] = True
    except ImportError:
        applied["numpy"] = False

    try:
        import torch

        torch.manual_seed(seed)
        applied["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            applied["torch_cuda"] = True
        else:
            applied["torch_cuda"] = False

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            applied["cudnn_deterministic"] = True
            # torch.use_deterministic_algorithms is intentionally NOT forced:
            # several fused kernels used by the training path have no
            # deterministic implementation and would raise. We record this
            # honestly rather than claiming bit-exact determinism.
            applied["use_deterministic_algorithms"] = False
            applied["determinism_caveat"] = (
                "cuDNN set deterministic; torch.use_deterministic_algorithms "
                "not enabled because some kernels lack deterministic variants. "
                "Greedy decoding is deterministic in practice; training loss "
                "may vary in the last decimal places across runs."
            )
    except ImportError:
        applied["torch"] = False

    return applied
