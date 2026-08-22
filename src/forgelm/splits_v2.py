"""v2 splitting and the seal that keeps the v2 test set honest.

Kept in its own module rather than extending `splits.py`, so that nothing in the
v1 split path can change. v1 evidence must keep reproducing byte-for-byte.

The v2 test set is *sealed*: it exists to be looked at once, at the end. The
enforcement is not a convention in a README -- `assert_not_sealed` raises, and
training and demonstration-selection paths call it, so reading a sealed example
is an error rather than a mistake somebody has to notice.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .seeding import rng

FAMILIES_PER_CATEGORY_V2 = 4

# Indexed by severity rank within a category (0 = lowest base_severity).
# 1 train / 1 validation / 2 test per category. The test arm deliberately takes
# ranks 1 and 3 so it spans mid and high severity rather than clustering.
SPLIT_PATTERN_V2: tuple[str, ...] = ("train", "test", "validation", "test")

SPLIT_NAMES_V2 = ("train", "validation", "test")

# Which split is sealed. Named rather than hard-coded at every call site so
# that "what is sealed" is a single, greppable fact.
SEALED_SPLIT_DEFAULT = "test"


def assign_families_v2(records: list[dict[str, Any]]) -> dict[str, str]:
    """Group-aware, category- and severity-stratified family assignment."""
    by_category: dict[str, dict[str, int]] = defaultdict(dict)
    for record in records:
        by_category[record["category"]][record["scenario_family"]] = \
            record["base_severity"]

    assignment: dict[str, str] = {}
    for category in sorted(by_category):
        fam_sev = by_category[category]
        if len(fam_sev) != FAMILIES_PER_CATEGORY_V2:
            raise ValueError(
                f"category {category!r} has {len(fam_sev)} v2 families, "
                f"expected {FAMILIES_PER_CATEGORY_V2}")
        r = rng("split_assignment", "v2", category)
        families = sorted(fam_sev)
        r.shuffle(families)
        families.sort(key=lambda fid: fam_sev[fid])
        for fid, split_name in zip(families, SPLIT_PATTERN_V2):
            assignment[fid] = split_name
    return assignment


def sealed_membership_checksum(example_split: dict[str, str]) -> str:
    """Checksum over the sealed test membership only.

    Narrow on purpose: it answers exactly one question -- "is this the same set
    of sealed examples?" -- so unrelated additions to the manifest cannot
    invalidate it, and a quiet change to what is sealed cannot hide.
    """
    sealed = sorted(eid for eid, split in example_split.items()
                    if split == "test")
    payload = json.dumps(sealed, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest_v2(records: list[dict[str, Any]]) -> dict[str, Any]:
    family_split = assign_families_v2(records)
    example_split = {r["example_id"]: family_split[r["scenario_family"]]
                     for r in records}

    counts = {name: 0 for name in SPLIT_NAMES_V2}
    per_split_category: dict[str, dict[str, int]] = {
        name: defaultdict(int) for name in SPLIT_NAMES_V2}
    per_split_priority: dict[str, dict[str, int]] = {
        name: defaultdict(int) for name in SPLIT_NAMES_V2}
    for record in records:
        split = example_split[record["example_id"]]
        counts[split] += 1
        per_split_category[split][record["category"]] += 1
        per_split_priority[split][record["expected_output"]["priority"]] += 1

    manifest = {
        "dataset": "forgelm-ticket-triage-v2",
        "split_pattern": list(SPLIT_PATTERN_V2),
        "grouping_unit": "scenario_family",
        "stratified_by": "category + base_severity_rank",
        "seed_name": "split_assignment/v2",
        "family_split": dict(sorted(family_split.items())),
        "example_split": dict(sorted(example_split.items())),
        "counts": counts,
        "counts_by_category": {n: dict(sorted(per_split_category[n].items()))
                               for n in SPLIT_NAMES_V2},
        "counts_by_priority": {n: dict(sorted(per_split_priority[n].items()))
                               for n in SPLIT_NAMES_V2},
        "sealed_split": SEALED_SPLIT_DEFAULT,
    }
    manifest["checksum"] = _manifest_checksum(manifest)
    manifest["test_membership_checksum"] = sealed_membership_checksum(example_split)
    return manifest


def _manifest_checksum(manifest: dict[str, Any]) -> str:
    payload = json.dumps({"family_split": manifest["family_split"],
                          "example_split": manifest["example_split"]},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest_v2(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    recomputed = _manifest_checksum(manifest)
    if recomputed != manifest.get("checksum"):
        raise ValueError(
            f"v2 split manifest checksum mismatch: stored="
            f"{manifest.get('checksum')} recomputed={recomputed}")
    sealed = sealed_membership_checksum(manifest["example_split"])
    if sealed != manifest.get("test_membership_checksum"):
        raise ValueError(
            f"v2 SEALED TEST MEMBERSHIP HAS CHANGED: stored="
            f"{manifest.get('test_membership_checksum')} recomputed={sealed}. "
            f"Any result measured against the previous sealed set is no longer "
            f"comparable.")
    return manifest


def apply_split_v2(records: list[dict[str, Any]],
                   manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    example_split = manifest["example_split"]
    out: dict[str, list[dict[str, Any]]] = {n: [] for n in SPLIT_NAMES_V2}
    missing = []
    for record in records:
        split = example_split.get(record["example_id"])
        if split is None:
            missing.append(record["example_id"])
            continue
        out[split].append(record)
    if missing:
        raise ValueError(f"{len(missing)} v2 example(s) absent from the "
                         f"manifest, first few: {missing[:5]}")
    return out


# --------------------------------------------------------------------------
# The seal
# --------------------------------------------------------------------------

class SealedTestAccessError(RuntimeError):
    """Raised when code that must not see the sealed test set touches it."""


def sealed_example_ids(manifest: dict[str, Any]) -> frozenset[str]:
    return frozenset(eid for eid, split in manifest["example_split"].items()
                     if split == manifest.get("sealed_split", SEALED_SPLIT_DEFAULT))


def assert_not_sealed(example_ids: Iterable[str], manifest: dict[str, Any],
                      context: str) -> None:
    """Refuse to proceed if any id belongs to the sealed v2 test set.

    Called from training and demonstration-selection paths. The point is that a
    leak becomes an exception at the moment it happens, rather than a number
    somebody has to be suspicious of afterwards.
    """
    sealed = sealed_example_ids(manifest)
    offending = sorted(set(example_ids) & sealed)
    if offending:
        raise SealedTestAccessError(
            f"{context} attempted to read {len(offending)} sealed v2 test "
            f"example(s), e.g. {offending[:5]}. The v2 test split is sealed "
            f"until final evaluation; training on it or drawing "
            f"demonstrations from it would invalidate every v2 result."
        )
