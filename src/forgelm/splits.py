"""Deterministic, group-aware, category-stratified splitting.

The generalisation axis we care about is *scenario*: can the adapted model
handle a helpdesk situation it has never been trained on? So the grouping unit
is `scenario_family`, and no family may appear in more than one split.

Stratification is by `category`: each of the 8 categories has exactly 7
families, split 4 train / 1 validation / 2 test. That keeps every category
present in every split (so macro-F1 is defined everywhere) while still
guaranteeing disjoint scenarios.

The resulting manifest maps every example_id -> split and is checksummed. Once
written, the test assignment must never be regenerated with different logic;
the test suite asserts the checksum is stable.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .seeding import rng

# Families per category, and how they are apportioned.
FAMILIES_PER_CATEGORY = 7

# Families within a category are ordered by base_severity (ascending) and then
# dealt to splits using this fixed pattern. Position i therefore corresponds to
# severity rank i, so every split receives a spread of severities rather than
# whichever ones a shuffle happened to hand it.
#
# Why this and not a plain shuffle: the first implementation used a seeded
# shuffle per category. It produced a validation split containing zero
# `critical` and one `medium` example, which would have made checkpoint
# selection blind to the high-priority classes. This was observed *before any
# model was run* -- no test-set information influenced the change. See
# DECISIONS.md, D-006.
SPLIT_PATTERN: tuple[str, ...] = (
    "train",       # severity rank 0 (lowest)
    "train",       # rank 1
    "test",        # rank 2
    "train",       # rank 3
    "validation",  # rank 4
    "test",        # rank 5
    "train",       # rank 6 (highest)
)

SPLIT_PLAN: tuple[tuple[str, int], ...] = (
    ("train", SPLIT_PATTERN.count("train")),
    ("validation", SPLIT_PATTERN.count("validation")),
    ("test", SPLIT_PATTERN.count("test")),
)

SPLIT_NAMES = ("train", "validation", "test")


def assign_families(records: list[dict[str, Any]]) -> dict[str, str]:
    """Map scenario_family -> split name.

    Group-aware (unit = scenario_family), stratified by category *and* by
    severity rank. Ties in base_severity are broken by a seeded shuffle so the
    assignment is not an artefact of alphabetical family naming, while the
    severity spread across splits is guaranteed rather than left to chance.
    """
    by_category: dict[str, dict[str, int]] = defaultdict(dict)
    for rec in records:
        by_category[rec["category"]][rec["scenario_family"]] = rec["base_severity"]

    assignment: dict[str, str] = {}
    for category in sorted(by_category):
        fam_sev = by_category[category]
        if len(fam_sev) != FAMILIES_PER_CATEGORY:
            raise ValueError(
                f"category {category!r} has {len(fam_sev)} families, "
                f"expected {FAMILIES_PER_CATEGORY}. The split pattern assumes a "
                f"balanced family count; fix the generator or the pattern."
            )

        # Shuffle first (breaks alphabetical ties reproducibly), then sort by
        # severity. Python's sort is stable, so the shuffled order survives
        # within each severity band.
        r = rng("split_assignment", category)
        families = sorted(fam_sev)
        r.shuffle(families)
        families.sort(key=lambda fid: fam_sev[fid])

        for fid, split_name in zip(families, SPLIT_PATTERN):
            assignment[fid] = split_name

    return assignment


def build_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce the full, checksummed split manifest."""
    family_split = assign_families(records)

    example_split: dict[str, str] = {}
    for rec in records:
        example_split[rec["example_id"]] = family_split[rec["scenario_family"]]

    counts: dict[str, int] = {name: 0 for name in SPLIT_NAMES}
    per_split_category: dict[str, dict[str, int]] = {
        name: defaultdict(int) for name in SPLIT_NAMES
    }
    for rec in records:
        split = example_split[rec["example_id"]]
        counts[split] += 1
        per_split_category[split][rec["category"]] += 1

    manifest = {
        "split_plan": {name: count for name, count in SPLIT_PLAN},
        "split_pattern": list(SPLIT_PATTERN),
        "grouping_unit": "scenario_family",
        "stratified_by": "category + base_severity_rank",
        "seed_name": "split_assignment",
        "family_split": dict(sorted(family_split.items())),
        "example_split": dict(sorted(example_split.items())),
        "counts": counts,
        "counts_by_category": {
            name: dict(sorted(per_split_category[name].items()))
            for name in SPLIT_NAMES
        },
    }
    manifest["checksum"] = manifest_checksum(manifest)
    return manifest


def manifest_checksum(manifest: dict[str, Any]) -> str:
    """Checksum over the assignment only -- not over counts or metadata.

    Deliberately narrow: it answers "did any example move between splits?"
    and nothing else, so cosmetic additions to the manifest do not invalidate
    it.
    """
    payload = json.dumps(
        {
            "family_split": manifest["family_split"],
            "example_split": manifest["example_split"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(records: list[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(records)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    recomputed = manifest_checksum(manifest)
    if recomputed != manifest.get("checksum"):
        raise ValueError(
            f"split manifest checksum mismatch: stored={manifest.get('checksum')} "
            f"recomputed={recomputed}. The split has been altered since it was "
            f"frozen; results computed against it are not comparable."
        )
    return manifest


def subsample_stratified(records: list[dict[str, Any]], fraction: float,
                         seed_name: str = "training",
                         seed_context: str = "subsample",
                         stratify_by: str = "category") -> list[dict[str, Any]]:
    """Deterministically take a stratified fraction of a split.

    Used by the training-data-size ablation. Stratifying by category keeps the
    label distribution comparable between the two arms, so the only variable
    that changes is how many examples the model saw.

    Note the limitation, which the ablation report states explicitly: because
    stratification is by *category* rather than by *scenario_family*, a small
    family can lose all of its examples by chance. That makes the manipulation
    "mostly depth" rather than "purely depth". `subsample_coverage` reports
    exactly what was lost so the effect is measured, not assumed.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda r: r["example_id"]):
        buckets[record[stratify_by]].append(record)

    kept: list[dict[str, Any]] = []
    for key in sorted(buckets):
        pool = list(buckets[key])
        rng(seed_name, seed_context, key).shuffle(pool)
        take = max(1, round(len(pool) * fraction))
        kept.extend(pool[:take])

    kept.sort(key=lambda r: r["example_id"])
    return kept


def subsample_coverage(original: list[dict[str, Any]],
                       kept: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe what a subsample removed: depth, coverage, or both."""
    families_before = {r["scenario_family"] for r in original}
    families_after = {r["scenario_family"] for r in kept}
    dropped = sorted(families_before - families_after)
    return {
        "n_before": len(original),
        "n_after": len(kept),
        "families_before": len(families_before),
        "families_after": len(families_after),
        "families_dropped": dropped,
        "examples_per_family_before": round(
            len(original) / len(families_before), 3) if families_before else None,
        "examples_per_family_after": round(
            len(kept) / len(families_after), 3) if families_after else None,
    }


def apply_split(records: list[dict[str, Any]],
                manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Partition records using a frozen manifest."""
    example_split = manifest["example_split"]
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    missing = []
    for rec in records:
        split = example_split.get(rec["example_id"])
        if split is None:
            missing.append(rec["example_id"])
            continue
        out[split].append(rec)
    if missing:
        raise ValueError(
            f"{len(missing)} example(s) absent from the split manifest, "
            f"first few: {missing[:5]}. The dataset and manifest are out of "
            f"sync -- regenerate the manifest or restore the dataset."
        )
    return out
