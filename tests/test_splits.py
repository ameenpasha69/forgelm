"""Split determinism, group-awareness, stratification and leakage prevention."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from forgelm import validate
from forgelm.schema import CATEGORIES, PRIORITIES
from forgelm.splits import (
    FAMILIES_PER_CATEGORY, SPLIT_NAMES, SPLIT_PATTERN, apply_split,
    assign_families, build_manifest, manifest_checksum, subsample_coverage,
    subsample_stratified,
)

# The frozen split. If this value changes, the test set has moved and every
# previously reported number is invalid. Updating it is a deliberate act that
# requires bumping DATASET_VERSION.
FROZEN_CHECKSUM = "994141c6e09e98667f0d57375cfbf552afdb2e8a4ed85c68663191e289846060"


def test_split_is_deterministic(records):
    assert assign_families(records) == assign_families(records)
    assert build_manifest(records)["checksum"] == build_manifest(records)["checksum"]


def test_split_checksum_is_frozen(records):
    """Guards against an accidental reshuffle of the test set."""
    assert build_manifest(records)["checksum"] == FROZEN_CHECKSUM, (
        "the split assignment has changed. If this is intentional, bump "
        "DATASET_VERSION and update FROZEN_CHECKSUM together, and treat all "
        "previously recorded results as belonging to the old split."
    )


def test_manifest_checksum_detects_tampering(manifest):
    tampered = {**manifest, "example_split": dict(manifest["example_split"])}
    first_id = next(iter(tampered["example_split"]))
    current = tampered["example_split"][first_id]
    tampered["example_split"][first_id] = (
        "test" if current != "test" else "train")
    assert manifest_checksum(tampered) != manifest["checksum"]


def test_no_scenario_family_spans_splits(records, manifest):
    fam_splits = defaultdict(set)
    for rec in records:
        fam_splits[rec["scenario_family"]].add(
            manifest["example_split"][rec["example_id"]])
    straddling = {f: s for f, s in fam_splits.items() if len(s) > 1}
    assert straddling == {}, f"families in multiple splits: {straddling}"


def test_every_example_is_assigned_exactly_once(records, manifest):
    assert len(manifest["example_split"]) == len(records)
    assert set(manifest["example_split"]) == {r["example_id"] for r in records}
    assert set(manifest["example_split"].values()) <= set(SPLIT_NAMES)


def test_split_sizes(manifest):
    assert manifest["counts"] == {"train": 171, "validation": 43, "test": 86}
    assert sum(manifest["counts"].values()) == 300


def test_every_category_appears_in_every_split(manifest):
    for split in SPLIT_NAMES:
        present = set(manifest["counts_by_category"][split])
        assert present == set(CATEGORIES), \
            f"{split} is missing categories: {set(CATEGORIES) - present}"


def test_every_priority_appears_in_every_split(records, manifest):
    """The reason the severity-stratified pattern exists.

    A validation split with no `critical` examples cannot inform checkpoint
    selection about the class that matters most.
    """
    by_split = apply_split(records, manifest)
    for split, rows in by_split.items():
        present = {r["expected_output"]["priority"] for r in rows}
        assert present == set(PRIORITIES), \
            f"{split} is missing priorities: {set(PRIORITIES) - present}"


def test_split_pattern_shape():
    assert len(SPLIT_PATTERN) == FAMILIES_PER_CATEGORY
    counts = Counter(SPLIT_PATTERN)
    assert counts == {"train": 4, "test": 2, "validation": 1}


def test_families_per_split(manifest):
    counts = Counter(manifest["family_split"].values())
    assert counts == {"train": 32, "validation": 8, "test": 16}


def test_no_cross_split_near_duplicates(records, manifest):
    findings = validate.check_cross_split_leakage(records, manifest)
    errors = [f for f in findings if f.severity == "error"]
    assert errors == [], f"cross-split leakage: {[f.message for f in errors]}"


def test_cross_split_similarity_has_a_wide_margin(records, manifest):
    """Not just below threshold -- comfortably below it."""
    findings = validate.check_cross_split_leakage(records, manifest)
    info = [f for f in findings if f.check == "cross_split_near_duplicate"][0]
    observed = info.detail["max_similarity"]
    assert observed < 0.6, (
        f"max cross-split similarity {observed} is uncomfortably close to the "
        f"{validate.NEAR_DUP_THRESHOLD} threshold"
    )


def test_apply_split_rejects_an_unknown_example(records, manifest):
    extra = [*records, {**records[0], "example_id": "ghost-99"}]
    with pytest.raises(ValueError, match="absent from the split manifest"):
        apply_split(extra, manifest)


# --------------------------------------------------------------------------
# Training-data-size ablation subsampling
# --------------------------------------------------------------------------

def _train(records, manifest):
    return apply_split(records, manifest)["train"]


def test_subsample_is_deterministic(records, manifest):
    train = _train(records, manifest)
    first = [r["example_id"] for r in subsample_stratified(train, 0.5)]
    second = [r["example_id"] for r in subsample_stratified(train, 0.5)]
    assert first == second


def test_subsample_size_is_about_the_requested_fraction(records, manifest):
    train = _train(records, manifest)
    kept = subsample_stratified(train, 0.5)
    assert abs(len(kept) - len(train) * 0.5) <= len(set(
        r["category"] for r in train))  # at most one rounding unit per stratum


def test_subsample_preserves_category_balance(records, manifest):
    """The ablation must change example count and nothing else it can help."""
    train = _train(records, manifest)
    kept = subsample_stratified(train, 0.5)
    before = Counter(r["category"] for r in train)
    after = Counter(r["category"] for r in kept)
    assert set(before) == set(after), "a category vanished from the subsample"
    for category in before:
        expected = max(1, round(before[category] * 0.5))
        assert after[category] == expected


def test_subsample_is_a_subset(records, manifest):
    train = _train(records, manifest)
    kept = subsample_stratified(train, 0.5)
    train_ids = {r["example_id"] for r in train}
    kept_ids = {r["example_id"] for r in kept}
    assert kept_ids <= train_ids
    assert len(kept_ids) == len(kept), "subsample contains duplicates"


def test_subsample_never_touches_held_out_splits(records, manifest):
    train = _train(records, manifest)
    kept = subsample_stratified(train, 0.5)
    for record in kept:
        assert manifest["example_split"][record["example_id"]] == "train"


def test_subsample_full_fraction_is_identity(records, manifest):
    train = _train(records, manifest)
    kept = subsample_stratified(train, 1.0)
    assert [r["example_id"] for r in kept] == \
        sorted(r["example_id"] for r in train)


def test_subsample_rejects_bad_fractions(records, manifest):
    train = _train(records, manifest)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            subsample_stratified(train, bad)


def test_subsample_coverage_reports_what_was_lost(records, manifest):
    """The ablation's interpretation depends on this being measured.

    Halving the data could mean fewer examples per scenario, or fewer
    scenarios. Those support opposite conclusions, so the report must state
    which happened rather than assume.
    """
    train = _train(records, manifest)
    kept = subsample_stratified(train, 0.5)
    coverage = subsample_coverage(train, kept)

    assert coverage["n_before"] == len(train)
    assert coverage["n_after"] == len(kept)
    assert coverage["families_after"] <= coverage["families_before"]
    assert coverage["examples_per_family_after"] < \
        coverage["examples_per_family_before"], \
        "a depth-reducing subsample must lower examples per family"
    # Whatever the value, it must be internally consistent.
    assert len(coverage["families_dropped"]) == \
        coverage["families_before"] - coverage["families_after"]


def test_leakage_detector_catches_a_planted_family_straddle(records, manifest):
    """Verify the detector fails when it should."""
    tampered = {**manifest, "example_split": dict(manifest["example_split"])}
    train_example = next(r for r in records
                         if manifest["example_split"][r["example_id"]] == "train")
    tampered["example_split"][train_example["example_id"]] = "test"
    findings = validate.check_cross_split_leakage(records, tampered)
    assert any(f.check == "scenario_family_leakage" for f in findings)
