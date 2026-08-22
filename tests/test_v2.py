"""v2: the new catalogue, the split, the seal, and the diagnostic suites.

The seal tests matter most. A sealed test set enforced only by convention is
enforced by whoever remembers at the moment it matters; these assert that the
code refuses.
"""

from __future__ import annotations

from collections import Counter

import pytest

from forgelm import diagnostics as D
from forgelm import validate
from forgelm.datagen_v2 import (
    EXAMPLES_PER_FAMILY_V2, FAMILIES_V2, TARGET_TOTAL_V2, generate_dataset_v2,
)
from forgelm.schema import CATEGORIES, PRIORITIES, validate_output
from forgelm.splits_v2 import (
    FAMILIES_PER_CATEGORY_V2, SEALED_SPLIT_DEFAULT, SPLIT_PATTERN_V2,
    SealedTestAccessError, apply_split_v2, assert_not_sealed, build_manifest_v2,
    sealed_example_ids, test_membership_checksum,
)


@pytest.fixture(scope="module")
def v2_records():
    return generate_dataset_v2()


@pytest.fixture(scope="module")
def v2_manifest(v2_records):
    return build_manifest_v2(v2_records)


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------

def test_generation_is_deterministic():
    assert generate_dataset_v2() == generate_dataset_v2()


def test_size_and_shape(v2_records):
    assert len(v2_records) == TARGET_TOTAL_V2 == 192
    assert len({r["example_id"] for r in v2_records}) == len(v2_records)
    assert len(FAMILIES_V2) == 32


def test_four_families_per_category():
    counts = Counter(f.category for f in FAMILIES_V2)
    assert set(counts) == set(CATEGORIES)
    assert set(counts.values()) == {FAMILIES_PER_CATEGORY_V2}


def test_family_ids_unique_and_v2_prefixed():
    ids = [f.fid for f in FAMILIES_V2]
    assert len(ids) == len(set(ids))
    assert all(fid.startswith("v2_") for fid in ids)


def test_no_family_name_collides_with_v1():
    from forgelm.datagen import FAMILIES as V1_FAMILIES

    overlap = {f.fid for f in FAMILIES_V2} & {f.fid for f in V1_FAMILIES}
    assert overlap == set(), f"v2 reuses v1 family ids: {overlap}"


def test_every_expected_output_is_schema_valid(v2_records):
    for record in v2_records:
        assert validate_output(record["expected_output"]) == [], \
            record["example_id"]


def test_quality_control_passes(v2_records):
    report = validate.run_all(v2_records, manifest=None)
    errors = [f for f in report["findings"] if f["severity"] == "error"]
    assert errors == [], errors[:3]


def test_priority_balance_beats_v1(v2_records):
    """The whole point of the new catalogue's severity/scale choices."""
    counts = Counter(r["expected_output"]["priority"] for r in v2_records)
    assert set(counts) == set(PRIORITIES)
    medium_share = counts["medium"] / len(v2_records)
    assert medium_share > 0.20, (
        f"medium is {medium_share:.1%} of v2; the balance fix did not take")


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def test_split_is_deterministic(v2_records):
    assert build_manifest_v2(v2_records)["checksum"] == \
        build_manifest_v2(v2_records)["checksum"]


def test_no_family_spans_splits(v2_records, v2_manifest):
    from collections import defaultdict

    spans = defaultdict(set)
    for record in v2_records:
        spans[record["scenario_family"]].add(
            v2_manifest["example_split"][record["example_id"]])
    straddling = {f: s for f, s in spans.items() if len(s) > 1}
    assert straddling == {}


def test_split_counts(v2_manifest):
    assert v2_manifest["counts"] == {"train": 48, "validation": 48, "test": 96}


def test_split_pattern_shape():
    assert len(SPLIT_PATTERN_V2) == FAMILIES_PER_CATEGORY_V2
    assert Counter(SPLIT_PATTERN_V2) == {"train": 1, "validation": 1, "test": 2}


def test_sealed_split_has_all_priorities(v2_records, v2_manifest):
    by_split = apply_split_v2(v2_records, v2_manifest)
    present = {r["expected_output"]["priority"] for r in by_split["test"]}
    assert present == set(PRIORITIES)


def test_sealed_split_has_more_medium_than_v1_did(v2_records, v2_manifest):
    by_split = apply_split_v2(v2_records, v2_manifest)
    medium = sum(1 for r in by_split["test"]
                 if r["expected_output"]["priority"] == "medium")
    assert medium > 3, (
        f"v1's test split had 3 medium examples; v2 has {medium}, which is no "
        f"improvement")


def test_no_cross_split_leakage(v2_records, v2_manifest):
    findings = validate.check_cross_split_leakage(v2_records, v2_manifest)
    errors = [f for f in findings if f.severity == "error"]
    assert errors == [], [f.message for f in errors]


# --------------------------------------------------------------------------
# The seal
# --------------------------------------------------------------------------

def test_seal_checksum_is_stable(v2_records, v2_manifest):
    assert test_membership_checksum(v2_manifest["example_split"]) == \
        v2_manifest["test_membership_checksum"]


def test_seal_checksum_changes_if_membership_changes(v2_manifest):
    tampered = dict(v2_manifest["example_split"])
    a_test_id = next(e for e, s in tampered.items() if s == "test")
    tampered[a_test_id] = "train"
    assert test_membership_checksum(tampered) != \
        v2_manifest["test_membership_checksum"]


def test_seal_checksum_ignores_unrelated_reordering(v2_manifest):
    """Narrow on purpose: it answers only 'is this the same sealed set?'."""
    shuffled = dict(reversed(list(v2_manifest["example_split"].items())))
    assert test_membership_checksum(shuffled) == \
        v2_manifest["test_membership_checksum"]


def test_assert_not_sealed_allows_training_examples(v2_records, v2_manifest):
    train_ids = [e for e, s in v2_manifest["example_split"].items()
                 if s == "train"]
    assert_not_sealed(train_ids, v2_manifest, "unit test")   # must not raise


def test_assert_not_sealed_raises_on_a_sealed_example(v2_manifest):
    sealed = sorted(sealed_example_ids(v2_manifest))
    with pytest.raises(SealedTestAccessError, match="sealed"):
        assert_not_sealed(sealed[:1], v2_manifest, "unit test")


def test_assert_not_sealed_raises_even_for_one_leaked_id(v2_manifest):
    """A single leaked example is still a leak."""
    sealed = sorted(sealed_example_ids(v2_manifest))
    train = [e for e, s in v2_manifest["example_split"].items() if s == "train"]
    with pytest.raises(SealedTestAccessError):
        assert_not_sealed(train + sealed[:1], v2_manifest, "unit test")


def test_sealed_ids_match_the_test_split(v2_records, v2_manifest):
    by_split = apply_split_v2(v2_records, v2_manifest)
    assert sealed_example_ids(v2_manifest) == \
        frozenset(r["example_id"] for r in by_split["test"])


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def suites(v2_records, v2_manifest):
    by_split = apply_split_v2(v2_records, v2_manifest)
    base = sorted(by_split["train"] + by_split["validation"],
                  key=lambda r: r["example_id"])
    return D.build_suites(base), base


def test_diagnostics_never_touch_the_sealed_split(suites, v2_manifest):
    built, _ = suites
    sealed = sealed_example_ids(v2_manifest)
    for name, rows in built.items():
        sources = {r["source_example_id"] for r in rows
                   if r.get("source_example_id")}
        assert not (sources & sealed), \
            f"suite {name} was built from sealed examples"


def test_every_suite_declares_a_scoring_mode(suites):
    built, _ = suites
    for name in D.SUITE_ORDER:
        assert name in built
        for row in built[name]:
            assert row["scoring_mode"] == D.SCORING_MODE[name]


def test_label_preserving_suites_keep_the_label(suites):
    built, base = suites
    by_id = {r["example_id"]: r for r in base}
    for name in ("noisy_text", "irrelevant_detail", "long_tickets"):
        for row in built[name]:
            original = by_id[row["source_example_id"]]
            assert row["expected_output"] == original["expected_output"]


def test_noisy_text_actually_changes_the_text(suites):
    built, base = suites
    by_id = {r["example_id"]: r for r in base}
    changed = sum(1 for r in built["noisy_text"]
                  if r["ticket_text"] != by_id[r["source_example_id"]]["ticket_text"])
    assert changed == len(built["noisy_text"])


def test_long_tickets_are_substantially_longer(suites):
    built, base = suites
    by_id = {r["example_id"]: r for r in base}
    for row in built["long_tickets"]:
        original = by_id[row["source_example_id"]]["ticket_text"]
        assert len(row["ticket_text"]) > 3 * len(original)


def test_missing_user_count_removes_the_number(suites):
    """If the count survived, the suite would not be testing anything."""
    built, base = suites
    by_id = {r["example_id"]: r for r in base}
    assert built["missing_user_count"], "suite is empty"
    for row in built["missing_user_count"]:
        original = by_id[row["source_example_id"]]
        n = original["expected_output"]["users_affected"]
        if n > 1:
            assert str(n) not in row["ticket_text"], row["ticket_text"]
        assert row["scoring_mode"] == "except_users"
        assert "users_affected" in row["unknowable_fields"]


def test_contradictory_suite_states_two_different_counts(suites):
    import re

    built, _ = suites
    assert built["contradictory"], "suite is empty"
    for row in built["contradictory"]:
        numbers = set(re.findall(r"\b(\d+) (?:people|users)\b",
                                 row["ticket_text"]))
        assert len(numbers) >= 2, row["ticket_text"]


def test_out_of_domain_has_no_ground_truth(suites):
    built, _ = suites
    rows = built["out_of_domain"]
    assert rows
    for row in rows:
        assert row["scoring_mode"] == "schema_only"
        assert len(row["unknowable_fields"]) == 5
        assert row["source_example_id"] is None


def test_out_of_domain_includes_an_instruction_override_attempt(suites):
    """Worth probing: does non-ticket input still yield confident triage?"""
    built, _ = suites
    texts = " ".join(r["ticket_text"] for r in built["out_of_domain"]).lower()
    assert "ignore your instructions" in texts


def test_suite_example_ids_are_unique_and_traceable(suites):
    built, _ = suites
    for name, rows in built.items():
        ids = [r["example_id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
        assert all(i.endswith(f"::{name}") for i in ids)
