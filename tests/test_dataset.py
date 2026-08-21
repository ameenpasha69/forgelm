"""Dataset schema, generation determinism, quality control and leakage."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from forgelm import validate
from forgelm.datagen import (
    FAMILIES, TARGET_TOTAL, TEMPLATE_FAMILIES, compute_priority,
    examples_per_family, generate_dataset,
)
from forgelm.schema import (
    AFFECTED_SERVICES, CATEGORIES, FIELD_ORDER, PRIORITIES, canonical_json,
    is_valid, outputs_equal, validate_output,
)


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------

def _good() -> dict:
    return {"category": "network", "priority": "high", "affected_service": "vpn",
            "is_security_incident": False, "users_affected": 12}


def test_valid_output_passes():
    assert validate_output(_good()) == []
    assert is_valid(_good())


@pytest.mark.parametrize("mutation,expected_prefix", [
    ({"category": "networking"}, "invalid_enum:category"),
    ({"priority": "urgent"}, "invalid_enum:priority"),
    ({"affected_service": "slack"}, "invalid_enum:affected_service"),
    ({"users_affected": 0}, "out_of_range:users_affected"),
    ({"users_affected": -3}, "out_of_range:users_affected"),
    ({"users_affected": "12"}, "wrong_type:users_affected"),
    ({"users_affected": 1.5}, "wrong_type:users_affected"),
    ({"is_security_incident": "false"}, "wrong_type:is_security_incident"),
    ({"is_security_incident": 0}, "wrong_type:is_security_incident"),
])
def test_invalid_values_are_rejected(mutation, expected_prefix):
    obj = {**_good(), **mutation}
    violations = validate_output(obj)
    assert any(v.startswith(expected_prefix.split(":")[0]) for v in violations), \
        f"{mutation} should have produced {expected_prefix}, got {violations}"


def test_missing_and_extra_fields_detected():
    obj = _good()
    del obj["priority"]
    obj["confidence"] = 0.9
    violations = validate_output(obj)
    assert "missing_field:priority" in violations
    assert "extra_field:confidence" in violations


def test_non_object_rejected():
    assert validate_output([1, 2, 3]) == ["not_an_object"]
    assert validate_output("hello") == ["not_an_object"]


def test_bool_is_not_int():
    """True == 1 in Python; the schema must not accept 1 for a boolean."""
    obj = {**_good(), "is_security_incident": 1}
    assert "wrong_type:is_security_incident" in validate_output(obj)


def test_outputs_equal_is_type_sensitive():
    a = {**_good(), "is_security_incident": False}
    b = {**_good(), "is_security_incident": 0}
    assert not outputs_equal(a, b)
    assert outputs_equal(a, dict(a))


def test_canonical_json_key_order_is_fixed():
    shuffled = {k: _good()[k] for k in reversed(FIELD_ORDER)}
    assert canonical_json(shuffled) == canonical_json(_good())
    assert json.loads(canonical_json(_good())) == _good()
    assert list(json.loads(canonical_json(_good())).keys()) == list(FIELD_ORDER)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def test_generation_is_deterministic():
    a = generate_dataset()
    b = generate_dataset()
    assert a == b, "generate_dataset() is not reproducible across calls"


def test_dataset_size_and_shape(records):
    assert len(records) == TARGET_TOTAL == 300
    assert len({r["example_id"] for r in records}) == len(records)
    assert sum(examples_per_family(i) for i in range(len(FAMILIES))) == TARGET_TOTAL


def test_every_family_and_template_used(records):
    assert {r["scenario_family"] for r in records} == {f.fid for f in FAMILIES}
    assert {r["template_family"] for r in records} == set(TEMPLATE_FAMILIES)


def test_family_ids_are_unique():
    fids = [f.fid for f in FAMILIES]
    assert len(fids) == len(set(fids))


def test_categories_are_balanced_by_family():
    counts = Counter(f.category for f in FAMILIES)
    assert set(counts) == set(CATEGORIES)
    assert set(counts.values()) == {7}, f"expected 7 families per category: {counts}"


def test_every_expected_output_is_schema_valid(records):
    for rec in records:
        assert validate_output(rec["expected_output"]) == [], rec["example_id"]


def test_expected_output_json_matches_object(records):
    for rec in records:
        assert json.loads(rec["expected_output_json"]) == rec["expected_output"]


def test_label_enums_are_in_range(records):
    for rec in records:
        out = rec["expected_output"]
        assert out["category"] in CATEGORIES
        assert out["priority"] in PRIORITIES
        assert out["affected_service"] in AFFECTED_SERVICES


@pytest.mark.parametrize("severity,security,users,expected", [
    (0, False, 1, "low"),
    (0, False, 60, "medium"),
    (1, False, 1, "low"),
    (1, False, 12, "medium"),
    (2, False, 1, "medium"),
    (2, False, 12, "high"),
    (3, False, 1, "high"),
    (3, False, 60, "critical"),
    (3, True, 12, "critical"),
    (1, True, 1, "medium"),
])
def test_priority_rule_table(severity, security, users, expected):
    assert compute_priority(severity, security, users) == expected


def test_priority_labels_follow_the_published_rule(records):
    assert validate.check_priority_rule(records) == []


def test_security_flag_is_not_a_proxy_for_category(records):
    """If is_security_incident were exactly category=='security', the field
    would be free and the metric would be misleading."""
    sec_true = {r["category"] for r in records
                if r["expected_output"]["is_security_incident"]}
    assert sec_true - {"security"}, \
        "no non-security category carries is_security_incident=True"


# --------------------------------------------------------------------------
# Quality control
# --------------------------------------------------------------------------

def test_no_structural_errors(records):
    report = validate.run_all(records, manifest=None)
    errors = [f for f in report["findings"] if f["severity"] == "error"]
    assert errors == [], f"structural QC errors: {errors[:5]}"


def test_no_duplicates(records):
    assert validate.check_duplicates(records) == []


def test_users_affected_is_recoverable_from_text(records):
    assert validate.check_users_recoverable(records) == []


def test_no_priority_terms_leak_into_ticket_text(records):
    assert validate.check_priority_leak_terms(records) == []


def test_single_user_examples_do_not_contradict_themselves(records):
    assert validate.check_scope_consistency(records) == []


def test_duplicate_detector_actually_catches_duplicates(records):
    """A QC check that cannot fail is not a QC check."""
    poisoned = [dict(r) for r in records[:5]]
    poisoned.append({**poisoned[0], "example_id": "dupe-01"})
    findings = validate.check_duplicates(poisoned)
    assert any(f.check == "exact_duplicate" for f in findings)


def test_priority_rule_detector_catches_a_wrong_label(records):
    poisoned = [dict(r) for r in records[:3]]
    poisoned[0] = {**poisoned[0],
                   "expected_output": {**poisoned[0]["expected_output"],
                                       "priority": "critical"
                                       if poisoned[0]["expected_output"]["priority"]
                                       != "critical" else "low"}}
    assert validate.check_priority_rule(poisoned)


def test_leak_term_detector_catches_a_planted_term(records):
    poisoned = [dict(r) for r in records[:3]]
    poisoned[0] = {**poisoned[0],
                   "ticket_text": poisoned[0]["ticket_text"] + " This is critical."}
    assert validate.check_priority_leak_terms(poisoned)


def test_ticket_text_lengths_are_reasonable(records):
    lengths = [len(r["ticket_text"]) for r in records]
    assert min(lengths) >= validate.MIN_CHARS
    assert max(lengths) <= validate.MAX_CHARS


def test_acronyms_are_not_lowercased_by_capitalisation(records):
    """Regression test for the str.capitalize() defect found in manual review."""
    offenders = [r["example_id"] for r in records
                 if r["template_family"] != "chat_style"
                 and (" crm " in r["ticket_text"] or " vpn " in r["ticket_text"]
                      or " mfa " in r["ticket_text"])]
    assert offenders == [], f"acronyms lower-cased in: {offenders[:5]}"
