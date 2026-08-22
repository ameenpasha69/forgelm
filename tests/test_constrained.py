"""Constrained decoding: the prefix automaton and the tokenizer-level mask.

The automaton is the load-bearing part. If it wrongly accepts, the constraint
guarantees nothing; if it wrongly rejects, generation dead-ends and the model is
handicapped. Both directions are tested.
"""

from __future__ import annotations

import pytest

from forgelm import constrained as C
from forgelm.schema import (
    AFFECTED_SERVICES, CATEGORIES, MAX_USERS_AFFECTED, PRIORITIES,
    canonical_json,
)

LEGAL = ('{"category": "network", "priority": "high", '
         '"affected_service": "vpn", "is_security_incident": false, '
         '"users_affected": 34}')


# --------------------------------------------------------------------------
# The template must match canonical_json exactly
# --------------------------------------------------------------------------

def test_literals_reconstruct_canonical_json():
    """If canonical_json's separators ever change, this must fail loudly."""
    obj = {"category": "network", "priority": "high",
           "affected_service": "vpn", "is_security_incident": False,
           "users_affected": 34}
    rebuilt = (C.LITERALS[0] + "network" + C.LITERALS[1] + "high"
               + C.LITERALS[2] + "vpn" + C.LITERALS[3] + "false"
               + C.LITERALS[4] + "34" + C.LITERALS[5])
    assert rebuilt == canonical_json(obj)


def test_no_enum_value_is_a_prefix_of_another():
    """The slot matcher takes the longest match and then commits.

    That is only safe while no permitted value is a strict prefix of another.
    If a future schema breaks this, the matcher needs backtracking.
    """
    for options in (CATEGORIES, PRIORITIES, AFFECTED_SERVICES, C.BOOLEANS):
        pairs = [(a, b) for a in options for b in options
                 if a != b and b.startswith(a)]
        assert pairs == [], f"strict-prefix pairs would break the matcher: {pairs}"


# --------------------------------------------------------------------------
# Acceptance
# --------------------------------------------------------------------------

def test_every_real_expected_output_is_complete(records):
    for record in records:
        text = canonical_json(record["expected_output"])
        assert C.is_complete(text), text


def test_every_prefix_of_every_real_output_is_valid(records):
    """Rejecting a legal prefix would dead-end generation mid-answer."""
    for record in records:
        text = canonical_json(record["expected_output"])
        for i in range(len(text) + 1):
            assert C.is_valid_prefix(text[:i]), \
                f"rejected legal prefix {text[:i]!r} of {text!r}"


def test_complete_output_is_also_a_valid_prefix():
    assert C.is_valid_prefix(LEGAL)
    assert C.is_complete(LEGAL)


def test_empty_string_is_a_valid_prefix():
    assert C.is_valid_prefix("")
    assert not C.is_complete("")


@pytest.mark.parametrize("value", [1, 2, 9, 10, 99, 100, MAX_USERS_AFFECTED])
def test_integer_slot_accepts_legal_values(value):
    obj = {"category": "network", "priority": "high",
           "affected_service": "vpn", "is_security_incident": False,
           "users_affected": value}
    assert C.is_complete(canonical_json(obj))


# --------------------------------------------------------------------------
# Rejection -- including the exact failure modes v1 exhibited
# --------------------------------------------------------------------------

_HEAD = ('{"category": "network", "priority": "high", '
         '"affected_service": ')
_TAIL_HEAD = ('{"category": "network", "priority": "high", '
              '"affected_service": "vpn", "is_security_incident": ')


@pytest.mark.parametrize("bad,reason", [
    (_HEAD + '"dns', "invented enum value observed in v1"),
    (_HEAD + '"internet', "invented enum value observed in v1"),
    (_HEAD + '"download', "invented enum value observed in v1"),
    ('{"category": "display', "invented category observed in v1"),
    ('{"category": "audio', "invented category observed in v1"),
    ('{"category": "file_system', "invented category observed in v1"),
    ('{"priority"', "wrong key order"),
    ('{"Category": "network"', "wrong key casing"),
    ("```json", "markdown fence"),
    ("Sure! Here is the triage:", "prose before the object"),
    ('[{"category"', "array instead of object"),
    (_TAIL_HEAD + '"false', "boolean emitted as a string"),
    (_TAIL_HEAD + "0", "boolean emitted as an int"),
])
def test_rejects_illegal_prefix(bad, reason):
    assert not C.is_valid_prefix(bad), f"should have rejected ({reason}): {bad!r}"


@pytest.mark.parametrize("digits", ["0", "01", "007", "99999", "5001"])
def test_integer_slot_rejects_illegal_values(digits):
    bad = (_TAIL_HEAD + "false" + C.LITERALS[4] + digits)
    assert not C.is_valid_prefix(bad), digits


def test_rejects_extra_field():
    bad = LEGAL[:-1] + ', "confidence": 0.9}'
    assert not C.is_valid_prefix(bad)
    assert not C.is_complete(bad)


def test_rejects_missing_field():
    bad = ('{"category": "network", "priority": "high", '
           '"affected_service": "vpn", "users_affected": 3}')
    assert not C.is_valid_prefix(bad)


def test_rejects_trailing_content_after_close():
    assert not C.is_valid_prefix(LEGAL + " extra")
    assert not C.is_valid_prefix(LEGAL + "\n```")


def test_rejects_duplicated_object():
    assert not C.is_valid_prefix(LEGAL + LEGAL)


# --------------------------------------------------------------------------
# Tokenizer-level masking
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_candidate_set_is_a_useful_subset(tokenizer):
    ids = C.candidate_token_ids(tokenizer)
    assert 0 < len(ids) < len(tokenizer) / 2, (
        f"candidate set of {len(ids)} out of {len(tokenizer)} is not "
        f"restricting anything useful")


@pytest.mark.slow
def test_constraint_allows_a_legal_first_token(tokenizer):
    constraint = C.SchemaConstraint(tokenizer)
    allowed, finished = constraint.mask_for("")
    assert not finished
    assert allowed, "no legal first token"
    # Every allowed token must keep the string legal.
    for token_id in allowed[:200]:
        text = constraint._token_text[token_id]
        assert C.is_valid_prefix(text), text


@pytest.mark.slow
def test_constraint_forces_stop_once_complete(tokenizer):
    constraint = C.SchemaConstraint(tokenizer)
    allowed, finished = constraint.mask_for(LEGAL)
    assert finished
    assert allowed == [tokenizer.eos_token_id]


@pytest.mark.slow
def test_constraint_never_allows_an_illegal_enum_token(tokenizer):
    """The decisive test: at the affected_service slot, no token may start
    spelling a value outside the enum."""
    constraint = C.SchemaConstraint(tokenizer)
    prefix = _HEAD + '"'
    allowed, _ = constraint.mask_for(prefix)
    for token_id in allowed:
        text = constraint._token_text[token_id]
        extended = prefix + text
        assert C.is_valid_prefix(extended), extended
        # Nothing may begin an illegal value such as "dns" or "internet".
        value_so_far = extended[len(_HEAD) + 1:]
        assert any(option.startswith(value_so_far) or
                   value_so_far.startswith(option)
                   for option in AFFECTED_SERVICES), value_so_far


@pytest.mark.slow
def test_constraint_cache_is_effective(tokenizer):
    constraint = C.SchemaConstraint(tokenizer)
    for _ in range(5):
        constraint.mask_for("")
    assert constraint.stats["cache_hits"] >= 4
