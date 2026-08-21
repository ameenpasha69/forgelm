"""Response parsing, the failure taxonomy, and metric arithmetic."""

from __future__ import annotations

import math

import pytest

from forgelm import metrics as M
from forgelm.parsing import ERROR_CATEGORIES, classify, evaluate_one, parse_response
from forgelm.schema import CATEGORIES, canonical_json

EXPECTED = {"category": "network", "priority": "high", "affected_service": "vpn",
            "is_security_incident": False, "users_affected": 12}
GOOD_JSON = canonical_json(EXPECTED)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_clean_json_is_strict():
    r = parse_response(GOOD_JSON)
    assert r.strict_json and r.lenient_json
    assert not r.had_fence and not r.had_prose
    assert r.parsed == EXPECTED
    assert r.schema_valid


def test_markdown_fence_is_recovered_but_flagged():
    r = parse_response(f"```json\n{GOOD_JSON}\n```")
    assert r.had_fence
    assert not r.strict_json          # it was not a bare object
    assert r.lenient_json             # but we still recovered it
    assert r.parsed == EXPECTED


def test_prose_around_json_is_recovered_but_flagged():
    r = parse_response(f"Sure! Here is the triage:\n{GOOD_JSON}\nHope that helps.")
    assert r.lenient_json and r.had_prose and not r.strict_json
    assert r.parsed == EXPECTED
    assert classify(r, EXPECTED) == "prose_outside_json"


def test_empty_response():
    assert parse_response("").error_category == "empty_or_refusal"
    assert parse_response("   \n ").error_category == "empty_or_refusal"


def test_unclosed_object_is_truncated():
    r = parse_response('{"category": "network", "priority": "hi')
    assert r.truncated
    assert r.error_category == "truncated"


def test_finish_reason_length_marks_truncation():
    r = parse_response("I think the category is probably", finish_reason="length")
    assert r.error_category == "truncated"


def test_prose_only_is_invalid_json():
    r = parse_response("I cannot help with that request.")
    assert not r.lenient_json
    assert r.error_category == "invalid_json"


def test_json_array_is_not_an_object():
    assert parse_response('[1, 2, 3]').error_category == "not_an_object"


def test_braces_inside_strings_do_not_confuse_the_scanner():
    payload = '{"category": "network", "note": "a } brace", "priority": "high"}'
    r = parse_response(f"prefix {payload} suffix")
    assert r.lenient_json
    assert r.parsed["note"] == "a } brace"


def test_escaped_quote_inside_string():
    payload = '{"category": "network", "note": "say \\"hi\\"", "priority": "low"}'
    r = parse_response(payload)
    assert r.lenient_json and r.parsed["note"] == 'say "hi"'


@pytest.mark.parametrize("bad,expected_category", [
    ('{"category":"networking","priority":"high","affected_service":"vpn",'
     '"is_security_incident":false,"users_affected":12}', "invalid_enum"),
    ('{"category":"network","priority":"high","affected_service":"vpn",'
     '"is_security_incident":false}', "missing_field"),
    ('{"category":"network","priority":"high","affected_service":"vpn",'
     '"is_security_incident":false,"users_affected":12,"confidence":0.9}',
     "extra_field"),
    ('{"category":"network","priority":"high","affected_service":["vpn"],'
     '"is_security_incident":false,"users_affected":12}', "wrong_type"),
    ('{"category":"network","priority":"high","affected_service":"vpn",'
     '"is_security_incident":false,"users_affected":0}', "out_of_range"),
])
def test_failure_taxonomy(bad, expected_category):
    r = parse_response(bad)
    assert classify(r, EXPECTED) == expected_category


def test_correct_answer_classifies_as_correct():
    assert classify(parse_response(GOOD_JSON), EXPECTED) == "correct"


def test_valid_but_wrong_answer():
    wrong = dict(EXPECTED, priority="low")
    assert classify(parse_response(canonical_json(wrong)), EXPECTED) == \
        "wrong_values_only"


def test_every_taxonomy_label_is_declared():
    labels = set()
    for text in (GOOD_JSON, "", "not json", '[1]',
                 f"```json\n{GOOD_JSON}\n```",
                 '{"category":"network"}',
                 '{"a":1,"category":"network","priority":"high",'
                 '"affected_service":"vpn","is_security_incident":false,'
                 '"users_affected":1}'):
        labels.add(classify(parse_response(text), EXPECTED))
    assert labels <= set(ERROR_CATEGORIES)


def test_evaluate_one_field_level_correctness():
    partial = dict(EXPECTED, priority="low", users_affected=99)
    result = evaluate_one(canonical_json(partial), EXPECTED)
    assert result["field_correct"] == {
        "category": True, "priority": False, "affected_service": True,
        "is_security_incident": True, "users_affected": False,
    }
    assert not result["exact_match"]


def test_evaluate_one_exact_match():
    result = evaluate_one(GOOD_JSON, EXPECTED)
    assert result["exact_match"]
    assert all(result["field_correct"].values())
    assert not result["constraint_violation"]


def test_fenced_correct_answer_counts_as_constraint_violation():
    result = evaluate_one(f"```json\n{GOOD_JSON}\n```", EXPECTED)
    assert result["exact_match"]           # values are right
    assert result["constraint_violation"]  # but the format contract was broken


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _record(eid, correct, **overrides):
    base = {
        "example_id": eid,
        "expected": EXPECTED,
        "predicted_fields": dict(EXPECTED) if correct else dict(EXPECTED,
                                                                priority="low"),
        "field_correct": {k: True for k in EXPECTED} if correct
        else {**{k: True for k in EXPECTED}, "priority": False},
        "exact_match": correct,
        "strict_json": True, "lenient_json": True, "schema_valid": True,
        "constraint_violation": False, "had_fence": False, "had_prose": False,
        "truncated": False, "error_category": "correct" if correct
        else "wrong_values_only",
    }
    base.update(overrides)
    return base


def test_macro_f1_perfect():
    y = ["network", "hardware", "email"]
    result = M.macro_f1(y, y, CATEGORIES)
    # Only 3 of 8 categories have support; the other 5 score 0 and are averaged
    # in, which is the intended penalty for a fixed label set.
    assert result["per_label"]["network"]["f1"] == 1.0
    assert result["macro_f1"] == pytest.approx(3 / 8)


def test_macro_f1_penalises_never_predicting_a_class():
    y_true = ["network", "network", "hardware"]
    y_pred = ["network", "network", "network"]
    result = M.macro_f1(y_true, y_pred, ["network", "hardware"])
    assert result["per_label"]["hardware"]["f1"] == 0.0
    assert result["per_label"]["hardware"]["support"] == 1
    assert result["macro_f1"] < 1.0


def test_macro_f1_handles_invalid_predictions():
    y_true = ["network", "hardware"]
    y_pred = ["network", None]
    result = M.macro_f1(y_true, y_pred, ["network", "hardware"])
    assert result["per_label"]["hardware"]["f1"] == 0.0


def test_confusion_buckets_out_of_vocabulary():
    matrix = M.confusion(["network"], ["banana"], ["network", "hardware"])
    assert matrix["network"]["<invalid>"] == 1


def test_compute_metrics_rates():
    records = [_record(f"e{i}", i < 3) for i in range(10)]
    m = M.compute_metrics(records)
    assert m["n"] == 10
    assert m["exact_match"] == 0.3
    assert m["field_accuracy"]["priority"] == 0.3
    assert m["field_accuracy"]["category"] == 1.0
    assert m["error_categories"] == {"correct": 3, "wrong_values_only": 7}


def test_bootstrap_ci_brackets_the_mean():
    values = [1.0] * 30 + [0.0] * 70
    ci = M.bootstrap_ci(values, n_resamples=2000)
    assert ci["mean"] == 0.3
    assert ci["lo"] < 0.3 < ci["hi"]
    assert 0.0 <= ci["lo"] and ci["hi"] <= 1.0


def test_bootstrap_ci_is_deterministic():
    values = [1.0, 0.0] * 25
    assert M.bootstrap_ci(values, n_resamples=500) == \
        M.bootstrap_ci(values, n_resamples=500)


def test_paired_bootstrap_detects_a_real_difference():
    a = [0.0] * 100
    b = [1.0] * 100
    diff = M.paired_bootstrap_diff(a, b, n_resamples=2000)
    assert diff["diff"] == 1.0
    assert diff["excludes_zero"]


def test_paired_bootstrap_on_identical_systems():
    a = b = [1.0, 0.0] * 40
    diff = M.paired_bootstrap_diff(a, b, n_resamples=2000)
    assert diff["diff"] == 0.0
    assert not diff["excludes_zero"]


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        M.paired_bootstrap_diff([1.0], [1.0, 0.0])


def test_mcnemar_counts_and_significance():
    a = [True] * 5 + [False] * 45
    b = [True] * 40 + [False] * 10
    result = M.mcnemar(a, b)
    assert result["both_correct"] == 5
    assert result["only_b_correct"] == 35
    assert result["only_a_correct"] == 0
    assert result["p_value"] < 0.001


def test_mcnemar_no_discordant_pairs():
    a = b = [True, False, True]
    result = M.mcnemar(a, b)
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_symmetric_case_is_not_significant():
    a = [True] * 5 + [False] * 5
    b = [False] * 5 + [True] * 5
    assert M.mcnemar(a, b)["p_value"] == pytest.approx(1.0)


def test_compare_requires_identical_example_sets():
    a = [_record("e1", True)]
    b = [_record("e2", True)]
    with pytest.raises(ValueError, match="shared example ids"):
        M.compare(a, b, "a", "b")


def test_compare_end_to_end():
    a = [_record(f"e{i}", False) for i in range(20)]
    b = [_record(f"e{i}", True) for i in range(20)]
    result = M.compare(a, b, "base", "lora")
    assert result["a_rate"] == 0.0 and result["b_rate"] == 1.0
    assert result["paired_diff"]["excludes_zero"]
    assert result["mcnemar"]["p_value"] < 0.001
    assert result["n_examples"] == 20
