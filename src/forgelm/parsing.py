"""Turning raw model text into a parsed object, and classifying how it failed.

Design rule: **the same parser is applied to every condition.** A lenient
parser applied only to the base model, or a strict one applied only to the
adapted model, would manufacture a result. So `parse_response` is deliberately
somewhat forgiving (it will strip a markdown fence and find the first balanced
JSON object) and both the strict and lenient outcomes are reported:

    json_parse_rate_strict   -- the whole response is exactly one JSON object
    json_parse_rate_lenient  -- a JSON object could be recovered from the text

The gap between them is itself a finding: it measures how much scaffolding the
model wraps around its answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import outputs_equal, validate_output

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


# Failure taxonomy. Ordered: the first matching category wins, so a response
# that is both unparseable and truncated is reported as unparseable.
ERROR_CATEGORIES = (
    "correct",
    "empty_or_refusal",
    "invalid_json",
    "truncated",
    "not_an_object",
    "extra_field",
    "missing_field",
    "invalid_enum",
    "wrong_type",
    "out_of_range",
    "wrong_values_only",
    "prose_outside_json",
)


@dataclass
class ParseResult:
    raw: str
    parsed: dict[str, Any] | None = None
    strict_json: bool = False           # whole response was one JSON object
    lenient_json: bool = False          # an object was recoverable
    had_fence: bool = False
    had_prose: bool = False             # text outside the JSON object
    truncated: bool = False
    schema_violations: list[str] = field(default_factory=list)
    error_category: str = "invalid_json"

    @property
    def schema_valid(self) -> bool:
        return self.lenient_json and not self.schema_violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "parsed": self.parsed,
            "strict_json": self.strict_json,
            "lenient_json": self.lenient_json,
            "had_fence": self.had_fence,
            "had_prose": self.had_prose,
            "truncated": self.truncated,
            "schema_valid": self.schema_valid,
            "schema_violations": self.schema_violations,
            "error_category": self.error_category,
        }


def _find_balanced_object(text: str) -> tuple[str | None, int, int]:
    """Return the first balanced {...} span, respecting strings and escapes."""
    start = text.find("{")
    if start == -1:
        return None, -1, -1
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], start, i + 1
    return None, start, -1  # opened but never closed -> truncated


def parse_response(raw: str, finish_reason: str | None = None) -> ParseResult:
    """Parse one raw generation into a ParseResult.

    `finish_reason` (if the caller knows generation hit the token cap) makes
    truncation detection reliable rather than heuristic.
    """
    result = ParseResult(raw=raw)
    text = raw.strip()

    if not text:
        result.error_category = "empty_or_refusal"
        return result

    fence = _FENCE_RE.match(text)
    if fence:
        result.had_fence = True
        text = fence.group(1).strip()

    # Strict: the entire (de-fenced) response parses as one JSON object.
    try:
        obj = json.loads(text)
        result.strict_json = not result.had_fence
        result.lenient_json = True
        result.parsed = obj if isinstance(obj, dict) else None
        if not isinstance(obj, dict):
            result.error_category = "not_an_object"
            return result
    except json.JSONDecodeError:
        span, start, end = _find_balanced_object(text)
        if span is None:
            if start != -1 and end == -1:
                result.truncated = True
                result.error_category = "truncated"
            else:
                result.error_category = ("truncated" if finish_reason == "length"
                                         else "invalid_json")
            return result
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            result.error_category = ("truncated" if finish_reason == "length"
                                     else "invalid_json")
            return result
        if not isinstance(obj, dict):
            result.error_category = "not_an_object"
            return result
        result.lenient_json = True
        result.parsed = obj
        result.had_prose = bool(text[:start].strip() or text[end:].strip())

    if finish_reason == "length" and not result.lenient_json:
        result.truncated = True
        result.error_category = "truncated"
        return result

    result.schema_violations = validate_output(result.parsed)
    return result


def classify(result: ParseResult, expected: dict[str, Any]) -> str:
    """Assign the single error category used in the failure analysis."""
    if not result.lenient_json:
        return result.error_category

    violations = result.schema_violations
    if violations:
        for prefix, label in (
            ("extra_field", "extra_field"),
            ("missing_field", "missing_field"),
            ("invalid_enum", "invalid_enum"),
            ("wrong_type", "wrong_type"),
            ("out_of_range", "out_of_range"),
            ("not_an_object", "not_an_object"),
        ):
            if any(v.startswith(prefix) for v in violations):
                return label
        return "wrong_type"

    if outputs_equal(result.parsed, expected):
        # A fully correct object still gets flagged if it arrived wrapped in
        # prose or a code fence, because that is a real constraint violation
        # for a machine-consumed endpoint.
        if result.had_prose:
            return "prose_outside_json"
        return "correct"

    return "wrong_values_only"


def evaluate_one(raw: str, expected: dict[str, Any],
                 finish_reason: str | None = None) -> dict[str, Any]:
    """Full per-example record: parse, validate, classify, field-level match."""
    result = parse_response(raw, finish_reason=finish_reason)
    category = classify(result, expected)

    field_correct: dict[str, bool] = {}
    predicted_fields: dict[str, Any] = {}
    for key in expected:
        pred = (result.parsed or {}).get(key, None)
        predicted_fields[key] = pred
        exp = expected[key]
        if isinstance(exp, bool) != isinstance(pred, bool):
            field_correct[key] = False
        else:
            field_correct[key] = pred == exp

    return {
        **result.as_dict(),
        "error_category": category,
        "expected": expected,
        "predicted_fields": predicted_fields,
        "field_correct": field_correct,
        "exact_match": category == "correct" or (
            result.schema_valid and outputs_equal(result.parsed, expected)
        ),
        "constraint_violation": (not result.schema_valid) or result.had_prose
        or result.had_fence,
    }
