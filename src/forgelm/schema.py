"""Task schema for ForgeLM: IT helpdesk ticket triage -> strict JSON.

This module is the single source of truth for what a *valid* model output looks
like. Everything else (data generation, prompting, parsing, metrics, tests)
imports from here so that the definition can never drift between components.

The output object has exactly five keys, in a fixed order, with closed
enumerations for three of them. A closed enum is deliberate: it makes
"the model invented a label that does not exist" a *measurable* failure mode
rather than a judgement call.
"""

from __future__ import annotations

import json
import re
from typing import Any

SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

CATEGORIES: tuple[str, ...] = (
    "access_management",
    "hardware",
    "network",
    "software",
    "security",
    "account_billing",
    "email",
    "other",
)

PRIORITIES: tuple[str, ...] = ("low", "medium", "high", "critical")

AFFECTED_SERVICES: tuple[str, ...] = (
    "vpn",
    "email",
    "crm",
    "printer",
    "laptop",
    "wifi",
    "payroll",
    "database",
    "phone",
    "none",
)

# Fixed key order. Used for canonical serialisation so that string comparison
# of two semantically identical objects never fails for ordering reasons.
FIELD_ORDER: tuple[str, ...] = (
    "category",
    "priority",
    "affected_service",
    "is_security_incident",
    "users_affected",
)

REQUIRED_FIELDS = frozenset(FIELD_ORDER)

# Fields used for per-field accuracy reporting, with their type family.
FIELD_TYPES: dict[str, str] = {
    "category": "enum",
    "priority": "enum",
    "affected_service": "enum",
    "is_security_incident": "bool",
    "users_affected": "int",
}

ENUM_VALUES: dict[str, tuple[str, ...]] = {
    "category": CATEGORIES,
    "priority": PRIORITIES,
    "affected_service": AFFECTED_SERVICES,
}

MAX_USERS_AFFECTED = 5000

# A JSON-Schema style description, emitted into the prompt and the dataset card
# so that humans and the model see the same contract.
JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(FIELD_ORDER),
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "priority": {"type": "string", "enum": list(PRIORITIES)},
        "affected_service": {"type": "string", "enum": list(AFFECTED_SERVICES)},
        "is_security_incident": {"type": "boolean"},
        "users_affected": {"type": "integer", "minimum": 1,
                           "maximum": MAX_USERS_AFFECTED},
    },
}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

class SchemaViolation(str):
    """A single, human-readable schema violation code."""


def validate_output(obj: Any) -> list[str]:
    """Validate a parsed object against the ForgeLM output schema.

    Returns a list of violation codes. An empty list means the object is
    schema-valid. We return *all* violations rather than raising on the first
    one, because the failure taxonomy wants to distinguish, for example,
    "missing field" from "bad enum value" even when both occur.
    """
    violations: list[str] = []

    if not isinstance(obj, dict):
        return ["not_an_object"]

    keys = set(obj.keys())

    for missing in sorted(REQUIRED_FIELDS - keys):
        violations.append(f"missing_field:{missing}")

    for extra in sorted(keys - REQUIRED_FIELDS):
        violations.append(f"extra_field:{extra}")

    for field, allowed in ENUM_VALUES.items():
        if field not in obj:
            continue
        value = obj[field]
        if not isinstance(value, str):
            violations.append(f"wrong_type:{field}")
        elif value not in allowed:
            violations.append(f"invalid_enum:{field}")

    if "is_security_incident" in obj:
        # Explicitly reject 0/1: bool is a subclass of int in Python, so the
        # isinstance order here matters and is intentional.
        if not isinstance(obj["is_security_incident"], bool):
            violations.append("wrong_type:is_security_incident")

    if "users_affected" in obj:
        value = obj["users_affected"]
        if isinstance(value, bool) or not isinstance(value, int):
            violations.append("wrong_type:users_affected")
        elif value < 1:
            violations.append("out_of_range:users_affected")
        elif value > MAX_USERS_AFFECTED:
            violations.append("out_of_range:users_affected")

    return violations


def is_valid(obj: Any) -> bool:
    return not validate_output(obj)


# --------------------------------------------------------------------------
# Canonical serialisation
# --------------------------------------------------------------------------

def canonical_json(obj: dict[str, Any]) -> str:
    """Serialise a schema-valid object deterministically.

    Keys are emitted in FIELD_ORDER (not alphabetical order) because that is
    the order the model is taught to produce, and a stable textual form is
    what makes exact-match comparison meaningful.
    """
    ordered = {k: obj[k] for k in FIELD_ORDER if k in obj}
    # Include any stray keys at the end so canonicalising an invalid object
    # does not silently drop evidence of the violation.
    for k in obj:
        if k not in ordered:
            ordered[k] = obj[k]
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


def outputs_equal(a: Any, b: Any) -> bool:
    """Exact structured match: same keys, same values, type-sensitive."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if set(a.keys()) != set(b.keys()):
        return False
    for key in a:
        av, bv = a[key], b[key]
        # bool/int must not compare equal (True == 1 in Python).
        if isinstance(av, bool) != isinstance(bv, bool):
            return False
        if av != bv:
            return False
    return True


# --------------------------------------------------------------------------
# Human-readable schema block for prompts
# --------------------------------------------------------------------------

def schema_prompt_block() -> str:
    """The schema description that is injected into every prompt.

    Kept in one place so the zero-shot baseline, the few-shot baseline and the
    fine-tuning data are provably given the same contract. If this string
    changed between baseline and training, the comparison would be invalid.
    """
    return (
        "Output a single JSON object with exactly these five keys:\n"
        f'  "category": one of {list(CATEGORIES)}\n'
        f'  "priority": one of {list(PRIORITIES)}\n'
        f'  "affected_service": one of {list(AFFECTED_SERVICES)}\n'
        '  "is_security_incident": true or false\n'
        '  "users_affected": an integer >= 1\n'
        "Output only the JSON object. No explanation, no markdown, no code fences."
    )


_WHITESPACE_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation runs.

    Used for duplicate detection, not for model input.
    """
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text
