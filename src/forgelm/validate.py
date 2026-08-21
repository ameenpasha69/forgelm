"""Dataset quality control and leakage detection.

Two jobs:

1. **Integrity** -- every record has the required fields, the expected output is
   schema-valid, labels are drawn from the declared enums, and the priority
   label actually matches the documented rule (this catches generator bugs that
   would otherwise silently teach the model a wrong rule).

2. **Leakage** -- exact, normalised and near-duplicate text overlap, both within
   and across splits. Group-aware splitting already guarantees no scenario
   family straddles a split boundary; the near-duplicate scan is the empirical
   check that this was sufficient.

Checks return structured findings rather than raising, so a report can list
everything at once. `severity` distinguishes a blocking defect from an
observation worth recording.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from .datagen import compute_priority
from .schema import (
    CATEGORIES,
    normalise_text,
    validate_output,
)

# Words that would give the priority label away if they appeared in the ticket
# text. The model must infer priority from severity + blast radius, not read it.
PRIORITY_LEAK_TERMS = (
    "priority", "critical", "p1", "p2", "p3", "sev1", "sev2",
    "severity", "low priority", "high priority",
)

REQUIRED_RECORD_FIELDS = (
    "example_id", "ticket_text", "expected_output", "expected_output_json",
    "category", "scenario_family", "template_family", "user_scale",
    "base_severity", "source", "generator_version", "schema_version",
    "generation_seed", "qc_status",
    "symptom_text", "detail_text", "ask_text", "scope_text",
)

# Phrases that assert other people are experiencing the problem. If one of
# these appears in a clause while users_affected == 1, the example contradicts
# its own label. Found by manual review round 1; now checked automatically.
MULTI_USER_MARKERS = (
    "report the same thing",
    "others have reported",
    "several people",
    "at least two people",
    "colleagues on the same",
    "two graduates",
    "everyone who tries",
    "people have reported",
)

MIN_CHARS = 40
MAX_CHARS = 700

# Near-duplicate threshold on character 4-gram Jaccard similarity. 0.80 is
# strict for this data: independently generated examples from different
# scenario families sit far below it (see reports/dataset_validation.json).
NEAR_DUP_THRESHOLD = 0.80


@dataclass
class Finding:
    check: str
    severity: str          # "error" | "warning" | "info"
    message: str
    detail: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Structural checks
# --------------------------------------------------------------------------

def check_required_fields(records: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    for rec in records:
        missing = [f for f in REQUIRED_RECORD_FIELDS if f not in rec]
        if missing:
            findings.append(Finding(
                "required_fields", "error",
                f"{rec.get('example_id', '<no id>')} missing {missing}",
                {"example_id": rec.get("example_id"), "missing": missing},
            ))
    return findings


def check_expected_output_schema(records: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    for rec in records:
        violations = validate_output(rec.get("expected_output"))
        if violations:
            findings.append(Finding(
                "expected_output_schema", "error",
                f"{rec['example_id']} expected_output violates schema: {violations}",
                {"example_id": rec["example_id"], "violations": violations},
            ))
    return findings


def check_priority_rule(records: list[dict[str, Any]]) -> list[Finding]:
    """The labels must match the rule published in the dataset card."""
    findings = []
    for rec in records:
        exp = rec["expected_output"]
        recomputed = compute_priority(
            rec["base_severity"], exp["is_security_incident"],
            exp["users_affected"],
        )
        if recomputed != exp["priority"]:
            findings.append(Finding(
                "priority_rule", "error",
                f"{rec['example_id']} priority={exp['priority']} but rule gives "
                f"{recomputed}",
                {"example_id": rec["example_id"], "stored": exp["priority"],
                 "recomputed": recomputed},
            ))
    return findings


def check_users_recoverable(records: list[dict[str, Any]]) -> list[Finding]:
    """A reader must be able to recover users_affected from the text alone.

    Either the exact integer is written out, or the ticket is unambiguously
    about a single person. Anything else makes the label unguessable and the
    metric unfair.
    """
    findings = []
    for rec in records:
        n = rec["expected_output"]["users_affected"]
        text = rec["ticket_text"]
        if n == 1:
            singular = any(
                phrase in text.lower()
                for phrase in ("just me", "only affecting me", "only person",
                               "nobody else", "only one person")
            )
            if not singular:
                findings.append(Finding(
                    "users_recoverable", "error",
                    f"{rec['example_id']} users_affected=1 but text has no "
                    f"singular marker",
                    {"example_id": rec["example_id"]},
                ))
        else:
            if not re.search(rf"\b{n}\b", text):
                findings.append(Finding(
                    "users_recoverable", "error",
                    f"{rec['example_id']} users_affected={n} not present in text",
                    {"example_id": rec["example_id"], "users_affected": n},
                ))
    return findings


def check_scope_consistency(records: list[dict[str, Any]]) -> list[Finding]:
    """A single-user ticket must not contain clauses asserting others are hit.

    Checks the symptom / detail / ask clauses rather than the whole rendered
    text, because the scope clause itself legitimately mentions other people
    ("nobody else reports this").
    """
    findings = []
    for rec in records:
        if rec["expected_output"]["users_affected"] != 1:
            continue
        clauses = " | ".join(
            rec.get(k, "") for k in ("symptom_text", "detail_text", "ask_text")
        ).lower()
        hits = [m for m in MULTI_USER_MARKERS if m in clauses]
        if hits:
            findings.append(Finding(
                "scope_consistency", "error",
                f"{rec['example_id']} has users_affected=1 but its clauses imply "
                f"multiple affected people: {hits}",
                {"example_id": rec["example_id"], "markers": hits,
                 "detail_text": rec.get("detail_text")},
            ))
    return findings


def check_priority_leak_terms(records: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    for rec in records:
        low = rec["ticket_text"].lower()
        hits = [t for t in PRIORITY_LEAK_TERMS if t in low]
        if hits:
            findings.append(Finding(
                "priority_leak_terms", "error",
                f"{rec['example_id']} ticket text contains priority-revealing "
                f"term(s) {hits}",
                {"example_id": rec["example_id"], "terms": hits},
            ))
    return findings


def check_length(records: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    for rec in records:
        n = len(rec["ticket_text"])
        if n < MIN_CHARS:
            findings.append(Finding(
                "length", "warning",
                f"{rec['example_id']} unusually short ({n} chars)",
                {"example_id": rec["example_id"], "chars": n}))
        elif n > MAX_CHARS:
            findings.append(Finding(
                "length", "warning",
                f"{rec['example_id']} unusually long ({n} chars)",
                {"example_id": rec["example_id"], "chars": n}))
    return findings


def check_balance(records: list[dict[str, Any]]) -> list[Finding]:
    """Report class balance; flag only genuinely degenerate distributions."""
    findings = []
    cat_counts = Counter(r["category"] for r in records)
    missing = [c for c in CATEGORIES if cat_counts[c] == 0]
    if missing:
        findings.append(Finding(
            "balance", "error", f"categories with no examples: {missing}",
            {"missing": missing}))

    if cat_counts:
        lo, hi = min(cat_counts.values()), max(cat_counts.values())
        ratio = hi / lo if lo else float("inf")
        severity = "warning" if ratio > 2.0 else "info"
        findings.append(Finding(
            "balance", severity,
            f"category counts min={lo} max={hi} imbalance_ratio={ratio:.2f}",
            {"counts": dict(sorted(cat_counts.items())), "ratio": round(ratio, 3)}))

    pri_counts = Counter(r["expected_output"]["priority"] for r in records)
    findings.append(Finding(
        "balance", "info", f"priority distribution: {dict(sorted(pri_counts.items()))}",
        {"counts": dict(sorted(pri_counts.items()))}))
    return findings


# --------------------------------------------------------------------------
# Duplicate / leakage detection
# --------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 4) -> set[str]:
    t = normalise_text(text)
    if len(t) < n:
        return {t}
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def check_duplicates(records: list[dict[str, Any]]) -> list[Finding]:
    findings = []

    exact: dict[str, list[str]] = defaultdict(list)
    norm: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        exact[rec["ticket_text"]].append(rec["example_id"])
        norm[normalise_text(rec["ticket_text"])].append(rec["example_id"])

    for text, ids in exact.items():
        if len(ids) > 1:
            findings.append(Finding(
                "exact_duplicate", "error",
                f"{len(ids)} examples share identical text: {ids}",
                {"example_ids": ids, "text": text[:160]}))

    for text, ids in norm.items():
        if len(ids) > 1 and not any(
            f.check == "exact_duplicate" and set(f.detail["example_ids"]) == set(ids)
            for f in findings
        ):
            findings.append(Finding(
                "normalised_duplicate", "error",
                f"{len(ids)} examples identical after normalisation: {ids}",
                {"example_ids": ids}))

    return findings


def near_duplicate_pairs(records: list[dict[str, Any]],
                         threshold: float = NEAR_DUP_THRESHOLD,
                         same_family_only: bool | None = None
                         ) -> list[dict[str, Any]]:
    """All record pairs above the similarity threshold.

    O(n^2) on 300 records is ~45k comparisons -- fast enough that an exact scan
    beats an approximate index, and an exact answer is what an auditor wants.
    """
    grams = [(r, _char_ngrams(r["ticket_text"])) for r in records]
    pairs = []
    for i in range(len(grams)):
        ri, gi = grams[i]
        for j in range(i + 1, len(grams)):
            rj, gj = grams[j]
            same_family = ri["scenario_family"] == rj["scenario_family"]
            if same_family_only is True and not same_family:
                continue
            if same_family_only is False and same_family:
                continue
            sim = jaccard(gi, gj)
            if sim >= threshold:
                pairs.append({
                    "a": ri["example_id"], "b": rj["example_id"],
                    "similarity": round(sim, 4),
                    "same_scenario_family": same_family,
                    "same_template_family":
                        ri["template_family"] == rj["template_family"],
                })
    return sorted(pairs, key=lambda p: -p["similarity"])


def check_cross_split_leakage(records: list[dict[str, Any]],
                              manifest: dict[str, Any],
                              threshold: float = NEAR_DUP_THRESHOLD
                              ) -> list[Finding]:
    """The decisive leakage check: does any test example resemble a train one?"""
    findings = []
    example_split = manifest["example_split"]
    family_split = manifest["family_split"]

    # 1. No scenario family may span splits.
    fam_to_splits: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        fam_to_splits[rec["scenario_family"]].add(example_split[rec["example_id"]])
    for fam, splits in sorted(fam_to_splits.items()):
        if len(splits) > 1:
            findings.append(Finding(
                "scenario_family_leakage", "error",
                f"family {fam} appears in multiple splits: {sorted(splits)}",
                {"scenario_family": fam, "splits": sorted(splits)}))

    # 2. No example id may be assigned two splits (structural sanity).
    if len(example_split) != len(set(example_split)):
        findings.append(Finding(
            "duplicate_assignment", "error",
            "an example id appears more than once in the manifest", None))

    # 3. Empirical near-duplicate scan across split boundaries.
    grams = [(r, _char_ngrams(r["ticket_text"]), example_split[r["example_id"]])
             for r in records]
    worst = 0.0
    worst_pair = None
    offending = []
    for i in range(len(grams)):
        ri, gi, si = grams[i]
        for j in range(i + 1, len(grams)):
            rj, gj, sj = grams[j]
            if si == sj:
                continue
            sim = jaccard(gi, gj)
            if sim > worst:
                worst, worst_pair = sim, (ri["example_id"], rj["example_id"], si, sj)
            if sim >= threshold:
                offending.append({
                    "a": ri["example_id"], "a_split": si,
                    "b": rj["example_id"], "b_split": sj,
                    "similarity": round(sim, 4),
                })

    if offending:
        findings.append(Finding(
            "cross_split_near_duplicate", "error",
            f"{len(offending)} cross-split pair(s) at or above similarity "
            f"{threshold}",
            {"pairs": offending[:25], "total": len(offending)}))
    else:
        findings.append(Finding(
            "cross_split_near_duplicate", "info",
            f"no cross-split pair reaches similarity {threshold}; "
            f"maximum observed = {worst:.4f}",
            {"max_similarity": round(worst, 4),
             "max_pair": worst_pair,
             "threshold": threshold}))

    # 4. Template families are expected to span splits by design; record the
    #    fact explicitly rather than letting a reader assume otherwise.
    tmpl_to_splits: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        tmpl_to_splits[rec["template_family"]].add(example_split[rec["example_id"]])
    spanning = {t: sorted(s) for t, s in sorted(tmpl_to_splits.items()) if len(s) > 1}
    findings.append(Finding(
        "template_family_spread", "info",
        f"{len(spanning)}/{len(tmpl_to_splits)} template families appear in more "
        f"than one split. This is intended: template families are surface styles "
        f"and carry no label information. The generalisation axis under test is "
        f"scenario_family, which is held out strictly.",
        {"spanning": spanning}))

    findings.append(Finding(
        "family_split_summary", "info",
        f"{len(family_split)} scenario families assigned across splits",
        {"counts": dict(Counter(family_split.values()))}))

    return findings


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_all(records: list[dict[str, Any]],
            manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    findings: list[Finding] = []
    findings += check_required_fields(records)
    findings += check_expected_output_schema(records)
    findings += check_priority_rule(records)
    findings += check_users_recoverable(records)
    findings += check_scope_consistency(records)
    findings += check_priority_leak_terms(records)
    findings += check_length(records)
    findings += check_balance(records)
    findings += check_duplicates(records)

    if manifest is not None:
        findings += check_cross_split_leakage(records, manifest)

    by_severity = Counter(f.severity for f in findings)
    return {
        "n_records": len(records),
        "n_findings": len(findings),
        "errors": by_severity.get("error", 0),
        "warnings": by_severity.get("warning", 0),
        "info": by_severity.get("info", 0),
        "passed": by_severity.get("error", 0) == 0,
        "findings": [f.as_dict() for f in findings],
    }


def summarise(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    lengths = [len(r["ticket_text"]) for r in records]
    return {
        "n": len(records),
        "categories": dict(sorted(Counter(r["category"] for r in records).items())),
        "priorities": dict(sorted(Counter(
            r["expected_output"]["priority"] for r in records).items())),
        "affected_services": dict(sorted(Counter(
            r["expected_output"]["affected_service"] for r in records).items())),
        "is_security_incident": dict(sorted(Counter(
            str(r["expected_output"]["is_security_incident"]) for r in records
        ).items())),
        "template_families": dict(sorted(Counter(
            r["template_family"] for r in records).items())),
        "scenario_families": len({r["scenario_family"] for r in records}),
        "ticket_chars": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        },
        "users_affected": {
            "min": min(r["expected_output"]["users_affected"] for r in records)
            if records else 0,
            "max": max(r["expected_output"]["users_affected"] for r in records)
            if records else 0,
        },
    }
