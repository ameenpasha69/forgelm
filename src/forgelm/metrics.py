"""Metrics, uncertainty and paired significance testing.

Every metric here is computed from the saved per-example prediction records, so
any number in the report can be regenerated from `reports/predictions/*.jsonl`
without re-running a model. `scripts/06_report.py` does exactly that as an
audit step.

On uncertainty: the test set has 86 examples. That is small. A difference of a
few points is noise. Reporting a bare point estimate would invite exactly the
over-claiming this project is trying to avoid, so every headline metric carries
a bootstrap interval and every model-vs-model comparison carries a paired test.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Sequence

from .schema import CATEGORIES, FIELD_ORDER, PRIORITIES
from .seeding import SEEDS


# --------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------

def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any],
             labels: Sequence[Any]) -> dict[str, Any]:
    """Macro-averaged F1 over a fixed label set.

    The label set is fixed (not inferred from the data) so that a model which
    never predicts a rare class is penalised for it rather than having the
    class quietly dropped from the average.
    """
    per_label: dict[str, dict[str, float]] = {}
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        support = sum(1 for t in y_true if t == label)
        per_label[str(label)] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": support,
            "predicted": sum(1 for p in y_pred if p == label),
        }
        f1s.append(f1)
    return {
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_label": per_label,
    }


def confusion(y_true: Sequence[Any], y_pred: Sequence[Any],
              labels: Sequence[Any]) -> dict[str, dict[str, int]]:
    """Confusion matrix with an explicit bucket for out-of-vocabulary output."""
    label_set = set(labels)
    matrix = {str(t): {str(p): 0 for p in labels} | {"<invalid>": 0}
              for t in labels}
    for t, p in zip(y_true, y_pred):
        row = matrix.setdefault(str(t), {str(x): 0 for x in labels} | {"<invalid>": 0})
        key = str(p) if p in label_set else "<invalid>"
        row[key] = row.get(key, 0) + 1
    return matrix


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics over per-example prediction records."""
    n = len(records)
    if n == 0:
        return {"n": 0}

    def rate(pred: Callable[[dict[str, Any]], bool]) -> float:
        return round(sum(1 for r in records if pred(r)) / n, 4)

    field_accuracy = {}
    for key in FIELD_ORDER:
        correct = sum(1 for r in records if r["field_correct"].get(key, False))
        field_accuracy[key] = round(correct / n, 4)

    y_true_cat = [r["expected"]["category"] for r in records]
    y_pred_cat = [r["predicted_fields"].get("category") for r in records]
    y_true_pri = [r["expected"]["priority"] for r in records]
    y_pred_pri = [r["predicted_fields"].get("priority") for r in records]

    metrics: dict[str, Any] = {
        "n": n,
        "json_parse_rate_strict": rate(lambda r: r["strict_json"]),
        "json_parse_rate_lenient": rate(lambda r: r["lenient_json"]),
        "schema_valid_rate": rate(lambda r: r["schema_valid"]),
        "exact_match": rate(lambda r: r["exact_match"]),
        "constraint_violation_rate": rate(lambda r: r["constraint_violation"]),
        "markdown_fence_rate": rate(lambda r: r["had_fence"]),
        "prose_outside_json_rate": rate(lambda r: r["had_prose"]),
        "truncation_rate": rate(lambda r: r["truncated"]),
        "field_accuracy": field_accuracy,
        "mean_field_accuracy": round(
            sum(field_accuracy.values()) / len(field_accuracy), 4),
        "category": macro_f1(y_true_cat, y_pred_cat, CATEGORIES),
        "priority": macro_f1(y_true_pri, y_pred_pri, PRIORITIES),
        "category_accuracy": field_accuracy["category"],
        "priority_accuracy": field_accuracy["priority"],
        "error_categories": dict(sorted(
            Counter(r["error_category"] for r in records).items())),
        "confusion_category": confusion(y_true_cat, y_pred_cat, CATEGORIES),
        "confusion_priority": confusion(y_true_pri, y_pred_pri, PRIORITIES),
    }

    # Per-dataset-category breakdown of the headline metric.
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cat[r["expected"]["category"]].append(r)
    metrics["exact_match_by_category"] = {
        cat: {"n": len(rows),
              "exact_match": round(
                  sum(1 for x in rows if x["exact_match"]) / len(rows), 4)}
        for cat, rows in sorted(by_cat.items())
    }

    if any("latency_seconds" in r for r in records):
        lat = [r["latency_seconds"] for r in records if "latency_seconds" in r]
        metrics["latency_seconds"] = {
            "mean": round(sum(lat) / len(lat), 4),
            "min": round(min(lat), 4), "max": round(max(lat), 4),
            "note": ("Observation only. Measured on one consumer GPU with "
                     "batched generation; not a serving-performance claim."),
        }
    return metrics


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = q * (len(sorted_values) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000,
                 alpha: float = 0.05, seed: int | None = None
                 ) -> dict[str, Any]:
    """Percentile bootstrap CI for the mean of a 0/1 (or continuous) vector."""
    import random

    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    r = random.Random(SEEDS["bootstrap"] if seed is None else seed)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[r.randrange(n)]
        means.append(total / n)
    means.sort()
    return {
        "mean": round(sum(values) / n, 4),
        "lo": round(_percentile(means, alpha / 2), 4),
        "hi": round(_percentile(means, 1 - alpha / 2), 4),
        "n": n,
        "n_resamples": n_resamples,
        "alpha": alpha,
    }


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n_resamples: int = 10000, alpha: float = 0.05,
                          seed: int | None = None) -> dict[str, Any]:
    """CI for mean(b) - mean(a) resampling *example indices*, not conditions.

    Paired resampling is the right choice here because both systems are scored
    on the identical test examples; treating them as independent samples would
    overstate the uncertainty.
    """
    import random

    if len(a) != len(b):
        raise ValueError(f"paired inputs must be equal length: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    r = random.Random(SEEDS["bootstrap"] if seed is None else seed)
    diffs = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            idx = r.randrange(n)
            total += b[idx] - a[idx]
        diffs.append(total / n)
    diffs.sort()
    lo = _percentile(diffs, alpha / 2)
    hi = _percentile(diffs, 1 - alpha / 2)
    return {
        "diff": round(sum(b) / n - sum(a) / n, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "n": n,
        "n_resamples": n_resamples,
        "alpha": alpha,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def hierarchical_bootstrap(per_seed_values: Sequence[Sequence[float]],
                           n_resamples: int = 10000, alpha: float = 0.05,
                           seed: int | None = None) -> dict[str, Any]:
    """Interval accounting for BOTH test-example sampling and training-seed choice.

    v1 reported a paired bootstrap over test examples only. That answers "if I
    drew a different test set, how would this number move?" -- but not "if I had
    trained with a different seed, how would it move?". With one training run
    the second question is unanswerable; with several it is estimable.

    Two-level resample: draw seeds with replacement, then draw example indices
    with replacement (the same indices for every drawn seed, preserving the
    pairing), and average.

    Honesty note: with only a handful of seeds the seed level is a *coarse*
    estimate. The returned dict carries `n_seeds` so a reader can weigh it, and
    `seed_spread` reports the raw observed range, which with n=3 is often the
    more truthful summary.
    """
    import random

    n_seeds = len(per_seed_values)
    if n_seeds == 0:
        return {"mean": float("nan"), "n_seeds": 0}
    n_examples = len(per_seed_values[0])
    if any(len(v) != n_examples for v in per_seed_values):
        raise ValueError("every seed must be scored on the same examples")

    seed_means = [sum(v) / len(v) for v in per_seed_values]
    r = random.Random(SEEDS["bootstrap"] if seed is None else seed)

    means = []
    for _ in range(n_resamples):
        drawn = [per_seed_values[r.randrange(n_seeds)] for _ in range(n_seeds)]
        indices = [r.randrange(n_examples) for _ in range(n_examples)]
        total = 0.0
        for values in drawn:
            total += sum(values[i] for i in indices) / n_examples
        means.append(total / n_seeds)
    means.sort()

    spread = max(seed_means) - min(seed_means)
    if n_seeds > 1:
        mean_of_means = sum(seed_means) / n_seeds
        variance = sum((m - mean_of_means) ** 2 for m in seed_means) / (n_seeds - 1)
        std = math.sqrt(variance)
    else:
        std = float("nan")

    return {
        "mean": round(sum(seed_means) / n_seeds, 4),
        "lo": round(_percentile(means, alpha / 2), 4),
        "hi": round(_percentile(means, 1 - alpha / 2), 4),
        "n_seeds": n_seeds,
        "n_examples": n_examples,
        "per_seed_means": [round(m, 4) for m in seed_means],
        "seed_std": round(std, 4) if not math.isnan(std) else None,
        "seed_min": round(min(seed_means), 4),
        "seed_max": round(max(seed_means), 4),
        "seed_spread": round(spread, 4),
        "n_resamples": n_resamples,
        "caveat": (
            f"seed level estimated from {n_seeds} runs; treat the interval as "
            f"indicative and prefer the raw per-seed spread "
            f"({min(seed_means):.4f} to {max(seed_means):.4f}) as the honest "
            f"summary"),
    }


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, Any]:
    """Exact McNemar test on paired binary outcomes (system A vs system B).

    Uses the exact binomial form rather than the chi-square approximation,
    because with n=86 the discordant counts can be small enough that the
    approximation is unreliable.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired inputs must be equal length")

    both = sum(1 for x, y in zip(a_correct, b_correct) if x and y)
    only_a = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    only_b = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    neither = sum(1 for x, y in zip(a_correct, b_correct) if not x and not y)

    n_discordant = only_a + only_b
    result: dict[str, Any] = {
        "both_correct": both,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "neither_correct": neither,
        "n_discordant": n_discordant,
        "test": "exact binomial (McNemar)",
    }

    if n_discordant == 0:
        result["p_value"] = 1.0
        result["note"] = "no discordant pairs; the systems agree on every example"
        return result

    try:
        from scipy.stats import binomtest

        result["p_value"] = float(
            binomtest(only_b, n_discordant, 0.5, alternative="two-sided").pvalue
        )
        result["backend"] = "scipy.stats.binomtest"
    except ImportError:
        # Exact two-sided binomial without scipy.
        k = min(only_a, only_b)
        tail = sum(math.comb(n_discordant, i) for i in range(k + 1))
        p = min(1.0, 2.0 * tail / (2 ** n_discordant))
        result["p_value"] = p
        result["backend"] = "math.comb fallback"

    return result


def compare(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]],
            name_a: str, name_b: str,
            metric: str = "exact_match") -> dict[str, Any]:
    """Full paired comparison of two systems on the same examples."""
    index_a = {r["example_id"]: r for r in records_a}
    index_b = {r["example_id"]: r for r in records_b}
    shared = sorted(set(index_a) & set(index_b))

    if len(shared) != len(records_a) or len(shared) != len(records_b):
        raise ValueError(
            f"cannot compare {name_a} ({len(records_a)}) with {name_b} "
            f"({len(records_b)}): only {len(shared)} shared example ids. "
            f"Both systems must be evaluated on the identical frozen set."
        )

    a_vals = [1.0 if index_a[e][metric] else 0.0 for e in shared]
    b_vals = [1.0 if index_b[e][metric] else 0.0 for e in shared]

    return {
        "metric": metric,
        "system_a": name_a,
        "system_b": name_b,
        "a_rate": round(sum(a_vals) / len(a_vals), 4),
        "b_rate": round(sum(b_vals) / len(b_vals), 4),
        "a_ci": bootstrap_ci(a_vals),
        "b_ci": bootstrap_ci(b_vals),
        "paired_diff": paired_bootstrap_diff(a_vals, b_vals),
        "mcnemar": mcnemar([bool(v) for v in a_vals], [bool(v) for v in b_vals]),
        "n_examples": len(shared),
    }


def headline_intervals(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Bootstrap intervals for the metrics that appear in the results table."""
    out = {}
    for name, key in (
        ("exact_match", "exact_match"),
        ("schema_valid_rate", "schema_valid"),
        ("json_parse_rate_lenient", "lenient_json"),
        ("constraint_violation_rate", "constraint_violation"),
    ):
        out[name] = bootstrap_ci([1.0 if r[key] else 0.0 for r in records])
    for field in FIELD_ORDER:
        out[f"field_accuracy.{field}"] = bootstrap_ci(
            [1.0 if r["field_correct"].get(field) else 0.0 for r in records])
    return out
