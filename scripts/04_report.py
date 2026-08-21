"""Recompute every metric from saved predictions and build the results report.

This script is also the audit step: it recomputes metrics from
`reports/predictions/*.jsonl` and compares them against the values recorded at
evaluation time. If they disagree, something changed between the run and the
report, and the script says so loudly instead of quietly printing new numbers.

It needs no GPU and no model.

    python scripts/04_report.py

Outputs
    reports/results.json     machine-readable results, comparisons, error analysis
    reports/RESULTS.md       the table that goes in the README
    reports/figures/*.png    figures generated from the real metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M  # noqa: E402
from forgelm.config import SUCCESS_CRITERIA  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run, load_runs  # noqa: E402
from forgelm.schema import FIELD_ORDER  # noqa: E402

REPORTS = REPO_ROOT / "reports"
PRED_DIR = REPORTS / "predictions"
FIG_DIR = REPORTS / "figures"

CONDITIONS = [
    ("zeroshot", "Base model, zero-shot"),
    ("fewshot", "Base model, few-shot (k=8)"),
    ("lora", "Base model + LoRA adapter"),
]


def load_predictions(condition: str, split: str) -> list[dict] | None:
    path = PRED_DIR / f"{condition}_{split}.jsonl"
    if not path.exists():
        return None
    return dataio.read_jsonl(path)


def audit_against_recorded(condition: str, split: str,
                           recomputed: dict) -> dict:
    """Compare freshly recomputed metrics with what was recorded at run time."""
    path = REPORTS / "metrics" / f"{condition}_{split}.json"
    if not path.exists():
        return {"checked": False, "reason": "no recorded metric file"}
    recorded = dataio.read_json(path).get("metrics", {})
    mismatches = {}
    for key in ("exact_match", "schema_valid_rate", "json_parse_rate_strict",
                "constraint_violation_rate", "n"):
        if key in recorded and recorded[key] != recomputed.get(key):
            mismatches[key] = {"recorded": recorded[key],
                               "recomputed": recomputed.get(key)}
    return {"checked": True, "agrees": not mismatches, "mismatches": mismatches}


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fmt_ci(ci: dict) -> str:
    return f"{fmt_pct(ci['mean'])} [{fmt_pct(ci['lo'])}, {fmt_pct(ci['hi'])}]"


def make_figures(results: dict, split: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    present = [(c, label) for c, label in CONDITIONS
               if c in results["conditions"]]
    if not present:
        return []

    colours = {"zeroshot": "#9aa5b1", "fewshot": "#4c78a8", "lora": "#e45756"}

    # --- headline metrics with bootstrap intervals -------------------------
    metric_keys = [
        ("json_parse_rate_strict", "Strict JSON"),
        ("schema_valid_rate", "Schema valid"),
        ("exact_match", "Exact match"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.26
    for i, (cond, label) in enumerate(present):
        m = results["conditions"][cond]["metrics"]
        intervals = results["conditions"][cond]["intervals"]
        xs = [j + (i - len(present) / 2 + 0.5) * width
              for j in range(len(metric_keys))]
        heights = [m[k] for k, _ in metric_keys]
        errs_lo, errs_hi = [], []
        for k, _ in metric_keys:
            ci = intervals.get(k)
            if ci:
                errs_lo.append(max(0.0, m[k] - ci["lo"]))
                errs_hi.append(max(0.0, ci["hi"] - m[k]))
            else:
                errs_lo.append(0.0)
                errs_hi.append(0.0)
        ax.bar(xs, heights, width, label=label, color=colours.get(cond),
               yerr=[errs_lo, errs_hi], capsize=3, ecolor="#333333")
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels([lbl for _, lbl in metric_keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title(f"ForgeLM headline metrics on the frozen {split} split "
                 f"(n={results['n_examples']}, 95% bootstrap CI)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / f"headline_metrics_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path.relative_to(REPO_ROOT)))

    # --- per-field accuracy ------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (cond, label) in enumerate(present):
        m = results["conditions"][cond]["metrics"]
        xs = [j + (i - len(present) / 2 + 0.5) * width
              for j in range(len(FIELD_ORDER))]
        ax.bar(xs, [m["field_accuracy"][f] for f in FIELD_ORDER], width,
               label=label, color=colours.get(cond))
    ax.set_xticks(range(len(FIELD_ORDER)))
    ax.set_xticklabels(FIELD_ORDER, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Per-field accuracy on the frozen {split} split")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / f"field_accuracy_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path.relative_to(REPO_ROOT)))

    # --- error taxonomy ----------------------------------------------------
    all_cats = sorted({c for cond, _ in present
                       for c in results["conditions"][cond]["metrics"]
                       ["error_categories"]})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (cond, label) in enumerate(present):
        counts = results["conditions"][cond]["metrics"]["error_categories"]
        xs = [j + (i - len(present) / 2 + 0.5) * width
              for j in range(len(all_cats))]
        ax.bar(xs, [counts.get(c, 0) for c in all_cats], width, label=label,
               color=colours.get(cond))
    ax.set_xticks(range(len(all_cats)))
    ax.set_xticklabels(all_cats, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("examples")
    ax.set_title(f"Failure taxonomy on the frozen {split} split")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / f"error_taxonomy_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path.relative_to(REPO_ROOT)))

    # --- training curve ----------------------------------------------------
    history_path = REPORTS / "training_history.json"
    if history_path.exists():
        history = dataio.read_json(history_path)
        train = history.get("train_losses") or []
        evals = history.get("eval_losses") or []
        if train:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            ax.plot([e for e, _ in train], [l for _, l in train],
                    label="training loss", color="#4c78a8", alpha=0.8)
            if evals:
                ax.plot([e for e, _ in evals], [l for _, l in evals],
                        label="validation loss", color="#e45756",
                        marker="o", markersize=4)
                best = min(evals, key=lambda x: x[1])
                ax.axvline(best[0], ls="--", color="#666666", lw=1)
                ax.annotate(f"selected\nepoch {best[0]:.0f}", (best[0], best[1]),
                            textcoords="offset points", xytext=(8, 14),
                            fontsize=8, color="#333333")
            ax.set_xlabel("epoch")
            ax.set_ylabel("cross-entropy loss")
            ax.set_title("LoRA training and validation loss")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            path = FIG_DIR / "training_curve.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(str(path.relative_to(REPO_ROOT)))

    return written


def build_markdown(results: dict, split: str) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"# ForgeLM results -- frozen `{split}` split")
    add("")
    add(f"All numbers below were recomputed from the raw prediction files in "
        f"`reports/predictions/` by `scripts/04_report.py`. "
        f"n = {results['n_examples']} examples. "
        f"Intervals are 95% percentile bootstrap.")
    add("")

    present = [(c, label) for c, label in CONDITIONS if c in results["conditions"]]

    add("## Headline")
    add("")
    add("| System | Strict JSON | Schema valid | **Exact match** | "
        "Constraint violations |")
    add("|---|---|---|---|---|")
    for cond, label in present:
        m = results["conditions"][cond]["metrics"]
        iv = results["conditions"][cond]["intervals"]
        add(f"| {label} | {fmt_pct(m['json_parse_rate_strict'])} "
            f"| {fmt_ci(iv['schema_valid_rate'])} "
            f"| **{fmt_ci(iv['exact_match'])}** "
            f"| {fmt_pct(m['constraint_violation_rate'])} |")
    add("")

    add("## Per-field accuracy")
    add("")
    add("| System | " + " | ".join(f"`{f}`" for f in FIELD_ORDER) + " | mean |")
    add("|---" * (len(FIELD_ORDER) + 2) + "|")
    for cond, label in present:
        m = results["conditions"][cond]["metrics"]
        cells = " | ".join(fmt_pct(m["field_accuracy"][f]) for f in FIELD_ORDER)
        add(f"| {label} | {cells} | {fmt_pct(m['mean_field_accuracy'])} |")
    add("")

    add("## Classification quality")
    add("")
    add("| System | category macro-F1 | category acc | priority macro-F1 | "
        "priority acc |")
    add("|---|---|---|---|---|")
    for cond, label in present:
        m = results["conditions"][cond]["metrics"]
        add(f"| {label} | {m['category']['macro_f1']:.3f} "
            f"| {fmt_pct(m['category_accuracy'])} "
            f"| {m['priority']['macro_f1']:.3f} "
            f"| {fmt_pct(m['priority_accuracy'])} |")
    add("")

    def comparison_table(title: str, comparisons: dict, note: str = "") -> None:
        if not comparisons:
            return
        add(f"## {title}")
        add("")
        if note:
            add(note)
            add("")
        add("| Comparison | A | B | Difference (B-A) | 95% CI | McNemar p | "
            "Verdict |")
        add("|---|---|---|---|---|---|---|")
        for cmp in comparisons.values():
            diff = cmp["paired_diff"]
            verdict = ("difference detected" if diff["excludes_zero"]
                       else "not distinguishable from zero")
            add(f"| {cmp['system_a']} -> {cmp['system_b']} "
                f"| {fmt_pct(cmp['a_rate'])} | {fmt_pct(cmp['b_rate'])} "
                f"| {diff['diff'] * 100:+.1f} pp "
                f"| [{diff['lo'] * 100:+.1f}, {diff['hi'] * 100:+.1f}] pp "
                f"| {cmp['mcnemar']['p_value']:.2e} | {verdict} |")
        add("")

    comparison_table(
        "Paired comparisons -- exact match (primary)",
        results["comparisons"],
        "Every field correct. Paired bootstrap, 10,000 resamples of the "
        "example indices; McNemar is the exact binomial form.")

    comparison_table(
        "Paired comparisons -- schema validity (secondary)",
        results.get("comparisons_schema_valid", {}),
        "Reported alongside exact match because the two measure different "
        "abilities: producing a well-formed object with legal enum values, "
        "versus producing the *right* object. A system can gain a lot of the "
        "first while gaining almost none of the second.")

    add("## Failure taxonomy")
    add("")
    all_cats = sorted({c for cond, _ in present
                       for c in results["conditions"][cond]["metrics"]
                       ["error_categories"]})
    add("| Failure mode | " + " | ".join(label for _, label in present) + " |")
    add("|---" * (len(present) + 1) + "|")
    for cat in all_cats:
        cells = " | ".join(
            str(results["conditions"][cond]["metrics"]["error_categories"]
                .get(cat, 0)) for cond, _ in present)
        add(f"| `{cat}` | {cells} |")
    add("")

    verdict = results["success_criteria_verdict"]
    add("## Against the pre-declared success criteria")
    add("")
    add(f"**{verdict['headline']}**")
    add("")
    for check in verdict["checks"]:
        mark = "PASS" if check["met"] else "FAIL"
        add(f"- **{mark}** -- {check['criterion']}")
        add(f"  - {check['evidence']}")
    add("")

    if results.get("audit"):
        add("## Audit: recomputed vs recorded")
        add("")
        for cond, audit in results["audit"].items():
            if not audit.get("checked"):
                add(f"- `{cond}`: not checked ({audit.get('reason')})")
            elif audit["agrees"]:
                add(f"- `{cond}`: recomputed metrics match the values recorded "
                    f"at run time")
            else:
                add(f"- `{cond}`: **MISMATCH** {audit['mismatches']}")
        add("")

    add("## Figures")
    add("")
    for fig in results.get("figures", []):
        add(f"- `{fig}`")
    add("")
    return "\n".join(lines)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["train", "validation", "test"])
    args = ap.parse_args()
    split = args.split

    run = Run(kind="report").start()
    try:
        results: dict = {"split": split, "conditions": {}, "comparisons": {},
                         "audit": {}, "n_examples": None}

        loaded: dict[str, list[dict]] = {}
        for cond, label in CONDITIONS:
            preds = load_predictions(cond, split)
            if preds is None:
                print(f"  (skipping {cond}: no predictions for {split})")
                continue
            loaded[cond] = preds
            computed = M.compute_metrics(preds)
            intervals = M.headline_intervals(preds)
            results["conditions"][cond] = {
                "label": label,
                "n": len(preds),
                "metrics": computed,
                "intervals": intervals,
            }
            results["audit"][cond] = audit_against_recorded(cond, split, computed)
            results["n_examples"] = len(preds)
            print(f"  {cond:9s} n={len(preds):3d} "
                  f"exact_match={computed['exact_match']:.4f} "
                  f"schema_valid={computed['schema_valid_rate']:.4f} "
                  f"strict_json={computed['json_parse_rate_strict']:.4f}")

        # ---- paired comparisons ------------------------------------------
        # exact_match is the headline, but comparing on it alone would hide
        # where a system actually gained: few-shot took schema validity from
        # 27.9% to 61.6% while barely moving exact match, because getting the
        # shape right and getting every value right are different abilities.
        pairs = [("zeroshot", "fewshot"), ("zeroshot", "lora"),
                 ("fewshot", "lora")]
        for a, b in pairs:
            if a in loaded and b in loaded:
                results["comparisons"][f"{a}_vs_{b}"] = M.compare(
                    loaded[a], loaded[b], a, b, metric="exact_match")
                results["comparisons_schema_valid"] = \
                    results.get("comparisons_schema_valid", {})
                results["comparisons_schema_valid"][f"{a}_vs_{b}"] = M.compare(
                    loaded[a], loaded[b], a, b, metric="schema_valid")

        # ---- success criteria --------------------------------------------
        checks = []
        if "lora" in loaded:
            lora_m = results["conditions"]["lora"]["metrics"]
            baselines = {c: results["conditions"][c]["metrics"]
                         for c in ("zeroshot", "fewshot") if c in loaded}
            stronger = max(baselines, key=lambda c: baselines[c]["exact_match"]) \
                if baselines else None

            beats_all = all(lora_m["exact_match"] > m["exact_match"]
                            for m in baselines.values())
            checks.append({
                "criterion": "LoRA exact match exceeds BOTH unchanged-model "
                             "baselines",
                "met": beats_all,
                "evidence": "; ".join(
                    f"{c}={fmt_pct(m['exact_match'])}" for c, m in baselines.items())
                + f"; lora={fmt_pct(lora_m['exact_match'])}",
            })

            if stronger:
                cmp = results["comparisons"].get(f"{stronger}_vs_lora")
                excludes = bool(cmp and cmp["paired_diff"]["excludes_zero"])
                checks.append({
                    "criterion": f"paired bootstrap 95% CI vs the stronger "
                                 f"baseline ({stronger}) excludes zero",
                    "met": excludes,
                    "evidence": (
                        f"difference {cmp['paired_diff']['diff'] * 100:+.1f} pp, "
                        f"95% CI [{cmp['paired_diff']['lo'] * 100:+.1f}, "
                        f"{cmp['paired_diff']['hi'] * 100:+.1f}] pp, "
                        f"McNemar p={cmp['mcnemar']['p_value']:.3e}"
                    ) if cmp else "comparison unavailable",
                })
                strong_m = baselines[stronger]
                checks.append({
                    "criterion": "schema_valid_rate does not regress against "
                                 "the stronger baseline",
                    "met": lora_m["schema_valid_rate"] >= strong_m["schema_valid_rate"],
                    "evidence": f"{stronger}={fmt_pct(strong_m['schema_valid_rate'])}, "
                                f"lora={fmt_pct(lora_m['schema_valid_rate'])}",
                })
                checks.append({
                    "criterion": "constraint_violation_rate does not increase "
                                 "against the stronger baseline",
                    "met": lora_m["constraint_violation_rate"]
                    <= strong_m["constraint_violation_rate"],
                    "evidence":
                        f"{stronger}={fmt_pct(strong_m['constraint_violation_rate'])}, "
                        f"lora={fmt_pct(lora_m['constraint_violation_rate'])}",
                })

        all_met = bool(checks) and all(c["met"] for c in checks)
        results["success_criteria"] = SUCCESS_CRITERIA
        results["success_criteria_verdict"] = {
            "headline": (
                "All pre-declared success criteria were met."
                if all_met else
                ("Some pre-declared success criteria were NOT met."
                 if checks else
                 "Not yet evaluable: the LoRA condition has not been run.")),
            "all_met": all_met,
            "checks": checks,
        }

        results["figures"] = make_figures(results, split)
        results["ledger_runs"] = [
            {"run_id": r["run_id"], "kind": r["kind"], "status": r["status"],
             "elapsed_seconds": r.get("elapsed_seconds")}
            for r in load_runs()
        ]

        dataio.write_json(results, REPORTS / f"results_{split}.json")
        markdown = build_markdown(results, split)
        (REPORTS / f"RESULTS_{split}.md").write_text(markdown, encoding="utf-8")
        if split == "test":
            (REPORTS / "RESULTS.md").write_text(markdown, encoding="utf-8")

        mismatches = {c: a for c, a in results["audit"].items()
                      if a.get("checked") and not a["agrees"]}
        if mismatches:
            print(f"\n!! AUDIT MISMATCH: {mismatches}")
            run.warn(f"audit mismatch: {mismatches}")
        else:
            print("\naudit: recomputed metrics match every recorded value")

        print(f"\n{results['success_criteria_verdict']['headline']}")
        for check in checks:
            print(f"  [{'PASS' if check['met'] else 'FAIL'}] {check['criterion']}")
            print(f"         {check['evidence']}")

        run.metrics = {"verdict": results["success_criteria_verdict"],
                       "audit": results["audit"]}
        run.add_artifact("results_json", REPORTS / f"results_{split}.json")
        run.add_artifact("results_md", REPORTS / f"RESULTS_{split}.md")
        run.finish("success")
        print(f"\nwrote reports/RESULTS_{split}.md and "
              f"reports/results_{split}.json")
        print(f"figures: {results['figures']}")
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
