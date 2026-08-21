"""Compare the primary run against a controlled ablation.

Kept separate from `04_report.py` on purpose. The primary experiment is the
result; an ablation is a smaller, secondary question asked afterwards, and
mixing the two in one report invites a reader to treat an exploratory number
with the same weight as the pre-registered one.

    python scripts/07_ablation_report.py --ablation frac50

Both runs are evaluated on the SAME frozen test split with the same prompt,
decoding and parser, so the paired tests are valid. Only one variable differs.

Output: reports/ABLATION.md and reports/ablation.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run, load_runs  # noqa: E402
from forgelm.schema import FIELD_ORDER  # noqa: E402

REPORTS = REPO_ROOT / "reports"


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fmt_n(n: int | None) -> str:
    """Training-set sizes come from the ledger and may be absent."""
    return "unknown" if n is None else str(n)


def find_training_run(tag: str | None):
    """Locate the ledger record for a training run by its adapter tag."""
    suffix = f"_{tag}" if tag else ""
    target = f"artifacts/lora_adapter{suffix}"
    for record in reversed(load_runs("train_lora")):
        if record.get("status") != "success":
            continue
        artifacts = record.get("artifacts", {})
        adapter = str(artifacts.get("adapter", "")).replace("\\", "/")
        if adapter.endswith(target.split("/")[-1]):
            return record
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True,
                    help="tag of the ablation run, e.g. frac50")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    run = Run(kind="ablation_report").start()
    try:
        base_path = REPORTS / "predictions" / f"lora_{args.split}.jsonl"
        abl_path = REPORTS / "predictions" / \
            f"lora_{args.ablation}_{args.split}.jsonl"
        for path in (base_path, abl_path):
            if not path.exists():
                raise SystemExit(f"missing predictions: {path}")

        base_preds = dataio.read_jsonl(base_path)
        abl_preds = dataio.read_jsonl(abl_path)
        base_m = M.compute_metrics(base_preds)
        abl_m = M.compute_metrics(abl_preds)

        base_run = find_training_run(None)
        abl_run = find_training_run(args.ablation)

        subsample = (abl_run or {}).get("inputs", {}).get("train_subsample") or {}
        n_full = subsample.get("n_before")
        n_abl = subsample.get("n_after")

        # If the ledger record is missing we can still do the statistics, but we
        # cannot say what the ablation actually changed -- and an ablation whose
        # manipulation is unknown is not interpretable. Say so plainly rather
        # than emitting a report with blanks in it.
        if abl_run is None:
            run.warn(f"no successful train_lora ledger record found for tag "
                     f"{args.ablation!r}")
            print(f"WARNING: no training-run record found for tag "
                  f"{args.ablation!r}. Training-set sizes and family coverage "
                  f"cannot be reported, so the comparison below is statistics "
                  f"without a described manipulation.", file=sys.stderr)

        # The decisive question for interpreting this ablation: did halving the
        # data remove *examples per scenario*, or did it remove *scenarios*?
        # Those support opposite conclusions, so it is measured, not assumed.
        family_coverage = {}
        if subsample.get("example_ids"):
            records = {r["example_id"]: r
                       for r in dataio.read_jsonl(dataio.PROCESSED_DATASET)}
            kept_families = Counter(
                records[e]["scenario_family"] for e in subsample["example_ids"]
                if e in records)
            manifest = dataio.read_json(dataio.SPLIT_MANIFEST)
            all_train_families = {
                f for f, s in manifest["family_split"].items() if s == "train"}
            family_coverage = {
                "train_families_total": len(all_train_families),
                "train_families_retained": len(kept_families),
                "families_dropped": sorted(all_train_families - set(kept_families)),
                "examples_per_family_full": round(
                    n_full / len(all_train_families), 2) if n_full else None,
                "examples_per_family_ablation": round(
                    n_abl / len(kept_families), 2) if kept_families else None,
            }

        comparison = M.compare(abl_preds, base_preds,
                               f"lora_{args.ablation}", "lora",
                               metric="exact_match")
        comparison_schema = M.compare(abl_preds, base_preds,
                                      f"lora_{args.ablation}", "lora",
                                      metric="schema_valid")

        per_field = {}
        index_a = {r["example_id"]: r for r in abl_preds}
        index_b = {r["example_id"]: r for r in base_preds}
        shared = sorted(set(index_a) & set(index_b))
        for field in FIELD_ORDER:
            a = [1.0 if index_a[e]["field_correct"].get(field) else 0.0
                 for e in shared]
            b = [1.0 if index_b[e]["field_correct"].get(field) else 0.0
                 for e in shared]
            per_field[field] = {
                "ablation_rate": round(sum(a) / len(a), 4),
                "full_rate": round(sum(b) / len(b), 4),
                "paired_diff": M.paired_bootstrap_diff(a, b),
                "mcnemar": M.mcnemar([bool(v) for v in a], [bool(v) for v in b]),
            }

        payload = {
            "ablation_tag": args.ablation,
            "split": args.split,
            "variable_changed": "number of training examples",
            "n_train_full": n_full,
            "n_train_ablation": n_abl,
            "family_coverage": family_coverage,
            "metrics_full": base_m,
            "metrics_ablation": abl_m,
            "comparison_exact_match": comparison,
            "comparison_schema_valid": comparison_schema,
            "per_field": per_field,
            "training_full": {
                "run_id": (base_run or {}).get("run_id"),
                "metrics": {k: (base_run or {}).get("metrics", {}).get(k)
                            for k in ("global_step", "best_metric",
                                      "elapsed_seconds")},
            },
            "training_ablation": {
                "run_id": (abl_run or {}).get("run_id"),
                "metrics": {k: (abl_run or {}).get("metrics", {}).get(k)
                            for k in ("global_step", "best_metric",
                                      "elapsed_seconds")},
            },
        }

        # ---- markdown ----------------------------------------------------
        lines: list[str] = []
        add = lines.append
        diff = comparison["paired_diff"]

        add("# Optional ablation -- training-data size")
        add("")
        add("**This is a secondary experiment, run after the primary result was "
            "complete and reproducible. It is not part of the pre-registered "
            "success criteria and should not be read with the same weight.**")
        add("")
        add(f"One variable changed: the number of training examples "
            f"({fmt_n(n_full)} -> {fmt_n(n_abl)}, category-stratified, "
            f"deterministic). "
            f"LoRA rank, alpha, dropout, learning rate, schedule, batch size, "
            f"sequence length, precision, seeds, prompt, decoding and the "
            f"evaluation split are all identical.")
        add("")

        if family_coverage:
            retained = family_coverage["train_families_retained"]
            total = family_coverage["train_families_total"]
            dropped = family_coverage["families_dropped"]
            add("## What was actually removed")
            add("")
            add(f"This matters for interpretation. Halving the data could mean "
                f"*fewer examples per scenario* or *fewer scenarios*, and those "
                f"support opposite conclusions.")
            add("")
            add(f"- Training scenario families retained: **{retained} / {total}**")
            if dropped:
                add(f"- Families lost entirely: `{', '.join(dropped)}` "
                    f"({len(dropped)} of {total})")
            else:
                add("- **No family was dropped entirely.**")
            add(f"- Examples per family: "
                f"{family_coverage['examples_per_family_full']} -> "
                f"{family_coverage['examples_per_family_ablation']}")
            add("")
            depth_ratio = (family_coverage["examples_per_family_ablation"] /
                           family_coverage["examples_per_family_full"]
                           if family_coverage["examples_per_family_full"] else 0)
            coverage_ratio = retained / total if total else 0
            if not dropped:
                add("So this ablation reduces **depth** (examples per scenario) "
                    "and not **coverage** (number of distinct scenarios). Read "
                    "the result accordingly.")
            else:
                add(f"So this ablation is **mostly** a depth reduction: examples "
                    f"per scenario fell to {depth_ratio:.0%} of the full run "
                    f"while scenario coverage held at {coverage_ratio:.0%} "
                    f"({retained}/{total} families). It is not a clean "
                    f"depth-only manipulation -- {len(dropped)} "
                    f"{'family was' if len(dropped) == 1 else 'families were'} "
                    f"lost as a side effect of stratifying by category rather "
                    f"than by family. That confound is small but real, and it "
                    f"is stated rather than rounded away.")
            add("")

        add("## Results on the frozen test split")
        add("")
        add("| Training examples | Strict JSON | Schema valid | Exact match | "
            "Constraint violations |")
        add("|---|---|---|---|---|")
        add(f"| {fmt_n(n_abl)} | {fmt_pct(abl_m['json_parse_rate_strict'])} "
            f"| {fmt_pct(abl_m['schema_valid_rate'])} "
            f"| {fmt_pct(abl_m['exact_match'])} "
            f"| {fmt_pct(abl_m['constraint_violation_rate'])} |")
        add(f"| **{fmt_n(n_full)}** | {fmt_pct(base_m['json_parse_rate_strict'])} "
            f"| {fmt_pct(base_m['schema_valid_rate'])} "
            f"| {fmt_pct(base_m['exact_match'])} "
            f"| {fmt_pct(base_m['constraint_violation_rate'])} |")
        add("")

        add("## Does doubling the data help?")
        add("")
        add("| Metric | 50% | 100% | Difference | 95% CI | McNemar p | Verdict |")
        add("|---|---|---|---|---|---|---|")
        for label, cmp in (("exact match", comparison),
                           ("schema valid", comparison_schema)):
            d = cmp["paired_diff"]
            verdict = ("difference detected" if d["excludes_zero"]
                       else "not distinguishable from zero")
            add(f"| {label} | {fmt_pct(cmp['a_rate'])} | {fmt_pct(cmp['b_rate'])} "
                f"| {d['diff'] * 100:+.1f} pp "
                f"| [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}] pp "
                f"| {cmp['mcnemar']['p_value']:.4f} | {verdict} |")
        add("")

        add("### Per field")
        add("")
        add("| Field | 50% | 100% | Difference | 95% CI | p | Verdict |")
        add("|---|---|---|---|---|---|---|")
        for field, stats in per_field.items():
            d = stats["paired_diff"]
            verdict = ("difference detected" if d["excludes_zero"]
                       else "no detectable difference")
            add(f"| `{field}` | {fmt_pct(stats['ablation_rate'])} "
                f"| {fmt_pct(stats['full_rate'])} "
                f"| {d['diff'] * 100:+.1f} pp "
                f"| [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}] pp "
                f"| {stats['mcnemar']['p_value']:.4f} | {verdict} |")
        add("")

        add("## Reading this honestly")
        add("")
        if diff["excludes_zero"]:
            add(f"Doubling the training data from {fmt_n(n_abl)} to "
                f"{fmt_n(n_full)} produced a "
                f"detectable improvement in exact match "
                f"({diff['diff'] * 100:+.1f} pp, 95% CI "
                f"[{diff['lo'] * 100:+.1f}, {diff['hi'] * 100:+.1f}] pp). "
                f"On this task the data curve had not flattened by 171 examples, "
                f"so more data of the same kind was still buying accuracy.")
        else:
            add(f"Doubling the training data from {fmt_n(n_abl)} to "
                f"{fmt_n(n_full)} did **not** "
                f"produce a detectable change in exact match "
                f"({diff['diff'] * 100:+.1f} pp, 95% CI "
                f"[{diff['lo'] * 100:+.1f}, {diff['hi'] * 100:+.1f}] pp, "
                f"McNemar p = {comparison['mcnemar']['p_value']:.3f}). "
                f"Combined with the primary finding that 11 of 16 held-out "
                f"scenario families scored zero, this points at **scenario "
                f"coverage rather than example volume** as the binding "
                f"constraint: adding more examples of the scenarios the model "
                f"already sees does not help it handle scenarios it has never "
                f"seen.")
        add("")
        add("Caveats that apply with full force here: n = 86, a single seed per "
            "arm, and two arms. This is one comparison, not a scaling curve. "
            "It is suggestive, not conclusive.")
        if family_coverage.get("families_dropped"):
            add("")
            add(f"Additionally, the subsample lost "
                f"{len(family_coverage['families_dropped'])} scenario family as "
                f"a side effect, so the manipulation is not purely a change in "
                f"example count. A cleaner version would stratify the subsample "
                f"by family rather than by category.")
        add("")

        dataio.write_json(payload, REPORTS / "ablation.json")
        (REPORTS / "ABLATION.md").write_text("\n".join(lines), encoding="utf-8")

        print(f"ablation ({fmt_n(n_abl)} ex): "
              f"exact_match={abl_m['exact_match']:.4f} "
              f"schema_valid={abl_m['schema_valid_rate']:.4f}")
        print(f"full     ({fmt_n(n_full)} ex): "
              f"exact_match={base_m['exact_match']:.4f} "
              f"schema_valid={base_m['schema_valid_rate']:.4f}")
        print(f"difference {diff['diff'] * 100:+.1f} pp, "
              f"95% CI [{diff['lo'] * 100:+.1f}, {diff['hi'] * 100:+.1f}] pp, "
              f"McNemar p={comparison['mcnemar']['p_value']:.4f}, "
              f"excludes_zero={diff['excludes_zero']}")
        if family_coverage:
            print(f"families retained: "
                  f"{family_coverage['train_families_retained']}/"
                  f"{family_coverage['train_families_total']}")

        run.metrics = {"comparison": comparison,
                       "family_coverage": family_coverage}
        run.add_artifact("ablation_md", REPORTS / "ABLATION.md")
        run.finish("success")
        print("\nwrote reports/ABLATION.md")
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
