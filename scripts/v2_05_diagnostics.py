"""E4 -- build and evaluate the diagnostic suites.

Diagnostics are secondary. They are reported separately from the primary
benchmark and are never pooled into it.

    python scripts/v2_05_diagnostics.py --build              # CPU only
    python scripts/v2_05_diagnostics.py --evaluate --adapter artifacts/lora_adapter
    python scripts/v2_05_diagnostics.py --report

Interpretation rule, fixed in the pre-registration: a drop on a suite means the
model handles that specific synthetic perturbation worse. It is not evidence
about safety, robustness, or real-world reliability, and this script does not
phrase it as such.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M  # noqa: E402
from forgelm.config import DECODING  # noqa: E402
from forgelm.diagnostics import (  # noqa: E402
    SCORING_MODE, SUITE_ORDER, SUITE_QUESTION, build_suites,
)
from forgelm.ledger import REPO_ROOT, Run  # noqa: E402
from forgelm.schema import CATEGORIES, FIELD_ORDER, PRIORITIES  # noqa: E402

V2 = REPO_ROOT / "experiments" / "v2"
SUITE_DIR = V2 / "data" / "diagnostics"
OUT = V2 / "reports" / "diagnostics"


def build() -> int:
    from forgelm.splits_v2 import apply_split_v2, load_manifest_v2

    records = dataio.read_jsonl(V2 / "data" / "tickets_v2.jsonl")
    manifest = load_manifest_v2(V2 / "data" / "split_manifest_v2.json")
    by_split = apply_split_v2(records, manifest)

    # Deliberately NOT the sealed test split.
    base = by_split["train"] + by_split["validation"]
    base.sort(key=lambda r: r["example_id"])
    print(f"base for diagnostics: {len(base)} examples from v2 train+validation "
          f"(the v2 test split stays sealed)")

    suites = build_suites(base)
    total = 0
    for name in SUITE_ORDER:
        rows = suites[name]
        dataio.write_jsonl(rows, SUITE_DIR / f"{name}.jsonl")
        total += len(rows)
        print(f"  {name:22s} n={len(rows):4d}  scoring={SCORING_MODE[name]}")
    print(f"total diagnostic examples: {total}")
    return 0


def evaluate(adapter: str | None, condition: str, tag: str,
             batch_size: int, limit: int | None) -> int:
    from forgelm.generate import run_evaluation
    from forgelm.modeling import (load_adapted_model, load_base_model,
                                  load_tokenizer)
    from forgelm.prompts import FEWSHOT_K, select_demonstrations
    from forgelm.seeding import SEEDS, seed_everything
    from forgelm.splits import apply_split, load_manifest

    run = Run(kind=f"diagnostics_{condition}", seeds=dict(SEEDS)).start()
    try:
        seed_everything(SEEDS["training"])
        tokenizer = load_tokenizer()

        demonstrations = None
        if condition == "fewshot":
            v1 = dataio.read_jsonl(dataio.PROCESSED_DATASET)
            v1_manifest = load_manifest(dataio.SPLIT_MANIFEST)
            demonstrations = select_demonstrations(
                apply_split(v1, v1_manifest)["train"], k=FEWSHOT_K)

        if condition == "lora":
            model, verification = load_adapted_model(adapter)
            print(f"adapter active: {verification['n_nonzero_lora_B_tensors']}"
                  f"/{verification['n_lora_B_tensors']} lora_B non-zero")
        else:
            model = load_base_model()
            verification = None

        summary = {}
        for name in SUITE_ORDER:
            path = SUITE_DIR / f"{name}.jsonl"
            if not path.exists():
                print(f"  (missing suite {name}; run --build first)")
                continue
            rows = dataio.read_jsonl(path)
            if limit:
                rows = rows[:limit]
            print(f"  {name} (n={len(rows)})...", end=" ", flush=True)
            preds = run_evaluation(
                model, tokenizer, rows, demonstrations=demonstrations,
                max_new_tokens=DECODING["max_new_tokens"],
                batch_size=batch_size)
            for row, source in zip(preds, rows):
                row["diagnostic_suite"] = name
                row["scoring_mode"] = source["scoring_mode"]
                row["unknowable_fields"] = source.get("unknowable_fields", [])
            dataio.write_jsonl(preds, OUT / f"{tag}_{name}.jsonl")
            metrics = M.compute_metrics(preds)
            summary[name] = {"n": len(preds),
                             "exact_match": metrics["exact_match"],
                             "schema_valid_rate": metrics["schema_valid_rate"]}
            print(f"schema_valid={metrics['schema_valid_rate']:.3f} "
                  f"exact={metrics['exact_match']:.3f}")

        run.metrics = summary
        run.finish("success")
        return 0
    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


def _score(preds: list[dict], mode: str) -> dict:
    """Suite-aware scoring. Only report what the suite can legitimately support."""
    n = len(preds)
    out: dict = {"n": n, "scoring_mode": mode}
    rate = lambda key: round(sum(1 for r in preds if r[key]) / n, 4)  # noqa: E731

    out["schema_valid_rate"] = rate("schema_valid")
    out["json_parse_rate_strict"] = rate("strict_json")
    out["constraint_violation_rate"] = rate("constraint_violation")
    out["invalid_enum_rate"] = round(
        sum(1 for r in preds
            if any(v.startswith("invalid_enum")
                   for v in r.get("schema_violations", []))) / n, 4)
    out["wrong_values_only_rate"] = round(
        sum(1 for r in preds if r["error_category"] == "wrong_values_only") / n, 4)

    if mode == "schema_only":
        out["note"] = ("input is ambiguous or not a helpdesk ticket, so there "
                       "is no correct answer; only format compliance is scored")
        return out

    scored_fields = [f for f in FIELD_ORDER]
    if mode == "except_users":
        scored_fields = [f for f in FIELD_ORDER if f != "users_affected"]
        out["note"] = ("users_affected was removed from the ticket text, so it "
                       "is unknowable and excluded from scoring")

    out["field_accuracy"] = {
        f: round(sum(1 for r in preds if r["field_correct"].get(f)) / n, 4)
        for f in scored_fields
    }
    out["mean_field_accuracy"] = round(
        sum(out["field_accuracy"].values()) / len(scored_fields), 4)
    out["exact_match_scored_fields"] = round(
        sum(1 for r in preds
            if all(r["field_correct"].get(f) for f in scored_fields)) / n, 4)
    if mode == "full":
        out["exact_match"] = rate("exact_match")
        y_true = [r["expected"]["category"] for r in preds]
        y_pred = [r["predicted_fields"].get("category") for r in preds]
        out["category_macro_f1"] = M.macro_f1(y_true, y_pred, CATEGORIES)["macro_f1"]
        out["priority_macro_f1"] = M.macro_f1(
            [r["expected"]["priority"] for r in preds],
            [r["predicted_fields"].get("priority") for r in preds],
            PRIORITIES)["macro_f1"]

        fams: dict[str, list] = {}
        for r in preds:
            fams.setdefault(r["scenario_family"], []).append(r)
        per_family = {f: sum(1 for x in rows if x["exact_match"]) / len(rows)
                      for f, rows in fams.items()}
        out["worst_family_accuracy"] = round(min(per_family.values()), 4)
        out["n_families"] = len(per_family)
        out["n_families_zero_scoring"] = sum(1 for v in per_family.values()
                                             if v == 0)
    return out


def report(tags: list[str]) -> int:
    run = Run(kind="diagnostics_report").start()
    try:
        results: dict[str, dict] = {}
        for tag in tags:
            per_suite = {}
            for name in SUITE_ORDER:
                path = OUT / f"{tag}_{name}.jsonl"
                if not path.exists():
                    continue
                preds = dataio.read_jsonl(path)
                per_suite[name] = _score(preds, SCORING_MODE[name])
            if per_suite:
                results[tag] = per_suite

        if not results:
            raise SystemExit("no diagnostic predictions found")

        L: list[str] = []
        add = L.append
        add("# E4 -- diagnostic suites")
        add("")
        add("**Secondary. Not the primary benchmark, and never pooled with it.**")
        add("")
        add("Built from the v2 train and validation splits; the v2 test split "
            "stays sealed. Three suites destroy the ground truth on purpose, so "
            "each declares what it can legitimately be scored on.")
        add("")
        add("| Suite | Question | Scoring |")
        add("|---|---|---|")
        for name in SUITE_ORDER:
            add(f"| `{name}` | {SUITE_QUESTION[name]} | `{SCORING_MODE[name]}` |")
        add("")

        for tag, per_suite in results.items():
            add(f"## {tag}")
            add("")
            add("| Suite | n | Schema valid | Strict JSON | Invalid enum | "
                "Wrong values only | Exact match (scored fields) |")
            add("|---|---|---|---|---|---|---|")
            for name in SUITE_ORDER:
                s = per_suite.get(name)
                if not s:
                    continue
                em = s.get("exact_match_scored_fields")
                add(f"| `{name}` | {s['n']} "
                    f"| {s['schema_valid_rate']*100:.1f}% "
                    f"| {s['json_parse_rate_strict']*100:.1f}% "
                    f"| {s['invalid_enum_rate']*100:.1f}% "
                    f"| {s['wrong_values_only_rate']*100:.1f}% "
                    f"| {f'{em*100:.1f}%' if em is not None else 'n/a'} |")
            add("")

            add("### Per-field accuracy where the labels survive")
            add("")
            add("| Suite | " + " | ".join(f"`{f}`" for f in FIELD_ORDER) + " |")
            add("|---" * (len(FIELD_ORDER) + 1) + "|")
            for name in SUITE_ORDER:
                s = per_suite.get(name)
                if not s or "field_accuracy" not in s:
                    continue
                cells = " | ".join(
                    f"{s['field_accuracy'][f]*100:.1f}%"
                    if f in s["field_accuracy"] else "n/a"
                    for f in FIELD_ORDER)
                add(f"| `{name}` | {cells} |")
            add("")

            add("### Generalisation detail (label-preserving suites only)")
            add("")
            add("| Suite | category macro-F1 | priority macro-F1 | "
                "worst-family accuracy | zero-scoring families |")
            add("|---|---|---|---|---|")
            for name in SUITE_ORDER:
                s = per_suite.get(name)
                if not s or "category_macro_f1" not in s:
                    continue
                add(f"| `{name}` | {s['category_macro_f1']:.3f} "
                    f"| {s['priority_macro_f1']:.3f} "
                    f"| {s['worst_family_accuracy']*100:.1f}% "
                    f"| {s['n_families_zero_scoring']}/{s['n_families']} |")
            add("")

        add("## What these do and do not mean")
        add("")
        add("A drop on `noisy_text` means this model handles **this synthetic "
            "perturbation** worse. It is not a robustness claim. A result on "
            "`out_of_domain` shows whether the model emits confident, "
            "well-formed triage for input that deserves none -- which is a "
            "useful thing to know and still not a safety claim. Nothing here "
            "is evidence about real tickets, real users, or deployment.")
        add("")

        dataio.write_json(results, OUT / "diagnostics.json")
        (OUT / "DIAGNOSTICS.md").write_text("\n".join(L), encoding="utf-8")
        run.metrics = {"tags": list(results)}
        run.finish("success")
        print(f"wrote {(OUT / 'DIAGNOSTICS.md').relative_to(REPO_ROOT)}")
        return 0
    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--condition", default="lora",
                    choices=["zeroshot", "fewshot", "lora"])
    ap.add_argument("--adapter", default="artifacts/lora_adapter")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch-size", type=int, default=DECODING["batch_size"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tags", nargs="*", default=["lora"])
    args = ap.parse_args()

    if args.build:
        return build()
    if args.evaluate:
        return evaluate(args.adapter, args.condition,
                        args.tag or args.condition, args.batch_size, args.limit)
    if args.report:
        return report(args.tags)
    ap.error("choose --build, --evaluate or --report")


if __name__ == "__main__":
    raise SystemExit(main())
