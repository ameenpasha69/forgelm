"""E3 -- how much of the remaining failure is the decoder, not the model?

Five conditions on the same frozen test split, same prompt, same parser:

    1  base zero-shot          unconstrained   (v1 evidence, reused)
    2  base few-shot k=8       unconstrained   (v1 evidence, reused)
    3  base + LoRA             unconstrained   (v1 evidence, reused)
    4  base few-shot k=8       CONSTRAINED
    5  base + LoRA             CONSTRAINED

Fairness rule, fixed before the runs: the identical constraint mechanism is
applied to 4 and 5. The quantity of interest is **4 vs 5** -- LoRA is never
credited with an improvement the decoder produced.

Metrics are separated so the decoder's contribution cannot masquerade as the
model's: syntactic validity, schema validity, semantic field correctness and
exact match are reported apart. Constraining makes the first two trivially
100%; only the last two say anything about the model.

    python scripts/v2_06_constrained_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run  # noqa: E402
from forgelm.schema import FIELD_ORDER  # noqa: E402

REPORTS = REPO_ROOT / "reports"
OUT = REPO_ROOT / "experiments" / "v2" / "reports" / "constrained"

CONDITIONS = [
    ("zeroshot", "reports/predictions/zeroshot_test.jsonl",
     "base zero-shot", False),
    ("fewshot", "reports/predictions/fewshot_test.jsonl",
     "base few-shot k=8", False),
    ("lora", "reports/predictions/lora_test.jsonl",
     "base + LoRA", False),
    ("fewshot_constrained",
     "reports/predictions/fewshot_constrained_test.jsonl",
     "base few-shot k=8", True),
    ("lora_constrained", "reports/predictions/lora_constrained_test.jsonl",
     "base + LoRA", True),
]


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> int:
    run = Run(kind="constrained_report").start()
    try:
        loaded: dict[str, list] = {}
        for key, rel, _, _ in CONDITIONS:
            path = REPO_ROOT / rel
            if path.exists():
                loaded[key] = dataio.read_jsonl(path)
            else:
                print(f"  (missing: {rel})")

        metrics = {k: M.compute_metrics(v) for k, v in loaded.items()}
        for key, m in metrics.items():
            print(f"  {key:22s} strict={m['json_parse_rate_strict']:.3f} "
                  f"schema={m['schema_valid_rate']:.3f} "
                  f"exact={m['exact_match']:.4f}")

        # The decisive, fair comparison.
        fair = None
        if "fewshot_constrained" in loaded and "lora_constrained" in loaded:
            fair = M.compare(loaded["fewshot_constrained"],
                             loaded["lora_constrained"],
                             "fewshot_constrained", "lora_constrained",
                             metric="exact_match")

        # What constraining buys each model on its own.
        gains = {}
        for base, constrained in (("fewshot", "fewshot_constrained"),
                                  ("lora", "lora_constrained")):
            if base in loaded and constrained in loaded:
                gains[base] = M.compare(loaded[base], loaded[constrained],
                                        base, constrained,
                                        metric="exact_match")

        # The unfair comparison, computed only so it can be shown and rejected.
        unfair = None
        if "fewshot" in loaded and "lora_constrained" in loaded:
            unfair = M.compare(loaded["fewshot"], loaded["lora_constrained"],
                               "fewshot_unconstrained", "lora_constrained",
                               metric="exact_match")

        invalid_enum = {}
        for key, preds in loaded.items():
            invalid_enum[key] = round(sum(
                1 for r in preds
                if any(v.startswith("invalid_enum")
                       for v in r.get("schema_violations", []))) / len(preds), 4)

        payload = {
            "experiment": "E3 constrained decoding",
            "metrics": {k: {kk: vv for kk, vv in m.items()
                            if kk not in ("confusion_category",
                                          "confusion_priority", "intervals")}
                        for k, m in metrics.items()},
            "invalid_enum_rate": invalid_enum,
            "fair_comparison_constrained_fewshot_vs_constrained_lora": fair,
            "gain_from_constraining": gains,
            "unfair_comparison_shown_and_rejected": unfair,
        }

        L: list[str] = []
        add = L.append
        add("# E3 -- constrained decoding")
        add("")
        add("v1 left 18 of 86 outputs containing an enum value that does not "
            "exist -- `\"dns\"`, `\"internet\"`, `\"display\"`, `\"audio\"` -- "
            "almost always a noun copied from the ticket. A decoder that cannot "
            "emit an illegal token makes that failure class impossible. This "
            "measures how much of the remaining gap that removes, and how much "
            "it does not.")
        add("")

        add("## Separated metrics")
        add("")
        add("Constraining makes syntactic and schema validity trivially 100%. "
            "Those columns therefore say nothing about the model once "
            "constrained; only the last two do.")
        add("")
        add("| # | Condition | Decoding | Syntactic (strict JSON) | "
            "Schema valid | Invalid enum | Mean field acc | **Exact match** |")
        add("|---|---|---|---|---|---|---|---|")
        for i, (key, _, label, is_constrained) in enumerate(CONDITIONS, 1):
            if key not in metrics:
                continue
            m = metrics[key]
            add(f"| {i} | {label} | "
                f"{'**constrained**' if is_constrained else 'unconstrained'} "
                f"| {fmt_pct(m['json_parse_rate_strict'])} "
                f"| {fmt_pct(m['schema_valid_rate'])} "
                f"| {fmt_pct(invalid_enum[key])} "
                f"| {fmt_pct(m['mean_field_accuracy'])} "
                f"| **{fmt_pct(m['exact_match'])}** |")
        add("")

        if fair:
            d = fair["paired_diff"]
            add("## The fair comparison: constrained few-shot vs constrained LoRA")
            add("")
            add("Both conditions use the identical constraint mechanism, so any "
                "difference is attributable to the adapter and not the decoder.")
            add("")
            add(f"- constrained few-shot: **{fmt_pct(fair['a_rate'])}** exact match")
            add(f"- constrained LoRA: **{fmt_pct(fair['b_rate'])}** exact match")
            add(f"- difference **{d['diff']*100:+.1f} pp**, 95% CI "
                f"[{d['lo']*100:+.1f}, {d['hi']*100:+.1f}] pp, "
                f"McNemar p = {fair['mcnemar']['p_value']:.4f} -- "
                f"{'difference detected' if d['excludes_zero'] else 'not distinguishable from zero'}")
            add("")

        if gains:
            add("## What constraining buys each model")
            add("")
            add("| Model | Unconstrained | Constrained | Difference | 95% CI | p |")
            add("|---|---|---|---|---|---|")
            for base, cmp in gains.items():
                d = cmp["paired_diff"]
                add(f"| {base} | {fmt_pct(cmp['a_rate'])} "
                    f"| {fmt_pct(cmp['b_rate'])} | {d['diff']*100:+.1f} pp "
                    f"| [{d['lo']*100:+.1f}, {d['hi']*100:+.1f}] pp "
                    f"| {cmp['mcnemar']['p_value']:.4f} |")
            add("")

        if unfair and fair:
            add("## The comparison this report refuses to headline")
            add("")
            d_unfair = unfair["paired_diff"]
            d_fair = fair["paired_diff"]
            add(f"Comparing **unconstrained few-shot** against **constrained "
                f"LoRA** gives {d_unfair['diff']*100:+.1f} pp -- against "
                f"{d_fair['diff']*100:+.1f} pp for the like-for-like "
                f"comparison. The difference between those two numbers is the "
                f"decoder's contribution, and quoting the first as evidence for "
                f"the adapter would be crediting LoRA with work the grammar "
                f"did. It is computed here only so it can be shown and set "
                f"aside.")
            add("")

        add("## What this does not show")
        add("")
        add("Constrained decoding guarantees a well-formed object with legal "
            "values. It cannot make a value *correct*. Any remaining gap after "
            "constraining is a model problem, not a format problem -- and that "
            "is precisely what the exact-match column isolates.")
        add("")

        dataio.write_json(payload, OUT / "constrained.json")
        (OUT / "CONSTRAINED.md").write_text("\n".join(L), encoding="utf-8")
        run.metrics = {"fair_comparison": fair, "invalid_enum": invalid_enum}
        run.finish("success")
        print(f"\nwrote {(OUT / 'CONSTRAINED.md').relative_to(REPO_ROOT)}")
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
