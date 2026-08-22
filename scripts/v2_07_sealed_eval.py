"""E5 -- the single evaluation on the sealed v2 test split.

This is the one look. The seal policy fixed in the pre-registration is that v2
test predictions are not inspected until configurations, seeds, prompts,
decoding and checkpoint-selection rules are frozen and committed. They are: v2
introduces no new training configuration, and every model evaluated here was
trained on v1 data before the v2 catalogue existed.

That last point is what makes this evaluation strong. The v2 families are not
merely held out -- they did not exist when these adapters were trained.

    python scripts/v2_07_sealed_eval.py --condition zeroshot
    python scripts/v2_07_sealed_eval.py --condition fewshot
    python scripts/v2_07_sealed_eval.py --condition lora --adapter artifacts/lora_adapter
    python scripts/v2_07_sealed_eval.py --report

Outputs experiments/v2/reports/sealed/*.jsonl and SEALED.md
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M  # noqa: E402
from forgelm.config import DECODING  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run  # noqa: E402
from forgelm.schema import CATEGORIES, FIELD_ORDER, PRIORITIES  # noqa: E402

V2 = REPO_ROOT / "experiments" / "v2"
OUT = V2 / "reports" / "sealed"

CONDITIONS = ("zeroshot", "fewshot", "lora")


def load_sealed():
    from forgelm.splits_v2 import (apply_split_v2, load_manifest_v2,
                                   sealed_membership_checksum)

    records = dataio.read_jsonl(V2 / "data" / "tickets_v2.jsonl")
    manifest = load_manifest_v2(V2 / "data" / "split_manifest_v2.json")

    # Re-derive the seal independently of the stored value.
    recomputed = sealed_membership_checksum(manifest["example_split"])
    if recomputed != manifest["test_membership_checksum"]:
        raise RuntimeError("sealed membership changed; results not comparable")

    sealed = apply_split_v2(records, manifest)["test"]
    return records, manifest, sealed


def evaluate(condition: str, adapter: str | None, batch_size: int,
             constrained: bool) -> int:
    from forgelm.generate import run_evaluation
    from forgelm.modeling import (BASE_MODEL_FACTS, load_adapted_model,
                                  load_base_model, load_tokenizer)
    from forgelm.prompts import (FEWSHOT_K, SYSTEM_PROMPT,
                                 select_demonstrations)
    from forgelm.seeding import SEEDS, seed_everything
    from forgelm.splits import apply_split, load_manifest
    from forgelm.splits_v2 import assert_not_sealed

    run = Run(kind=f"sealed_eval_{condition}", seeds=dict(SEEDS)).start()
    try:
        seed_everything(SEEDS["training"])
        records, manifest, sealed = load_sealed()
        print(f"sealed v2 test: {len(sealed)} examples, "
              f"{len({r['scenario_family'] for r in sealed})} families")
        print(f"seal checksum : {manifest['test_membership_checksum'][:32]}...")

        tokenizer = load_tokenizer()
        demonstrations = None
        demo_ids: list[str] = []
        if condition == "fewshot":
            # Demonstrations come from the v1 TRAIN split, never from v2 at all.
            v1 = dataio.read_jsonl(dataio.PROCESSED_DATASET)
            v1_manifest = load_manifest(dataio.SPLIT_MANIFEST)
            demonstrations = select_demonstrations(
                apply_split(v1, v1_manifest)["train"], k=FEWSHOT_K)
            demo_ids = [d["example_id"] for d in demonstrations]
            # And prove it: none of them may be a sealed v2 example.
            assert_not_sealed(demo_ids, manifest, "few-shot demonstrations")
            print(f"demonstrations (from v1 train): {demo_ids}")

        if condition == "lora":
            model, verification = load_adapted_model(adapter)
            print(f"adapter active: {verification['n_nonzero_lora_B_tensors']}"
                  f"/{verification['n_lora_B_tensors']} lora_B non-zero")
        else:
            model = load_base_model()
            verification = None

        predictions = run_evaluation(
            model, tokenizer, sealed, demonstrations=demonstrations,
            max_new_tokens=DECODING["max_new_tokens"],
            batch_size=batch_size, constrained=constrained,
            progress=lambda d, t: print(f"  {d}/{t}", end="\r", flush=True))
        print()

        name = condition + ("_constrained" if constrained else "")
        metrics = M.compute_metrics(predictions)
        dataio.write_jsonl(predictions, OUT / f"{name}_v2test.jsonl")
        dataio.write_json({
            "condition": name,
            "split": "v2 sealed test",
            "run_id": run.run_id,
            "seal_checksum": manifest["test_membership_checksum"],
            "base_model": BASE_MODEL_FACTS,
            "system_prompt_sha": hashlib.sha256(
                SYSTEM_PROMPT.encode()).hexdigest()[:16],
            "adapter_dir": adapter,
            "adapter_verification": verification,
            "fewshot_demo_ids": demo_ids,
            "constrained": constrained,
            "metrics": metrics,
        }, OUT / f"{name}_v2test.json")

        for key in ("json_parse_rate_strict", "schema_valid_rate",
                    "exact_match", "constraint_violation_rate"):
            print(f"  {key:28s} {metrics[key]:.4f}")
        print(f"  field accuracy: {metrics['field_accuracy']}")

        run.metrics = {k: v for k, v in metrics.items()
                       if k not in ("confusion_category", "confusion_priority",
                                    "intervals")}
        run.finish("success")
        return 0
    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


def report() -> int:
    run = Run(kind="sealed_report").start()
    try:
        _, manifest, sealed = load_sealed()
        loaded = {}
        for name in ("zeroshot", "fewshot", "lora",
                     "fewshot_constrained", "lora_constrained"):
            path = OUT / f"{name}_v2test.jsonl"
            if path.exists():
                loaded[name] = dataio.read_jsonl(path)
        if not loaded:
            raise SystemExit("no sealed predictions; run --condition first")

        metrics = {k: M.compute_metrics(v) for k, v in loaded.items()}

        comparisons = {}
        for a, b in (("zeroshot", "lora"), ("fewshot", "lora"),
                     ("fewshot_constrained", "lora_constrained")):
            if a in loaded and b in loaded:
                comparisons[f"{a}_vs_{b}"] = M.compare(
                    loaded[a], loaded[b], a, b, metric="exact_match")

        families = {}
        if "lora" in loaded:
            fam = defaultdict(list)
            for row in loaded["lora"]:
                fam[row["scenario_family"]].append(row)
            families = {f: {"n": len(rows),
                            "n_correct": sum(1 for r in rows if r["exact_match"])}
                        for f, rows in sorted(fam.items())}

        L: list[str] = []
        add = L.append
        add("# E5 -- the sealed v2 test evaluation")
        add("")
        add(f"**{len(sealed)} examples, "
            f"{len({r['scenario_family'] for r in sealed})} scenario families, "
            f"evaluated once.**")
        add("")
        add(f"Seal checksum `{manifest['test_membership_checksum'][:32]}...`, "
            f"re-derived at evaluation time rather than trusted.")
        add("")
        add("Every model here was trained on v1 data **before the v2 catalogue "
            "existed**. These families are not merely held out -- they were not "
            "available to be trained on. Maximum similarity between any v2 "
            "ticket and any v1 ticket is 0.3313.")
        add("")

        add("## Results")
        add("")
        add("| Condition | Strict JSON | Schema valid | **Exact match** | "
            "Mean field acc |")
        add("|---|---|---|---|---|")
        for name, m in metrics.items():
            add(f"| {name} | {m['json_parse_rate_strict']*100:.1f}% "
                f"| {m['schema_valid_rate']*100:.1f}% "
                f"| **{m['exact_match']*100:.1f}%** "
                f"| {m['mean_field_accuracy']*100:.1f}% |")
        add("")

        add("### Per field")
        add("")
        add("| Condition | " + " | ".join(f"`{f}`" for f in FIELD_ORDER) + " |")
        add("|---" * (len(FIELD_ORDER) + 1) + "|")
        for name, m in metrics.items():
            cells = " | ".join(f"{m['field_accuracy'][f]*100:.1f}%"
                               for f in FIELD_ORDER)
            add(f"| {name} | {cells} |")
        add("")

        if comparisons:
            add("## Paired comparisons")
            add("")
            add("| Comparison | A | B | Difference | 95% CI | McNemar p | "
                "Verdict |")
            add("|---|---|---|---|---|---|---|")
            for cmp in comparisons.values():
                d = cmp["paired_diff"]
                add(f"| {cmp['system_a']} -> {cmp['system_b']} "
                    f"| {cmp['a_rate']*100:.1f}% | {cmp['b_rate']*100:.1f}% "
                    f"| {d['diff']*100:+.1f} pp "
                    f"| [{d['lo']*100:+.1f}, {d['hi']*100:+.1f}] pp "
                    f"| {cmp['mcnemar']['p_value']:.4f} "
                    f"| {'difference detected' if d['excludes_zero'] else 'not distinguishable from zero'} |")
            add("")

        if families:
            zero = [f for f, s in families.items() if s["n_correct"] == 0]
            add("## Generalisation on families that did not exist at training time")
            add("")
            add(f"**{len(zero)} of {len(families)} sealed families produced zero "
                f"fully correct outputs.**")
            add("")
            add("| Family | correct / n |")
            add("|---|---|")
            for f, s in sorted(families.items(),
                               key=lambda kv: -kv[1]["n_correct"]):
                add(f"| `{f}` | {s['n_correct']} / {s['n']} |")
            add("")

        add("## Priority balance, which v1 could not measure")
        add("")
        add("v1's test split held 3 `medium` examples, so priority macro-F1 on "
            "it was close to meaningless. The sealed v2 split holds 17.")
        add("")
        if "lora" in metrics:
            m = metrics["lora"]
            add(f"- category macro-F1: {m['category']['macro_f1']:.3f}")
            add(f"- priority macro-F1: {m['priority']['macro_f1']:.3f}")
            add("")

        dataio.write_json({"metrics": metrics, "comparisons": comparisons,
                           "by_family": families,
                           "seal_checksum": manifest["test_membership_checksum"]},
                          OUT / "sealed.json")
        (OUT / "SEALED.md").write_text("\n".join(L), encoding="utf-8")
        run.finish("success")
        print(f"wrote {(OUT / 'SEALED.md').relative_to(REPO_ROOT)}")
        return 0
    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS)
    ap.add_argument("--adapter", default="artifacts/lora_adapter")
    ap.add_argument("--batch-size", type=int, default=DECODING["batch_size"])
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        return report()
    if not args.condition:
        ap.error("choose --condition or --report")
    if args.condition == "lora" and not args.adapter:
        ap.error("--adapter required for lora")
    return evaluate(args.condition, args.adapter, args.batch_size,
                    args.constrained)


if __name__ == "__main__":
    raise SystemExit(main())
