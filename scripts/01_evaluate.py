"""Evaluate one model condition on one split.

Conditions
    zeroshot  unchanged base model, fixed instruction, no demonstrations
    fewshot   unchanged base model, same instruction + K demonstrations
              drawn ONLY from the training split
    lora      base model + a saved LoRA adapter, same instruction, no demos

Usage
    python scripts/01_evaluate.py --condition zeroshot --split test
    python scripts/01_evaluate.py --condition fewshot  --split test
    python scripts/01_evaluate.py --condition lora --split test \
        --adapter artifacts/lora_adapter

Outputs
    reports/predictions/<condition>_<split>.jsonl   one record per example
    reports/metrics/<condition>_<split>.json        aggregate metrics
    runs/<run_id>/run.json                          ledger record

The prediction file is the primary artefact: every metric can be recomputed
from it by scripts/06_report.py without a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M  # noqa: E402
from forgelm.config import DECODING, SUCCESS_CRITERIA  # noqa: E402
from forgelm.generate import run_evaluation  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run  # noqa: E402
from forgelm.modeling import (  # noqa: E402
    BASE_MODEL_FACTS, BASE_MODEL_ID, BASE_MODEL_REVISION,
    load_adapted_model, load_base_model, load_tokenizer, parameter_report,
    select_precision,
)
from forgelm.prompts import (  # noqa: E402
    FEWSHOT_K, PROMPT_VERSION, SYSTEM_PROMPT, select_demonstrations,
)
from forgelm.seeding import SEEDS, seed_everything  # noqa: E402
from forgelm.splits import apply_split, load_manifest  # noqa: E402

PRED_DIR = REPO_ROOT / "reports" / "predictions"
METRIC_DIR = REPO_ROOT / "reports" / "metrics"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["zeroshot", "fewshot", "lora"])
    ap.add_argument("--split", default="test",
                    choices=["train", "validation", "test"])
    ap.add_argument("--adapter", default=None,
                    help="adapter directory (required for --condition lora)")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N examples (smoke testing)")
    ap.add_argument("--batch-size", type=int, default=DECODING["batch_size"])
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filenames, e.g. an ablation name")
    args = ap.parse_args()

    if args.condition == "lora" and not args.adapter:
        ap.error("--adapter is required when --condition lora")

    name = args.condition + (f"_{args.tag}" if args.tag else "")
    stem = f"{name}_{args.split}" + ("_smoke" if args.limit else "")

    run = Run(kind=f"eval_{args.condition}", seeds=dict(SEEDS)).start()
    try:
        seed_applied = seed_everything(SEEDS["training"])
        run.seeds["applied"] = seed_applied

        # ---- data -------------------------------------------------------
        records = dataio.read_jsonl(dataio.PROCESSED_DATASET)
        manifest = load_manifest(dataio.SPLIT_MANIFEST)   # verifies checksum
        by_split = apply_split(records, manifest)
        eval_records = by_split[args.split]
        if args.limit:
            eval_records = eval_records[:args.limit]

        run.inputs = {
            "split": args.split,
            "n_examples": len(eval_records),
            "dataset_fingerprint": dataio.dataset_fingerprint(),
            "split_manifest_checksum": manifest["checksum"],
            "limit": args.limit,
        }

        # ---- model ------------------------------------------------------
        precision = select_precision()
        tokenizer = load_tokenizer()
        adapter_verification = None

        if args.condition == "lora":
            model, adapter_verification = load_adapted_model(args.adapter)
            print(f"adapter loaded and verified active: "
                  f"{adapter_verification['n_nonzero_lora_tensors']}"
                  f"/{adapter_verification['n_lora_tensors']} LoRA tensors "
                  f"non-zero, max|W|={adapter_verification['max_abs_lora_weight']:.5f}")
        else:
            model = load_base_model()

        params = parameter_report(model)

        # ---- demonstrations (few-shot only) ------------------------------
        demonstrations = None
        demo_ids: list[str] = []
        if args.condition == "fewshot":
            demonstrations = select_demonstrations(by_split["train"], k=FEWSHOT_K)
            demo_ids = [d["example_id"] for d in demonstrations]
            # Hard guarantee: demonstrations may never come from a held-out split.
            for d in demonstrations:
                assigned = manifest["example_split"][d["example_id"]]
                if assigned != "train":
                    raise RuntimeError(
                        f"demonstration {d['example_id']} is in split "
                        f"{assigned!r}, not 'train'. That would leak held-out "
                        f"data into a base-model baseline."
                    )
            print(f"few-shot demonstrations ({len(demo_ids)}): {demo_ids}")

        run.config = {
            "condition": args.condition,
            "prompt_version": PROMPT_VERSION,
            # Recorded so a reader can prove every condition saw the identical
            # instruction, rather than taking it on trust.
            "system_prompt_sha": hashlib.sha256(
                SYSTEM_PROMPT.encode()).hexdigest()[:16],
            "decoding": {**DECODING, "batch_size": args.batch_size},
            "base_model": BASE_MODEL_FACTS,
            "precision": precision,
            "adapter_dir": args.adapter,
            "adapter_verification": adapter_verification,
            "fewshot_k": FEWSHOT_K if args.condition == "fewshot" else 0,
            "fewshot_demo_ids": demo_ids,
            "parameters": {k: v for k, v in params.items()
                           if k != "trainable_module_names"},
        }

        print(f"condition={args.condition} split={args.split} "
              f"n={len(eval_records)} device={precision['device']} "
              f"dtype={precision['dtype']}")

        # ---- generate ----------------------------------------------------
        def progress(done: int, total: int) -> None:
            print(f"  {done}/{total}", end="\r", flush=True)

        predictions = run_evaluation(
            model, tokenizer, eval_records,
            demonstrations=demonstrations,
            max_new_tokens=DECODING["max_new_tokens"],
            batch_size=args.batch_size,
            progress=progress,
        )
        print()

        # ---- score -------------------------------------------------------
        computed = M.compute_metrics(predictions)
        computed["intervals"] = M.headline_intervals(predictions)

        pred_path = PRED_DIR / f"{stem}.jsonl"
        metric_path = METRIC_DIR / f"{stem}.json"
        dataio.write_jsonl(predictions, pred_path)
        dataio.write_json({
            "condition": name,
            "split": args.split,
            "run_id": run.run_id,
            "config": run.config,
            "success_criteria": SUCCESS_CRITERIA,
            "metrics": computed,
        }, metric_path)

        run.add_artifact("predictions", pred_path)
        run.add_artifact("metrics", metric_path)
        run.metrics = {k: v for k, v in computed.items()
                       if k not in ("confusion_category", "confusion_priority",
                                    "intervals")}
        run.finish("success")

        # ---- console summary ---------------------------------------------
        print(f"\n=== {name} / {args.split} (n={computed['n']}) ===")
        for key in ("json_parse_rate_strict", "json_parse_rate_lenient",
                    "schema_valid_rate", "exact_match",
                    "constraint_violation_rate", "markdown_fence_rate",
                    "prose_outside_json_rate", "truncation_rate"):
            print(f"  {key:28s} {computed[key]:.4f}")
        print(f"  {'category_macro_f1':28s} {computed['category']['macro_f1']:.4f}")
        print(f"  {'priority_macro_f1':28s} {computed['priority']['macro_f1']:.4f}")
        print(f"  field accuracy: {computed['field_accuracy']}")
        print(f"  errors: {computed['error_categories']}")
        print(f"\nrun_id: {run.run_id}")
        print(f"predictions -> {pred_path.relative_to(REPO_ROOT)}")
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
