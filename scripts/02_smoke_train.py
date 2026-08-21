"""Pre-flight gate. Everything that must pass BEFORE a full training run.

Full training is cheap here (minutes, not hours), but the failures this script
catches -- broken label masking, a dead gradient path, silent truncation, NaN
loss, out-of-memory -- would all produce a plausible-looking loss curve and a
worthless adapter. Each check is fast and each one answers a specific question.

    1. tokenisation        do prompt and answer tokenise the way we think?
    2. truncation          does anything get cut off at max_seq_len?
    3. label masking       is the loss computed over the answer and only the answer?
    4. one-batch forward   is the initial loss finite and in a sane range?
    5. tiny overfit        do gradients actually reach and change the adapter?
    6. smoke training      does the real Trainer loop run end to end?

The adapter produced here is deliberately thrown away: step 5 mutates the
weights, so the real run in 03_train_lora.py builds a fresh model.

    python scripts/02_smoke_train.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, training as T  # noqa: E402
from forgelm.config import TRAINING  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run  # noqa: E402
from forgelm.modeling import (  # noqa: E402
    BASE_MODEL_FACTS, LORA_TARGET_MODULES, build_lora_model, load_base_model,
    load_tokenizer, parameter_report, select_precision,
)
from forgelm.seeding import SEEDS, seed_everything  # noqa: E402
from forgelm.splits import apply_split, load_manifest  # noqa: E402

REPORTS = REPO_ROOT / "reports"


def main() -> int:
    run = Run(kind="smoke_train", seeds=dict(SEEDS)).start()
    gate_results: dict[str, object] = {}
    failures: list[str] = []

    try:
        precision = select_precision()
        run.seeds["applied"] = seed_everything(SEEDS["training"])
        config = {**TRAINING, "fp16": precision["fp16"], "bf16": precision["bf16"]}
        run.config = {"training": config, "precision": precision,
                      "base_model": BASE_MODEL_FACTS,
                      "target_modules": LORA_TARGET_MODULES}

        print(f"device={precision['device']} dtype={precision['dtype']}")
        print(f"precision rationale: {precision['reason']}\n")

        # ---- data --------------------------------------------------------
        records = dataio.read_jsonl(dataio.PROCESSED_DATASET)
        manifest = load_manifest(dataio.SPLIT_MANIFEST)
        by_split = apply_split(records, manifest)
        train_records, val_records = by_split["train"], by_split["validation"]
        run.inputs = {"n_train": len(train_records), "n_validation": len(val_records),
                      "dataset_fingerprint": dataio.dataset_fingerprint()}

        tokenizer = load_tokenizer()
        tokenizer.padding_side = "right"   # training pads on the right

        # ---- gate 1 + 2: tokenisation and truncation ---------------------
        max_seq_len = config["max_seq_len"]
        train_encoded = T.build_dataset(tokenizer, train_records, max_seq_len)
        val_encoded = T.build_dataset(tokenizer, val_records, max_seq_len)

        trunc_train = T.truncation_stats(train_encoded, max_seq_len)
        trunc_val = T.truncation_stats(val_encoded, max_seq_len)
        gate_results["truncation_train"] = trunc_train
        gate_results["truncation_validation"] = trunc_val
        print(f"[1] tokenised {len(train_encoded)} train / {len(val_encoded)} val")
        print(f"[2] truncation: train {trunc_train['n_truncated']}/"
              f"{trunc_train['n']}, val {trunc_val['n_truncated']}/{trunc_val['n']} "
              f"(max total tokens = {trunc_train['total_tokens']['max']}, "
              f"limit = {max_seq_len})")
        if trunc_train["n_truncated"] or trunc_val["n_truncated"]:
            failures.append("truncation: examples are being cut off")

        sample = T.inspect_tokenization(tokenizer, train_encoded[0])
        gate_results["tokenisation_sample"] = sample
        print(f"    sample {sample['example_id']}: "
              f"{sample['n_masked']} masked + {sample['n_unmasked']} supervised "
              f"= {sample['n_total_tokens']} tokens")
        print(f"    supervised span -> {sample['unmasked_text']!r}")
        print(f"    masked tail     -> ...{sample['masked_text_tail'][-90:]!r}")

        # ---- gate 3: label masking ---------------------------------------
        masking = T.verify_masking(tokenizer, train_encoded + val_encoded)
        gate_results["label_masking"] = masking
        print(f"[3] label masking: {'PASS' if masking['passed'] else 'FAIL'} "
              f"({masking['n_checked']} examples, {masking['n_problems']} problems)")
        if not masking["passed"]:
            failures.append(f"label masking: {masking['problems'][:3]}")

        # ---- model -------------------------------------------------------
        base = load_base_model()
        model, lora_config, upcast = build_lora_model(base, config)
        params = parameter_report(model)
        gate_results["parameters"] = {
            k: v for k, v in params.items() if k != "trainable_module_names"}
        gate_results["n_upcast_to_fp32"] = upcast
        print(f"\n    LoRA r={config['lora_r']} alpha={config['lora_alpha']} "
              f"dropout={config['lora_dropout']}")
        print(f"    trainable {params['trainable_params']:,} / "
              f"{params['total_params']:,} = {params['trainable_percent']}% "
              f"({params['n_trainable_tensors']} tensors, {upcast} upcast to fp32)")
        # Names look like
        #   base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
        # so the adapted projection is the part immediately before "lora_A"/"lora_B".
        distinct = sorted({
            parts[i - 1]
            for n in params["trainable_module_names"]
            for parts in [n.split(".")]
            for i, p in enumerate(parts)
            if p.startswith("lora_") and i > 0
        })
        print(f"    adapted projections: {distinct}")
        print(f"    trainable dtypes: {params['trainable_param_dtypes']} "
              f"| all params: {params['param_dtypes']}")
        gate_results["adapted_projections"] = distinct
        gate_results["trainable_param_dtypes"] = params["trainable_param_dtypes"]

        # ---- gate 4: one-batch forward ------------------------------------
        forward = T.one_batch_forward(model, tokenizer, train_encoded, batch_size=2)
        gate_results["one_batch_forward"] = forward
        print(f"\n[4] forward pass: loss={forward['loss']} "
              f"ppl={forward['perplexity']} shape={forward['batch_shape']} "
              f"supervised_tokens={forward['n_supervised_tokens']} "
              f"-> {'PASS' if forward['healthy'] else 'FAIL'}")
        if not forward["healthy"]:
            failures.append(f"forward pass unhealthy: loss={forward['loss']}")

        # ---- gate 5: tiny overfit -----------------------------------------
        overfit = T.tiny_overfit(model, tokenizer, train_encoded,
                                 steps=30, lr=1e-3, n_examples=4)
        gate_results["tiny_overfit"] = overfit
        print(f"[5] tiny overfit on {overfit['n_examples']} examples, "
              f"{overfit['steps']} steps: {overfit['first_loss']} -> "
              f"{overfit['final_loss']} (min {overfit['min_loss']}) "
              f"-> {'PASS' if overfit['passed'] else 'FAIL'}")
        if not overfit["passed"]:
            failures.append("tiny overfit: loss did not fall by half; the "
                            "gradient path may be broken")

        # ---- gate 6: smoke training loop ----------------------------------
        smoke_dir = run.dir / "smoke_output"
        smoke_config = {**config, "num_train_epochs": 1,
                        "eval_strategy": "no", "save_strategy": "no",
                        "load_best_model_at_end": False,
                        "early_stopping_patience": 1}
        trainer, applied = T.build_trainer(
            model, tokenizer, train_encoded[:16], val_encoded[:4],
            str(smoke_dir), smoke_config, seed=SEEDS["training"])
        gate_results["training_arguments"] = applied
        if applied["dropped"]:
            run.warn(f"TrainingArguments dropped unsupported keys: "
                     f"{applied['dropped']}")
            print(f"    note: unsupported TrainingArguments dropped: "
                  f"{applied['dropped']}")

        print("[6] smoke training (16 examples, 1 epoch)...")
        result = trainer.train()
        smoke = {"train_loss": round(float(result.training_loss), 6),
                 "steps": int(result.global_step),
                 "runtime_seconds": round(float(result.metrics.get(
                     "train_runtime", 0.0)), 2)}
        gate_results["smoke_training"] = smoke
        print(f"    train_loss={smoke['train_loss']} steps={smoke['steps']} "
              f"runtime={smoke['runtime_seconds']}s -> PASS")

        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gate_results["peak_gpu_gib"] = round(peak, 3)
            print(f"\n    peak GPU memory {peak:.2f} / {total:.2f} GiB")

        shutil.rmtree(smoke_dir, ignore_errors=True)

    except Exception as exc:  # noqa: BLE001
        run.metrics = {"gates": gate_results, "failures": failures}
        run.fail(exc)
        dataio.write_json({"gates": gate_results, "failures": failures,
                           "status": "exception", "run_id": run.run_id},
                          REPORTS / "smoke_train.json")
        raise

    passed = not failures
    run.metrics = {"gates": gate_results, "failures": failures, "passed": passed}
    run.finish("success" if passed else "failed")
    dataio.write_json({"gates": gate_results, "failures": failures,
                       "passed": passed, "run_id": run.run_id},
                      REPORTS / "smoke_train.json")

    print("\n" + "=" * 60)
    if passed:
        print("ALL PRE-FLIGHT GATES PASSED -- safe to run 03_train_lora.py")
    else:
        print("PRE-FLIGHT FAILED:")
        for f in failures:
            print(f"  - {f}")
    print(f"run_id: {run.run_id}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
