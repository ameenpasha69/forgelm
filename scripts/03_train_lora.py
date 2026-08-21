"""Train the LoRA adapter.

Run `scripts/02_smoke_train.py` first; this script refuses to start unless the
pre-flight gates have passed, because every failure that script detects would
otherwise waste a full run and produce a plausible-looking but worthless
adapter.

Checkpoint selection uses **validation loss**, never test data. Validation loss
is a cheap proxy evaluated every epoch by the Trainer; after the best checkpoint
is restored, this script also runs a full generative evaluation on the
validation split so that the selected checkpoint is confirmed against the actual
task metric. The frozen test split is not touched anywhere in this file.

    python scripts/03_train_lora.py
    python scripts/03_train_lora.py --config configs/ablation_r8.json --tag r8

Outputs
    artifacts/lora_adapter[_tag]/     adapter weights + config + tokenizer
    reports/training_history[_tag].json
    reports/metrics/lora[_tag]_validation.json
    runs/<run_id>/run.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M, training as T  # noqa: E402
from forgelm.config import DECODING, SUCCESS_CRITERIA, TRAINING  # noqa: E402
from forgelm.generate import run_evaluation  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run  # noqa: E402
from forgelm.modeling import (  # noqa: E402
    BASE_MODEL_FACTS, BASE_MODEL_ID, BASE_MODEL_REVISION, LORA_TARGET_MODULES,
    build_lora_model, load_adapted_model, load_base_model, load_tokenizer,
    parameter_report, select_precision,
)
from forgelm.prompts import PROMPT_VERSION  # noqa: E402
from forgelm.seeding import SEEDS, seed_everything  # noqa: E402
from forgelm.splits import apply_split, load_manifest  # noqa: E402

REPORTS = REPO_ROOT / "reports"
ARTIFACTS = REPO_ROOT / "artifacts"


def require_smoke_pass(force: bool) -> dict:
    path = REPORTS / "smoke_train.json"
    if not path.exists():
        if force:
            return {"skipped": True, "reason": "--force given, no smoke report"}
        raise SystemExit(
            "reports/smoke_train.json not found. Run scripts/02_smoke_train.py "
            "first (or pass --force to override, which is not recommended)."
        )
    report = dataio.read_json(path)
    if not report.get("passed") and not force:
        raise SystemExit(
            f"pre-flight gates did not pass: {report.get('failures')}. "
            f"Fix them, or pass --force to train anyway."
        )
    return {"skipped": False, "run_id": report.get("run_id"),
            "passed": report.get("passed")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="JSON file with training overrides (for ablations)")
    ap.add_argument("--tag", default=None,
                    help="suffix for output paths, e.g. 'r8'")
    ap.add_argument("--force", action="store_true",
                    help="train even if pre-flight gates did not pass")
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="use only this fraction of the training split "
                         "(category-stratified, deterministic). For the "
                         "training-data-size ablation.")
    args = ap.parse_args()

    if not 0.0 < args.train_fraction <= 1.0:
        ap.error("--train-fraction must be in (0, 1]")

    suffix = f"_{args.tag}" if args.tag else ""
    adapter_dir = ARTIFACTS / f"lora_adapter{suffix}"

    smoke = require_smoke_pass(args.force)
    print(f"pre-flight: {smoke}")

    run = Run(kind="train_lora", seeds=dict(SEEDS)).start()
    try:
        precision = select_precision()
        run.seeds["applied"] = seed_everything(SEEDS["training"])

        overrides = dataio.read_json(args.config) if args.config else {}
        config = {**TRAINING, **overrides,
                  "fp16": precision["fp16"], "bf16": precision["bf16"]}

        # ---- data --------------------------------------------------------
        records = dataio.read_jsonl(dataio.PROCESSED_DATASET)
        manifest = load_manifest(dataio.SPLIT_MANIFEST)
        by_split = apply_split(records, manifest)
        train_records = by_split["train"]
        val_records = by_split["validation"]

        # Training-data-size ablation. Subsampling is category-stratified and
        # deterministic so the smaller run differs from the full run in exactly
        # one variable: how many examples it saw.
        subsample = None
        if args.train_fraction < 1.0:
            from collections import defaultdict

            from forgelm.seeding import rng

            by_category: dict[str, list] = defaultdict(list)
            for rec in sorted(train_records, key=lambda r: r["example_id"]):
                by_category[rec["category"]].append(rec)

            kept: list = []
            for category in sorted(by_category):
                pool = list(by_category[category])
                rng("training", "subsample", category).shuffle(pool)
                take = max(1, round(len(pool) * args.train_fraction))
                kept.extend(pool[:take])
            kept.sort(key=lambda r: r["example_id"])

            subsample = {
                "fraction": args.train_fraction,
                "n_before": len(train_records),
                "n_after": len(kept),
                "stratified_by": "category",
                "example_ids": [r["example_id"] for r in kept],
            }
            train_records = kept
            print(f"training-data-size ablation: using {len(kept)}/"
                  f"{subsample['n_before']} training examples "
                  f"({args.train_fraction:.0%}, category-stratified)")

        tokenizer = load_tokenizer()
        tokenizer.padding_side = "right"
        train_encoded = T.build_dataset(tokenizer, train_records,
                                        config["max_seq_len"])
        val_encoded = T.build_dataset(tokenizer, val_records,
                                      config["max_seq_len"])

        trunc = T.truncation_stats(train_encoded, config["max_seq_len"])
        masking = T.verify_masking(tokenizer, train_encoded + val_encoded)
        if not masking["passed"]:
            raise RuntimeError(f"label masking invalid: {masking['problems'][:3]}")

        steps_per_epoch = max(1, len(train_encoded) // (
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]))

        run.inputs = {
            "n_train": len(train_records),
            "n_validation": len(val_records),
            "dataset_fingerprint": dataio.dataset_fingerprint(),
            "split_manifest_checksum": manifest["checksum"],
            "truncation": trunc,
            "label_masking": {k: v for k, v in masking.items() if k != "problems"},
            "smoke_gate": smoke,
            "train_subsample": subsample,
        }

        # ---- model -------------------------------------------------------
        base = load_base_model()
        model, lora_config, upcast = build_lora_model(base, config)
        params = parameter_report(model)

        effective_batch = (config["per_device_train_batch_size"]
                           * config["gradient_accumulation_steps"])
        run.config = {
            "training": config,
            "precision": precision,
            "base_model": BASE_MODEL_FACTS,
            "prompt_version": PROMPT_VERSION,
            "lora": {
                "r": config["lora_r"], "alpha": config["lora_alpha"],
                "dropout": config["lora_dropout"],
                "target_modules": config.get("target_modules",
                                             LORA_TARGET_MODULES),
                "bias": "none", "task_type": "CAUSAL_LM",
            },
            "effective_batch_size": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "planned_total_steps": steps_per_epoch * config["num_train_epochs"],
            "checkpoint_selection": {
                "metric": config["metric_for_best_model"],
                "greater_is_better": config["greater_is_better"],
                "computed_on": "validation split only",
                "rationale": (
                    "Validation loss is evaluated every epoch at negligible "
                    "cost and is monotonically related to the token-level fit "
                    "we are optimising. The selected checkpoint is then "
                    "confirmed with a full generative evaluation on the same "
                    "validation split. Test data is never consulted."
                ),
            },
            "parameters": {k: v for k, v in params.items()
                           if k != "trainable_module_names"},
            "success_criteria": SUCCESS_CRITERIA,
        }

        print(f"\ntrain={len(train_encoded)} val={len(val_encoded)} "
              f"effective_batch={effective_batch} "
              f"steps/epoch={steps_per_epoch} "
              f"epochs={config['num_train_epochs']}")
        print(f"trainable {params['trainable_params']:,}/{params['total_params']:,} "
              f"({params['trainable_percent']}%), {upcast} tensors upcast to fp32")
        print(f"precision: {precision['dtype']} ({precision['reason'][:70]}...)\n")

        # ---- train -------------------------------------------------------
        output_dir = run.dir / "checkpoints"
        trainer, applied = T.build_trainer(
            model, tokenizer, train_encoded, val_encoded,
            str(output_dir), config, seed=SEEDS["training"])
        if applied["dropped"]:
            run.warn(f"TrainingArguments dropped: {applied['dropped']}")
        run.config["training_arguments_applied"] = applied

        t0 = time.time()
        result = trainer.train()
        elapsed = time.time() - t0

        history = trainer.state.log_history
        train_losses = [(h.get("epoch"), h["loss"]) for h in history if "loss" in h]
        eval_losses = [(h.get("epoch"), h["eval_loss"])
                       for h in history if "eval_loss" in h]

        print(f"\ntraining finished in {elapsed:.1f}s, "
              f"{result.global_step} optimiser steps")
        print("epoch  train_loss  eval_loss")
        eval_by_epoch = {round(e, 3): v for e, v in eval_losses}
        for epoch, loss in train_losses[-40:]:
            ev = eval_by_epoch.get(round(epoch, 3))
            print(f"  {epoch:5.2f}  {loss:10.4f}  "
                  f"{('%.4f' % ev) if ev is not None else '':>9s}")

        best_ckpt = trainer.state.best_model_checkpoint
        best_metric = trainer.state.best_metric
        print(f"\nbest checkpoint: {best_ckpt}")
        print(f"best {config['metric_for_best_model']}: {best_metric}")

        # ---- save adapter --------------------------------------------------
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        adapter_files = sorted(
            (p.name, p.stat().st_size) for p in adapter_dir.iterdir() if p.is_file())
        adapter_bytes = sum(size for _, size in adapter_files)
        print(f"adapter saved to {adapter_dir.relative_to(REPO_ROOT)} "
              f"({adapter_bytes/1e6:.1f} MB)")

        # Provenance next to the weights, so the adapter is self-describing
        # even if it is copied out of the repository.
        dataio.write_json({
            "experiment_id": run.config["training"].get("experiment_id",
                                                        "forgelm-lora-ticket-triage-v1"),
            "run_id": run.run_id,
            "base_model_id": BASE_MODEL_ID,
            "base_model_revision": BASE_MODEL_REVISION,
            "base_model_licence": BASE_MODEL_FACTS["licence"],
            "prompt_version": PROMPT_VERSION,
            "dataset_fingerprint": dataio.dataset_fingerprint(),
            "split_manifest_checksum": manifest["checksum"],
            "lora": run.config["lora"],
            "selected_by": config["metric_for_best_model"],
            "best_metric": best_metric,
            "best_checkpoint": str(best_ckpt),
            "seeds": dict(SEEDS),
        }, adapter_dir / "forgelm_provenance.json")

        history_payload = {
            "run_id": run.run_id,
            "config": run.config,
            "log_history": history,
            "train_losses": train_losses,
            "eval_losses": eval_losses,
            "best_checkpoint": str(best_ckpt),
            "best_metric": best_metric,
            "global_step": result.global_step,
            "elapsed_seconds": round(elapsed, 2),
            "train_metrics": {k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in result.metrics.items()},
        }
        dataio.write_json(history_payload, REPORTS / f"training_history{suffix}.json")

        # ---- verify reload through a clean path -----------------------------
        del trainer, model, base
        import torch
        torch.cuda.empty_cache()

        reloaded, verification = load_adapted_model(str(adapter_dir))
        print(f"\nadapter reload verified: "
              f"{verification['n_nonzero_lora_tensors']}/"
              f"{verification['n_lora_tensors']} LoRA tensors non-zero, "
              f"max|W|={verification['max_abs_lora_weight']:.5f}")

        # ---- confirm the selected checkpoint on VALIDATION -------------------
        print("\nconfirming selected checkpoint on the validation split "
              "(generative evaluation)...")
        val_predictions = run_evaluation(
            reloaded, tokenizer, val_records,
            demonstrations=None,
            max_new_tokens=DECODING["max_new_tokens"],
            batch_size=DECODING["batch_size"],
        )
        val_metrics = M.compute_metrics(val_predictions)
        name = f"lora{suffix}"
        dataio.write_jsonl(val_predictions,
                           REPORTS / "predictions" / f"{name}_validation.jsonl")
        dataio.write_json({"condition": name, "split": "validation",
                           "run_id": run.run_id, "metrics": val_metrics},
                          REPORTS / "metrics" / f"{name}_validation.json")

        print(f"validation exact_match={val_metrics['exact_match']:.4f} "
              f"schema_valid={val_metrics['schema_valid_rate']:.4f} "
              f"strict_json={val_metrics['json_parse_rate_strict']:.4f}")

        run.metrics = {
            "global_step": result.global_step,
            "elapsed_seconds": round(elapsed, 2),
            "best_metric": best_metric,
            "best_checkpoint": str(best_ckpt),
            "final_train_loss": train_losses[-1][1] if train_losses else None,
            "eval_losses": eval_losses,
            "adapter_bytes": adapter_bytes,
            "adapter_files": adapter_files,
            "adapter_verification": verification,
            "validation_metrics": {
                k: v for k, v in val_metrics.items()
                if k not in ("confusion_category", "confusion_priority")},
        }
        run.add_artifact("adapter", adapter_dir)
        run.add_artifact("training_history", REPORTS / f"training_history{suffix}.json")
        run.finish("success")

        # Checkpoints are large and fully reproducible; the selected adapter is
        # already saved separately.
        shutil.rmtree(output_dir, ignore_errors=True)

        print(f"\nrun_id: {run.run_id}")
        print(f"next: python scripts/01_evaluate.py --condition lora "
              f"--split test --adapter {adapter_dir.relative_to(REPO_ROOT)}")
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
