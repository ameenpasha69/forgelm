"""Supervised fine-tuning data pipeline and Trainer construction.

The important part of this file is **prompt masking**. A causal language model
predicts every next token, so if the prompt tokens are left in the loss the
model spends most of its gradient learning to reproduce the instruction it was
given -- which it already knows, and which is not the task. Labels for prompt
positions are set to -100 (PyTorch's ignore index) so the loss is computed only
over the JSON the model is supposed to emit.

This is easy to get wrong and impossible to notice from the loss curve alone,
so `inspect_tokenization` decodes the masked and unmasked spans back to text and
`verify_masking` asserts the invariants. Both run before training starts.
"""

from __future__ import annotations

import inspect
from typing import Any

from .prompts import render_training_text

IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def encode_example(tokenizer, record: dict[str, Any],
                   max_seq_len: int) -> dict[str, Any]:
    """Tokenise one record into input_ids / labels with the prompt masked."""
    prompt, completion = render_training_text(
        tokenizer, record["ticket_text"], record["expected_output"])

    # add_special_tokens=False throughout: the chat template has already
    # inserted every special token this model expects. Letting the tokenizer
    # add more would put a stray BOS in the middle of the sequence.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)

    truncated = len(input_ids) > max_seq_len
    if truncated:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "n_prompt_tokens": len(prompt_ids),
        "n_completion_tokens": len(completion_ids),
        "n_total_tokens": len(prompt_ids) + len(completion_ids),
        "truncated": truncated,
        "example_id": record["example_id"],
    }


def build_dataset(tokenizer, records: list[dict[str, Any]],
                  max_seq_len: int) -> list[dict[str, Any]]:
    return [encode_example(tokenizer, r, max_seq_len) for r in records]


def truncation_stats(encoded: list[dict[str, Any]],
                     max_seq_len: int) -> dict[str, Any]:
    totals = [e["n_total_tokens"] for e in encoded]
    n_trunc = sum(1 for e in encoded if e["truncated"])
    return {
        "n": len(encoded),
        "max_seq_len": max_seq_len,
        "n_truncated": n_trunc,
        "truncation_rate": round(n_trunc / len(encoded), 4) if encoded else 0.0,
        "total_tokens": {
            "min": min(totals), "max": max(totals),
            "mean": round(sum(totals) / len(totals), 1),
        },
        "prompt_tokens_mean": round(
            sum(e["n_prompt_tokens"] for e in encoded) / len(encoded), 1),
        "completion_tokens_mean": round(
            sum(e["n_completion_tokens"] for e in encoded) / len(encoded), 1),
    }


def verify_masking(tokenizer, encoded: list[dict[str, Any]]) -> dict[str, Any]:
    """Assert the loss is computed over the answer and nothing else.

    Checks, for every encoded example:
      * at least one masked position and at least one unmasked position exist;
      * the masked region is a strict prefix (no interleaving);
      * decoding the unmasked positions reproduces the JSON answer, not the
        instruction.
    """
    problems: list[str] = []
    for e in encoded:
        labels = e["labels"]
        masked = [i for i, l in enumerate(labels) if l == IGNORE_INDEX]
        unmasked = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]

        if not masked:
            problems.append(f"{e['example_id']}: nothing masked")
            continue
        if not unmasked:
            problems.append(f"{e['example_id']}: everything masked -> zero loss")
            continue
        if masked != list(range(len(masked))):
            problems.append(f"{e['example_id']}: masked region is not a prefix")
        if unmasked[0] != len(masked):
            problems.append(f"{e['example_id']}: mask/answer boundary misaligned")

        answer = tokenizer.decode([labels[i] for i in unmasked],
                                  skip_special_tokens=True)
        if "{" not in answer or "category" not in answer:
            problems.append(
                f"{e['example_id']}: unmasked span does not look like the JSON "
                f"answer: {answer[:80]!r}")

    return {
        "n_checked": len(encoded),
        "n_problems": len(problems),
        "problems": problems[:20],
        "passed": not problems,
    }


def inspect_tokenization(tokenizer, encoded: dict[str, Any]) -> dict[str, Any]:
    """Human-readable view of one encoded example."""
    labels = encoded["labels"]
    ids = encoded["input_ids"]
    masked_ids = [i for i, l in zip(ids, labels) if l == IGNORE_INDEX]
    unmasked_ids = [i for i, l in zip(ids, labels) if l != IGNORE_INDEX]
    return {
        "example_id": encoded["example_id"],
        "n_total_tokens": len(ids),
        "n_masked": len(masked_ids),
        "n_unmasked": len(unmasked_ids),
        "masked_text_tail": tokenizer.decode(masked_ids[-60:]),
        "unmasked_text": tokenizer.decode(unmasked_ids),
        "first_unmasked_tokens": [
            tokenizer.decode([t]) for t in unmasked_ids[:12]
        ],
    }


# --------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------

class CausalCollator:
    """Pad a batch to its longest member.

    Padding to the batch maximum rather than to max_seq_len matters on a 4 GiB
    card: the logits tensor is batch x seq x 151936, so every avoided pad token
    saves ~600 KB of fp32 logits.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.pad_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, labels, attention = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_id] * pad)
            labels.append(list(f["labels"]) + [IGNORE_INDEX] * pad)
            attention.append(list(f["attention_mask"]) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


# --------------------------------------------------------------------------
# Trainer construction
# --------------------------------------------------------------------------

def supported_training_arguments(requested: dict[str, Any]) -> tuple[dict, list[str]]:
    """Filter kwargs against the installed TrainingArguments signature.

    transformers renames arguments across major versions (`evaluation_strategy`
    became `eval_strategy`, `tokenizer` became `processing_class`). Rather than
    pinning to one version or guessing, we introspect and report what was
    dropped, so a silently-ignored setting cannot corrupt a run.
    """
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    valid = set(signature.parameters)

    aliases = {
        "eval_strategy": ("eval_strategy", "evaluation_strategy"),
        "evaluation_strategy": ("eval_strategy", "evaluation_strategy"),
    }

    accepted: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in requested.items():
        candidates = aliases.get(key, (key,))
        for candidate in candidates:
            if candidate in valid:
                accepted[candidate] = value
                break
        else:
            dropped.append(key)
    return accepted, dropped


def build_trainer(model, tokenizer, train_encoded, eval_encoded,
                  output_dir: str, config: dict[str, Any], seed: int,
                  data_seed: int | None = None):
    """Assemble a Trainer, returning it plus the arguments actually applied.

    `seed` drives parameter initialisation; `data_seed` drives batch ordering.
    They default to the same value (v1 behaviour) but are separable so a
    variance study can attribute run-to-run differences to one or the other.
    """
    import math

    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    valid = set(inspect.signature(TrainingArguments.__init__).parameters)

    # Warm-up. transformers 5.x removed `warmup_ratio` and keeps only
    # `warmup_steps`, so the ratio is converted against the real step count
    # rather than being silently dropped (which would start training at the
    # full learning rate on step 1).
    effective_batch = (config["per_device_train_batch_size"]
                       * config["gradient_accumulation_steps"])
    steps_per_epoch = max(1, math.ceil(len(train_encoded) / effective_batch))
    total_steps = max(1, steps_per_epoch * int(config["num_train_epochs"]))
    warmup: dict[str, Any]
    if "warmup_ratio" in valid:
        warmup = {"warmup_ratio": config["warmup_ratio"]}
    else:
        warmup = {"warmup_steps": max(1, round(
            config["warmup_ratio"] * total_steps))}

    requested = {
        "output_dir": output_dir,
        "num_train_epochs": config["num_train_epochs"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "lr_scheduler_type": config["lr_scheduler_type"],
        **warmup,
        "weight_decay": config["weight_decay"],
        "max_grad_norm": config["max_grad_norm"],
        "optim": config["optimizer"],
        "logging_steps": config["logging_steps"],
        "eval_strategy": config["eval_strategy"],
        "save_strategy": config["save_strategy"],
        "save_total_limit": config["save_total_limit"],
        "load_best_model_at_end": config["load_best_model_at_end"],
        "metric_for_best_model": config["metric_for_best_model"],
        "greater_is_better": config["greater_is_better"],
        "fp16": config["fp16"],
        "bf16": config["bf16"],
        "seed": seed,
        "data_seed": seed if data_seed is None else data_seed,
        "report_to": [],
        "remove_unused_columns": False,
        # PEFT-wrapped models confuse Trainer's automatic label detection;
        # naming the label column explicitly keeps eval_loss populated.
        "label_names": ["labels"],
        "disable_tqdm": False,
    }

    accepted, dropped = supported_training_arguments(requested)
    args = TrainingArguments(**accepted)

    # EarlyStoppingCallback asserts that evaluation is enabled; adding it with
    # eval_strategy="no" raises at on_train_begin rather than being ignored.
    callbacks = []
    if str(config["eval_strategy"]).lower() not in ("no", "none"):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config["early_stopping_patience"]))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_encoded,
        "eval_dataset": eval_encoded,
        "data_collator": CausalCollator(tokenizer),
        "callbacks": callbacks,
    }
    # transformers>=4.46 renamed Trainer(tokenizer=...) to processing_class=...
    trainer_signature = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    return trainer, {"accepted": accepted, "dropped": dropped}


# --------------------------------------------------------------------------
# Pre-flight checks
# --------------------------------------------------------------------------

def one_batch_forward(model, tokenizer, encoded: list[dict[str, Any]],
                      batch_size: int = 2) -> dict[str, Any]:
    """Run a single forward pass and confirm the loss is finite and sane.

    A randomly-initialised adapter over a trained base model should give a loss
    in roughly the 0.5-5 range for this task. A loss of 0.0, NaN or inf means
    the labels are broken -- catching that here costs seconds; catching it after
    a full training run costs the run.
    """
    import math
    import torch

    collator = CausalCollator(tokenizer)
    batch = collator(encoded[:batch_size])
    batch = {k: v.to(model.device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        output = model(**batch)
    loss = float(output.loss.item())

    n_supervised = int((batch["labels"] != IGNORE_INDEX).sum().item())
    return {
        "loss": round(loss, 6),
        "finite": math.isfinite(loss),
        "batch_shape": list(batch["input_ids"].shape),
        "n_supervised_tokens": n_supervised,
        "perplexity": round(math.exp(loss), 2) if math.isfinite(loss) and loss < 20
        else None,
        "healthy": math.isfinite(loss) and 0.0 < loss < 20.0 and n_supervised > 0,
    }


def tiny_overfit(model, tokenizer, encoded: list[dict[str, Any]],
                 steps: int = 30, lr: float = 1e-3,
                 n_examples: int = 4, micro_batch: int = 1) -> dict[str, Any]:
    """Can the model drive loss towards zero on a handful of examples?

    If it cannot overfit four examples, the gradient path is broken and full
    training is pointless. This is the cheapest possible proof that gradients
    actually reach the adapter and change the output.

    Micro-batching with gradient accumulation is not a stylistic choice: a
    single forward+backward over 4 examples of ~280 tokens allocates
    4 x 280 x 151936 logits in fp16 *and* fp32 *and* their gradient, which
    exceeds 4 GiB. Accumulating over micro-batches of 1 keeps the peak at the
    same level as real training, which is also what we want this gate to
    exercise.
    """
    import math
    import torch

    subset = encoded[:n_examples]
    collator = CausalCollator(tokenizer)
    micro_batches = [
        {k: v.to(model.device)
         for k, v in collator(subset[i:i + micro_batch]).items()}
        for i in range(0, len(subset), micro_batch)
    ]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=lr)
    use_cuda = model.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    model.train()
    losses = []
    oom = False
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        total = 0.0
        try:
            for batch in micro_batches:
                with torch.autocast(device_type=model.device.type,
                                    dtype=torch.float16, enabled=use_cuda):
                    loss = model(**batch).loss / len(micro_batches)
                scaler.scale(loss).backward()
                total += float(loss.item())
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimiser)
            scaler.update()
        except torch.OutOfMemoryError:
            oom = True
            torch.cuda.empty_cache()
            break
        losses.append(total)

    model.eval()
    del optimiser, micro_batches
    if use_cuda:
        torch.cuda.empty_cache()

    if not losses:
        return {"n_examples": n_examples, "steps": steps, "lr": lr,
                "passed": False, "oom": oom,
                "note": "no step completed"}

    first, last = losses[0], losses[-1]
    return {
        "n_examples": n_examples,
        "micro_batch": micro_batch,
        "steps": len(losses),
        "lr": lr,
        "first_loss": round(first, 6),
        "final_loss": round(last, 6),
        "min_loss": round(min(losses), 6),
        "loss_history": [round(x, 5) for x in losses],
        "all_finite": all(math.isfinite(x) for x in losses),
        "oom": oom,
        # A working gradient path should cut the loss substantially. The task
        # loss starts low (most JSON tokens are structural and already
        # predictable under teacher forcing), so the bar is a relative drop.
        "passed": (not oom and math.isfinite(last) and last < first * 0.5),
    }
