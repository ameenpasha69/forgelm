"""Batched, deterministic generation.

Decoding is greedy (`do_sample=False`) in every condition. That is a deliberate
choice: sampling would add run-to-run variance that this experiment has no
budget to average away, and the task has exactly one correct answer, so there is
nothing for sampling temperature to buy.

Left padding is mandatory for batched decoder-only generation. With right
padding, shorter sequences end in pad tokens and the model continues from a pad
rather than from the real final token, which quietly degrades the baseline.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .parsing import evaluate_one
from .prompts import render_prompt


def _oom_safe_batches(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def generate_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int) -> list[dict[str, Any]]:
    """Greedy-decode one batch, returning text plus a finish reason."""
    import torch

    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=False)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        tokenizer.padding_side = previous_side

    results = []
    for row in output:
        new_tokens = row[prompt_len:]
        # Trim at the first eos so the decoded text excludes trailing padding.
        eos_positions = (new_tokens == tokenizer.eos_token_id).nonzero()
        if len(eos_positions):
            cut = int(eos_positions[0].item())
            emitted = new_tokens[:cut]
            finish_reason = "stop"
        else:
            emitted = new_tokens
            finish_reason = "length" if len(new_tokens) >= max_new_tokens else "stop"
        text = tokenizer.decode(emitted, skip_special_tokens=True)
        results.append({
            "text": text,
            "finish_reason": finish_reason,
            "n_generated_tokens": int(len(emitted)),
        })
    return results


def run_evaluation(model, tokenizer, records: list[dict[str, Any]],
                   demonstrations: list[dict[str, Any]] | None = None,
                   max_new_tokens: int = 96,
                   batch_size: int = 8,
                   progress: Callable[[int, int], None] | None = None
                   ) -> list[dict[str, Any]]:
    """Evaluate a model on a list of dataset records.

    Returns one fully self-describing record per example: the rendered prompt,
    the raw response, the parsed object, the expected object, per-field
    correctness and an error category. Every metric in the report is derived
    from these records, so they -- not the metrics -- are the primary artefact.
    """
    import torch

    model.eval()
    outputs: list[dict[str, Any]] = []
    done = 0

    for chunk in _oom_safe_batches(records, batch_size):
        prompts = [render_prompt(tokenizer, r["ticket_text"], demonstrations)
                   for r in chunk]

        start = time.perf_counter()
        try:
            generations = generate_batch(model, tokenizer, prompts, max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            # Fall back to one-at-a-time rather than losing the whole run.
            torch.cuda.empty_cache()
            generations = []
            for p in prompts:
                generations.extend(
                    generate_batch(model, tokenizer, [p], max_new_tokens))
        elapsed = time.perf_counter() - start
        per_example_latency = elapsed / len(chunk)

        for record, prompt, gen in zip(chunk, prompts, generations):
            evaluated = evaluate_one(
                gen["text"], record["expected_output"],
                finish_reason=gen["finish_reason"],
            )
            outputs.append({
                "example_id": record["example_id"],
                "scenario_family": record["scenario_family"],
                "template_family": record["template_family"],
                "ticket_text": record["ticket_text"],
                "prompt": prompt,
                "finish_reason": gen["finish_reason"],
                "n_generated_tokens": gen["n_generated_tokens"],
                "latency_seconds": round(per_example_latency, 4),
                **evaluated,
            })

        done += len(chunk)
        if progress:
            progress(done, len(records))

    return outputs
