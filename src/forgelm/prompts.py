"""Prompt construction.

Fairness rule that governs this whole module: **the system prompt and the task
instruction are byte-identical across zero-shot, few-shot and fine-tuned
conditions.** The only thing that varies is whether demonstrations are present
and whether the weights have been adapted. If the instruction differed, any
measured improvement could be an artefact of better prompting rather than of
adaptation, and the experiment would answer nothing.

Everything is rendered through the tokenizer's *official* chat template
(`apply_chat_template`), never a hand-rolled string. Hand-rolling a chat format
is the single most common way to accidentally handicap a base-model baseline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import canonical_json, schema_prompt_block
from .seeding import rng

# The one instruction used everywhere. Changing this invalidates comparisons
# against previously recorded results, so it is versioned.
PROMPT_VERSION = "1.0.0"

SYSTEM_PROMPT = (
    "You are an IT service desk triage assistant. "
    "You read a helpdesk ticket and return structured triage data as JSON.\n\n"
    + schema_prompt_block()
)

USER_TEMPLATE = "Ticket:\n{ticket}\n\nTriage JSON:"

# Number of demonstrations in the few-shot baseline: one per category, so the
# baseline sees every label at least once. This is a deliberately *strong*
# few-shot baseline -- a weak one would flatter the fine-tuned model.
FEWSHOT_K = 8


def user_turn(ticket_text: str) -> str:
    return USER_TEMPLATE.format(ticket=ticket_text)


def build_messages(ticket_text: str,
                   demonstrations: list[dict[str, Any]] | None = None
                   ) -> list[dict[str, str]]:
    """Assemble a chat message list.

    Demonstrations are rendered as real alternating user/assistant turns rather
    than being pasted into the system prompt. That is what the chat template is
    designed for, and it matches how the model saw multi-turn data in its own
    instruction tuning.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demo in (demonstrations or []):
        messages.append({"role": "user", "content": user_turn(demo["ticket_text"])})
        messages.append({"role": "assistant",
                         "content": canonical_json(demo["expected_output"])})
    messages.append({"role": "user", "content": user_turn(ticket_text)})
    return messages


def render_prompt(tokenizer, ticket_text: str,
                  demonstrations: list[dict[str, Any]] | None = None) -> str:
    """The exact string fed to the model, generation prompt included."""
    messages = build_messages(ticket_text, demonstrations)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def render_training_text(tokenizer, ticket_text: str,
                         expected_output: dict[str, Any]) -> tuple[str, str]:
    """Return (prompt, completion) for supervised fine-tuning.

    The prompt half is identical to what the zero-shot baseline receives. The
    completion half is the canonical JSON plus the end-of-turn token, so the
    adapted model learns to stop cleanly instead of rambling past the object.
    """
    prompt = render_prompt(tokenizer, ticket_text, demonstrations=None)
    completion = canonical_json(expected_output) + tokenizer.eos_token
    return prompt, completion


def select_demonstrations(train_records: list[dict[str, Any]],
                          k: int = FEWSHOT_K,
                          seed_name: str = "fewshot_selection"
                          ) -> list[dict[str, Any]]:
    """Pick k demonstrations from the TRAINING split only.

    Two constraints:
      * Category coverage first -- one example per category until we run out of
        categories, so the few-shot prompt demonstrates the full label space.
      * Deterministic given the seed, so the baseline is reproducible.

    Drawing demonstrations from validation or test would leak held-out data
    into a "base model" baseline and make the comparison meaningless.
    """
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in train_records:
        by_category[rec["category"]].append(rec)

    chosen: list[dict[str, Any]] = []
    categories = sorted(by_category)
    r = rng(seed_name)

    # Round-robin over categories so coverage degrades gracefully if k < 8.
    round_index = 0
    while len(chosen) < k:
        progressed = False
        for category in categories:
            if len(chosen) >= k:
                break
            pool = sorted(by_category[category], key=lambda x: x["example_id"])
            if round_index >= len(pool):
                continue
            # Shuffle once per category with a stable seed, then take by index.
            local = list(pool)
            rng(seed_name, category).shuffle(local)
            chosen.append(local[round_index])
            progressed = True
        if not progressed:
            break
        round_index += 1

    # Present demonstrations in a fixed, mixed order (not grouped by category)
    # so the model does not infer a positional pattern.
    r.shuffle(chosen)
    return chosen


def prompt_stats(tokenizer, records: list[dict[str, Any]],
                 demonstrations: list[dict[str, Any]] | None = None
                 ) -> dict[str, Any]:
    """Token-length statistics, used to choose max_seq_len honestly."""
    lengths = [
        len(tokenizer(render_prompt(tokenizer, r["ticket_text"], demonstrations),
                      add_special_tokens=False)["input_ids"])
        for r in records
    ]
    target_lengths = [
        len(tokenizer(canonical_json(r["expected_output"]),
                      add_special_tokens=False)["input_ids"])
        for r in records
    ]
    return {
        "n": len(records),
        "prompt_tokens": {
            "min": min(lengths), "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
        },
        "target_tokens": {
            "min": min(target_lengths), "max": max(target_lengths),
            "mean": round(sum(target_lengths) / len(target_lengths), 1),
        },
        "prompt_plus_target_max": max(
            p + t for p, t in zip(lengths, target_lengths)
        ),
    }
