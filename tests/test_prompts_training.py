"""Prompt construction, official chat-template usage, and training encoding.

These tests need the tokenizer (a small download, cached after the first run)
but never the model weights, so they stay fast.
"""

from __future__ import annotations

import pytest

from forgelm import training as T
from forgelm.config import DECODING, TRAINING
from forgelm.prompts import (
    FEWSHOT_K, SYSTEM_PROMPT, USER_TEMPLATE, build_messages, render_prompt,
    render_training_text, select_demonstrations,
)
from forgelm.schema import CATEGORIES, canonical_json
from forgelm.splits import apply_split

pytestmark = pytest.mark.usefixtures("tokenizer")


# --------------------------------------------------------------------------
# Message construction (no tokenizer needed)
# --------------------------------------------------------------------------

def test_zero_shot_messages_shape():
    messages = build_messages("my laptop is broken")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "my laptop is broken" in messages[1]["content"]


def test_few_shot_messages_alternate_roles(records):
    demos = records[:3]
    messages = build_messages("a ticket", demos)
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "user", "assistant", "user", "assistant",
        "user"]
    assert messages[2]["content"] == canonical_json(demos[0]["expected_output"])


def test_system_prompt_declares_every_enum_value():
    for category in CATEGORIES:
        assert category in SYSTEM_PROMPT
    assert "No explanation, no markdown, no code fences" in SYSTEM_PROMPT


def test_system_prompt_is_identical_in_every_condition(records):
    zero = build_messages("t")[0]["content"]
    few = build_messages("t", records[:2])[0]["content"]
    assert zero == few == SYSTEM_PROMPT, (
        "the instruction must not vary between conditions, or a measured "
        "difference could be an artefact of prompting rather than adaptation"
    )


# --------------------------------------------------------------------------
# Demonstration selection
# --------------------------------------------------------------------------

def test_demonstrations_come_only_from_train(records, manifest):
    by_split = apply_split(records, manifest)
    demos = select_demonstrations(by_split["train"], k=FEWSHOT_K)
    for demo in demos:
        assert manifest["example_split"][demo["example_id"]] == "train"


def test_demonstrations_cover_every_category(records, manifest):
    by_split = apply_split(records, manifest)
    demos = select_demonstrations(by_split["train"], k=8)
    assert {d["category"] for d in demos} == set(CATEGORIES)


def test_demonstration_selection_is_deterministic(records, manifest):
    by_split = apply_split(records, manifest)
    first = [d["example_id"] for d in select_demonstrations(by_split["train"])]
    second = [d["example_id"] for d in select_demonstrations(by_split["train"])]
    assert first == second


def test_demonstrations_are_distinct(records, manifest):
    by_split = apply_split(records, manifest)
    demos = select_demonstrations(by_split["train"], k=FEWSHOT_K)
    assert len({d["example_id"] for d in demos}) == FEWSHOT_K


# --------------------------------------------------------------------------
# Official chat template
# --------------------------------------------------------------------------

def test_render_uses_the_official_chat_template(tokenizer):
    rendered = render_prompt(tokenizer, "printer is jammed")
    assert tokenizer.chat_template, "tokenizer has no chat template"
    assert "<|im_start|>system" in rendered
    assert "<|im_start|>user" in rendered
    assert rendered.endswith("<|im_start|>assistant\n"), (
        "the prompt must end with the generation prompt, or the model is being "
        "asked to continue the user's turn instead of answering"
    )


def test_render_matches_tokenizer_apply_chat_template_directly(tokenizer):
    """Guards against the prompt module hand-rolling the format."""
    ticket = "vpn keeps dropping"
    expected = tokenizer.apply_chat_template(
        build_messages(ticket), tokenize=False, add_generation_prompt=True)
    assert render_prompt(tokenizer, ticket) == expected


def test_few_shot_prompt_contains_the_demonstrations(tokenizer, records):
    demos = records[:2]
    rendered = render_prompt(tokenizer, "a new ticket", demos)
    for demo in demos:
        assert demo["ticket_text"] in rendered
        assert canonical_json(demo["expected_output"]) in rendered
    assert rendered.count("<|im_start|>assistant") == len(demos) + 1


def test_training_text_prompt_half_equals_zero_shot_prompt(tokenizer, records):
    record = records[0]
    prompt, completion = render_training_text(
        tokenizer, record["ticket_text"], record["expected_output"])
    assert prompt == render_prompt(tokenizer, record["ticket_text"])
    assert completion == canonical_json(record["expected_output"]) + \
        tokenizer.eos_token


# --------------------------------------------------------------------------
# Training encoding and masking
# --------------------------------------------------------------------------

def test_encoding_masks_the_prompt(tokenizer, records):
    encoded = T.encode_example(tokenizer, records[0], TRAINING["max_seq_len"])
    labels = encoded["labels"]
    assert labels[:encoded["n_prompt_tokens"]] == \
        [T.IGNORE_INDEX] * encoded["n_prompt_tokens"]
    assert all(l != T.IGNORE_INDEX
               for l in labels[encoded["n_prompt_tokens"]:])


def test_supervised_span_decodes_to_the_json_answer(tokenizer, records):
    record = records[0]
    encoded = T.encode_example(tokenizer, record, TRAINING["max_seq_len"])
    supervised = [l for l in encoded["labels"] if l != T.IGNORE_INDEX]
    decoded = tokenizer.decode(supervised, skip_special_tokens=True)
    assert decoded.strip() == canonical_json(record["expected_output"])


def test_input_ids_and_labels_are_aligned(tokenizer, records):
    encoded = T.encode_example(tokenizer, records[0], TRAINING["max_seq_len"])
    assert len(encoded["input_ids"]) == len(encoded["labels"]) \
        == len(encoded["attention_mask"])


def test_completion_ends_with_eos(tokenizer, records):
    encoded = T.encode_example(tokenizer, records[0], TRAINING["max_seq_len"])
    assert encoded["input_ids"][-1] == tokenizer.eos_token_id, (
        "without a trailing eos the adapted model never learns to stop"
    )


def test_verify_masking_passes_on_real_data(tokenizer, records):
    encoded = T.build_dataset(tokenizer, records[:40], TRAINING["max_seq_len"])
    report = T.verify_masking(tokenizer, encoded)
    assert report["passed"], report["problems"]


def test_verify_masking_catches_an_unmasked_prompt(tokenizer, records):
    encoded = T.build_dataset(tokenizer, records[:3], TRAINING["max_seq_len"])
    encoded[0]["labels"] = list(encoded[0]["input_ids"])  # nothing masked
    report = T.verify_masking(tokenizer, encoded)
    assert not report["passed"]
    assert "nothing masked" in report["problems"][0]


def test_verify_masking_catches_a_fully_masked_example(tokenizer, records):
    encoded = T.build_dataset(tokenizer, records[:3], TRAINING["max_seq_len"])
    encoded[0]["labels"] = [T.IGNORE_INDEX] * len(encoded[0]["input_ids"])
    report = T.verify_masking(tokenizer, encoded)
    assert not report["passed"]
    assert "zero loss" in report["problems"][0]


def test_no_truncation_at_the_configured_length(tokenizer, records):
    encoded = T.build_dataset(tokenizer, records, TRAINING["max_seq_len"])
    stats = T.truncation_stats(encoded, TRAINING["max_seq_len"])
    assert stats["n_truncated"] == 0, (
        f"{stats['n_truncated']} examples truncated at "
        f"{TRAINING['max_seq_len']}; longest is {stats['total_tokens']['max']}"
    )


def test_max_seq_len_has_headroom(tokenizer, records):
    encoded = T.build_dataset(tokenizer, records, TRAINING["max_seq_len"])
    longest = T.truncation_stats(encoded, TRAINING["max_seq_len"])["total_tokens"]["max"]
    assert longest <= TRAINING["max_seq_len"]


def test_generation_budget_exceeds_target_length(tokenizer, records):
    """max_new_tokens must not manufacture truncation failures."""
    targets = [len(tokenizer(canonical_json(r["expected_output"]),
                             add_special_tokens=False)["input_ids"])
               for r in records]
    assert DECODING["max_new_tokens"] > max(targets) * 2


# --------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------

def test_collator_pads_consistently(tokenizer, records):
    torch = pytest.importorskip("torch")
    encoded = T.build_dataset(tokenizer, records[:4], TRAINING["max_seq_len"])
    batch = T.CausalCollator(tokenizer)(encoded)
    assert batch["input_ids"].shape == batch["labels"].shape \
        == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape[1] % 8 == 0


def test_collator_masks_padding_in_labels(tokenizer, records):
    encoded = T.build_dataset(tokenizer, records[:4], TRAINING["max_seq_len"])
    batch = T.CausalCollator(tokenizer)(encoded)
    for row_labels, row_attention in zip(batch["labels"], batch["attention_mask"]):
        for label, attend in zip(row_labels.tolist(), row_attention.tolist()):
            if attend == 0:
                assert label == T.IGNORE_INDEX, \
                    "padding positions must never contribute to the loss"


def test_training_arguments_filter_reports_dropped_keys():
    pytest.importorskip("transformers")
    accepted, dropped = T.supported_training_arguments(
        {"output_dir": "x", "learning_rate": 1e-4,
         "a_setting_that_does_not_exist": True})
    assert "learning_rate" in accepted
    assert "a_setting_that_does_not_exist" in dropped


def test_eval_strategy_alias_is_resolved():
    pytest.importorskip("transformers")
    accepted, dropped = T.supported_training_arguments(
        {"output_dir": "x", "eval_strategy": "epoch"})
    assert "eval_strategy" in dropped or any(
        k in accepted for k in ("eval_strategy", "evaluation_strategy"))
