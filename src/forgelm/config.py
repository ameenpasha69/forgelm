"""Experiment configuration in one place.

Values here were chosen from measurements, not defaults, and each carries the
measurement that justified it. `configs/*.json` can override any of them for an
ablation; the effective config is written into every ledger record.
"""

from __future__ import annotations

from typing import Any

EXPERIMENT_ID = "forgelm-lora-ticket-triage-v1"

# ---------------------------------------------------------------------------
# Decoding -- identical in every condition
# ---------------------------------------------------------------------------
DECODING: dict[str, Any] = {
    "strategy": "greedy",
    "do_sample": False,
    "num_beams": 1,
    # Targets are 35-38 tokens (measured over all 300 examples). 160 leaves
    # ~120 tokens of headroom so a base model that prefixes prose can still
    # finish its JSON object. Deliberately generous: a stingy cap would
    # manufacture "truncated" failures for the baseline and flatter the
    # adapted model.
    "max_new_tokens": 160,
    "batch_size": 8,
    "rationale": (
        "Greedy decoding removes sampling variance; the task has exactly one "
        "correct answer. The token budget is identical across zero-shot, "
        "few-shot and adapted conditions."
    ),
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAINING: dict[str, Any] = {
    # Measured: max(prompt + target) = 283 tokens over the whole dataset.
    # 320 covers it with headroom and truncates nothing.
    "max_seq_len": 320,

    # LoRA. r=16 over all seven linear projections gives ~8.8M trainable
    # parameters (~1.8% of the model).
    "lora_r": 16,
    "lora_alpha": 32,          # alpha = 2r, the conventional pairing
    "lora_dropout": 0.05,

    # Optimisation. 2e-4 is the standard LoRA learning rate for models of this
    # size; the base model's own weights are frozen so a higher rate than full
    # fine-tuning is appropriate.
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "optimizer": "adamw_torch",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # Effective batch 8 via accumulation. per_device=1 because the 151,936-token
    # vocabulary makes the logits tensor the dominant memory cost on a 4 GiB
    # card: batch x seq x vocab x 4 bytes = 1 x 320 x 151936 x 4 ~= 195 MB in
    # fp32 for the loss computation alone.
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "per_device_eval_batch_size": 1,

    "num_train_epochs": 8,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "early_stopping_patience": 3,

    "logging_steps": 5,
    "seed_name": "training",
}

# ---------------------------------------------------------------------------
# Success criteria -- fixed BEFORE any training run
# ---------------------------------------------------------------------------
SUCCESS_CRITERIA: dict[str, Any] = {
    "declared_at": "before any training run; see DECISIONS.md D-008",
    "primary": {
        "metric": "exact_match on the frozen test split",
        "requirement": (
            "The LoRA-adapted model must beat BOTH unchanged-model baselines "
            "(zero-shot and few-shot), and the paired bootstrap 95% CI for the "
            "difference against the stronger baseline must exclude zero."
        ),
    },
    "secondary": [
        "schema_valid_rate must not regress against the stronger baseline",
        "constraint_violation_rate must not increase against the stronger baseline",
    ],
    "explicitly_not_claimed": [
        "production or deployment readiness",
        "generalisation beyond this synthetic dataset",
        "improved safety, alignment, fairness or robustness",
        "latency or cost improvements",
        "any capability on real helpdesk tickets",
    ],
}


def effective_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = {
        "experiment_id": EXPERIMENT_ID,
        "decoding": dict(DECODING),
        "training": dict(TRAINING),
    }
    if overrides:
        for section, values in overrides.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values
    return config
