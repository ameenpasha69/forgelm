---
base_model: Qwen/Qwen2.5-0.5B-Instruct
base_model_revision: 7ae557604adf67be50417f59c2c2f167def9a775
library_name: peft
license: mit
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- structured-output
- json
---

# ForgeLM ticket-triage LoRA adapter

**A LoRA adapter, not a model.** It must be loaded on top of
`Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. On its own it does nothing.

Converts one free-text IT helpdesk ticket into a strict five-field JSON object.

```json
{"category": "network", "priority": "high", "affected_service": "vpn", "is_security_incident": false, "users_affected": 34}
```

## Measured results

Frozen test split, 86 held-out synthetic examples, evaluated once. Same prompt,
same greedy decoding and same parser in every condition.

| System | Strict JSON | Schema valid | Exact match |
|---|---|---|---|
| base, zero-shot | 5.8% | 27.9% | 0.0% |
| base, few-shot k=8 | 100.0% | 61.6% | 1.2% |
| **base + this adapter** | 100.0% | **79.1%** | **11.6%** |

Against the stronger (few-shot) baseline: **+10.5 pp exact match, 95% CI
[+3.5, +18.6], McNemar p = 0.012.**

## What it did not learn

**11 of the 16 held-out scenario families produced zero fully correct outputs.**
The adapter learned the *output contract* well and the *task* only partially.
Per-field, the significant gains are `affected_service` (+20.9 pp) and
`is_security_incident` (+10.5 pp); `category`, `priority` and `users_affected`
show no detectable difference.

## Training

171 synthetic examples, LoRA r=16 / alpha=32 / dropout=0.05 on all seven linear
projections (8,798,208 trainable parameters, 1.75% of the base model). fp16 with
fp32 adapter parameters, effective batch 8, lr 2e-4 cosine. Early stopping
selected the epoch-2 checkpoint on validation loss.

## Do not use this on real tickets

It was trained on invented data against an invented priority rule, and evaluated
only on 86 synthetic examples. Nothing about safety, alignment, fairness,
robustness or real-world reliability was measured.

## Full documentation

`forgelm_provenance.json` in this directory records the required base model,
its revision, the dataset checksum, the split manifest checksum and every seed.

See the project repository for `MODEL_CARD.md`, `DATASET_CARD.md`,
`EXPERIMENT_CARD.md`, the raw per-example predictions, and the evidence audit.
