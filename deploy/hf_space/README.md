---
title: ForgeLM Ticket Triage
emoji: 🎫
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: mit
short_description: LoRA-adapted Qwen2.5-0.5B turning helpdesk tickets into structured JSON
---

# ForgeLM — structured ticket triage with a LoRA adapter

A hosted demonstration of [ForgeLM](https://github.com/ameenpasha69/forgelm):
`Qwen/Qwen2.5-0.5B-Instruct` with a LoRA adapter trained on 171 synthetic
helpdesk tickets to emit a strict triage object.

Paste a ticket, get back:

```json
{"category": "network", "priority": "high", "affected_service": "vpn",
 "is_security_incident": false, "users_affected": 14}
```

## What this is, and what it is not

This is a **demonstration of a research artifact**, not a product:

- The adapter was trained on **synthetic** tickets. Do not use it to triage
  real ones.
- There is no authentication and no rate limiting beyond a single-slot queue.
- The output schema has no `not_a_ticket` value, so refusal is not
  representable — the model will confidently triage `2 + 2 = ?`. That is a
  real finding from the project's diagnostics, reported rather than hidden.

## Measured results

Held-out split, schema validity:

| System | Schema valid | Exact match |
|---|---|---|
| base, zero-shot | 27.9% | 0.0% |
| base, few-shot k=8 | 61.6% | 1.2% |
| base + LoRA | 79.1% | 11.6% |

Full method, ablations, dataset card and limitations live in the
[GitHub repository](https://github.com/ameenpasha69/forgelm).

## Notes on this deployment

Inference runs greedy on CPU, so a ticket takes a few seconds. The model is
loaded once at startup and requests are queued one at a time — a single model
instance on a shared CPU cannot serve two generations concurrently.

The Space is generated from `deploy/hf_space/` in the GitHub repository; see
`deploy/build.py` there for how it is assembled.
