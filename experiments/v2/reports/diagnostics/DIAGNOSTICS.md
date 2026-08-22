# E4 -- diagnostic suites

**Secondary. Not the primary benchmark, and never pooled with it.**

Built from the v2 train and validation splits; the v2 test split stays sealed. Three suites destroy the ground truth on purpose, so each declares what it can legitimately be scored on.

| Suite | Question | Scoring |
|---|---|---|
| `unseen_families` | Does it handle situations it has never trained on? | `full` |
| `noisy_text` | Does typo-ridden, lower-case, informal writing break it? | `full` |
| `irrelevant_detail` | Is it distracted by a sentence that carries no signal? | `full` |
| `long_tickets` | Does a much longer ticket degrade it? | `full` |
| `missing_user_count` | When a field is genuinely unknowable, what does it do? | `except_users` |
| `contradictory` | When the ticket contradicts itself, does it still emit well-formed output? | `schema_only` |
| `out_of_domain` | Given input that is not a helpdesk ticket at all, does it produce confident nonsense? | `schema_only` |

## lora

| Suite | n | Schema valid | Strict JSON | Invalid enum | Wrong values only | Exact match (scored fields) |
|---|---|---|---|---|---|---|
| `unseen_families` | 96 | 78.1% | 100.0% | 21.9% | 63.5% | 14.6% |
| `noisy_text` | 96 | 83.3% | 100.0% | 16.7% | 71.9% | 11.5% |
| `irrelevant_detail` | 96 | 91.7% | 100.0% | 8.3% | 79.2% | 12.5% |
| `long_tickets` | 96 | 97.9% | 100.0% | 2.1% | 88.5% | 9.4% |
| `missing_user_count` | 96 | 71.9% | 100.0% | 22.9% | 65.6% | 14.6% |
| `contradictory` | 96 | 86.5% | 100.0% | 13.5% | 86.5% | n/a |
| `out_of_domain` | 12 | 50.0% | 100.0% | 25.0% | 50.0% | n/a |

### Per-field accuracy where the labels survive

| Suite | `category` | `priority` | `affected_service` | `is_security_incident` | `users_affected` |
|---|---|---|---|---|---|
| `unseen_families` | 56.2% | 38.5% | 66.7% | 89.6% | 100.0% |
| `noisy_text` | 51.0% | 37.5% | 61.5% | 87.5% | 99.0% |
| `irrelevant_detail` | 56.2% | 38.5% | 67.7% | 86.5% | 97.9% |
| `long_tickets` | 49.0% | 32.3% | 56.2% | 83.3% | 100.0% |
| `missing_user_count` | 57.3% | 41.7% | 65.6% | 89.6% | n/a |

### Generalisation detail (label-preserving suites only)

| Suite | category macro-F1 | priority macro-F1 | worst-family accuracy | zero-scoring families |
|---|---|---|---|---|
| `unseen_families` | 0.567 | 0.261 | 0.0% | 11/16 |
| `noisy_text` | 0.506 | 0.246 | 0.0% | 11/16 |
| `irrelevant_detail` | 0.532 | 0.251 | 0.0% | 11/16 |
| `long_tickets` | 0.439 | 0.203 | 0.0% | 13/16 |

## What these do and do not mean

A drop on `noisy_text` means this model handles **this synthetic perturbation** worse. It is not a robustness claim. A result on `out_of_domain` shows whether the model emits confident, well-formed triage for input that deserves none -- which is a useful thing to know and still not a safety claim. Nothing here is evidence about real tickets, real users, or deployment.
