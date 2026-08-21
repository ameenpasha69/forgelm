# ForgeLM results -- frozen `test` split

All numbers below were recomputed from the raw prediction files in `reports/predictions/` by `scripts/04_report.py`. n = 86 examples. Intervals are 95% percentile bootstrap.

## Headline

| System | Strict JSON | Schema valid | **Exact match** | Constraint violations |
|---|---|---|---|---|
| Base model, zero-shot | 5.8% | 27.9% [18.6%, 37.2%] | **0.0% [0.0%, 0.0%]** | 95.3% |
| Base model, few-shot (k=8) | 100.0% | 61.6% [51.2%, 72.1%] | **1.2% [0.0%, 3.5%]** | 38.4% |
| Base model + LoRA adapter | 100.0% | 79.1% [69.8%, 87.2%] | **11.6% [5.8%, 18.6%]** | 20.9% |

## Per-field accuracy

| System | `category` | `priority` | `affected_service` | `is_security_incident` | `users_affected` | mean |
|---|---|---|---|---|---|---|
| Base model, zero-shot | 19.8% | 40.7% | 17.4% | 54.6% | 94.2% | 45.4% |
| Base model, few-shot (k=8) | 59.3% | 41.9% | 37.2% | 81.4% | 100.0% | 63.9% |
| Base model + LoRA adapter | 53.5% | 46.5% | 58.1% | 91.9% | 100.0% | 70.0% |

## Classification quality

| System | category macro-F1 | category acc | priority macro-F1 | priority acc |
|---|---|---|---|---|
| Base model, zero-shot | 0.140 | 19.8% | 0.145 | 40.7% |
| Base model, few-shot (k=8) | 0.544 | 59.3% | 0.200 | 41.9% |
| Base model + LoRA adapter | 0.506 | 53.5% | 0.286 | 46.5% |

## Paired comparisons -- exact match (primary)

Every field correct. Paired bootstrap, 10,000 resamples of the example indices; McNemar is the exact binomial form.

| Comparison | A | B | Difference (B-A) | 95% CI | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| zeroshot -> fewshot | 0.0% | 1.2% | +1.2 pp | [+0.0, +3.5] pp | 1.00e+00 | not distinguishable from zero |
| zeroshot -> lora | 0.0% | 11.6% | +11.6 pp | [+4.7, +18.6] pp | 1.95e-03 | difference detected |
| fewshot -> lora | 1.2% | 11.6% | +10.5 pp | [+3.5, +18.6] pp | 1.17e-02 | difference detected |

## Paired comparisons -- schema validity (secondary)

Reported alongside exact match because the two measure different abilities: producing a well-formed object with legal enum values, versus producing the *right* object. A system can gain a lot of the first while gaining almost none of the second.

| Comparison | A | B | Difference (B-A) | 95% CI | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| zeroshot -> fewshot | 27.9% | 61.6% | +33.7 pp | [+20.9, +46.5] pp | 2.43e-06 | difference detected |
| zeroshot -> lora | 27.9% | 79.1% | +51.2 pp | [+40.7, +61.6] pp | 1.14e-13 | difference detected |
| fewshot -> lora | 61.6% | 79.1% | +17.4 pp | [+7.0, +27.9] pp | 4.08e-03 | difference detected |

## Per-field paired tests: fewshot -> lora

An aggregate win can hide a per-field loss, and an apparent per-field loss can turn out to be noise. Each field gets its own paired test rather than an eyeball comparison of two percentages.

| Field | fewshot | LoRA | Difference | 95% CI | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| `category` | 59.3% | 53.5% | -5.8 pp | [-17.4, +7.0] pp | 0.4583 | no detectable difference |
| `priority` | 41.9% | 46.5% | +4.7 pp | [-11.6, +20.9] pp | 0.6778 | no detectable difference |
| `affected_service` | 37.2% | 58.1% | +20.9 pp | [+9.3, +32.6] pp | 0.0009 | improvement |
| `is_security_incident` | 81.4% | 91.9% | +10.5 pp | [+2.3, +18.6] pp | 0.0225 | improvement |
| `users_affected` | 100.0% | 100.0% | +0.0 pp | [+0.0, +0.0] pp | 1.0000 | no detectable difference |

## Failure taxonomy

| Failure mode | Base model, zero-shot | Base model, few-shot (k=8) | Base model + LoRA adapter |
|---|---|---|---|
| `correct` | 0 | 1 | 10 |
| `invalid_enum` | 15 | 33 | 18 |
| `out_of_range` | 2 | 0 | 0 |
| `wrong_type` | 45 | 0 | 0 |
| `wrong_values_only` | 24 | 52 | 58 |

## Generalisation: LoRA exact match by held-out scenario family

The headline rate averages over 16 unseen situations. **11 of 16 held-out families produced zero fully correct outputs.** The aggregate gain is concentrated in a few families rather than spread evenly, which is a materially different claim from "the model learned the task".

| Scenario family (unseen in training) | correct / n |
|---|---|
| `ab_expense_reimbursement` | 5 / 5 |
| `ab_invoice_dispute` | 2 / 5 |
| `oth_meeting_room_av` | 1 / 5 |
| `sw_file_corrupt` | 1 / 5 |
| `am_account_lockout` | 1 / 6 |
| `am_sso_failure` | 0 / 6 |
| `email_calendar_sync` | 0 / 5 |
| `email_spoofed_sender` | 0 / 5 |
| `hw_monitor_dead` | 0 / 6 |
| `hw_stolen_laptop` | 0 / 6 |
| `net_dns_failure` | 0 / 6 |
| `net_slow_internet` | 0 / 6 |
| `oth_asset_inventory` | 0 / 5 |
| `sec_malware_detected` | 0 / 5 |
| `sec_suspicious_login` | 0 / 5 |
| `sw_update_broke_app` | 0 / 5 |

### Invented enum values

Where the adapted model still breaks the schema, it is mostly copying a salient noun out of the ticket instead of mapping it onto the closed enum:

| Field | Value emitted (not in the enum) | Count |
|---|---|---|
| `affected_service` | `dns` | 6 |
| `affected_service` | `internet` | 3 |
| `category` | `display` | 2 |
| `affected_service` | `invoice` | 2 |
| `category` | `audio` | 2 |
| `affected_service` | `download` | 1 |
| `category` | `file_system` | 1 |
| `category` | `phone` | 1 |

## Against the pre-declared success criteria

**All pre-declared success criteria were met.**

- **PASS** -- LoRA exact match exceeds BOTH unchanged-model baselines
  - zeroshot=0.0%; fewshot=1.2%; lora=11.6%
- **PASS** -- paired bootstrap 95% CI vs the stronger baseline (fewshot) excludes zero
  - difference +10.5 pp, 95% CI [+3.5, +18.6] pp, McNemar p=1.172e-02
- **PASS** -- schema_valid_rate does not regress against the stronger baseline
  - fewshot=61.6%, lora=79.1%
- **PASS** -- constraint_violation_rate does not increase against the stronger baseline
  - fewshot=38.4%, lora=20.9%

## Audit: recomputed vs recorded

- `zeroshot`: recomputed metrics match the values recorded at run time
- `fewshot`: recomputed metrics match the values recorded at run time
- `lora`: recomputed metrics match the values recorded at run time

## Figures

- `reports\figures\headline_metrics_test.png`
- `reports\figures\field_accuracy_test.png`
- `reports\figures\error_taxonomy_test.png`
- `reports\figures\training_curve.png`
