# ForgeLM results -- frozen `test` split

All numbers below were recomputed from the raw prediction files in `reports/predictions/` by `scripts/04_report.py`. n = 86 examples. Intervals are 95% percentile bootstrap.

## Headline

| System | Strict JSON | Schema valid | **Exact match** | Constraint violations |
|---|---|---|---|---|
| Base model, zero-shot | 5.8% | 27.9% [18.6%, 37.2%] | **0.0% [0.0%, 0.0%]** | 95.3% |
| Base model, few-shot (k=8) | 100.0% | 61.6% [51.2%, 72.1%] | **1.2% [0.0%, 3.5%]** | 38.4% |

## Per-field accuracy

| System | `category` | `priority` | `affected_service` | `is_security_incident` | `users_affected` | mean |
|---|---|---|---|---|---|---|
| Base model, zero-shot | 19.8% | 40.7% | 17.4% | 54.6% | 94.2% | 45.4% |
| Base model, few-shot (k=8) | 59.3% | 41.9% | 37.2% | 81.4% | 100.0% | 63.9% |

## Classification quality

| System | category macro-F1 | category acc | priority macro-F1 | priority acc |
|---|---|---|---|---|
| Base model, zero-shot | 0.140 | 19.8% | 0.145 | 40.7% |
| Base model, few-shot (k=8) | 0.544 | 59.3% | 0.200 | 41.9% |

## Paired comparisons -- exact match (primary)

Every field correct. Paired bootstrap, 10,000 resamples of the example indices; McNemar is the exact binomial form.

| Comparison | A | B | Difference (B-A) | 95% CI | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| zeroshot -> fewshot | 0.0% | 1.2% | +1.2 pp | [+0.0, +3.5] pp | 1.00e+00 | not distinguishable from zero |

## Paired comparisons -- schema validity (secondary)

Reported alongside exact match because the two measure different abilities: producing a well-formed object with legal enum values, versus producing the *right* object. A system can gain a lot of the first while gaining almost none of the second.

| Comparison | A | B | Difference (B-A) | 95% CI | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| zeroshot -> fewshot | 27.9% | 61.6% | +33.7 pp | [+20.9, +46.5] pp | 2.43e-06 | difference detected |

## Failure taxonomy

| Failure mode | Base model, zero-shot | Base model, few-shot (k=8) |
|---|---|---|
| `correct` | 0 | 1 |
| `invalid_enum` | 15 | 33 |
| `out_of_range` | 2 | 0 |
| `wrong_type` | 45 | 0 |
| `wrong_values_only` | 24 | 52 |

## Against the pre-declared success criteria

**Not yet evaluable: the LoRA condition has not been run.**


## Audit: recomputed vs recorded

- `zeroshot`: recomputed metrics match the values recorded at run time
- `fewshot`: recomputed metrics match the values recorded at run time

## Figures

- `reports\figures\headline_metrics_test.png`
- `reports\figures\field_accuracy_test.png`
- `reports\figures\error_taxonomy_test.png`
