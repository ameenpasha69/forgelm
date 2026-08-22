# E5 -- the sealed v2 test evaluation

**96 examples, 16 scenario families, evaluated once.**

Seal checksum `0122062c5432ed0253fb34c45a4164c7...`, re-derived at evaluation time rather than trusted.

Every model here was trained on v1 data **before the v2 catalogue existed**. These families are not merely held out -- they were not available to be trained on. Maximum similarity between any v2 ticket and any v1 ticket is 0.3313.

## Results

| Condition | Strict JSON | Schema valid | **Exact match** | Mean field acc |
|---|---|---|---|---|
| zeroshot | 4.2% | 28.1% | **0.0%** | 40.6% |
| fewshot | 100.0% | 67.7% | **3.1%** | 63.5% |
| lora | 100.0% | 83.3% | **29.2%** | 77.1% |
| fewshot_constrained | 100.0% | 100.0% | **9.4%** | 69.4% |
| lora_constrained | 100.0% | 100.0% | **40.6%** | 79.4% |

### Per field

| Condition | `category` | `priority` | `affected_service` | `is_security_incident` | `users_affected` |
|---|---|---|---|---|---|
| zeroshot | 15.6% | 18.8% | 14.6% | 58.3% | 95.8% |
| fewshot | 55.2% | 26.0% | 50.0% | 86.5% | 100.0% |
| lora | 61.5% | 61.5% | 70.8% | 91.7% | 100.0% |
| fewshot_constrained | 55.2% | 26.0% | 79.2% | 86.5% | 100.0% |
| lora_constrained | 66.7% | 59.4% | 79.2% | 91.7% | 100.0% |

## Paired comparisons

| Comparison | A | B | Difference | 95% CI | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| zeroshot -> lora | 0.0% | 29.2% | +29.2 pp | [+20.8, +38.5] pp | 0.0000 | difference detected |
| fewshot -> lora | 3.1% | 29.2% | +26.0 pp | [+16.7, +35.4] pp | 0.0000 | difference detected |
| fewshot_constrained -> lora_constrained | 9.4% | 40.6% | +31.2 pp | [+20.8, +41.7] pp | 0.0000 | difference detected |

## Generalisation on families that did not exist at training time

**7 of 16 sealed families produced zero fully correct outputs.**

| Family | correct / n |
|---|---|
| `v2_ab_po_mismatch` | 6 / 6 |
| `v2_sec_supply_chain_alert` | 6 / 6 |
| `v2_net_cable_damage` | 4 / 6 |
| `v2_am_group_nesting` | 3 / 6 |
| `v2_net_ipv6_misconfig` | 3 / 6 |
| `v2_oth_relocation_survey` | 3 / 6 |
| `v2_ab_currency_mismatch` | 1 / 6 |
| `v2_email_shared_mailbox_sync` | 1 / 6 |
| `v2_oth_power_outage_plan` | 1 / 6 |
| `v2_am_federation_cert_rotation` | 0 / 6 |
| `v2_email_autoreply_loop` | 0 / 6 |
| `v2_hw_fan_noise` | 0 / 6 |
| `v2_hw_ups_battery` | 0 / 6 |
| `v2_sec_shadow_it` | 0 / 6 |
| `v2_sw_api_deprecation` | 0 / 6 |
| `v2_sw_macro_blocked` | 0 / 6 |

## Priority balance, which v1 could not measure

v1's test split held 3 `medium` examples, so priority macro-F1 on it was close to meaningless. The sealed v2 split holds 17.

- category macro-F1: 0.603
- priority macro-F1: 0.481
