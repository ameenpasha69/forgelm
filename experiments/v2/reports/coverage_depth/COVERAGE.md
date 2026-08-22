# E2 -- coverage versus depth at a fixed example budget

**Verdict: no detectable difference.**

Two arms, **64 training examples each**, differing only in how those examples are spread over scenario families.

| | Arm A -- high coverage | Arm B -- low coverage |
|---|---|---|
| Scenario families | 32 | 16 |
| Examples per family | 2 | 4 |
| Total examples | 64 | 64 |
| Category distribution | {'access_management': 8, 'account_billing': 8, 'email': 8, 'hardware': 8, 'network': 8, 'other': 8, 'security': 8, 'software': 8} | {'access_management': 8, 'account_billing': 8, 'email': 8, 'hardware': 8, 'network': 8, 'other': 8, 'security': 8, 'software': 8} |
| Priority distribution | {'critical': 9, 'high': 9, 'low': 37, 'medium': 9} | {'critical': 8, 'high': 14, 'low': 35, 'medium': 7} |

Category distribution is identical by construction. Priority distribution was matched at design time by choosing Arm B's families to minimise distance to Arm A's; the residual gap is reported above rather than smoothed over, because only six family pairings exist per category and an exact match is not achievable.

## Per seed (frozen v1 test split, n=86)

| Seed | High coverage | Low coverage | Difference (high-low) | 95% CI | McNemar p | Detected? |
|---|---|---|---|---|---|---|
| 1337 | 19.8% | 18.6% | +1.2 pp | [-7.0, +8.1] pp | 1.0000 | no |
| 2718 | 18.6% | 17.4% | +1.2 pp | [-5.8, +9.3] pp | 1.0000 | no |
| 3141 | 17.4% | 19.8% | -2.3 pp | [-10.5, +5.8] pp | 0.7744 | no |

## Pooled across seeds

| Metric | Arm | Mean | Seed min | Seed max | Spread |
|---|---|---|---|---|---|
| exact_match | high | 18.6% | 17.4% | 19.8% | 2.3 pp |
| exact_match | low | 18.6% | 17.4% | 19.8% | 2.3 pp |
| schema_valid_rate | high | 80.2% | 77.9% | 83.7% | 5.8 pp |
| schema_valid_rate | low | 74.8% | 68.6% | 77.9% | 9.3 pp |

Mean difference (high - low) across seeds: **-0.0 pp**. 0 of 3 seeds show a difference whose CI excludes zero. Direction: high better in 2, low better in 1.

## Reading this honestly

At this budget the experiment **does not detect** a difference between spreading 64 examples over 32 scenarios and concentrating them in 16. That is *not* the same as showing the two are equivalent -- it is a statement about what this design could resolve.

The design is underpowered on purpose-built grounds, not as an excuse: 64 training examples per arm is a third of v1's, the coverage contrast is only 2:1, there are 3 seeds, and the test split is 86 examples. E1 measured the seed spread on the full configuration at 14 pp, which is larger than most effects this comparison could hope to see.

Zero-scoring held-out families, per seed:

| Seed | High coverage | Low coverage |
|---|---|---|
| 1337 | 8/16 | 10/16 |
| 2718 | 8/16 | 9/16 |
| 3141 | 8/16 | 8/16 |
