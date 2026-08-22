# E1 -- multi-seed variance of the primary configuration

**Verdict: the v1 conclusion survives.** 3 of 3 seeds independently meet v1's pre-declared primary criterion.

The rule above was fixed in `PREREGISTRATION.md` before any of these runs existed. Every seed is reported; none is promoted.

## Per seed

| Seed | Selected epoch | Best val loss | Runtime | Exact match | Schema valid | Zero-scoring families | Adapter sha256 |
|---|---|---|---|---|---|---|---|
| 1337 | 2.0 | 0.0743 | 2202s | **11.6%** | 79.1% | 11/16 | `4ee00d3b9be3` |
| 2718 | 1.0 | 0.1006 | 1666s | **8.1%** | 68.6% | 12/16 | `20a4fec77492` |
| 3141 | 3.0 | 0.0766 | 2474s | **22.1%** | 94.2% | 7/16 | `63f0688f2d27` |

## Does each seed clear v1's bar?

v1's criterion: beat **both** baselines on exact match (zero-shot 0.0%, few-shot 1.2%) **and** have the paired 95% CI against few-shot exclude zero.

| Seed | Exact match | Diff vs few-shot | 95% CI | McNemar p | Meets criterion |
|---|---|---|---|---|---|
| 1337 | 11.6% | +10.5 pp | [+3.5, +18.6] pp | 0.0117 | **yes** |
| 2718 | 8.1% | +7.0 pp | [+1.2, +14.0] pp | 0.0703 | **yes** |
| 3141 | 22.1% | +20.9 pp | [+11.6, +30.2] pp | 0.0000 | **yes** |

## Across seeds

Intervals below resample **both** test examples and training seeds. v1's interval resampled test examples only.

| Metric | Mean | Seed min | Seed max | Seed spread | Seed SD | 95% CI (both sources) |
|---|---|---|---|---|---|---|
| exact_match | 14.0% | 8.1% | 22.1% | 14.0 pp | 7.3 pp | [5.8%, 24.0%] |
| schema_valid_rate | 80.6% | 68.6% | 94.2% | 25.6 pp | 12.9 pp | [66.3%, 93.8%] |
| constraint_violation_rate | 19.4% | 5.8% | 31.4% | 25.6 pp | 12.9 pp | [6.2%, 33.7%] |

With 3 seeds the seed level of that bootstrap is coarse. The raw spread column is the more truthful summary.

### Per field, across seeds

| Field | Mean | Seed min | Seed max | Spread |
|---|---|---|---|---|
| `category` | 58.9% | 53.5% | 65.1% | 11.6 pp |
| `priority` | 57.0% | 46.5% | 67.4% | 20.9 pp |
| `affected_service` | 58.9% | 44.2% | 74.4% | 30.2 pp |
| `is_security_incident` | 87.6% | 81.4% | 91.9% | 10.5 pp |
| `users_affected` | 100.0% | 100.0% | 100.0% | 0.0 pp |

## How to read this

**The direction is robust; the magnitude is not.** Every seed beats few-shot, but the seed-to-seed spread in exact match is **14.0 pp**, which is larger than the weakest seed's entire effect (+7.0 pp). The per-seed effect ranges from +7.0 pp to +20.9 pp.

So the defensible claim is *"LoRA beats few-shot on this task"*, which held in 3/3 runs. The claim *"LoRA beats few-shot by about 10 points"* -- which v1's single run invited -- is **not** supported: v1's seed happened to land mid-range, and a different seed would have reported anywhere from +7 to +21 pp.

**Two tests disagree on some seeds.** The pre-registered criterion is the paired bootstrap CI, so that is what decides the verdict -- but where McNemar's exact test disagrees, both are reported:

- seed 2718: bootstrap CI [+1.2, +14.0] pp excludes zero, but McNemar p = 0.0703 (above 0.05). This seed sits at the margin and should not be described as a clear win on its own.

**Generalisation stays poor in every seed.** Zero-scoring held-out scenario families: 11/16 (seed 1337), 12/16 (seed 2718), 7/16 (seed 3141). The v1 finding that the adapter learns the output contract far better than the task is not a seed artefact.

## Deviation from the pre-registration

Seed 1337 reuses the v1 training run rather than repeating it. It is the identical configuration and seed, so it is a valid member of the seed set, but the pre-registration said it would be re-run as a reproducibility check and it was not. Reason: the machine has 5.9 GB of RAM and a repeat run costs ~40 minutes of strictly serial time.
