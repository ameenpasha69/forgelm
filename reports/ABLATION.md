# Optional ablation -- training-data size

**This is a secondary experiment, run after the primary result was complete and reproducible. It is not part of the pre-registered success criteria and should not be read with the same weight.**

One variable changed: the number of training examples (171 -> 86, category-stratified, deterministic). LoRA rank, alpha, dropout, learning rate, schedule, batch size, sequence length, precision, seeds, prompt, decoding and the evaluation split are all identical.

## What was actually removed

This matters for interpretation. Halving the data could mean *fewer examples per scenario* or *fewer scenarios*, and those support opposite conclusions.

- Training scenario families retained: **31 / 32**
- Families lost entirely: `email_dl_update` (1 of 32)
- Examples per family: 5.34 -> 2.77

So this ablation is **mostly** a depth reduction: examples per scenario fell to 52% of the full run while scenario coverage held at 97% (31/32 families). It is not a clean depth-only manipulation -- 1 family was lost as a side effect of stratifying by category rather than by family. That confound is small but real, and it is stated rather than rounded away.

## Results on the frozen test split

| Training examples | Strict JSON | Schema valid | Exact match | Constraint violations |
|---|---|---|---|---|
| 86 | 100.0% | 82.6% | 17.4% | 17.4% |
| **171** | 100.0% | 79.1% | 11.6% | 20.9% |

## Does doubling the data help?

| Metric | 50% | 100% | Difference | 95% CI | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| exact match | 17.4% | 11.6% | -5.8 pp | [-12.8, +1.2] pp | 0.1797 | not distinguishable from zero |
| schema valid | 82.6% | 79.1% | -3.5 pp | [-11.6, +4.7] pp | 0.5811 | not distinguishable from zero |

### Per field

| Field | 50% | 100% | Difference | 95% CI | p | Verdict |
|---|---|---|---|---|---|---|
| `category` | 67.4% | 53.5% | -14.0 pp | [-22.1, -7.0] pp | 0.0005 | difference detected |
| `priority` | 57.0% | 46.5% | -10.5 pp | [-20.9, +0.0] pp | 0.0931 | no detectable difference |
| `affected_service` | 59.3% | 58.1% | -1.2 pp | [-11.6, +9.3] pp | 1.0000 | no detectable difference |
| `is_security_incident` | 87.2% | 91.9% | +4.7 pp | [-1.2, +11.6] pp | 0.2891 | no detectable difference |
| `users_affected` | 100.0% | 100.0% | +0.0 pp | [+0.0, +0.0] pp | 1.0000 | no detectable difference |

### Surprising: on some fields, half the data was *better*

- **`category`**: 67.4% at 50% vs 53.5% at 100% (+14.0 pp in favour of the smaller training set, 95% CI [+7.0, +22.1] pp, p = 0.0005)

Two explanations fit, and this experiment cannot separate them:

1. **Overfitting to training scenarios.** At 171 examples the model sees each scenario ~5.3 times; at 86 it sees each ~2.8 times. More repetitions of the same 32 situations may push it to memorise scenario-specific surface cues that do not transfer to the 16 held-out situations. `category` is the field most dependent on recognising the *kind* of situation, so it is the one you would expect to suffer first. That is consistent with the primary run's training curve, where validation loss bottomed at epoch 2 and rose thereafter.

2. **Run-to-run variance.** Each arm is a single training run with a single seed. Nothing here separates a real effect from seed noise, and a 14-point swing on one field from one run per arm is well within what seed variance can produce for a 0.5B model on 86 test examples.

**The honest position is that this is a hypothesis, not a result.** Settling it needs 3-5 seeds per arm, which was not run. It is reported because deleting a surprising number because it is inconvenient is worse than reporting it with its caveat.

## Reading this honestly

Doubling the training data from 86 to 171 did **not** produce a detectable change in exact match (-5.8 pp, 95% CI [-12.8, +1.2] pp, McNemar p = 0.180). Combined with the primary finding that 11 of 16 held-out scenario families scored zero, this points at **scenario coverage rather than example volume** as the binding constraint: adding more examples of the scenarios the model already sees does not help it handle scenarios it has never seen.

Caveats that apply with full force here: n = 86, a single seed per arm, and two arms. This is one comparison, not a scaling curve. It is suggestive, not conclusive.

### The headline result does not change

The ablation arm scored higher on exact match (17.4% vs 11.6%). It would be trivially easy, and wrong, to now present the ablation as "the result".

The primary experiment was pre-registered against the **full** training split, its success criteria were fixed before any training run, and the frozen test split was evaluated once for it. This ablation was run afterwards, as a secondary question, and its arm was not pre-registered. Promoting a post-hoc arm because it scored better is exactly the selection this project exists to avoid -- it turns the test split into a model-selection set and destroys the meaning of the number.

**The reported result therefore remains the 171-example run.** The ablation's contribution is the *direction* it points, not a better score to substitute in.

Additionally, the subsample lost 1 scenario family as a side effect, so the manipulation is not purely a change in example count. A cleaner version would stratify the subsample by family rather than by category.
