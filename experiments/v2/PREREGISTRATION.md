# ForgeLM v2 -- pre-registration

**Written and committed before any v2 training run.** Its purpose is to fix the
analysis plan while the answers are still unknown, so that a result cannot be
selected after the fact.

| | |
|---|---|
| v1 commit preserved | `3314025787773d851feb70e8b5ed36fde4af3dd3` |
| v1 audit at that commit | 20/20 checks, 142 tests, working tree clean |
| Branch | `v2-continuation` |
| Hardware | NVIDIA GTX 1650, 4 GiB, sm_75 (unchanged from v1) |

v1 evidence is immutable. No v1 dataset, prediction file, run record, report or
adapter is modified. The v1 headline result is never recomputed or replaced.

---

## Why v2 exists

v1 answered its question and, in doing so, exposed four specific weaknesses.
v2 addresses those and nothing else.

| v1 weakness | v2 response |
|---|---|
| One training seed per condition; "a sample of size one" | E1: three seeds, all reported |
| The data-size ablation accidentally dropped one scenario family | E2: equal-count, category-balanced coverage-vs-depth arms |
| 18/86 outputs still contained invented enum values | E3: constrained decoding, applied symmetrically to base and adapted |
| Generalisation measured only on one held-out slice | E4: separate diagnostic suites, kept out of the primary benchmark |

---

## E1 -- Multi-seed variance of the primary configuration

**Question.** Does the v1 headline conclusion survive training-seed variance?

**Design.** Three independent training runs of the *exact* v1 primary
configuration (r=16, alpha=32, dropout=0.05, lr 2e-4, cosine, 8 epochs max,
early stopping patience 3, effective batch 8, max_seq_len 320, fp16), differing
only in seed.

**Seeds, fixed now:** `1337`, `2718`, `3141`.
`1337` is v1's seed; re-running it also serves as a reproducibility check
against the v1 run. Initialisation seed and data-order seed are set to the same
value in each run and recorded separately.

**Recorded per seed:** initialisation seed, data-order seed, runtime, selected
checkpoint, full training and validation loss history, exact match, schema
validity, per-field accuracy, per-scenario-family accuracy, adapter sha256.

**Analysis, fixed now.**
- Report **every** seed. No seed is dropped, and the best seed is never
  promoted to headline.
- Report mean, standard deviation, min, max across seeds for each metric.
- Report an interval accounting for **both** sources of variation: test-example
  sampling (paired bootstrap, as in v1) and training-seed variation. Seed
  variation is summarised by the observed spread across the three runs; with
  n=3 this is a descriptive range, not a precise variance estimate, and will be
  labelled as such.
- Count how many seeds independently meet v1's pre-declared primary criterion
  (LoRA exact match beats both baselines **and** the paired 95% CI against the
  stronger baseline excludes zero).

**Conclusion rule, fixed now.**
- **Survives** if all 3 seeds meet the primary criterion.
- **Partially survives** if 2 of 3 do -- reported as seed-dependent.
- **Does not survive** if fewer than 2 do.

The v1 published number is not edited whatever the outcome; E1 is reported
alongside it as a robustness assessment.

---

## E2 -- Coverage versus depth, properly controlled

**Question.** At a fixed training-example budget, does covering more scenario
families beat seeing more examples of fewer families?

v1's ablation gestured at this but confounded it: the smaller arm lost the
`email_dl_update` family as a side effect of stratifying by category.

**Design.** Two arms with **exactly equal example counts** and **identical
category distributions**.

| | Arm A -- high coverage | Arm B -- low coverage |
|---|---|---|
| Scenario families | 32 (all train families, 4 per category) | 16 (2 per category) |
| Examples per family | 2 | 4 |
| **Total examples** | **64** | **64** |
| Examples per category | 8 | 8 |

Everything else is identical by construction: prompt, LoRA settings, decoding,
validation split, test split, seeds, and the number of optimiser steps implied
by the example count.

**Why 64 and not more.** Equal counts with category balance require Arm B's
families to supply 4 examples each. Only 6 of the 32 training families hold 6
or more examples, so the 32x3 / 16x6 = 96 design is infeasible. 32x2 / 16x4 = 64
is the largest clean design the v1 training split supports.

**Seeds:** the same three -- `1337`, `2718`, `3141` -- in both arms, so the
comparison is paired on seed as well as on test example.

**Analysis, fixed now.** Paired bootstrap and exact McNemar on the frozen v1
test split, per seed and pooled. Primary metric exact match; secondary schema
validity and per-field accuracy.

**Reporting rule, fixed now.** A null result is reported as
**"no detectable difference"**. It is not reported as evidence that coverage
and depth are equivalent. With 64 training examples per arm, 3 seeds and 86
test examples, this design is underpowered for small effects, and that will be
stated with the result rather than discovered afterwards.

---

## E3 -- Constrained decoding

**Question.** How much of the remaining schema failure is a decoding problem
rather than a model problem?

**Motivation.** 18 of 86 v1 LoRA outputs contained an invented enum value --
`"dns"`, `"internet"`, `"display"`, `"audio"` -- i.e. a noun copied from the
ticket instead of mapped onto the closed enum. A decoder that cannot emit an
illegal token makes that failure class impossible.

**Guarantee required of the constrained path:** exactly one JSON object,
exactly the five permitted keys in order, correct value types, enum values
drawn only from the permitted sets, no additional fields, no markdown fence,
no surrounding prose.

**Conditions, all on the same frozen test split with the same prompt:**

| # | Model | Decoding |
|---|---|---|
| 1 | base, zero-shot | unconstrained (v1 evidence, reused) |
| 2 | base, few-shot k=8 | unconstrained (v1 evidence, reused) |
| 3 | base + LoRA | unconstrained (v1 evidence, reused) |
| 4 | base, few-shot k=8 | **constrained** |
| 5 | base + LoRA | **constrained** |

**Fairness rule, fixed now.** The identical constraint mechanism is applied to
condition 4 and condition 5. LoRA is never credited with an improvement the
decoder produced: the quantity of interest is the *difference between* 4 and 5,
not the difference between 3 and 5.

**Metrics separated, fixed now:** syntactic validity (parses as JSON), schema
validity (satisfies the contract), semantic field correctness (per-field
accuracy), and exact match. Constrained decoding is expected to make the first
two trivially 100%; only the last two carry information about the model.

**Tests required before results are reported:** the constrained path must be
shown to reject an illegal enum value and an extra key.

---

## E4 -- Diagnostic suites (secondary, not the primary benchmark)

Seven small diagnostic sets, generated deterministically and kept **separate**
from the primary benchmark so they cannot inflate or deflate the headline:

unseen scenario families; noisy spelling and informal writing; irrelevant
detail; missing user-count information; contradictory information;
out-of-domain input; unusually long tickets.

**Reported:** exact match, schema validity, per-field accuracy, macro-F1,
worst-family accuracy, number of zero-scoring families, invalid-enum rate,
wrong-values-only rate, and a breakdown by input type.

**Interpretation rule, fixed now.** These are diagnostics of *this model on this
synthetic data*. They will not be converted into claims about safety,
robustness, or real-world reliability. A drop on the noisy-input set means the
model handles that synthetic perturbation worse; it does not mean the model is
"not robust" in any general sense.

---

## E5 -- Sealed v2 test set

New scenario families, genuinely new situations rather than paraphrases of v1
families. Frozen **before** any v2 training, with:

stable scenario IDs; a versioned scenario catalogue; deterministic generation;
family-disjoint splitting; exact and normalised duplicate checks; exhaustive
cross-split similarity analysis; improved priority balance (v1's test split had
only 3 `medium` examples); a test-membership checksum; and an automated guard
that makes a training script reading a sealed example an error rather than a
mistake.

**Sealing rule, fixed now.** v2 test predictions are not inspected until
configurations, seeds, prompts, decoding and checkpoint-selection rules are
frozen and committed.

---

## What v2 will not do

No QLoRA, MLflow, vLLM, RAG, agents, preference optimisation, cloud deployment,
production API, or distributed training. No claim of production readiness,
safety, alignment, fairness, real-ticket reliability, general capability
improvement, or latency/cost improvement. None of those are measured, and none
are needed to resolve the weaknesses listed above.
