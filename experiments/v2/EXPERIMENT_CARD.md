# Experiment card -- ForgeLM v2

| | |
|---|---|
| **Experiment ID** | `forgelm-v2-continuation` |
| **Branch** | `v2-continuation` |
| **v1 baseline preserved at** | `3314025787773d851feb70e8b5ed36fde4af3dd3` |
| **Hardware** | unchanged from v1: GTX 1650, 4 GiB VRAM, sm_75, **5.9 GiB system RAM** |

v2 does not restate v1. It addresses four specific weaknesses that v1's own
evidence exposed, and nothing else. The root `EXPERIMENT_CARD.md` remains the
record for v1 and is not edited.

---

## What v1 could not answer, and why

| v1 weakness | Why it mattered | v2 experiment |
|---|---|---|
| One training seed | The reported +10.5 pp was a sample of size one | **E1** three seeds |
| The data-size ablation dropped a scenario family | Coverage was confounded with label mix | **E2** equal-count, label-matched arms |
| 18/86 outputs contained invented enum values | Unclear whether that was a model or a decoder failure | **E3** constrained decoding |
| Generalisation measured on one held-out slice | The test split had been seen | **E4** diagnostics + **E5** a sealed set |

---

## Shared protocol

Everything below is inherited from v1 unchanged, so v2 results are comparable
to v1 results:

- **Base model** `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775`
- **Prompt** identical system prompt in every condition, SHA recorded per run
- **Decoding** greedy, `do_sample=False`, `num_beams=1`, `max_new_tokens=160`
- **LoRA** r=16, alpha=32, dropout=0.05, all seven linear projections,
  8,798,208 trainable parameters (1.75%)
- **Training** effective batch 8 (1 x 8 accumulation), lr 2e-4 cosine, 10%
  warm-up, up to 8 epochs, early stopping patience 3, fp16 with fp32 adapter
  parameters, `max_seq_len` 320
- **Checkpoint selection** validation loss only; test data never consulted
- **Parser** identical in every condition

Deviations are listed under each experiment rather than buried.

---

## E1 -- multi-seed variance

**Question.** Does the v1 conclusion survive training-seed variance?

**Design.** Three runs of the exact v1 primary configuration, differing only in
seed. Initialisation seed and data-order seed set to the same value per run and
recorded separately.

**Conclusion rule, fixed before the runs:** survives (all seeds meet v1's
primary criterion) / partially survives (a majority) / does not survive.

### Result

| Seed | Selected epoch | Best val loss | Exact match | vs few-shot | 95% CI | McNemar p | Meets criterion |
|---|---|---|---|---|---|---|---|
| 1337 | 2 | 0.0743 | 11.6% | +10.5 pp | [+3.5, +18.6] | 0.0117 | yes |
| 2718 | 1 | 0.1006 | 8.1% | +7.0 pp | [+1.2, +14.0] | 0.0703 | yes |
| 3141 | 3 | 0.0824 | 22.1% | +20.9 pp | [+11.6, +30.2] | <0.0001 | yes |

**Verdict: the v1 conclusion SURVIVES (3/3).**

| Metric | Mean | Min | Max | Spread | SD |
|---|---|---|---|---|---|
| exact match | 14.0% | 8.1% | 22.1% | **14.0 pp** | 7.3 pp |
| schema valid | 80.6% | 68.6% | 94.2% | 25.6 pp | 12.9 pp |

**What this changes about v1's claim.** The seed spread (14.0 pp) is *larger*
than the weakest seed's entire effect (+7.0 pp). So:

- supported: **"LoRA beats few-shot on this task"** -- held in 3/3 runs
- **not** supported: "LoRA beats few-shot by about 10 points" -- v1's single run
  invited that reading; a different seed would have reported +7 to +21 pp

Validation exact match ranged 2.3% to 25.6% and the selected epoch moved between
1 and 3, so **checkpoint selection is itself seed-sensitive**.

On seed 2718 the paired bootstrap CI and McNemar's exact test disagree (CI
excludes zero, p = 0.070). The pre-registered criterion is the CI, so it counts;
the report states that this seed is at the margin and is not a clear win alone.

**Generalisation stays poor in every seed:** 11/16, 12/16 and 7/16 held-out
families score zero. v1's central negative finding is not a seed artefact.

**Disclosed deviation.** Seed 1337 reuses the v1 run rather than repeating it
(identical configuration and seed). The pre-registration said it would be re-run
as a reproducibility check; it was not, because system RAM made serial time the
binding constraint.

---

## E2 -- coverage versus depth

**Question.** At a fixed example budget, does covering more scenario families
beat seeing more examples of fewer?

**Design.** Two arms, **64 examples each**, **8 per category in both**:

| | Arm A -- high coverage | Arm B -- low coverage |
|---|---|---|
| Families | 32 | 16 |
| Examples per family | 2 | 4 |

Three seeds per arm, the same three as E1, so the comparison is paired on seed
as well as on test example.

**Controls.** Equal counts; identical category distribution by construction;
priority distribution matched at design time by choosing Arm B's families to
minimise L1 distance to Arm A's (residual distance 10, down from 20 under a
random draw); identical prompt, LoRA settings, decoding, validation and test
splits, and seeds.

**Design limit recorded before running.** 32x3 / 16x6 = 96 per arm is
infeasible: only 6 of the 32 training families hold 6 or more examples. 64 is
the largest clean equal-count design the v1 training split supports, and it is
underpowered for small effects.

**Reporting rule, fixed before running.** A null result is reported as
"no detectable difference", never as evidence that the arms are equivalent.

### Result

See `reports/coverage_depth/COVERAGE.md`.

---

## E3 -- constrained decoding

**Question.** How much of the remaining failure is the decoder rather than the
model?

**Mechanism.** An exact prefix automaton over the single legal output template,
with a `LogitsProcessor` masking any token that would leave the language.
Guarantees by construction: one JSON object, exactly the five permitted keys in
order, correct types, only permitted enum values, no extra fields, no fence, no
prose.

**Verified in both directions before use:** all 300 real expected outputs
classify as complete; every prefix of every one classifies as a valid prefix (a
wrongly rejected legal prefix would dead-end generation mid-answer); and it
rejects the exact v1 failures (`"dns"`, `"internet"`, `"download"`), invented
categories, extra keys, wrong key order, fences, prose prefixes,
boolean-as-string, and out-of-range integers. 35 tests.

**Conditions.** zero-shot / few-shot / LoRA unconstrained (v1 evidence, reused),
plus few-shot and LoRA **constrained**.

**Fairness rule, fixed before running.** The identical mechanism is applied to
both constrained conditions. The headline is constrained-few-shot vs
constrained-LoRA. The unfair comparison (unconstrained few-shot vs constrained
LoRA) is computed only so it can be displayed and set aside; the gap between the
two is the decoder's contribution.

**Metrics separated** so the decoder's work cannot masquerade as the model's:
syntactic validity, schema validity, field correctness and exact match are
reported apart. Constraining makes the first two trivially 100%.

### Result

See `reports/constrained/CONSTRAINED.md`.

---

## E4 -- diagnostic suites

**Secondary. Never pooled into the primary benchmark.**

Seven suites built from the v2 **train and validation** splits only, so the
sealed test split stays reserved: unseen families, noisy text, irrelevant
detail, long tickets, missing user count, contradictory information,
out-of-domain input.

**Honest scoring.** Three suites destroy the ground truth by design, so each
declares what it can be scored on: `full`, `except_users` (that field removed
from the text and therefore excluded), or `schema_only` (no correct answer
exists; only format compliance is meaningful).

**Interpretation rule, fixed before running.** A drop on a suite means this
model handles that specific synthetic perturbation worse. It is not a claim
about safety, robustness or real-world reliability.

### Result

See `reports/diagnostics/DIAGNOSTICS.md`.

---

## E5 -- the sealed v2 test set

**192 examples, 32 new scenario families, 96 sealed test examples.**

| | |
|---|---|
| dataset sha256 | `b1001027089c4751457da1ce0430b3bdd097e97e005108855f2a114ba59aca3a` |
| split checksum | `587018f5a81bf60c3c1f0756e7a94480b157cf3793a968e8686e1a59342bc4f9` |
| **seal checksum** | `0122062c5432ed0253fb34c45a4164c79c230680121368e80db8aa04b9463489` |
| **max similarity to any v1 ticket** | **0.3313** across 57,600 exhaustive comparisons |
| family-name overlap with v1 | none |

**Why this evaluation is strong.** Every model evaluated on it was trained on v1
data *before the v2 catalogue existed*. These families were not merely held
out -- they were unavailable to train on.

**The seal is enforced in code:** `assert_not_sealed` raises if a training or
demonstration-selection path touches a sealed example, and `load_manifest_v2`
verifies a dedicated membership checksum on every load. CI re-checks it.

**Priority balance fixed without changing the rule** (which would have destroyed
v1/v2 comparability): `medium` is 31.8% of v2 against 12.7% of v1, and the
sealed test split holds 17 `medium` examples against v1's 3.

**Structural limitation, stated up front.** With 4 families per category and 1
allocated to train, the v2 training split necessarily holds a single severity
rank per category and contains no `critical` examples. That was accepted to get
96 sealed test examples rather than 48.

### Result

See `reports/sealed/SEALED.md`.

---

## Environment finding

The binding constraint on this machine is **5.9 GiB of system RAM**, not 4 GiB
of VRAM. Running a second Python process alongside a trainer paged it out and
made it **~19x slower** -- 138 minutes per epoch instead of 8, resident set 144
MB -- while `nvidia-smi` still reported 98% GPU utilisation throughout. The
failure is silent; the only symptoms are wall-clock and `OSError 1455` in
unrelated processes. All v2 GPU and CPU work is strictly serialised.
`PORTABILITY.md` documents the symptom for anyone reproducing on a small-memory
machine.

---

## Not claimed

No claim of production readiness, deployment readiness, alignment, safety,
fairness, real-ticket reliability, general capability improvement, or
latency/cost improvement. No QLoRA, MLflow, vLLM, RAG, agents, preference
optimisation or distributed training was added. The diagnostic suites are
explicitly not robustness claims.
