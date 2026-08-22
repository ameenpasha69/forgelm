# ForgeLM v2 -- decision log

Continues `DECISIONS.md` at the repository root, which covers v1 (D-001 to
D-015) and is not edited. v2 decisions are numbered V-001 onward.

Append-only. A reversal gets a new entry rather than an edit.

---

## V-001 -- Continue on a branch; v1 evidence is immutable

**Decision.** All v2 work happens on `v2-continuation`, in `experiments/v2/`.
No v1 dataset, prediction file, run record, report, adapter or metric is
modified, and the v1 headline result is never recomputed or replaced.

**Gate run before touching anything**, and passing:

```
commit 3314025787773d851feb70e8b5ed36fde4af3dd3
20/20 evidence audit, 142 tests, working tree clean
raw sha256 aba72022...  processed sha256 63c56f50...  split 994141c6...
```

**Why.** A continuation that quietly improves the numbers it is auditing is
worthless. Separating v1 evidence from v2 evidence is what makes it possible
to say "the original conclusion survived" and mean something.

---

## V-002 -- Pre-register before running

**Decision.** `PREREGISTRATION.md` was written and committed **before** any v2
training run, fixing seeds, conclusion rules, fairness rules and reporting
rules while the answers were still unknown.

**Why it mattered in practice.** E1's conclusion rule
(survives / partially survives / does not survive) was fixed before the seeds
ran. Seed 2718 later landed at the margin -- its bootstrap CI excludes zero
while McNemar gives p = 0.070. Had the rule not been fixed in advance, there
would have been an obvious temptation to pick whichever test gave the tidier
answer. Instead the CI decides (as pre-registered) and the disagreement is
disclosed.

---

## V-003 -- Three seeds, all reported, none promoted

**Decision.** Seeds `1337`, `2718`, `3141` on the exact v1 primary
configuration. Every seed reported; the best is never promoted.

**Result.** Exact match 11.6% / 8.1% / 22.1%. All three clear v1's criterion,
so the conclusion **survives** -- but the spread (14.0 pp) is **larger than the
weakest seed's entire effect** (+7.0 pp).

**What that changes.** "LoRA beats few-shot" is supported. "LoRA beats few-shot
by about 10 points" -- which v1's single run invited -- is not. v1's seed
happened to land mid-range.

**Disclosed deviation.** Seed 1337 reuses the v1 run rather than repeating it.
It is the identical configuration and seed, so it is a legitimate member of the
set, but the pre-registration said it would be re-run as a reproducibility
check and it was not. Reason: the machine has 5.9 GB of RAM and a repeat costs
~40 minutes of strictly serial time. Recorded here because an undisclosed
deviation is how pre-registration stops meaning anything.

---

## V-004 -- The binding constraint is system RAM, not VRAM

**Context.** v1 attributed its sizing decisions to 4 GiB of VRAM. That was
incomplete.

**Observed.** Running a second Python process (a test suite, a dataset build)
alongside a trainer on this 5.9 GiB machine pushed the trainer into swap. It did
not crash -- it slowed by **~19x**, from 8 minutes per epoch to 138, with a
resident set of 144 MB. Seed 1337's first attempt reached epoch 3 of 8 in 415
minutes before being killed.

**Decision.** One Python process at a time. GPU work and CPU work are
serialised, not overlapped.

**Why it is recorded rather than quietly fixed.** The failure mode is silent.
`nvidia-smi` showed 98% GPU utilisation throughout, which looks healthy. The
only symptoms were `OSError 1455` in unrelated test runs -- which I initially
dismissed as environmental noise -- and wall-clock. Anyone reproducing this on a
small-memory machine will hit it, so `PORTABILITY.md` documents the symptom.

---

## V-005 -- Constrained decoding as an exact prefix automaton

**Decision.** Rather than a general JSON grammar, an exact automaton over the
one legal output template, with a `LogitsProcessor` masking every token that
would leave the language.

**Why.** The target language is tiny and completely fixed: five keys in a fixed
order, three closed enums, a boolean, a bounded integer. A general grammar would
be more code and weaker guarantees. This gives, by construction: one object,
exactly the permitted keys in order, correct types, only permitted enum values,
no extra fields, no fence, no prose.

**Verified in both directions before use.** All 300 real expected outputs
classify as complete, and every prefix of every one classifies as a valid
prefix -- a wrongly rejected legal prefix would dead-end generation mid-answer,
which is a silent way to handicap the model. It rejects the exact v1 failures
(`"dns"`, `"internet"`, `"download"`), invented categories, extra keys, wrong
key order, fences, prose prefixes, boolean-as-string, and `users_affected` of 0,
`01` or `99999`.

**Assumption made explicit.** The slot matcher takes the longest match and
commits, which is only safe while no permitted value is a strict prefix of
another. A test asserts that property, so a future schema change that breaks it
fails loudly rather than silently mis-parsing.

**Fairness rule.** The identical mechanism is applied to base and adapted
conditions. The headline is constrained-few-shot vs constrained-LoRA. The unfair
comparison (unconstrained few-shot vs constrained LoRA) is computed only so it
can be shown and set aside.

---

## V-006 -- v2 priority balance fixed without changing the rule

**Problem.** v1's test split held only 3 `medium` examples, because `medium`
requires a rule score of exactly 2. Priority macro-F1 on that split was close to
meaningless.

**Rejected.** Changing the priority rule. It would have improved the balance and
destroyed comparability between v1 and v2 -- the two datasets would no longer be
measuring the same task.

**Decision.** Keep the v1 rule unchanged and choose family severities and
user-count scales so that score-2 combinations occur often (severity 1 with
10-49 users; severity 2 with fewer than 10; severity 0 with 50+).

**Result.** `medium` is 31.8% of v2 against 12.7% of v1, and the sealed v2 test
split holds 17 `medium` examples against v1's 3.

---

## V-007 -- The v2 test set is sealed in code, not by convention

**Decision.** `splits_v2.assert_not_sealed()` raises if a training or
demonstration-selection path touches a sealed example, and the manifest carries
a dedicated `test_membership_checksum` that `load_manifest_v2` verifies on every
load.

**Why not just a README note.** A convention is something someone has to
remember at the moment it matters. An exception is not. The checksum is narrow
on purpose -- it answers only "is this the same set of sealed examples?" -- so
unrelated manifest additions cannot invalidate it and a quiet change to what is
sealed cannot hide.

**Not-a-paraphrase, measured.** Every v2 ticket was compared against every v1
ticket: 57,600 comparisons, maximum similarity **0.3313**, zero family-name
overlap. "These are new situations" is a measurement here, not a claim.

---

## V-008 -- Coverage-vs-depth arms are matched on labels, not just on count

**Context.** v1's data-size ablation was confounded: the smaller arm lost a
scenario family as a side effect of stratifying by category.

**First attempt at v2.** Equal counts (64 each) and identical category
distributions, with Arm B's two families per category drawn by a seeded shuffle.
That produced a **10-point gap in critical+high** between the arms -- coverage
confounded with label mix, the same class of defect v1 had.

**Decision.** Choose Arm B's families to minimise L1 distance to Arm A's
priority distribution. This is a design-time balance on an *input* distribution,
decided before any model ran and without reference to any result.

**Result.** Residual L1 distance halved, 20 -> 10. Not zero: only six family
pairings exist per category, so an exact match is unreachable. The residual is
reported in the arm configs and in the report rather than smoothed over.

**Design limit, recorded before results.** 32x3 / 16x6 = 96 per arm is
infeasible because only 6 of the 32 training families hold 6 or more examples.
32x2 / 16x4 = 64 is the largest clean equal-count design the v1 training split
supports, and it is underpowered for small effects.

---

## V-009 -- Diagnostics declare what they can honestly be scored on

**Problem.** Three of the seven diagnostic suites destroy the ground truth by
design: removing the user count makes `users_affected` unknowable; adding a
contradictory count makes it ambiguous; out-of-domain input has no correct
answer at all. Scoring exact match on those would produce numbers that look
meaningful and are not.

**Decision.** Each suite declares a scoring mode -- `full`, `except_users`
(that field excluded), or `schema_only` (format compliance only) -- and the
report honours it.

**Interpretation rule.** A drop on `noisy_text` means this model handles that
specific synthetic perturbation worse. It is not a robustness claim, and the
report says so explicitly. Diagnostics are never pooled into the primary
benchmark.

---

## V-010 -- Portability without a single requirements file

**Decision.** Five files: `base`, `windows-cuda` (the exact historical stack,
preserved verbatim), `linux-cuda`, `colab` (deliberately no torch, since
replacing Colab's build is the usual way to break its GPU runtime), and `cpu`.

**Why.** One `requirements.txt` pinning `torch==2.13.0+cu126` forces a Windows
CUDA build on every platform, including CI. The CPU file matters most for a
reader who wants to *check* the work rather than repeat it: every headline
number recomputes from the committed per-example predictions with no GPU and no
model weights, and CI runs exactly that path offline.
