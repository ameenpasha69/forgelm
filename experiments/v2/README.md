# ForgeLM v2 — auditing v1 rather than extending it

v1 answered its research question and, in doing so, exposed four specific
weaknesses. v2 addresses those and nothing else. **No v1 artefact is modified**:
the v1 dataset, predictions, run records, reports, adapter and headline result
all still reproduce byte-for-byte.

Preserved baseline: commit `3314025787773d851feb70e8b5ed36fde4af3dd3`,
re-verified before any v2 work began — 20/20 evidence audit, 142 tests, clean
tree, all three checksums matching.

---

## The headline correction

**v1 reported one training run.** Three seeds of the identical configuration:

| Seed | Test exact match | vs few-shot | 95% CI | Meets v1's criterion |
|---|---|---|---|---|
| 1337 | 11.6% | +10.5 pp | [+3.5, +18.6] | yes |
| 2718 | 8.1% | +7.0 pp | [+1.2, +14.0] | yes |
| 3141 | 22.1% | +20.9 pp | [+11.6, +30.2] | yes |

**The conclusion survives (3/3).** The magnitude does not: the seed spread
(**14.0 pp**) is larger than the weakest seed's entire effect (**+7.0 pp**).

So *"LoRA beats few-shot on this task"* is supported. *"LoRA beats few-shot by
about 10 points"* — which v1's single run invited — is **not**. v1's seed landed
mid-range by luck.

Validation exact match ranged **2.3% → 25.6%** and the selected checkpoint moved
between epoch 1 and 3, so checkpoint selection is itself seed-sensitive.
Generalisation stayed poor in every seed (11/16, 12/16, 7/16 held-out families
scored zero) — v1's central negative finding is not a seed artefact.

---

## The five experiments

| | Experiment | Addresses |
|---|---|---|
| **E1** | Multi-seed variance | v1's single training run |
| **E2** | Coverage vs depth, equal budget | v1's ablation accidentally dropped a family |
| **E3** | Constrained decoding | 18/86 v1 outputs had invented enum values |
| **E4** | Diagnostic suites | generalisation measured on one slice only |
| **E5** | Sealed v2 test set | v1's test split had been seen |

Everything was fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) — seeds,
conclusion rules, fairness rules, reporting rules — **before any v2 run
existed**.

### Why that mattered in practice

Seed 2718 landed at the margin: its bootstrap CI excludes zero while McNemar
gives p = 0.070. With the rule fixed in advance, the CI decides (as
pre-registered) and the disagreement is disclosed. Without it, there would have
been an obvious temptation to quote whichever test read better.

---

## Reading the evidence

| Report | What it holds |
|---|---|
| [`reports/multiseed/MULTISEED.md`](reports/multiseed/MULTISEED.md) | E1, every seed, none promoted |
| [`reports/coverage_depth/COVERAGE.md`](reports/coverage_depth/COVERAGE.md) | E2 |
| [`reports/constrained/CONSTRAINED.md`](reports/constrained/CONSTRAINED.md) | E3 |
| [`reports/diagnostics/DIAGNOSTICS.md`](reports/diagnostics/DIAGNOSTICS.md) | E4 |
| [`reports/sealed/SEALED.md`](reports/sealed/SEALED.md) | E5, the single sealed look |
| [`STATUS.md`](STATUS.md) | per-component status with evidence |
| [`DECISIONS.md`](DECISIONS.md) | every v2 choice, including the wrong ones |
| [`PORTABILITY.md`](PORTABILITY.md) | running this elsewhere |

Raw per-example predictions back every number. `scripts/06_audit.py` re-derives
them and fails if any recorded value cannot be reproduced — v2 faces the same
bar as v1.

---

## The sealed test set

192 examples across **32 scenario families that did not exist when any of these
adapters was trained**. Not merely held out — unavailable to train on.

| | |
|---|---|
| Sealed test examples | **96** (16 families) |
| Max similarity to *any* v1 ticket | **0.3313** across 57,600 exhaustive comparisons |
| Family-name overlap with v1 | none |
| Seal checksum | `0122062c5432ed02...` |

The seal is enforced in code, not by convention: `assert_not_sealed()` raises if
a training or demonstration path touches a sealed example, and the membership
checksum is re-derived on every load and in CI.

**Priority balance was fixed without changing the rule** — changing it would have
destroyed v1/v2 comparability. Choosing family severities and user-count scales
instead took `medium` from 12.7% to **31.8%**, and from **3 to 17** examples in
the test split.

---

## Two things I got wrong, and fixed

Recorded in [`DECISIONS.md`](DECISIONS.md) rather than quietly corrected.

**V-004 — the binding constraint was never VRAM.** It is **5.9 GiB of system
RAM**. Running a second Python process alongside a trainer paged it out and made
it **~19× slower** — 138 min/epoch instead of 8, resident set 144 MB — while
`nvidia-smi` reported 98% GPU utilisation throughout. One run reached epoch 3 of
8 in 415 minutes before being killed. The failure is silent; the only symptoms
are wall-clock and `OSError 1455` elsewhere, which I initially dismissed as
noise.

**V-008 — my first coverage-arm design was confounded.** Drawing arm B's
families at random left a 10-point critical+high gap against arm A, entangling
coverage with label mix — the *same* defect that made v1's ablation hard to
read. Fixed by matching priority distributions at design time; the residual is
reported because only six family pairings exist per category.

---

## Not claimed

No production readiness, deployment readiness, alignment, safety, fairness,
real-ticket reliability, general capability improvement, or latency/cost
improvement. No QLoRA, MLflow, vLLM, RAG, agents, preference optimisation or
distributed training. The E4 diagnostics are explicitly **not** robustness
claims: a drop on a perturbed suite means this model handles that specific
synthetic perturbation worse, and nothing more.
