# ForgeLM v2 -- status

Same vocabulary as the root `STATUS.md`. "Verified" means it ran and left
evidence; writing the code is never sufficient.

**v1 evidence is untouched.** Preserved at commit
`3314025787773d851feb70e8b5ed36fde4af3dd3`, re-verified before any v2 work
began: 20/20 evidence audit, 142 tests, clean working tree, all three dataset
checksums matching.

---

## v2 components

| Component | Status | Evidence | Limitations |
|---|---|---|---|
| v1 preservation gate | **verified** | 20/20 audit + 142 tests at `3314025` before branching | -- |
| Pre-registration | **verified** | `PREREGISTRATION.md` committed before any v2 run | -- |
| **E1 multi-seed** | **verified** | 3 seeds; `reports/multiseed/MULTISEED.md` | 3 seeds is a coarse variance estimate |
| **E5 sealed v2 dataset** | **verified** | 192 examples, 32 new families, seal checksum `0122062c...` | v2 train split has no `critical` examples (structural) |
| **E3 constrained decoding** | **verified** | 5 conditions; `reports/constrained/CONSTRAINED.md`; 35 tests | v1 split underpowered for the like-for-like comparison -- resolved by E5 |
| **E2 coverage vs depth** | **verified** | 6 runs; `reports/coverage_depth/COVERAGE.md` | 64 examples/arm, underpowered for small effects |
| **E4 diagnostics** | **verified** | 588 examples, 7 suites; `reports/diagnostics/DIAGNOSTICS.md` | 3 suites have no ground truth by design |
| Portability | **verified** | 5 requirements files; `PORTABILITY.md` | Linux/Colab paths written but not executed here |
| CI workflow | **verified** | run [32582858525](https://github.com/ameenpasha69/forgelm/actions/runs/32582858525) passed in 1m34s on a clean Ubuntu CPU runner, offline, no GPU and no model weights: checksums matched, split frozen (max cross-split similarity 0.3255), **v2 seal intact** (`0122062c5432ed02`), all three v1 conditions recomputed == recorded, report regenerated, 26/26 audit checks | Node 20 deprecation warning from `actions/checkout@v4` |
| Gradio demo checks | **verified** | **11/11** checks; `reports/demo_checks.json` | Gradio server not booted (optional dependency) |
| Full test suite | **verified** | **215 passed** | -- |
| Final evidence audit | **verified** | **27/27 checks passed**, `reports/EVIDENCE.md` | -- |
| Colab verification | **verified** | executed end to end on a real Tesla T4, 2026-08-23; training reproduced the local run's step count, early-stopping epoch and selected checkpoint, and all three exact-match figures matched to 4 dp | 1 Colab-only bug found and fixed (torchao/peft) |

---

## E1 result: the v1 conclusion survives, its magnitude does not

| Seed | Exact match | vs few-shot | 95% CI | McNemar p | Meets v1 criterion |
|---|---|---|---|---|---|
| 1337 | 11.6% | +10.5 pp | [+3.5, +18.6] | 0.0117 | yes |
| 2718 | 8.1% | +7.0 pp | [+1.2, +14.0] | 0.0703 | yes |
| 3141 | 22.1% | +20.9 pp | [+11.6, +30.2] | <0.0001 | yes |

**Verdict: survives (3/3).** Applied from the rule fixed before these runs
existed.

The finding v1 could not have made:

- **Seed spread is 14.0 pp -- larger than the weakest seed's entire effect
  (+7.0 pp).** "LoRA beats few-shot" holds. "LoRA beats few-shot by about 10
  points" does not; v1's seed landed mid-range.
- Validation exact match ranged **2.3% to 25.6%**, and the selected checkpoint
  moved between epoch 1 and 2. Checkpoint selection is itself seed-sensitive.
- On seed 2718 the bootstrap CI and McNemar disagree (CI excludes zero,
  p = 0.070). The pre-registered criterion is the CI, so it counts; the report
  states plainly that this seed is at the margin.
- **Generalisation stays poor in every seed** (11/16, 12/16, 7/16 held-out
  families score zero). The v1 finding that the adapter learns the output
  contract far better than the task is not a seed artefact.

---

## E2, E3, E4, E5 results

**E2 -- coverage vs depth: no detectable difference.** Mean **-0.0 pp**;
direction flips across seeds (+1.2, +1.2, -2.3); pooled means identical at
18.6%. Per the pre-registered rule this is *not* a claim of equivalence -- the
design is underpowered by construction.

**E3 -- constrained decoding.** A grammar requiring no training took schema
validity to 100% for both systems and lifted the *base* model by **+7.0 pp
(p = 0.031)**, more than it lifted the adapted one. So v1's headline was
measuring, in substantial part, format compliance available for free.

**E5 -- the sealed evaluation, which resolves E3.** 96 examples from 32 families
that did not exist when any adapter was trained:

| Condition | Schema valid | Exact match |
|---|---|---|
| zero-shot | 28.1% | 0.0% |
| few-shot | 67.7% | 3.1% |
| **LoRA** | 83.3% | **29.2%** |
| few-shot + grammar | 100% | 9.4% |
| **LoRA + grammar** | 100% | **40.6%** |

| Comparison | Difference | 95% CI | p |
|---|---|---|---|
| few-shot -> LoRA | +26.0 pp | [+16.7, +35.4] | <0.0001 |
| few-shot+grammar -> LoRA+grammar | **+31.2 pp** | [+20.8, +41.7] | **<0.0001** |

**Correction to an intermediate v2 interpretation.** When E3's like-for-like
comparison on the v1 split came back non-significant (+8.1 pp, p = 0.19), the
tempting summary was "the adapter's advantage dissolves once the decoder is
equalised". That was an over-generalisation from one underpowered test set. The
identical comparison on the sealed split gives +31.2 pp at p < 0.0001.
**Constraining the decoder does not eliminate the adapter's advantage.**

LoRA also scores *higher* on genuinely-unseen families (29.2%) than on v1's own
test split (11.6%), and zero-scoring families fall from 11/16 to 7/16 -- v1's
test families were harder than the task in general.

**E4 -- diagnostics.** Perturbation costs are small: typos, irrelevant detail
and length each cost ~2-3 pp, and long tickets *raise* schema validity to 97.9%.
The notable result is out-of-domain: **half of non-tickets still receive a
well-formed triage object**, and `2 + 2 = ?` produced an invented category
`"math"`. This is recorded as a **schema design limitation** -- there is no
`not_a_ticket` value, so refusal is not representable -- and explicitly not as a
robustness or safety claim.

---

## Findings that changed how the work was done

1. **The binding constraint is 5.9 GB of system RAM, not 4 GB of VRAM.**
   Running a test suite alongside a trainer paged it out: **~19x slower**, 138
   min/epoch instead of 8, resident set 144 MB, while `nvidia-smi` still showed
   98% GPU utilisation. Seed 1337's first attempt reached epoch 3 of 8 in 415
   minutes before being killed. The `OSError 1455` skips in earlier test runs
   were the early warning and were initially dismissed as noise. GPU and CPU
   work are now strictly serialised.
2. **The first coverage-arm construction was confounded.** Drawing Arm B's
   families at random left a 10-point critical+high gap against Arm A --
   coverage entangled with label mix, the same defect that made v1's ablation
   hard to interpret. Fixed by matching priority distributions at design time
   (L1 distance 20 -> 10); the residual is reported because only six family
   pairings exist per category.
3. **v2's priority balance was fixed without touching the rule.** Changing the
   rule would have improved balance and destroyed v1/v2 comparability. Choosing
   family severities and user scales instead took `medium` from 12.7% to 31.8%,
   and from 3 to 17 examples in the sealed test split.
4. **The v2 catalogue is measurably not a paraphrase of v1.** 57,600 exhaustive
   comparisons, maximum similarity 0.3313, zero family-name overlap.

---

## Colab: VERIFIED on a Tesla T4

Executed end to end on a real Colab T4 by the repository owner, 2026-08-23.
Every cell ran; the results below are theirs, not a local proxy.

| | Colab | Local reference |
|---|---|---|
| GPU | **Tesla T4**, 15,360 MiB, sm_75 | GTX 1650, 4,096 MiB, sm_75 |
| Driver / CUDA | 580.82.07 / 13.0 | 592.82 / 12.6 |
| OS | Linux 6.6.122 | Windows 11 |
| Python | 3.13.15 | 3.13.14 |
| torch | **2.11.0+cu128** (build 12.8) | 2.13.0+cu126 |
| `bf16 native` | **False** | False |

### Training reproduced independently

| | Local | Colab |
|---|---|---|
| Optimiser steps | 110 | **110** |
| Early stopping | epoch 5 of 8 | **epoch 5 of 8** |
| Selected checkpoint | epoch 2 (`checkpoint-44`) | **epoch 2 (`checkpoint-44`)** |
| Best eval loss | 0.0743 | **0.0732** |
| Wall clock | 2202 s | **204 s** (10.8x faster) |

Per-epoch validation loss: local `0.0891 / 0.0743 / 0.0954 / 0.0939 / 0.0977`
against Colab `0.0902 / 0.0732 / 0.1038 / 0.0991 / 0.1136`. Different OS,
different torch major-minor, different CUDA build, different GPU -- and the run
selected the same epoch and stopped at the same point.

### The headline metric reproduced exactly

| metric | zero-shot | few-shot | LoRA |
|---|---|---|---|
| `json_parse_rate_strict` | 0.0581 | 1.0000 | 1.0000 |
| **`exact_match`** | **0.0000** | **0.0116** | **0.1163** |
| `markdown_fence_rate` | 0.9419 | 0.0000 | 0.0000 |

All three exact-match figures are **identical to the reported v1 values** to
four decimal places.

### What did NOT reproduce bit-for-bit, and why

A handful of metrics moved by one to three examples out of 86:

| metric | local | Colab |
|---|---|---|
| `schema_valid_rate` (few-shot) | 0.6163 | 0.6279 |
| `schema_valid_rate` (LoRA) | 0.7907 | 0.8023 |
| `field: is_security_incident` (LoRA) | 0.9186 | 0.8837 |
| `field: category` (LoRA) | 0.5349 | 0.5465 |

This is expected and was predicted. Greedy decoding is deterministic *given
identical arithmetic*, but floating-point matmul results differ across GPU
architectures, so a few borderline generations diverge. Note that the zero-shot
condition is **identical on every metric** -- it has the least room for a
near-tie to flip -- and that `exact_match` is unmoved throughout: the flips
changed *wrong* answers into differently-wrong answers.

`EXPERIMENT_CARD.md` stated in v1 that bit-exact determinism was not claimed and
that results "may differ in the last decimal places across runs, and more on
different hardware". That is now measured rather than asserted.

### One Colab-only bug this found

peft raised `ImportError: Found an incompatible version of torchao ... 0.10.0,
but only versions above 0.16.0 are supported` inside `get_peft_model()`. Colab
preinstalls torchao 0.10.0; the notebook installed the latest peft, which probes
for torchao and raises on anything older than 0.16. **Invisible locally** -- this
machine has no torchao at all, so the probe returns False cleanly. Fixed by
uninstalling torchao before installing peft (ForgeLM never uses quantisation).

Two further Colab-fatal defects were found and fixed by inspection beforehand: a
`YOUR-USERNAME` placeholder in the clone URL, and `check=False` on that clone
swallowing the failure so it surfaced as a confusing `ImportError` nine cells
later.

---

## Not claimed

No claim is made anywhere in v2 about production readiness, deployment,
alignment, safety, fairness, real-ticket reliability, general capability
improvement, or latency/cost. The diagnostic suites in E4 in particular are
**not** robustness claims: a drop on a perturbed suite means this model handles
that specific synthetic perturbation worse, and nothing more.
