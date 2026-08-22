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
| **E3 constrained decoding** | **implemented, tests verified** | 35 tests; accepts all 300 legal outputs and every prefix, rejects all v1 failure modes | evaluation runs pending |
| E2 coverage vs depth | **in progress** | 6 runs (2 arms x 3 seeds) executing | 64 examples/arm, underpowered for small effects |
| E4 diagnostics | **implemented, not yet run** | 7 suites built from v2 train+validation | 3 suites have no ground truth by design |
| Portability | **verified** | 5 requirements files; `PORTABILITY.md` | Linux/Colab paths written but not executed here |
| CI workflow | **implemented but unverified** | `.github/workflows/ci.yml`, CPU-only, offline | not yet run on GitHub Actions |
| Gradio demo checks | **implemented, not yet run** | `scripts/v2_03_demo_checks.py` | needs a GPU gap |
| Colab verification | **blocked** | no Colab access from this environment | see below |

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

## Colab: honestly unverified

This environment has no Colab access, so no Colab result is claimed and none is
invented. The notebook's GPU cells remain **implemented but unverified**; its
CPU-only cells were executed locally and passed.

To verify, open `notebooks/forgelm_colab.ipynb` on a T4 runtime and run all
cells. Record GPU, VRAM, Python, CUDA, torch, transformers, peft, runtime,
seeds, adapter sha256 and final metrics. A T4 is also sm_75, so
`modeling.select_precision()` takes the same fp16 branch as the reference
machine.

---

## Not claimed

No claim is made anywhere in v2 about production readiness, deployment,
alignment, safety, fairness, real-ticket reliability, general capability
improvement, or latency/cost. The diagnostic suites in E4 in particular are
**not** robustness claims: a drop on a perturbed suite means this model handles
that specific synthetic perturbation worse, and nothing more.
