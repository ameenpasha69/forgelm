# Model card -- ForgeLM v2 adapters

Supplements the root `MODEL_CARD.md`, which describes the v1 adapter and is not
edited. This card covers what v2 adds: **a set of adapters rather than one**,
and what having several changes about the claims that can be made.

**These are LoRA adapters, not models.** Each must be loaded on top of
`Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. No foundation model was created and
no base weights are redistributed.

---

## The important change: there is no single adapter to point at

v1 shipped one adapter and one number. v2 trained the identical configuration
three times, changing only the seed:

| Seed | Selected epoch | Test exact match | Test schema valid | Zero-scoring families |
|---|---|---|---|---|
| 1337 | 2 | 11.6% | 79.1% | 11/16 |
| 2718 | 1 | 8.1% | 68.6% | 12/16 |
| 3141 | 3 | 22.1% | 94.2% | 7/16 |

**Spread: 14.0 pp on exact match, 25.6 pp on schema validity.**

The v1 adapter (seed 1337) is not the best of these and was never selected for
being good -- it was simply the first run. It happens to sit mid-range, which is
luck rather than merit.

### What this means if you use one

- **Do not read a single adapter's score as the method's score.** Any one of
  these three would have produced a defensible-looking v1 report with a number
  between 8% and 22%.
- **The direction is reliable, the magnitude is not.** All three beat few-shot
  prompting on exact match with a paired CI excluding zero. None of them
  supports a precise effect size.
- **Checkpoint selection is itself seed-sensitive.** The best validation epoch
  moved between 1 and 3, and validation exact match ranged 2.3% to 25.6%.

## Which adapter is shipped

`artifacts/lora_adapter/` (seed 1337) remains the committed artefact, unchanged
from v1, so v1's evidence keeps reproducing. The seed 2718 and 3141 adapters are
**not committed** -- they are ~35 MB each and add nothing a reader needs, since
what carries the finding is their per-example predictions, which *are*
committed under `reports/predictions/lora_seed*_test.jsonl`.

Regenerate either with:

```bash
python scripts/03_train_lora.py --seed 2718 --tag seed2718
```

## Coverage-arm adapters (E2)

Six further adapters exist transiently for the coverage-versus-depth
experiment (2 arms x 3 seeds, 64 training examples each). They are diagnostic
instruments, not deliverables, and are likewise not committed; their predictions
are.

## Evaluation surfaces

| Surface | n | What it tests |
|---|---|---|
| v1 frozen test | 86 | the original benchmark; seen, so no longer usable for development decisions |
| **v2 sealed test** | **96** | 32 scenario families that **did not exist** when these adapters were trained |
| v2 diagnostics | 7 suites | specific input perturbations, reported separately and never pooled |

The v2 sealed set is the stronger evidence of the two: its families were not
merely held out, they were unavailable to train on. Maximum similarity between
any v2 ticket and any v1 ticket is **0.3313** across 57,600 exhaustive
comparisons.

## Constrained decoding changes what "schema valid" means

With the constrained decoder (`forgelm.constrained`), schema validity is
**guaranteed by construction** rather than achieved by the model. Any report
using it must therefore separate:

- syntactic validity and schema validity -- trivially 100%, and say nothing
  about the model once constrained
- field correctness and exact match -- the only figures that remain informative

The same mechanism is applied to base and adapted conditions, so no adapter is
credited with what the grammar did.

## Limitations

Everything in the root `MODEL_CARD.md` still applies. v2 adds:

- **A single adapter's headline number is not reproducible to better than
  ~14 pp** across seeds on this task and dataset size.
- The v2 training split contains no `critical` examples (structural: 4 families
  per category, 1 allocated to train). Train on v1 + v2 combined if you train
  on v2 at all.
- Diagnostic suite results are **not** robustness claims. A drop on
  `noisy_text` means this model handles that specific synthetic perturbation
  worse, and nothing more.

## Not claimed

No production readiness, deployment readiness, alignment, safety, fairness,
real-ticket reliability, general capability improvement, or latency/cost
improvement. No QLoRA, quantisation, RAG, agents or preference optimisation was
used.

## Licence

Adapter weights and code MIT. Base model apache-2.0, governed by its own
licence, not redistributed here.
