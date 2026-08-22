# E3 -- constrained decoding

v1 left 18 of 86 outputs containing an enum value that does not exist -- `"dns"`, `"internet"`, `"display"`, `"audio"` -- almost always a noun copied from the ticket. A decoder that cannot emit an illegal token makes that failure class impossible. This measures how much of the remaining gap that removes, and how much it does not.

## Separated metrics

Constraining makes syntactic and schema validity trivially 100%. Those columns therefore say nothing about the model once constrained; only the last two do.

| # | Condition | Decoding | Syntactic (strict JSON) | Schema valid | Invalid enum | Mean field acc | **Exact match** |
|---|---|---|---|---|---|---|---|
| 1 | base zero-shot | unconstrained | 5.8% | 27.9% | 17.4% | 45.4% | **0.0%** |
| 2 | base few-shot k=8 | unconstrained | 100.0% | 61.6% | 38.4% | 63.9% | **1.2%** |
| 3 | base + LoRA | unconstrained | 100.0% | 79.1% | 20.9% | 70.0% | **11.6%** |
| 4 | base few-shot k=8 | **constrained** | 100.0% | 100.0% | 0.0% | 70.2% | **8.1%** |
| 5 | base + LoRA | **constrained** | 100.0% | 100.0% | 0.0% | 72.8% | **16.3%** |

## The fair comparison: constrained few-shot vs constrained LoRA

Both conditions use the identical constraint mechanism, so any difference is attributable to the adapter and not the decoder.

- constrained few-shot: **8.1%** exact match
- constrained LoRA: **16.3%** exact match
- difference **+8.1 pp**, 95% CI [-2.3, +18.6] pp, McNemar p = 0.1892 -- not distinguishable from zero

## What constraining buys each model

| Model | Unconstrained | Constrained | Difference | 95% CI | p |
|---|---|---|---|---|---|
| fewshot | 1.2% | 8.1% | +7.0 pp | [+2.3, +12.8] pp | 0.0312 |
| lora | 11.6% | 16.3% | +4.7 pp | [+1.2, +9.3] pp | 0.1250 |

## The comparison this report refuses to headline

Comparing **unconstrained few-shot** against **constrained LoRA** gives +15.1 pp -- against +8.1 pp for the like-for-like comparison. The difference between those two numbers is the decoder's contribution, and quoting the first as evidence for the adapter would be crediting LoRA with work the grammar did. It is computed here only so it can be shown and set aside.

## What this does not show

Constrained decoding guarantees a well-formed object with legal values. It cannot make a value *correct*. Any remaining gap after constraining is a model problem, not a format problem -- and that is precisely what the exact-match column isolates.
