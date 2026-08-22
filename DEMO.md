# Demoing ForgeLM

A 10-minute walkthrough for showing this to someone. Ordered so that if you get
cut off after two minutes, the two minutes that happened were the right ones.

**Setup once, before the meeting:**

```bash
cd forgelm
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux
pip install gradio              # only needed for the visual demo
```

---

## Minute 0-2 — show the model working

```bash
python scripts/demo_app.py --compare
```

Open **http://127.0.0.1:7860**. Takes ~40 s to load both models.

Paste this ticket:

```
The VPN client disconnects every few minutes and has to be reconnected.
So far 14 people have reported it. Please investigate the concentrator
```

You get the adapted model's answer plus the unchanged base model's, side by
side. The base model wraps its answer in a markdown fence or invents an enum
value; the adapted one returns a clean object.

```json
{"category": "network", "priority": "high", "affected_service": "vpn",
 "is_security_incident": false, "users_affected": 14}
```

Then try something that is **not** a ticket -- `2 + 2 = ?` -- and watch it
confidently triage it anyway. That is a real finding from the diagnostics, not a
gotcha: the schema has no `not_a_ticket` value, so refusal is not representable.
Being able to say that about your own system is worth more than the demo working.

**If the laptop is not available:** the same notebook runs in Colab, verified on
a T4 --
[open it](https://colab.research.google.com/github/ameenpasha69/forgelm/blob/main/notebooks/forgelm_colab.ipynb).

---

## Minute 2-4 — the result, and what it is not

Open `README.md`. One table:

| System | Schema valid | Exact match |
|---|---|---|
| base, zero-shot | 27.9% | 0.0% |
| base, few-shot k=8 | 61.6% | 1.2% |
| base + LoRA | 79.1% | 11.6% |

Then say the thing that makes it credible:

> "The few-shot baseline is the one that matters. Zero-shot scored 0% mostly
> because the model wrapped its JSON in markdown fences -- eight examples in the
> prompt fixed that for free. If I'd only compared against zero-shot I'd have
> credited fine-tuning with solving a problem prompting already solved."

Then the correction v2 made:

> "v1 was one training run. Three seeds give 11.6%, 8.1% and 22.1%. All beat
> few-shot, so the conclusion holds -- but the spread is 14 points, wider than
> the weakest seed's whole effect. So 'LoRA beats few-shot' is supported;
> 'by about 10 points' isn't."

---

## Minute 4-6 — the strongest result

On the **sealed** v2 test set: 96 examples from 32 scenario families that did
not exist when the adapter was trained.

| | Exact match |
|---|---|
| few-shot + grammar | 9.4% |
| **LoRA + grammar** | **40.6%** |

+31.2 pp, 95% CI [+20.8, +41.7], p < 0.0001.

Why the grammar matters:

> "Constrained decoding makes illegal output unrepresentable. I apply it to both
> systems, because otherwise I'd be crediting the adapter with what the decoder
> did. The report computes the flattering comparison -- unconstrained few-shot
> against constrained LoRA, +15.1 pp -- purely so it can be shown and rejected."

---

## Minute 6-8 — prove the numbers are real

```bash
python scripts/06_audit.py
```

27 checks: dataset regenerates from its seed, checksums match, the split has not
moved, the sealed test membership verifies, every metric recomputes from raw
per-example predictions, no secrets, no placeholders, no unsupported claims.

Then the line that usually lands:

> "The CI job runs all of that on a GitHub runner with no GPU and no model
> weights. Every headline number re-derives from the committed predictions on a
> machine that has never downloaded the model."

And:

> "The notebook was run on a Colab T4 -- different OS, different torch, different
> CUDA build. It selected the same checkpoint, stopped at the same epoch, and
> produced exact-match identical to four decimal places."

---

## Minute 8-10 — the part most projects skip

Open `experiments/v2/DECISIONS.md` and `STATUS.md`.

> "Every mistake I made is written down. The adapter-liveness check that would
> have accepted an untrained adapter, because PEFT zero-initialises B and I was
> counting any non-zero tensor. The 19x training stall from running tests
> alongside training on a 6 GB machine, which looked healthy in nvidia-smi the
> whole time. A coverage experiment whose first design reproduced the exact
> confound it was meant to fix. And a conclusion I got wrong mid-project and
> corrected."

That last one is worth telling properly:

> "When I equalised the decoder, the advantage stopped being significant on the
> v1 test set, and I wrote that up as 'the gain was mostly formatting'. Then the
> same comparison on the sealed set gave +31 points at p<0.0001. It was an
> underpowered test set, not a dissolving effect. The correction is in the
> experiment card, not quietly edited out."

---

## Questions you should expect

**"Why such a small model?"**
0.5B is the smallest ungated Apache-2.0 model that follows instructions well
enough for a real zero-shot baseline. Bigger would have been slower and no more
informative. `DECISIONS.md` D-004 has the four candidates and why each was
rejected -- Qwen3-0.6B was rejected for having thinking mode on by default,
which conflicts with deterministic greedy decoding.

**"Synthetic data is a cop-out."**
Real tickets are personal data. Synthetic gives a clean licence story and labels
correct by construction. The cost is stated in the dataset card: nothing here
predicts behaviour on real tickets, and the priority rule is invented.

**"Is 86 test examples enough?"**
No, and that is why every number carries a bootstrap interval and a paired
McNemar test, and why the sealed v2 set was built with 96 better-balanced
examples. v1's test split had 3 `medium` examples; v2's has 17.

**"Did it actually generalise?"**
Partly. 7 of 16 sealed families still score zero. The adapter learned the output
contract far better than the task. That's in the README, not buried.

**"What would you do next?"**
More scenario families rather than more examples per family -- E2 showed that at
a fixed budget, redistribution makes no detectable difference, so coverage is
the lever. Multiple seeds as standard. And an explicit `not_a_ticket` enum value
so refusal is representable.

---

## If something breaks live

- **Gradio missing** -> `python scripts/demo_app.py --cli --compare`
- **No GPU** -> everything except the demo still runs: `pytest tests/ -m "not slow"`,
  `python scripts/04_report.py --split test`, `python scripts/06_audit.py`
- **Nothing works** -> the reports are committed markdown. `reports/RESULTS.md`
  and `experiments/v2/reports/SEALED.md` read fine on GitHub.
