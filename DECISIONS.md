# ForgeLM -- decision log

Every non-obvious choice, why it was made, what was rejected, and what evidence
existed at the time. Decisions are append-only: if one is reversed, a new entry
records the reversal rather than editing history.

Dates are the date the decision was taken (2026-08-22 unless stated).

---

## D-001 -- Greenfield start

**Context.** The working directory `D:\projects` was empty. `git status` reported
"not a git repository". There was no prior ForgeLM code, notebook, dataset or
result to audit, preserve or reuse.

**Decision.** Build from scratch in `D:\projects\forgelm`, `git init` there.

**Evidence.** `ls -laR` returned only `.` and `..`; see `STATUS.md` audit row.
No file was deleted, because none existed.

---

## D-002 -- Run the experiment locally rather than only writing a Colab notebook

**Context.** The brief anticipated that GPU training might be unavailable and
that full training might have to be marked unverified. Probing the machine found
an NVIDIA GeForce GTX 1650: 4 GiB VRAM, compute capability sm_75 (Turing),
driver 592.82.

**Decision.** Execute the whole experiment locally on that GPU, and additionally
ship a Colab notebook for reproduction elsewhere.

**Why it matters.** It converts "implemented but unverified" into "verified"
for training and evaluation -- the difference between a repository that
describes an experiment and one that has run it.

**Evidence.**
```
torch 2.13.0+cu126   cuda build 12.6   available True
arch_list ['sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','sm_90']
gpu NVIDIA GeForce GTX 1650 sm_75 4.0 GiB
fp16 matmul OK -> CUDA FULLY FUNCTIONAL
```

**Constraint this imposes.** 4 GiB is the binding limit on every subsequent
sizing decision. See D-009.

---

## D-003 -- Task: helpdesk ticket triage to a strict five-field JSON schema

**Candidates compared.**

| Candidate | Deterministic eval | Data-gen difficulty | Leakage risk | Colab/4 GiB feasible | Legal/ethical risk | Likely to show an adaptation effect |
|---|---|---|---|---|---|---|
| **Ticket triage -> typed JSON** | **High** -- closed enums, exact match, macro-F1 | Medium | Controllable via scenario families | Yes -- short inputs, ~38-token outputs | Low (synthetic) | **High** -- format + label + rule learning all measurable |
| Structured extraction from product requests | High | Medium | Medium | Yes | Low | Medium -- mostly copying spans |
| NL instruction -> constrained action schema | High | High -- schema design is the whole job | Medium | Yes | Low | Medium |
| Structured summarisation of synthetic records | Low -- needs BLEU/ROUGE or a judge | Low | High | Yes | Low | Low -- metrics are unreliable |

**Decision.** Ticket triage to strict JSON.

**Why.** It is the only candidate where *every* output field is objectively
checkable without a judge model, and it exercises three distinct abilities that
can be measured separately: format compliance (valid JSON, no fences), closed-set
classification (`category`, `affected_service`), and an inferable rule
(`priority` as a function of severity and blast radius). Open-ended conversation
was excluded explicitly, per the brief.

**Rejected.** Summarisation -- its only cheap metrics are BLEU/ROUGE, which the
brief forbids as a sole evaluation, and rightly so.

---

## D-004 -- Base model: Qwen2.5-0.5B-Instruct pinned to a commit

**Candidates researched from primary sources** (Hugging Face model cards and the
`/api/models` endpoint, 2026-08-22):

| Model | Licence | Gated | Params | Verdict |
|---|---|---|---|---|
| **Qwen/Qwen2.5-0.5B-Instruct** | apache-2.0 | No | 0.49B (0.36B non-emb) | **Selected** |
| HuggingFaceTB/SmolLM2-360M-Instruct | apache-2.0 | No | 0.36B | Backup; smaller, weaker instruction following |
| Qwen/Qwen3-0.6B | apache-2.0 | No | 0.6B | Rejected -- thinking mode on by default |
| meta-llama/Llama-3.2-1B-Instruct | llama3.2 | **manual** | 1.24B | Rejected -- not permissive, needs credentials |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | apache-2.0 | No | 1.1B | Rejected -- 2x memory, older, weaker |

**Decision.** `Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`.

**Why this one.**
- It is the *smallest* ungated Apache-2.0 model that still follows instructions
  well enough for a meaningful zero-shot baseline. The brief says not to pick a
  larger model because it looks more impressive, and 0.5B is genuinely enough.
- Qwen3-0.6B was rejected for a specific technical reason, not size: its chat
  template enables thinking mode by default and its own documentation warns
  against greedy decoding in that mode. This experiment requires deterministic
  greedy decoding in every condition, so Qwen3 would have introduced either a
  confound or a non-default configuration to explain away.
- Llama-3.2-1B is `gated: manual` and carries the Llama 3.2 Community Licence,
  which is not permissive. It would also have made the repository depend on a
  credential.

**Pinning.** The revision is a commit SHA, not `main`. A branch can move; a
commit cannot. The SHA is written into every ledger record.

**Verified directly, not from memory:**
```
tokenizer class Qwen2Tokenizer   vocab 151643
pad <|endoftext|>   eos <|im_end|>   has chat_template: True
rendered: '<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSER<|im_end|>\n<|im_start|>assistant\n'
layers 24  hidden 896  max_pos 32768  tie_word_embeddings True
```

**Note on terminology.** No quantisation is used, so this is **LoRA**, not
QLoRA. The brief's caution applies and is respected: the term QLoRA appears
nowhere as a description of what was run.

---

## D-005 -- Synthetic data, generated programmatically

**Decision.** 300 examples generated by `src/forgelm/datagen.py` from 56 scenario
families x 8 template families.

**Why synthetic.** Real helpdesk tickets are personal data and could not be used
under the brief's constraints. Programmatic generation additionally gives labels
that are correct *by construction* rather than by annotator agreement, and full
lineage: every example records the scenario family, template family, and the
exact symptom/detail/ask/scope clauses it was assembled from.

**Why 300.** The brief allows 150-300. The top of the range was chosen because
the group-aware split leaves only 86 test examples, and statistical power was
already the tightest constraint on the conclusions.

---

## D-006 -- Split: group-aware on scenario family, stratified by category *and severity*

**First implementation.** Seeded shuffle of families within each category, dealt
4 train / 1 validation / 2 test.

**Problem observed.** That produced a validation split containing **zero
`critical` examples and one `medium` example**:
```
validation  n=43  priorities={'high': 10, 'low': 32, 'medium': 1}
```
Checkpoint selection would have been blind to the highest-priority class.

**Decision.** Order families within each category by `base_severity` (ties broken
by a seeded shuffle, so the result is not an artefact of alphabetical naming) and
deal them using a fixed pattern indexed by severity rank:
`[train, train, test, train, validation, test, train]`.

**Result.** Every split now contains all four priority classes:
```
train       n=171  {'critical': 24, 'high': 28, 'low': 98, 'medium': 21}
validation  n= 43  {'critical':  5, 'high': 13, 'low': 11, 'medium': 14}
test        n= 86  {'critical':  8, 'high': 35, 'low': 40, 'medium':  3}
```

**Integrity note.** This change was made **before any model was run**. No test
result influenced it, and the change was motivated by a property of the
*validation* split. It is recorded here rather than silently applied because
"we changed the split and then got better numbers" is exactly the pattern a
reader should be suspicious of.

**Residual limitation.** The test split has only 3 `medium` examples. Priority
macro-F1 is therefore noisy and is reported with that caveat rather than
smoothed away by re-engineering the label distribution.

---

## D-007 -- Template families intentionally span splits

**Decision.** The held-out unit is `scenario_family`. All 8 `template_family`
surface styles appear in all three splits.

**Why.** Template families are writing styles ("terse", "frustrated", "formal
ticket") and carry no label information. Holding a style out would measure
"can it handle unfamiliar prose", which is a different research question.
The generalisation axis under test is *situation*, and that is held out strictly.

**How it is checked rather than asserted.** An exhaustive O(n^2) character
4-gram Jaccard scan over all 44,850 cross-split pairs. Result:
```
no cross-split pair reaches similarity 0.8; maximum observed = 0.3255
```
A test asserts the maximum stays below 0.6, so the margin is monitored, not just
the threshold.

---

## D-008 -- Success criteria fixed before training

Declared in `src/forgelm/config.py::SUCCESS_CRITERIA` and committed before the
first training run.

**Primary.** LoRA must beat **both** unchanged-model baselines on test exact
match, *and* the paired bootstrap 95% CI of the difference against the
**stronger** baseline must exclude zero.

Requiring it to beat the stronger baseline is the point. The zero-shot baseline
scored 0.0% exact match; clearing that bar would prove nothing. The few-shot
baseline is the real competitor, and it is deliberately strong (k=8, one
demonstration per category, all drawn from train).

**Secondary.** `schema_valid_rate` must not regress and
`constraint_violation_rate` must not increase against the stronger baseline.

**Explicitly not claimed:** production readiness, deployment readiness,
generalisation beyond this synthetic dataset, safety, alignment, fairness,
robustness, latency or cost improvements, or any capability on real tickets.

---

## D-009 -- Precision fp16, batch 1 x accumulation 8, max_seq_len 320

**The memory constraint.** With a 151,936-token vocabulary, the logits tensor
dominates: `batch x seq x vocab x 4 bytes`. At batch 1 and seq 320 that is
~195 MB in fp32 for the loss alone; at batch 4 it would be ~780 MB, on a card
with ~3.9 GiB usable.

**Decision.** `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`
(effective batch 8), `max_seq_len=320`.

**Why 320 specifically.** Measured, not guessed. Over all 300 examples:
prompt 206-245 tokens, target 35-38 tokens, `max(prompt+target) = 283`.
320 covers the longest example with headroom and truncates nothing -- asserted
by a test.

**Why fp16 and not bf16.** `torch.cuda.is_bf16_supported()` returns `True` on
this device, but sm_75 (Turing) has **no native bf16 datapath** -- the support
is emulated. `select_precision()` therefore keys off compute capability
(`>= 8.0`) rather than trusting that flag, selects fp16 autocast with a gradient
scaler, and returns the reason in a dict that is written into the ledger.
LoRA parameters are explicitly upcast to fp32 so the optimiser never steps
half-precision master weights.

---

## D-010 -- Decoding is greedy and identical in every condition

`do_sample=False`, `num_beams=1`, `max_new_tokens=160`, identical system prompt
(SHA recorded per run), identical parser.

**Why 160 when targets are 38 tokens.** Deliberate generosity to the baselines.
A tight cap would convert base-model verbosity into `truncated` failures and
flatter the adapted model. 160 leaves ~120 tokens of headroom for a base model
that prefixes prose before its JSON. Observed truncation rate: 0.0% in every
condition.

**Why greedy.** The task has exactly one correct answer, so sampling temperature
buys nothing, and run-to-run variance would need averaging this project cannot
afford.

---

## D-011 -- The parser is lenient, and leniency is measured

The same parser runs in every condition. It strips markdown fences and recovers
the first balanced JSON object from surrounding prose. Both outcomes are
reported:

- `json_parse_rate_strict` -- the whole response was one bare JSON object
- `json_parse_rate_lenient` -- an object could be recovered

**Why.** A strict-only parser would score the zero-shot baseline at ~6% purely
for wrapping its answer in ```json fences, which is a formatting habit rather
than a triage failure. A lenient-only parser would hide that habit entirely.
Reporting both makes the gap itself a finding -- and the gap turned out to be
the single largest zero-shot/few-shot difference.

---

## D-012 -- Checkpoint selection on validation loss, confirmed generatively

**Decision.** `metric_for_best_model = eval_loss`, computed on the validation
split every epoch, with early stopping (patience 3). After the best checkpoint is
restored, the training script runs a *generative* evaluation on the same
validation split to confirm the selection against the real task metric.

**Why not select directly on validation exact match.** Generation is ~50x more
expensive than a forward pass on this GPU, and doing it every epoch for 8 epochs
would have dominated the training budget. Validation loss is a cheap monotone
proxy; the generative confirmation catches the case where it is not.

**What is never done.** Test data is not read during training, checkpoint
selection, prompt design or hyperparameter choice. `03_train_lora.py` does not
import the test split at all.

---

## D-013 -- Defects found in manual review round 1, and fixed

Manual inspection of a stratified sample (one example per category per split)
found four real defects that the automated checks had not been written to catch:

1. **`str.capitalize()` lower-cased the rest of the string**, turning "the CRM"
   into "the crm", "the VPN" into "the vpn", and "I try" into "i try". Fixed with
   an explicit first-character upcase. A regression test now asserts acronyms
   survive.
2. **Clause/scope contradictions.** Examples with `users_affected = 1` could draw
   a detail clause asserting others were affected ("colleagues on the same
   handset model report the same thing"). Four such clauses were rewritten, and
   `check_scope_consistency` now fails the build if one reappears.
3. **A priority term leaked into ticket text.** `email_spoofed_sender` had the
   ask "treat this as a priority please". Caught automatically by
   `check_priority_leak_terms` on the very first build, and rewritten.
4. **Awkward singular scope phrasing** on request-type tickets; a neutral variant
   ("Only one person is affected.") was added.

Recorded here because "manual review was performed" is a claim, and these are
the receipts.

---

## D-014 -- transformers 5.x compatibility handled by introspection

The installed stack resolved to `transformers 5.15.1`, a major version that has
renamed arguments over time (`evaluation_strategy` -> `eval_strategy`,
`Trainer(tokenizer=)` -> `Trainer(processing_class=)`).

**Decision.** Rather than pinning to an old version or guessing,
`supported_training_arguments()` introspects the installed
`TrainingArguments.__init__` signature, resolves known aliases, and **reports
every dropped key** into the run ledger.

**Why.** A silently-ignored training argument -- `eval_strategy` being dropped,
say -- would disable evaluation and checkpoint selection while the run still
appeared to succeed. Making the drop list an explicit artefact turns a silent
failure into a visible one.

**What it actually caught.** `warmup_ratio` (removed in transformers 5.x, only
`warmup_steps` survives) and `save_safetensors`. Warm-up is now converted to a
step count against the real total, instead of training starting at the full
learning rate on step 1 with nobody noticing.

---

## D-015 -- Adapter liveness must be checked on B, not on "any LoRA tensor"

**Context.** `load_adapted_model` refuses to return an inert adapter, because an
adapter that is mathematically identical to the base model would load cleanly,
reproduce base-model results exactly, and be misread as "fine-tuning did not
help". The original check was: *at least one tensor whose name contains `lora_`
is non-zero.*

**The defect.** That is not sufficient. The LoRA update is

```
delta_W = (alpha / r) * B @ A
```

PEFT initialises `A` randomly and `B` to **zeros**, precisely so an untrained
adapter is a no-op. A freshly initialised adapter therefore has roughly half its
LoRA tensors non-zero -- every `A` matrix -- while contributing exactly nothing.
The old check passed it.

**How it was found.** `tests/test_model_integration.py::
test_inert_adapter_is_rejected` builds a LoRA model, saves it *without training*,
and asserts the loader raises. It did not raise. The test was written to prove
the guard worked and instead proved it did not.

**Decision.** Liveness is now determined by `lora_B` alone: at least one `lora_B`
tensor must be non-zero. Both counts are still reported.

**Impact on results: none.** The trained adapter has 168/168 `lora_B` tensors
non-zero and passed under either rule. The bug was in the guard, not in the
experiment. It is recorded because a safety check that cannot catch the failure
it exists for is worse than no check -- it produces false confidence.
