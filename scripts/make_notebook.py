"""Generate notebooks/forgelm_colab.ipynb.

The notebook is generated rather than hand-edited so that it stays consistent
with the package: the same functions, the same config values, the same
explanations. Regenerate with:

    python scripts/make_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "forgelm_colab.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ===========================================================================
md("""
# ForgeLM -- LoRA adaptation of a small instruction model, end to end

**Research question.** Can parameter-efficient LoRA fine-tuning measurably
improve a small, permissively licensed instruction model on a narrow
structured-output task, compared with zero-shot *and* few-shot use of the
unchanged base model, within a free Colab budget?

This notebook runs the whole experiment: build the data, prove it is not leaking,
measure two baselines, train an adapter, and evaluate once on a frozen test set.

**What this is not.** It is not a new foundation model, and nothing here is
production-ready. We are measuring one narrow thing carefully.

---

### Before you start: the concepts you need

Read this once. Every definition is tied to what this notebook actually does,
rather than being generic.

**Pre-training vs fine-tuning.** Qwen2.5-0.5B-Instruct was *pre-trained* on a
huge amount of text to learn language in general, then *instruction-tuned* to
follow requests. We are doing a third, much smaller step: adapting it to one
specific output format on ~171 examples. We are not teaching it English; we are
teaching it *our* JSON contract.

**Fine-tuning vs RAG.** Retrieval-augmented generation looks facts up at
question time and pastes them into the prompt. It is the right tool when the
model lacks *knowledge*. Our model does not lack knowledge -- it knows what a
VPN is. It fails to follow a *format and a labelling convention*. That is a
behaviour problem, and behaviour is what fine-tuning changes. Retrieval would not
help here.

**Full fine-tuning vs LoRA.** Full fine-tuning updates all 494 million weights:
you need optimiser state for every one, which will not fit in 4 GB, and with 171
examples you would overwrite the model's general ability. LoRA freezes the base
model and inserts small trainable low-rank matrices beside each weight matrix.
We train **8,798,208 parameters -- 1.75% of the model**. The frozen part cannot
be damaged, and the thing you save is tiny.

**LoRA vs QLoRA.** QLoRA additionally *quantises* the frozen base model to 4-bit
to save memory. We do **not** do that -- the fp16 model fits in 4 GB with room to
spare. So this is LoRA. Calling it QLoRA would be a false claim about what ran.

**Train / validation / test.** Train (171) updates the adapter. Validation (43)
chooses which epoch's checkpoint to keep. Test (86) is touched **once**, at the
very end. If you use test to pick anything, your test score stops being an
estimate of unseen performance and becomes a number you tuned towards.

**Data leakage.** If a test example resembles a training example, you measure
memorisation instead of generalisation. Our defence is structural: we split on
*scenario family*, so the test set contains 16 situations the model has never
seen. We then verify it empirically -- all 44,850 cross-split pairs compared,
highest similarity **0.326** against a 0.80 alarm threshold.

**Overfitting.** With 8.8M trainable parameters and 171 examples, the model can
memorise. That is why validation loss is checked every epoch and training stops
when it stops improving.

**Causal-LM loss and teacher forcing.** The model predicts each token from the
ones before it. During training we feed the *correct* previous tokens rather than
its own guesses -- teacher forcing. It makes training stable, but it also means
training loss is optimistic: at generation time one early mistake changes
everything after it. This is exactly why we report generated exact-match, not
just loss.

**Tokenization.** Text becomes integer tokens. Our prompts are 206-245 tokens and
our JSON answers are 35-38 tokens.

**Chat templates.** Instruction models expect a specific layout of special
tokens. Qwen uses ChatML:
`<|im_start|>system ... <|im_end|><|im_start|>user ... <|im_end|><|im_start|>assistant`.
We always build prompts with `tokenizer.apply_chat_template`. Hand-rolling this
format is the most common way to accidentally cripple a baseline and then
"beat" it.

**Truncation.** Cutting sequences at a length limit. We measured the longest
example at 283 tokens and set the limit to 320, so **nothing is truncated** --
verified, not assumed.

**Prompt masking.** A causal model would happily learn to predict the
instruction back at you. We set the label for every prompt token to `-100` so the
loss is computed **only over the JSON answer**. Get this wrong and the loss curve
still looks fine while the model learns the wrong thing.

**Why few-shot is a mandatory baseline.** If showing the model 8 examples in the
prompt works as well as training an adapter, the adapter is not worth it. Skipping
this baseline is the easiest way to make fine-tuning look better than it is. In
our run the few-shot baseline turned out to be *dramatically* stronger than
zero-shot, which is precisely why it needed to be measured.

**Why generation metrics can mislead.** A model can produce perfectly valid JSON
with entirely wrong values. That is why we report a ladder: parseable -> schema
valid -> exact match, plus per-field accuracy. Only the last one means "right".

**Why a small improvement proves little.** Our test set is 86 examples. A
difference of a few points is inside the noise. Every headline number here carries
a bootstrap confidence interval and every comparison carries a paired McNemar
test.

**What the adapter actually contains.** Not a model. A few hundred small matrix
pairs (A and B) that get added alongside the frozen base weights. It is useless
without `Qwen/Qwen2.5-0.5B-Instruct` at the exact pinned revision.
""")

# ---------------------------------------------------------------------------
md("""
## 1. Verify the environment

**What this does.** Prints the GPU, its compute capability, and how much memory
it has.

**Why it is needed.** Every sizing decision downstream depends on the answer.
The reference run used a GTX 1650 (4 GiB, sm_75). Colab's T4 (16 GiB, sm_75) is
the same architecture with more memory, so everything here fits comfortably.

**What may fail.** If `cuda available: False`, go to
*Runtime -> Change runtime type -> T4 GPU*. The notebook will still run on CPU
but training will take hours instead of minutes.

**How to read it.** Note `bf16 native`. On sm_75 this is `False`, and that is why
we use fp16 -- see the precision cell later.
""")
code("""
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
""")

code("""
import sys, platform
print("python  :", sys.version.split()[0])
print("platform:", platform.platform())
try:
    import torch
    print("torch   :", torch.__version__, "| cuda build:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        cap = p.major * 10 + p.minor
        print(f"gpu     : {p.name}  sm_{p.major}{p.minor}  "
              f"{p.total_memory/1024**3:.1f} GiB")
        print("bf16 native (needs sm_80+):", cap >= 80)
except ImportError:
    print("torch not installed yet -- the next cell installs it")
""")

# ---------------------------------------------------------------------------
md("""
## 2. Install dependencies and get the ForgeLM code

**What this does.** Installs the libraries and puts `src/forgelm` on the import
path.

**Why it is needed.** All the real logic lives in the package, not in this
notebook. That is deliberate: notebooks are for teaching and orchestration, but
logic that only exists in a notebook cannot be unit-tested. Every function you
call below is covered by the test suite in `tests/`.

**Inputs.** Nothing. **Outputs.** An importable `forgelm` package.

**What may fail.** If the `git clone` line 404s, either replace the URL with your
own fork, or upload the repository folder to the Colab file browser and set
`REPO` to that path.
""")
code("""
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.44", "peft", "accelerate", "datasets",
                    "scipy", "matplotlib"], check=False)
    # Replace with your fork if you have one.
    REPO = Path("/content/forgelm")
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/YOUR-USERNAME/forgelm.git",
                        str(REPO)], check=False)
else:
    REPO = Path.cwd()
    if not (REPO / "src").exists():
        REPO = REPO.parent           # running from notebooks/

sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)
print("repo:", REPO)

import forgelm
print("forgelm version:", forgelm.__version__)
""")

# ---------------------------------------------------------------------------
md("""
## 3. Set every seed

**What this does.** Seeds Python, NumPy and PyTorch, and reports what was
actually applied.

**Why it is needed.** So a rerun gives the same answer, and so the run record can
state exactly which seeds produced which number.

**How to read it.** Note `determinism_caveat`. We do *not* claim bit-exact
determinism: some fused CUDA kernels have no deterministic implementation, and
forcing `torch.use_deterministic_algorithms(True)` would crash. Greedy decoding
is deterministic in practice; training loss can wobble in the last decimal
places. Saying so is more useful than pretending otherwise.
""")
code("""
from forgelm.seeding import SEEDS, seed_everything
import json

print("named seeds:", json.dumps(SEEDS, indent=2))
applied = seed_everything(SEEDS["training"])
print("\\napplied:", json.dumps(applied, indent=2))
""")

# ---------------------------------------------------------------------------
md("""
## 4. Build the dataset

**What this does.** Generates all 300 examples from 56 scenario families and 8
writing styles, then writes them to `data/raw/`.

**Why it is needed.** The data is synthetic and generated by code, not
downloaded. That gives a clean licence story (no personal data, nothing scraped)
and labels that are correct *by construction*.

**Inputs.** The seed `20240517`. **Outputs.** 300 records with full lineage.

**How to read it.** Look at the example printed below: `expected_output` is the
target, and every other field is provenance -- which scenario, which style, which
clauses it was assembled from.
""")
code("""
from forgelm.datagen import generate_dataset, FAMILIES, TEMPLATE_FAMILIES

records = generate_dataset()
print(f"{len(records)} examples from {len(FAMILIES)} scenario families "
      f"x {len(TEMPLATE_FAMILIES)} writing styles")

r = records[0]
print("\\n--- one example ---")
print("ticket:", r["ticket_text"])
print("target:", r["expected_output_json"])
print("lineage:", {k: r[k] for k in
                   ("example_id", "scenario_family", "template_family",
                    "user_scale", "base_severity")})
""")

md("""
**Determinism check.** Generating twice must give identical output. If this
fails, something depends on dict ordering or an unseeded random call, and no
result from this notebook would be reproducible.
""")
code("""
assert generate_dataset() == records, "generation is not reproducible!"
print("PASS -- generation is deterministic")
""")

# ---------------------------------------------------------------------------
md("""
## 5. Validate the data

**What this does.** Runs every quality check: required fields, schema validity,
whether the `priority` labels match the published rule, whether
`users_affected` can actually be recovered from the text, duplicates, class
balance, and whether any priority-revealing word leaked into a ticket.

**Why it is needed.** A model trained on subtly broken labels produces a
confident, meaningless result. These checks run before anything expensive.

**What may fail.** `errors > 0` means stop. On the very first real build this
check caught a genuine bug: the ticket text for `email_spoofed_sender` contained
the word "priority", handing the model the answer.

**How to read it.** `errors: 0` is the gate. `info` findings are observations
worth recording, not problems.
""")
code("""
from forgelm import validate

report = validate.run_all(records, manifest=None)
print(f"errors={report['errors']}  warnings={report['warnings']}  "
      f"info={report['info']}")
for f in report["findings"]:
    if f["severity"] != "info":
        print(f"  {f['severity'].upper()}: {f['message']}")
assert report["errors"] == 0, "dataset failed validation"
print("\\nPASS -- no structural errors")
""")

# ---------------------------------------------------------------------------
md("""
## 6. Split the data -- group-aware and stratified

**What this does.** Assigns every example to train / validation / test, grouping
by **scenario family** so no situation appears in two splits.

**Why it is needed.** This is the single most important decision for whether the
result means anything. If `net_vpn_drops` appeared in both train and test, a good
test score would only prove memorisation.

Families are ordered by severity within each category and dealt with a fixed
pattern, which guarantees every split gets a spread of severities. An earlier
plain-shuffle version produced a validation split with **zero `critical`
examples** -- checkpoint selection would have been blind to the class that
matters most.

**How to read it.** Check that all four priority classes appear in all three
splits.
""")
code("""
from forgelm.splits import build_manifest, apply_split
from collections import Counter

manifest = build_manifest(records)
by_split = apply_split(records, manifest)

print("checksum:", manifest["checksum"])
for name, rows in by_split.items():
    pri = Counter(x["expected_output"]["priority"] for x in rows)
    print(f"{name:11s} n={len(rows):3d}  families={len({x['scenario_family'] for x in rows}):2d}  "
          f"priorities={dict(sorted(pri.items()))}")
""")

# ---------------------------------------------------------------------------
md("""
## 7. Prove there is no leakage

**What this does.** Two checks. First, structural: no scenario family may span
splits. Second, empirical: compare **every** cross-split pair of tickets on
character 4-gram Jaccard similarity and report the maximum.

**Why it is needed.** The structural guarantee is only as good as the assumption
that different families produce different text. The similarity scan tests that
assumption instead of trusting it.

**How to read it.** `max similarity` well below the 0.80 alarm threshold is the
evidence. In the reference run it was **0.326** -- a wide margin, not a near miss.
""")
code("""
findings = validate.check_cross_split_leakage(records, manifest)
for f in findings:
    print(f"[{f.severity:5s}] {f.check}: {f.message}")

errors = [f for f in findings if f.severity == "error"]
assert not errors, f"LEAKAGE DETECTED: {[f.message for f in errors]}"
print("\\nPASS -- no cross-split leakage")
""")

# ---------------------------------------------------------------------------
md("""
## 8. Load the model and tokenizer

**What this does.** Downloads Qwen2.5-0.5B-Instruct at a **pinned commit hash**
and chooses a numeric precision appropriate to the actual GPU.

**Why pin a commit.** `main` can move. A commit cannot. Every result records the
exact revision it was produced against.

**Why the precision logic matters.** `torch.cuda.is_bf16_supported()` returns
`True` on sm_75 hardware, but Turing has no native bf16 datapath -- the support is
*emulated*. So we key the decision off compute capability instead, pick fp16, and
keep the LoRA parameters in fp32 so the optimiser never steps half-precision
weights. Trusting that flag is a real trap.

**What may fail.** First run downloads ~1 GB.
""")
code("""
from forgelm.modeling import (BASE_MODEL_FACTS, load_tokenizer, load_base_model,
                              select_precision)
import json

print(json.dumps(BASE_MODEL_FACTS, indent=2))
precision = select_precision()
print("\\nprecision choice:", json.dumps(precision, indent=2))

tokenizer = load_tokenizer()
print("\\ntokenizer:", type(tokenizer).__name__, "| vocab:", tokenizer.vocab_size)
print("pad:", tokenizer.pad_token, "| eos:", tokenizer.eos_token)
""")

md("""
**Inspect the official chat template.** This is the exact string the model sees.
Note that it ends with `<|im_start|>assistant\\n` -- the *generation prompt*. Without
it, the model would try to continue the user's turn instead of answering.
""")
code("""
from forgelm.prompts import render_prompt, SYSTEM_PROMPT

example = by_split["test"][0]
prompt = render_prompt(tokenizer, example["ticket_text"])
print(prompt)
print("\\nprompt length:", len(tokenizer(prompt)["input_ids"]), "tokens")
""")

# ---------------------------------------------------------------------------
md("""
## 9. Baseline A -- zero-shot

**What this does.** Asks the unchanged model to do the task with only the
instruction, using greedy decoding.

**Why it is needed.** This is the "before" picture. Without it there is no
improvement to measure.

**Fairness rules, all enforced in code.** Identical system prompt in every
condition; identical decoding settings; identical parser; a generous 160-token
budget (targets are 38 tokens) so the baseline is never truncated unfairly.

**How to read it.** Watch the ladder: *parseable* -> *schema valid* -> *exact
match*. The reference run scored 100% / 27.9% / **0.0%**, and wrapped its answer
in ```json fences 94% of the time despite being told not to.
""")
code("""
from forgelm.generate import run_evaluation
from forgelm import metrics as M
from forgelm.config import DECODING

model = load_base_model()
test_records = by_split["test"]

zs = run_evaluation(model, tokenizer, test_records,
                    max_new_tokens=DECODING["max_new_tokens"],
                    batch_size=DECODING["batch_size"],
                    progress=lambda d, t: print(f"  {d}/{t}", end="\\r"))
zs_metrics = M.compute_metrics(zs)
print("\\n--- zero-shot ---")
for k in ("json_parse_rate_strict", "schema_valid_rate", "exact_match",
          "markdown_fence_rate"):
    print(f"  {k:26s} {zs_metrics[k]:.4f}")
print("  field accuracy:", zs_metrics["field_accuracy"])
""")

md("""
**Look at what it actually produced.** Metrics summarise; raw output explains.
This is where you see *why* the score is what it is.
""")
code("""
for row in zs[:2]:
    print("=" * 70)
    print("RAW      :", repr(row["raw"][:220]))
    print("parsed   :", row["parsed"])
    print("expected :", row["expected"])
    print("error    :", row["error_category"])
""")

# ---------------------------------------------------------------------------
md("""
## 10. Baseline B -- few-shot

**What this does.** Same model, same instruction, but with 8 worked examples in
the prompt -- one per category, **drawn only from the training split**.

**Why this is the baseline that matters.** If prompting alone matches
fine-tuning, the adapter is not worth its complexity. A weak few-shot baseline
would make LoRA look good for free, so this one is deliberately strong.

The code *asserts* that no demonstration comes from validation or test. Pulling
a demo from the test set would leak held-out data into a "base model" baseline.

**How to read it.** In the reference run this fixed the formatting problem
completely: strict JSON went 5.8% -> **100%** and category macro-F1 0.14 -> 0.54.
That is a big, honest result, and it raises the bar LoRA has to clear.
""")
code("""
from forgelm.prompts import select_demonstrations, FEWSHOT_K

demos = select_demonstrations(by_split["train"], k=FEWSHOT_K)
for d in demos:
    assert manifest["example_split"][d["example_id"]] == "train"
print("demonstrations:", [d["example_id"] for d in demos])
print("categories covered:", sorted({d["category"] for d in demos}))

fs = run_evaluation(model, tokenizer, test_records, demonstrations=demos,
                    max_new_tokens=DECODING["max_new_tokens"],
                    batch_size=DECODING["batch_size"],
                    progress=lambda d, t: print(f"  {d}/{t}", end="\\r"))
fs_metrics = M.compute_metrics(fs)
print("\\n--- few-shot (k=8) ---")
for k in ("json_parse_rate_strict", "schema_valid_rate", "exact_match",
          "markdown_fence_rate"):
    print(f"  {k:26s} {fs_metrics[k]:.4f}")
print("  field accuracy:", fs_metrics["field_accuracy"])
""")

# ---------------------------------------------------------------------------
md("""
## 11. Configure LoRA

**What this does.** Wraps the frozen base model with trainable low-rank adapters
on all seven linear projections in each transformer block.

**Why these settings.**
- `r=16` -- the rank of the low-rank update. Higher means more capacity and more
  overfitting risk on 171 examples.
- `alpha=32` -- the scaling factor, conventionally `2r`.
- `dropout=0.05` -- mild regularisation, because 8.8M parameters on 171 examples
  can memorise.
- Targeting the MLP projections as well as attention matters for *format*
  learning, which is most of what we are teaching.

**How to read it.** The trainable percentage should be ~1.75%, and the trainable
tensors should be **fp32** while the frozen base stays fp16. If the trainable
tensors were fp16, small optimiser updates would vanish into rounding and the
loss would mysteriously plateau.
""")
code("""
from forgelm.modeling import build_lora_model, parameter_report
from forgelm.config import TRAINING

config = {**TRAINING, "fp16": precision["fp16"], "bf16": precision["bf16"]}
del model  # free the baseline copy first
import torch, gc; gc.collect(); torch.cuda.empty_cache()

base = load_base_model()
lora_model, lora_config, n_upcast = build_lora_model(base, config)
params = parameter_report(lora_model)

print(f"trainable {params['trainable_params']:,} / {params['total_params']:,} "
      f"= {params['trainable_percent']}%")
print("trainable dtypes:", params["trainable_param_dtypes"])
print("frozen  dtypes:", params["param_dtypes"])
adapted = sorted({p.split(".")[-1] for n in params["trainable_module_names"]
                  for p in [n.split(".lora_")[0]]})
print("adapted projections:", adapted)
""")

# ---------------------------------------------------------------------------
md("""
## 12. Inspect the tokenized training data

**What this does.** Encodes the training examples and shows exactly which tokens
contribute to the loss.

**Why this is the most important cell in the notebook.** Prompt masking is
invisible when wrong. If the prompt tokens are left in the loss, the model spends
most of its gradient learning to recite the instruction back -- and the loss curve
looks completely normal while it happens.

**How to read it.** The *supervised span* must be the JSON answer and nothing
else. The *masked tail* should end with the generation prompt. `truncated: 0` is
required -- if anything is cut off, the model is being trained on incomplete
targets.
""")
code("""
from forgelm import training as T

train_enc = T.build_dataset(tokenizer, by_split["train"], config["max_seq_len"])
val_enc = T.build_dataset(tokenizer, by_split["validation"], config["max_seq_len"])

stats = T.truncation_stats(train_enc, config["max_seq_len"])
print("truncation:", {k: stats[k] for k in
                      ("n", "n_truncated", "truncation_rate", "total_tokens")})

sample = T.inspect_tokenization(tokenizer, train_enc[0])
print(f"\\n{sample['example_id']}: {sample['n_masked']} masked + "
      f"{sample['n_unmasked']} supervised")
print("\\nSUPERVISED (contributes to loss):")
print(" ", repr(sample["unmasked_text"]))
print("\\nMASKED tail (no loss):")
print(" ", repr("..." + sample["masked_text_tail"][-110:]))

check = T.verify_masking(tokenizer, train_enc + val_enc)
assert check["passed"], check["problems"]
print(f"\\nPASS -- masking verified on {check['n_checked']} examples")
""")

# ---------------------------------------------------------------------------
md("""
## 13. Smoke tests before committing to a full run

**What this does.** Three cheap checks: a forward pass (is the loss finite and
sane?), and a tiny overfit (can gradients actually change the output?).

**Why it is needed.** Full training takes tens of minutes. These take seconds and
catch broken labels, dead gradients and out-of-memory before you spend that time.

**How to read it.**
- The starting loss is **low (~0.19)**, and that is expected, not a bug: most
  tokens in the JSON are structural (`{"category": "`) and trivially predictable
  under teacher forcing. Only the *value* tokens are uncertain. This is exactly
  why loss is a poor headline metric here and exact match is the real one.
- The tiny overfit must drive loss down sharply. If it cannot memorise four
  examples, something is broken and full training is pointless.

**What may fail.** On a 4 GiB card this originally hit CUDA OOM: 4 examples x
280 tokens x 151,936 vocab in fp16 *and* fp32 *and* their gradients exceeds the
card. The fix is gradient accumulation over micro-batches of 1, which is also
what real training does.
""")
code("""
fwd = T.one_batch_forward(lora_model, tokenizer, train_enc, batch_size=2)
print("forward:", {k: fwd[k] for k in
                   ("loss", "perplexity", "n_supervised_tokens", "healthy")})
assert fwd["healthy"]

overfit = T.tiny_overfit(lora_model, tokenizer, train_enc,
                         steps=30, lr=1e-3, n_examples=4, micro_batch=1)
print(f"tiny overfit: {overfit['first_loss']} -> {overfit['final_loss']} "
      f"(passed={overfit['passed']})")
assert overfit["passed"], "gradients are not reaching the adapter"
print("\\nPASS -- safe to train")
""")

md("""
The tiny-overfit test **modified the adapter weights**, so we rebuild a fresh
model for the real run. Training on top of a deliberately overfitted adapter
would corrupt the experiment.
""")
code("""
del lora_model, base
gc.collect(); torch.cuda.empty_cache()

base = load_base_model()
lora_model, lora_config, _ = build_lora_model(base, config)
print("fresh LoRA model rebuilt")
""")

# ---------------------------------------------------------------------------
md("""
## 14. Train

**What this does.** Trains the adapter for up to 8 epochs, evaluating on the
validation split each epoch and keeping the best checkpoint by validation loss.

**Why these settings.** Effective batch 8 comes from `batch_size=1 x
accumulation=8`. Batch size 1 is not timidity -- with a 151,936-token vocabulary
the logits tensor dominates memory. Early stopping (patience 3) guards against
overfitting 171 examples.

**Test data is never touched here.** Checkpoint selection uses validation only.

**How to read it.** Training loss should fall steadily. Validation loss should
fall and then flatten or rise -- the epoch before it rises is the one worth
keeping. If validation loss rises immediately, the learning rate is too high.
""")
code("""
import time
trainer, applied = T.build_trainer(
    lora_model, tokenizer, train_enc, val_enc,
    output_dir="runs/notebook_train", config=config, seed=SEEDS["training"])
if applied["dropped"]:
    print("NOTE: TrainingArguments not supported by this version:",
          applied["dropped"])

t0 = time.time()
result = trainer.train()
print(f"\\ntrained in {time.time()-t0:.0f}s, {result.global_step} steps")
print("best checkpoint:", trainer.state.best_model_checkpoint)
print("best eval_loss :", trainer.state.best_metric)
""")

code("""
import matplotlib.pyplot as plt

hist = trainer.state.log_history
tr = [(h["epoch"], h["loss"]) for h in hist if "loss" in h]
ev = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]

plt.figure(figsize=(7, 4))
plt.plot([e for e, _ in tr], [l for _, l in tr], label="train loss")
if ev:
    plt.plot([e for e, _ in ev], [l for _, l in ev], "o-", label="validation loss")
    best = min(ev, key=lambda x: x[1])
    plt.axvline(best[0], ls="--", c="grey")
plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.legend(); plt.grid(alpha=.3)
plt.title("LoRA training"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 15. Save the adapter, then reload it from scratch

**What this does.** Saves the adapter, deletes everything from memory, and
reloads base + adapter through a clean path.

**Why it is needed.** "It worked in the training process" is not the same as
"it can be loaded later". Reloading is the only way to know the saved artefact
is usable.

**The critical check.** `load_adapted_model` verifies the LoRA weights are
actually non-zero and **raises** if they are not. A saved adapter whose B
matrices are all zero is mathematically identical to the base model -- it would
load without error and silently reproduce base-model results, which you would
then misread as "fine-tuning did not help".

**How to read it.** The adapter is ~35 MB, not 1 GB. That is the whole point of
LoRA: you ship the delta, not the model.
""")
code("""
from forgelm.modeling import load_adapted_model
from pathlib import Path

adapter_dir = Path("artifacts/lora_adapter_notebook")
trainer.model.save_pretrained(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))

size = sum(f.stat().st_size for f in adapter_dir.iterdir() if f.is_file())
print(f"saved {size/1e6:.1f} MB to {adapter_dir}")
for f in sorted(adapter_dir.iterdir()):
    print(f"  {f.name:32s} {f.stat().st_size/1e6:8.2f} MB")

del trainer, lora_model, base
gc.collect(); torch.cuda.empty_cache()

adapted_model, verification = load_adapted_model(str(adapter_dir))
print("\\nverification:", json.dumps(verification, indent=2))
""")

# ---------------------------------------------------------------------------
md("""
## 16. Final evaluation on the frozen test set

**What this does.** Runs the adapted model on the same 86 test examples, with
the same prompt, the same decoding settings and the same parser as both
baselines.

**Why "once" matters.** Every extra look at the test set with a tweak in between
turns it into a validation set. This is the single evaluation that gets reported.

**How to read it.** Compare all three systems on the same ladder. Pay attention
to *which* fields improved -- a jump in `category` with no change in `priority`
tells a different story from uniform improvement.
""")
code("""
lora_preds = run_evaluation(adapted_model, tokenizer, test_records,
                            max_new_tokens=DECODING["max_new_tokens"],
                            batch_size=DECODING["batch_size"],
                            progress=lambda d, t: print(f"  {d}/{t}", end="\\r"))
lora_metrics = M.compute_metrics(lora_preds)

print("\\n" + "=" * 76)
print(f"{'metric':28s} {'zero-shot':>12s} {'few-shot':>12s} {'LoRA':>12s}")
print("=" * 76)
for k in ("json_parse_rate_strict", "schema_valid_rate", "exact_match",
          "constraint_violation_rate", "markdown_fence_rate"):
    print(f"{k:28s} {zs_metrics[k]:12.4f} {fs_metrics[k]:12.4f} "
          f"{lora_metrics[k]:12.4f}")
print("-" * 76)
for f in ("category", "priority", "affected_service", "is_security_incident",
          "users_affected"):
    print(f"{'  field: ' + f:28s} {zs_metrics['field_accuracy'][f]:12.4f} "
          f"{fs_metrics['field_accuracy'][f]:12.4f} "
          f"{lora_metrics['field_accuracy'][f]:12.4f}")
""")

# ---------------------------------------------------------------------------
md("""
## 17. Is the difference real? Confidence intervals and a paired test

**What this does.** Computes a paired bootstrap confidence interval for the
difference, and an exact McNemar test.

**Why it is needed.** With 86 test examples, a higher number is not automatically
a better model. Reporting a bare point estimate is how small experiments
over-claim.

- **Paired bootstrap** -- resample *examples* (not systems) 10,000 times. If the
  95% interval for the difference excludes zero, the difference survives
  resampling.
- **McNemar** -- looks only at examples where the two systems *disagree*. If one
  system wins 40 disagreements and the other wins 2, that is signal. If it is
  21 vs 21, it is noise.

**How to read it.** `excludes_zero: True` plus a small p-value means the
difference is real *on this test set*. It says nothing about real tickets.
""")
code("""
cmp_fs = M.compare(fs, lora_preds, "fewshot", "lora", metric="exact_match")
cmp_zs = M.compare(zs, lora_preds, "zeroshot", "lora", metric="exact_match")

for cmp in (cmp_zs, cmp_fs):
    d = cmp["paired_diff"]
    print(f"{cmp['system_a']} ({cmp['a_rate']:.1%}) -> "
          f"{cmp['system_b']} ({cmp['b_rate']:.1%})")
    print(f"   difference {d['diff']:+.1%}  95% CI [{d['lo']:+.1%}, {d['hi']:+.1%}]"
          f"  excludes zero: {d['excludes_zero']}")
    m = cmp["mcnemar"]
    print(f"   McNemar: only_A={m['only_a_correct']} only_B={m['only_b_correct']} "
          f"p={m['p_value']:.3e}\\n")
""")

# ---------------------------------------------------------------------------
md("""
## 18. Error analysis

**What this does.** Groups the remaining failures by category and shows real
examples of each.

**Why it is needed.** An aggregate score tells you *how much* is wrong. Only the
errors tell you *what* is wrong, and that is what would guide the next
experiment.

**How to read it.** Look for whether the failures moved from *format* problems
(invalid JSON, fences, bad enum values) to *judgement* problems (valid schema,
wrong value). That shift is the signature of format learning without full task
mastery -- and it is the most likely honest outcome of adapting on 171 examples.
""")
code("""
print("failure taxonomy (test set):")
print(f"{'category':24s} {'zero-shot':>10s} {'few-shot':>10s} {'LoRA':>10s}")
cats = sorted(set(zs_metrics["error_categories"]) |
              set(fs_metrics["error_categories"]) |
              set(lora_metrics["error_categories"]))
for c in cats:
    print(f"{c:24s} {zs_metrics['error_categories'].get(c,0):10d} "
          f"{fs_metrics['error_categories'].get(c,0):10d} "
          f"{lora_metrics['error_categories'].get(c,0):10d}")

print("\\n--- remaining LoRA failures ---")
shown = set()
for row in lora_preds:
    c = row["error_category"]
    if c == "correct" or c in shown:
        continue
    shown.add(c)
    print("=" * 70)
    print(f"[{c}] {row['example_id']}")
    print("ticket  :", row["ticket_text"][:150])
    print("got     :", row["parsed"])
    print("expected:", row["expected"])
    wrong = [k for k, ok in row["field_correct"].items() if not ok]
    print("wrong fields:", wrong)
""")

# ---------------------------------------------------------------------------
md("""
## 19. Export artefacts

**What this does.** Writes the raw per-example predictions and the metrics to
disk, and zips the adapter.

**Why raw predictions matter more than metrics.** Anyone can recompute every
number in this notebook from the prediction files. A metric you cannot recompute
is a claim; a metric you can recompute from saved predictions is evidence.
`scripts/04_report.py` does exactly that, and flags any disagreement with what
was recorded at run time.
""")
code("""
from forgelm import dataio

for name, preds in (("zeroshot", zs), ("fewshot", fs), ("lora", lora_preds)):
    dataio.write_jsonl(preds, f"reports/predictions/{name}_test.jsonl")
    dataio.write_json({"condition": name, "split": "test",
                       "metrics": M.compute_metrics(preds)},
                      f"reports/metrics/{name}_test.json")
    print(f"wrote reports/predictions/{name}_test.jsonl ({len(preds)} rows)")

import shutil
shutil.make_archive("forgelm_lora_adapter", "zip", adapter_dir)
print("\\nzipped adapter -> forgelm_lora_adapter.zip")

if IN_COLAB:
    from google.colab import files
    files.download("forgelm_lora_adapter.zip")
""")

# ---------------------------------------------------------------------------
md("""
## 20. What this did and did not show

**Shown, if the criteria above passed:** on this synthetic 86-example test set,
with this base model at this revision, LoRA on 171 examples produced a
measurable improvement over both zero-shot and few-shot prompting of the same
unchanged model, with the paired confidence interval excluding zero.

**Not shown, and not claimed:**

- Anything about real helpdesk tickets. The data is synthetic.
- Anything about deployment, latency, cost or serving. Latency was recorded as an
  observation on one consumer GPU, not as a performance claim.
- Anything about safety, alignment, fairness or robustness. Those were not
  measured, so nothing can be said about them.
- Generalisation beyond the 16 held-out scenario families in the test set.
- That r=16 is optimal. Only one configuration was trained; the ablation is a
  separate, clearly-labelled experiment.

**The most interesting honest finding** is the few-shot baseline. Prompting alone
took strict-JSON compliance from 5.8% to 100%. If we had only compared against
zero-shot, we would have credited LoRA with fixing a problem that eight examples
in the prompt already fixed for free. That is why the brief insists on this
baseline, and why it was worth the compute.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"wrote {OUT.relative_to(REPO_ROOT)}: {n_md} markdown + {n_code} code cells")
