# ForgeLM -- status

Status vocabulary, used strictly:

| Status | Means |
|---|---|
| **verified** | It ran, and there is a log, artefact, test result or metric file proving it |
| **implemented but unverified** | Code exists and is reviewed, but has not been executed end to end |
| **in progress** | Being worked on now |
| **not started** | No code |
| **blocked** | Cannot proceed without something unavailable |

"I wrote the code" is never sufficient for **verified**. Neither is "the config
says so".

---

## Repository audit (initial)

`D:\projects` was empty and was not a git repository. There was no prior ForgeLM
code, notebook, dataset, artefact or result to classify, preserve or reuse.
**Nothing was deleted, because nothing existed.** No secrets, tokens, hard-coded
paths, placeholder logic or fabricated metrics were found, because there were no
files to contain them. The repository was initialised at `D:\projects\forgelm`.

---

## Component status

| Component | Status | Evidence | Limitations | Next action |
|---|---|---|---|---|
| Environment / GPU | **verified** | `torch 2.13.0+cu126`, `sm_75` in `arch_list`, fp16 matmul executed on GTX 1650 4 GiB | 4 GiB caps batch size; no native bf16 on Turing | none |
| Task selection | **verified** | 4 candidates compared in `DECISIONS.md` D-003 | single narrow task by design | none |
| Base model selection | **verified** | 5 candidates compared against HF API + model cards; pinned to commit `7ae5576...` | 0.5B is small; results will not transfer to larger models | none |
| Output schema | **verified** | `src/forgelm/schema.py`; 14 schema tests pass | 5 fields only | none |
| Dataset generation | **verified** | 300 examples, `raw_sha256=aba7202...`, regeneration reproduces it byte-for-byte | synthetic, templated, one domain, British English | none |
| Dataset QC | **verified** | `reports/dataset_validation.json`: 0 errors, 0 warnings | checks cover what we thought to check; manual review found 4 more | none |
| Manual review | **verified** | `reports/manual_review_sample.md` (24 examples); 4 defects found and fixed, listed in `DATASET_CARD.md` | one reviewer, one round | second reviewer would help |
| Deterministic split | **verified** | manifest checksum `994141c6...`, asserted frozen by `test_split_checksum_is_frozen` | 86 test examples is small | none |
| Leakage controls | **verified** | 0 families straddle splits; all 44,850 cross-split pairs scanned; max similarity **0.3255** vs 0.80 threshold | template families intentionally shared (D-007) | none |
| Evaluation protocol | **verified** | metrics recomputed from raw predictions by `scripts/04_report.py` and audited against recorded values | n=86 gives wide intervals | none |
| Zero-shot baseline | **verified** | `reports/predictions/zeroshot_test.jsonl`, run `20260821T200256Z_eval_zeroshot_812934` | greedy decoding only | none |
| Few-shot baseline | **verified** | `reports/predictions/fewshot_test.jsonl`, run `20260821T200656Z_eval_fewshot_3f8d1e` | k=8 only; no k sweep | none |
| Pre-flight training gates | **verified** | `reports/smoke_train.json`, all 6 gates pass, peak 2.14/4.00 GiB | -- | none |
| LoRA training | **verified** | run `20260821T203003Z_train_lora_b61876`; 110 steps, 2202 s; early stopping fired at epoch 5; best val loss 0.0743 @ epoch 2 | one seed, one configuration | multi-seed run |
| Adapter save / reload | **verified** | 336/336 LoRA tensors non-zero, max abs 0.03833; reloaded through a clean path and re-verified in `06_audit.py --with-model` | 46.7 MB incl. tokenizer | none |
| Frozen test evaluation | **verified** | run `20260821T211024Z_eval_lora_73a19c`; `reports/predictions/lora_test.jsonl`; evaluated once | n=86, wide intervals | none |
| Success criteria | **verified met** | all 4 pre-declared criteria pass; exact match +10.5 pp vs few-shot, 95% CI [+3.5, +18.6], McNemar p=0.0117 | criteria are narrow by design | none |
| Generalisation | **verified negative** | 11 of 16 held-out scenario families produced zero fully correct outputs | this is the main limitation | broader scenario coverage |
| Reproducibility ledger | **verified** | every run writes `runs/<run_id>/run.json` incl. failed runs | wall-clock timings are machine-specific | none |
| Colab notebook | **partially verified** | 50 cells; JSON-valid; every code cell parses; CPU-only cells (seeds, generation, validation, splitting, leakage) executed locally and passed | GPU cells not executed on Colab hardware | run it on a T4 |
| Published repository | **verified** | pushed to GitHub, then cloned back into a clean directory and all three dataset checksums re-verified | -- | none |
| Optional ablation | **not started** | -- | compute was spent on the primary experiment; no ablation result is claimed | `--train-fraction 0.5` or `configs/ablation_r8.json` are wired up and ready |
| Optional demo app | **partially verified** | `--cli` executed against the trained adapter: loaded 336/336 LoRA tensors and returned a schema-valid, correct triage for the VPN example; missing-adapter error path also executed | Gradio UI itself not launched (gradio is an optional dependency and is not installed) | `pip install gradio` and run without `--cli` |
| Test suite | **verified** | `134 passed in 144.30s`, recorded in `reports/EVIDENCE.md` | no test asserts a *result value*, by design | none |
| Final evidence audit | **verified** | `20/20 checks passed`, `reports/EVIDENCE.md` | -- | none |

---

## Verified negative / surprising findings

Recorded because a repository that only lists successes is not being honest
about what happened.

1. **The few-shot baseline is far stronger than the zero-shot baseline.**
   Strict-JSON compliance went 5.8% -> 100% and category macro-F1 0.14 -> 0.54
   from eight in-prompt demonstrations alone. Had only zero-shot been measured,
   LoRA would have received credit for fixing a problem that prompting already
   fixed for free.
2. **The QC suite caught a real label leak on its first run.** The ticket text
   for `email_spoofed_sender` contained the word "priority". Build aborted,
   defect fixed.
3. **The first split design was defective.** A plain seeded shuffle produced a
   validation split with zero `critical` examples. Found and fixed before any
   model was run (D-006).
4. **`torch.cuda.is_bf16_supported()` returns `True` on sm_75**, where bf16 is
   emulated rather than native. Trusting it would have selected a slow, unstable
   precision. `select_precision()` keys off compute capability instead.
5. **`tiny_overfit` hit CUDA OOM at batch 4.** The 151,936-token vocabulary makes
   the logits tensor dominant. Fixed with gradient accumulation over
   micro-batches of 1 -- which is also what real training does here.
6. **transformers 5.15.1 silently dropped `warmup_ratio` and
   `save_safetensors`.** The argument-introspection layer reported them. Warm-up
   is now converted to `warmup_steps` against the real step count instead of
   being lost.
7. **`EarlyStoppingCallback` asserts evaluation is enabled**, so the smoke
   configuration (`eval_strategy="no"`) crashed at `on_train_begin`. The callback
   is now attached conditionally.
8. **Training loss starts low (~0.19), which is expected, not a bug.** Most
   tokens in the target JSON are structural and trivially predictable under
   teacher forcing. This is precisely why generated exact-match, not loss, is the
   headline metric.
9. **Loading a second model while training was running exhausted the Windows
   paging file** (`OSError 1455`). This was operator error, not a code defect,
   and it is listed here because the ledger caught it: the failed run is
   preserved at `runs/20260821T205221Z_eval_lora_*` with its traceback. On a
   4 GiB card, run one model process at a time.
10. **The adapter-liveness check was itself wrong, and a test caught it.**
   `load_adapted_model` verified that "some LoRA tensor is non-zero". That is
   insufficient: the LoRA update is `delta_W = B @ A`, and PEFT
   *zero-initialises B* so an untrained adapter is a deliberate no-op. A fresh
   adapter therefore has ~half its tensors non-zero (all the A matrices) while
   being mathematically identical to the base model -- and the old check passed
   it. `test_inert_adapter_is_rejected` saves an untrained adapter and asserts
   the loader refuses it; it failed, exposing the bug. The check now requires a
   non-zero `lora_B`. The trained adapter passes either way (168/168 B tensors
   non-zero), so no reported result changes -- but the guard would not have
   caught the failure it was written to catch.
11. **Git would have silently broken every dataset checksum.** On Windows, git
   converts LF to CRLF on checkout by default. The dataset files and split
   manifest are written with explicit LF newlines and verified by sha256, so a
   fresh clone would have failed the evidence audit for reasons that had nothing
   to do with the experiment. Fixed by `.gitattributes` marking every
   checksum-verified format `-text`. **Verified by cloning the published
   repository into a clean directory and re-checking all three checksums --
   all matched.**

---

## Not claimed

No claim is made anywhere in this repository about production readiness,
deployment readiness, alignment, safety, fairness, robustness, latency
improvement, cost reduction, general capability improvement, domain expertise,
QLoRA, quantisation, vLLM, MLflow, distributed training, human-preference
optimisation, RAG, agentic capability, or real-world reliability. None of those
were measured, so none can be asserted.

Latency appears in the metrics as a recorded **observation** on one consumer
GPU, explicitly flagged as not a serving-performance claim.
