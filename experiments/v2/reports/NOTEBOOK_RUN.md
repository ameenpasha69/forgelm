# Notebook execution record

**22 cells executed, 2 skipped, 1 failed** out of 25 code cells.

## Scope -- read this before quoting the result

Executed on a GTX 1650 (sm_75), the same compute capability as a Colab T4, so select_precision() takes the identical fp16 branch. This verifies the notebook's code paths on real CUDA hardware. It does NOT verify Colab's environment: different torch build, preinstalled packages and driver. The Colab GPU cells remain unverified ON COLAB.

**Training was shortened to fit this machine.** cell 14: num_train_epochs 8 -> 1, early_stopping_patience -> 1. The notebook's training cell therefore executed for real, but the adapter it produced is not the reported one -- that comes from the full 8-epoch runs in `runs/`. What this verifies is that the cell runs, not that a 1-epoch adapter is any good.

## Environment

| | |
|---|---|
| python | `3.13.14` |
| torch | `2.13.0+cu126` |
| cuda_available | `True` |
| gpu | `NVIDIA GeForce GTX 1650` |
| compute_capability | `sm_75` |
| vram_gib | `4.0` |
| transformers | `5.15.1` |
| peft | `0.20.0` |
| accelerate | `1.14.0` |

## Cells

| Cell | Status | Time | First line |
|---|---|---|---|
| 0 | skipped | -- | `shells out to nvidia-smi; environment is captured separately below` |
| 1 | ok | 0.0s | `import sys, platform` |
| 2 | skipped | -- | `installs packages and git-clones the repo; on local hardware the noteb` |
| 3 | ok | 0.2s | `from forgelm.seeding import SEEDS, seed_everything` |
| 4 | ok | 0.0s | `from forgelm.datagen import generate_dataset, FAMILIES, TEMPLATE_FAMIL` |
| 5 | ok | 0.0s | `assert generate_dataset() == records, "generation is not reproducible!` |
| 6 | ok | 0.0s | `from forgelm import validate` |
| 7 | ok | 0.0s | `from forgelm.splits import build_manifest, apply_split` |
| 8 | ok | 0.8s | `findings = validate.check_cross_split_leakage(records, manifest)` |
| 9 | ok | 26.2s | `from forgelm.modeling import (BASE_MODEL_FACTS, load_tokenizer, load_b` |
| 10 | ok | 0.1s | `from forgelm.prompts import render_prompt, SYSTEM_PROMPT` |
| 11 | ok | 127.8s | `from forgelm.generate import run_evaluation` |
| 12 | ok | 0.0s | `for row in zs[:2]:` |
| 13 | ok | 515.5s | `from forgelm.prompts import select_demonstrations, FEWSHOT_K` |
| 14 | ok | 20.6s | `from forgelm.modeling import build_lora_model, parameter_report` |
| 15 | ok | 1.2s | `from forgelm import training as T` |
| 16 | ok | 260.3s | `fwd = T.one_batch_forward(lora_model, tokenizer, train_enc, batch_size` |
| 17 | ok | 14.7s | `del lora_model, base` |
| 18 | ok | 459.2s | `import time` |
| 19 | ok | 2.1s | `import matplotlib.pyplot as plt` |
| 20 | ok | 11.4s | `from forgelm.modeling import load_adapted_model` |
| 21 | ok | 111.6s | `lora_preds = run_evaluation(adapted_model, tokenizer, test_records,` |
| 22 | ok | 5.2s | `cmp_fs = M.compare(fs, lora_preds, "fewshot", "lora", metric="exact_ma` |
| 23 | ok | 0.0s | `print("failure taxonomy (test set):")` |
| 24 | **FAILED** | 3.5s | `from forgelm import dataio` |

## Failure

```
Traceback (most recent call last):
  File "D:\projects\forgelm\scripts\v2_08_run_notebook.py", line 129, in main
    exec(compile(source, f"<cell {index}>", "exec"), namespace)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<cell 24>", line 14, in <module>
NameError: name 'IN_COLAB' is not defined

```

## Verifying on Colab itself

Open `notebooks/forgelm_colab.ipynb` on a T4 runtime and run all cells. Record GPU, VRAM, Python, CUDA, torch, transformers, peft, runtime, seeds, adapter sha256 and final metrics. Until that is done, `STATUS.md` keeps the Colab GPU cells marked **implemented but unverified on Colab**.
