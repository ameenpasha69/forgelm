# Running ForgeLM somewhere other than the machine it was built on

Every recorded v1 result came from one specific machine. That specification is
preserved verbatim below and is **not** the recommended setup for anyone else --
it is the historical record of what produced the numbers.

## The environment the results came from

| | |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.13.14 |
| GPU | NVIDIA GeForce GTX 1650, 4.0 GiB, compute capability **sm_75** (Turing) |
| Driver | 592.82 |
| System RAM | **5.9 GiB** |
| torch | 2.13.0+cu126 (CUDA 12.6, cuDNN 91002) |
| transformers / peft / accelerate | 5.15.1 / 0.20.0 / 1.14.0 |

Re-recorded in every `runs/<run_id>/run.json`, so any number traces to the stack
that produced it.

### The constraint that actually bit

Not the 4 GiB of VRAM -- the **5.9 GiB of system RAM**. A single trainer needs
2-3 GiB resident. Running a second Python process alongside it (a test suite, a
dataset build) pushed the trainer into swap and made it **~19x slower**: 138
minutes per epoch instead of 8, with a resident set of 144 MB. It never
crashed; it just crawled, which is harder to notice.

**If you have under ~8 GiB of RAM: run one Python process at a time.** The
symptom to watch for is `OSError 1455` ("paging file is too small") on Windows,
or an `nvidia-smi` showing high GPU utilisation while wall-clock per epoch
climbs.

---

## Choose the right requirements file

There is deliberately no single `requirements.txt` that works everywhere; one
would force a Windows CUDA build on a Linux box or a CPU box.

| File | For | Can train? |
|---|---|---|
| `requirements/windows-cuda.txt` | reproducing the historical results exactly | yes |
| `requirements/linux-cuda.txt` | Linux + NVIDIA | yes |
| `requirements/colab.txt` | Google Colab T4 | yes |
| `requirements/cpu.txt` | tests, dataset build, report regeneration, CI | no |

### Windows + CUDA (the historical environment)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements/windows-cuda.txt
```

### Linux + CUDA

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements/linux-cuda.txt
```

Check your GPU is covered by the wheel before training:

```bash
python -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.get_device_capability())"
```

Expect small differences in training loss against the recorded numbers. The
deterministic parts -- dataset generation, splitting, parsing, metrics, bootstrap
-- reproduce exactly, and the CI job asserts that.

### Google Colab (T4)

Do **not** install torch; Colab's build matches its own CUDA runtime and
replacing it is the usual way to break the GPU runtime.

```bash
!pip install -q -r requirements/colab.txt
```

A T4 is also sm_75, so `modeling.select_precision()` takes the same fp16 branch
as the reference machine. It has 16 GiB of VRAM against 4, so the batch sizes in
`config.py` are conservative rather than necessary there.

### CPU only -- audit without a GPU

This is the important one for a reader who wants to check the work rather than
repeat it.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/cpu.txt

python scripts/00_build_dataset.py       # regenerates and re-verifies checksums
pytest tests/ -m "not slow"              # deterministic checks
python scripts/04_report.py --split test # every metric, from raw predictions
python scripts/06_audit.py --skip-tests  # evidence audit
```

None of that needs model weights. Every headline number is recomputed from the
committed per-example predictions, so the evidence is auditable on a laptop with
no GPU at all.

---

## Continuous integration

`.github/workflows/ci.yml` runs exactly the CPU-only path above on every push,
with `HF_HUB_OFFLINE=1` set so nothing can quietly reach for the network. It
checks:

- the dataset regenerates deterministically and both checksums match
- the v1 split has not moved and no cross-split leakage exists
- the v2 sealed test membership checksum still verifies
- schemas, parsing, metrics and the statistical utilities pass
- every metric recomputes from the committed raw predictions
- the report regenerates
- no placeholders, secrets, or unsupported claims

Tests that need the tokenizer skip themselves offline rather than downloading,
so a green CI run never depends on a model artefact.
