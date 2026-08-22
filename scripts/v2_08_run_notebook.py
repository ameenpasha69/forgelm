"""Execute the Colab notebook's code cells in order and record what happened.

Honest scope, stated up front
-----------------------------
This runs the notebook on **local hardware**, not on Colab. It therefore
verifies that every code path executes and produces sane output on a real
CUDA device. It does **not** verify Colab's environment: different torch
build, different preinstalled packages, different driver.

The local GPU is a GTX 1650 -- compute capability **sm_75**, the same as a
Colab T4 -- so `modeling.select_precision()` takes the identical fp16 branch.
That makes this a strong proxy and still not the thing itself, and the report
says so.

    python scripts/v2_08_run_notebook.py            # all cells, ~50-60 min
    python scripts/v2_08_run_notebook.py --epochs 1   # shortened training

Output: experiments/v2/reports/notebook_execution.json and NOTEBOOK_RUN.md
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forgelm import dataio  # noqa: E402
from forgelm.ledger import Run  # noqa: E402

NOTEBOOK = REPO_ROOT / "notebooks" / "forgelm_colab.ipynb"
OUT = REPO_ROOT / "experiments" / "v2" / "reports"

# Cells that only make sense inside Colab. Skipping them is itself a finding,
# so each carries the reason that goes into the report.
SKIP_REASONS = {
    0: "shells out to nvidia-smi; environment is captured separately below",
    2: "installs packages and git-clones the repo; on local hardware the "
       "notebook's own IN_COLAB guard already takes the no-op branch, but "
       "running pip here would disturb the verified environment",
}

# The cell that defines `config` (LoRA settings). Cells after it depend on that
# name, so it can never be skipped -- an earlier version of this script had the
# training index wrong, skipped this one instead, and cell 15 died with
# `NameError: config`. Recorded because the failure looked like a notebook bug
# and was a bug in the harness.
CONFIG_CELL = 14

# The cell that actually calls trainer.train().
TRAINING_CELL = 18

# Cells that depend on a trained trainer existing. Skipping training alone
# would cascade into these, so the notebook cannot be verified by skipping a
# cell in the middle; the training must be shortened instead.
TRAINING_DEPENDENT = {19, 20, 21, 22, 23, 24}


def cell_sources() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code"]


def environment() -> dict:
    info: dict = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["compute_capability"] = f"sm_{p.major}{p.minor}"
            info["vram_gib"] = round(p.total_memory / (1024 ** 3), 2)
    except ImportError:
        info["torch"] = None
    for pkg in ("transformers", "peft", "accelerate"):
        try:
            from importlib import metadata

            info[pkg] = metadata.version(pkg)
        except Exception:
            info[pkg] = None
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None,
                    help="shorten the notebook's training to N epochs. The "
                         "notebook's cells are stateful -- everything after the "
                         "training cell needs a trained trainer -- so the only "
                         "honest way to fit a small machine is to shorten "
                         "training, not to skip it.")
    args = ap.parse_args()

    run = Run(kind="notebook_execution").start()
    try:
        sources = cell_sources()
        env = environment()
        print(f"executing {len(sources)} code cells on "
              f"{env.get('gpu', 'CPU')} ({env.get('compute_capability', 'n/a')})")

        namespace: dict = {"__name__": "__main__"}
        results = []
        patched: list[str] = []
        failed_at = None

        for index, source in enumerate(sources):
            if index in SKIP_REASONS:
                print(f"  cell {index:2d}: SKIPPED ({SKIP_REASONS[index]})")
                results.append({"cell": index, "status": "skipped",
                                "reason": SKIP_REASONS[index]})
                continue

            buffer = io.StringIO()
            start = time.perf_counter()
            try:
                with redirect_stdout(buffer), redirect_stderr(buffer):
                    exec(compile(source, f"<cell {index}>", "exec"), namespace)
                elapsed = time.perf_counter() - start
                out = buffer.getvalue()

                if (index == CONFIG_CELL and args.epochs
                        and isinstance(namespace.get("config"), dict)):
                    namespace["config"]["num_train_epochs"] = args.epochs
                    namespace["config"]["early_stopping_patience"] = 1
                    patched.append(
                        f"cell {index}: num_train_epochs 8 -> {args.epochs}, "
                        f"early_stopping_patience -> 1")
                    print(f"           [patched] training shortened to "
                          f"{args.epochs} epoch(s)")
                print(f"  cell {index:2d}: ok ({elapsed:6.1f}s) "
                      f"{out.strip().splitlines()[-1][:70] if out.strip() else ''}")
                results.append({
                    "cell": index, "status": "ok",
                    "seconds": round(elapsed, 2),
                    "stdout_tail": out.strip()[-600:],
                    "first_line": source.strip().splitlines()[0][:90],
                })
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - start
                print(f"  cell {index:2d}: FAILED ({elapsed:.1f}s) "
                      f"{type(exc).__name__}: {exc}")
                results.append({
                    "cell": index, "status": "failed",
                    "seconds": round(elapsed, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2000:],
                    "stdout_tail": buffer.getvalue().strip()[-600:],
                    "first_line": source.strip().splitlines()[0][:90],
                })
                failed_at = index
                break

        ok = sum(1 for r in results if r["status"] == "ok")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = sum(1 for r in results if r["status"] == "failed")

        payload = {
            "notebook": str(NOTEBOOK.relative_to(REPO_ROOT)),
            "executed_on": "local hardware, NOT Google Colab",
            "environment": env,
            "n_code_cells": len(sources),
            "ok": ok, "skipped": skipped, "failed": failed,
            "failed_at_cell": failed_at,
            "patches_applied": patched,
            "cells": results,
            "scope_caveat": (
                "Executed on a GTX 1650 (sm_75), the same compute capability as "
                "a Colab T4, so select_precision() takes the identical fp16 "
                "branch. This verifies the notebook's code paths on real CUDA "
                "hardware. It does NOT verify Colab's environment: different "
                "torch build, preinstalled packages and driver. The Colab GPU "
                "cells remain unverified ON COLAB."),
        }

        L = [
            "# Notebook execution record",
            "",
            f"**{ok} cells executed, {skipped} skipped, {failed} failed** "
            f"out of {len(sources)} code cells.",
            "",
            "## Scope -- read this before quoting the result",
            "",
            payload["scope_caveat"],
            "",
            ("**Training was shortened to fit this machine.** "
             + "; ".join(patched)
             + ". The notebook's training cell therefore executed for real, but "
               "the adapter it produced is not the reported one -- that comes "
               "from the full 8-epoch runs in `runs/`. What this verifies is "
               "that the cell runs, not that a 1-epoch adapter is any good.")
            if patched else
            "Training ran at the notebook's own settings, unmodified.",
            "",
            "## Environment",
            "",
            "| | |", "|---|---|",
        ]
        for k, v in env.items():
            L.append(f"| {k} | `{v}` |")
        L += ["", "## Cells", "",
              "| Cell | Status | Time | First line |", "|---|---|---|---|"]
        for r in results:
            mark = {"ok": "ok", "skipped": "skipped",
                    "failed": "**FAILED**"}[r["status"]]
            secs = f"{r['seconds']:.1f}s" if "seconds" in r else "--"
            first = r.get("first_line", r.get("reason", ""))[:70]
            L.append(f"| {r['cell']} | {mark} | {secs} | `{first}` |")
        L.append("")
        if failed:
            L += ["## Failure", "",
                  "```", results[-1].get("traceback", "")[-1500:], "```", ""]
        L += ["## Verifying on Colab itself", "",
              "Open `notebooks/forgelm_colab.ipynb` on a T4 runtime and run all "
              "cells. Record GPU, VRAM, Python, CUDA, torch, transformers, "
              "peft, runtime, seeds, adapter sha256 and final metrics. Until "
              "that is done, `STATUS.md` keeps the Colab GPU cells marked "
              "**implemented but unverified on Colab**.", ""]

        dataio.write_json(payload, OUT / "notebook_execution.json")
        (OUT / "NOTEBOOK_RUN.md").write_text("\n".join(L), encoding="utf-8")

        run.metrics = {"ok": ok, "skipped": skipped, "failed": failed}
        run.finish("success" if failed == 0 else "failed")
        print(f"\n{ok} ok, {skipped} skipped, {failed} failed")
        print(f"wrote {(OUT / 'NOTEBOOK_RUN.md').relative_to(REPO_ROOT)}")
        return 0 if failed == 0 else 1

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
