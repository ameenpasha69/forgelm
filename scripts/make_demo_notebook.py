"""Generate notebooks/forgelm_demo_colab.ipynb.

Generated rather than hand-edited, for the same reason as
`scripts/make_notebook.py`: the notebook should stay consistent with the
package instead of drifting into a second, untested copy of the logic.

    python scripts/make_demo_notebook.py

This is the *demonstration* notebook, not the experiment one. It does not
train anything -- it clones the repository, installs the inference
dependencies, and serves the committed adapter through `deploy/app.py` on a
public Gradio URL. Two cells and about two minutes, against
`forgelm_colab.ipynb`'s full end-to-end run.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "forgelm_demo_colab.ipynb"

GITHUB_REPO = "https://github.com/ameenpasha69/forgelm.git"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ===========================================================================
md("""
# ForgeLM -- run the demonstration

This serves the **already-trained** adapter as a web app with a public URL.
It does not train anything; that is
[`forgelm_colab.ipynb`](https://colab.research.google.com/github/ameenpasha69/forgelm/blob/main/notebooks/forgelm_colab.ipynb),
which runs the whole experiment end to end.

**Run the two cells below.** The second prints a `*.gradio.live` link.

---

### Before you open that link to anyone

The URL is **public and unauthenticated** -- anyone who has it can send
requests to the model while your runtime is alive. It expires after 72 hours,
or as soon as you stop the cell or the Colab runtime is recycled.

The adapter was trained on **171 synthetic** helpdesk tickets. It is a research
artifact. Do not use it to triage real ones, and do not paste anything
confidential into it.

**Runtime.** CPU is fine -- a ticket takes a few seconds. *Runtime > Change
runtime type > T4 GPU* makes it near-instant, if you would rather not spend
GPU quota on a demo, leave it on CPU.
""")

# ---------------------------------------------------------------------------
md("""
## 1. Get the code and install dependencies

Clones the repository -- which carries the trained adapter in
`artifacts/lora_adapter` -- and installs what inference needs. Torch is already
present on Colab, so this only adds the libraries around it.

Takes about a minute. `pip`'s dependency-resolver warnings about preinstalled
Colab packages are expected and harmless here.
""")
code(f"""
import os, subprocess, sys
from pathlib import Path

REPO = Path("/content/forgelm")
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "{GITHUB_REPO}", str(REPO)], check=True)
os.chdir(REPO)

# requirements/colab.txt is the Colab-shaped dependency set: it deliberately
# does not pin torch, because Colab ships a build matched to its own CUDA.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "-r", "requirements/colab.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "gradio==6.25.0"], check=True)

print("adapter present:", (REPO / "artifacts/lora_adapter"
                           / "adapter_model.safetensors").exists())
""")

# ---------------------------------------------------------------------------
md("""
## 2. Serve it

Stages the same tree that the Cloud Run and Spaces deployments use -- one app,
one code path, so what you see here is what those hosts would serve -- and
launches it with a public tunnel.

`FORGELM_SHARE=1` is what opts into the public URL; without it the app binds
locally, which is useless inside a Colab VM. Look for the line reading
**`Running on public URL: https://....gradio.live`**.

The first request loads the base model from the Hub, so it is slower than the
rest. **Leave this cell running** -- the link dies when it stops.
""")
code("""
import os, subprocess, sys

subprocess.run([sys.executable, "deploy/build.py",
                "--target", "cloudrun", "--out", "build/demo"], check=True)

env = dict(os.environ, FORGELM_SHARE="1", GRADIO_SERVER_NAME="0.0.0.0")
env.pop("PORT", None)          # Colab sets no PORT; keep Gradio's default
env["GRADIO_SERVER_PORT"] = "7860"

# Streamed rather than captured so the public URL appears as soon as Gradio
# prints it, instead of after the process ends.
process = subprocess.Popen([sys.executable, "app.py"], cwd="build/demo",
                           env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for line in process.stdout:
        print(line, end="")
except KeyboardInterrupt:
    process.terminate()
    print("\\nstopped -- the public URL is now dead")
""")

# ---------------------------------------------------------------------------
md("""
## What to try

The three tickets below are the ones in `DEMO.md`, with the objects the model
is recorded as returning:

| Ticket | Expected |
|---|---|
| VPN disconnecting, 14 reports | `network` / `high` / `vpn` / 14 |
| Swollen laptop battery, chemical smell | `hardware` / `critical` / `laptop` / 1 |
| Lookalike-domain password phish, 62 reports | `security` / `critical` / security incident / 62 |

Then type `2 + 2 = ?` and watch it get confidently triaged anyway. The output
schema has no `not_a_ticket` value, so refusal is **not representable** -- the
model cannot decline, only mislabel. That is a real finding from the project's
own diagnostics, and the app reports the result as schema-invalid rather than
hiding it.

Being able to say that about your own system is worth more than a demo that
only shows the happy path.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False),
               encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"wrote {OUT.relative_to(REPO_ROOT)}: {n_md} markdown + {n_code} code cells")
