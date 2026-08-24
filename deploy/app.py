"""ForgeLM on Hugging Face Spaces -- a hosted demonstration, not a service.

This is the deployed sibling of `scripts/demo_app.py`. It differs in the ways a
hosted app has to differ, and in no others:

  * the model is loaded once, at import, and reused for every request;
  * requests are queued with concurrency 1, because one model instance on a
    shared CPU cannot serve two generations at the same time;
  * it binds 0.0.0.0 rather than 127.0.0.1, because a container has to;
  * comparison mode is off -- loading the base model as well would roughly
    double memory on a 16 GiB shared CPU box for a contrast that the README
    already reports in numbers.

What has NOT changed is the honesty of the thing. The adapter was trained on
171 synthetic helpdesk tickets. It should not be used to triage real ones, and
the UI says so where a user will actually read it.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT / "src"))

import gradio as gr  # noqa: E402
import torch  # noqa: E402

# Torch's default CPU thread count is conservative in containers -- it reads the
# host's core count but the cgroup quota is what actually applies. Generation
# here is a long chain of small matmuls, so threads are the only lever that
# matters on a CPU Space. Precision is deliberately left alone: `select_precision`
# picks float32 on CPU for a documented reason (fp16 on CPU is slow and poorly
# supported), and this is not the place to second-guess it.
_threads = os.cpu_count() or 2
torch.set_num_threads(_threads)
print(f"torch CPU threads: {torch.get_num_threads()} (of {_threads} visible)",
      flush=True)

from forgelm.config import DECODING  # noqa: E402
from forgelm.generate import generate_batch  # noqa: E402
from forgelm.modeling import (  # noqa: E402
    BASE_MODEL_ID, BASE_MODEL_REVISION, load_adapted_model, load_tokenizer,
)
from forgelm.parsing import parse_response  # noqa: E402
from forgelm.prompts import render_prompt  # noqa: E402
from forgelm.schema import canonical_json, validate_output  # noqa: E402

ADAPTER_DIR = APP_ROOT / "artifacts" / "lora_adapter"

EXAMPLES = [
    "The VPN client disconnects every few minutes and has to be reconnected. "
    "So far 14 people have reported it. Please investigate the concentrator",
    "Issue summary: the battery in my laptop has visibly swollen and lifted the "
    "trackpad. Additional detail: there is a faint chemical smell coming from "
    "the vents. Impact: Only one person is affected. Request: what should I do "
    "with it right now?",
    "hey, a message claiming our mailbox is full is asking people to re-enter "
    "their password. the sender domain is a lookalike with one letter changed. "
    "around 62 colleagues have reported the same thing. pls confirm whether "
    "anyone clicked it",
    "2 + 2 = ?",
]

# A ticket long enough to be a denial-of-wallet on a free CPU box is not a
# ticket. Truncation is visible in the UI rather than silent.
MAX_TICKET_CHARS = 2000


class Triager:
    """Loads the model once, at startup, and answers one ticket at a time."""

    def __init__(self, adapter_dir: Path) -> None:
        self.tokenizer = load_tokenizer()
        self.model, self.verification = load_adapted_model(str(adapter_dir))
        self.provenance = {}
        provenance_path = adapter_dir / "forgelm_provenance.json"
        if provenance_path.exists():
            self.provenance = json.loads(
                provenance_path.read_text(encoding="utf-8"))

    def triage(self, ticket: str) -> dict:
        prompt = render_prompt(self.tokenizer, ticket)
        out = generate_batch(self.model, self.tokenizer, [prompt],
                             DECODING["max_new_tokens"])[0]
        result = parse_response(out["text"],
                                finish_reason=out["finish_reason"])
        violations = validate_output(result.parsed) \
            if result.parsed is not None else ["unparseable"]
        return {
            "raw": out["text"],
            "pretty": canonical_json(result.parsed) if result.parsed else None,
            "schema_valid": not violations,
            "violations": violations,
            "had_fence": result.had_fence,
            "truncated": result.truncated,
            "n_tokens": out["n_generated_tokens"],
        }


# Loaded at import so the first visitor does not pay for it. A failure here is
# captured rather than raised: a Space that boots and explains why it is broken
# is more useful than one that shows a stack trace and a dead port.
TRIAGER: Triager | None = None
LOAD_ERROR: str | None = None

try:
    print(f"loading {BASE_MODEL_ID} @ {BASE_MODEL_REVISION[:12]} "
          f"+ adapter from {ADAPTER_DIR} ...", flush=True)
    TRIAGER = Triager(ADAPTER_DIR)
    _v = TRIAGER.verification
    print(f"adapter live: {_v['n_nonzero_lora_B_tensors']}"
          f"/{_v['n_lora_B_tensors']} lora_B tensors non-zero", flush=True)
except Exception:  # noqa: BLE001
    LOAD_ERROR = traceback.format_exc()
    print(LOAD_ERROR, file=sys.stderr, flush=True)


def handler(ticket: str):
    """Returns (status, json_or_raw, detail)."""
    if TRIAGER is None:
        return ("Model failed to load.", "",
                f"The Space started but the model did not load:\n\n{LOAD_ERROR}")

    ticket = (ticket or "").strip()
    if not ticket:
        return "Enter a ticket first.", "", ""

    notes = []
    if len(ticket) > MAX_TICKET_CHARS:
        ticket = ticket[:MAX_TICKET_CHARS]
        notes.append(f"Ticket truncated to {MAX_TICKET_CHARS} characters.")

    try:
        result = TRIAGER.triage(ticket)
    except Exception as exc:  # noqa: BLE001
        return (f"Generation failed: {type(exc).__name__}", "", str(exc))

    if result["schema_valid"]:
        status = "Schema VALID"
    else:
        status = f"Schema INVALID -> {', '.join(result['violations'])}"

    body = result["pretty"] or result["raw"] or "(empty response)"

    if result["had_fence"]:
        notes.append("The response arrived wrapped in a markdown code fence; "
                     "it was unwrapped before parsing.")
    if result["truncated"]:
        notes.append("Generation hit the token cap before closing the object.")
    if not result["schema_valid"]:
        notes.append(f"Raw model output:\n{result['raw'][:600]}")
    notes.append(f"{result['n_tokens']} tokens generated, greedy decoding, "
                 f"max_new_tokens={DECODING['max_new_tokens']}.")

    return status, body, "\n\n".join(notes)


def build_ui() -> gr.Blocks:
    if TRIAGER is not None:
        v = TRIAGER.verification
        liveness = (f"{v['n_nonzero_lora_B_tensors']}/{v['n_lora_B_tensors']} "
                    f"`lora_B` tensors non-zero")
    else:
        liveness = "model failed to load"

    with gr.Blocks(title="ForgeLM -- LoRA ticket triage") as demo:
        gr.Markdown(
            "# ForgeLM -- structured ticket triage with a LoRA adapter\n\n"
            f"`{BASE_MODEL_ID}` @ `{BASE_MODEL_REVISION[:12]}` with a LoRA "
            f"adapter trained on **171 synthetic** helpdesk tickets. "
            f"Adapter liveness at boot: {liveness}.\n\n"
            "Paste a helpdesk ticket and the model returns a triage object: "
            "category, priority, affected service, whether it is a security "
            "incident, and how many users are affected.\n\n"
            "*Inference is greedy on CPU and requests are served one at a "
            "time, so expect a few seconds per ticket.*"
        )

        gr.Markdown(
            "> **This is a demonstration of a research artifact, not a "
            "product.** The adapter was trained on synthetic tickets, so do "
            "not use it to triage real ones. There is no authentication and no "
            "rate limiting beyond a single-slot queue. The schema has no "
            "`not_a_ticket` value, so the model will confidently triage input "
            "that is not a ticket at all -- try `2 + 2 = ?` and watch it "
            "happen. That is a known finding from the project's diagnostics, "
            "reported rather than hidden."
        )

        ticket_box = gr.Textbox(
            label="Helpdesk ticket", lines=6,
            placeholder="Describe the problem the way a user would report it...")
        run_button = gr.Button("Triage", variant="primary")

        status_box = gr.Textbox(label="Schema check", interactive=False)
        output_box = gr.Code(label="Triage JSON", language="json")
        detail_box = gr.Textbox(label="Detail", lines=6, interactive=False)

        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[ticket_box])

        gr.Markdown(
            "Measured on a held-out split: base zero-shot **27.9%** schema "
            "valid, few-shot k=8 **61.6%**, base + LoRA **79.1%**. Full method, "
            "ablations and limitations: "
            "[github.com/ameenpasha69/forgelm]"
            "(https://github.com/ameenpasha69/forgelm)."
        )

        run_button.click(handler, inputs=[ticket_box],
                         outputs=[status_box, output_box, detail_box])
        ticket_box.submit(handler, inputs=[ticket_box],
                          outputs=[status_box, output_box, detail_box])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    # One model instance on a shared CPU cannot serve two generations at once;
    # the queue serialises them instead of letting them contend. `api_open`
    # closes the programmatic endpoint -- this is a demonstration, and an open
    # unauthenticated inference API is not part of it. (In Gradio 6 this moved
    # onto queue(); `launch(show_api=...)` no longer exists.)
    ui.queue(default_concurrency_limit=1, max_size=16, api_open=False)
    # PORT is what Cloud Run (and most container hosts) inject and it is not
    # negotiable there; GRADIO_SERVER_PORT is what Spaces and a local run use.
    # Checked in that order so one file serves every target unedited.
    port = int(os.environ.get("PORT")
               or os.environ.get("GRADIO_SERVER_PORT")
               or 7860)

    # share=True opens a public tunnel to this process through Gradio's relay.
    # That is the only way to reach a Colab runtime from outside, and it is how
    # the demo notebook uses this file -- but it publishes an unauthenticated
    # model endpoint, so it stays opt-in and off by default rather than
    # something a stray environment could turn on by accident.
    share = os.environ.get("FORGELM_SHARE", "").strip().lower() in {
        "1", "true", "yes", "on"}
    if share:
        print("FORGELM_SHARE is set: publishing a public *.gradio.live URL. "
              "It is unauthenticated and expires after 72 hours.", flush=True)

    ui.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
              server_port=port, share=share)
