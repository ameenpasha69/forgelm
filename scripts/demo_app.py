"""Local demonstration of the ForgeLM adapter. NOT a deployment.

This is a small local UI for looking at what the adapter does on a ticket you
type. It is not a service, it has no authentication, no rate limiting, no
batching and no error budget, and it must not be exposed to a network.

    pip install gradio
    python scripts/demo_app.py                 # Gradio UI at http://127.0.0.1:7860
    python scripts/demo_app.py --cli           # no Gradio needed
    python scripts/demo_app.py --cli --compare # base vs adapted, side by side

The comparison mode loads the base model as well, which needs roughly twice the
memory. On a 4 GiB card, run it without `--compare` if you hit an OOM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm.config import DECODING  # noqa: E402
from forgelm.generate import generate_batch  # noqa: E402
from forgelm.ledger import REPO_ROOT  # noqa: E402
from forgelm.modeling import (  # noqa: E402
    BASE_MODEL_ID, BASE_MODEL_REVISION, load_adapted_model, load_base_model,
    load_tokenizer,
)
from forgelm.parsing import parse_response  # noqa: E402
from forgelm.prompts import render_prompt  # noqa: E402
from forgelm.schema import canonical_json, validate_output  # noqa: E402

DEFAULT_ADAPTER = REPO_ROOT / "artifacts" / "lora_adapter"

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
]


class Triager:
    """Loads the model once and answers one ticket at a time."""

    def __init__(self, adapter_dir: str, compare: bool = False,
                 constrained: bool = False) -> None:
        self.tokenizer = load_tokenizer()
        self.model, self.verification = load_adapted_model(adapter_dir)
        self.base = load_base_model() if compare else None
        self.constrained = constrained
        # Built once and shared, so its prefix cache stays warm across tickets.
        # Applied to BOTH models when comparing -- constraining only the adapted
        # one would credit the adapter with what the grammar did.
        self.constraint = None
        if constrained:
            from forgelm.constrained import SchemaConstraint

            self.constraint = SchemaConstraint(self.tokenizer)

        self.provenance = {}
        provenance_path = Path(adapter_dir) / "forgelm_provenance.json"
        if provenance_path.exists():
            self.provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    def _run(self, model, ticket: str) -> dict:
        prompt = render_prompt(self.tokenizer, ticket)
        out = generate_batch(model, self.tokenizer, [prompt],
                             DECODING["max_new_tokens"],
                             constraint=self.constraint)[0]
        result = parse_response(out["text"], finish_reason=out["finish_reason"])
        violations = validate_output(result.parsed) if result.parsed is not None \
            else ["unparseable"]
        return {
            "raw": out["text"],
            "parsed": result.parsed,
            "pretty": canonical_json(result.parsed) if result.parsed else None,
            "schema_valid": not violations,
            "violations": violations,
            "had_fence": result.had_fence,
            "n_tokens": out["n_generated_tokens"],
        }

    def triage(self, ticket: str) -> dict:
        ticket = (ticket or "").strip()
        if not ticket:
            return {"error": "Enter a ticket first."}
        payload = {"adapted": self._run(self.model, ticket)}
        if self.base is not None:
            payload["base"] = self._run(self.base, ticket)
        return payload


def format_block(title: str, result: dict) -> str:
    lines = [f"--- {title} ---"]
    if result["schema_valid"]:
        lines.append(f"schema: VALID")
        lines.append(result["pretty"])
    else:
        lines.append(f"schema: INVALID -> {result['violations']}")
        lines.append(f"raw: {result['raw'][:400]}")
    if result["had_fence"]:
        lines.append("note: response was wrapped in a markdown code fence")
    lines.append(f"({result['n_tokens']} tokens generated)")
    return "\n".join(lines)


def run_cli(triager: Triager, compare: bool) -> int:
    print(f"base model : {BASE_MODEL_ID} @ {BASE_MODEL_REVISION[:12]}")
    print(f"adapter    : {triager.verification['n_nonzero_lora_B_tensors']}"
          f"/{triager.verification['n_lora_B_tensors']} lora_B tensors active")
    decoding = ("CONSTRAINED (illegal output unrepresentable)"
                if triager.constrained else "unconstrained")
    print(f"decoding   : {decoding}")
    print("\nThis is a LOCAL DEMONSTRATION, not a deployment.\n")
    print("Type a ticket and press Enter. Blank line + Enter to quit.")
    print("Type 'example' for a sample ticket.\n")

    example_index = 0
    while True:
        try:
            ticket = input("ticket> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not ticket:
            return 0
        if ticket == "example":
            ticket = EXAMPLES[example_index % len(EXAMPLES)]
            example_index += 1
            print(f"\n{ticket}\n")

        payload = triager.triage(ticket)
        if "error" in payload:
            print(payload["error"])
            continue
        print()
        if "base" in payload:
            print(format_block("unchanged base model", payload["base"]))
            print()
        print(format_block("base + LoRA adapter", payload["adapted"]))
        print()


def run_gradio(triager: Triager, compare: bool) -> int:
    try:
        import gradio as gr
    except ImportError:
        print("gradio is not installed. Either:\n"
              "  pip install gradio\n"
              "or run the dependency-free version:\n"
              "  python scripts/demo_app.py --cli", file=sys.stderr)
        return 1

    def handler(ticket: str):
        payload = triager.triage(ticket)
        if "error" in payload:
            return payload["error"], "", ""
        adapted = payload["adapted"]
        status = ("Schema VALID" if adapted["schema_valid"]
                  else f"Schema INVALID -> {adapted['violations']}")
        pretty = adapted["pretty"] or adapted["raw"]
        base_text = format_block("unchanged base model", payload["base"]) \
            if "base" in payload else "(comparison disabled -- start with --compare)"
        return status, pretty, base_text

    with gr.Blocks(title="ForgeLM local demonstration") as demo:
        gr.Markdown(
            f"# ForgeLM -- local demonstration\n\n"
            f"**This is not a deployment.** It is a local tool for inspecting "
            f"what the adapter does on one ticket at a time.\n\n"
            f"Base model `{BASE_MODEL_ID}` @ `{BASE_MODEL_REVISION[:12]}` "
            f"+ LoRA adapter trained on 171 synthetic examples. "
            f"The model was trained on **synthetic** helpdesk tickets and should "
            f"not be used to triage real ones."
        )
        ticket_box = gr.Textbox(label="Helpdesk ticket", lines=5,
                                placeholder="Describe the problem...")
        run_button = gr.Button("Triage", variant="primary")
        status_box = gr.Textbox(label="Schema check", interactive=False)
        output_box = gr.Code(label="Triage JSON", language="json")
        base_box = gr.Textbox(label="Unchanged base model (comparison)",
                              lines=8, interactive=False)
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[ticket_box])

        run_button.click(handler, inputs=[ticket_box],
                         outputs=[status_box, output_box, base_box])
        ticket_box.submit(handler, inputs=[ticket_box],
                          outputs=[status_box, output_box, base_box])

    # Bound to localhost deliberately. share=True would expose an unauthenticated
    # model endpoint to the public internet.
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False,
                inbrowser=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--cli", action="store_true",
                    help="run in the terminal instead of launching Gradio")
    ap.add_argument("--compare", action="store_true",
                    help="also load the unchanged base model for comparison "
                         "(roughly doubles memory use)")
    ap.add_argument("--constrained", action="store_true",
                    help="mask decoding so only schema-legal output can be "
                         "produced. Applied to both models when comparing.")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter)
    if not adapter_dir.exists():
        print(f"No adapter at {adapter_dir}.\n"
              f"Train one first:  python scripts/03_train_lora.py",
              file=sys.stderr)
        return 1

    print(f"loading adapter from {adapter_dir}...")
    try:
        triager = Triager(str(adapter_dir), compare=args.compare,
                          constrained=args.constrained)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not load the model or adapter: {type(exc).__name__}: {exc}\n"
              f"Check that {adapter_dir} contains adapter_config.json and "
              f"adapter_model.safetensors, and that the base model "
              f"{BASE_MODEL_ID} can be downloaded.", file=sys.stderr)
        return 1

    return run_cli(triager, args.compare) if args.cli \
        else run_gradio(triager, args.compare)


if __name__ == "__main__":
    raise SystemExit(main())
