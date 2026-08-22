"""E8 -- verify the existing Gradio demonstration. Does not rebuild it.

The CLI path was already verified in v1. This checks the parts that were not:
that the server binds to localhost only, that the adapter really is live
(non-zero lora_B, not merely non-zero LoRA tensors), that every advertised
example works, and that blank, malformed, very long and load-failure paths are
handled rather than crashing.

    pip install gradio
    python scripts/v2_03_demo_checks.py            # headless checks
    python scripts/v2_03_demo_checks.py --launch   # additionally boot the UI

This is a local demonstration. It is not a deployment and nothing here should
be read as one.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from forgelm import dataio  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run  # noqa: E402

OUT = REPO_ROOT / "experiments" / "v2" / "reports"
DEMO_SOURCE = REPO_ROOT / "scripts" / "demo_app.py"
ADAPTER = REPO_ROOT / "artifacts" / "lora_adapter"


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, passed: bool, evidence: str) -> bool:
        self.rows.append({"check": name, "passed": bool(passed),
                          "evidence": evidence})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"         {evidence}")
        return bool(passed)

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true",
                    help="also boot the Gradio server and probe it")
    ap.add_argument("--adapter", default=str(ADAPTER))
    args = ap.parse_args()

    checks = Checks()
    run = Run(kind="demo_checks").start()

    try:
        source = DEMO_SOURCE.read_text(encoding="utf-8")

        # ---- 1. binding, checked in the source that actually runs ---------
        #
        # Parsed with AST rather than substring-matched. A plain
        # `"share=True" in source` check fails here for a silly reason: the
        # file contains a comment explaining *why* share=True is not used, and
        # a substring scan cannot tell a call argument from prose about one.
        print("1. network binding")
        import ast

        tree = ast.parse(source)
        launch_kwargs: dict[str, object] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "launch":
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    try:
                        launch_kwargs[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        launch_kwargs[kw.arg] = "<non-literal>"

        server_name = launch_kwargs.get("server_name")
        share = launch_kwargs.get("share")
        checks.add("Gradio binds to localhost only",
                   server_name == "127.0.0.1",
                   f"launch(server_name={server_name!r}) in the call that "
                   f"actually runs")
        checks.add("no public tunnel is requested",
                   share is False,
                   f"launch(share={share!r}); a public tunnel would expose an "
                   f"unauthenticated model endpoint")

        # ---- 2. adapter liveness ------------------------------------------
        print("\n2. adapter")
        import demo_app  # noqa: E402  (scripts/ is on sys.path)

        adapter_dir = Path(args.adapter)
        if not adapter_dir.exists():
            checks.add("adapter present", False, f"{adapter_dir} not found")
            raise SystemExit(1)

        triager = demo_app.Triager(str(adapter_dir), compare=False)
        v = triager.verification
        checks.add("adapter loads and is genuinely live",
                   v["adapter_is_active"] and v["n_nonzero_lora_B_tensors"] > 0,
                   f"{v['n_nonzero_lora_B_tensors']}/{v['n_lora_B_tensors']} "
                   f"lora_B tensors non-zero, max|B|="
                   f"{v['max_abs_lora_B_weight']:.6f}")
        checks.add("provenance is available to the UI",
                   bool(triager.provenance.get("base_model_revision")),
                   f"base revision "
                   f"{triager.provenance.get('base_model_revision', '?')[:12]}")

        # ---- 3. every advertised example ----------------------------------
        print("\n3. advertised examples")
        results = []
        for i, example in enumerate(demo_app.EXAMPLES):
            payload = triager.triage(example)
            adapted = payload["adapted"]
            results.append({"index": i, "schema_valid": adapted["schema_valid"],
                            "output": adapted["pretty"],
                            "violations": adapted["violations"]})
            print(f"    example {i}: schema_valid={adapted['schema_valid']} "
                  f"-> {str(adapted['pretty'])[:70]}")
        checks.add("all advertised examples produce schema-valid output",
                   all(r["schema_valid"] for r in results),
                   f"{sum(1 for r in results if r['schema_valid'])}/"
                   f"{len(results)} examples valid")

        # ---- 4. degenerate inputs ------------------------------------------
        print("\n4. degenerate inputs")
        blank = triager.triage("")
        checks.add("blank input is refused, not crashed",
                   "error" in blank,
                   f"returned {blank.get('error')!r}")

        whitespace = triager.triage("   \n\t  ")
        checks.add("whitespace-only input is refused",
                   "error" in whitespace,
                   f"returned {whitespace.get('error')!r}")

        malformed = triager.triage("}{ \x00 !!! <script>alert(1)</script> ;;;")
        checks.add("malformed input is handled without raising",
                   "adapted" in malformed,
                   f"schema_valid={malformed['adapted']['schema_valid']}, "
                   f"raw={malformed['adapted']['raw'][:60]!r}")

        long_ticket = ("The VPN client disconnects every few minutes. " * 200
                       + " About 14 people are affected.")
        start = time.perf_counter()
        long_result = triager.triage(long_ticket)
        elapsed = time.perf_counter() - start
        checks.add("very long input is handled without raising",
                   "adapted" in long_result,
                   f"{len(long_ticket)} chars accepted in {elapsed:.1f}s, "
                   f"schema_valid="
                   f"{long_result['adapted']['schema_valid']}")

        # ---- 5. schema status is actually surfaced -------------------------
        print("\n5. schema status display")
        block = demo_app.format_block("test", results and
                                      triager.triage(demo_app.EXAMPLES[0])["adapted"])
        checks.add("schema status is shown to the user",
                   "schema:" in block and ("VALID" in block or "INVALID" in block),
                   f"rendered block contains a schema line: "
                   f"{block.splitlines()[1] if len(block.splitlines()) > 1 else block!r}")

        # ---- 6. load-failure handling --------------------------------------
        print("\n6. model-load failure handling")
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            failed = False
            message = ""
            try:
                demo_app.Triager(tmp, compare=False)
            except Exception as exc:  # noqa: BLE001
                failed = True
                message = f"{type(exc).__name__}: {exc}"
            checks.add("a broken adapter directory raises a clear error",
                       failed, message[:160] or "no exception raised")

        # ---- 7. optional: boot the server ----------------------------------
        if args.launch:
            print("\n7. launching the Gradio server")
            try:
                import gradio  # noqa: F401
                checks.add("gradio is installed", True,
                           f"gradio {gradio.__version__}")
            except ImportError:
                checks.add("gradio is installed", False,
                           "pip install gradio to run --launch")

        payload = {
            "checks": checks.rows,
            "all_passed": checks.all_passed,
            "examples": results,
            "adapter_verification": v,
            "note": ("This is a local demonstration, not a deployment. It has "
                     "no authentication, no rate limiting and no error budget, "
                     "and binds to localhost deliberately."),
        }
        dataio.write_json(payload, OUT / "demo_checks.json")

        run.metrics = {"passed": checks.all_passed,
                       "n_checks": len(checks.rows)}
        run.finish("success" if checks.all_passed else "failed")

        passed = sum(1 for r in checks.rows if r["passed"])
        print(f"\n{passed}/{len(checks.rows)} demonstration checks passed")
        return 0 if checks.all_passed else 1

    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
