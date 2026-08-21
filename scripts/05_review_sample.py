"""Dump a stratified sample of the dataset for human reading.

Automated checks prove structural properties. They cannot tell you that a ticket
reads like nonsense, or that a detail clause quietly contradicts the label. Only
reading does that, so this script produces the artefact that a reviewer actually
reads -- and that a sceptic can re-read.

    python scripts/05_review_sample.py

Output: reports/manual_review_sample.md -- one example per (split, category),
24 examples, deterministic given the dataset.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio  # noqa: E402
from forgelm.ledger import REPO_ROOT  # noqa: E402
from forgelm.schema import CATEGORIES  # noqa: E402
from forgelm.splits import apply_split, load_manifest  # noqa: E402

OUT = REPO_ROOT / "reports" / "manual_review_sample.md"


def main() -> int:
    records = dataio.read_jsonl(dataio.PROCESSED_DATASET)
    manifest = load_manifest(dataio.SPLIT_MANIFEST)
    by_split = apply_split(records, manifest)

    lines: list[str] = [
        "# Manual review sample",
        "",
        "One example per (split, category) -- 24 in total. Deterministic: the "
        "same 24 examples every time, so a second reviewer reads exactly what "
        "the first one did.",
        "",
        "Regenerate with `python scripts/05_review_sample.py`.",
        "",
        "## What to check when reading",
        "",
        "1. Does the ticket read like something a person would actually write?",
        "2. Is `users_affected` genuinely recoverable from the text?",
        "3. Do the clauses contradict each other or the label?",
        "4. Does the text give away `priority` directly rather than implying it?",
        "5. Are acronyms (CRM, VPN, MFA, SSO, DNS) correctly cased?",
        "",
        "Defects found in round 1 and since fixed are listed in "
        "`DATASET_CARD.md` under *Manual review*.",
        "",
    ]

    for split in ("train", "validation", "test"):
        lines.append(f"## {split} (n={len(by_split[split])})")
        lines.append("")
        seen: dict[str, dict] = {}
        for rec in sorted(by_split[split], key=lambda r: r["example_id"]):
            seen.setdefault(rec["category"], rec)
        for category in CATEGORIES:
            rec = seen.get(category)
            if rec is None:
                lines.append(f"### {category}\n\n_(no example in this split)_\n")
                continue
            lines.append(f"### {category} -- `{rec['example_id']}`")
            lines.append("")
            lines.append(f"- style: `{rec['template_family']}` | scenario: "
                         f"`{rec['scenario_family']}` | base_severity: "
                         f"{rec['base_severity']} | user_scale: `{rec['user_scale']}`")
            lines.append("")
            lines.append("```")
            lines.append(rec["ticket_text"])
            lines.append("```")
            lines.append("")
            lines.append(f"Expected: `{rec['expected_output_json']}`")
            lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} "
          f"({sum(1 for l in lines if l.startswith('### '))} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
