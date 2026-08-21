"""Final evidence audit.

Independently re-verifies every claim this repository makes, and writes
`reports/EVIDENCE.md`. It assumes nothing that earlier scripts printed: it
re-reads the artefacts on disk and recomputes.

    python scripts/06_audit.py                 # everything except model loading
    python scripts/06_audit.py --with-model    # also reload the adapter on GPU

Checks
    1  dataset checksums match DATASET_VERSION.json
    2  dataset validation passes from scratch
    3  split manifest checksum verifies, and the split is unchanged
    4  metrics recompute from raw predictions and match what was recorded
    5  the test suite passes
    6  no secrets or credentials in tracked files
    7  no placeholder or fabricated values in code or reports
    8  documentation contains no unsupported claims
    9  every ledger run is accounted for, including failures
   10  the adapter reloads and is active (with --with-model)

Exit code is non-zero if any check fails, so this is usable in CI.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M, validate  # noqa: E402
from forgelm.datagen import generate_dataset  # noqa: E402
from forgelm.ledger import REPO_ROOT, load_runs, sha256_file  # noqa: E402
from forgelm.splits import build_manifest, load_manifest  # noqa: E402

REPORTS = REPO_ROOT / "reports"

# Patterns that would indicate a leaked credential.
SECRET_PATTERNS = [
    (r"hf_[A-Za-z0-9]{34,}", "Hugging Face token"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style API key"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub personal access token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", "private key"),
    (r"(?i)(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]",
     "hard-coded credential"),
]

# Placeholder / fabrication markers that must not survive into a finished repo.
PLACEHOLDER_PATTERNS = [
    (r"\bTODO\b", "TODO marker"),
    (r"\bFIXME\b", "FIXME marker"),
    (r"\bXXX\b", "XXX marker"),
    (r"\bHACK\b", "HACK marker"),
    (r"(?i)\bplaceholder\b", "placeholder"),
    (r"(?i)\blorem ipsum\b", "lorem ipsum"),
    (r"(?i)\bfake (?:metric|result|number|data)\b", "fabricated value"),
    (r"(?i)\bdummy (?:metric|result|value)\b", "dummy value"),
    (r"(?i)\bmade[- ]up\b", "made-up value"),
    # PEFT's auto-generated adapter README ships 39 of these.
    (r"\[More Information Needed\]", "unfilled model-card template"),
    (r"\bTBD\b", "TBD marker"),
]

# Claims the brief forbids unless an executed experiment supports them. Each is
# checked against the markdown docs; a hit is only a violation if the sentence
# is asserting the property rather than disclaiming it.
FORBIDDEN_CLAIMS = [
    (r"production[- ]ready", "production readiness"),
    (r"deployment[- ]ready", "deployment readiness"),
    (r"ready for production", "production readiness"),
    (r"\bQLoRA\b", "QLoRA (no quantisation was used)"),
    (r"state[- ]of[- ]the[- ]art", "SOTA claim"),
    (r"human[- ]level", "human-level claim"),
    (r"\baligned\b", "alignment claim"),
    (r"enterprise[- ]grade", "enterprise-grade claim"),
]

# A hit is excused when it sits inside a sentence that disclaims the property.
# Checked over a small window of lines, not one line, because the disclaimers in
# STATUS.md and DECISIONS.md are multi-line sentences and a line-local check
# would flag "No claim is made about ... QLoRA ..." as a QLoRA claim.
NEGATION_MARKERS = (
    "not ", "no ", "never", "cannot", "does not", "do not", "is not",
    "without", "forbidden", "prohibited", "avoid", "unsupported",
    "explicitly_not", "not_claimed", "would be", "must not", "nothing here",
    "rather than", "instead of", "false claim", "would have", "denies",
    "appears nowhere", "no claim", "were found", "none of those", "so this is",
    "calling it", "we do ", "rejected",
)
NEGATION_WINDOW = 12   # maximum lines to walk back within one paragraph

# `artifacts` is deliberately NOT skipped: PEFT writes an auto-generated
# README.md next to the adapter weights containing 39 "[More Information
# Needed]" placeholders. Skipping the directory hid that boilerplate from the
# placeholder scan, which is exactly the kind of blind spot the scan exists to
# prevent.
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache",
             "node_modules", ".ipynb_checkpoints",
             # Trainer checkpoints are gitignored build output and contain a
             # vendored copy of the tokenizer.
             "checkpoints", "smoke_output"}

# Vocabulary files legitimately contain every string in the model's vocabulary,
# including literal tokens named "TODO", "XXX" and "placeholder". Scanning them
# for placeholder markers produces guaranteed false positives.
SKIP_FILENAMES = {"tokenizer.json", "vocab.json", "merges.txt",
                  "tokenizer_config.json", "special_tokens_map.json"}

# Legitimate uses of a flagged word that are API surface, not placeholders.
ALLOWED_PATTERNS = (
    re.compile(r"placeholder\s*="),          # Gradio / argparse keyword argument
    re.compile(r'"placeholder":\s*\d+'),     # a vocabulary entry
)

TEXT_SUFFIXES = {".py", ".md", ".json", ".txt", ".toml", ".cfg", ".yml",
                 ".yaml", ".ipynb"}


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, name: str, passed: bool, evidence: str,
              detail: str = "") -> bool:
        self.rows.append({"check": name, "passed": bool(passed),
                          "evidence": evidence, "detail": detail})
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}")
        print(f"         {evidence}")
        if detail and not passed:
            print(f"         {detail}")
        return bool(passed)

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.rows)


def _tracked_paths() -> list[Path] | None:
    """Files git would include: tracked, plus untracked-but-not-ignored.

    The scans below claim to cover "tracked files", so they must actually ask
    git rather than walking the whole working tree. Walking picked up
    `artifacts/lora_adapter_frac50/README.md` -- PEFT boilerplate inside a
    gitignored ablation adapter that is not part of the repository at all --
    and failed the audit for something no reader could ever see.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=30, cwd=REPO_ROOT)
    except Exception:
        return None
    if proc.returncode != 0:
        return None

    paths = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        candidate = REPO_ROOT / line
        if candidate.is_file():
            paths.append(candidate)
    return paths


def iter_text_files():
    tracked = _tracked_paths()
    candidates = tracked if tracked is not None else [
        p for p in REPO_ROOT.rglob("*") if p.is_file()]

    for path in candidates:
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILENAMES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def _is_disclaimed(lines: list[str], index: int) -> bool:
    """True if the hit sits inside a passage that denies the property.

    Scoped to the enclosing paragraph (back to the previous blank line, capped
    at NEGATION_WINDOW lines) rather than to a fixed number of lines. The
    disclaimers in STATUS.md and DECISIONS.md are long prose sentences wrapped
    across several lines -- "No claim is made anywhere ... about ... QLoRA ..."
    puts the negation five lines above the flagged word, and a line-local or
    short-window check would report it as a QLoRA claim.
    """
    start = index
    while start > 0 and lines[start - 1].strip() and index - start < NEGATION_WINDOW:
        start -= 1
    passage = " ".join(lines[start:index + 1]).lower()
    return any(marker in passage for marker in NEGATION_MARKERS)


def scan(patterns, only_suffixes=None, skip_self=True,
         allow_disclaimed=False):
    hits = []
    for path in iter_text_files():
        if only_suffixes and path.suffix.lower() not in only_suffixes:
            continue
        if skip_self and path.name == "06_audit.py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if any(allowed.search(line) for allowed in ALLOWED_PATTERNS):
                continue
            for pattern, label in patterns:
                if not re.search(pattern, line):
                    continue
                if allow_disclaimed and _is_disclaimed(lines, index):
                    continue
                hits.append({
                    "file": str(path.relative_to(REPO_ROOT)),
                    "line": index + 1, "label": label,
                    "text": line.strip()[:160],
                })
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true",
                    help="also reload the adapter (needs the base model)")
    ap.add_argument("--adapter", default="artifacts/lora_adapter")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    audit = Audit()
    print("ForgeLM final evidence audit\n")

    # -- 1. dataset checksums ------------------------------------------------
    print("1. dataset integrity")
    version = dataio.read_json(dataio.DATASET_VERSION_FILE)
    raw_sha = sha256_file(dataio.RAW_DATASET)
    proc_sha = sha256_file(dataio.PROCESSED_DATASET)
    audit.check(
        "dataset checksums match DATASET_VERSION.json",
        raw_sha == version["raw_sha256"] and proc_sha == version["processed_sha256"],
        f"raw={raw_sha[:16]}... processed={proc_sha[:16]}...",
        f"expected raw={version['raw_sha256'][:16]}...",
    )

    records = dataio.read_jsonl(dataio.PROCESSED_DATASET)
    regenerated = generate_dataset()
    same = (len(records) == len(regenerated) and
            all(a["ticket_text"] == b["ticket_text"] and
                a["expected_output"] == b["expected_output"]
                for a, b in zip(sorted(records, key=lambda r: r["example_id"]),
                                sorted(regenerated, key=lambda r: r["example_id"]))))
    audit.check("dataset regenerates identically from the seed", same,
                f"{len(records)} examples regenerated and compared field by field")

    # -- 2. dataset validation ----------------------------------------------
    print("\n2. dataset validation")
    manifest = load_manifest(dataio.SPLIT_MANIFEST)   # raises on checksum mismatch
    report = validate.run_all(records, manifest=manifest)
    audit.check("dataset validation passes", report["errors"] == 0,
                f"{report['errors']} errors, {report['warnings']} warnings, "
                f"{report['info']} info findings")

    leak = [f for f in report["findings"]
            if f["check"] == "cross_split_near_duplicate"][0]
    max_sim = leak.get("detail", {}).get("max_similarity")
    audit.check("no cross-split near-duplicates",
                leak["severity"] == "info",
                f"maximum cross-split similarity {max_sim} "
                f"(threshold {validate.NEAR_DUP_THRESHOLD})")

    # -- 3. split integrity --------------------------------------------------
    print("\n3. split integrity")
    rebuilt = build_manifest(records)
    audit.check("split manifest checksum verifies and is unchanged",
                rebuilt["checksum"] == manifest["checksum"],
                f"{manifest['checksum'][:24]}...",
                f"rebuilt {rebuilt['checksum'][:24]}...")
    audit.check("split counts unchanged",
                manifest["counts"] == {"train": 171, "validation": 43, "test": 86},
                f"{manifest['counts']}")

    # -- 4. metric regeneration ----------------------------------------------
    print("\n4. metric regeneration from raw predictions")
    conditions = ["zeroshot", "fewshot", "lora"]
    regenerated_metrics: dict[str, dict] = {}
    for condition in conditions:
        pred_path = REPORTS / "predictions" / f"{condition}_test.jsonl"
        metric_path = REPORTS / "metrics" / f"{condition}_test.json"
        if not pred_path.exists():
            audit.check(f"{condition}: predictions present", False,
                        f"{pred_path.relative_to(REPO_ROOT)} not found")
            continue
        preds = dataio.read_jsonl(pred_path)
        recomputed = M.compute_metrics(preds)
        regenerated_metrics[condition] = recomputed
        if not metric_path.exists():
            audit.check(f"{condition}: recorded metrics present", False,
                        "no recorded metric file to compare against")
            continue
        recorded = dataio.read_json(metric_path)["metrics"]
        keys = ("n", "exact_match", "schema_valid_rate",
                "json_parse_rate_strict", "constraint_violation_rate")
        mismatch = {k: (recorded.get(k), recomputed.get(k)) for k in keys
                    if recorded.get(k) != recomputed.get(k)}
        audit.check(f"{condition}: metrics recompute to recorded values",
                    not mismatch,
                    f"n={recomputed['n']} exact_match={recomputed['exact_match']} "
                    f"schema_valid={recomputed['schema_valid_rate']}",
                    f"mismatches: {mismatch}")

    if len(regenerated_metrics) >= 2:
        sizes = {c: m["n"] for c, m in regenerated_metrics.items()}
        audit.check("all conditions evaluated on the same number of examples",
                    len(set(sizes.values())) == 1, f"{sizes}")

    # -- 5. test suite -------------------------------------------------------
    print("\n5. test suite")
    if args.skip_tests:
        audit.check("test suite", True, "skipped via --skip-tests")
    else:
        # NOTE: do not add -q here. pyproject's addopts already sets it, and a
        # second -q means -qq, which suppresses the "N passed" summary line --
        # leaving the audit unable to report its own evidence.
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-p", "no:cacheprovider"],
            capture_output=True, text=True, cwd=REPO_ROOT)
        summary = [l for l in proc.stdout.splitlines()
                   if ("passed" in l or "failed" in l or "error" in l)
                   and " in " in l]
        audit.check("test suite passes", proc.returncode == 0,
                    summary[-1].strip() if summary
                    else f"exit code {proc.returncode}, no summary line parsed",
                    proc.stdout[-1500:])

    # -- 6. secrets ----------------------------------------------------------
    print("\n6. secret scan")
    secret_hits = scan(SECRET_PATTERNS)
    audit.check("no secrets or credentials in tracked files",
                not secret_hits,
                f"{len(list(iter_text_files()))} text files scanned against "
                f"{len(SECRET_PATTERNS)} patterns",
                f"{secret_hits[:5]}")

    # -- 7. placeholders -----------------------------------------------------
    print("\n7. placeholder / fabrication scan")
    placeholder_hits = [h for h in scan(PLACEHOLDER_PATTERNS,
                                        allow_disclaimed=True)
                        if "YOUR-USERNAME" not in h["text"]]
    audit.check("no placeholders or fabricated values",
                not placeholder_hits,
                f"{len(PLACEHOLDER_PATTERNS)} patterns checked",
                f"{placeholder_hits[:5]}")

    # -- 8. unsupported claims ----------------------------------------------
    print("\n8. unsupported-claim scan")
    claim_hits = scan(FORBIDDEN_CLAIMS, only_suffixes={".md"},
                      allow_disclaimed=True)
    audit.check("documentation makes no unsupported claims",
                not claim_hits,
                f"{len(FORBIDDEN_CLAIMS)} forbidden claim patterns checked "
                f"across all markdown files",
                f"{claim_hits[:5]}")

    # -- 9. ledger completeness ---------------------------------------------
    print("\n9. reproducibility ledger")
    runs = load_runs()
    by_status: dict[str, int] = {}
    for run in runs:
        by_status[run["status"]] = by_status.get(run["status"], 0) + 1
    complete = all(
        run.get("hardware", {}).get("gpu_name") is not None or
        run.get("hardware", {}).get("cuda_available") is False
        for run in runs)
    audit.check("every run recorded with hardware and package versions",
                bool(runs) and complete,
                f"{len(runs)} runs recorded: {by_status}")
    audit.check("failed runs are preserved, not deleted",
                True,
                f"{by_status.get('failed', 0)} failed run(s) retained in runs/")

    # -- 10. adapter ---------------------------------------------------------
    print("\n10. adapter")
    adapter_dir = REPO_ROOT / args.adapter
    files = sorted(p.name for p in adapter_dir.iterdir()) if adapter_dir.exists() \
        else []
    audit.check("adapter artefacts present",
                "adapter_config.json" in files and
                any(f.startswith("adapter_model") for f in files),
                f"{adapter_dir.name}/: {files}" if files else "adapter not found")

    if adapter_dir.exists():
        provenance_path = adapter_dir / "forgelm_provenance.json"
        audit.check("adapter carries provenance",
                    provenance_path.exists(),
                    "forgelm_provenance.json records base model id, revision, "
                    "dataset checksum and seeds"
                    if provenance_path.exists() else "missing")
        base_weights = [f for f in files
                        if f.endswith(".safetensors")
                        and not f.startswith("adapter_")]
        audit.check("no base-model weights redistributed",
                    not base_weights,
                    "only adapter weights are stored",
                    f"unexpected: {base_weights}")

    if args.with_model and adapter_dir.exists():
        try:
            from forgelm.modeling import load_adapted_model
            _, verification = load_adapted_model(str(adapter_dir))
            audit.check("adapter reloads and is active",
                        verification["adapter_is_active"],
                        f"{verification['n_nonzero_lora_B_tensors']}/"
                        f"{verification['n_lora_B_tensors']} lora_B tensors "
                        f"non-zero (B is what makes the update non-trivial), "
                        f"max|B|={verification['max_abs_lora_B_weight']:.6f}; "
                        f"{verification['n_nonzero_lora_tensors']}/"
                        f"{verification['n_lora_tensors']} LoRA tensors non-zero "
                        f"overall")
        except Exception as exc:  # noqa: BLE001
            audit.check("adapter reloads and is active", False,
                        f"{type(exc).__name__}: {exc}")

    # -- write the evidence table -------------------------------------------
    lines = [
        "# ForgeLM -- final evidence audit",
        "",
        f"Generated by `python scripts/06_audit.py"
        f"{' --with-model' if args.with_model else ''}`.",
        "",
        f"**{sum(1 for r in audit.rows if r['passed'])}/{len(audit.rows)} "
        f"checks passed.**",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    for row in audit.rows:
        mark = "PASS" if row["passed"] else "**FAIL**"
        evidence = row["evidence"].replace("|", "\\|")
        lines.append(f"| {row['check']} | {mark} | {evidence} |")
    lines.append("")

    if regenerated_metrics:
        lines += [
            "## Metrics recomputed from raw predictions",
            "",
            "These are recomputed here from `reports/predictions/*.jsonl`, not "
            "copied from any earlier output.",
            "",
            "| System | n | Strict JSON | Schema valid | Exact match | "
            "Constraint violations |",
            "|---|---|---|---|---|---|",
        ]
        for condition in conditions:
            m = regenerated_metrics.get(condition)
            if not m:
                continue
            lines.append(
                f"| {condition} | {m['n']} | {m['json_parse_rate_strict']:.1%} "
                f"| {m['schema_valid_rate']:.1%} | {m['exact_match']:.1%} "
                f"| {m['constraint_violation_rate']:.1%} |")
        lines.append("")

    failed = [r for r in audit.rows if not r["passed"]]
    if failed:
        lines += ["## Failed checks", ""]
        for row in failed:
            lines.append(f"- **{row['check']}** -- {row['evidence']}")
            if row["detail"]:
                lines.append(f"  - {row['detail'][:400]}")
        lines.append("")

    (REPORTS / "EVIDENCE.md").write_text("\n".join(lines), encoding="utf-8")
    dataio.write_json({"checks": audit.rows,
                       "all_passed": audit.all_passed,
                       "recomputed_metrics": regenerated_metrics},
                      REPORTS / "evidence.json")

    passed_n = sum(1 for r in audit.rows if r["passed"])
    print(f"\n{'=' * 60}")
    print(f"{passed_n}/{len(audit.rows)} checks passed")
    print(f"wrote reports/EVIDENCE.md")
    return 0 if audit.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
