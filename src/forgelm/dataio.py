"""Reading and writing dataset artefacts, with checksums attached.

Anything that crosses a process boundary goes through here so that a checksum
is always available to record in the run ledger. A metric without a dataset
checksum next to it cannot be audited.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ledger import REPO_ROOT, sha256_file

DATA_DIR = REPO_ROOT / "data"
RAW_DATASET = DATA_DIR / "raw" / "tickets_v1.jsonl"
PROCESSED_DATASET = DATA_DIR / "processed" / "tickets_v1.validated.jsonl"
SPLIT_MANIFEST = DATA_DIR / "splits" / "split_manifest_v1.json"
DATASET_VERSION_FILE = DATA_DIR / "DATASET_VERSION.json"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_fingerprint() -> dict[str, Any]:
    """Everything needed to prove which data produced a result."""
    fp: dict[str, Any] = {}
    for name, path in (
        ("raw_dataset", RAW_DATASET),
        ("processed_dataset", PROCESSED_DATASET),
        ("split_manifest", SPLIT_MANIFEST),
    ):
        if path.exists():
            fp[name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        else:
            fp[name] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": None,
                        "missing": True}
    if DATASET_VERSION_FILE.exists():
        fp["version"] = read_json(DATASET_VERSION_FILE)
    return fp
