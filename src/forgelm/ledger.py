"""Reproducibility ledger.

Every executed run writes one JSON record under runs/<run_id>/run.json capturing
the environment, inputs, configuration and outcome. The point is that a reader
can tell, months later, exactly what produced a number -- including runs that
failed, which are recorded rather than deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

# Packages whose versions materially affect results.
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "tokenizers",
    "numpy",
    "safetensors",
    "scipy",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_cmd(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=20, cwd=REPO_ROOT
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_info() -> dict[str, Any]:
    commit = _run_cmd(["git", "rev-parse", "HEAD"])
    dirty = _run_cmd(["git", "status", "--porcelain"])
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty) if dirty is not None else None,
        "dirty_files": (dirty.splitlines() if dirty else []),
    }


def package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version_torch"] = torch.version.cuda
        info["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_total_memory_bytes"] = props.total_memory
            info["gpu_total_memory_gib"] = round(props.total_memory / (1024 ** 3), 3)
            info["gpu_capability"] = f"sm_{props.major}{props.minor}"
            info["gpu_multi_processor_count"] = props.multi_processor_count
            # bf16 needs Ampere (sm_80+). Recording this explicitly because it
            # dictates the training precision choice.
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
        else:
            info["gpu_name"] = None
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False

    smi = _run_cmd(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"]
    )
    info["nvidia_smi"] = smi
    return info


@dataclass
class Run:
    """A single executed unit of work with a durable audit record."""

    kind: str                       # e.g. "baseline_zeroshot", "train_lora"
    config: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default="")
    inputs: dict[str, Any] = field(default_factory=dict)
    seeds: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    status: str = "created"

    _t0: float = field(default=0.0, repr=False)
    _started: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.run_id:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.run_id = f"{stamp}_{self.kind}_{uuid.uuid4().hex[:6]}"
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "Run":
        self._t0 = time.time()
        self._started = datetime.now(timezone.utc).isoformat()
        self.status = "running"
        self._flush()
        return self

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def add_artifact(self, name: str, path: str | Path) -> None:
        p = Path(path)
        self.artifacts[name] = str(p.relative_to(REPO_ROOT)) if p.is_absolute() and \
            str(p).startswith(str(REPO_ROOT)) else str(p)

    def finish(self, status: str = "success") -> dict[str, Any]:
        self.status = status
        return self._flush()

    def fail(self, exc: BaseException) -> dict[str, Any]:
        self.status = "failed"
        self.notes.append(f"exception: {type(exc).__name__}: {exc}")
        (self.dir / "traceback.txt").write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )
        return self._flush()

    # -- serialisation -----------------------------------------------------

    def peak_gpu_memory(self) -> dict[str, Any]:
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
                    "peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved() / (1024 ** 3), 3),
                }
        except Exception:
            pass
        return {"peak_allocated_bytes": None,
                "note": "not measurable (no CUDA device)"}

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "started_utc": self._started or None,
            "ended_utc": datetime.now(timezone.utc).isoformat()
            if self.status not in ("created", "running") else None,
            "elapsed_seconds": round(now - self._t0, 3) if self._t0 else None,
            "git": git_info(),
            "hardware": hardware_info(),
            "packages": package_versions(),
            "seeds": self.seeds,
            "inputs": self.inputs,
            "config": self.config,
            "metrics": self.metrics,
            "peak_memory": self.peak_gpu_memory(),
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "notes": self.notes,
        }

    def _flush(self) -> dict[str, Any]:
        payload = self.to_dict()
        (self.dir / "run.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return payload


def load_runs(kind: str | None = None) -> list[dict[str, Any]]:
    """Read every ledger record, optionally filtered by kind."""
    out = []
    if not RUNS_DIR.exists():
        return out
    for rec in sorted(RUNS_DIR.glob("*/run.json")):
        try:
            data = json.loads(rec.read_text(encoding="utf-8"))
        except Exception:
            continue
        if kind is None or data.get("kind") == kind:
            out.append(data)
    return out
