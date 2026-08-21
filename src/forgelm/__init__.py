"""ForgeLM: a small, reproducible LoRA adaptation experiment.

Research question
-----------------
Can parameter-efficient LoRA fine-tuning measurably improve a small,
permissively licensed instruction model on a narrow structured-output task,
compared with zero-shot and few-shot use of the unchanged base model, within a
free or low-cost compute budget?

This package holds the reusable implementation. Scripts in `scripts/` and the
notebook in `notebooks/` orchestrate it; neither contains logic that is not
also importable and testable from here.
"""

__version__ = "1.0.0"

EXPERIMENT_ID = "forgelm-lora-ticket-triage-v1"
