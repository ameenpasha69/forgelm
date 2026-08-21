"""Integration tests that touch the model.

Marked `slow` (and `gpu` where a CUDA device is genuinely required) so the
default suite stays fast:

    pytest tests/ -m "not slow"      # fast, no model download
    pytest tests/                    # everything

These cover the claims that cannot be checked without real weights: that LoRA
attaches to the modules we intended, that a saved adapter reloads and is
actually active, and that the whole train-save-reload-generate path works
end to end.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from forgelm import training as T
from forgelm.config import TRAINING
from forgelm.modeling import (
    BASE_MODEL_ID, BASE_MODEL_REVISION, LORA_TARGET_MODULES, build_lora_model,
    load_adapted_model, load_base_model, parameter_report, select_precision,
)
from forgelm.parsing import parse_response
from forgelm.prompts import render_prompt

pytestmark = pytest.mark.slow

ADAPTER_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "lora_adapter"


# --------------------------------------------------------------------------
# Precision selection
# --------------------------------------------------------------------------

def test_precision_never_selects_bf16_on_pre_ampere(cuda_available):
    """The whole point of select_precision(): do not trust
    torch.cuda.is_bf16_supported() on Turing."""
    precision = select_precision()
    if not cuda_available:
        assert precision["device"] == "cpu"
        return
    import torch

    props = torch.cuda.get_device_properties(0)
    capability = props.major * 10 + props.minor
    if capability < 80:
        assert precision["bf16"] is False, (
            f"bf16 selected on sm_{props.major}{props.minor}, which has no "
            f"native bf16 datapath")
        assert precision["fp16"] is True
    else:
        assert precision["bf16"] is True


# --------------------------------------------------------------------------
# LoRA attachment
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lora_model():
    pytest.importorskip("peft")
    precision = select_precision()
    config = {**TRAINING, "fp16": precision["fp16"], "bf16": precision["bf16"]}
    base = load_base_model()
    model, _, _ = build_lora_model(base, config)
    yield model
    del model, base
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pass


def test_lora_targets_every_intended_projection(lora_model):
    params = parameter_report(lora_model)
    adapted = {
        parts[i - 1]
        for name in params["trainable_module_names"]
        for parts in [name.split(".")]
        for i, p in enumerate(parts)
        if p.startswith("lora_") and i > 0
    }
    assert adapted == set(LORA_TARGET_MODULES), (
        f"expected {sorted(LORA_TARGET_MODULES)}, adapted {sorted(adapted)}")


def test_only_lora_parameters_are_trainable(lora_model):
    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name, f"non-LoRA parameter is trainable: {name}"


def test_trainable_fraction_is_small(lora_model):
    params = parameter_report(lora_model)
    assert 0.5 < params["trainable_percent"] < 5.0, (
        f"{params['trainable_percent']}% trainable is outside the expected "
        f"range for r={TRAINING['lora_r']}")
    assert params["trainable_params"] > 0


def test_trainable_parameters_are_fp32(lora_model):
    """fp16 master weights lose small optimiser updates to rounding."""
    params = parameter_report(lora_model)
    assert set(params["trainable_param_dtypes"]) == {"torch.float32"}, (
        f"trainable dtypes: {params['trainable_param_dtypes']}")


def test_base_weights_remain_half_precision(lora_model, cuda_available):
    if not cuda_available:
        pytest.skip("cpu path uses float32 throughout")
    params = parameter_report(lora_model)
    assert "torch.float16" in params["param_dtypes"]


def test_forward_pass_produces_finite_loss(lora_model, tokenizer, records):
    encoded = T.build_dataset(tokenizer, records[:4], TRAINING["max_seq_len"])
    result = T.one_batch_forward(lora_model, tokenizer, encoded, batch_size=1)
    assert result["healthy"], result
    assert result["n_supervised_tokens"] > 0


# --------------------------------------------------------------------------
# End-to-end: train a tiny adapter, save it, reload it, generate with it
# --------------------------------------------------------------------------

@pytest.mark.gpu
def test_train_save_reload_generate_roundtrip(tokenizer, records, tmp_path,
                                              cuda_available):
    """The smallest possible complete run of the real pipeline.

    Two examples, a handful of steps. It proves the plumbing, not the science:
    that gradients flow, that saving produces a loadable artefact, and that the
    reloaded model generates text.
    """
    if not cuda_available:
        pytest.skip("needs a CUDA device")
    pytest.importorskip("peft")
    import gc
    import torch

    precision = select_precision()
    config = {**TRAINING, "fp16": precision["fp16"], "bf16": precision["bf16"],
              "lora_r": 4, "lora_alpha": 8}
    tokenizer.padding_side = "right"
    encoded = T.build_dataset(tokenizer, records[:2], config["max_seq_len"])

    base = load_base_model()
    model, _, _ = build_lora_model(base, config)

    overfit = T.tiny_overfit(model, tokenizer, encoded, steps=10, lr=1e-3,
                             n_examples=2, micro_batch=1)
    assert overfit["all_finite"], overfit

    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(str(adapter_dir))
    assert (adapter_dir / "adapter_config.json").exists()
    assert any(p.name.startswith("adapter_model")
               for p in adapter_dir.iterdir())

    # The adapter must be far smaller than the base model it modifies.
    size = sum(p.stat().st_size for p in adapter_dir.iterdir() if p.is_file())
    assert size < 200_000_000, f"adapter is {size/1e6:.0f} MB"

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    reloaded, verification = load_adapted_model(str(adapter_dir))
    assert verification["adapter_is_active"]
    assert verification["n_nonzero_lora_tensors"] > 0

    from forgelm.generate import generate_batch

    prompt = render_prompt(tokenizer, records[0]["ticket_text"])
    out = generate_batch(reloaded, tokenizer, [prompt], max_new_tokens=64)[0]
    assert isinstance(out["text"], str)
    assert out["finish_reason"] in ("stop", "length")

    del reloaded
    gc.collect()
    torch.cuda.empty_cache()


def test_inert_adapter_is_rejected(tmp_path, tokenizer, records, cuda_available):
    """A saved adapter with all-zero B matrices is identical to the base model.

    It would load without error and silently reproduce base-model results, which
    a reader would misinterpret as 'fine-tuning did not help'. load_adapted_model
    must refuse it.
    """
    if not cuda_available:
        pytest.skip("needs a CUDA device")
    pytest.importorskip("peft")
    import gc
    import torch

    precision = select_precision()
    config = {**TRAINING, "fp16": precision["fp16"], "bf16": precision["bf16"],
              "lora_r": 4, "lora_alpha": 8}
    base = load_base_model()
    model, _, _ = build_lora_model(base, config)

    # A freshly initialised LoRA has zero B matrices by construction, so an
    # untrained adapter is exactly the inert case we want to reject.
    adapter_dir = tmp_path / "inert"
    model.save_pretrained(str(adapter_dir))

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    with pytest.raises(RuntimeError, match="inert"):
        load_adapted_model(str(adapter_dir))

    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# The trained adapter shipped in artifacts/
# --------------------------------------------------------------------------

@pytest.mark.skipif(not ADAPTER_DIR.exists(),
                    reason="no trained adapter; run scripts/03_train_lora.py")
def test_shipped_adapter_reloads_and_is_active(cuda_available):
    if not cuda_available:
        pytest.skip("needs a CUDA device")
    model, verification = load_adapted_model(str(ADAPTER_DIR))
    assert verification["adapter_is_active"]
    assert verification["n_nonzero_lora_tensors"] == \
        verification["n_lora_tensors"]

    del model
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not ADAPTER_DIR.exists(),
                    reason="no trained adapter; run scripts/03_train_lora.py")
def test_shipped_adapter_has_provenance():
    import json

    path = ADAPTER_DIR / "forgelm_provenance.json"
    assert path.exists(), "adapter has no provenance file"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    assert provenance["base_model_id"] == BASE_MODEL_ID
    assert provenance["base_model_revision"] == BASE_MODEL_REVISION
    assert provenance["split_manifest_checksum"]
    assert provenance["seeds"]


@pytest.mark.skipif(not ADAPTER_DIR.exists(),
                    reason="no trained adapter; run scripts/03_train_lora.py")
def test_shipped_adapter_does_not_redistribute_base_weights():
    base_weights = [p.name for p in ADAPTER_DIR.iterdir()
                    if p.suffix == ".safetensors"
                    and not p.name.startswith("adapter_")]
    assert base_weights == [], (
        f"base-model weights present in the adapter directory: {base_weights}")


@pytest.mark.gpu
@pytest.mark.skipif(not ADAPTER_DIR.exists(),
                    reason="no trained adapter; run scripts/03_train_lora.py")
def test_shipped_adapter_emits_parseable_json(tokenizer, records, cuda_available):
    """The headline behavioural claim, checked directly on a few examples."""
    if not cuda_available:
        pytest.skip("needs a CUDA device")
    import gc
    import torch

    from forgelm.generate import generate_batch

    model, _ = load_adapted_model(str(ADAPTER_DIR))
    prompts = [render_prompt(tokenizer, r["ticket_text"]) for r in records[:4]]
    outputs = generate_batch(model, tokenizer, prompts, max_new_tokens=160)

    parsed = [parse_response(o["text"], o["finish_reason"]) for o in outputs]
    assert all(p.lenient_json for p in parsed), \
        [o["text"][:100] for o in outputs]

    del model
    gc.collect()
    torch.cuda.empty_cache()
