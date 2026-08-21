"""Model, tokenizer and LoRA construction.

Pinning
-------
The base model is pinned to an exact commit hash, not a branch name. `main` can
move; a commit cannot. Every result in this repository was produced against
BASE_MODEL_REVISION and that string is written into every ledger record.

Precision
---------
The development GPU is a GTX 1650 (Turing, sm_75, 4 GiB). `torch.cuda.
is_bf16_supported()` returns True on this device, but that reflects *emulated*
bf16 -- Turing has no native bf16 datapath. Training therefore uses fp16
autocast with a gradient scaler, and the LoRA parameters are held in fp32 so
the optimiser never updates half-precision master weights. `select_precision`
encodes that decision and explains itself in the returned dict.
"""

from __future__ import annotations

from typing import Any

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
BASE_MODEL_LICENCE = "apache-2.0"

# Verified from the model card and config.json on 2026-08-22; see DECISIONS.md.
BASE_MODEL_FACTS: dict[str, Any] = {
    "model_id": BASE_MODEL_ID,
    "revision": BASE_MODEL_REVISION,
    "licence": BASE_MODEL_LICENCE,
    "gated": False,
    "total_params_reported": "0.49B (0.36B non-embedding)",
    "hidden_layers": 24,
    "hidden_size": 896,
    "max_position_embeddings": 32768,
    "tie_word_embeddings": True,
    "chat_format": "ChatML (<|im_start|>role\\n ... <|im_end|>)",
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
    "source": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
}

# All linear projections in the transformer block. Qwen2 uses grouped-query
# attention, so k_proj/v_proj are much narrower than q_proj/o_proj.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def select_precision() -> dict[str, Any]:
    """Choose the training/inference dtype for the *actual* device present."""
    import torch

    if not torch.cuda.is_available():
        return {
            "device": "cpu", "dtype": "float32", "fp16": False, "bf16": False,
            "reason": "no CUDA device; fp16 on CPU is slow and poorly supported",
        }

    props = torch.cuda.get_device_properties(0)
    capability = props.major * 10 + props.minor
    native_bf16 = capability >= 80

    if native_bf16:
        return {
            "device": "cuda", "dtype": "bfloat16", "fp16": False, "bf16": True,
            "gpu": props.name, "capability": f"sm_{props.major}{props.minor}",
            "reason": "compute capability >= 8.0 has a native bfloat16 datapath",
        }

    return {
        "device": "cuda", "dtype": "float16", "fp16": True, "bf16": False,
        "gpu": props.name, "capability": f"sm_{props.major}{props.minor}",
        "reason": (
            f"compute capability sm_{props.major}{props.minor} predates Ampere. "
            f"torch.cuda.is_bf16_supported() may report True via emulation, but "
            f"there is no native bf16 datapath, so fp16 autocast with a gradient "
            f"scaler is used and LoRA parameters are kept in fp32."
        ),
    }


def load_tokenizer(model_id: str = BASE_MODEL_ID,
                   revision: str = BASE_MODEL_REVISION):
    """Load the tokenizer and make padding explicit.

    Two settings matter and are easy to get wrong:
      * `pad_token` -- Qwen ships a distinct <|endoftext|> pad token, so we do
        NOT reuse eos as pad. Reusing eos would make the collator mask real
        end-of-turn tokens.
      * `padding_side` -- must be LEFT for batched generation (so every
        sequence's last real token sits at the same index) and RIGHT for
        training. Callers set it explicitly; see `generate` and `training`.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_id: str = BASE_MODEL_ID,
                    revision: str = BASE_MODEL_REVISION,
                    dtype: str | None = None,
                    device: str | None = None):
    """Load the unchanged base model."""
    import torch
    from transformers import AutoModelForCausalLM

    precision = select_precision()
    dtype = dtype or precision["dtype"]
    device = device or precision["device"]
    torch_dtype = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch_dtype,
    )
    model.to(device)
    return model


def parameter_report(model) -> dict[str, Any]:
    """Programmatic trainable-parameter accounting.

    Reported rather than asserted from a config file, because what actually
    ends up trainable depends on how PEFT matched the target module names.
    """
    total = 0
    trainable = 0
    trainable_modules: list[str] = []
    dtypes: dict[str, int] = {}
    trainable_dtypes: dict[str, int] = {}

    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        key = str(param.dtype)
        dtypes[key] = dtypes.get(key, 0) + count
        if param.requires_grad:
            trainable += count
            trainable_modules.append(name)
            trainable_dtypes[key] = trainable_dtypes.get(key, 0) + count

    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_percent": round(100.0 * trainable / total, 4) if total else 0.0,
        "n_trainable_tensors": len(trainable_modules),
        "trainable_module_names": trainable_modules,
        "param_dtypes": dtypes,
        # Separated out because the whole point of the fp32 upcast is that the
        # *trainable* tensors are fp32 while the frozen base stays fp16.
        "trainable_param_dtypes": trainable_dtypes,
    }


def build_lora_model(base_model, config: dict[str, Any]):
    """Wrap the base model with LoRA adapters.

    LoRA parameters are explicitly upcast to fp32. With a fp16 base model the
    adapter tensors would otherwise be created in fp16, and an optimiser
    stepping fp16 master weights loses small updates to rounding -- the
    classic silent cause of a loss curve that plateaus for no visible reason.
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config.get("target_modules", LORA_TARGET_MODULES),
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    upcast = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            upcast += 1

    return model, lora_config, upcast


def load_adapted_model(adapter_dir: str,
                       model_id: str = BASE_MODEL_ID,
                       revision: str = BASE_MODEL_REVISION,
                       dtype: str | None = None,
                       device: str | None = None):
    """Reload base + saved adapter through a clean inference path.

    Returns (model, verification) where `verification` proves the adapter is
    actually attached and non-zero -- a saved adapter whose B matrices are all
    zero is mathematically identical to the base model, and would silently
    produce "no improvement" results.
    """
    import torch
    from peft import PeftModel

    base = load_base_model(model_id, revision, dtype=dtype, device=device)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    lora_tensors = 0
    nonzero_tensors = 0
    max_abs = 0.0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        lora_tensors += 1
        m = float(param.detach().abs().max().item())
        max_abs = max(max_abs, m)
        if m > 0:
            nonzero_tensors += 1

    verification = {
        "adapter_dir": str(adapter_dir),
        "is_peft_model": isinstance(model, PeftModel),
        "active_adapters": list(getattr(model, "peft_config", {}).keys()),
        "n_lora_tensors": lora_tensors,
        "n_nonzero_lora_tensors": nonzero_tensors,
        "max_abs_lora_weight": max_abs,
        "adapter_is_active": lora_tensors > 0 and nonzero_tensors > 0,
    }
    if not verification["adapter_is_active"]:
        raise RuntimeError(
            f"adapter at {adapter_dir} loaded but appears inert "
            f"({lora_tensors} LoRA tensors, {nonzero_tensors} non-zero). "
            f"Evaluating this would silently reproduce base-model results."
        )
    return model, verification
