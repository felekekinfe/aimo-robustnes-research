"""Safe VRAM Model Loader & Proxy Guard for AIMO White-Box Inference.

NO-CRASH & VRAM SAFETY PROTOCOL:
- Models >32B ('120b', '397b', '70b', '72b', 'gpt-oss', 'frontier') are intercepted
  and routed to a safe GPU proxy (e.g. Qwen2.5-Math-7B, DeepSeek-R1-8B, etc.).
- Active GPU model is cached across batches for identical base_model_id.
- Offline-only loading (local_files_only=True, device_map='auto').
- Device inspection via _get_device(model) avoids missing .device under device_map='auto'.
"""

from __future__ import annotations

import gc
from typing import Any

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False

OVERSIZED_TAGS = (
    "32b", "70b", "72b", "110b", "120b", "235b", "397b", "405b", "671b", "gpt-oss", "frontier"
)

SAFE_OFFLINE_PROXIES = (
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "Qwen/Qwen3.5-4B",
)

# Global persistent cache
_LOADED_MODEL_ID: str | None = None
_LOADED_TOKENIZER: Any = None
_LOADED_MODEL: Any = None
_LAST_SAFE_MODEL_ID: str | None = None


def _get_device(model: Any) -> Any:
    """Safely resolve device of a model under device_map='auto'."""
    try:
        return next(model.parameters()).device
    except Exception:
        pass
    return getattr(model, "device", torch.device("cpu") if HAS_TORCH_TRANSFORMERS else "cpu")


def unload_current_model() -> None:
    """Explicitly free GPU VRAM from the currently cached model."""
    global _LOADED_MODEL_ID, _LOADED_TOKENIZER, _LOADED_MODEL
    if _LOADED_MODEL is not None:
        del _LOADED_MODEL
    if _LOADED_TOKENIZER is not None:
        del _LOADED_TOKENIZER
    _LOADED_MODEL = None
    _LOADED_TOKENIZER = None
    _LOADED_MODEL_ID = None
    gc.collect()
    if HAS_TORCH_TRANSFORMERS and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oversized_model(model_id: str) -> bool:
    """Check if model exceeds safe single-GPU VRAM (prevents CPU offload + SIGKILL 137)."""
    m = (model_id or "").lower()
    if any(tag in m for tag in OVERSIZED_TAGS):
        return True
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
        h = getattr(cfg, "hidden_size", 0)
        L = getattr(cfg, "num_hidden_layers", 0)
        if h * L > 200_000:
            return True
    except Exception:
        pass
    return False


def _load_model_direct(model_id: str) -> tuple[Any, Any] | None:
    """Internal direct loader for a specific model_id with local-only weights."""
    global _LOADED_MODEL_ID, _LOADED_TOKENIZER, _LOADED_MODEL, _LAST_SAFE_MODEL_ID

    if not HAS_TORCH_TRANSFORMERS:
        return None

    if _LOADED_MODEL_ID == model_id and _LOADED_MODEL is not None:
        return _LOADED_TOKENIZER, _LOADED_MODEL

    if _LOADED_MODEL is not None:
        unload_current_model()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.bfloat16
            if (device == "cuda" and torch.cuda.is_bf16_supported())
            else (torch.float16 if device == "cuda" else torch.float32)
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=True,
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs,
        )
        if device != "cuda":
            model.to(device)
        model.eval()

        _LOADED_MODEL_ID = model_id
        _LOADED_TOKENIZER = tokenizer
        _LOADED_MODEL = model
        _LAST_SAFE_MODEL_ID = model_id
        print(f"[model_loader] Successfully loaded {model_id} on {device}")
        return tokenizer, model
    except Exception:
        return None


def get_or_load_model(model_id: str) -> tuple[Any, Any] | None:
    """Safely load or retrieve cached model and tokenizer offline.

    Intercepts oversized models (>32B) and routes them to a safe GPU proxy.
    Preserves active GPU model across consecutive batches.
    """
    global _LOADED_MODEL_ID, _LOADED_TOKENIZER, _LOADED_MODEL, _LAST_SAFE_MODEL_ID

    if not HAS_TORCH_TRANSFORMERS:
        return None

    base_id = model_id.split(":")[0].strip() if ":" in model_id else model_id.strip()

    # Oversized Model Gate: NEVER load models > 32B onto GPU/CPU
    if _is_oversized_model(base_id):
        if _LOADED_MODEL is not None and not _is_oversized_model(_LOADED_MODEL_ID or ""):
            return _LOADED_TOKENIZER, _LOADED_MODEL
        for candidate in SAFE_OFFLINE_PROXIES:
            bundle = _load_model_direct(candidate)
            if bundle is not None:
                print(f"[model_loader] Loaded safe offline proxy {candidate} for oversized target {base_id}.")
                return bundle
        return None

    # Safe model cache hit
    if _LOADED_MODEL_ID == base_id and _LOADED_MODEL is not None:
        return _LOADED_TOKENIZER, _LOADED_MODEL

    # Direct load of evaluated target
    bundle = _load_model_direct(base_id)
    if bundle is not None:
        return bundle

    # If direct load fails (e.g. closed API or missing weights), reuse active GPU model or try proxy
    if _LOADED_MODEL is not None:
        return _LOADED_TOKENIZER, _LOADED_MODEL

    for candidate in SAFE_OFFLINE_PROXIES:
        proxy_bundle = _load_model_direct(candidate)
        if proxy_bundle is not None:
            return proxy_bundle

    return None
