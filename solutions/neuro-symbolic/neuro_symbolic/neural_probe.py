"""Neural hidden-state extraction and linear-probe scoring with safe fallbacks."""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False

RECOMMENDED_STRATEGY = "best_layer_mean_margin"

# Module-level model cache to avoid re-loading weights for each batch problem
_LOADED_MODEL_ID: str | None = None
_LOADED_TOKENIZER: Any = None
_LOADED_MODEL: Any = None


@dataclass(frozen=True)
class ProbeArtifact:
    model_id: str
    system_prompt: str
    data: dict[str, Any]
    kind: str


def _safe_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def _artifact_candidates(model_id: str) -> list[Path]:
    root_dir = Path(__file__).resolve().parents[1]
    safe_id = _safe_model_id(model_id)
    candidates = []
    if safe_id:
        candidates.extend([
            root_dir / "probe_artifacts" / f"{safe_id}.pkl",
            root_dir / "probe_artifacts" / f"{safe_id}.npz",
        ])
    candidates.extend([
        root_dir / "probe_artifact.pkl",
        root_dir / "probe_artifacts" / "probe_artifact.pkl",
        root_dir / "probe_artifact.npz",
    ])
    return candidates


def load_probe_artifact(model_id: str) -> ProbeArtifact | None:
    path = next((c for c in _artifact_candidates(model_id) if c.is_file()), None)
    if path is None:
        return None
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict) or data.get("schema_version") != 2:
            return None
        return ProbeArtifact(
            model_id=str(data.get("model_id") or ""),
            system_prompt=str(data.get("system_prompt") or ""),
            data=data,
            kind="pickle",
        )
    except Exception:
        return None


def _layer_index(artifact: ProbeArtifact) -> int:
    strategy = artifact.data.get("recommended_strategy") or {}
    return int(strategy.get("layer_index", artifact.data["best_layer_index"]))


def get_or_load_model(model_id: str) -> tuple[Any, Any] | None:
    """Safely load or retrieve the cached model and tokenizer offline."""
    global _LOADED_MODEL_ID, _LOADED_TOKENIZER, _LOADED_MODEL

    if not HAS_TORCH_TRANSFORMERS:
        return None

    if _LOADED_MODEL_ID == model_id and _LOADED_MODEL is not None:
        return _LOADED_TOKENIZER, _LOADED_MODEL

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device == "cuda" else torch.float32
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            local_files_only=True,
        )
        model.to(device)
        model.eval()

        _LOADED_MODEL_ID = model_id
        _LOADED_TOKENIZER = tokenizer
        _LOADED_MODEL = model
        return tokenizer, model
    except Exception:
        # Model is not in local cache, corrupted, or out of memory
        return None


def _build_prompt(tokenizer: Any, problem_text: str, system_prompt: str) -> str:
    if system_prompt and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{system_prompt}\n\n{problem_text}" if system_prompt else problem_text


def _encode_problem(tokenizer: Any, model: Any, problem_text: str, artifact: ProbeArtifact) -> np.ndarray:
    prompt = _build_prompt(tokenizer, problem_text, artifact.system_prompt)
    model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    model_inputs = {name: val.to(model.device) for name, val in model_inputs.items()}

    with torch.no_grad():
        output = model(**model_inputs, output_hidden_states=True, return_dict=True, use_cache=False)

    hidden_states = output.hidden_states
    layer = _layer_index(artifact)
    return hidden_states[layer][0, -1, :].detach().float().cpu().numpy()


def compute_ensemble_margin(vector: np.ndarray, artifact: ProbeArtifact) -> float:
    layer = _layer_index(artifact)
    margins = []
    for group in artifact.data.get("groups", []):
        probes = group.get("probes", {})
        probe = probes.get(layer, probes.get(str(layer)))
        if probe is None:
            continue
        weights = np.asarray(probe["weights"], dtype=np.float32)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        bias = np.asarray(probe["bias"], dtype=np.float32).reshape(-1)
        threshold = np.asarray(probe["threshold"], dtype=np.float32).reshape(-1)

        margins.extend((weights @ vector + bias - threshold).astype(float).tolist())

    if not margins:
        return 0.0
    return float(np.mean(margins))


def score_neural_probe_batch(
    model_id: str,
    problems: list[str],
) -> list[float] | None:
    """Score problems using the neural probe.

    Returns:
        List of continuous margins in problem order, or None if the neural branch
        is unavailable (missing weights, artifact, or error).
    """
    artifact = load_probe_artifact(model_id)
    if artifact is None:
        return None

    target_model_id = artifact.model_id or model_id
    model_bundle = get_or_load_model(target_model_id)
    if model_bundle is None:
        return None

    tokenizer, model = model_bundle
    try:
        margins = []
        for problem in problems:
            vector = _encode_problem(tokenizer, model, problem, artifact)
            margin = compute_ensemble_margin(vector, artifact)
            margins.append(margin)
        return margins
    except Exception:
        return None
