"""Neuro-Symbolic decision fusion and safe fallback mechanism."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np

from neuro_symbolic.config import (
    DECISION_THRESHOLD,
    FALLBACK_DECISION_THRESHOLD,
    FUSION_ALPHA,
    MODEL_ALIASES,
    SYMBOLIC_ARTIFACT_NAME,
)
from neuro_symbolic.neural_probe import score_neural_probe_batch
from neuro_symbolic.symbolic_extractor import extract_feature_dataframe

_LOADED_SYMBOLIC_MODEL = None


def _load_symbolic_model():
    global _LOADED_SYMBOLIC_MODEL
    if _LOADED_SYMBOLIC_MODEL is not None:
        return _LOADED_SYMBOLIC_MODEL

    root_dir = Path(__file__).resolve().parents[1]
    candidate_paths = [
        root_dir / "artifacts" / SYMBOLIC_ARTIFACT_NAME,
        root_dir / SYMBOLIC_ARTIFACT_NAME,
    ]
    path = next((p for p in candidate_paths if p.is_file()), None)
    if path is not None:
        try:
            payload = joblib.load(path)
            _LOADED_SYMBOLIC_MODEL = payload.get("model", payload)
            return _LOADED_SYMBOLIC_MODEL
        except Exception:
            pass
    return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def predict_neuro_symbolic(
    model_id: str,
    reasoning_effort: str,
    problems: list[str],
) -> list[bool]:
    """Execute neuro-symbolic robustness inference with safe fallback.

    Args:
        model_id: Target model identifier or Codabench alias.
        reasoning_effort: Evaluated reasoning effort ("low", "medium", "default").
        problems: List of mathematical problem statement strings.

    Returns:
        List of native Python bools in exact input problem order.
    """
    if not problems:
        return []

    resolved_model_id = MODEL_ALIASES.get(model_id, model_id)

    # 1. Compute Symbolic Probabilities
    symbolic_model = _load_symbolic_model()
    features_df = extract_feature_dataframe(problems)

    if symbolic_model is not None and hasattr(symbolic_model, "predict_proba"):
        try:
            probs = symbolic_model.predict_proba(features_df)
            p_symbolic = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
        except Exception:
            p_symbolic = np.full(len(problems), 0.5)
    elif symbolic_model is not None and hasattr(symbolic_model, "predict"):
        try:
            preds = symbolic_model.predict(features_df)
            p_symbolic = np.asarray(preds, dtype=float)
        except Exception:
            p_symbolic = np.full(len(problems), 0.5)
    else:
        p_symbolic = np.full(len(problems), 0.5)

    # 2. Compute Neural Probe Margins (if weights and artifact are accessible)
    neural_margins = score_neural_probe_batch(resolved_model_id, problems)

    # 3. Decision Fusion & Fallback
    if neural_margins is not None and len(neural_margins) == len(problems):
        p_neural = _sigmoid(np.asarray(neural_margins, dtype=float))
        p_fused = (FUSION_ALPHA * p_neural) + ((1.0 - FUSION_ALPHA) * p_symbolic)
        decisions = [bool(p >= DECISION_THRESHOLD) for p in p_fused]
    else:
        # Graceful fallback: neural branch not available (offline CPU / missing weights)
        decisions = [bool(p >= FALLBACK_DECISION_THRESHOLD) for p in p_symbolic]

    # Guarantee strict native bool output
    return [bool(d) for d in decisions]
