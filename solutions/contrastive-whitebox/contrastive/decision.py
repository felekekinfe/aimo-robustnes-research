"""Contrastive decision logic fusing latent invariance, probe margin, and symbolic priors."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np

from contrastive.contrastive_detector import extract_contrastive_batch
from contrastive.perturbation_engine import generate_contrastive_variant
from contrastive.symbolic_extractor import extract_feature_dataframe

MODEL_ALIASES: dict[str, str] = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

SYMBOLIC_ARTIFACT_NAME = "symbolic_pipeline.joblib"
FALLBACK_THRESHOLD = 0.45

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


def _sigmoid(x: float) -> float:
    clipped = max(-15.0, min(15.0, x))
    return 1.0 / (1.0 + math.exp(-clipped))


def predict_contrastive_robustness(
    model_id: str,
    reasoning_effort: str,
    problems: list[str],
) -> list[bool]:
    """Predict robustness using contrastive latent invariance with symbolic fallback.

    Args:
        model_id: Codabench model alias or exact checkpoint name.
        reasoning_effort: Reasoning effort metadata ("low", "medium", "default").
        problems: Ordered list of mathematical problem strings.

    Returns:
        List of native Python bools aligned with ``problems``.
    """
    if not problems:
        return []

    resolved_model_id = MODEL_ALIASES.get(model_id, model_id)

    # 1. Compute Symbolic Probabilities as Baseline Priors
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
            p_symbolic = np.asarray(symbolic_model.predict(features_df), dtype=float)
        except Exception:
            p_symbolic = np.full(len(problems), 0.5)
    else:
        p_symbolic = np.full(len(problems), 0.5)

    # 2. Generate Meaning-Preserving Contrastive Variants
    variants = [generate_contrastive_variant(p) for p in problems]

    # 3. Extract Contrastive White-Box Metrics (if GPU and weights exist)
    contrastive_metrics = extract_contrastive_batch(
        model_id=resolved_model_id,
        original_problems=problems,
        variant_problems=variants,
    )

    decisions: list[bool] = []

    if contrastive_metrics is not None and len(contrastive_metrics) == len(problems):
        # White-Box Contrastive Mode Active
        for metric, p_sym in zip(contrastive_metrics, p_symbolic):
            # Margin probability
            p_margin = _sigmoid(metric.margin_orig)
            # Latent stability probability (penalizes drift > 0.05)
            p_stability = max(0.0, min(1.0, 1.0 - (metric.latent_drift / 0.15)))

            # Agreement penalty: if original and variant disagree, penalize
            agreement_bonus = 0.10 if metric.margin_agreement else -0.15

            # Fused score
            fused_score = (
                0.45 * p_margin
                + 0.35 * p_stability
                + 0.20 * float(p_sym)
                + agreement_bonus
            )
            decisions.append(bool(fused_score >= 0.50))
    else:
        # Graceful fallback to calibrated symbolic model (CPU / missing weights)
        decisions = [bool(p >= FALLBACK_THRESHOLD) for p in p_symbolic]

    return [bool(d) for d in decisions]
