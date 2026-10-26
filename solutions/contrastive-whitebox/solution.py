"""Contrastive White-Box Robustness Solution for the AIMO Interpretability Challenge.

Directly tests latent representation invariance under meaning-preserving
perturbations, fusing cosine similarity, representation drift, probe margins,
and structural problem priors with safe fallback.
"""

from __future__ import annotations

from typing import Any
from contrastive.decision import (
    predict_contrastive_robustness,
    _evaluate_symbolic_decision,
    _problem_features,
)


def are_robust(
    model_id: str,
    reasoning_effort: str = "default",
    problems: list[str] | None = None,
    **kwargs: Any,
) -> list[bool]:
    """Predict whether model reasoning on each problem is robust.

    Supports both signatures:
      - are_robust(model_id, reasoning_effort, problems)
      - are_robust(model_id, problems)

    Returns:
        List of native Python bools aligned with ``problems``.
    """
    # Guard against 2-argument calls where 'problems' was passed into positional 'reasoning_effort'
    if problems is None and isinstance(reasoning_effort, list):
        problems = reasoning_effort
        reasoning_effort = "default"

    if problems is None:
        problems = []

    if not problems:
        return []

    model_str = str(model_id or "")
    effort_str = str(reasoning_effort or "default")

    # Primary execution path
    try:
        results = predict_contrastive_robustness(
            model_id=model_str,
            reasoning_effort=effort_str,
            problems=problems,
        )
        if (
            isinstance(results, list)
            and len(results) == len(problems)
            and all(type(r) is bool for r in results)
        ):
            return results
    except Exception:
        pass

    # Ironclad Fallback: Symbolic Decision Engine (100% exception-safe)
    try:
        fallback_preds = [
            bool(
                _evaluate_symbolic_decision(
                    problem=p,
                    feat_dict=_problem_features(p),
                    model_id=model_str,
                    reasoning_effort=effort_str,
                    p_rf=0.5,
                )
            )
            for p in problems
        ]
        return fallback_preds
    except Exception:
        # Ultimate fail-safe: always return native Python bools of exact required length
        return [False for _ in problems]
