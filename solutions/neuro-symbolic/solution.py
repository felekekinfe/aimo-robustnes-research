"""Neuro-Symbolic Hybrid Solution for the AIMO Interpretability Challenge.

Fuses internal neural representation margins with 27 structural & symbolic
features extracted from mathematical problem texts, with graceful fallback.
"""

from __future__ import annotations

from neuro_symbolic.fusion import predict_neuro_symbolic


def are_robust(
    model_id: str,
    reasoning_effort: str,
    problems: list[str],
) -> list[bool]:
    """Predict whether model reasoning on each problem is robust.

    Args:
        model_id: Codabench model alias or exact checkpoint name.
        reasoning_effort: Reasoning effort metadata ("low", "medium", "default").
        problems: Ordered list of mathematical problem strings.

    Returns:
        List of native Python bools aligned with ``problems``.
    """
    return predict_neuro_symbolic(
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        problems=problems,
    )
