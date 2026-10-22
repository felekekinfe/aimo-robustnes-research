"""Contrastive White-Box Robustness Solution for the AIMO Interpretability Challenge.

Directly tests latent representation invariance under meaning-preserving
perturbations, fusing cosine similarity, representation drift, probe margins,
and structural problem priors with safe fallback.
"""

from __future__ import annotations

from contrastive.decision import predict_contrastive_robustness


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
    return predict_contrastive_robustness(
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        problems=problems,
    )
