"""Contrastive White-Box Robustness Solution for the AIMO Interpretability Challenge.

Directly tests latent representation invariance under meaning-preserving
perturbations, fusing cosine similarity, representation drift, probe margins,
and structural problem priors with safe fallback.

NO-CRASH PROTOCOL:
- Zero pickle/joblib loads (avoids sklearn version mismatch)
- Offline-only model loading (local_files_only=True, device_map="auto")
- Explicit VRAM cleanup between model_id groups
- Global try/except with traceback logging
- Every prediction is independent per-problem (no batch distribution hacking)
"""

from __future__ import annotations

import gc
import traceback
from typing import Any


def are_robust(
    model_id: str,
    reasoning_effort: str = "default",
    problems: list[str] | None = None,
    **kwargs: Any,
) -> list[bool]:
    """Organizer-compliant entrypoint.

    Supports both signatures:
      - are_robust(model_id, reasoning_effort, problems)
      - are_robust(model_id, problems)  [legacy 2-arg form]

    Every problem is scored independently. Model loading is batched for
    VRAM efficiency but the prediction for problem A never depends on
    the predicted distribution of the rest of the batch.

    Returns:
        List of native Python ``bool`` values aligned with ``problems``.
    """
    # --- Signature flexibility: handle 2-arg legacy calls ---
    if problems is None and isinstance(reasoning_effort, list):
        problems = reasoning_effort
        reasoning_effort = "default"

    if problems is None:
        problems = []
    if not problems:
        return []

    model_str = str(model_id or "")
    effort_str = str(reasoning_effort or "default")

    # === PRIMARY EXECUTION PATH ===
    try:
        from contrastive.decision import predict_contrastive_robustness

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
        traceback.print_exc()

    # === FALLBACK 1: Symbolic Decision Engine (zero imports beyond stdlib+re) ===
    try:
        from contrastive.decision import _evaluate_symbolic_decision, _problem_features

        fallback = [
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
        return fallback
    except Exception:
        traceback.print_exc()

    # === FALLBACK 2: Ultimate fail-safe — always valid output ===
    print(f"FATAL: all fallbacks failed for model_id={model_str}, returning [False]*{len(problems)}")
    return [False for _ in problems]
