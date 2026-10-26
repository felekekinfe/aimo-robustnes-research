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
    model_id: Any = None,
    reasoning_effort: Any = "default",
    problems: list[Any] | None = None,
    **kwargs: Any,
) -> list[bool]:
    """Organizer-compliant entrypoint supporting all Codabench calling conventions:

    1. are_robust(model_id: str, reasoning_effort: str, problems: list[str])  [Official]
    2. are_robust(problems: list[dict])                                       [Batch-dict]
    3. are_robust(model_id: str, problems: list[str])                         [Legacy 2-arg]

    Returns:
        List of native Python bools aligned with input problems.
    """
    # Convention 2: are_robust(problems: list[dict])
    if isinstance(model_id, list) and len(model_id) > 0 and isinstance(model_id[0], dict):
        items = model_id
        predictions = [False] * len(items)
        groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for idx, it in enumerate(items):
            m = str(it.get("model_id", ""))
            e = str(it.get("reasoning_effort", "default"))
            p = str(it.get("problem", ""))
            groups.setdefault((m, e), []).append((idx, p))

        for (m_id, eff), grp in groups.items():
            prob_list = [p for _, p in grp]
            preds = are_robust(model_id=m_id, reasoning_effort=eff, problems=prob_list)
            for (orig_idx, _), pr in zip(grp, preds):
                predictions[orig_idx] = bool(pr)
        return predictions

    # Convention 3: legacy 2-arg are_robust(model_id, problems)
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
