"""Hybrid Ensemble Solution for AIMO Interpretability Challenge.

Combines the strongest signals from multiple approaches:
1. SCSP Contrastive Drift (CDI_latent, CDI_digits, Spectral Rank)
2. Synthetic Tree Expert Sensitivity (ablate/mutate causal boundaries)
3. Streamlined-Hybrid Early CoT Dynamics (entropy, margin, volatility)
4. Structural Complexity Priors (model-conditioned, effort-aware)

Zero pickle dependencies - pure analytical ensemble with pre-computed weights.
White-box signals throughout for White-Box Award eligibility.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

from extractor import HybridFeatureExtractor
from decision import compute_hybrid_decision_logit, get_model_prior_log_odds

logger = logging.getLogger("AIMO-Hybrid-Ensemble")

# Track configuration
OVERSIZED_OR_CLOSED_TAGS = (
    "gemini", "gpt-5", "claude", "397b", "405b", "671b", "frontier"
)

PER_PROBLEM_TIMEOUT = 10.0  # seconds (aggressive for 3600s total budget)

# Global persistent cache across batches
_CACHED_MODEL_ID: Optional[str] = None
_CACHED_EXTRACTOR: Optional[HybridFeatureExtractor] = None


def _is_unsupported_model(model_id: str) -> bool:
    """Identify models that cannot or should not be loaded on single GPU."""
    m_clean = (model_id or "").split(":")[0].strip().lower()
    if "gpt-oss-120b" in m_clean or "120b" in m_clean:
        return False  # This one IS supported on 96GB
    return any(tag in m_clean for tag in OVERSIZED_OR_CLOSED_TAGS)


def _get_or_load_extractor(model_id: str) -> HybridFeatureExtractor:
    """Retrieve cached extractor or initialize safely with fallback."""
    global _CACHED_MODEL_ID, _CACHED_EXTRACTOR

    base_id = (model_id or "").split(":")[0].strip()
    if _CACHED_MODEL_ID == base_id and _CACHED_EXTRACTOR is not None:
        return _CACHED_EXTRACTOR

    if _is_unsupported_model(base_id):
        _CACHED_MODEL_ID = base_id
        _CACHED_EXTRACTOR = HybridFeatureExtractor(model=None, tokenizer=None, model_id=base_id)
        return _CACHED_EXTRACTOR

    try:
        extractor = HybridFeatureExtractor.from_pretrained(base_id, device="cuda")
        _CACHED_MODEL_ID = base_id
        _CACHED_EXTRACTOR = extractor
        return extractor
    except Exception as e:
        logger.warning("Could not load weights for %s (%s). Using structural fallback.", base_id, e)
        extractor = HybridFeatureExtractor(model=None, tokenizer=None, model_id=base_id)
        _CACHED_MODEL_ID = base_id
        _CACHED_EXTRACTOR = extractor
        return extractor


def evaluate_single_problem(
    problem: str,
    model_id: str,
    reasoning_effort: str,
    extractor: HybridFeatureExtractor,
) -> bool:
    """Score an individual problem independently via hybrid ensemble."""
    t_start = time.time()
    try:
        # Extract all feature groups
        features = extractor.extract_all_features(problem, model_id, reasoning_effort)

        # Compute hybrid decision logit
        logit = compute_hybrid_decision_logit(
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            features=features,
        )

        # Hard timeout check
        if (time.time() - t_start) > PER_PROBLEM_TIMEOUT:
            logger.warning("Timeout on problem, using prior fallback")
            prior = get_model_prior_log_odds(model_id, reasoning_effort)
            return bool(prior >= 0.0)

        return bool(logit >= 0.0)
    except Exception as e:
        logger.warning("Error evaluating problem: %s. Using prior fallback.", e)
        prior = get_model_prior_log_odds(model_id, reasoning_effort)
        return bool(prior >= 0.0)


def are_robust(
    model_id: str,
    reasoning_effort: str = "default",
    problems: list[str] | None = None,
    **kwargs: Any,
) -> list[bool]:
    """Codabench official entrypoint.

    Accepts both 3-arg and legacy 2-arg calls:
      - are_robust(model_id, reasoning_effort, problems)
      - are_robust(model_id, problems)
    """
    # Handle legacy 2-arg signature
    if problems is None and isinstance(reasoning_effort, list):
        problems = reasoning_effort
        reasoning_effort = "default"

    if problems is None:
        problems = []
    if not problems:
        return []

    effort = str(reasoning_effort or "default")
    extractor = _get_or_load_extractor(model_id)

    results: List[bool] = []
    for p in problems:
        pred = evaluate_single_problem(p, model_id, effort, extractor)
        results.append(bool(pred))

    # Telemetry
    n_total = len(problems)
    n_true = sum(results)
    pct = (n_true / n_total * 100.0) if n_total > 0 else 0.0
    logger.info(
        f"[hybrid_ensemble] model={model_id} effort={effort} n={n_total} "
        f"robust={n_true}/{n_total} ({pct:.1f}%)"
    )
    return results