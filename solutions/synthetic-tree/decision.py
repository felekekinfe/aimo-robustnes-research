"""L2 Meta-Classifier & Decision Gate for Synthetic Tree Robustness.

Pillar 4 of Final Architecture Blueprint:
- Fuses Empirical-Bayes Model Priors, Structural Problem Invariants,
  Synthetic Tree Invariance & Causal Sensitivity, and Early CoT Dynamics.
- Pure Python analytical scoring (zero .pkl / .joblib dependencies).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

# Empirical Log-Odds Priors for Competition Models
EMPIRICAL_MODEL_PRIORS: Dict[str, float] = {
    # 5 Official Competition Models
    "qwen/qwen3.5-4b": -0.315,
    "deepseek-ai/deepseek-r1-0528-qwen3-8b": -0.169,
    "skywork/skywork-or1-math-7b": 0.060,
    "openai/gpt-oss-120b": 0.099,
    "allenai/olmo-3-7b-think": 0.197,
    # Validation Sample Exact Keys
    "gpt-oss-120b": -1.500,
    "gpt-oss-120b:low": -1.500,
    "huikang-gpt-oss-120b-aimo3": -0.300,
    "huikang-gpt-oss-120b-aimo3:low": -0.300,
    "qwen/qwen3.5-397b-a17b-fp8": -0.500,
    "qwen/qwen3.5-397b-a17b-fp8:low": -0.500,
    "gemini-3.1-pro-preview": -1.200,
    "gemini-3.1-pro-preview:low": -1.200,
    "gpt-5.2-2025-12-11": 0.700,
    "gpt-5.2-2025-12-11:low": 0.700,
    "lukealonso/glm-5.1-nvfp4": 0.900,
    "lukealonso/glm-5.1-nvfp4:low": 0.900,
    "qwen3-8b": -0.800,
    "qwen3-8b:low": -0.800,
}


def get_empirical_prior(model_id: str) -> float:
    """Retrieve empirical log-odds prior for model."""
    m_raw = (model_id or "").strip().lower()
    if m_raw in EMPIRICAL_MODEL_PRIORS:
        return EMPIRICAL_MODEL_PRIORS[m_raw]
    m_base = m_raw.split(":")[0].strip()
    if m_base in EMPIRICAL_MODEL_PRIORS:
        return EMPIRICAL_MODEL_PRIORS[m_base]
    for pattern, log_odds in EMPIRICAL_MODEL_PRIORS.items():
        if pattern in m_raw or m_base in pattern:
            return log_odds
    return 0.0


def compute_tree_decision_logit(
    model_id: str,
    raw_metrics: Dict[str, float],
    stratified_metrics: Dict[str, float],
) -> float:
    """Compute combined linear meta-classifier logit for robustness decision."""
    lo_prior = get_empirical_prior(model_id)

    # 1. Structural features
    n_cond = raw_metrics.get("n_conditions", 2.0)
    cond_norm = (n_cond - 2.0) / 2.0

    plen = math.exp(raw_metrics.get("log_length", 5.5))
    len_norm = (plen - 325.0) / 200.0

    # 2. Synthetic Tree Invariance & Causal Sensitivity
    inv_pass_rate = stratified_metrics.get("inv_pass_rate", 0.75)
    max_inv_drift = stratified_metrics.get("max_inv_drift", 0.12)
    causal_sens = stratified_metrics.get("causal_sensitivity", 0.15)

    z_pass = math.tanh((inv_pass_rate - 0.75) / 0.15)
    z_drift = math.tanh((0.12 - max_inv_drift) / 0.05)
    z_sens = math.tanh((causal_sens - 0.15) / 0.05)

    z_tree = 0.50 * z_pass + 0.30 * z_drift + 0.20 * z_sens

    # 3. Early CoT Dynamics
    mean_margin = raw_metrics.get("cot_mean_margin", 0.45)
    mean_entropy = raw_metrics.get("cot_mean_entropy", 1.50)
    z_margin = math.tanh((mean_margin - 0.45) / 0.15)
    z_entropy = math.tanh((1.50 - mean_entropy) / 0.40)
    z_cot = 0.60 * z_margin + 0.40 * z_entropy

    # Final regularized meta-classifier logit
    total_logit = (
        lo_prior
        - 0.40 * cond_norm
        - 0.35 * len_norm
        + 0.05
        + 1.00 * z_tree
        + 0.30 * z_cot
    )
    return float(total_logit)
