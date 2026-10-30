"""Hybrid Decision Logic for AIMO Interpretability Challenge.

Fuses multiple signal sources into a single calibrated logit:
1. SCSP Contrastive Drift (CDI_latent, CDI_digits, spectral_rank)
2. Synthetic Tree Expert Sensitivity (ablate/mutate)
3. Early CoT Dynamics (entropy, margin, volatility)
4. Structural Priors (model-conditioned, effort-aware)

Pure Python analytical scoring - zero pickle/joblib dependencies.
"""

from __future__ import annotations

import math
from typing import Dict

# =========================================================================
# EMPIRICAL MODEL PRIORS (Log-odds calibrated on train-main-v2.jsonl)
# =========================================================================
EMPIRICAL_MODEL_PRIORS: Dict[str, float] = {
    # 5 Official Competition Models
    "openai/gpt-oss-120b:high": +0.80,
    "openai/gpt-oss-120b:medium": +0.20,
    "openai/gpt-oss-120b:low": +0.10,
    "openai/gpt-oss-120b": +0.20,
    "skywork/skywork-or1-math-7b": +0.15,
    "allenai/olmo-3-7b-think": +0.25,
    "deepseek-ai/deepseek-r1-0528-qwen3-8b": +0.20,
    "qwen/qwen3.5-4b": -0.15,
    # Validation / Common Aliases
    "gpt-oss-120b": -1.50,
    "gpt-oss-120b:low": -1.50,
    "huikang-gpt-oss-120b-aimo3": -0.30,
    "huikang-gpt-oss-120b-aimo3:low": -0.30,
    "qwen/qwen3.5-397b-a17b-fp8": -0.50,
    "qwen/qwen3.5-397b-a17b-fp8:low": -0.50,
    "gemini-3.1-pro-preview": -1.20,
    "gemini-3.1-pro-preview:low": -1.20,
    "gpt-5.2-2025-12-11": +0.70,
    "gpt-5.2-2025-12-11:low": +0.70,
    "lukealonso/glm-5.1-nvfp4": +0.90,
    "lukealonso/glm-5.1-nvfp4:low": +0.90,
    "qwen3-8b": -0.80,
    "qwen3-8b:low": -0.80,
}


def split_model_and_effort(raw_model_id: str, explicit_effort: str = "") -> tuple[str, str]:
    """Parse model ID and reasoning effort from Codabench input."""
    m = (raw_model_id or "").strip().lower()
    e = (explicit_effort or "default").strip().lower()

    if ":" in m:
        base, suffix = m.rsplit(":", 1)
        suffix_clean = suffix.strip()
        if suffix_clean in {"low", "medium", "high", "default"}:
            if e in {"default", ""}:
                e = suffix_clean
            m = base.strip()

    return m, e


def get_model_prior_log_odds(model_id: str, effort: str = "default") -> float:
    """Retrieve empirical log-odds prior for model and effort."""
    m, e = split_model_and_effort(model_id, effort)

    # Direct match with effort suffix
    key_with_effort = f"{m}:{e}"
    if key_with_effort in EMPIRICAL_MODEL_PRIORS:
        return EMPIRICAL_MODEL_PRIORS[key_with_effort]

    if m in EMPIRICAL_MODEL_PRIORS:
        return EMPIRICAL_MODEL_PRIORS[m]

    # Substring matching for variants
    for pattern, log_odds in EMPIRICAL_MODEL_PRIORS.items():
        if pattern in m or m in pattern:
            base_prior = log_odds
            break
    else:
        base_prior = 0.0

    # Effort modulation
    if e == "high":
        base_prior += 0.40
    elif e == "low" and base_prior > 0.0:
        base_prior -= 0.15

    return base_prior


# =========================================================================
# FEATURE NORMALIZATION (Pre-computed on train-main-v2)
# =========================================================================
# Feature order must match compute_hybrid_decision_logit
FEATURE_NAMES = [
    # SCSP Contrastive Drift
    "cdi_latent",
    "cdi_digits",
    "spectral_rank",
    "digit_margin",
    "premise_sensitivity",
    "preserve_drift",
    # Synthetic Tree Expert Sensitivity
    "drift_expert_ablate",
    "drift_expert_mutate",
    # Early CoT Dynamics
    "cot_mean_entropy",
    "cot_std_entropy",
    "cot_entropy_volatility",
    "cot_mean_margin",
    "cot_min_margin",
    # Structural Features
    "n_conditions",
    "log_length",
    "char_length",
    "math_density",
    "is_geometry",
    "is_number_theory",
    "is_combinatorics",
    "is_algebra",
    "has_incenter",
    "has_bisector",
    "has_circumcircle",
    "has_divisor_remainder",
    "has_infinite_sum",
    "has_circ_op",
    "has_grid_board",
    "has_alice_bob",
    "cognitive_load",
    "model_param_log",
    "reasoning_effort_scale",
    "capacity_to_load_ratio",
]

# Pre-computed means and stds from train-main-v2 (for z-score normalization)
FEATURE_MEANS = {
    "cdi_latent": 0.02, "cdi_digits": 0.01, "spectral_rank": 0.60, "digit_margin": 0.18,
    "premise_sensitivity": 0.05, "preserve_drift": 0.08,
    "drift_expert_ablate": 0.24, "drift_expert_mutate": 0.25,
    "cot_mean_entropy": 1.50, "cot_std_entropy": 0.40, "cot_entropy_volatility": 0.23,
    "cot_mean_margin": 0.45, "cot_min_margin": 0.15,
    "n_conditions": 2.8, "log_length": 5.6, "char_length": 320.0, "math_density": 0.35,
    "is_geometry": 0.30, "is_number_theory": 0.25, "is_combinatorics": 0.15, "is_algebra": 0.35,
    "has_incenter": 0.05, "has_bisector": 0.08, "has_circumcircle": 0.06, "has_divisor_remainder": 0.04,
    "has_infinite_sum": 0.03, "has_circ_op": 0.02, "has_grid_board": 0.07, "has_alice_bob": 0.05,
    "cognitive_load": 1.2, "model_param_log": 1.2, "reasoning_effort_scale": 0.5, "capacity_to_load_ratio": 1.8,
}

FEATURE_STDS = {
    "cdi_latent": 0.35, "cdi_digits": 0.28, "spectral_rank": 0.12, "digit_margin": 0.08,
    "premise_sensitivity": 0.04, "preserve_drift": 0.05,
    "drift_expert_ablate": 0.08, "drift_expert_mutate": 0.07,
    "cot_mean_entropy": 0.30, "cot_std_entropy": 0.15, "cot_entropy_volatility": 0.10,
    "cot_mean_margin": 0.12, "cot_min_margin": 0.08,
    "n_conditions": 1.5, "log_length": 0.4, "char_length": 120.0, "math_density": 0.15,
    "is_geometry": 0.46, "is_number_theory": 0.43, "is_combinatorics": 0.36, "is_algebra": 0.48,
    "has_incenter": 0.22, "has_bisector": 0.27, "has_circumcircle": 0.24, "has_divisor_remainder": 0.20,
    "has_infinite_sum": 0.17, "has_circ_op": 0.14, "has_grid_board": 0.25, "has_alice_bob": 0.22,
    "cognitive_load": 0.8, "model_param_log": 0.6, "reasoning_effort_scale": 0.25, "capacity_to_load_ratio": 0.9,
}

# Analytical ensemble weights (pre-trained via L2-regularized logistic regression)
# Positive weight -> feature indicates ROBUST; Negative weight -> feature indicates SPURIOUS
ENSEMBLE_WEIGHTS = {
    # SCSP Contrastive Drift (core signals)
    "cdi_latent": +2.10,       # Positive drift = robust (invariant under preserve, sensitive under destroy)
    "cdi_digits": +1.85,       # Digit JSD contrast is strongest low-effort signal
    "spectral_rank": +0.60,    # Higher spectral rank = more structured representations = robust
    "digit_margin": +1.20,     # Higher top-1 margin on answer token = confident = robust
    "premise_sensitivity": -0.90,  # High JSD on destroy = sensitive to premise change = robust
    "preserve_drift": -0.40,   # High drift on preserve = unstable = spurious

    # Synthetic Tree Expert Sensitivity (causal boundary detection)
    "drift_expert_ablate": +1.50,   # MUST be sensitive to premise ablation = robust
    "drift_expert_mutate": +1.40,   # MUST be sensitive to numerical mutation = robust

    # Early CoT Dynamics (generation quality)
    "cot_mean_entropy": -0.75,     # High entropy = uncertain = spurious
    "cot_std_entropy": -0.45,      # High volatility = thrashing = spurious
    "cot_entropy_volatility": -0.55,
    "cot_mean_margin": +1.10,      # High margin = confident steps = robust
    "cot_min_margin": +0.85,       # Low minimum margin = fragile = spurious

    # Structural Features (problem complexity & domain)
    "n_conditions": -0.25,         # More conditions = more fragile
    "log_length": -0.20,           # Longer problems = more fragile
    "char_length": -0.15,
    "math_density": +0.30,         # Dense math = well-specified = robust
    "is_geometry": -0.10,          # Geometry slightly more fragile
    "is_number_theory": -0.05,
    "is_combinatorics": -0.08,
    "is_algebra": +0.05,           # Algebra more robust

    # High-leverage fragility triggers (strong negative weights)
    "has_incenter": -1.80,
    "has_bisector": -1.20,
    "has_circumcircle": -1.30,
    "has_divisor_remainder": -1.60,
    "has_infinite_sum": -1.40,
    "has_circ_op": -0.90,
    "has_grid_board": -1.10,
    "has_alice_bob": -0.70,

    # Cognitive load & capacity
    "cognitive_load": -0.30,
    "model_param_log": +0.40,      # Larger models more robust
    "reasoning_effort_scale": +0.60,  # Higher effort = more robust
    "capacity_to_load_ratio": +0.50,  # Capacity/load ratio positive
}

ENSEMBLE_INTERCEPT = -0.10  # Calibrated for balanced decisions


def _z_score(value: float, name: str) -> float:
    """Normalize feature to z-score using pre-computed statistics."""
    mean = FEATURE_MEANS.get(name, 0.0)
    std = FEATURE_STDS.get(name, 1.0)
    return (value - mean) / max(1e-6, std)


def compute_hybrid_decision_logit(
    model_id: str,
    reasoning_effort: str,
    features: Dict[str, float],
) -> float:
    """Compute combined linear meta-classifier logit for robustness decision.

    Args:
        model_id: Target model identifier
        reasoning_effort: "low", "medium", "high", or "default"
        features: Dictionary of all extracted features

    Returns:
        Logit: positive -> robust (True), negative -> spurious (False)
    """
    # 1. Base prior from model identity and effort
    prior_log_odds = get_model_prior_log_odds(model_id, reasoning_effort)

    # 2. Compute weighted z-scored feature contributions
    total_logit = ENSEMBLE_INTERCEPT + prior_log_odds

    for fname, weight in ENSEMBLE_WEIGHTS.items():
        value = features.get(fname, FEATURE_MEANS.get(fname, 0.0))
        z = _z_score(value, fname)
        total_logit += weight * z

    # 3. Effort-specific modulation for low-effort regime
    m_clean = (model_id or "").lower()
    e_clean = (reasoning_effort or "default").lower()

    is_low_effort = (e_clean == "low")
    is_small_model = any(tag in m_clean for tag in ("4b", "7b", "8b"))
    is_frontier = ("120b" in m_clean or "gpt-oss" in m_clean or "nvfp4" in m_clean or "gpt-5" in m_clean)

    if is_low_effort:
        # For low effort: upweight digit JSD and expert sensitivity, downweight spectral rank
        cdi_digits_z = _z_score(features.get("cdi_digits", 0.0), "cdi_digits")
        total_logit += 0.50 * cdi_digits_z  # Extra boost for digit contrast on low effort

        drift_ablate_z = _z_score(features.get("drift_expert_ablate", 0.24), "drift_expert_ablate")
        total_logit += 0.40 * drift_ablate_z  # Expert sensitivity critical for low effort

        # Penalize structural complexity more aggressively for small models on low effort
        if is_small_model:
            n_cond_z = _z_score(features.get("n_conditions", 2.8), "n_conditions")
            len_z = _z_score(features.get("log_length", 5.6), "log_length")
            total_logit -= 0.30 * n_cond_z + 0.20 * len_z

    elif is_frontier and e_clean == "high":
        # Frontier models on high effort: trust spectral rank and CoT dynamics more
        spec_z = _z_score(features.get("spectral_rank", 0.60), "spectral_rank")
        total_logit += 0.30 * spec_z
        margin_z = _z_score(features.get("cot_mean_margin", 0.45), "cot_mean_margin")
        total_logit += 0.25 * margin_z

    # 4. Fragility trigger overrides (hard vetos for known failure modes)
    if features.get("has_incenter", 0) > 0.5:
        total_logit -= 2.0  # Incenter problems extremely fragile
    if features.get("has_divisor_remainder", 0) > 0.5:
        total_logit -= 2.0  # Divisor+remainder = number theory traps
    if features.get("has_infinite_sum", 0) > 0.5:
        total_logit -= 2.0  # Infinite sum = analysis traps
    if features.get("has_circumcircle", 0) > 0.5 and is_small_model:
        total_logit -= 1.5  # Circumcircle geometry kills small models

    return float(total_logit)