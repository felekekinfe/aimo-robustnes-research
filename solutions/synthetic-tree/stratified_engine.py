"""Type-Stratified Threshold Engine for AIMO Synthetic Tree.

Pillar 3 of Final Architecture Blueprint:
- Evaluates each branch against its dedicated baseline threshold (tau_t).
- Computes branch pass flags, group pass rates, worst-case invariant drift, and causal sensitivity.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

# Type-stratified threshold baselines
TYPE_THRESHOLDS = {
    # Group 1: Surface & Typos
    "typo_char": 0.040,
    "typo_delim": 0.040,
    "format_delims": 0.040,
    "format_whitespace": 0.040,
    # Group 2: Rename & Rephrase
    "rename_algebraic": 0.090,
    "rename_geometric": 0.090,
    "rephrase_commute": 0.090,
    "rephrase_query": 0.090,
    # Group 3: Domain & Distract
    "domain_academic": 0.170,
    "domain_entities": 0.170,
    "distract_prepend": 0.170,
    "distract_append": 0.170,
    # Group 4: Expert & Boundary Sensitivity
    "expert_ablate": 0.220,
    "expert_mutate": 0.220,
}

INVARIANT_BRANCHES = [
    "typo_char", "typo_delim", "format_delims", "format_whitespace",
    "rename_algebraic", "rename_geometric", "rephrase_commute", "rephrase_query",
    "domain_academic", "domain_entities", "distract_prepend", "distract_append",
]


def evaluate_stratified_metrics(raw_metrics: Dict[str, float]) -> Dict[str, float]:
    """Process raw branch drifts into type-stratified pass flags and summary features."""
    pass_flags: Dict[str, float] = {}
    invariant_drifts: List[float] = []

    # 1. Invariant branches pass flags (drifts must be BELOW threshold)
    for b in INVARIANT_BRANCHES:
        d = raw_metrics.get(f"drift_{b}", 0.05)
        tau = TYPE_THRESHOLDS.get(b, 0.10)
        passed = 1.0 if d <= tau else 0.0
        pass_flags[f"pass_{b}"] = passed
        invariant_drifts.append(d)

    # 2. Expert sensitivity branches (drifts should be ABOVE threshold to show logical sensitivity)
    d_ablate = raw_metrics.get("drift_expert_ablate", 0.22)
    d_mutate = raw_metrics.get("drift_expert_mutate", 0.22)
    sens_ablate = 1.0 if d_ablate >= TYPE_THRESHOLDS["expert_ablate"] else 0.0
    sens_mutate = 1.0 if d_mutate >= TYPE_THRESHOLDS["expert_mutate"] else 0.0

    # 3. Aggregate summary features
    max_inv_drift = float(np.max(invariant_drifts)) if invariant_drifts else 0.10
    mean_inv_drift = float(np.mean(invariant_drifts)) if invariant_drifts else 0.08
    std_inv_drift = float(np.std(invariant_drifts)) if invariant_drifts else 0.02
    inv_pass_rate = float(np.mean(list(pass_flags.values()))) if pass_flags else 0.80

    # Causal sensitivity: sensitivity drift minus surface drift
    mean_surface_drift = float(np.mean([
        raw_metrics.get("drift_typo_char", 0.03),
        raw_metrics.get("drift_format_delims", 0.03),
    ]))
    mean_expert_drift = float(0.50 * d_ablate + 0.50 * d_mutate)
    causal_sensitivity = float(mean_expert_drift - mean_surface_drift)

    summary = {
        "max_inv_drift": max_inv_drift,
        "mean_inv_drift": mean_inv_drift,
        "std_inv_drift": std_inv_drift,
        "inv_pass_rate": inv_pass_rate,
        "causal_sensitivity": causal_sensitivity,
        "sens_ablate": sens_ablate,
        "sens_mutate": sens_mutate,
    }
    summary.update(pass_flags)
    return summary
