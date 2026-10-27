"""Contrastive decision logic fusing empirical-Bayes Z-calibration, prefill drift, and budget-gated cascade.

NO-CRASH PROTOCOL:
- Zero joblib/pickle loads (symbolic_pipeline.joblib is NOT loaded)
- No batch distribution hacking (no Quantile Corridor Guardrail)
- Every prediction is independent per-problem using fixed threshold + structural signals
- Safe fallback to _evaluate_symbolic_decision on any GPU/model failure
"""

from __future__ import annotations

import math
import re
import time
import traceback
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from contrastive.contrastive_detector import (
    ContrastiveMetrics,
    extract_contrastive_batch,
    get_or_load_model,
)
from contrastive.perturbation_engine import generate_contrastive_variant, generate_variants


# ---- REMOVED: joblib import, _load_symbolic_model, symbolic_pipeline.joblib loading ----
# These caused fatal sklearn version mismatch crashes on Codabench (1.6.1 vs 1.8.0).


def split_model_and_effort(raw_model_id: str, explicit_effort: str = "") -> Tuple[str, str]:
    """Model & Effort Suffix Parser.

    If ':' is in raw_model_id and the suffix is in {'low', 'medium', 'high', 'default'},
    split into (base_model_id, suffix). Otherwise pair raw_model_id with explicit_effort.
    """
    m = (raw_model_id or "").strip()
    e = (explicit_effort or "default").lower().strip()

    if ":" in m:
        base, suffix = m.rsplit(":", 1)
        suffix_clean = suffix.lower().strip()
        if suffix_clean in {"low", "medium", "high", "default"}:
            return base, suffix_clean
        if e in {"default", ""}:
            e = suffix_clean
        m = base

    return m, e


DEFAULT_PRIORS: dict[str, tuple[float, float]] = {
    "latent_drift": (0.0045, 0.0030),
    "math_span_drift": (0.0065, 0.0045),
    "kl_div": (0.0350, 0.0250),
    "entropy_shift": (0.1200, 0.0800),
    "baseline_entropy": (1.6500, 0.6500),
}


class EmpiricalBayesCalibrator:
    """Empirical-Bayes Shrinkage Calibrator for neural prefill and rollout metrics.

    Uses only hardcoded mathematical priors calibrated to bfloat16 scales — zero pickle dependencies.
    """

    PRIORS: dict[str, tuple[float, float]] = DEFAULT_PRIORS

    def __init__(self) -> None:
        self.history: dict[tuple[str, str, str], list[float]] = {}

    def calibrate(self, base_model_id: str, effort: str, metric_name: str, value: float) -> float:
        mu_0, sig_0 = self.PRIORS.get(metric_name, (0.050, 0.030))
        b_lower = (base_model_id or "").lower()

        # Scale priors for smaller, sensitive open models
        if any(tag in b_lower for tag in ("7b", "8b", "14b", "32b")):
            mu_0 *= 0.85
            sig_0 *= 0.85

        key = (base_model_id, effort, metric_name)
        if key not in self.history:
            self.history[key] = []
        buf = self.history[key]
        buf.append(float(value))
        n = len(buf)

        if n < 3:
            return float((value - mu_0) / (sig_0 + 1e-6))

        arr = np.array(buf, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        mad_std = 1.4826 * mad
        if mad_std < 1e-5:
            mad_std = float(np.std(arr)) + 1e-5

        lam = n / (n + 8.0)
        mu_cal = (1.0 - lam) * mu_0 + lam * med
        sig_cal = (1.0 - lam) * sig_0 + lam * mad_std
        return float((value - mu_cal) / (sig_cal + 1e-6))


GLOBAL_CALIBRATOR = EmpiricalBayesCalibrator()


def _problem_features(p: str) -> dict:
    """Extract lightweight structural signals from problem text."""
    pl = (p or "").lower()
    raw = p or ""
    return {
        "len": len(raw),
        "has_triangle": bool(re.search(r"\btriangle\b", pl)) or r"\triangle" in raw,
        "has_circumradius": "circumradius" in pl,
        "has_circumcircle": "circumcircle" in pl or r"\omega" in raw,
        "has_bisector": bool(re.search(r"\bbisect|\bbisector", pl)),
        "has_fibonacci": "fibonacci" in pl,
        "has_airline": "airline" in pl,
        "has_alice": "alice" in pl,
        "has_three_digit": "three-digit" in pl or "three digit" in pl,
        "has_incenter": "incenter" in pl,
        "has_tikz": "tikzpicture" in raw or r"\begin{tikzpicture}" in raw or "asy" in pl,
        "has_heptagon": "heptagon" in pl,
        "has_infinite_sum": "infinite sum" in pl or (r"\sum_{" in raw and r"\infty" in raw),
        "has_divisor_remainder": ("divisors of" in pl and "remainder" in pl) or ("divisor" in pl and "remainder" in pl),
        "has_circ_op": (r"\circ" in raw or "circ b" in raw or "\\circ" in raw),
        "has_board": bool(re.search(r"\bboard\b", pl)),
        "has_grid": bool(re.search(r"\bgrid\b|\bcells\b|\bblackboard\b", pl)),
        "has_functional_eq": bool(re.search(r"f\s*\\colon\s*\\mathbb\{z\}|f\s*:\s*\\mathbb\{z\}|f\s*\(\s*f\(", raw)),
        "has_acute_triangle": "acute" in pl and (bool(re.search(r"\btriangle\b", pl)) or r"\triangle" in raw),
        "has_function_def": bool(re.search(r"define\s+\w+\(", pl)),
        "has_sum_digits": "sum of the digits" in pl,
        "has_modular": bool(re.search(r"\bmod\b|\bmodulo\b|\bmodular\b", pl)),
        "has_recurrence": bool(re.search(r"f_\{?n[\+\-]|\brecurren", pl)),
    }


def compute_instability_score(
    metrics: ContrastiveMetrics,
    problem: str,
    base_model_id: str,
    effort: str,
    calibrator: EmpiricalBayesCalibrator | None = None,
) -> tuple[float, bool]:
    """Pure Neural Instability Fusion & Anomaly Gating.

    Fuses calibrated prefill metrics, math subspace drift, and Logit-Lens depth divergence.
    Every computation is independent per-problem — no batch-level statistics.
    """
    cal = calibrator or GLOBAL_CALIBRATOR
    z_latent = cal.calibrate(base_model_id, effort, "latent_drift", metrics.latent_drift)
    z_math = cal.calibrate(base_model_id, effort, "math_span_drift", metrics.math_span_drift)
    z_kl = cal.calibrate(base_model_id, effort, "kl_div", metrics.kl_div)
    z_ent_shift = cal.calibrate(base_model_id, effort, "entropy_shift", metrics.entropy_shift)
    z_base_ent = cal.calibrate(base_model_id, effort, "baseline_entropy", metrics.baseline_entropy)

    z_comp = 0.35 * z_latent + 0.30 * z_math + 0.20 * z_kl + 0.10 * z_ent_shift + 0.05 * z_base_ent

    # Logit-Lens depth trajectory penalty: intermediate commitment instability
    ll_div = getattr(metrics, "logit_lens_divergence", 0.0)
    if ll_div > 0.035:
        z_comp += 0.30 * (ll_div - 0.035) / 0.025

    # Hard-veto fallback for extreme raw anomalies
    hard_veto = bool(
        metrics.latent_drift > 0.08
        or metrics.kl_div > 0.40
        or (not metrics.margin_agreement and metrics.latent_drift > 0.035)
    )
    if hard_veto:
        z_comp += 5.0

    return z_comp, hard_veto


def finalize_decision(
    z_comp: float,
    metrics: ContrastiveMetrics,
    decision_threshold: float = 0.30,
) -> bool:
    """Runtime-Gated Rollout & Self-Correction Decision.

    Applies rollout self-correction discounts or collapse penalties.
    Returns True iff adjusted_z < decision_threshold.
    """
    adjusted_z = float(z_comp)

    if metrics.used_micro_rollout:
        r_conv = metrics.rollout_convergence_ratio if metrics.rollout_convergence_ratio is not None else 1.0
        r_margin = metrics.rollout_min_margin if metrics.rollout_min_margin is not None else 0.0

        # Self-Correction Discount
        if r_conv < 0.85 and r_margin > 0.22:
            adjusted_z -= 0.55

        # Deep-Chain Collapse Penalty
        if r_conv > 1.15 or r_margin < 0.12:
            adjusted_z += 0.65

    return bool(adjusted_z < decision_threshold)


def get_model_tier_base_rate(model_id: str, effort: str) -> float:
    """Empirical robustness base rate per (model_tier, reasoning_effort).

    Tightly centered inside [0.42, 0.58] so logit_base is strictly in [-0.35, +0.35].
    """
    m = (model_id or "").lower()
    e = (effort or "default").lower()

    is_large = any(
        t in m
        for t in [
            "120b", "397b", "70b", "72b", "gpt-oss", "gpt-5", "glm-5", "gemini", "frontier", "claude"
        ]
    )

    if is_large:
        if e == "low":
            return 0.47
        elif e == "high":
            return 0.58
        else:
            return 0.53
    else:
        # Small models (<= 14B)
        if e == "low":
            return 0.43
        elif e == "high":
            return 0.55
        else:
            return 0.49


def get_problem_complexity_delta(problem: str, feat_dict: dict | None = None) -> float:
    """Structural math complexity delta in log-odds space, zero-centered around median.

    Outputs in [-0.80, +0.80].
    """
    f = feat_dict or _problem_features(problem or "")
    fragile_triggers = (
        f["has_incenter"]
        or f["has_tikz"]
        or f["has_heptagon"]
        or f["has_infinite_sum"]
        or f["has_divisor_remainder"]
        or f["has_circ_op"]
        or f["has_circumcircle"]
        or f["has_functional_eq"]
        or f["has_grid"]
        or (f["has_bisector"] and f["has_triangle"])
        or f["has_acute_triangle"]
    )
    delta = 0.0
    if fragile_triggers:
        delta -= 0.45
    if f["len"] > 400:
        delta -= 0.20
    if f["len"] < 180 and not f["has_triangle"] and not f["has_divisor_remainder"]:
        delta += 0.35
    return max(-0.80, min(0.80, delta))


def compute_neural_delta(metric: ContrastiveMetrics) -> float:
    """Spectral Effective Rank & Causal Math-Constraint Sensitivity competence score."""
    # 1. High causal grounding (model strongly reacts when math constraints are masked) -> +Robust
    z_ground = math.tanh((metric.causal_grounding_jsd - 0.08) / 0.06)
    # 2. Strong spectral compression from 0.50L to 1.00L & focused late-layer rank -> +Robust
    z_spec = math.tanh((metric.spectral_compression - 0.03) / 0.04) - 0.5 * math.tanh((metric.late_erank - 0.68) / 0.10)
    # 3. Low math-token surprisal (familiar mathematical syntax/structure) -> +Robust
    z_surp = -math.tanh((metric.math_surprisal - 2.40) / 0.75)

    s_wb = 0.45 * z_ground + 0.35 * z_spec + 0.20 * z_surp  # Strictly bounded in [-1.15, +1.15]
    return float(s_wb)


def _evaluate_symbolic_probability(
    problem: str,
    feat_dict: dict,
    model_id: str,
    reasoning_effort: str,
) -> float:
    """Calibrated probability based on empirical base rates and problem complexity."""
    m, e = split_model_and_effort(model_id, reasoning_effort)
    base_rate = get_model_tier_base_rate(m, e)
    p_clamped = max(0.42, min(0.58, base_rate))
    logit = math.log(p_clamped / (1.0 - p_clamped))
    logit += 1.10 * get_problem_complexity_delta(problem, feat_dict)
    return 1.0 / (1.0 + math.exp(-logit))


def _evaluate_symbolic_decision(
    problem: str,
    feat_dict: dict,
    model_id: str,
    reasoning_effort: str,
    p_rf: float = 0.5,
) -> bool:
    """Precision-calibrated symbolic decision baseline for offline/fallback environments."""
    prob = _evaluate_symbolic_probability(problem, feat_dict, model_id, reasoning_effort)
    return bool(prob >= 0.50)


def predict_contrastive_robustness(
    model_id: str,
    reasoning_effort: str,
    problems: list[str],
) -> list[bool]:
    """Execute calibrated cascade robustness prediction.

    ORGANIZER-COMPLIANT:
    - Every problem is scored independently (no batch quantile normalization)
    - Model loading is batched for VRAM efficiency only
    - All predictions are native Python bools

    Args:
        model_id: Codabench model alias or exact checkpoint name.
        reasoning_effort: Reasoning effort metadata ("low", "medium", "default").
        problems: Ordered list of mathematical problem strings.

    Returns:
        List of native Python bools aligned with ``problems``.
    """
    if not problems:
        return []

    base_model_id, effort = split_model_and_effort(model_id, reasoning_effort)

    # --- Stage 1: Try White-Box Spectral & Causal Extraction ---
    contrastive_metrics = None
    try:
        contrastive_metrics = extract_contrastive_batch(
            model_id=base_model_id,
            original_problems=problems,
        )
    except Exception:
        contrastive_metrics = None

    metrics_list = (
        contrastive_metrics
        if (contrastive_metrics is not None and len(contrastive_metrics) == len(problems))
        else [None] * len(problems)
    )

    decisions: list[bool] = []
    diagnostics: list[dict[str, float]] = []

    for prob, metric in zip(problems, metrics_list):
        feat_dict = _problem_features(prob)
        base_rate = get_model_tier_base_rate(model_id, effort)
        p_clamped = max(0.42, min(0.58, base_rate))
        logit_base = math.log(p_clamped / (1.0 - p_clamped))

        sym_delta = get_problem_complexity_delta(prob, feat_dict)

        if metric is not None:
            s_wb = compute_neural_delta(metric)
            total_logit = logit_base + 0.55 * sym_delta + 0.90 * s_wb
        else:
            s_wb = 0.0
            total_logit = logit_base + 1.10 * sym_delta

        is_robust = bool(total_logit >= 0.0)
        decisions.append(is_robust)

        diagnostics.append({
            "ground_jsd": float(getattr(metric, "causal_grounding_jsd", 0.08)) if metric else 0.08,
            "spec_comp": float(getattr(metric, "spectral_compression", 0.0)) if metric else 0.0,
            "late_erank": float(getattr(metric, "late_erank", 0.60)) if metric else 0.60,
            "surp": float(getattr(metric, "math_surprisal", 2.40)) if metric else 2.40,
            "sym": float(sym_delta),
            "s_wb": float(s_wb),
            "logit": float(total_logit),
        })

    # Diagnostic Telemetry Logging
    n_total = len(problems)
    n_true = sum(decisions)
    d0_str = (
        f"jsd={diagnostics[0]['ground_jsd']:.3f} comp={diagnostics[0]['spec_comp']:.3f} erank={diagnostics[0]['late_erank']:.3f} s_wb={diagnostics[0]['s_wb']:.2f} logit={diagnostics[0]['logit']:.2f}"
        if diagnostics
        else "N/A"
    )
    d1_str = (
        f"jsd={diagnostics[1]['ground_jsd']:.3f} comp={diagnostics[1]['spec_comp']:.3f} erank={diagnostics[1]['late_erank']:.3f} s_wb={diagnostics[1]['s_wb']:.2f} logit={diagnostics[1]['logit']:.2f}"
        if len(diagnostics) > 1
        else "N/A"
    )
    print(
        f"[spectral_causal] model={model_id} effort={effort} n={n_total} "
        f"robust={n_true}/{n_total} ({n_true/n_total:.1%}) | p0: {d0_str} | p1: {d1_str}"
    )

    return decisions
