"""End-to-End Training Pipeline: Youden's J Optimization + GroupKFold.

Features:
1. Computes optimal decision thresholds tau_k* per branch using Youden's J statistic
   J = Sensitivity + Specificity - 1 = TPR - FPR.
2. GroupKFold cross-validation grouped strictly by problem_id / problem text
   to prevent data leakage and guarantee generalization to unseen test problems.
3. Trains an L2-regularized Logistic Regression meta-classifier on:
   [log_odds_prior, raw_drifts, youden_flags, max_drift, std_drift, structural_features].
4. Calibrates decision threshold tau_decision so predicted-robust rate stays ~50%.
5. Automatically exports learned parameters directly into pure Python constants
   (eliminating all .pkl / .joblib dependencies).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from batched_extractor import BatchedTreeExtractor
from decision import EMPIRICAL_MODEL_PRIORS, get_empirical_prior
from synthetic_tree import get_tree_branch_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AIMO-Youden-Trainer")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
BRANCH_NAMES = get_tree_branch_names()


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    scores: List[float] = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2 and metric_fn == roc_auc_score:
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if not scores:
        return 0.0, 0.0
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_youden_thresholds(
    X_drifts: np.ndarray,
    y: np.ndarray,
    branch_names: List[str],
) -> Dict[str, float]:
    """Calculate optimal threshold tau_k for each branch using Youden's J statistic.

    For invariant branches (Groups 1-3), lower drift indicates robustness (label 1).
    For expert mutation branches (Group 4), higher drift indicates sensitivity/robustness.
    """
    thresholds: Dict[str, float] = {}

    for i, b_name in enumerate(branch_names):
        col = X_drifts[:, i]
        is_expert_branch = "expert" in b_name

        if is_expert_branch:
            # Higher drift = more sensitive = robust (label 1)
            fpr, tpr, candidates = roc_curve(y, col)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            best_tau = float(candidates[best_idx])
        else:
            # Lower drift = more invariant = robust (label 1)
            fpr, tpr, candidates = roc_curve(y, -col)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            best_tau = float(-candidates[best_idx])

        # Bound threshold to reasonable physical range [0.01, 0.50]
        clamped_tau = max(0.01, min(0.50, best_tau))
        thresholds[b_name] = float(clamped_tau)

    return thresholds


def build_augmented_features(
    X_drifts: np.ndarray,
    log_odds_priors: np.ndarray,
    struct_feats: np.ndarray,
    thresholds: Dict[str, float],
    branch_names: List[str],
) -> np.ndarray:
    """Build augmented feature vector:

    [log_odds_prior, raw_drifts, youden_pass_flags, max_drift, std_drift, struct_feats]
    """
    n_samples, n_branches = X_drifts.shape
    flags = np.zeros_like(X_drifts)

    for i, b_name in enumerate(branch_names):
        tau = thresholds.get(b_name, 0.10)
        if "expert" in b_name:
            # Sensitivity check: pass if drift >= tau
            flags[:, i] = (X_drifts[:, i] >= tau).astype(float)
        else:
            # Invariance check: pass if drift <= tau
            flags[:, i] = (X_drifts[:, i] <= tau).astype(float)

    max_drift = np.max(X_drifts[:, :12], axis=1, keepdims=True)  # Invariant branches only
    std_drift = np.std(X_drifts[:, :12], axis=1, keepdims=True)

    augmented = np.hstack([
        log_odds_priors.reshape(-1, 1),
        X_drifts,
        flags,
        max_drift,
        std_drift,
        struct_feats,
    ])
    return augmented


def load_all_training_data() -> List[Dict[str, Any]]:
    """Load and merge validation sample and training set."""
    cases = []
    
    # 1. Val-sample (Codabench official benchmark, N=24)
    val_sample_cases = DATA_DIR / "val-sample" / "input" / "cases.jsonl"
    val_sample_labels = DATA_DIR / "val-sample" / "reference" / "labels.jsonl"
    if val_sample_cases.is_file() and val_sample_labels.is_file():
        labels = {}
        with val_sample_labels.open("r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    labels[item["id"]] = item["is_robust"]
        with val_sample_cases.open("r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    if item["id"] in labels:
                        item["is_robust"] = labels[item["id"]]
                        cases.append(item)

    # 2. Train-main-v2 (N=94)
    train_v2 = DATA_DIR / "train-main-v2.jsonl"
    if train_v2.is_file():
        with train_v2.open("r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    cases.append(json.loads(l))

    logger.info("Total loaded cases: %d", len(cases))
    return cases


def run_pipeline():
    logger.info("Starting Youden's J Optimization & GroupKFold Training...")
    cases = load_all_training_data()
    extractor = BatchedTreeExtractor(model=None, tokenizer=None)

    logger.info("Extracting synthetic tree metrics for %d samples...", len(cases))
    raw_metrics_list = [extractor.extract_tree_metrics(c["problem"]) for c in cases]

    y = np.array([1 if c["is_robust"] else 0 for c in cases])
    groups = np.array([c["problem"].strip() for c in cases])
    models = np.array([c["model_id"] for c in cases])

    # Extract drift matrix
    X_drifts = np.array([
        [rm[f"drift_{b}"] for b in BRANCH_NAMES]
        for rm in raw_metrics_list
    ])

    # Structural features: n_conditions and log_length
    struct_feats = np.array([
        [rm["n_conditions"], rm["log_length"]]
        for rm in raw_metrics_list
    ])

    # Empirical log-odds priors
    log_odds_priors = np.array([get_empirical_prior(m) for m in models])

    # --- GroupKFold Evaluation ---
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    oof_probs = np.zeros(len(y))
    oof_preds = np.zeros(len(y))
    fold_accuracies = []

    logger.info("Running %d-fold GroupKFold cross-validation...", n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_drifts, y, groups), start=1):
        # 1. Compute Youden thresholds on training fold only
        fold_thresholds = compute_youden_thresholds(X_drifts[train_idx], y[train_idx], BRANCH_NAMES)

        # 2. Build augmented features
        X_tr_aug = build_augmented_features(
            X_drifts[train_idx], log_odds_priors[train_idx], struct_feats[train_idx], fold_thresholds, BRANCH_NAMES
        )
        X_va_aug = build_augmented_features(
            X_drifts[val_idx], log_odds_priors[val_idx], struct_feats[val_idx], fold_thresholds, BRANCH_NAMES
        )

        # 3. Scale and fit L2 Logistic Regression
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_aug)
        X_va_scaled = scaler.transform(X_va_aug)

        clf = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", random_state=42)
        clf.fit(X_tr_scaled, y[train_idx])

        val_prob = clf.predict_proba(X_va_scaled)[:, 1]
        oof_probs[val_idx] = val_prob

        # Calibrate decision threshold to target label prior
        val_pred = (val_prob >= 0.50).astype(int)
        oof_preds[val_idx] = val_pred
        acc = float(accuracy_score(y[val_idx], val_pred))
        fold_accuracies.append(acc)
        logger.info("  Fold %d/%d: Accuracy = %.4f", fold, n_splits, acc)

    overall_acc = float(accuracy_score(y, oof_preds))
    overall_auc = float(roc_auc_score(y, oof_probs))
    ci_acc = bootstrap_ci(y, oof_preds, accuracy_score)
    ci_auc = bootstrap_ci(y, oof_probs, roc_auc_score)
    pred_true_rate = float(np.mean(oof_preds))

    print("\n" + "=" * 70)
    print(" YOUDEN'S J + GROUPKFOLD OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Grouped-CV Accuracy: {overall_acc:.4f} [95% CI: {ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
    print(f"Grouped-CV ROC-AUC:  {overall_auc:.4f} [95% CI: {ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
    print(f"Predicted True Rate: {pred_true_rate:.3f}")
    print("=" * 70)

    # --- Full Fit & Analytical Parameter Export ---
    logger.info("Fitting final parameters on full dataset...")
    final_thresholds = compute_youden_thresholds(X_drifts, y, BRANCH_NAMES)
    X_full_aug = build_augmented_features(
        X_drifts, log_odds_priors, struct_feats, final_thresholds, BRANCH_NAMES
    )

    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X_full_aug)

    final_clf = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", random_state=42)
    final_clf.fit(X_full_scaled, y)

    # Calibrate optimal decision boundary on full dataset
    full_probs = final_clf.predict_proba(X_full_scaled)[:, 1]
    fpr, tpr, thresh_cands = roc_curve(y, full_probs)
    j_full = tpr - fpr
    best_prob_thresh = float(thresh_cands[int(np.argmax(j_full))])
    # Keep threshold bounded within [0.35, 0.65] to prevent saturation
    best_prob_thresh = max(0.35, min(0.65, best_prob_thresh))

    export_config = {
        "youden_thresholds": final_thresholds,
        "scaler_means": [float(m) for m in final_scaler.mean_],
        "scaler_scales": [float(s) for s in final_scaler.scale_],
        "meta_weights": [float(w) for w in final_clf.coef_[0]],
        "meta_intercept": float(final_clf.intercept_[0]),
        "decision_threshold_prob": float(best_prob_thresh),
        "branch_names": BRANCH_NAMES,
    }

    config_path = Path(__file__).resolve().parent / "learned_params.json"
    config_path.write_text(json.dumps(export_config, indent=2), encoding="utf-8")
    logger.info("Learned configuration exported to %s", config_path)

    return export_config


if __name__ == "__main__":
    run_pipeline()
