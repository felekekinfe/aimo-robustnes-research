"""Training and Cross-Validation Pipeline for Streamlined Hybrid Robustness Ensemble.

Features:
- GroupKFold CV strictly grouped by problem text (zero data leakage).
- L2-regularized Logistic Regression + Ridge Classifier Ensemble.
- Bootstrap 95% confidence intervals for AUC and Accuracy.
- Formats analytical weights directly into Python constants (eliminating .pkl / .joblib dependencies).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from extractor import FEATURE_NAMES, StreamlinedHybridExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AIMO-Train-Evaluate")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval for a metric function."""
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


def load_dataset(cases_path: Path, labels_path: Path | None = None) -> List[Dict[str, Any]]:
    """Load jsonl cases and optionally merge labels."""
    cases = []
    with cases_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    if labels_path and labels_path.is_file():
        labels = {}
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    labels[item["id"]] = item["is_robust"]
        for c in cases:
            if c["id"] in labels:
                c["is_robust"] = labels[c["id"]]

    return cases


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Run GroupKFold cross validation with an L2 regularized ensemble."""
    n_samples = len(y)
    unique_groups = len(np.unique(groups))
    splits = min(n_splits, unique_groups)
    gkf = GroupKFold(n_splits=splits)

    # 1. Empirical-Bayes Model Prior
    y_prob_prior = np.zeros(n_samples)
    for train_idx, val_idx in gkf.split(X, y, groups):
        train_y = y[train_idx]
        global_p = float(np.mean(train_y))
        model_counts = defaultdict(lambda: [0, 0])
        for m, label in zip(models[train_idx], train_y):
            model_counts[m][label] += 1
        for vi in val_idx:
            m = models[vi]
            n0, n1 = model_counts[m]
            # Empirical Bayes smoothing with k=1.0
            prob = (n1 + 1.0 * global_p) / (n0 + n1 + 1.0)
            y_prob_prior[vi] = prob

    prior_clipped = np.clip(y_prob_prior, 0.05, 0.95)
    log_odds_prior = np.log(prior_clipped / (1.0 - prior_clipped)).reshape(-1, 1)

    # Combine Prior with extracted features
    X_full = np.hstack([log_odds_prior, X])

    # 2. Fit L2 Logistic Regression & Ridge Classifier inside CV
    y_prob_cv = np.zeros(n_samples)
    
    for train_idx, val_idx in gkf.split(X_full, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_full[train_idx])
        X_va = scaler.transform(X_full[val_idx])

        # L2-regularized Logistic Regression (C=0.1)
        clf_lr = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", random_state=42)
        clf_lr.fit(X_tr, y[train_idx])
        prob_lr = clf_lr.predict_proba(X_va)[:, 1]

        # L2-regularized Ridge Classifier (alpha=10.0)
        clf_ridge = RidgeClassifier(alpha=10.0, random_state=42)
        clf_ridge.fit(X_tr, y[train_idx])
        dec_ridge = clf_ridge.decision_function(X_va)
        prob_ridge = 1.0 / (1.0 + np.exp(-dec_ridge))  # Sigmoid conversion

        # Equal-weighted ensemble
        y_prob_cv[val_idx] = 0.50 * prob_lr + 0.50 * prob_ridge

    # Decision threshold aligned to training prior
    y_pred_cv = (y_prob_cv >= 0.50).astype(int)

    acc = float(accuracy_score(y, y_pred_cv))
    auc = float(roc_auc_score(y, y_prob_cv)) if len(np.unique(y)) > 1 else 0.50
    ci_acc = bootstrap_ci(y, y_pred_cv, accuracy_score)
    ci_auc = bootstrap_ci(y, y_prob_cv, roc_auc_score)
    pred_true_rate = float(np.mean(y_pred_cv))

    return {
        "accuracy": acc,
        "ci_acc": ci_acc,
        "auc": auc,
        "ci_auc": ci_auc,
        "pred_true_rate": pred_true_rate,
        "y_prob_cv": y_prob_cv,
    }


def fit_and_export_analytical_weights(
    X: np.ndarray,
    y: np.ndarray,
    models: np.ndarray,
) -> Dict[str, Any]:
    """Fit full dataset model and extract explicit weights for pure Python deployment."""
    # Compute global empirical-Bayes model priors
    global_p = float(np.mean(y))
    model_counts = defaultdict(lambda: [0, 0])
    for m, label in zip(models, y):
        model_counts[m][label] += 1
    
    model_priors: Dict[str, float] = {}
    for m in np.unique(models):
        n0, n1 = model_counts[m]
        prob = (n1 + 1.0 * global_p) / (n0 + n1 + 1.0)
        prob_clip = min(0.95, max(0.05, prob))
        log_odds = float(np.log(prob_clip / (1.0 - prob_clip)))
        model_priors[str(m)] = log_odds

    log_odds_vector = np.array([model_priors[str(m)] for m in models]).reshape(-1, 1)
    X_full = np.hstack([log_odds_vector, X])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    clf_lr = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", random_state=42)
    clf_lr.fit(X_scaled, y)

    clf_ridge = RidgeClassifier(alpha=10.0, random_state=42)
    clf_ridge.fit(X_scaled, y)

    # Average ensemble coefficients
    ensemble_weights = 0.50 * clf_lr.coef_[0] + 0.50 * clf_ridge.coef_[0]
    ensemble_intercept = float(0.50 * clf_lr.intercept_[0] + 0.50 * clf_ridge.intercept_[0])

    analytical_config = {
        "feature_names": ["log_odds_prior"] + FEATURE_NAMES,
        "scaler_means": [float(m) for m in scaler.mean_],
        "scaler_scales": [float(s) for s in scaler.scale_],
        "ensemble_weights": [float(w) for w in ensemble_weights],
        "ensemble_intercept": float(ensemble_intercept),
        "model_priors": model_priors,
    }
    return analytical_config


if __name__ == "__main__":
    val_sample_cases = DATA_DIR / "val-sample" / "input" / "cases.jsonl"
    val_sample_labels = DATA_DIR / "val-sample" / "reference" / "labels.jsonl"
    
    logger.info("Loading validation dataset from %s...", val_sample_cases)
    cases = load_dataset(val_sample_cases, val_sample_labels)
    
    extractor = StreamlinedHybridExtractor(model=None, tokenizer=None)
    
    logger.info("Extracting features for %d samples...", len(cases))
    X_list = [extractor.extract_features(c["problem"]) for c in cases]
    X = np.vstack(X_list)
    y = np.array([1 if c["is_robust"] else 0 for c in cases])
    groups = np.array([c["problem"].strip() for c in cases])
    models = np.array([c["model_id"] for c in cases])

    logger.info("Running 5-fold Grouped-CV evaluation...")
    metrics = run_cross_validation(X, y, groups, models, n_splits=5)

    print("\n" + "=" * 65)
    print(" STREAMLINED HYBRID ENSEMBLE EVALUATION (GROUPED 5-FOLD CV)")
    print("=" * 65)
    print(f"Accuracy:        {metrics['accuracy']:.4f} [95% CI: {metrics['ci_acc'][0]:.4f}, {metrics['ci_acc'][1]:.4f}]")
    print(f"ROC-AUC:         {metrics['auc']:.4f} [95% CI: {metrics['ci_auc'][0]:.4f}, {metrics['ci_auc'][1]:.4f}]")
    print(f"Pred True Rate:  {metrics['pred_true_rate']:.3f}")
    print("=" * 65)

    config = fit_and_export_analytical_weights(X, y, models)
    logger.info("Analytical configuration successfully generated.")
