"""Test literature-derived features (PutnamGAP parametric load, GSM-Symbolic distractor risk) in Grouped-CV."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MATH_SPAN_RE = re.compile(r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
RELATION_RE = re.compile(r"(=|\\le|\\ge|\\leq|\\geq|<|>|\\equiv|\\mid)")

def extract_features(problem: str) -> dict[str, float]:
    p = problem.strip()
    words = p.split()
    n_words = len(words)
    numbers = NUMBER_RE.findall(p)
    n_numbers = len(numbers)
    relations = RELATION_RE.findall(p)
    n_conditions = len(relations)

    # 1. PutnamGAP: Parametric Load Ratio
    parametric_load = (n_numbers * n_conditions) / max(1.0, float(n_words))

    # 2. GSM-Symbolic: Premise Distractor Risk
    # Split text into sentences preserving math
    sentences = re.split(r"(?<=[.?!])\s+", p)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(1, len(sentences))
    zero_math_sentences = sum(1 for s in sentences if not MATH_SPAN_RE.search(s))
    distractor_risk = zero_math_sentences / n_sentences

    return {
        "n_conditions": float(n_conditions),
        "parametric_load": float(parametric_load),
        "distractor_risk": float(distractor_risk),
    }

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_bootstraps: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    scores = []
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

def evaluate(cases_path: str, labels_path: str):
    cases = [json.loads(l) for l in open(cases_path, encoding="utf-8")]
    labels = {json.loads(l)["id"]: json.loads(l)["is_robust"] for l in open(labels_path, encoding="utf-8")}
    for c in cases:
        c["is_robust"] = labels[c["id"]]

    y = np.array([1 if c["is_robust"] else 0 for c in cases])
    groups = np.array([c["problem"].strip() for c in cases])
    models = np.array([c["model_id"] for c in cases])
    n_samples = len(y)

    feat_dicts = [extract_features(c["problem"]) for c in cases]

    gkf = GroupKFold(n_splits=5)

    # Base log-odds prior
    y_prob_prior = np.zeros(n_samples)
    for train_idx, val_idx in gkf.split(cases, y, groups):
        train_y = y[train_idx]
        global_p = np.mean(train_y)
        m_counts = defaultdict(lambda: [0, 0])
        for m, yi in zip(models[train_idx], train_y):
            m_counts[m][yi] += 1
        for vi in val_idx:
            m = models[vi]
            n0, n1 = m_counts[m]
            y_prob_prior[vi] = (n1 + 1.0 * global_p) / (n0 + n1 + 1.0)

    prior_clipped = np.clip(y_prob_prior, 0.05, 0.95)
    log_odds_prior = np.log(prior_clipped / (1.0 - prior_clipped)).reshape(-1, 1)

    print("=================================================================")
    print(" TESTING CANDIDATE LITERATURE FEATURES IN GROUPED-CV (5-FOLD)")
    print(" Baseline Floor (Prior + n_conditions): AUC = 0.8926, Acc = 0.8750")
    print("=================================================================")

    # Configurations to test:
    test_configs = [
        ("Prior + n_conditions (Baseline Champion)", ["n_conditions"]),
        ("Prior + parametric_load (PutnamGAP)", ["parametric_load"]),
        ("Prior + distractor_risk (GSM-Symbolic)", ["distractor_risk"]),
        ("Prior + n_conditions + parametric_load", ["n_conditions", "parametric_load"]),
        ("Prior + n_conditions + distractor_risk", ["n_conditions", "distractor_risk"]),
        ("Prior + n_conditions + parametric_load + distractor_risk", ["n_conditions", "parametric_load", "distractor_risk"]),
    ]

    for name, f_list in test_configs:
        X_feats = np.array([[fd[fn] for fn in f_list] for fd in feat_dicts])
        X_all = np.hstack([log_odds_prior, X_feats])

        y_prob_cv = np.zeros(n_samples)
        for train_idx, val_idx in gkf.split(cases, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_all[train_idx])
            X_va = scaler.transform(X_all[val_idx])

            clf = LogisticRegression(C=1.0)
            clf.fit(X_tr, y[train_idx])
            y_prob_cv[val_idx] = clf.predict_proba(X_va)[:, 1]

        auc = roc_auc_score(y, y_prob_cv)
        y_pred = (y_prob_cv >= 0.5).astype(int)
        acc = accuracy_score(y, y_pred)
        ci_acc = bootstrap_ci(y, y_pred, accuracy_score)
        ci_auc = bootstrap_ci(y, y_prob_cv, roc_auc_score)
        ptr = np.mean(y_pred)

        delta_auc = auc - 0.8926
        status = "IMPROVED" if delta_auc > 0.001 else ("TIED" if abs(delta_auc) <= 0.001 else "DEGRADED")

        print(f"\nConfiguration: {name}")
        print(f"  AUC:      {auc:.4f} (delta: {delta_auc:+.4f}) [95% CI: {ci_auc[0]:.4f}, {ci_auc[1]:.4f}] -> {status}")
        print(f"  Accuracy: {acc:.4f} [95% CI: {ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
        print(f"  Pred True Rate: {ptr:.3f}")

if __name__ == "__main__":
    evaluate("data/val-sample/input/cases.jsonl", "data/val-sample/reference/labels.jsonl")
