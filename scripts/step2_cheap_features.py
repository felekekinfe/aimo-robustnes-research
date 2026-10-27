"""STEP 2: Cheap Features & Grouped-CV Feature Selection."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
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

GEOM_WORDS = {"triangle", "circle", "radius", "diameter", "circumcircle", "incenter", "angle", "polygon", "area"}
NT_WORDS = {"prime", "primes", "divisible", "divisor", "modulo", "mod", "remainder", "integer", "integers", "gcd"}
COMB_WORDS = {"probability", "marbles", "ways", "permutation", "combination", "urn", "draw", "random"}

def extract_metadata_features(problem: str) -> dict[str, float]:
    p = problem.strip()
    p_lower = p.lower()
    n_chars = len(p)
    n_words = len(p.split())

    math_spans = MATH_SPAN_RE.findall(p)
    math_chars = sum(len(m) for m in math_spans)
    density = math_chars / max(1, n_chars)

    numbers = NUMBER_RE.findall(p)
    relations = RELATION_RE.findall(p)

    is_geom = float(any(w in p_lower for w in GEOM_WORDS))
    is_nt = float(any(w in p_lower for w in NT_WORDS))
    is_comb = float(any(w in p_lower for w in COMB_WORDS))

    has_incenter = float("incenter" in p_lower or "circumcircle" in p_lower)
    has_grid = float("grid" in p_lower or "table" in p_lower)

    return {
        "log_length": np.log(max(10, n_chars)),
        "n_words": float(n_words),
        "n_math_spans": float(len(math_spans)),
        "math_density": float(density),
        "n_numbers": float(len(numbers)),
        "n_conditions": float(len(relations)),
        "is_geom": is_geom,
        "is_nt": is_nt,
        "is_comb": is_comb,
        "fragile_cues": float(has_incenter or has_grid),
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

def run_step2(dataset_name: str, rows: list[dict], prob_key: str, model_key: str, label_key: str):
    print(f"\n=======================================================")
    print(f" STEP 2 FEATURE EVALUATION FOR: {dataset_name} (N={len(rows)})")
    print(f"=======================================================")

    valid_rows = [r for r in rows if r.get(label_key) is not None]
    y = np.array([1 if r[label_key] is True else 0 for r in valid_rows])
    groups = np.array([r[prob_key].strip() for r in valid_rows])
    models = np.array([r[model_key] for r in valid_rows])

    n_samples = len(y)
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    # 1. Base Prior Floor (GroupKFold with k=1.0 empirical-Bayes)
    y_prob_prior = np.zeros(n_samples)
    for train_idx, val_idx in gkf.split(valid_rows, y, groups):
        train_y = y[train_idx]
        global_p = np.mean(train_y)
        model_counts = defaultdict(lambda: [0, 0])
        for m, val in zip(models[train_idx], train_y):
            model_counts[m][val] += 1
        for vi in val_idx:
            m = models[vi]
            n0, n1 = model_counts[m]
            prob = (n1 + 1.0 * global_p) / (n0 + n1 + 1.0)
            y_prob_prior[vi] = prob

    y_pred_prior = (y_prob_prior >= 0.5).astype(int)
    auc_prior = roc_auc_score(y, y_prob_prior) if len(np.unique(y_prob_prior)) > 1 else 0.5
    acc_prior = accuracy_score(y, y_pred_prior)
    ci_prior_acc = bootstrap_ci(y, y_pred_prior, accuracy_score)
    ci_prior_auc = bootstrap_ci(y, y_prob_prior, roc_auc_score)
    print(f"Base Prior Alone (Grouped-CV {n_splits}-fold):")
    print(f"  Accuracy: {acc_prior:.4f} (95% CI: [{ci_prior_acc[0]:.4f}, {ci_prior_acc[1]:.4f}])")
    print(f"  AUC:      {auc_prior:.4f} (95% CI: [{ci_prior_auc[0]:.4f}, {ci_prior_auc[1]:.4f}])")
    print(f"  Pred True Rate: {np.mean(y_pred_prior):.3f}")

    # 2. Extract features
    all_feats = [extract_metadata_features(r[prob_key]) for r in valid_rows]
    feature_names = list(all_feats[0].keys())

    # Transform prior to log-odds
    prior_clipped = np.clip(y_prob_prior, 0.05, 0.95)
    log_odds_prior = np.log(prior_clipped / (1.0 - prior_clipped)).reshape(-1, 1)

    # 3. Test individual features + prior
    print(f"\nEvaluating individual features added to prior log-odds:")
    selected_features = []
    best_auc = auc_prior

    for feat_name in feature_names:
        feat_vals = np.array([f[feat_name] for f in all_feats]).reshape(-1, 1)
        X_test = np.hstack([log_odds_prior, feat_vals])

        y_prob_cv = np.zeros(n_samples)
        for train_idx, val_idx in gkf.split(valid_rows, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_test[train_idx])
            X_va = scaler.transform(X_test[val_idx])

            clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs")
            clf.fit(X_tr, y[train_idx])
            y_prob_cv[val_idx] = clf.predict_proba(X_va)[:, 1]

        auc_f = roc_auc_score(y, y_prob_cv) if len(np.unique(y_prob_cv)) > 1 else 0.5
        acc_f = accuracy_score(y, (y_prob_cv >= 0.5).astype(int))
        delta_auc = auc_f - auc_prior
        status = "KEPT" if delta_auc > 0.001 else "DROPPED"
        print(f"  + {feat_name:15s}: AUC = {auc_f:.4f} (delta = {delta_auc:+.4f}), Acc = {acc_f:.4f} -> {status}")
        if delta_auc > 0.001:
            selected_features.append(feat_name)

    # 4. Joint model with selected features
    if selected_features:
        print(f"\nSelected features for joint model: {selected_features}")
        feat_mat = np.array([[f[name] for name in selected_features] for f in all_feats])
        X_joint = np.hstack([log_odds_prior, feat_mat])

        y_prob_joint = np.zeros(n_samples)
        for train_idx, val_idx in gkf.split(valid_rows, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_joint[train_idx])
            X_va = scaler.transform(X_joint[val_idx])

            clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs")
            clf.fit(X_tr, y[train_idx])
            y_prob_joint[val_idx] = clf.predict_proba(X_va)[:, 1]

        auc_joint = roc_auc_score(y, y_prob_joint) if len(np.unique(y_prob_joint)) > 1 else 0.5
        acc_joint = accuracy_score(y, (y_prob_joint >= 0.5).astype(int))
        ci_joint_acc = bootstrap_ci(y, (y_prob_joint >= 0.5).astype(int), accuracy_score)
        ci_joint_auc = bootstrap_ci(y, y_prob_joint, roc_auc_score)
        print(f"\nJoint Model (Prior + {len(selected_features)} Selected Features):")
        print(f"  Accuracy: {acc_joint:.4f} (95% CI: [{ci_joint_acc[0]:.4f}, {ci_joint_acc[1]:.4f}])")
        print(f"  AUC:      {auc_joint:.4f} (95% CI: [{ci_joint_auc[0]:.4f}, {ci_joint_auc[1]:.4f}])")
        print(f"  Pred True Rate: {np.mean(y_prob_joint >= 0.5):.3f}")
    else:
        print(f"\nNo features improved grouped-CV AUC over the prior floor ({auc_prior:.4f}). Retaining prior alone.")

def main() -> None:
    # 1. val-sample (24 rows, 8 problems)
    cases_path = DATA_DIR / "val-sample" / "input" / "cases.jsonl"
    labels_path = DATA_DIR / "val-sample" / "reference" / "labels.jsonl"
    val_cases = [json.loads(l) for l in cases_path.open(encoding="utf-8")]
    val_labels = {json.loads(l)["id"]: json.loads(l)["is_robust"] for l in labels_path.open(encoding="utf-8")}
    for c in val_cases:
        c["is_robust"] = val_labels[c["id"]]
    run_step2("val-sample Codabench", val_cases, "problem", "model_id", "is_robust")

    # 2. train-main-v2 (80 labeled rows, 33 problems)
    train_path = DATA_DIR / "train-main-v2.jsonl"
    train_v2 = [json.loads(l) for l in train_path.open(encoding="utf-8")]
    run_step2("train-main-v2", train_v2, "problem", "model_id", "is_robust")

if __name__ == "__main__":
    main()
