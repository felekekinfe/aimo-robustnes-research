"""STEP 1: Compute baselines and shrunk priors with GroupKFold cross-validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_bootstraps: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        # Check if only one class sampled
        if len(np.unique(y_true[idx])) < 2 and metric_fn == roc_auc_score:
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if not scores:
        return 0.0, 0.0
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def evaluate_dataset(rows: list[dict], name: str, prob_key: str, model_key: str, delta_key: str | None, label_key: str):
    print(f"\n=======================================================")
    print(f" BASELINE EVALUATION FOR: {name} (N={len(rows)})")
    print(f"=======================================================")

    # Filter out None labels
    valid_rows = [r for r in rows if r.get(label_key) is not None]
    y = np.array([1 if r[label_key] is True else 0 for r in valid_rows])
    groups = np.array([r[prob_key].strip() for r in valid_rows])
    models = np.array([r[model_key] for r in valid_rows])
    deltas = np.array([str(r.get(delta_key, 'unknown')) if delta_key else 'default' for r in valid_rows])

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_samples = len(y)
    print(f"Samples: {n_samples}, Distinct Problem Groups: {n_groups}")
    print(f"Overall Label Distribution: {Counter(y)} (Robust/1: {np.mean(y):.3f})")

    # 1. Global Majority Class Baseline
    majority_class = Counter(y).most_common(1)[0][0]
    y_pred_majority = np.full_like(y, majority_class)
    acc_maj = accuracy_score(y, y_pred_majority)
    ci_maj_acc = bootstrap_ci(y, y_pred_majority, accuracy_score)
    print(f"\n1. Global Majority Class ({majority_class}):")
    print(f"   Accuracy: {acc_maj:.4f} (95% CI: [{ci_maj_acc[0]:.4f}, {ci_maj_acc[1]:.4f}])")
    print(f"   AUC: 0.5000 (constant predictor)")

    # GroupKFold setup
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    # 2. Per-Delta Majority with GroupKFold
    if delta_key:
        y_pred_delta = np.zeros(n_samples)
        y_prob_delta = np.zeros(n_samples)

        for train_idx, val_idx in gkf.split(valid_rows, y, groups):
            # Fit on train_idx
            train_deltas = deltas[train_idx]
            train_y = y[train_idx]
            delta_counts = defaultdict(list)
            for d, val in zip(train_deltas, train_y):
                delta_counts[d].append(val)

            global_train_prior = np.mean(train_y)
            for vi in val_idx:
                d = deltas[vi]
                if d in delta_counts and len(delta_counts[d]) > 0:
                    prob = np.mean(delta_counts[d])
                else:
                    prob = global_train_prior
                y_prob_delta[vi] = prob
                y_pred_delta[vi] = 1 if prob >= 0.5 else 0

        acc_delta = accuracy_score(y, y_pred_delta)
        auc_delta = roc_auc_score(y, y_prob_delta) if len(np.unique(y_prob_delta)) > 1 else 0.5
        ci_delta_acc = bootstrap_ci(y, y_pred_delta, accuracy_score)
        ci_delta_auc = bootstrap_ci(y, y_prob_delta, roc_auc_score)
        print(f"\n2. Per-Delta Majority (Grouped-CV {n_splits}-fold):")
        print(f"   Accuracy: {acc_delta:.4f} (95% CI: [{ci_delta_acc[0]:.4f}, {ci_delta_acc[1]:.4f}])")
        print(f"   AUC:      {auc_delta:.4f} (95% CI: [{ci_delta_auc[0]:.4f}, {ci_delta_auc[1]:.4f}])")
        print(f"   Predicted True Rate: {np.mean(y_pred_delta):.3f}")

    # 3. Per-(Model, Delta) Majority / Empirical-Bayes Shrunk Prior
    for k_smooth in [0.0, 1.0, 3.0]:
        y_pred_shrunk = np.zeros(n_samples)
        y_prob_shrunk = np.zeros(n_samples)

        for train_idx, val_idx in gkf.split(valid_rows, y, groups):
            train_y = y[train_idx]
            global_prior = np.mean(train_y)

            # Cell counts: (model, delta)
            cell_counts = defaultdict(lambda: [0, 0]) # [n_0, n_1]
            model_counts = defaultdict(lambda: [0, 0])
            for mi, di, yi in zip(models[train_idx], deltas[train_idx], train_y):
                cell_counts[(mi, di)][yi] += 1
                model_counts[mi][yi] += 1

            for vi in val_idx:
                mi = models[vi]
                di = deltas[vi]
                n_0, n_1 = cell_counts[(mi, di)]
                n_cell = n_0 + n_1

                if k_smooth == 0.0:
                    # Unshrunk MLE
                    if n_cell > 0:
                        prob = n_1 / n_cell
                    else:
                        m_n0, m_n1 = model_counts[mi]
                        prob = m_n1 / (m_n0 + m_n1) if (m_n0 + m_n1) > 0 else global_prior
                else:
                    # Shrunk empirical Bayes (m-estimate / add-k smoothing toward model prior or global prior)
                    m_n0, m_n1 = model_counts[mi]
                    m_prior = (m_n1 + k_smooth * global_prior) / (m_n0 + m_n1 + k_smooth) if (m_n0 + m_n1) > 0 else global_prior
                    prob = (n_1 + k_smooth * m_prior) / (n_cell + k_smooth)

                y_prob_shrunk[vi] = prob
                y_pred_shrunk[vi] = 1 if prob >= 0.5 else 0

        acc_shrunk = accuracy_score(y, y_pred_shrunk)
        auc_shrunk = roc_auc_score(y, y_prob_shrunk) if len(np.unique(y_prob_shrunk)) > 1 else 0.5
        ci_acc = bootstrap_ci(y, y_pred_shrunk, accuracy_score)
        ci_auc = bootstrap_ci(y, y_prob_shrunk, roc_auc_score)
        desc = "MLE (unshrunk)" if k_smooth == 0.0 else f"Shrunk Prior (k={k_smooth})"
        print(f"\n3. Per-(Model, Delta) {desc} (Grouped-CV {n_splits}-fold):")
        print(f"   Accuracy: {acc_shrunk:.4f} (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}])")
        print(f"   AUC:      {auc_shrunk:.4f} (95% CI: [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}])")
        print(f"   Predicted True Rate: {np.mean(y_pred_shrunk):.3f}")

def main() -> None:
    # Evaluate val-sample-raw
    val_raw_path = DATA_DIR / "val-sample-raw.jsonl"
    if val_raw_path.is_file():
        val_raw = [json.loads(l) for l in val_raw_path.open(encoding="utf-8")]
        evaluate_dataset(val_raw, "val-sample raw (HF)", "original_problem", "model_id", "permutation_type", "model_is_robust")

    # Evaluate val-sample Codabench cases
    cases_path = DATA_DIR / "val-sample" / "input" / "cases.jsonl"
    labels_path = DATA_DIR / "val-sample" / "reference" / "labels.jsonl"
    val_cases = [json.loads(l) for l in cases_path.open(encoding="utf-8")]
    val_labels = {json.loads(l)["id"]: json.loads(l)["is_robust"] for l in labels_path.open(encoding="utf-8")}
    for c in val_cases:
        c["is_robust"] = val_labels[c["id"]]
    evaluate_dataset(val_cases, "val-sample Codabench", "problem", "model_id", None, "is_robust")

    # Evaluate augmented-sample-math-agg
    aug_path = DATA_DIR / "augmented-sample-math-agg.jsonl"
    if aug_path.is_file():
        aug = [json.loads(l) for l in aug_path.open(encoding="utf-8")]
        evaluate_dataset(aug, "augmented-sample-math-agg", "original_problem", "model_id", "permutation_type", "model_is_robust")

    # Evaluate train-main-v2
    train_path = DATA_DIR / "train-main-v2.jsonl"
    if train_path.is_file():
        train_v2 = [json.loads(l) for l in train_path.open(encoding="utf-8")]
        evaluate_dataset(train_v2, "train-main-v2 (labeled rows)", "problem", "model_id", None, "is_robust")

if __name__ == "__main__":
    main()
