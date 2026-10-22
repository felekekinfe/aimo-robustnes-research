"""Train and evaluate symbolic robustness classifiers using leakage-safe CV.

This script:
1. Loads training datasets (aimo-interp/train-main-v2 and/or aimo-interp/augmented-sample-math-agg).
2. Extracts the 27 symbolic and structural features.
3. Runs StratifiedGroupKFold cross-validation grouped by problem to prevent data leakage.
4. Evaluates multiple classifiers against the majority-class baseline.
5. Computes balanced accuracy, macro-F1, ROC-AUC, and feature importances.
6. Serializes the best model artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aimo_symbolic.features import SYMBOLIC_FEATURE_NAMES, extract_symbolic_features


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load dataset, extract symbolic features, labels, and grouping identifiers."""
    data_dir = ROOT / "data"

    if dataset_name == "train-main-v2":
        path = data_dir / "train-main-v2.jsonl"
        if not path.is_file():
            print(f"Data file {path} not found. Fetching from Hugging Face...")
            import subprocess
            subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_training_data.py")], check=True)
        with path.open("r", encoding="utf-8") as f:
            raw_rows = [json.loads(line) for line in f if line.strip()]
        valid_rows = [r for r in raw_rows if isinstance(r.get("is_robust"), bool)]
        problems = [r["problem"] for r in valid_rows]
        labels = np.array([r["is_robust"] for r in valid_rows], dtype=bool)
        model_ids = [r.get("model_id", "default") for r in valid_rows]
        efforts = [r.get("reasoning_effort", "default") for r in valid_rows]
        groups = np.array([r["problem"][:100] for r in valid_rows])
        rows = [extract_symbolic_features(p, m, e) for p, m, e in zip(problems, model_ids, efforts)]
        features_df = pd.DataFrame(rows, columns=list(SYMBOLIC_FEATURE_NAMES))

    elif dataset_name == "augmented-sample-math-agg":
        path = data_dir / "augmented-sample-math-agg.jsonl"
        if not path.is_file():
            print(f"Data file {path} not found. Fetching from Hugging Face...")
            import subprocess
            subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_training_data.py")], check=True)
        with path.open("r", encoding="utf-8") as f:
            raw_rows = [json.loads(line) for line in f if line.strip()]
        valid_rows = [r for r in raw_rows if isinstance(r.get("model_is_robust"), bool)]
        problems = [r["original_problem"] for r in valid_rows]
        labels = np.array([r["model_is_robust"] for r in valid_rows], dtype=bool)
        model_ids = [r.get("model_id", "default") for r in valid_rows]
        groups = np.array([r.get("problem_id", r["original_problem"][:100]) for r in valid_rows])
        rows = [extract_symbolic_features(p, m, "default") for p, m in zip(problems, model_ids)]
        features_df = pd.DataFrame(rows, columns=list(SYMBOLIC_FEATURE_NAMES))

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return features_df, labels, groups, problems


def evaluate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, dict]:
    """Run StratifiedGroupKFold cross-validation across model candidates."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    candidate_models = {
        "majority_dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=5,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=5,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=100,
            random_state=seed,
        ),
    }

    results = {}

    for name, model in candidate_models.items():
        oof_preds = np.zeros(len(y), dtype=bool)
        oof_probs = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in sgkf.split(X, y, groups):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            oof_preds[val_idx] = preds

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val)
                # Probability of positive class (True)
                if probs.shape[1] == 2:
                    oof_probs[val_idx] = probs[:, 1]
                else:
                    oof_probs[val_idx] = probs[:, 0]

        acc = accuracy_score(y, oof_preds)
        bal_acc = balanced_accuracy_score(y, oof_preds)
        f1 = f1_score(y, oof_preds, average="macro")
        cm = confusion_matrix(y, oof_preds).tolist()

        try:
            auc = roc_auc_score(y, oof_probs) if len(np.unique(oof_probs)) > 1 else 0.5
        except Exception:
            auc = 0.5

        results[name] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm,
        }

    return results


def analyze_feature_importances(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> pd.DataFrame:
    """Fit a tree ensemble on full features and rank feature importances."""
    rf = ExtraTreesClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=seed)
    rf.fit(X, y)

    importances = rf.feature_importances_
    df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["train-main-v2", "augmented-sample-math-agg"],
        default="augmented-sample-math-agg",
    )
    parser.add_argument("--export-artifact", action="store_true")
    args = parser.parse_args()

    print(f"=== Evaluating Symbolic Robustness Predictors on '{args.dataset}' ===")
    X, y, groups, problems = load_dataset(args.dataset)
    print(f"Total instances: {len(y)}, Groups: {len(np.unique(groups))}")
    print(f"Labels: True={np.sum(y)}, False={np.sum(~y)} (Positive rate: {np.mean(y):.2%})\n")

    results = evaluate_models(X, y, groups)
    print("--- 5-Fold Stratified Group CV Results ---")
    print(f"{'Model':<25} | {'Acc':<8} | {'Bal Acc':<8} | {'Macro F1':<8} | {'ROC-AUC':<8}")
    print("-" * 65)
    for name, m in sorted(results.items(), key=lambda item: item[1]["balanced_accuracy"], reverse=True):
        print(f"{name:<25} | {m['accuracy']:<8.4f} | {m['balanced_accuracy']:<8.4f} | {m['macro_f1']:<8.4f} | {m['roc_auc']:<8.4f}")

    print("\n--- Top 10 Most Predictive Symbolic Features ---")
    feat_df = analyze_feature_importances(X, y)
    for idx, row in feat_df.head(10).iterrows():
        print(f"  {row['feature']:<28}: {row['importance']:.4f}")

    if args.export_artifact:
        artifacts_dir = ROOT / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        best_model = RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=42)
        best_model.fit(X, y)
        artifact_path = artifacts_dir / f"symbolic_model_{args.dataset.replace('-', '_')}.joblib"
        joblib.dump(
            {
                "model": best_model,
                "feature_names": SYMBOLIC_FEATURE_NAMES,
                "cv_results": results,
                "dataset": args.dataset,
            },
            artifact_path,
        )
        print(f"\nSaved trained artifact to {artifact_path}")


if __name__ == "__main__":
    main()
