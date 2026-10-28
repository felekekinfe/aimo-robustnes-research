#!/usr/bin/env python3
"""
Audits and calibrates SCSP priors & decision thresholds on train-main-v2.jsonl
using GroupKFold by problem.
"""

from __future__ import annotations
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from scsp_core import (
    compute_scsp_robustness_logit,
    extract_structural_surface_stats,
    get_empirical_model_effort_prior,
)


def pure_python_accuracy(y_true, y_pred) -> float:
    return sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(1, len(y_true))


def load_training_rows(jsonl_path: str):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prob = r.get("problem") or r.get("problem_text") or ""
            # Deterministic problem hash for GroupKFold across Python processes
            pid = r.get("problem_id") or r.get("id") or hashlib.md5(prob.strip().encode("utf-8")).hexdigest()[:12]
            mid = r.get("model_id") or r.get("model") or "unknown"
            eff = (r.get("reasoning_effort") or r.get("effort") or "default").lower()
            
            raw_val = r["robust"] if "robust" in r else r.get("is_robust", r.get("label"))
            label = bool(raw_val) if raw_val is not None else False
            
            rows.append({
                "problem_id": pid,
                "problem": prob,
                "model_id": mid,
                "reasoning_effort": eff,
                "label": int(label),
            })
    return rows


def run_calibration_audit(jsonl_path: str = "data/train-main-v2.jsonl"):
    path = Path(jsonl_path)
    if not path.exists():
        # Search workspace for train-main-v2.jsonl
        matches = list(ROOT.rglob("train-main-v2.jsonl"))
        if not matches:
            raise FileNotFoundError("Could not find train-main-v2.jsonl in workspace.")
        path = matches[0]

    rows = load_training_rows(str(path))
    print(f"Loaded {len(rows)} instances from {path}")

    # 1. Empirical Ground-Truth Breakdown by (model_id, reasoning_effort)
    slice_counts = defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["model_id"].lower(), r["reasoning_effort"])
        slice_counts[k][1] += 1
        slice_counts[k][0] += r["label"]

    print("\n=== EMPIRICAL GROUND-TRUTH ROBUSTNESS BY (MODEL, EFFORT) ===")
    empirical_priors = {}
    for (mid, eff), (pos, tot) in sorted(slice_counts.items()):
        # Laplace-smoothed empirical probability -> log-odds
        p_emp = (pos + 2.0) / (tot + 4.0)
        log_odds = math.log(p_emp / (1.0 - p_emp))
        empirical_priors[f"prior_{mid}_{eff}"] = round(log_odds, 4)
        print(f"  {mid:42s} | effort={eff:8s} | {pos:3d}/{tot:3d} ({pos/tot:6.1%}) | log_odds={log_odds:+.3f}")

    # 2. GroupKFold Out-of-Fold Validation (Zero Leakage across problem_id)
    groups = np.array([r["problem_id"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    oof_preds = np.zeros(len(rows), dtype=int)

    n_splits = 5
    if HAS_SKLEARN:
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(rows, labels, groups=groups))
    else:
        # Pure-python deterministic GroupKFold
        unique_groups = sorted(list(set(groups)))
        group_to_fold = {g: int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16) % n_splits for g in unique_groups}
        splits = []
        for f in range(n_splits):
            val_idx = np.array([i for i, g in enumerate(groups) if group_to_fold[g] == f])
            train_idx = np.array([i for i, g in enumerate(groups) if group_to_fold[g] != f])
            splits.append((train_idx, val_idx))

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Compute fold-specific priors strictly on tr_idx
        fold_counts = defaultdict(lambda: [0, 0])
        for idx in tr_idx:
            r = rows[idx]
            k = (r["model_id"].lower(), r["reasoning_effort"])
            fold_counts[k][1] += 1
            fold_counts[k][0] += r["label"]

        fold_weights = {}
        for (mid, eff), (pos, tot) in fold_counts.items():
            p_emp = (pos + 2.0) / (tot + 4.0)
            fold_weights[f"prior_{mid}_{eff}"] = math.log(p_emp / (1.0 - p_emp))

        for idx in va_idx:
            r = rows[idx]
            _, pred = compute_scsp_robustness_logit(
                problem_text=r["problem"],
                model_id=r["model_id"],
                reasoning_effort=r["reasoning_effort"],
                whitebox_features=None,
                custom_weights=fold_weights,
            )
            oof_preds[idx] = int(pred)

    acc_fn = accuracy_score if HAS_SKLEARN else pure_python_accuracy
    oof_acc = acc_fn(labels, oof_preds)
    pred_true_rate = oof_preds.mean()
    gt_true_rate = labels.mean()
    print("\n=== 5-FOLD GROUPKFOLD OUT-OF-FOLD DIAGNOSTIC ===")
    print(f"  OOF Accuracy     : {oof_acc:.4f} ({oof_acc*100:.2f}%)")
    print(f"  Ground-Truth True: {gt_true_rate:.1%} | Predicted True: {pred_true_rate:.1%}")

    # Check effort='low' specifically
    low_mask = np.array([r["reasoning_effort"] == "low" for r in rows])
    if low_mask.any():
        low_acc = acc_fn(labels[low_mask], oof_preds[low_mask])
        print(
            f"  Effort='low' Acc : {low_acc:.4f} | GT True: {labels[low_mask].mean():.1%} "
            f"| Pred True: {oof_preds[low_mask].mean():.1%}"
        )

    # Check effort='high' specifically
    high_mask = np.array([r["reasoning_effort"] == "high" for r in rows])
    if high_mask.any():
        high_acc = acc_fn(labels[high_mask], oof_preds[high_mask])
        print(
            f"  Effort='high' Acc: {high_acc:.4f} | GT True: {labels[high_mask].mean():.1%} "
            f"| Pred True: {oof_preds[high_mask].mean():.1%}"
        )

    # Check effort='medium' specifically
    med_mask = np.array([r["reasoning_effort"] == "medium" for r in rows])
    if med_mask.any():
        med_acc = acc_fn(labels[med_mask], oof_preds[med_mask])
        print(
            f"  Effort='medium' Acc: {med_acc:.4f} | GT True: {labels[med_mask].mean():.1%} "
            f"| Pred True: {oof_preds[med_mask].mean():.1%}"
        )

    print("\n=== COPY THESE CALIBRATED PRIORS INTO decision.py & scsp_core.py ===")
    print(json.dumps(empirical_priors, indent=2))
    return empirical_priors


if __name__ == "__main__":
    run_calibration_audit()
