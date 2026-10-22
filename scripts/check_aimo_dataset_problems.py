"""Check Robustness and Fragility on Real AIMO Competition Dataset Problems.

Loads authentic problems from:
1. data/val-sample (the official 24-case validation benchmark with reference labels)
2. data/train-main-v2 (official multi-model training problems)

Analyzes why each real problem was labeled ROBUST vs FRAGILE, what perturbation
was applied, and compares predictions against ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solutions" / "contrastive-whitebox"
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from contrastive.decision import predict_contrastive_robustness
from contrastive.perturbation_engine import generate_contrastive_variant
from contrastive.symbolic_extractor import extract_symbolic_features


def load_val_sample_data():
    cases_path = ROOT / "data" / "val-sample" / "input" / "cases.jsonl"
    labels_path = ROOT / "data" / "val-sample" / "reference" / "labels.jsonl"

    if not cases_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("val-sample dataset files not found. Run scripts/import_hf_dataset.py first.")

    with cases_path.open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    with labels_path.open("r", encoding="utf-8") as f:
        labels = {json.loads(line)["id"]: json.loads(line)["is_robust"] for line in f if line.strip()}

    return cases, labels


def analyze_real_aimo_problems(limit: int = 10):
    print("=" * 85)
    print(" EVALUATING REAL AIMO DATASET PROBLEMS: ROBUST VS FRAGILE BREAKDOWN")
    print("=" * 85)

    cases, ground_truth = load_val_sample_data()
    print(f"Total problems available in val-sample: {len(cases)}")
    print(f"Displaying in-depth breakdown for first {min(limit, len(cases))} problems:\n")

    problems = [c["problem"] for c in cases]
    model_ids = [c["model_id"] for c in cases]

    # Batch cases by (model_id, reasoning_effort) identically to the official ingestion program
    batches: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for idx, case in enumerate(cases):
        effort = case.get("reasoning_effort", "low")
        batches.setdefault((case["model_id"], effort), []).append((idx, case))

    predictions = [False] * len(cases)
    for (model_id, effort), batch in batches.items():
        batch_problems = [case["problem"] for _, case in batch]
        batch_preds = predict_contrastive_robustness(
            model_id=model_id,
            reasoning_effort=effort,
            problems=batch_problems,
        )
        for (idx, _), pred in zip(batch, batch_preds):
            predictions[idx] = pred

    correct_count = 0
    total_evaluated = min(limit, len(cases))

    for i in range(total_evaluated):
        c = cases[i]
        cid = c["id"]
        model = c["model_id"]
        prob_text = c["problem"]
        gt = ground_truth.get(cid)
        pred = predictions[i]
        is_match = (gt == pred)
        if is_match:
            correct_count += 1

        variant = generate_contrastive_variant(prob_text)
        feats = extract_symbolic_features(prob_text)

        domain = "General Math"
        if feats["has_geometry_keyword"] > 0:
            domain = "Geometry"
        elif feats["has_number_theory_keyword"] > 0:
            domain = "Number Theory"
        elif feats["has_combinatorics_keyword"] > 0:
            domain = "Combinatorics"
        elif feats["has_algebra_keyword"] > 0:
            domain = "Algebra"

        match_badge = "[CORRECT PREDICTION]" if is_match else "[MISMATCH]"
        gt_label = "ROBUST (True)" if gt else "FRAGILE (False)"
        pred_label = "ROBUST (True)" if pred else "FRAGILE (False)"

        print(f"Problem #{i+1} [{cid[:14]}...] | Model: {model} | Domain: {domain}")
        print(f"  Ground Truth:  {gt_label:<18} | Prediction: {pred_label:<18} {match_badge}")
        print(f"  Original Text: {prob_text[:110]}...")
        print(f"  Perturbed Var: {variant[:110]}...")
        print(f"  Complexity:    Chars={int(feats['char_count'])}, MathRatio={feats['math_char_ratio']:.2f}, Equations={int(feats['equal_sign_count'])}, Vars={int(feats['distinct_variable_count'])}")
        print("-" * 85)

    # Full Dataset Summary Statistics
    all_gt = [ground_truth[c["id"]] for c in cases]
    full_correct = sum(1 for p, g in zip(predictions, all_gt) if p == g)
    accuracy = full_correct / len(cases)

    # Confusion matrix elements
    tp = sum(1 for p, g in zip(predictions, all_gt) if p and g)
    tn = sum(1 for p, g in zip(predictions, all_gt) if not p and not g)
    fp = sum(1 for p, g in zip(predictions, all_gt) if p and not g)
    fn = sum(1 for p, g in zip(predictions, all_gt) if not p and g)

    print("\n" + "=" * 85)
    print(" COMPLETE AIMO DATASET (24/24 CASES) EVALUATION SUMMARY")
    print("=" * 85)
    print(f" Overall Accuracy:      {accuracy:.2%} ({full_correct} / {len(cases)} correct)")
    print(f" Ground Truth Balance:  Robust: {sum(all_gt)} ({sum(all_gt)/len(cases):.1%}), Fragile: {len(all_gt)-sum(all_gt)} ({(len(all_gt)-sum(all_gt))/len(cases):.1%})")
    print(f" Predictions Balance:   Robust: {sum(predictions)} ({sum(predictions)/len(cases):.1%}), Fragile: {len(predictions)-sum(predictions)} ({(len(predictions)-sum(predictions))/len(cases):.1%})")
    print(f" Confusion Matrix:      TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("=" * 85)


if __name__ == "__main__":
    analyze_real_aimo_problems(limit=10)
