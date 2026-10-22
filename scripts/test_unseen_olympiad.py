"""Test Contrastive White-Box Robustness on Unseen Olympiad Math Problems (AIME / IMO).

Evaluates:
1. Genuine unseen Olympiad problems across Algebra, Number Theory, Combinatorics, and Geometry.
2. Meaning-preserving perturbation generation (x -> x').
3. 27 structural & symbolic complexity features.
4. Latent representation drift and cosine similarity (if GPU / model weights present).
5. Calibrated robustness decisions: True (robust reasoning) vs False (fragile).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solutions" / "contrastive-whitebox"
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from contrastive.decision import predict_contrastive_robustness
from contrastive.perturbation_engine import generate_contrastive_variant
from contrastive.symbolic_extractor import extract_feature_dataframe, extract_symbolic_features


# Curated set of genuine AIME / Olympiad problems across all 4 math disciplines
UNSEEN_OLYMPIAD_PROBLEMS = [
    {
        "id": "AIME-Algebra",
        "domain": "Algebra / Polynomials",
        "problem": (
            "Let $P(x)$ be a polynomial of degree 2024 such that $P(k) = \\frac{1}{k}$ "
            "for each $k = 1, 2, \\dots, 2025$. Find the value of $2026 \\cdot P(2026)$."
        ),
    },
    {
        "id": "AIME-NumberTheory",
        "domain": "Number Theory / Modular Powers",
        "problem": (
            "Find the least positive integer $n$ such that $2^n + 3^n$ is a multiple of $5^7$."
        ),
    },
    {
        "id": "AIME-Combinatorics",
        "domain": "Combinatorics / Extremal Sets",
        "problem": (
            "A subset $S$ of $\\{1, 2, \\dots, 100\\}$ is called sum-free if no two elements "
            "of $S$ sum to an element of $S$. What is the maximum possible number of elements "
            "in a sum-free subset $S$?"
        ),
    },
    {
        "id": "AIME-Geometry",
        "domain": "Geometry / Incircle & Similar Triangles",
        "problem": (
            "In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Let $I$ be the incenter "
            "of $\\triangle ABC$. The line passing through $I$ parallel to $BC$ intersects "
            "$AB$ at $P$ and $AC$ at $Q$. Determine the length of the segment $PQ$."
        ),
    },
]


def evaluate_unseen_olympiad():
    print("=" * 80)
    print(" EVALUATING CONTRASTIVE WHITE-BOX DETECTOR ON UNSEEN OLYMPIAD PROBLEMS")
    print("=" * 80)

    problems = [item["problem"] for item in UNSEEN_OLYMPIAD_PROBLEMS]

    # 1. Perturbation Engine Demo
    print("\n[1] Meaning-Preserving Equivalence Perturbations (x -> x'):")
    print("-" * 80)
    for item in UNSEEN_OLYMPIAD_PROBLEMS:
        orig = item["problem"]
        variant = generate_contrastive_variant(orig)
        item["variant"] = variant
        print(f"[{item['id']} - {item['domain']}]")
        print(f"  Original:  {orig}")
        print(f"  Variant:   {variant}")
        print()

    # 2. Symbolic Structural Feature Profiling
    print("[2] Symbolic Structural Complexity Profiling (Selected Features):")
    print("-" * 80)
    df_feats = extract_feature_dataframe(problems)
    summary_cols = [
        "char_count",
        "math_char_ratio",
        "latex_command_count",
        "fraction_count",
        "distinct_variable_count",
        "max_number_magnitude",
        "has_algebra_keyword",
        "has_number_theory_keyword",
        "has_combinatorics_keyword",
        "has_geometry_keyword",
    ]
    df_subset = df_feats[summary_cols].copy()
    df_subset.insert(0, "Problem_ID", [item["id"] for item in UNSEEN_OLYMPIAD_PROBLEMS])
    print(df_subset.to_string(index=False))

    # 3. Model Predictions via the Contrastive Solution Entrypoint
    print("\n[3] Model Robustness Predictions:")
    print("-" * 80)
    predictions = predict_contrastive_robustness(
        model_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        reasoning_effort="default",
        problems=problems,
    )

    for item, pred in zip(UNSEEN_OLYMPIAD_PROBLEMS, predictions):
        status = "ROBUST (Stable under perturbation)" if pred else "FRAGILE (Sensitive to perturbation)"
        print(f"  * {item['id']:<20} [{item['domain']:<32}] => {status}")

    print("\n" + "=" * 80)
    print(f" Total Problems Evaluated: {len(problems)}")
    print(f" Predicted Robust:         {predictions.count(True)} / {len(problems)} ({predictions.count(True)/len(problems):.1%})")
    print(f" Predicted Fragile:        {predictions.count(False)} / {len(problems)} ({predictions.count(False)/len(problems):.1%})")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_unseen_olympiad()
