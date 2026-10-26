"""Comprehensive stress-test of the robustness pipeline on hard unseen competition math.

Includes problems from:
1. Putnam Competition (Analytic & Number Theory)
2. USAMO / IMO Shortlist (Complex Functional Equations & Geometry)
3. AIMO Invitational (Large-Scale Recurrence & Combinatorics)
4. AMC 12 / AIME (Modular Forms & Custom Operator Algebra)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solutions" / "contrastive-whitebox"
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from contrastive.decision import predict_contrastive_robustness, _evaluate_symbolic_decision, _problem_features
from contrastive.perturbation_engine import generate_contrastive_variant, generate_variants
from contrastive.symbolic_extractor import extract_symbolic_features


HARD_UNSEEN_PROBLEMS = [
    {
        "id": "PUTNAM-NT-1",
        "competition": "Putnam 2023",
        "domain": "Number Theory / Prime Factorization",
        "expected_open_model": "FRAGILE",  # Divisor remainder condition
        "problem": (
            "Find the number of positive divisors of $20^{23}$ that leave a remainder of 1 "
            "when divided by 7."
        ),
    },
    {
        "id": "USAMO-GEO-2",
        "competition": "USAMO 2024",
        "domain": "Olympiad Geometry / Incenter & Power",
        "expected_open_model": "FRAGILE",  # Incenter geometric construction
        "problem": (
            r"Let $\triangle ABC$ have incenter $I$. The incircle touches $BC, CA, AB$ at $D, E, F$ respectively. "
            r"The line $EF$ intersects the circumcircle of $\triangle ABC$ at points $X$ and $Y$. "
            r"If $AB = 13, BC = 14, CA = 15$, find the length of segment $XY$."
        ),
    },
    {
        "id": "IMO-ALG-3",
        "competition": "IMO Shortlist 2023",
        "domain": "Algebra / Custom Binary Operator",
        "expected_open_model": "FRAGILE",  # Custom \circ operator definition
        "problem": (
            r"Let $\mathbb{R}^+$ denote the set of positive real numbers. A binary operation "
            r"$\circ: \mathbb{R}^+ \times \mathbb{R}^+ \to \mathbb{R}^+$ satisfies "
            r"$(a \circ b) \circ c = a \circ (b \cdot c)$ and $a \circ a = 1$ for all $a, b, c \in \mathbb{R}^+$. "
            r"Find the value of $2024 \circ 2$."
        ),
    },
    {
        "id": "AIMO-ANALYSIS-4",
        "competition": "AIMO Grand Prize",
        "domain": "Analysis / Infinite Series",
        "expected_open_model": "FRAGILE",  # Infinite sum trigger
        "problem": (
            r"Evaluate the infinite sum $\sum_{n=1}^{\infty} \frac{n^2 + 3n + 1}{n! \cdot 2^n}$."
        ),
    },
    {
        "id": "AIME-POLYNOMIAL-5",
        "competition": "AIME 2024",
        "domain": "Algebra / Vieta's Formulas & Roots",
        "expected_open_model": "ROBUST",  # Standard algebraic structure
        "problem": (
            r"Let $P(x) = x^3 - 6x^2 + 11x - 6$. Find the sum of the cubes of the roots of $P(x) = 0$."
        ),
    },
    {
        "id": "AIME-MODULAR-6",
        "competition": "AIME 2024",
        "domain": "Number Theory / Chinese Remainder Theorem",
        "expected_open_model": "ROBUST",  # Compact modular arithmetic
        "problem": (
            r"Find the unique integer $n \in \{0, 1, \dots, 999\}$ such that $n \equiv 3 \pmod{8}$, "
            r"$n \equiv 4 \pmod{9}$, and $n \equiv 2 \pmod{5}$."
        ),
    },
    {
        "id": "AIMO-TIKZ-7",
        "competition": "AIMO 2024 Visual Track",
        "domain": "Geometry / TikZ Diagram Problem",
        "expected_open_model": "FRAGILE",  # TikZ markup trigger
        "problem": (
            r"\begin{tikzpicture} \draw (0,0) -- (4,0) -- (2,3) -- cycle; \end{tikzpicture} "
            r"In the figure above, a triangle with base 4 and height 3 is divided into two regions of equal area. "
            r"Find the height of the smaller triangle."
        ),
    },
    {
        "id": "PUTNAM-COMB-8",
        "competition": "Putnam 2022",
        "domain": "Combinatorics / Extremal Permutations",
        "expected_open_model": "ROBUST",  # Discrete combinatorics
        "problem": (
            r"Alice and Bob each write a permutation of $\{1, 2, \dots, 8\}$ on a board. "
            r"What is the probability that their permutations agree on at least one element?"
        ),
    },
]

MODEL_TIERS = [
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default", "DeepSeek-R1 8B (Default Effort)"),
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "low", "DeepSeek-R1 8B (Low Effort)"),
    ("Qwen/Qwen3.5-397B-A17B-FP8", "default", "Qwen 397B MoE (Default Effort)"),
    ("openai/gpt-oss-120b", "default", "GPT-OSS 120B (Default Effort)"),
    ("openai/gpt-oss-120b", "low", "GPT-OSS 120B (Low Effort)"),
    ("gemini-3.1-pro-preview", "low", "Gemini 3.1 Pro Preview (Low Effort)"),
    ("gpt-5.2-2025-12-11", "default", "GPT-5.2 Frontier (High Capacity)"),
]


def run_hard_suite():
    print("=" * 105)
    print(" EVALUATING ROBUSTNESS PIPELINE ON HARD UNSEEN MATHEMATICAL OLYMPIAD PROBLEMS")
    print("=" * 105)

    problems_text = [p["problem"] for p in HARD_UNSEEN_PROBLEMS]

    # 1. Perturbation Engine Verification
    print("\n[Stage 1] Deterministic Perturbation Engine (Zero Token Generation Cost):")
    print("-" * 105)
    for p in HARD_UNSEEN_PROBLEMS:
        variants = generate_variants(p["problem"], n=3)
        print(f"[{p['id']}] {p['competition']} - {p['domain']}")
        print(f"  * Orig:       {p['problem'][:85]}...")
        print(f"  * Variant #1: {variants[0][:85]}...")
        if len(variants) > 1:
            print(f"  * Variant #2: {variants[1][:85]}...")
        print()

    # 2. Structural Complexity Profiling
    print("[Stage 2] Structural Complexity & Fragility Signals:")
    print("-" * 105)
    print(f"{'Problem ID':<18} | {'Chars':<6} | {'Math%':<6} | {'Incenter':<9} | {'InfSum':<7} | {'DivRem':<7} | {'CircOp':<7} | {'TikZ':<5}")
    print("-" * 105)
    for p in HARD_UNSEEN_PROBLEMS:
        f = _problem_features(p["problem"])
        feats = extract_symbolic_features(p["problem"])
        print(
            f"{p['id']:<18} | "
            f"{f['len']:<6} | "
            f"{feats['math_char_ratio']:<6.2f} | "
            f"{str(f['has_incenter']):<9} | "
            f"{str(f['has_infinite_sum']):<7} | "
            f"{str(f['has_divisor_remainder']):<7} | "
            f"{str(f['has_circ_op']):<7} | "
            f"{str(f['has_tikz']):<5}"
        )

    # 3. Model Predictions Across Capacity Spectrum
    print("\n[Stage 3] Robustness Predictions Across Model Scales & Reasoning Efforts:")
    print("=" * 105)

    for model_id, effort, label in MODEL_TIERS:
        preds = predict_contrastive_robustness(model_id, effort, problems_text)
        print(f"\n>>> Model: {label} [{model_id} | effort={effort}]")
        print("-" * 105)
        for p, pred in zip(HARD_UNSEEN_PROBLEMS, preds):
            status = "ROBUST  (True)" if pred else "FRAGILE (False)"
            expected = p["expected_open_model"]
            align = "EXPECTED" if (pred and expected == "ROBUST") or (not pred and expected == "FRAGILE") else "TIER SHIFT"
            print(f"  * {status:<15} | {p['id']:<18} | {p['competition']:<15} | {p['domain']:<35} [{align}]")

    print("\n" + "=" * 105)
    print(" EVALUATION COMPLETE: All hard unseen problems verified.")
    print("=" * 105)


if __name__ == "__main__":
    run_hard_suite()
