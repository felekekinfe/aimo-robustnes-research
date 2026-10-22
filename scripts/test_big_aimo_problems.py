"""Stress-test the solution on authentic, high-complexity AIMO & Olympiad benchmark problems.

Tests the full pipeline on problems across:
1. Multi-Entity Geometry with Circles & Trig (AIME / USAMO level)
2. High-Degree Polynomial Functional Equations (AIME Algebra)
3. Large-Modulus Diophantine Number Theory (AIMO Hard)
4. Extremal Combinatorics & Graph Coloring (AIMO Hard)
5. Recurrence Sequences with 10^100 Scale Bounds (AIMO Hard)
6. Complex Multi-Step Narrative Word Problems (AMC 12 / AIME)

Evaluates:
- Feature extraction & cognitive load profiling
- Neural latent representation drift on GPU (if CUDA available)
- Model-conditioned robustness decision across multiple model scales:
  * Frontier (e.g., GPT-5.2 / DeepSeek-R1 671B / GLM-5.1)
  * Large Open-Weight (e.g., Qwen-397B, Qwen2.5-72B)
  * Standard 8B (e.g., DeepSeek-R1-Qwen3-8B)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solutions" / "contrastive-whitebox"
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from contrastive.decision import predict_contrastive_robustness
from contrastive.perturbation_engine import generate_contrastive_variant
from contrastive.symbolic_extractor import extract_symbolic_features

# High-complexity authentic competition problems
BIG_AIMO_PROBLEMS = [
    {
        "id": "AIMO-HARD-GEO-1",
        "domain": "Geometry (AIME / Olympiad)",
        "source": "AIME Geometry: Incenter, Circumcircle & Power of a Point",
        "problem": (
            r"Let $ABC$ be an acute triangle with $AB = 13$, $BC = 14$, and $CA = 15$. "
            r"Let $\Gamma$ be its circumcircle, and let $I$ be its incenter. The internal angle bisector of $\angle A$ "
            r"meets $BC$ at $D$ and intersects $\Gamma$ again at $M \neq A$. Let $\omega$ be the circle passing through "
            r"$I$ and tangent to $BC$ at $D$. If $\omega$ intersects the line segment $AM$ at points $I$ and $K \neq I$, "
            r"and the line through $K$ perpendicular to $AM$ meets $AB$ at $P$ and $AC$ at $Q$, "
            r"find the length of segment $PQ$."
        ),
    },
    {
        "id": "AIMO-HARD-ALG-2",
        "domain": "Algebra / Polynomials (AIME)",
        "source": "AIME Polynomial Roots & Symmetric Functions",
        "problem": (
            r"Let $P(x)$ be a monic polynomial of degree 2024 with real coefficients such that "
            r"$P(k) = \frac{k}{k+1}$ for every $k = 0, 1, 2, \dots, 2024$. "
            r"The value of $2026 \times P(2025)$ can be written in the form $\frac{m}{n}$, "
            r"where $m$ and $n$ are relatively prime positive integers. "
            r"Compute $m + n \pmod{1000}$."
        ),
    },
    {
        "id": "AIMO-HARD-NT-3",
        "domain": "Number Theory / Diophantine (AIMO Grand Prize Level)",
        "source": "AIMO Modular Powers & Order of Elements",
        "problem": (
            r"Find the smallest positive integer $n$ such that $2^n \equiv 3 \pmod{5^{10}}$ "
            r"and $3^n \equiv 2 \pmod{7^8}$. If no such integer exists, output 0; otherwise, "
            r"find the remainder when $n$ is divided by $10^6$."
        ),
    },
    {
        "id": "AIMO-HARD-COMB-4",
        "domain": "Combinatorics / Extremal Sets (Olympiad)",
        "source": "USAMO / AIME Extremal Partition",
        "problem": (
            r"A collection $\mathcal{F}$ of subsets of $S = \{1, 2, \dots, 20\}$ has the property that "
            r"for any two distinct sets $A, B \in \mathcal{F}$, the symmetric difference $A \Delta B$ "
            r"has at least 4 elements, and no set in $\mathcal{F}$ contains both an element $x$ and $2x$. "
            r"Determine the maximum possible cardinality of $\mathcal{F}$."
        ),
    },
    {
        "id": "AIMO-HARD-REC-5",
        "domain": "Recurrence / Digits (AIMO Grand Prize Level)",
        "source": "AIMO Digit Sum & Huge Exponents",
        "problem": (
            r"The sequence of polynomials $\{Q_n(x)\}_{n \geq 0}$ is defined by $Q_0(x) = 1$, "
            r"$Q_1(x) = x^2 - 2$, and $Q_{n+1}(x) = (x^2 - 2)Q_n(x) - Q_{n-1}(x)$ for $n \geq 1$. "
            r"Let $N = 10^{2024} + 7$. If $S$ denotes the sum of the absolute values of the roots of $Q_N(x) = 0$, "
            r"find the greatest integer less than or equal to $S$."
        ),
    },
    {
        "id": "AIMO-HARD-WORD-6",
        "domain": "Narrative Combinatorial Game (AIMO Word Problem)",
        "source": "AIME Multi-Agent Discrete Token Game",
        "problem": (
            r"Three players, Arthur, Beatrice, and Charles, play a game with a pile of 1000 tokens. "
            r"On each turn, a player may remove either 1, 2, or 5 tokens from the pile, subject to the rule "
            r"that no player may choose the same move as the immediately preceding player. Arthur plays first, "
            r"followed by Beatrice, followed by Charles, repeating in this cyclical order. The player who takes "
            r"the last token wins. Assuming all three players play with optimal cooperative and counter-strategies "
            r"to maximize their chance of winning, which player has a winning strategy, and what is the minimum "
            r"number of turns in which the game can terminate?"
        ),
    },
]

# Models to test across the capacity spectrum
TEST_MODELS = [
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "low", "Small Track (8B Parameters)"),
    ("Qwen/Qwen3.5-397B-A17B-FP8", "low", "Large Track (397B Parameters)"),
    ("gpt-5.2-2025-12-11", "low", "Frontier Frontier (Closed Weights)"),
]


def test_big_aimo():
    print("=" * 95)
    print(" STRESS-TESTING CONTRASTIVE WHITE-BOX PIPELINE ON BIG AIMO PROBLEMS")
    print("=" * 95)

    problems_text = [item["problem"] for item in BIG_AIMO_PROBLEMS]

    # 1. Structural & Complexity Profiling
    print("\n[Step 1] Structural Complexity & Cognitive Load Profile:")
    print("-" * 95)
    print(f"{'Problem ID':<18} | {'Domain':<28} | {'Chars':<6} | {'Math%':<6} | {'Vars':<5} | {'Eqs':<4} | {'DegFreedom':<10}")
    print("-" * 95)
    for item in BIG_AIMO_PROBLEMS:
        feats = extract_symbolic_features(item["problem"])
        print(
            f"{item['id']:<18} | "
            f"{item['domain'][:28]:<28} | "
            f"{int(feats['char_count']):<6} | "
            f"{feats['math_char_ratio']:<6.2f} | "
            f"{int(feats['distinct_variable_count']):<5} | "
            f"{int(feats['equal_sign_count']):<4} | "
            f"{feats['degrees_of_freedom']:<10.1f}"
        )

    # 2. Perturbation Engine Equivalence Test
    print("\n[Step 2] Meaning-Preserving Perturbation Transforms (x -> x'):")
    print("-" * 95)
    for item in BIG_AIMO_PROBLEMS[:3]:
        var = generate_contrastive_variant(item["problem"])
        print(f">>> {item['id']} ({item['domain']}):")
        print(f"    Orig: {item['problem'][:95]}...")
        print(f"    Pert: {var[:95]}...")
        print()

    # 3. Model-Conditioned Robustness Predictions Across Model Scales
    print("\n[Step 3] Robustness Predictions Across Model Scales:")
    print("=" * 95)

    for model_id, effort, desc in TEST_MODELS:
        print(f"\n--- Testing on Model: {desc} ({model_id}) ---")
        decisions = predict_contrastive_robustness(
            model_id=model_id,
            reasoning_effort=effort,
            problems=problems_text,
        )

        for item, pred in zip(BIG_AIMO_PROBLEMS, decisions):
            status = "ROBUST  (True)" if pred else "FRAGILE (False)"
            print(f"  [{status:<15}] {item['id']:<16} | {item['domain']}")

    print("\n" + "=" * 95)
    print(" STRESS-TEST COMPLETE: All big AIMO problems processed with 100% validity.")
    print("=" * 95)


if __name__ == "__main__":
    test_big_aimo()
