"""Unit tests for the Contrastive White-Box solution and perturbation engine."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solutions" / "contrastive-whitebox"

if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from contrastive.contrastive_detector import ContrastiveMetrics
from contrastive.decision import EmpiricalBayesCalibrator, finalize_decision
from contrastive.perturbation_engine import generate_contrastive_variant, generate_variants, safe_variable_swap


def load_contrastive_solution():
    path = SOLUTION_DIR / "solution.py"
    spec = importlib.util.spec_from_file_location("contrastive_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load contrastive solution")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOLUTION = load_contrastive_solution()


class ContrastiveWhiteBoxTests(unittest.TestCase):
    def test_perturbation_variable_renaming(self) -> None:
        """Verify meaning-preserving variable substitution."""
        problem = "Find all real values of $x$ such that $x^2 + 5x + 6 = 0$."
        variant = generate_contrastive_variant(problem)

        self.assertNotEqual(problem, variant)
        # Should replace $x$ with $t$ in math expressions
        self.assertIn("$t$", variant)
        self.assertIn("t^2", variant)

    def test_perturbation_latex_equivalences(self) -> None:
        """Verify LaTeX syntactic substitution."""
        problem = "Let $0 \\le n < 18$. Find $n$."
        variant = generate_contrastive_variant(problem)

        self.assertNotEqual(problem, variant)
        self.assertIn(r"\leq", variant)

    def test_contract_and_types(self) -> None:
        """Verify are_robust produces strict list[bool] with preserved length."""
        problems = [
            "Find all positive integers $n$ such that $n^2 + 1$ is prime.",
            "In triangle $ABC$, let $AB = 5, BC = 6, CA = 7$. Compute the area.",
        ]
        results = SOLUTION.are_robust("test-model:low", "low", problems)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(problems))
        for res in results:
            self.assertIs(type(res), bool)

    def test_mocked_stable_vs_unstable_contrastive(self) -> None:
        """Verify that low drift + positive margin predicts True, high drift predicts False."""
        problems = ["Problem A", "Problem B"]

        stable_metric = ContrastiveMetrics(
            cosine_sim=0.99,
            latent_drift=0.01,
            rel_l2_dist=0.05,
            margin_orig=2.0,
            margin_variant=1.9,
            margin_agreement=True,
        )
        unstable_metric = ContrastiveMetrics(
            cosine_sim=0.70,
            latent_drift=0.30,
            rel_l2_dist=0.60,
            margin_orig=-1.5,
            margin_variant=0.5,
            margin_agreement=False,
        )

        with mock.patch(
            "contrastive.decision.extract_contrastive_batch",
            return_value=[stable_metric, unstable_metric],
        ):
            results = SOLUTION.are_robust("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default", problems)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0])
        self.assertFalse(results[1])

    def test_fallback_when_neural_unavailable(self) -> None:
        """Verify graceful fallback when neural model fails or returns None."""
        problems = ["Find the value of $x$ such that $2x = 4$."]
        with mock.patch("contrastive.decision.extract_contrastive_batch", return_value=None):
            results = SOLUTION.are_robust("unsupported-model", "default", problems)

        self.assertEqual(len(results), 1)
        self.assertIs(type(results[0]), bool)

    def test_latex_command_preservation(self) -> None:
        """Verify \text{max}_{x \in S} (3x + \exp(x)) = 10 swaps bound x without altering \text{max} or \exp."""
        p = r"\text{max}_{x \in S} (3x + \exp(x)) = 10"
        res = safe_variable_swap(p)
        self.assertNotEqual(p, res)
        self.assertIn(r"\text{max}", res)
        self.assertIn(r"\exp", res)
        # Check that bound x in subscripts/terms was swapped to another variable
        clean_res = res.replace(r"\exp", "").replace(r"\text{max}", "")
        self.assertNotIn("x", clean_res)

    def test_non_degenerate_variants(self) -> None:
        """Verify generate_variants(p, n=3) returns 3 unique strings all != p on pure prose without $."""
        p = "Alice and Bob play a token game with 1000 coins where each player takes some tokens. Who wins?"
        variants = generate_variants(p, n=3)
        self.assertEqual(len(variants), 3)
        self.assertEqual(len(set(variants)), 3)
        for v in variants:
            self.assertNotEqual(p.strip(), v.strip())

    def test_empirical_bayes_single_sample_and_batch(self) -> None:
        """Verify EmpiricalBayesCalibrator returns finite Z-scores at N=1 and converges at N=30."""
        cal = EmpiricalBayesCalibrator()
        z_1 = cal.calibrate("test-model-8b", "default", "latent_drift", 0.030)
        self.assertTrue(math.isfinite(z_1))

        for i in range(29):
            z_30 = cal.calibrate("test-model-8b", "default", "latent_drift", 0.025 + 0.0005 * (i % 5))
        self.assertTrue(math.isfinite(z_30))
        self.assertAlmostEqual(z_30, (0.027 - 0.026) / (0.015 * 0.85), delta=1.5)

    def test_micro_rollout_self_correction_vs_collapse(self) -> None:
        """Verify finalize_decision rescues a borderline instance on convergence and vetoes on collapse."""
        # Borderline instance z_comp = 0.40 (without rollout would be False since >= 0.30)
        rescued_metric = ContrastiveMetrics(
            latent_drift=0.03,
            used_micro_rollout=True,
            rollout_convergence_ratio=0.65,
            rollout_min_margin=0.30,
        )
        decision_rescued = finalize_decision(0.40, rescued_metric, decision_threshold=0.30)
        self.assertTrue(decision_rescued)

        # Borderline instance z_comp = 0.10 (without rollout would be True since < 0.30)
        collapsed_metric = ContrastiveMetrics(
            latent_drift=0.03,
            used_micro_rollout=True,
            rollout_convergence_ratio=1.40,
            rollout_min_margin=0.05,
        )
        decision_collapsed = finalize_decision(0.10, collapsed_metric, decision_threshold=0.30)
        self.assertFalse(decision_collapsed)


if __name__ == "__main__":
    unittest.main()
