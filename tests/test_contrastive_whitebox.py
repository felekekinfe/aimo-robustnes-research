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
        """Verify that high causal grounding + strong spectral compression predicts True, poor predicts False."""
        problems = ["Compute $2+2$.", "In $\\triangle ABC$, compute the circumcircle radius."]

        stable_metric = ContrastiveMetrics(
            causal_grounding_jsd=0.18,
            spectral_compression=0.12,
            late_erank=0.52,
            math_surprisal=1.80,
            causal_hidden_shift=0.08,
        )
        unstable_metric = ContrastiveMetrics(
            causal_grounding_jsd=0.01,
            spectral_compression=-0.02,
            late_erank=0.82,
            math_surprisal=3.60,
            causal_hidden_shift=0.01,
        )

        with mock.patch(
            "contrastive.decision.extract_contrastive_batch",
            return_value=[stable_metric, unstable_metric],
        ):
            results = SOLUTION.are_robust("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default", problems)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0])
        self.assertFalse(results[1])

    def test_realistic_bfloat16_prefill_discrimination(self) -> None:
        """Verify realistic Mechanistic Interpretability signals produce balanced, discriminating outputs."""
        problems = ["Find all real roots of $x^2 + 5x + 6 = 0$.", "Let $N$ be a 3-digit number..."]

        res_stable = ContrastiveMetrics(
            causal_grounding_jsd=0.16,
            spectral_compression=0.10,
            late_erank=0.55,
            math_surprisal=1.90,
        )
        res_unstable = ContrastiveMetrics(
            causal_grounding_jsd=0.02,
            spectral_compression=-0.01,
            late_erank=0.80,
            math_surprisal=3.50,
        )

        with mock.patch(
            "contrastive.decision.extract_contrastive_batch",
            return_value=[res_stable, res_unstable],
        ):
            predictions = SOLUTION.are_robust("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default", problems)

        self.assertEqual(len(predictions), 2)
        self.assertIn(True, predictions)
        self.assertIn(False, predictions)

    def test_spectral_effective_rank_computation(self) -> None:
        """Verify compute_spectral_effective_rank returns values in [0.0, 1.0]."""
        import torch
        from contrastive.contrastive_detector import compute_spectral_effective_rank

        # Highly correlated tokens (rank 1) -> erank near 0
        u = torch.randn(64)
        H_rank1 = torch.outer(torch.linspace(1, 10, 10), u)
        erank_low = compute_spectral_effective_rank(H_rank1)
        self.assertGreaterEqual(erank_low, 0.0)
        self.assertLess(erank_low, 0.10)

        # Random orthogonal-like tokens -> erank higher
        torch.manual_seed(42)
        H_rand = torch.randn(10, 64)
        erank_high = compute_spectral_effective_rank(H_rand)
        self.assertGreater(erank_high, erank_low)
        self.assertLessEqual(erank_high, 1.0)

    def test_constraint_ablation(self) -> None:
        """Verify build_constraint_ablated_problem ablates internal conditions."""
        from contrastive.contrastive_detector import build_constraint_ablated_problem

        problem = "Let $x$ and $y$ be positive integers such that $x + y = 10$. Find $x^2 - y^2$."
        ablated = build_constraint_ablated_problem(problem)
        self.assertNotEqual(problem, ablated)
        self.assertIn("$c$", ablated)

    def test_n_equals_one_invariance(self) -> None:
        """Verify strict N=1 invariance: predictions do not depend on other batch items."""
        p1 = "Find all positive integers $n$ such that $n^2 + 1$ is prime."
        p2 = "In triangle $ABC$, let $AB = 5, BC = 6, CA = 7$. Compute the area."

        with mock.patch("contrastive.decision.extract_contrastive_batch", return_value=None):
            r_both = SOLUTION.are_robust("test-model", "default", [p1, p2])
            r_1 = SOLUTION.are_robust("test-model", "default", [p1])
            r_2 = SOLUTION.are_robust("test-model", "default", [p2])

        self.assertEqual(r_both[0], r_1[0])
        self.assertEqual(r_both[1], r_2[0])

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
        z_1 = cal.calibrate("test-model-8b", "default", "latent_drift", 0.0045)
        self.assertTrue(math.isfinite(z_1))

        for i in range(29):
            z_30 = cal.calibrate("test-model-8b", "default", "latent_drift", 0.0040 + 0.0002 * (i % 5))
        self.assertTrue(math.isfinite(z_30))

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

    def test_oversized_model_gate(self) -> None:
        """Verify _is_oversized_model intercepts massive models and allows small models."""
        from contrastive.contrastive_detector import _is_oversized_model

        self.assertTrue(_is_oversized_model("openai/gpt-oss-120b"))
        self.assertTrue(_is_oversized_model("Qwen/Qwen3.5-397B-A17B-FP8"))
        self.assertTrue(_is_oversized_model("meta-llama/Llama-3-70B"))
        self.assertFalse(_is_oversized_model("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"))
        self.assertFalse(_is_oversized_model("Qwen/Qwen3.5-4B"))
        self.assertFalse(_is_oversized_model("Skywork/Skywork-OR1-Math-7B"))


    def test_extract_causal_nodes(self) -> None:
        """Verify extraction of surface_text, numbers, math_spans, and question."""
        from contrastive.causal_graph import extract_causal_nodes

        problem = "A farmer has 12 sheep and 8 cows. If each sheep drinks $3$ liters of water, find the total water."
        nodes = extract_causal_nodes(problem)

        self.assertIn("surface_text", nodes)
        self.assertIn("numbers", nodes)
        self.assertIn("math_spans", nodes)
        self.assertIn("question", nodes)
        self.assertIn("12", nodes["numbers"])
        self.assertIn("8", nodes["numbers"])
        self.assertTrue(len(nodes["math_spans"]) > 0)
        self.assertIn("total water", nodes["question"].lower())

    def test_generate_evolutionary_variants(self) -> None:
        """Verify that generate_evolutionary_variants returns 3 distinct, balanced strings."""
        from contrastive.perturbation_engine import generate_evolutionary_variants

        problem = "In a right triangle $ABC$ with legs $a = 6$ and $b = 8$, compute the hypotenuse $c$."
        v_orig, v_narr, v_stress = generate_evolutionary_variants(problem)

        self.assertEqual(v_orig, problem)
        self.assertNotEqual(v_orig, v_narr)
        self.assertNotEqual(v_orig, v_stress)
        self.assertNotEqual(v_narr, v_stress)
        # All variants should preserve LaTeX delimiter balance
        for variant in (v_orig, v_narr, v_stress):
            self.assertEqual(variant.count("$") % 2, 0, f"Unbalanced $ in {variant}")

    def test_causal_score_discrimination(self) -> None:
        """Verify high consistency + low sensitivity -> positive score, low consistency -> negative score."""
        from contrastive.contrastive_detector import CausalConsistencyMetrics
        from contrastive.decision import compute_causal_score

        robust_metric = CausalConsistencyMetrics(
            consistency=0.92,
            sensitivity=0.12,
        )
        fragile_metric = CausalConsistencyMetrics(
            consistency=0.70,
            sensitivity=0.45,
        )

        score_robust = compute_causal_score(robust_metric)
        score_fragile = compute_causal_score(fragile_metric)

        self.assertGreater(score_robust, 0.0)
        self.assertLess(score_fragile, 0.0)
        self.assertGreater(score_robust, score_fragile)


if __name__ == "__main__":
    unittest.main()
