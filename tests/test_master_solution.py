"""Comprehensive tests for the unified solution, triton kernels, and perceptors."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from solution import are_robust, GLOBAL_CLASSIFIER, split_model_and_effort, canonical_problem_hash
from triton_kernels import compute_linear_attention_drift, HAS_TRITON
from perceptors import SymbolicPerturber, SubspaceGeometrySensor, KernelizedRetrievalRouter


class MasterSolutionTests(unittest.TestCase):
    """Test suite for master solution and submodules."""

    def test_are_robust_contracts(self):
        problems = [
            "Let x and y be positive integers such that x + y = 10. Find max xy.",
            "Find the number of primes less than 50.",
        ]
        # Standard signature
        res1 = are_robust("Qwen/Qwen3.5-4B", reasoning_effort="default", problems=problems)
        self.assertEqual(len(res1), 2)
        self.assertTrue(all(type(r) is bool for r in res1))

        # Legacy 2-argument signature
        res2 = are_robust("openai/gpt-oss-120b", problems)
        self.assertEqual(len(res2), 2)
        self.assertTrue(all(type(r) is bool for r in res2))

        # Empty problem list
        self.assertEqual(are_robust("Qwen/Qwen3.5-4B", problems=[]), [])

    def test_n_equals_one_invariance(self):
        p1 = "Let ABC be a triangle with sides 3, 4, 5. Find the circumradius."
        p2 = "Compute the sum of divisors of 120."
        
        batch_res = are_robust("openai/gpt-oss-120b", reasoning_effort="medium", problems=[p1, p2])
        indiv1 = are_robust("openai/gpt-oss-120b", reasoning_effort="medium", problems=[p1])
        indiv2 = are_robust("openai/gpt-oss-120b", reasoning_effort="medium", problems=[p2])

        self.assertEqual(batch_res[0], indiv1[0])
        self.assertEqual(batch_res[1], indiv2[0])

    def test_triton_linear_attention_drift(self):
        b, seq, d = 2, 32, 64
        h_orig = torch.randn(b, seq, d)
        h_pert = h_orig + 0.05 * torch.randn(b, seq, d)
        drift = compute_linear_attention_drift(h_orig, h_pert)
        
        self.assertIsInstance(drift, float)
        self.assertGreaterEqual(drift, 0.0)
        self.assertLessEqual(drift, 2.0)

    def test_symbolic_perturber_quad(self):
        problem = "Let ABC be a triangle with side lengths 13, 14, 15. Find the inradius r."
        raw, probed = SymbolicPerturber.generate_contrastive_quad(problem)
        self.assertEqual(len(raw), 4)
        self.assertEqual(len(probed), 4)
        self.assertEqual(raw[0], problem)
        self.assertIn("boxed{", probed[0])
        self.assertNotEqual(raw[0], raw[3])  # Destroy altered numerical premises

    def test_subspace_geometry_sensor(self):
        H = torch.randn(20, 64)
        srank = SubspaceGeometrySensor.compute_stable_rank(H)
        spec_rank = SubspaceGeometrySensor.compute_spectral_effective_rank(H)
        self.assertGreaterEqual(srank, 0.0)
        self.assertLessEqual(srank, 1.0)
        self.assertGreaterEqual(spec_rank, 0.0)
        self.assertLessEqual(spec_rank, 1.0)

    def test_kernelized_retrieval_router(self):
        router = KernelizedRetrievalRouter()
        problem = "Find all integer pairs (x, y) satisfying x^2 - y^2 = 24."
        sim = router.query_perturbation_severity(problem)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)


if __name__ == "__main__":
    unittest.main()
