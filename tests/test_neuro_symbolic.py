"""Unit tests for the Neuro-Symbolic solution entrypoint and fusion contract."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solutions" / "neuro-symbolic"

if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))


def load_neuro_symbolic_solution():
    path = SOLUTION_DIR / "solution.py"
    spec = importlib.util.spec_from_file_location("neuro_symbolic_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load neuro_symbolic solution")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOLUTION = load_neuro_symbolic_solution()


class NeuroSymbolicSolutionTests(unittest.TestCase):
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

    def test_mocked_neural_fusion(self) -> None:
        """Verify that neural margins are fused when the neural branch is active."""
        problems = ["Problem 1", "Problem 2"]
        with mock.patch("neuro_symbolic.fusion.score_neural_probe_batch", return_value=[5.0, -5.0]):
            results = SOLUTION.are_robust("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default", problems)

        self.assertEqual(len(results), 2)
        # Margin +5.0 strongly drives probability towards True
        self.assertTrue(results[0])
        # Margin -5.0 strongly drives probability towards False
        self.assertFalse(results[1])

    def test_fallback_when_neural_unavailable(self) -> None:
        """Verify graceful fallback when neural model fails or returns None."""
        problems = ["Find the value of $x$ such that $2x = 4$."]
        with mock.patch("neuro_symbolic.fusion.score_neural_probe_batch", return_value=None):
            results = SOLUTION.are_robust("unsupported-model", "default", problems)

        self.assertEqual(len(results), 1)
        self.assertIs(type(results[0]), bool)


if __name__ == "__main__":
    unittest.main()
