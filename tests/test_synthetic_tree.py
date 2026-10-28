"""Unit tests for the Synthetic Tree Architecture."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solutions" / "synthetic-tree"

if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from decision import compute_tree_decision_logit, get_empirical_prior
from stratified_engine import evaluate_stratified_metrics
from synthetic_tree import generate_synthetic_tree, get_tree_branch_names


def load_tree_solution():
    path = SOLUTION_DIR / "solution.py"
    spec = importlib.util.spec_from_file_location("tree_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load synthetic tree solution")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOLUTION = load_tree_solution()


class SyntheticTreeTests(unittest.TestCase):
    def test_tree_branch_count_and_names(self) -> None:
        """Verify that exactly 14 branches across 7 categories are generated."""
        p = "Let $ABC$ be a triangle with positive integers $a, b, c$ such that $a^2 + b^2 = c^2$. Find the area."
        tree = generate_synthetic_tree(p)
        branch_names = get_tree_branch_names()

        self.assertEqual(len(branch_names), 14)
        self.assertIn("root", tree)
        for b in branch_names:
            self.assertIn(b, tree)
            self.assertTrue(len(tree[b]) > 0)

    def test_stratified_threshold_engine(self) -> None:
        """Verify that stratified threshold engine calculates pass rates and sensitivity."""
        raw_metrics = {
            "drift_typo_char": 0.02,
            "drift_typo_delim": 0.02,
            "drift_format_delims": 0.02,
            "drift_format_whitespace": 0.02,
            "drift_rename_algebraic": 0.05,
            "drift_rename_geometric": 0.05,
            "drift_rephrase_commute": 0.05,
            "drift_rephrase_query": 0.05,
            "drift_domain_academic": 0.10,
            "drift_domain_entities": 0.10,
            "drift_distract_prepend": 0.10,
            "drift_distract_append": 0.10,
            "drift_expert_ablate": 0.30,
            "drift_expert_mutate": 0.30,
        }
        res = evaluate_stratified_metrics(raw_metrics)
        self.assertEqual(res["inv_pass_rate"], 1.0)
        self.assertTrue(res["causal_sensitivity"] > 0.20)
        self.assertEqual(res["sens_ablate"], 1.0)
        self.assertEqual(res["sens_mutate"], 1.0)

    def test_contract_are_robust(self) -> None:
        """Verify are_robust produces strict list[bool] with preserved length."""
        problems = [
            "Find all positive integers $n$ such that $n^2 + 1$ is prime.",
            "In triangle $ABC$, let $AB = 5, BC = 6, CA = 7$. Compute the area.",
        ]
        results = SOLUTION.are_robust("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", problems=problems)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, bool)

    def test_closed_and_oversized_models(self) -> None:
        """Verify closed API models and oversized models are safely processed."""
        problems = ["What is the value of $x$ if $2x + 4 = 10$ ?"]
        for m in ["gemini-3.1-pro-preview", "gpt-5.2-2025-12-11", "Qwen/Qwen3.5-397B-A17B-FP8"]:
            res = SOLUTION.are_robust(m, problems=problems)
            self.assertEqual(len(res), 1)
            self.assertIsInstance(res[0], bool)


if __name__ == "__main__":
    unittest.main()
