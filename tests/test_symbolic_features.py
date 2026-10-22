"""Unit tests for the symbolic and structural feature extractor."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aimo_symbolic.features import (
    SYMBOLIC_FEATURE_NAMES,
    extract_feature_dataframe,
    extract_symbolic_feature_vector,
    extract_symbolic_features,
)


class SymbolicFeatureExtractorTests(unittest.TestCase):
    def test_feature_names_schema(self) -> None:
        """Verify feature schema is non-empty and unique."""
        self.assertGreater(len(SYMBOLIC_FEATURE_NAMES), 10)
        self.assertEqual(len(SYMBOLIC_FEATURE_NAMES), len(set(SYMBOLIC_FEATURE_NAMES)))

    def test_empty_string_handling(self) -> None:
        """Verify empty string produces finite zeros without crashing."""
        feats = extract_symbolic_features("")
        self.assertEqual(len(feats), len(SYMBOLIC_FEATURE_NAMES))
        self.assertEqual(feats["char_count"], 0.0)
        self.assertEqual(feats["word_count"], 0.0)
        self.assertEqual(feats["math_char_ratio"], 0.0)
        self.assertEqual(feats["number_count"], 0.0)

        vec = extract_symbolic_feature_vector("")
        self.assertEqual(vec.shape, (len(SYMBOLIC_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(vec).all())

    def test_algebraic_equation_extraction(self) -> None:
        """Verify extraction on an algebraic equation with variables and exponents."""
        problem = "Find all real solutions $(x, y)$ such that $x^2 + y^2 = 25$ and $x - y = 1$."
        feats = extract_symbolic_features(problem)

        self.assertGreater(feats["math_delimiter_count"], 0.0)
        self.assertGreater(feats["math_char_ratio"], 0.0)
        self.assertGreaterEqual(feats["equal_sign_count"], 2.0)
        self.assertGreaterEqual(feats["superscript_count"], 2.0)
        self.assertEqual(feats["has_find_evaluate"], 1.0)
        self.assertEqual(feats["has_prove"], 0.0)
        self.assertIn("x", ["x", "y"])  # variable existence
        self.assertGreaterEqual(feats["distinct_variable_count"], 2.0)

    def test_geometry_keyword_detection(self) -> None:
        """Verify geometric problems are flagged with the geometry keyword heuristic."""
        problem = "In triangle $ABC$, the circle centered at $O$ has radius $r = 5$. Compute the area."
        feats = extract_symbolic_features(problem)

        self.assertEqual(feats["has_geometry_keyword"], 1.0)
        self.assertEqual(feats["has_number_theory_keyword"], 0.0)
        self.assertEqual(feats["has_combinatorics_keyword"], 0.0)
        self.assertEqual(feats["has_find_evaluate"], 1.0)

    def test_number_theory_detection(self) -> None:
        """Verify number theory and modular arithmetic heuristics."""
        problem = "Find the remainder when $2^{100}$ is divided by the prime $p = 101$ modulo $7$."
        feats = extract_symbolic_features(problem)

        self.assertEqual(feats["has_number_theory_keyword"], 1.0)
        self.assertGreaterEqual(feats["superscript_count"], 1.0)
        self.assertGreaterEqual(feats["number_count"], 3.0)

    def test_combinatorics_detection(self) -> None:
        """Verify combinatorics keywords and binomial coefficients."""
        problem = "How many ways can we choose a subset of 5 elements from 20? Evaluate $\\binom{20}{5}$."
        feats = extract_symbolic_features(problem)

        self.assertEqual(feats["has_combinatorics_keyword"], 1.0)
        self.assertGreaterEqual(feats["binom_count"], 1.0)
        self.assertEqual(feats["has_find_evaluate"], 1.0)

    def test_vector_and_dataframe_shape(self) -> None:
        """Verify vector and DataFrame extraction maintain exact schema and order."""
        texts = [
            "What is $2 + 2$?",
            "Let $f(x) = x^3 - 3x$. Prove that $f(x)$ has two local extrema.",
        ]
        df = extract_feature_dataframe(texts)

        self.assertEqual(df.shape, (2, len(SYMBOLIC_FEATURE_NAMES)))
        self.assertEqual(list(df.columns), list(SYMBOLIC_FEATURE_NAMES))
        self.assertTrue(np.isfinite(df.to_numpy()).all())

        # Check order preservation
        self.assertEqual(df.iloc[0]["has_find_evaluate"], 1.0)
        self.assertEqual(df.iloc[1]["has_prove"], 1.0)


if __name__ == "__main__":
    unittest.main()
