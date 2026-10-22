"""Symbolic and structural feature extractor for mathematical problem texts.

This module extracts deterministic, interpretable structural features from
competition math problems on CPU without loading neural language models.
Features capture:
1. Textual scale and complexity (lengths, word counts).
2. LaTeX and mathematical markup density.
3. Algebraic and relational structures (equations, inequalities, variables, powers).
4. Numeric magnitude and arithmetic characteristics.
5. Problem domain signals (geometry, number theory, combinatorics, algebra).
6. Goal phrasing (finding a value vs proving a property).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

SYMBOLIC_FEATURE_NAMES: tuple[str, ...] = (
    # Textual scale
    "char_count",
    "word_count",
    "line_count",
    "avg_word_length",
    # LaTeX & Math markup density
    "math_delimiter_count",
    "math_char_ratio",
    "latex_command_count",
    "fraction_count",
    "sqrt_count",
    "sum_prod_int_count",
    "binom_count",
    "subscript_count",
    "superscript_count",
    # Algebraic & Relational structure
    "equal_sign_count",
    "inequality_sign_count",
    "relation_count",
    "distinct_variable_count",
    # Numeric characteristics
    "number_count",
    "max_number_magnitude",
    "has_float",
    # Domain keywords heuristics
    "has_geometry_keyword",
    "has_number_theory_keyword",
    "has_combinatorics_keyword",
    "has_algebra_keyword",
    # Goal type heuristics
    "has_find_evaluate",
    "has_prove",
    "question_mark_count",
)

# Precompiled regexes
RE_MATH_BLOCKS = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\[.*?|\\])", re.DOTALL)
RE_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+")
RE_NUMBERS = re.compile(r"\b\d+(?:\.\d+)?\b")
RE_VARIABLES = re.compile(r"\b[a-zA-Z]\b")

RE_GEOMETRY = re.compile(
    r"\b(triangle|circle|angle|radius|diameter|perimeter|area|polygon|vertex|vertices|quadrilateral|tangent|chord|secant|sphere|cone|cylinder|hypotenuse)\b",
    re.IGNORECASE,
)
RE_NUMBER_THEORY = re.compile(
    r"\b(prime|divisor|divisors|divisible|divides|modulo|mod|gcd|lcm|remainder|congruent|integers?|factorization)\b",
    re.IGNORECASE,
)
RE_COMBINATORICS = re.compile(
    r"\b(permutation|permutations|combination|combinations|probability|ways|arrangements?|choose|subset|subsets|graph|vertices|edges)\b",
    re.IGNORECASE,
)
RE_ALGEBRA = re.compile(
    r"\b(polynomial|polynomials|roots?|coefficients?|quadratic|cubic|function|functions|inequality|system of equations|matrix|determinant)\b",
    re.IGNORECASE,
)
RE_FIND_EVALUATE = re.compile(
    r"\b(find|evaluate|determine|compute|calculate|what is|how many|value of)\b",
    re.IGNORECASE,
)
RE_PROVE = re.compile(
    r"\b(prove|show that|demonstrate)\b",
    re.IGNORECASE,
)


def extract_symbolic_features(text: str) -> dict[str, float]:
    """Extract a dictionary of numerical structural features from problem text.

    Args:
        text: Problem statement string.

    Returns:
        Mapping from feature name to float value.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    char_count = float(len(text))
    words = text.split()
    word_count = float(len(words))
    line_count = float(len(text.splitlines())) if text else 0.0
    avg_word_length = char_count / max(1.0, word_count) if word_count > 0 else 0.0

    # Math markup detection
    math_matches = RE_MATH_BLOCKS.findall(text)
    math_delimiter_count = float(len(math_matches))
    math_chars = sum(len(m) for m in math_matches)
    math_char_ratio = (float(math_chars) / max(1.0, char_count)) if char_count > 0 else 0.0

    # LaTeX command frequency
    latex_commands = RE_LATEX_COMMAND.findall(text)
    latex_command_count = float(len(latex_commands))

    lower_text = text.lower()
    fraction_count = float(lower_text.count(r"\frac") + lower_text.count(r"\dfrac"))
    sqrt_count = float(lower_text.count(r"\sqrt"))
    sum_prod_int_count = float(
        lower_text.count(r"\sum") + lower_text.count(r"\prod") + lower_text.count(r"\int")
    )
    binom_count = float(lower_text.count(r"\binom") + lower_text.count(r"\choose"))
    subscript_count = float(text.count("_"))
    superscript_count = float(text.count("^"))

    # Relations and variables
    equal_sign_count = float(text.count("="))
    inequality_sign_count = float(
        text.count("<")
        + text.count(">")
        + lower_text.count(r"\le")
        + lower_text.count(r"\ge")
        + lower_text.count(r"\neq")
    )
    relation_count = equal_sign_count + inequality_sign_count

    # Extract single-letter variables within math spans or general text
    math_combined = " ".join(math_matches) if math_matches else text
    variables = set(RE_VARIABLES.findall(math_combined))
    distinct_variable_count = float(len(variables))

    # Numbers
    numbers = RE_NUMBERS.findall(text)
    number_count = float(len(numbers))
    float_values = []
    has_float = 0.0
    for n in numbers:
        try:
            val = float(n)
            float_values.append(val)
            if "." in n:
                has_float = 1.0
        except ValueError:
            pass

    max_number = max(float_values) if float_values else 0.0
    max_number_magnitude = float(math.log10(1.0 + abs(max_number))) if max_number >= 0 else 0.0

    # Domain keywords
    has_geometry = 1.0 if RE_GEOMETRY.search(text) else 0.0
    has_number_theory = 1.0 if RE_NUMBER_THEORY.search(text) else 0.0
    has_combinatorics = 1.0 if RE_COMBINATORICS.search(text) else 0.0
    has_algebra = 1.0 if RE_ALGEBRA.search(text) else 0.0

    # Goal phrasing
    has_find = 1.0 if RE_FIND_EVALUATE.search(text) else 0.0
    has_prv = 1.0 if RE_PROVE.search(text) else 0.0
    question_mark_count = float(text.count("?"))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "avg_word_length": avg_word_length,
        "math_delimiter_count": math_delimiter_count,
        "math_char_ratio": math_char_ratio,
        "latex_command_count": latex_command_count,
        "fraction_count": fraction_count,
        "sqrt_count": sqrt_count,
        "sum_prod_int_count": sum_prod_int_count,
        "binom_count": binom_count,
        "subscript_count": subscript_count,
        "superscript_count": superscript_count,
        "equal_sign_count": equal_sign_count,
        "inequality_sign_count": inequality_sign_count,
        "relation_count": relation_count,
        "distinct_variable_count": distinct_variable_count,
        "number_count": number_count,
        "max_number_magnitude": max_number_magnitude,
        "has_float": has_float,
        "has_geometry_keyword": has_geometry,
        "has_number_theory_keyword": has_number_theory,
        "has_combinatorics_keyword": has_combinatorics,
        "has_algebra_keyword": has_algebra,
        "has_find_evaluate": has_find,
        "has_prove": has_prv,
        "question_mark_count": question_mark_count,
    }


def extract_symbolic_feature_vector(text: str) -> np.ndarray:
    """Extract ordered numeric feature vector from problem text.

    Args:
        text: Problem statement string.

    Returns:
        1D float64 NumPy array with shape ``(len(SYMBOLIC_FEATURE_NAMES),)``.
    """
    feature_dict = extract_symbolic_features(text)
    return np.array([feature_dict[name] for name in SYMBOLIC_FEATURE_NAMES], dtype=np.float64)


def extract_feature_dataframe(texts: Sequence[str]) -> pd.DataFrame:
    """Extract structured DataFrame of symbolic features for a sequence of texts.

    Args:
        texts: Sequence of mathematical problem statements.

    Returns:
        DataFrame with columns matching ``SYMBOLIC_FEATURE_NAMES`` and index matching ``texts``.
    """
    rows = [extract_symbolic_features(t) for t in texts]
    return pd.DataFrame(rows, columns=list(SYMBOLIC_FEATURE_NAMES))
