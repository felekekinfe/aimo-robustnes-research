"""Model-Conditioned Symbolic & Structural Feature Extractor for Math Problems.

Extracts 35 deterministic numerical features capturing:
1. Textual scale and complexity (lengths, word counts).
2. LaTeX and mathematical markup density.
3. Deep algebraic and relational structures (equations, inequalities, variables, powers).
4. Recurrence sequences and syntactic nesting depth.
5. Numeric magnitude and Diophantine / integer constraints.
6. Problem domain signals (geometry, number theory, combinatorics, algebra).
7. Goal phrasing heuristics (finding vs proving).
8. Model capacity & compute budget priors (log parameter scale, reasoning effort scale, capacity-to-load ratio).
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
    "degrees_of_freedom",
    "max_bracket_depth",
    "has_recurrence",
    # Numeric & Diophantine characteristics
    "number_count",
    "max_number_magnitude",
    "has_float",
    "has_integer_constraint",
    # Domain keywords heuristics
    "has_geometry_keyword",
    "has_number_theory_keyword",
    "has_combinatorics_keyword",
    "has_algebra_keyword",
    # Goal type heuristics
    "has_find_evaluate",
    "has_prove",
    "question_mark_count",
    # Narrative & Word problem indicators
    "narrative_step_count",
    "is_word_problem",
    "subscript_density",
    # Cognitive load & Model-conditioned priors
    "cognitive_load",
    "model_param_log",
    "reasoning_effort_scale",
    "capacity_to_load_ratio",
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
RE_RECURRENCE = re.compile(
    r"([a-zA-Z])_\{?[nki]\s*[+-]\s*\d+\}?|recurrence|fibonacci|defined as follows",
    re.IGNORECASE,
)
RE_INTEGER_CONSTRAINT = re.compile(
    r"\b(integer|integers|positive integer|whole number|divisible by|multiple of)\b",
    re.IGNORECASE,
)
RE_NARRATIVE_STEPS = re.compile(
    r"\b(then|after|before|each|every|remaining|erases?|writes?|operates?|schedule|flights?|airline|board|consecutive|regardless)\b",
    re.IGNORECASE,
)


def parse_model_param_log(model_id: str) -> float:
    """Extract approximate log10 parameter scale (in billions) from model identifier."""
    if not model_id or not isinstance(model_id, str):
        return 0.9031  # Default to ~8B
    m_lower = model_id.lower()
    match = re.search(r"(\d+)\s*b\b", m_lower)
    if match:
        params_b = float(match.group(1))
    elif "gpt-5" in m_lower:
        params_b = 500.0
    elif "gemini" in m_lower:
        params_b = 100.0
    elif "glm" in m_lower:
        params_b = 50.0
    else:
        params_b = 8.0
    return float(math.log10(max(1.0, params_b)))


def parse_reasoning_effort_scale(reasoning_effort: str) -> float:
    """Map reasoning effort string to a continuous budget scale in [0.0, 1.0]."""
    if not reasoning_effort or not isinstance(reasoning_effort, str):
        return 0.50
    effort_lower = reasoning_effort.lower().strip()
    mapping = {
        "low": 0.33,
        "medium": 0.66,
        "high": 1.00,
        "default": 0.50,
    }
    return mapping.get(effort_lower, 0.50)


def compute_max_bracket_depth(text: str) -> float:
    """Compute maximum syntactic nesting depth of brackets, parentheses, and braces."""
    current = 0
    max_d = 0
    for ch in text:
        if ch in "({[":
            current += 1
            if current > max_d:
                max_d = current
        elif ch in ")}]":
            if current > 0:
                current -= 1
    return float(max_d)


def extract_symbolic_features(
    text: str,
    model_id: str = "default",
    reasoning_effort: str = "default",
) -> dict[str, float]:
    """Extract 35 model-conditioned and structural features from problem text."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    char_count = float(len(text))
    words = text.split()
    word_count = float(len(words))
    line_count = float(len(text.splitlines())) if text else 0.0
    avg_word_length = char_count / max(1.0, word_count) if word_count > 0 else 0.0

    math_matches = RE_MATH_BLOCKS.findall(text)
    math_delimiter_count = float(len(math_matches))
    math_chars = sum(len(m) for m in math_matches)
    math_char_ratio = (float(math_chars) / max(1.0, char_count)) if char_count > 0 else 0.0

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

    equal_sign_count = float(text.count("="))
    inequality_sign_count = float(
        text.count("<")
        + text.count(">")
        + lower_text.count(r"\le")
        + lower_text.count(r"\ge")
        + lower_text.count(r"\neq")
    )
    relation_count = equal_sign_count + inequality_sign_count

    math_combined = " ".join(math_matches) if math_matches else text
    variables = set(RE_VARIABLES.findall(math_combined))
    distinct_variable_count = float(len(variables))
    degrees_of_freedom = distinct_variable_count - equal_sign_count
    max_bracket_depth = compute_max_bracket_depth(text)
    has_recurrence = 1.0 if RE_RECURRENCE.search(text) else 0.0

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
    has_integer_constraint = 1.0 if RE_INTEGER_CONSTRAINT.search(text) else 0.0

    has_geometry = 1.0 if RE_GEOMETRY.search(text) else 0.0
    has_number_theory = 1.0 if RE_NUMBER_THEORY.search(text) else 0.0
    has_combinatorics = 1.0 if RE_COMBINATORICS.search(text) else 0.0
    has_algebra = 1.0 if RE_ALGEBRA.search(text) else 0.0

    has_find = 1.0 if RE_FIND_EVALUATE.search(text) else 0.0
    has_prv = 1.0 if RE_PROVE.search(text) else 0.0
    question_mark_count = float(text.count("?"))

    narrative_step_count = float(len(RE_NARRATIVE_STEPS.findall(text)))
    is_word_problem = 1.0 if (math_char_ratio < 0.20 and char_count > 220.0) else 0.0
    subscript_density = subscript_count / max(1.0, word_count)

    # Cognitive load and capacity ratios
    cognitive_load = (char_count * math_char_ratio) / 100.0
    model_param_log = parse_model_param_log(model_id)
    reasoning_effort_scale = parse_reasoning_effort_scale(reasoning_effort)
    capacity_to_load_ratio = (model_param_log + 1.0) / (cognitive_load + 1.0)

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
        "degrees_of_freedom": degrees_of_freedom,
        "max_bracket_depth": max_bracket_depth,
        "has_recurrence": has_recurrence,
        "number_count": number_count,
        "max_number_magnitude": max_number_magnitude,
        "has_float": has_float,
        "has_integer_constraint": has_integer_constraint,
        "has_geometry_keyword": has_geometry,
        "has_number_theory_keyword": has_number_theory,
        "has_combinatorics_keyword": has_combinatorics,
        "has_algebra_keyword": has_algebra,
        "has_find_evaluate": has_find,
        "has_prove": has_prv,
        "question_mark_count": question_mark_count,
        "narrative_step_count": narrative_step_count,
        "is_word_problem": is_word_problem,
        "subscript_density": subscript_density,
        "cognitive_load": cognitive_load,
        "model_param_log": model_param_log,
        "reasoning_effort_scale": reasoning_effort_scale,
        "capacity_to_load_ratio": capacity_to_load_ratio,
    }


def extract_symbolic_feature_vector(
    text: str,
    model_id: str = "default",
    reasoning_effort: str = "default",
) -> np.ndarray:
    """Extract ordered numeric feature vector from problem text and model priors."""
    feature_dict = extract_symbolic_features(text, model_id, reasoning_effort)
    return np.array([feature_dict[name] for name in SYMBOLIC_FEATURE_NAMES], dtype=np.float64)


def extract_feature_dataframe(
    texts: Sequence[str],
    model_id: str = "default",
    reasoning_effort: str = "default",
) -> pd.DataFrame:
    """Extract structured DataFrame of symbolic features for a sequence of texts."""
    rows = [extract_symbolic_features(t, model_id, reasoning_effort) for t in texts]
    return pd.DataFrame(rows, columns=list(SYMBOLIC_FEATURE_NAMES))
