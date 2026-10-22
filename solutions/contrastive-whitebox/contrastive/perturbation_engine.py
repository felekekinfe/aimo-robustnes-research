"""Deterministic, meaning-preserving perturbation engine for mathematical problems.

Applies benign transformations (canonical variable renaming, LaTeX syntactic
equivalences, and phrasing substitutions) that do not alter the underlying
mathematical problem or its solution.
"""

from __future__ import annotations

import re

# LaTeX command equivalences: (pattern, replacement)
SYNTACTIC_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\le\b"), r"\\leq"),
    (re.compile(r"\\ge\b"), r"\\geq"),
    (re.compile(r"\\cdot\b"), r"\\times"),
    (re.compile(r"\\dfrac\b"), r"\\frac"),
    (re.compile(r"\\pmod\b"), r"\\bmod"),
    (re.compile(r"\\left\("), r"("),
    (re.compile(r"\\right\)"), r")"),
]

# Question framing rephrasings
FRAMING_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bWhat is the value of\b", re.IGNORECASE), "Find the value of"),
    (re.compile(r"\bWhat is\b", re.IGNORECASE), "Compute"),
    (re.compile(r"\bFind all real solutions\b", re.IGNORECASE), "Determine all real solutions"),
    (re.compile(r"\bFind all possible values of\b", re.IGNORECASE), "Determine all possible values of"),
    (re.compile(r"\bFind\b", re.IGNORECASE), "Determine"),
    (re.compile(r"\bEvaluate\b", re.IGNORECASE), "Compute"),
    (re.compile(r"\bCompute:\b", re.IGNORECASE), "Evaluate:"),
    (re.compile(r"\bSolve for\b", re.IGNORECASE), "Find the value of"),
]


RE_MATH_BLOCKS = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\[.*?|\\])", re.DOTALL)
RE_MATH_VARS = re.compile(r"\b[a-zA-Z]\b")


def _get_math_variables(text: str) -> set[str]:
    """Find all single-letter variables appearing inside math blocks."""
    blocks = RE_MATH_BLOCKS.findall(text)
    combined = " ".join(blocks) if blocks else ""
    return set(RE_MATH_VARS.findall(combined))


def _rename_variable_in_math(text: str, old_var: str, new_var: str) -> str:
    """Rename a single-letter variable strictly inside LaTeX math blocks."""
    def replace_in_math_block(match: re.Match) -> str:
        block = match.group(0)
        # Replace old_var when surrounded by non-alphanumeric characters or start/end
        renamed = re.sub(rf"(?<![a-zA-Z\\]){old_var}(?![a-zA-Z])", new_var, block)
        return renamed

    return RE_MATH_BLOCKS.sub(replace_in_math_block, text)


def generate_contrastive_variant(problem_text: str) -> str:
    """Generate a semantically equivalent, perturbed variant of the problem text.

    Args:
        problem_text: Original mathematical problem string.

    Returns:
        Perturbed problem string representing the identical mathematical problem.
    """
    if not problem_text or not isinstance(problem_text, str):
        return problem_text

    variant = problem_text
    math_vars = _get_math_variables(problem_text)

    # 1. Try variable renaming: replace an active math variable with an unused candidate
    candidate_renames = [
        ("x", "t"),
        ("x", "u"),
        ("n", "k"),
        ("n", "m"),
        ("y", "w"),
        ("a", "u"),
        ("b", "v"),
    ]
    for old_v, new_v in candidate_renames:
        if old_v in math_vars and new_v not in math_vars:
            renamed_text = _rename_variable_in_math(variant, old_v, new_v)
            if renamed_text != variant:
                variant = renamed_text
                break

    # 2. Apply LaTeX syntactic equivalences
    for pattern, replacement in SYNTACTIC_REPLACEMENTS:
        new_variant, count = pattern.subn(replacement, variant)
        if count > 0:
            variant = new_variant
            break

    # 3. If no change occurred yet, apply framing rephrasing
    if variant == problem_text:
        for pattern, replacement in FRAMING_REPLACEMENTS:
            new_variant, count = pattern.subn(replacement, variant, count=1)
            if count > 0:
                variant = new_variant
                break

    # 4. Fallback: prepend a harmless standard directive if text remained identical
    if variant == problem_text:
        variant = f"Carefully solve: {problem_text}"

    return variant
