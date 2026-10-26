"""Deterministic, meaning-preserving perturbation engine for mathematical problems.

Applies benign transformations (canonical variable swapping, commutative equation flipping,
condition sentence reordering, LaTeX syntactic equivalences, and framing substitutions)
that preserve mathematical semantics without requiring external LLM generation.
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
        return re.sub(rf"(?<![a-zA-Z\\]){old_var}(?![a-zA-Z])", new_var, block)

    return RE_MATH_BLOCKS.sub(replace_in_math_block, text)


def perturb_variable_swap(text: str) -> str:
    """Swap the most common mathematical variable with an unused one."""
    math_vars = _get_math_variables(text)
    candidates = [
        ("x", "t"),
        ("x", "u"),
        ("n", "k"),
        ("n", "m"),
        ("y", "w"),
        ("a", "u"),
        ("b", "v"),
        ("k", "p"),
        ("m", "q"),
    ]
    for old_v, new_v in candidates:
        if old_v in math_vars and new_v not in math_vars:
            renamed = _rename_variable_in_math(text, old_v, new_v)
            if renamed != text:
                return renamed
    return text


def perturb_commutative_flip(text: str) -> str:
    """Flip simple two-sided equalities inside math blocks: $A = B$ -> $B = A$."""
    def flip_equation(match: re.Match) -> str:
        block = match.group(0)
        # Check if block is $left = right$ with exactly one equal sign
        if block.startswith("$") and block.endswith("$") and block.count("=") == 1:
            inner = block[1:-1].strip()
            # Avoid flipping inequalities or definitions
            if any(tok in inner for tok in ("<", ">", r"\le", r"\ge", r"\ne", ":=")):
                return block
            parts = inner.split("=")
            left, right = parts[0].strip(), parts[1].strip()
            if left and right and len(left) <= 30 and len(right) <= 30:
                return f"${right} = {left}$"
        return block

    flipped = re.sub(r"\$[^$=<\n]+=[^$=<\n]+\$", flip_equation, text, count=1)
    return flipped if flipped != text else text


def perturb_condition_reorder(text: str) -> str:
    """Reorder independent setup sentences before the final question."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) >= 3:
        # Swap the first two declarative sentences if neither ends with a question mark
        if not sentences[0].endswith("?") and not sentences[1].endswith("?"):
            reordered = [sentences[1], sentences[0]] + sentences[2:]
            return " ".join(reordered)
    return text


def perturb_syntactic(text: str) -> str:
    """Apply LaTeX syntactic equivalences."""
    variant = text
    for pattern, replacement in SYNTACTIC_REPLACEMENTS:
        new_variant, count = pattern.subn(replacement, variant)
        if count > 0:
            variant = new_variant
            break
    return variant


def perturb_framing(text: str) -> str:
    """Apply question framing substitutions."""
    variant = text
    for pattern, replacement in FRAMING_REPLACEMENTS:
        new_variant, count = pattern.subn(replacement, variant, count=1)
        if count > 0:
            variant = new_variant
            break
    return variant


def generate_contrastive_variant(problem_text: str) -> str:
    """Generate a single semantically equivalent, perturbed variant of the problem text.

    Args:
        problem_text: Original mathematical problem string.

    Returns:
        Perturbed problem string representing the identical mathematical problem.
    """
    if not problem_text or not isinstance(problem_text, str):
        return problem_text

    # Apply both variable swapping and LaTeX syntactic equivalences sequentially
    variant = perturb_variable_swap(problem_text)
    variant = perturb_syntactic(variant)

    if variant == problem_text:
        variant = perturb_framing(variant)

    if variant == problem_text:
        variant = perturb_commutative_flip(variant)

    if variant == problem_text:
        variant = perturb_condition_reorder(variant)

    if variant == problem_text:
        variant = f"Carefully solve: {problem_text}"

    return variant


def generate_variants(problem_text: str, n: int = 3) -> list[str]:
    """Generate up to n distinct meaning-preserving perturbed variants."""
    if not problem_text or not isinstance(problem_text, str):
        return [problem_text] * n

    pool: list[str] = []

    # Variant 1: Variable swap + syntax
    v1 = perturb_variable_swap(problem_text)
    v1 = perturb_syntactic(v1)
    if v1 != problem_text and v1 not in pool:
        pool.append(v1)

    # Variant 2: Commutative flip + framing
    v2 = perturb_commutative_flip(problem_text)
    v2 = perturb_framing(v2)
    if v2 != problem_text and v2 not in pool:
        pool.append(v2)

    # Variant 3: Condition reordering + variable swap
    v3 = perturb_condition_reorder(problem_text)
    v3 = perturb_variable_swap(v3)
    if v3 != problem_text and v3 not in pool:
        pool.append(v3)

    # Fallback variants if pool has fewer than n
    fallbacks = [
        perturb_framing(problem_text),
        perturb_syntactic(problem_text),
        f"Carefully evaluate: {problem_text}",
        f"Determine the answer: {problem_text}",
    ]
    for fb in fallbacks:
        if len(pool) >= n:
            break
        if fb != problem_text and fb not in pool:
            pool.append(fb)

    while len(pool) < n:
        pool.append(f"Carefully solve: {problem_text}")

    return pool[:n]
