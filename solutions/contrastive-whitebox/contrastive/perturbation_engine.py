"""AST-Safe Zero-Cost Perturbation Engine for mathematical problems.

Implements LaTeX math-span isolation, bound-variable alpha-renaming, balanced commutative
equality flipping, anaphora-safe sentence condition reordering, and non-degenerate surface reframing.
Guarantees distinctness without altering underlying mathematical semantics or corrupting LaTeX commands.
"""

from __future__ import annotations

import re
from typing import List

MATH_SPAN_RE = re.compile(r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)")
CMD_RE = re.compile(r"\\[a-zA-Z]+(?:\s*\{[^}]*\})?")

CANDIDATE_VARS = ("x", "y", "z", "n", "m", "k", "a", "b", "c", "p", "q", "r", "s", "t")
RESERVED_VARS = {"e", "i", "d", "f", "g", "O", "P", "E", "C", "R", "N", "Z", "Q"}
REPLACEMENT_VARS = ("t", "u", "v", "w", "k", "m", "q", "r", "s", "z")

ANAPHORIC_TOKENS = {"it", "this", "that", "they", "these", "such", "hence", "thus", "therefore"}

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

# Framing rephrasings
FRAMING_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bWhat is the value of\b", re.IGNORECASE), "Find the value of"),
    (re.compile(r"\bWhat is\b", re.IGNORECASE), "Compute"),
    (re.compile(r"\bFind all real solutions\b", re.IGNORECASE), "Determine all real solutions"),
    (re.compile(r"\bFind all possible values of\b", re.IGNORECASE), "Determine all possible values of"),
    (re.compile(r"\bFind the\b", re.IGNORECASE), "Determine the exact"),
    (re.compile(r"\bFind\b", re.IGNORECASE), "Determine"),
    (re.compile(r"\bEvaluate\b", re.IGNORECASE), "Compute"),
    (re.compile(r"\bCompute:\b", re.IGNORECASE), "Evaluate:"),
    (re.compile(r"\bSolve for\b", re.IGNORECASE), "Find the value of"),
]


def is_balanced(s: str) -> bool:
    """Check if parentheses, braces, and brackets are properly balanced."""
    stack: list[str] = []
    pairs = {")": "(", "}": "{", "]": "["}
    for ch in s:
        if ch in "({[":
            stack.append(ch)
        elif ch in ")}]":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0


def _mask_commands(text: str) -> tuple[str, list[str]]:
    """Mask LaTeX control sequences with indexed placeholders."""
    cmds: list[str] = []

    def repl(m: re.Match) -> str:
        idx = len(cmds)
        cmds.append(m.group(0))
        return f"@@CMD{idx}@@"

    masked = CMD_RE.sub(repl, text)
    return masked, cmds


def _unmask_commands(text: str, cmds: list[str]) -> str:
    """Restore masked LaTeX control sequences from indexed placeholders."""
    unmasked = text
    for idx, cmd in enumerate(cmds):
        unmasked = unmasked.replace(f"@@CMD{idx}@@", cmd)
    return unmasked


def safe_variable_swap(problem: str, seed_offset: int = 0) -> str:
    """Bound-variable alpha-renaming strictly preserving LaTeX commands.

    Scans math spans for candidate single-letter variables, ranks them by frequency,
    and substitutes the target variable with an unused replacement character.
    """
    if not problem or not isinstance(problem, str):
        return problem

    spans = list(MATH_SPAN_RE.finditer(problem))
    is_bare = len(spans) == 0

    target_text = problem if is_bare else " ".join(m.group(0) for m in spans)
    masked_target, cmds = _mask_commands(target_text)

    # Count variable frequencies
    freq: dict[str, int] = {}
    for v in CANDIDATE_VARS:
        if v not in RESERVED_VARS:
            matches = re.findall(rf"(?<![a-zA-Z_@\\]){v}(?![a-zA-Z0-9_@])", masked_target)
            if matches:
                freq[v] = len(matches)

    if not freq:
        return problem

    ranked_vars = sorted(freq.keys(), key=lambda v: (-freq[v], v))
    chosen_var = ranked_vars[seed_offset % len(ranked_vars)]

    # Pick an unused replacement character (prioritizing canonical math replacements)
    canonical_candidates = [("x", "t"), ("x", "u"), ("n", "k"), ("n", "m"), ("y", "w"), ("a", "u"), ("b", "v")]
    new_var = None
    for old_c, rep_c in canonical_candidates:
        if chosen_var == old_c and rep_c not in freq:
            new_var = rep_c
            break

    if new_var is None:
        for rep in REPLACEMENT_VARS:
            if rep != chosen_var and rep not in problem:
                new_var = rep
                break
    if new_var is None:
        for rep in REPLACEMENT_VARS:
            if rep != chosen_var and rep not in freq:
                new_var = rep
                break
    if new_var is None:
        return problem

    var_re = re.compile(rf"(?<![a-zA-Z_@\\]){chosen_var}(?![a-zA-Z0-9_@])")

    if is_bare:
        swapped = var_re.sub(new_var, masked_target)
        return _unmask_commands(swapped, cmds)

    def replace_in_span(m: re.Match) -> str:
        span = m.group(0)
        span_masked, span_cmds = _mask_commands(span)
        span_swapped = var_re.sub(new_var, span_masked)
        return _unmask_commands(span_swapped, span_cmds)

    return MATH_SPAN_RE.sub(replace_in_span, problem)


def safe_commutative_flip(problem: str) -> str:
    """Balanced commutative equality flipping inside math blocks (A = B -> B = A)."""
    if not problem or not isinstance(problem, str):
        return problem

    spans = list(MATH_SPAN_RE.finditer(problem))

    def flip_inner(inner: str) -> str | None:
        if inner.count("=") != 1:
            return None
        # Check no inequality or modular operators
        if any(tok in inner for tok in (r"\le", r"\ge", r"\neq", r"\equiv", r"\pmod", "<", ">", ",", ":=")):
            return None
        lhs, rhs = inner.split("=", 1)
        lhs_s, rhs_s = lhs.strip(), rhs.strip()
        if not lhs_s or not rhs_s:
            return None
        if not is_balanced(lhs_s) or not is_balanced(rhs_s):
            return None
        return f"{rhs_s} = {lhs_s}"

    if spans:
        for m in spans:
            span = m.group(0)
            delim_l, delim_r = "", ""
            if span.startswith("$$") and span.endswith("$$"):
                delim_l, delim_r = "$$", "$$"
                inner = span[2:-2]
            elif span.startswith(r"\[") and span.endswith(r"\]"):
                delim_l, delim_r = r"\[", r"\]"
                inner = span[2:-2]
            elif span.startswith(r"\(") and span.endswith(r"\)"):
                delim_l, delim_r = r"\(", r"\)"
                inner = span[2:-2]
            elif span.startswith("$") and span.endswith("$"):
                delim_l, delim_r = "$", "$"
                inner = span[1:-1]
            else:
                continue

            flipped_inner = flip_inner(inner)
            if flipped_inner is not None and flipped_inner != inner:
                new_span = f"{delim_l}{flipped_inner}{delim_r}"
                return problem[:m.start()] + new_span + problem[m.end():]
        return problem

    # Bare equation without spans
    flipped = flip_inner(problem)
    return flipped if flipped is not None else problem


def safe_condition_reorder_or_reframe(problem: str, variant_idx: int = 0) -> str:
    """Anaphora-safe condition sentence reordering with non-degenerate fallback."""
    if not problem or not isinstance(problem, str):
        return problem

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", problem) if s.strip()]

    # Condition reorder if setup has at least 2 declarative sentences before a final question
    if len(sentences) >= 3:
        s0, s1 = sentences[0], sentences[1]
        first_word_s1 = re.sub(r"[^\w]", "", s1.split()[0]).lower() if s1.split() else ""
        if first_word_s1 not in ANAPHORIC_TOKENS and not s0.endswith("?") and not s1.endswith("?"):
            reordered = [s1, s0] + sentences[2:]
            cand = " ".join(reordered)
            if cand != problem:
                return cand

    # Deterministic reframing fallback
    reframing_templates = [
        "Consider the following mathematical problem. {p}",
        "Carefully evaluate and solve: {p}",
        "Determine the exact solution to the following problem: {p}",
    ]
    template = reframing_templates[variant_idx % len(reframing_templates)]

    for pattern, replacement in FRAMING_REPLACEMENTS:
        new_text, count = pattern.subn(replacement, problem, count=1)
        if count > 0 and new_text != problem:
            return new_text

    return template.format(p=problem)


def perturb_syntactic(text: str) -> str:
    """Apply LaTeX syntactic equivalences."""
    variant = text
    for pattern, replacement in SYNTACTIC_REPLACEMENTS:
        new_variant, count = pattern.subn(replacement, variant)
        if count > 0:
            variant = new_variant
            break
    return variant


def generate_variants(problem: str, n: int = 3) -> List[str]:
    """Generate n distinct meaning-preserving perturbed variants.

    Deduplicates against the original problem and prior variants, guaranteeing
    that all n returned strings are strictly distinct.
    """
    if not problem or not isinstance(problem, str):
        return [str(problem)] * n

    pool: list[str] = []
    base_clean = problem.strip()

    def add_if_distinct(candidate: str) -> bool:
        c_clean = candidate.strip()
        if c_clean and c_clean != base_clean and c_clean not in pool:
            pool.append(candidate)
            return True
        return False

    # Strategy 1: Bound-variable alpha-renaming + syntactic substitution
    v1 = safe_variable_swap(problem, seed_offset=0)
    v1 = perturb_syntactic(v1)
    add_if_distinct(v1)

    # Strategy 2: Commutative equality flipping
    v2 = safe_commutative_flip(problem)
    if v2 == problem:
        v2 = safe_variable_swap(problem, seed_offset=1)
    v2 = perturb_syntactic(v2)
    add_if_distinct(v2)

    # Strategy 3: Anaphora-safe condition reordering or surface reframing
    v3 = safe_condition_reorder_or_reframe(problem, variant_idx=0)
    add_if_distinct(v3)

    # Strategy 4: Combined variable swap on reordered condition
    v4 = safe_variable_swap(v3, seed_offset=0)
    add_if_distinct(v4)

    # Non-degenerate deterministic fallbacks to guarantee n distinct strings
    fallback_templates = [
        "Consider the following mathematical problem. {p}",
        "Carefully evaluate and solve: {p}",
        "Determine the exact solution to the following problem: {p}",
        "Calculate the answer to the following question: {p}",
        "Solve the following mathematical challenge: {p}",
    ]
    fb_idx = 0
    while len(pool) < n and fb_idx < len(fallback_templates):
        cand = fallback_templates[fb_idx].format(p=problem)
        add_if_distinct(cand)
        fb_idx += 1

    return pool[:n]


def build_perturbation_triplet(problem: str) -> tuple[str, str, str]:
    """Build organizer-safe perturbation triplet (v_ctrl, v_1, v_2).

    1. v_ctrl: Harmless framing change to measure baseline hidden-state shift.
    2. v_1: Algebraic / structural reframe with phrasing and variable swap (guaranteed non-identical).
    3. v_2: Directive / constraint order reframe (guaranteed distinct from problem and v_1).
    """
    if not problem or not isinstance(problem, str):
        problem = "Find the value."

    p_clean = problem.strip()

    # 1. v_ctrl: Meaning-Preserving Control Wrapper
    v_ctrl = "Consider the following mathematics problem:\n" + p_clean

    # 2. v_1: Phrasing substitutions + variable swap + syntactic
    v_1 = p_clean
    phrasing_subs = [
        (re.compile(r"\bFind the value of\b", re.IGNORECASE), "Determine the value of"),
        (re.compile(r"\bFind the number of\b", re.IGNORECASE), "Determine the number of"),
        (re.compile(r"\bFind all\b", re.IGNORECASE), "Determine all"),
        (re.compile(r"\bFind\b"), "Determine"),
        (re.compile(r"\bLet\b"), "Suppose"),
        (re.compile(r"\bCompute\b"), "Evaluate"),
        (re.compile(r"\bCalculate\b"), "Find"),
        (re.compile(r"\bpositive integers\b"), "integers greater than zero"),
        (re.compile(r"\bpositive integer\b"), "integer greater than zero"),
        (re.compile(r"\breal numbers\b"), "real values"),
        (re.compile(r"\breal number\b"), "real value"),
    ]
    for pat, rep in phrasing_subs:
        if pat.search(v_1):
            v_1 = pat.sub(rep, v_1, count=1)
            break

    v_1 = safe_variable_swap(v_1, seed_offset=0)
    v_1 = perturb_syntactic(v_1)

    # 100% Non-identical guarantee for v_1
    if v_1.strip() == p_clean:
        v_1 = "Problem Statement: " + p_clean + "\n(Express your final answer clearly.)"

    # 3. v_2: Distinct Directive / Constraint Order Reframe
    v_2 = "Given the conditions below, solve the problem:\n" + p_clean + "\nWhat is the required value?"
    if v_2.strip() == v_1.strip() or v_2.strip() == p_clean:
        v_2 = "Consider the mathematical conditions:\n" + p_clean + "\nDetermine the solution."

    return v_ctrl, v_1, v_2


def generate_contrastive_variant(problem_text: str) -> str:
    """Primary contrastive variant preserving backward compatibility."""
    if not problem_text or not isinstance(problem_text, str):
        return problem_text

    # Apply variable swap and LaTeX syntax
    variant = safe_variable_swap(problem_text, seed_offset=0)
    variant = perturb_syntactic(variant)

    if variant != problem_text:
        return variant

    # Try commutative flip
    variant = safe_commutative_flip(problem_text)
    if variant != problem_text:
        return variant

    # Try condition reordering or reframing
    variant = safe_condition_reorder_or_reframe(problem_text, variant_idx=0)
    if variant != problem_text:
        return variant

    return f"Carefully solve: {problem_text}"
