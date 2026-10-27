"""Symbolic Extractor: Formal Mathematical Parser & Token-Span Mapper.

Pillar 2 of Neuro-Symbolic Causal AI:
- Deterministic extraction of formal mathematical AST (equations, variables, relations).
- Exact character interval partitioning: Formal Math Constraints (G) vs Narrative Story (S).
- Token-level alignment mapping character intervals to token positions in input_ids.
- Invariant structural analysis (geometry, number theory, symmetry, diophantine equations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Set, Tuple


MATH_DELIM_RE = re.compile(
    r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)"
)
VARIABLE_RE = re.compile(r"(?<![\\a-zA-Z])([a-zA-Z])(?![a-zA-Z])")
RELATION_RE = re.compile(r"(=|\\le|\\ge|\\leq|\\geq|<|>|\\equiv|\\mid|\\approx|\\neq)")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

GEOMETRY_KEYWORDS = {
    "triangle", "circle", "radius", "diameter", "circumcircle", "incenter",
    "centroid", "orthocenter", "perpendicular", "parallel", "angle", "polygon",
    "quadrilateral", "tangent", "chord", "hypotenuse", "vertex", "vertices",
    "collinear", "concyclic", "bisector", "area", "perimeter", "sphere",
}

NUMBER_THEORY_KEYWORDS = {
    "prime", "primes", "divisible", "divisor", "divisors", "gcd", "lcm",
    "modular", "modulo", "mod", "congruent", "congruence", "remainder",
    "integer", "integers", "coprime", "diophantine", "factorization",
}


@dataclass
class FormalMathAST:
    """Formal mathematical Abstract Syntax Tree representation of a problem."""
    variables: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    math_char_spans: list[tuple[int, int]] = field(default_factory=list)
    story_char_spans: list[tuple[int, int]] = field(default_factory=list)
    constraint_density: float = 0.0
    is_geometry: bool = False
    is_number_theory: bool = False
    is_algebra: bool = False
    has_cyclic_symmetry: bool = False
    is_diophantine: bool = False
    raw_problem: str = ""


def extract_formal_ast(problem: str) -> FormalMathAST:
    """Parse problem text into formal mathematical AST without external heavy binaries."""
    if not problem or not isinstance(problem, str):
        return FormalMathAST(raw_problem=str(problem or ""))

    raw = problem.strip()
    n_chars = len(raw)
    if n_chars == 0:
        return FormalMathAST()

    # 1. Identify all formal math spans (G) and their exact char spans
    math_matches = list(MATH_DELIM_RE.finditer(raw))
    math_char_spans: list[tuple[int, int]] = [m.span() for m in math_matches]

    # If no LaTeX delimiters present, find standalone equations/numbers
    if not math_char_spans:
        for m in re.finditer(r"\b[a-zA-Z]\s*=\s*[-+]?\d+\b", raw):
            math_char_spans.append(m.span())
        if not math_char_spans:
            for m in NUMBER_RE.finditer(raw):
                math_char_spans.append(m.span())

    # 2. Derive narrative story spans (S) as the complement intervals
    story_char_spans: list[tuple[int, int]] = []
    curr = 0
    for start, end in sorted(math_char_spans):
        if start > curr:
            story_char_spans.append((curr, start))
        curr = max(curr, end)
    if curr < n_chars:
        story_char_spans.append((curr, n_chars))

    # 3. Extract formal math content from math spans
    math_texts = [raw[s:e] for s, e in math_char_spans]
    math_blob = " ".join(math_texts)

    # Variables (filter out LaTeX command words)
    raw_vars = VARIABLE_RE.findall(math_blob)
    latex_escapes = set("abcdefghijklmnopqrstuvwxyz") - {"x", "y", "z", "n", "k", "m", "a", "b", "c", "p", "q", "r", "s", "t", "u", "v", "w"}
    variables = sorted(list(set(v for v in raw_vars if len(v) == 1 and v.lower() not in {"e", "d", "i"} or v in {"x", "y", "z", "n", "k"})))

    # Equations and relations
    equations: list[str] = []
    relations: list[str] = []
    for mt in math_texts:
        clean_mt = mt.strip("$").strip(r"\[\]\(\)")
        if RELATION_RE.search(clean_mt):
            equations.append(clean_mt)
        rels = RELATION_RE.findall(clean_mt)
        relations.extend(rels)

    # Numbers
    numbers = NUMBER_RE.findall(raw)

    # Constraint density: ratio of math characters to total characters
    total_math_chars = sum(e - s for s, e in math_char_spans)
    constraint_density = float(total_math_chars / max(1, n_chars))

    # Domain Classification
    raw_lower = raw.lower()
    is_geom = any(kw in raw_lower for kw in GEOMETRY_KEYWORDS)
    is_nt = any(kw in raw_lower for kw in NUMBER_THEORY_KEYWORDS)
    is_alg = bool(len(equations) > 0 and not is_geom)

    # Invariants
    has_cyclic = (
        ("x" in variables and "y" in variables and "z" in variables)
        or ("a" in variables and "b" in variables and "c" in variables)
    ) and any("+" in eq and ("=" in eq or r"\le" in eq or r"\ge" in eq) for eq in equations)

    is_diophantine = is_nt and ("integer" in raw_lower or "positive" in raw_lower) and len(equations) > 0

    return FormalMathAST(
        variables=variables,
        equations=equations,
        relations=sorted(list(set(relations))),
        numbers=numbers,
        math_char_spans=math_char_spans,
        story_char_spans=story_char_spans,
        constraint_density=constraint_density,
        is_geometry=is_geom,
        is_number_theory=is_nt,
        is_algebra=is_alg,
        has_cyclic_symmetry=has_cyclic,
        is_diophantine=is_diophantine,
        raw_problem=raw,
    )


def get_math_and_story_token_indices(
    tokenizer: Any,
    prompt: str,
    math_char_spans: list[tuple[int, int]],
) -> tuple[list[int], list[int]]:
    """Map character intervals of formal math (G) and narrative (S) to token indices.

    Guarantees non-empty index lists for both G and S.
    """
    if not prompt:
        return [0], [0]

    # Attempt FastTokenizer offset mapping
    try:
        enc = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
        if "offset_mapping" in enc and enc["offset_mapping"] is not None:
            offsets = enc["offset_mapping"][0].tolist()
            math_indices: list[int] = []
            story_indices: list[int] = []

            for idx, (t_start, t_end) in enumerate(offsets):
                if t_start == t_end:
                    continue  # Special token
                # Check overlap with any math span
                is_math = any(max(t_start, m_s) < min(t_end, m_e) for m_s, m_e in math_char_spans)
                if is_math:
                    math_indices.append(idx)
                else:
                    story_indices.append(idx)

            if math_indices and story_indices:
                return math_indices, story_indices
    except Exception:
        pass

    # Robust token-decoding fallback
    try:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    except Exception:
        return [0], [0]

    n_tokens = len(input_ids)
    if n_tokens <= 1:
        return [0], [0]

    math_indices = []
    story_indices = []
    curr_char = 0

    for idx, tid in enumerate(input_ids):
        try:
            tok_str = tokenizer.decode([tid])
        except Exception:
            tok_str = ""

        clean_tok = tok_str.strip()
        if not clean_tok:
            story_indices.append(idx)
            continue

        pos = prompt.find(clean_tok, curr_char)
        if pos == -1:
            pos = curr_char
        end_pos = pos + len(clean_tok)
        curr_char = pos

        is_math = any(max(pos, m_s) < min(end_pos, m_e) for m_s, m_e in math_char_spans)
        if is_math:
            math_indices.append(idx)
        else:
            story_indices.append(idx)

    # Fallback to prevent empty partition
    if not math_indices and story_indices:
        # Assign tokens containing math chars to math
        new_story = []
        for idx in story_indices:
            try:
                t_text = tokenizer.decode([input_ids[idx]])
            except Exception:
                t_text = ""
            if any(c in t_text for c in "0123456789+-*/=^_{}\\$"):
                math_indices.append(idx)
            else:
                new_story.append(idx)
        story_indices = new_story

    if not math_indices:
        math_indices = [idx for idx in range(n_tokens // 2, n_tokens)]
    if not story_indices:
        story_indices = [idx for idx in range(0, max(1, n_tokens // 2))]

    return math_indices, story_indices
