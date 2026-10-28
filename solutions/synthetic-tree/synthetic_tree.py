"""On-the-Fly Synthetic 7-Type Tree Generator for AIMO Mathematical Problems.

Generates 14 distinct synthetic branch variants across all 7 official perturbation categories:
  Group 1 (Surface & Noise):
    1. typo_char: Subtle typographical error in non-math English word
    2. typo_delim: LaTeX command or delimiter syntactic substitution
    3. format_delims: Inline vs display LaTeX delimiter swap ($...$ <-> \(...\))
    4. format_whitespace: Spacing and paragraph line break adjustments
  Group 2 (Representation & Syntax):
    5. rename_algebraic: Bound-variable alpha-renaming ($x, y$ -> $t, u$, $n$ -> $k$)
    6. rename_geometric: Geometric vertex/point renaming ($A, B, C$ -> $P, Q, R$)
    7. rephrase_commute: Commutative equality flipping ($A = B$ -> $B = A$)
    8. rephrase_query: Question framing transformation ("Find" -> "Determine the exact value of")
  Group 3 (Context & Distraction):
    9. domain_academic: Formal academic preamble replacing narrative intro
    10. domain_entities: Story entity relabeling (marbles -> tokens, sweets -> items)
    11. distract_prepend: Benign domain definition prefix
    12. distract_append: Simplification advisory suffix
  Group 4 (Expert & Semantic Boundary):
    13. expert_ablate: Premise boundary condition ablation
    14. expert_mutate: Numerical parameter increment / relational stress

All variants are generated deterministically in pure Python without external dependencies.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

MATH_SPAN_RE = re.compile(r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)")
CMD_RE = re.compile(r"\\[a-zA-Z]+(?:\s*\{[^}]*\})?")

# Common non-math English words for typo injection
TYPO_CANDIDATES = [
    ("triangle", "traingle"),
    ("positive", "posistive"),
    ("integers", "intgers"),
    ("integer", "intger"),
    ("number", "numbr"),
    ("numbers", "numbrs"),
    ("divisible", "divisble"),
    ("multiple", "multipl"),
    ("possible", "possble"),
    ("consecutive", "consecutve"),
    ("following", "follwing"),
    ("problem", "problm"),
    ("company", "compny"),
    ("product", "prodct"),
]

# Entity word substitutions
ENTITY_REPLACEMENTS = [
    (re.compile(r"\bmarbles\b", re.IGNORECASE), "tokens"),
    (re.compile(r"\bmarble\b", re.IGNORECASE), "token"),
    (re.compile(r"\bsweets\b", re.IGNORECASE), "items"),
    (re.compile(r"\bsweet\b", re.IGNORECASE), "item"),
    (re.compile(r"\bflights\b", re.IGNORECASE), "departures"),
    (re.compile(r"\bflight\b", re.IGNORECASE), "departure"),
    (re.compile(r"\bcoins\b", re.IGNORECASE), "tokens"),
    (re.compile(r"\bplayers\b", re.IGNORECASE), "participants"),
]

# Query rephrasings
QUERY_REPLACEMENTS = [
    (re.compile(r"\bWhat is the value of\b", re.IGNORECASE), "Find the exact value of"),
    (re.compile(r"\bWhat is\b", re.IGNORECASE), "Compute the value of"),
    (re.compile(r"\bFind the\b", re.IGNORECASE), "Determine the exact"),
    (re.compile(r"\bFind all\b", re.IGNORECASE), "Determine all"),
    (re.compile(r"\bFind\b", re.IGNORECASE), "Determine"),
    (re.compile(r"\bEvaluate\b", re.IGNORECASE), "Compute"),
    (re.compile(r"\bHow many\b", re.IGNORECASE), "Compute the total number of"),
]

# Syntactic LaTeX replacements
SYNTACTIC_REPLACEMENTS = [
    (re.compile(r"\\le\b"), r"\\leq"),
    (re.compile(r"\\ge\b"), r"\\geq"),
    (re.compile(r"\\cdot\b"), r"\\times"),
    (re.compile(r"\\dfrac\b"), r"\\frac"),
    (re.compile(r"\\pmod\b"), r"\\bmod"),
]


def _mask_commands(text: str) -> Tuple[str, List[str]]:
    cmds: List[str] = []
    def repl(m: re.Match) -> str:
        idx = len(cmds)
        cmds.append(m.group(0))
        return f"@@CMD{idx}@@"
    return CMD_RE.sub(repl, text), cmds


def _unmask_commands(text: str, cmds: List[str]) -> str:
    unmasked = text
    for idx, cmd in enumerate(cmds):
        unmasked = unmasked.replace(f"@@CMD{idx}@@", cmd)
    return unmasked


def generate_synthetic_tree(problem: str) -> Dict[str, str]:
    """Generate Root problem and 14 distinct variants across 7 official categories."""
    p = (problem or "").strip()
    if not p:
        p = "Find the value."

    variants: Dict[str, str] = {"root": p}

    # ----------------------------------------------------
    # 1. TYPOS (2 Variants)
    # ----------------------------------------------------
    # 1A: Lexical typo in English narrative
    v_typo_char = p
    for orig_w, typo_w in TYPO_CANDIDATES:
        if orig_w in v_typo_char.lower():
            v_typo_char = re.sub(rf"\b{orig_w}\b", typo_w, v_typo_char, count=1, flags=re.IGNORECASE)
            break
    if v_typo_char == p:
        # Fallback typo: duplicate a vowel in first English word
        words = p.split()
        if len(words) > 3 and not words[0].startswith("$"):
            words[0] = words[0] + "e"
            v_typo_char = " ".join(words)
        else:
            v_typo_char = p + " ."
    variants["typo_char"] = v_typo_char

    # 1B: Delimiter / syntactic variation
    v_typo_delim = p
    for pat, rep in SYNTACTIC_REPLACEMENTS:
        new_v, count = pat.subn(rep, v_typo_delim)
        if count > 0:
            v_typo_delim = new_v
            break
    if v_typo_delim == p:
        v_typo_delim = p.replace(r"\leq", r"\le").replace(r"\geq", r"\ge")
    if v_typo_delim == p:
        v_typo_delim = p.replace(r"\\", r"\newline ")
    variants["typo_delim"] = v_typo_delim

    # ----------------------------------------------------
    # 2. FORMATTING (2 Variants)
    # ----------------------------------------------------
    # 2A: LaTeX delimiter conversion ($...$ -> \(...\))
    def _swap_delim(m: re.Match) -> str:
        s = m.group(0)
        if s.startswith("$") and not s.startswith("$$") and s.endswith("$"):
            return r"\(" + s[1:-1] + r"\)"
        return s
    v_format_delims = MATH_SPAN_RE.sub(_swap_delim, p)
    if v_format_delims == p:
        v_format_delims = p.replace(r"\(", "$").replace(r"\)", "$")
    variants["format_delims"] = v_format_delims

    # 2B: Paragraph spacing & markdown block formatting
    variants["format_whitespace"] = "\n\n" + p.replace(". ", ".\n\n")

    # ----------------------------------------------------
    # 3. RENAME (2 Variants)
    # ----------------------------------------------------
    # 3A: Algebraic variable alpha-renaming (x -> t, n -> k)
    spans = list(MATH_SPAN_RE.finditer(p))
    v_rename_alg = p
    if spans:
        span_texts = [m.group(0) for m in spans]
        math_blob = " ".join(span_texts)
        masked_blob, cmds = _mask_commands(math_blob)
        
        target_var, rep_var = None, None
        for cand, rep in [("x", "t"), ("n", "k"), ("y", "u"), ("a", "m"), ("b", "q")]:
            if re.search(rf"(?<![a-zA-Z_@\\]){cand}(?![a-zA-Z0-9_@])", masked_blob):
                target_var, rep_var = cand, rep
                break
        
        if target_var and rep_var:
            def repl_var(m: re.Match) -> str:
                s, c_list = _mask_commands(m.group(0))
                s = re.sub(rf"(?<![a-zA-Z_@\\]){target_var}(?![a-zA-Z0-9_@])", rep_var, s)
                return _unmask_commands(s, c_list)
            v_rename_alg = MATH_SPAN_RE.sub(repl_var, p)
    if v_rename_alg == p:
        v_rename_alg = p.replace("$x$", "$t$").replace("$n$", "$k$")
    variants["rename_algebraic"] = v_rename_alg

    # 3B: Geometric point renaming (A, B, C -> P, Q, R)
    v_rename_geom = p
    if re.search(r"\b[A-C]\b", p) or "ABC" in p or r"\triangle ABC" in p:
        v_rename_geom = (
            p.replace("ABC", "PQR")
            .replace(r"\triangle ABC", r"\triangle PQR")
            .replace("AB", "PQ")
            .replace("BC", "QR")
            .replace("CA", "RP")
        )
    if v_rename_geom == p:
        # Fallback to swapping second variable
        v_rename_geom = p.replace("$a$", "$u$").replace("$y$", "$v$")
    variants["rename_geometric"] = v_rename_geom

    # ----------------------------------------------------
    # 4. REPHRASE (2 Variants)
    # ----------------------------------------------------
    # 4A: Commutative equality flipping (A = B -> B = A)
    v_commute = p
    for m in MATH_SPAN_RE.finditer(p):
        span = m.group(0)
        inner = span.strip("$").strip(r"\[\]\(\)")
        if inner.count("=") == 1 and not any(op in inner for op in (r"\le", r"\ge", "<", ">", r"\equiv")):
            lhs, rhs = inner.split("=", 1)
            if lhs.strip() and rhs.strip():
                flipped = f"{rhs.strip()} = {lhs.strip()}"
                v_commute = p[:m.start()] + span.replace(inner, flipped) + p[m.end():]
                break
    variants["rephrase_commute"] = v_commute

    # 4B: Query framing rewording
    v_query = p
    for pat, rep in QUERY_REPLACEMENTS:
        new_v, count = pat.subn(rep, v_query, count=1)
        if count > 0:
            v_query = new_v
            break
    if v_query == p:
        v_query = "Determine the exact solution: " + p
    variants["rephrase_query"] = v_query

    # ----------------------------------------------------
    # 5. DOMAIN (2 Variants)
    # ----------------------------------------------------
    # 5A: Formal academic framing replacing informal intro
    variants["domain_academic"] = (
        "Let the mathematical conditions and variables be formally specified as follows: " + p
    )

    # 5B: Entity substitution (marbles -> tokens, sweets -> items)
    v_entity = p
    for pat, rep in ENTITY_REPLACEMENTS:
        new_v, count = pat.subn(rep, v_entity)
        if count > 0:
            v_entity = new_v
    if v_entity == p:
        v_entity = "In an abstract mathematical universe, " + p
    variants["domain_entities"] = v_entity

    # ----------------------------------------------------
    # 6. DISTRACT (2 Variants)
    # ----------------------------------------------------
    # 6A: Benign domain definition prefix
    variants["distract_prepend"] = (
        "Recall that all computations are evaluated under standard mathematical axioms. " + p
    )

    # 6B: Clarification / simplification advisory suffix
    variants["distract_append"] = (
        p.rstrip() + " Provide the answer in its standard simplified numerical representation."
    )

    # ----------------------------------------------------
    # 7. EXPERT / SENSITIVITY CHECK (2 Variants)
    # ----------------------------------------------------
    # 7A: Premise boundary condition ablation
    v_expert_ablate = p
    if "positive" in v_expert_ablate:
        v_expert_ablate = v_expert_ablate.replace("positive ", "", 1)
    elif "integer" in v_expert_ablate:
        v_expert_ablate = v_expert_ablate.replace("integer ", "real ", 1)
    elif r"\ge" in v_expert_ablate:
        v_expert_ablate = v_expert_ablate.replace(r"\ge", ">", 1)
    elif "=" in v_expert_ablate:
        v_expert_ablate = v_expert_ablate.replace("=", r"\approx", 1)
    variants["expert_ablate"] = v_expert_ablate

    # 7B: Numerical parameter mutation (+1 increment to test logical sensitivity)
    v_expert_mutate = p
    num_matches = list(re.finditer(r"\b\d+\b", p))
    if num_matches:
        # Pick the second number or first
        target_m = num_matches[1] if len(num_matches) > 1 else num_matches[0]
        val = int(target_m.group(0))
        new_val_str = str(val + 1)
        s, e = target_m.span()
        v_expert_mutate = p[:s] + new_val_str + p[e:]
    variants["expert_mutate"] = v_expert_mutate

    return variants


def get_tree_branch_names() -> List[str]:
    """Ordered list of the 14 synthetic branch variant names."""
    return [
        "typo_char",
        "typo_delim",
        "format_delims",
        "format_whitespace",
        "rename_algebraic",
        "rename_geometric",
        "rephrase_commute",
        "rephrase_query",
        "domain_academic",
        "domain_entities",
        "distract_prepend",
        "distract_append",
        "expert_ablate",
        "expert_mutate",
    ]
