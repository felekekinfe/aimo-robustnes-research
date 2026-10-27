"""Lightweight Causal Graph & Semantic Node Parser for Mathematical Problems.

Decomposes an Olympiad mathematical problem into causal structural components:
- Surface Narrative (S): Contextual framing, story elements, or setup preamble.
- Numbers & Operands (N): Numeric values, constants, and relational operators.
- Mathematical Constraints (G): Inline LaTeX equations, conditions, and predicates.
- Question Clause (Q): The exact target quantity or property to be determined.
"""

from __future__ import annotations

import re
from typing import Any

MATH_SPAN_RE = re.compile(
    r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)"
)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
QUESTION_STARTERS = re.compile(
    r"\b(find|determine|compute|calculate|what\s+is|what\s+are|prove|show|evaluate|how\s+many)\b",
    re.IGNORECASE,
)


def _split_sentences_preserving_math(text: str) -> list[str]:
    """Split text into sentences while strictly preserving math spans from premature splitting."""
    spans: list[str] = []

    def repl(m: re.Match) -> str:
        idx = len(spans)
        spans.append(m.group(0))
        return f"__MATH_NODE_{idx}__"

    masked = MATH_SPAN_RE.sub(repl, text.strip())
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", masked) if s.strip()]

    # Restore math spans
    sentences: list[str] = []
    for s in raw_sentences:
        for idx, span in enumerate(spans):
            s = s.replace(f"__MATH_NODE_{idx}__", span)
        sentences.append(s)

    return sentences if sentences else [text.strip()]


def extract_causal_nodes(problem: str) -> dict[str, Any]:
    """Extract causal structural nodes from mathematical problem text.

    Returns:
        dict containing:
            - surface_text: Narrative framing / introductory prose (S).
            - numbers: List of numeric values and operands (N).
            - math_spans: List of inline/display LaTeX expressions (G).
            - question: Final question clause asking for the target answer (Q).
            - sentences: Ordered list of all segmented problem sentences.
            - raw_problem: The original problem text.
    """
    if not problem or not isinstance(problem, str):
        return {
            "surface_text": "",
            "numbers": [],
            "math_spans": [],
            "question": "Find the value.",
            "sentences": [],
            "raw_problem": str(problem or ""),
        }

    raw = problem.strip()
    math_spans = [m.group(0) for m in MATH_SPAN_RE.finditer(raw)]
    numbers = NUMBER_RE.findall(raw)
    sentences = _split_sentences_preserving_math(raw)

    # 1. Identify Question Node (Q)
    question = sentences[-1] if sentences else raw
    if len(sentences) > 1 and not QUESTION_STARTERS.search(question) and "?" not in question:
        for s in reversed(sentences):
            if QUESTION_STARTERS.search(s) or "?" in s:
                question = s
                break

    # 2. Identify Surface Narrative Node (S)
    surface_text = ""
    constraint_sentences = []
    for s in sentences:
        if s == question:
            continue
        # If sentence has no LaTeX math spans or minimal math, it is narrative framing
        s_maths = MATH_SPAN_RE.findall(s)
        if len(s_maths) == 0 and not surface_text:
            surface_text = s
        else:
            constraint_sentences.append(s)

    if not surface_text and sentences:
        # If every sentence has math, the first sentence serves as premise framing
        surface_text = sentences[0]

    return {
        "surface_text": surface_text,
        "numbers": numbers,
        "math_spans": math_spans,
        "question": question,
        "constraint_sentences": constraint_sentences,
        "sentences": sentences,
        "raw_problem": raw,
    }
