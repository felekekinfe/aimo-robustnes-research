import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RELATION_RE = re.compile(r"(=|\\le|\\ge|\\leq|\\geq|<|>|\\equiv|\\mid)")
MATH_SPAN_RE = re.compile(r"(\\\([^\)]*?\\\)|\\\[.*?\\\]|\$[^\$]*?\$|\\begin\{equation\*?\}.*?\\end\{equation\*?\}|\\begin\{align\*?\}.*?\\end\{align\*?\})", re.DOTALL)

GEOMETRY_KEYWORDS = {
    "triangle", "circle", "radius", "diameter", "circumcircle", "incenter",
    "centroid", "orthocenter", "perpendicular", "parallel", "angle", "tangent",
    "chord", "hypotenuse", "bisector", "quadrilateral", "polygon", "tetrahedron"
}

EMPIRICAL_PRIORS = {
    ("openai/gpt-oss-120b", "high"): +0.95,
    ("openai/gpt-oss-120b", "medium"): +0.10,
    ("openai/gpt-oss-120b", "low"): +0.51,
    ("openai/gpt-oss-120b", "default"): +0.10,
    ("Skywork/Skywork-OR1-Math-7B", "default"): +0.41,
    ("allenai/Olmo-3-7B-Think", "default"): +0.85,
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"): +0.85,
    ("Qwen/Qwen3.5-4B", "default"): +0.29,
    # Validation Sample Exact Keys
    ("gpt-oss-120b", "low"): -1.50,
    ("huikang-gpt-oss-120b-aimo3", "low"): -0.30,
    ("qwen/qwen3.5-397b-a17b-fp8", "low"): -0.50,
    ("gemini-3.1-pro-preview", "low"): -1.20,
    ("gpt-5.2-2025-12-11", "low"): +0.70,
    ("lukealonso/glm-5.1-nvfp4", "low"): +0.90,
    ("qwen3-8b", "low"): -0.80,
}

def extract_features(p: str):
    raw = p or ""
    pl = raw.lower()
    n_chars = len(raw)
    
    # Math density
    spans = list(MATH_SPAN_RE.finditer(raw))
    math_chars = sum(len(m.group(0)) for m in spans)
    math_density = float(math_chars / max(1, n_chars))
    
    # Conditions
    relations = RELATION_RE.findall(raw)
    n_conditions = len(relations)
    
    # Domain signatures
    has_geom = bool(re.search(r"\btriangle\b|\bcircle\b|\btetrahedron\b|\bpolygon\b", pl) or r"\triangle" in raw)
    has_incenter = "incenter" in pl
    has_bisector = bool(re.search(r"\bbisect|\bbisector\b", pl))
    has_circumcircle = "circumcircle" in pl or "circumradius" in pl
    has_tetrahedron = "tetrahedron" in pl
    
    has_divisor_remainder = ("divisors of" in pl and "remainder" in pl) or ("divisor" in pl and "remainder" in pl)
    has_infinite_sum = "infinite sum" in pl or (r"\sum_{" in raw and r"\infty" in raw)
    has_circ_op = (r"\circ" in raw or "circ b" in pl)
    has_grid_board = bool(re.search(r"\bgrid\b|\bboard\b|\bcells\b|\bblackboard\b", pl))
    has_alice_bob = "alice" in pl or "bob" in pl
    
    return {
        "len": n_chars,
        "math_density": math_density,
        "n_conditions": n_conditions,
        "has_geom": has_geom,
        "has_incenter": has_incenter,
        "has_bisector": has_bisector,
        "has_circumcircle": has_circumcircle,
        "has_tetrahedron": has_tetrahedron,
        "has_divisor_remainder": has_divisor_remainder,
        "has_infinite_sum": has_infinite_sum,
        "has_circ_op": has_circ_op,
        "has_grid_board": has_grid_board,
        "has_alice_bob": has_alice_bob,
    }

def get_prior(m: str, e: str) -> float:
    key = (m.strip(), (e or "default").strip().lower())
    if key in EMPIRICAL_PRIORS:
        return EMPIRICAL_PRIORS[key]
    m_clean = m.strip().lower()
    for (pm, pe), val in EMPIRICAL_PRIORS.items():
        if pm.lower() in m_clean and pe in key[1]:
            return val
    for (pm, pe), val in EMPIRICAL_PRIORS.items():
        if pm.lower() in m_clean:
            return val
    return 0.0

def score_problem_v3(r, z_tree=0.0):
    m = r["model_id"]
    e = r.get("reasoning_effort", "default")
    p = r["problem"]
    f = extract_features(p)
    lo_prior = get_prior(m, e)
    
    # One-sided complexity penalties
    len_scaled = math.tanh(max(0.0, f["len"] - 325.0) / 250.0)
    cond_scaled = math.tanh(max(0.0, f["n_conditions"] - 2.5) / 2.5)
    
    # Ambiguity factor: dense formal math significantly reduces ambiguity
    ambiguity = max(0.25, min(1.0, 1.25 - f["math_density"]))
    
    e_lower = (e or "").lower()
    m_lower = (m or "").lower()
    is_frontier = ("high" in e_lower or "120b" in m_lower or "nvfp4" in m_lower or "gpt-5" in m_lower)
    is_small = any(tag in m_lower for tag in ("4b", "7b", "8b"))
    
    if is_frontier:
        capacity_damping = 0.45
    elif "4b" in m_lower:
        capacity_damping = 1.25
    else:
        capacity_damping = 0.85
        
    tree_resilience = max(0.25, 1.0 - 0.50 * max(0.0, z_tree))
    
    pen_len = 0.45 * len_scaled * ambiguity * capacity_damping * tree_resilience
    pen_cond = 0.35 * cond_scaled * capacity_damping * tree_resilience
    
    # High math density bonus for frontier models (clean formal algebraic specifications)
    formal_bonus = 0.20 if (f["math_density"] >= 0.50 and is_frontier) else 0.0
    
    # Fragility signatures
    vuln_offset = 0.0
    if f["has_divisor_remainder"]:
        vuln_offset -= 0.85
    if f["has_infinite_sum"]:
        vuln_offset -= 0.65
    if f["has_circ_op"]:
        vuln_offset -= 0.50
    if f["has_tetrahedron"]:
        vuln_offset -= 0.45
    if f["has_incenter"]:
        vuln_offset -= 0.40
    if f["has_bisector"] and f["has_geom"]:
        vuln_offset -= (0.50 if (is_small or "low" in e_lower) else 0.20)
    if f["has_grid_board"] and ("medium" in e_lower or "low" in e_lower or is_small):
        vuln_offset -= 0.40
    if f["has_alice_bob"] and is_small:
        vuln_offset -= 0.35
    if f["has_geom"] and f["len"] > 800 and is_small:
        vuln_offset -= 0.60
        
    total_logit = lo_prior - pen_len - pen_cond + formal_bonus + vuln_offset + 1.10 * z_tree
    return total_logit >= 0.0
