"""Hybrid Feature Extractor for AIMO Interpretability Challenge.

Combines signals from multiple approaches:
1. SCSP Contrastive Drift (4-prompt prefill: clean, preserve1, preserve2, destroy)
2. Synthetic Tree Expert Sensitivity (expert_ablate, expert_mutate drifts)
3. Early CoT Dynamics (first 64 tokens entropy, margin, volatility)
4. Structural Features (model-conditioned priors, cognitive load)

Optimized for:
- Single-GPU RTX 6000 96GB execution within < 3600s total budget
- Zero GPU OOM: torch.inference_mode, manual cache cleanup, prompt length bounding
- Graceful CPU fallback when CUDA is unavailable
- Offline execution: local_files_only=True
"""

from __future__ import annotations

import gc
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False

logger = logging.getLogger("AIMO-Hybrid-Extractor")

# =========================================================================
# REGEX PATTERNS (pre-compiled for speed)
# =========================================================================
MATH_DELIM_RE = re.compile(r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)")
CMD_RE = re.compile(r"\\[a-zA-Z]+(?:\s*\{[^}]*\})?")
RELATION_RE = re.compile(r"(=|\\le|\\ge|\\leq|\\geq|<|>|\\equiv|\\mid)")
VARIABLE_RE = re.compile(r"(?<![\\a-zA-Z])([a-zA-Z])(?![a-zA-Z])")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# Safe variable swap (outside LaTeX commands)
_SAFE_VAR_RE = re.compile(r"(?<!\\)\b([ABCPQRXYZabcpqrxyzuv])\b")
_VAR_SWAP_MAP = {
    "A": "P", "B": "Q", "C": "R", "P": "A", "Q": "B", "R": "C",
    "X": "U", "Y": "V", "Z": "W", "a": "p", "b": "q", "c": "r",
    "p": "a", "q": "b", "r": "c", "x": "u", "y": "v", "z": "w",
    "u": "x", "v": "y",
}
_INT_NUMBER_RE = re.compile(r"(?<!\\)\b(\d+)\b")

# Domain keywords
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
COMBINATORICS_KEYWORDS = {
    "permutation", "permutations", "combination", "combinations", "probability",
    "ways", "arrangements", "choose", "subset", "subsets", "graph",
}
ALGEBRA_KEYWORDS = {
    "polynomial", "polynomials", "roots", "coefficients", "quadratic", "cubic",
    "function", "functions", "inequality", "system of equations", "matrix",
    "determinant", "monic",
}

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================
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


def safe_variable_swap(text: str, seed_offset: int = 0) -> str:
    """Bound-variable alpha-renaming strictly preserving LaTeX commands."""
    if not text or not isinstance(text, str):
        return text

    spans = list(MATH_DELIM_RE.finditer(text))
    is_bare = len(spans) == 0

    target_text = text if is_bare else " ".join(m.group(0) for m in spans)
    masked_target, cmds = _mask_commands(target_text)

    freq: dict[str, int] = {}
    for v in "xyzabcpqrstuvwnmk":
        if v in _VAR_SWAP_MAP:
            matches = re.findall(rf"(?<![a-zA-Z_@\\]){v}(?![a-zA-Z0-9_@])", masked_target)
            if matches:
                freq[v] = len(matches)

    if not freq:
        return text

    ranked_vars = sorted(freq.keys(), key=lambda v: (-freq[v], v))
    chosen_var = ranked_vars[seed_offset % len(ranked_vars)]
    new_var = _VAR_SWAP_MAP.get(chosen_var, chosen_var)

    var_re = re.compile(rf"(?<![a-zA-Z_@\\]){chosen_var}(?![a-zA-Z0-9_@])")

    if is_bare:
        swapped = var_re.sub(new_var, masked_target)
        return _unmask_commands(swapped, cmds)

    def replace_in_span(m: re.Match) -> str:
        span = m.group(0)
        span_masked, span_cmds = _mask_commands(span)
        span_swapped = var_re.sub(new_var, span_masked)
        return _unmask_commands(span_swapped, span_cmds)

    return MATH_DELIM_RE.sub(replace_in_span, text)


def safe_commutative_flip(text: str) -> str:
    """Balanced commutative equality flipping inside math blocks (A = B -> B = A)."""
    if not text or not isinstance(text, str):
        return text

    def flip_inner(inner: str) -> str | None:
        if inner.count("=") != 1:
            return None
        if any(tok in inner for tok in (r"\le", r"\ge", r"\neq", r"\equiv", r"\pmod", "<", ">", ",", ":=")):
            return None
        lhs, rhs = inner.split("=", 1)
        lhs_s, rhs_s = lhs.strip(), rhs.strip()
        if not lhs_s or not rhs_s:
            return None
        # Check balanced parentheses
        stack = []
        pairs = {")": "(", "}": "{", "]": "["}
        for ch in lhs_s:
            if ch in "({[":
                stack.append(ch)
            elif ch in ")}]":
                if not stack or stack[-1] != pairs[ch]:
                    return None
                stack.pop()
        if stack:
            return None
        stack = []
        for ch in rhs_s:
            if ch in "({[":
                stack.append(ch)
            elif ch in ")}]":
                if not stack or stack[-1] != pairs[ch]:
                    return None
                stack.pop()
        if stack:
            return None
        return f"{rhs_s} = {lhs_s}"

    spans = list(MATH_DELIM_RE.finditer(text))
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
                return text[:m.start()] + new_span + text[m.end():]
        return text

    flipped = flip_inner(text)
    return flipped if flipped is not None else text


def build_synthetic_tree_branches(problem: str) -> Dict[str, str]:
    """Generate expert_ablate and expert_mutate branches for causal sensitivity."""
    p = (problem or "").strip()
    if not p:
        p = "Find the value."

    branches = {}

    # expert_ablate: Premise boundary condition ablation
    v_ablate = p
    if "positive" in v_ablate:
        v_ablate = v_ablate.replace("positive ", "", 1)
    elif "integer" in v_ablate:
        v_ablate = v_ablate.replace("integer ", "real ", 1)
    elif r"\ge" in v_ablate:
        v_ablate = v_ablate.replace(r"\ge", ">", 1)
    elif "=" in v_ablate:
        v_ablate = v_ablate.replace("=", r"\approx", 1)
    branches["expert_ablate"] = v_ablate

    # expert_mutate: Numerical parameter mutation (+1 increment)
    v_mutate = p
    num_matches = list(_INT_NUMBER_RE.finditer(p))
    if num_matches:
        # Pick the second number if available, else first
        target_m = num_matches[1] if len(num_matches) > 1 else num_matches[0]
        try:
            val = int(target_m.group(1))
            new_val_str = str(val + 1)
            s, e = target_m.span(1)
            v_mutate = p[:s] + new_val_str + p[e:]
        except (ValueError, IndexError):
            pass
    branches["expert_mutate"] = v_mutate

    return branches


def compute_spectral_rank(hidden_states: torch.Tensor) -> float:
    """Compute normalized spectral entropy (effective rank) of hidden states."""
    try:
        h = hidden_states.float()
        if h.ndim != 2 or h.size(0) < 4:
            return 0.58

        if h.size(0) > 256:
            idx = torch.linspace(0, h.size(0) - 1, steps=256, device=h.device).long()
            h = h[idx]

        h = h - h.mean(dim=0, keepdim=True)
        h = F.normalize(h, p=2, dim=-1)
        gram = torch.matmul(h, h.transpose(0, 1))
        eigvals = torch.linalg.eigvalsh(gram).clamp(min=1e-8)
        probs = eigvals / eigvals.sum()
        entropy = -torch.sum(probs * torch.log(probs)).item()
        max_ent = math.log(float(h.size(0)))
        return float(np.clip(entropy / max(max_ent, 1.0), 0.0, 1.0))
    except Exception:
        return 0.58


def compute_linear_cka(H1: torch.Tensor, H2: torch.Tensor) -> float:
    """Compute Linear Centered Kernel Alignment between two hidden state matrices."""
    try:
        X = H1.float()
        Y = H2.float()
        min_len = min(X.size(0), Y.size(0))
        if min_len < 2:
            return 1.0
        X = X[:min_len]
        Y = Y[:min_len]
        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)
        cross_norm_sq = torch.norm(torch.mm(Y_c.t(), X_c), p="fro") ** 2
        norm_x_sq = torch.norm(torch.mm(X_c.t(), X_c), p="fro") ** 2
        norm_y_sq = torch.norm(torch.mm(Y_c.t(), Y_c), p="fro") ** 2
        denom = torch.sqrt(norm_x_sq * norm_y_sq) + 1e-8
        cka = (cross_norm_sq / denom).item()
        return float(max(0.0, min(1.0, cka)))
    except Exception:
        return 0.90


def compute_stable_rank(H: torch.Tensor) -> float:
    """Compute Stable Rank SR = ||H||_F^2 / ||H||_2^2 for activation matrix."""
    try:
        matrix = H.float()
        frobenius_sq = torch.sum(matrix ** 2).item()
        v = torch.randn(matrix.size(1), 1, device=matrix.device)
        v = v / (torch.norm(v) + 1e-8)
        for _ in range(3):
            u = torch.mm(matrix, v)
            u = u / (torch.norm(u) + 1e-8)
            v = torch.mm(matrix.t(), u)
            v = v / (torch.norm(v) + 1e-8)
        sigma_max = torch.norm(torch.mm(matrix, v)).item()
        spectral_sq = max(1e-6, sigma_max ** 2)
        sr = frobenius_sq / spectral_sq
        return float(sr)
    except Exception:
        return 1.0


def extract_structural_features(problem: str, model_id: str, reasoning_effort: str) -> Dict[str, float]:
    """Compute pure text structural features with zero GPU cost."""
    p = problem or ""
    p_lower = p.lower()
    n_chars = len(p)
    words = p.split()
    n_words = len(words)

    # Relations and math density
    relations = RELATION_RE.findall(p)
    n_conditions = len(relations)

    math_spans = list(MATH_DELIM_RE.finditer(p))
    math_chars = sum(len(m.group(0)) for m in math_spans)
    math_density = float(math_chars / max(1, n_chars))

    # Domain flags
    is_geom = 1.0 if any(kw in p_lower for kw in GEOMETRY_KEYWORDS) else 0.0
    is_nt = 1.0 if any(kw in p_lower for kw in NUMBER_THEORY_KEYWORDS) else 0.0
    is_comb = 1.0 if any(kw in p_lower for kw in COMBINATORICS_KEYWORDS) else 0.0
    is_alg = 1.0 if any(kw in p_lower for kw in ALGEBRA_KEYWORDS) else 0.0

    # High-leverage fragility signatures
    has_incenter = 1.0 if "incenter" in p_lower else 0.0
    has_bisector = 1.0 if bool(re.search(r"\bbisect|\bbisector\b", p_lower)) else 0.0
    has_circumcircle = 1.0 if ("circumcircle" in p_lower or "circumradius" in p_lower) else 0.0
    has_divisor_remainder = 1.0 if (("divisors of" in p_lower and "remainder" in p_lower) or ("divisor" in p_lower and "remainder" in p_lower)) else 0.0
    has_infinite_sum = 1.0 if ("infinite sum" in p_lower or (r"\sum_{" in p and r"\infty" in p)) else 0.0
    has_circ_op = 1.0 if (r"\circ" in p or "circ b" in p_lower) else 0.0
    has_grid_board = 1.0 if bool(re.search(r"\bgrid\b|\bboard\b|\bcells\b|\bblackboard\b", p_lower)) else 0.0
    has_alice_bob = 1.0 if ("alice" in p_lower or "bob" in p_lower) else 0.0

    # Cognitive load
    cognitive_load = (n_chars * math_density) / 100.0

    # Model parameter scale
    model_param_log = 0.9031  # default ~8B
    if "120b" in model_id.lower() or "gpt-oss" in model_id.lower():
        model_param_log = 2.08  # ~120B
    elif "397b" in model_id.lower():
        model_param_log = 2.60  # ~397B
    elif "7b" in model_id.lower() or "8b" in model_id.lower():
        model_param_log = 0.85  # ~7-8B
    elif "4b" in model_id.lower():
        model_param_log = 0.60  # ~4B

    # Reasoning effort scale
    effort_scale = {"low": 0.33, "medium": 0.66, "high": 1.00, "default": 0.50}
    reasoning_effort_scale = effort_scale.get(reasoning_effort.lower(), 0.50)

    capacity_to_load_ratio = (model_param_log + 1.0) / (cognitive_load + 1.0)

    return {
        "n_conditions": float(n_conditions),
        "log_length": float(math.log(max(10, n_chars))),
        "char_length": float(n_chars),
        "math_density": math_density,
        "is_geometry": is_geom,
        "is_number_theory": is_nt,
        "is_combinatorics": is_comb,
        "is_algebra": is_alg,
        "has_incenter": has_incenter,
        "has_bisector": has_bisector,
        "has_circumcircle": has_circumcircle,
        "has_divisor_remainder": has_divisor_remainder,
        "has_infinite_sum": has_infinite_sum,
        "has_circ_op": has_circ_op,
        "has_grid_board": has_grid_board,
        "has_alice_bob": has_alice_bob,
        "cognitive_load": cognitive_load,
        "model_param_log": model_param_log,
        "reasoning_effort_scale": reasoning_effort_scale,
        "capacity_to_load_ratio": capacity_to_load_ratio,
    }


# =========================================================================
# MAIN EXTRACTOR CLASS
# =========================================================================
class HybridFeatureExtractor:
    """High-performance feature extractor combining all signal sources."""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_id: str = "",
        device: str = "cuda",
    ):
        self.device = device if (HAS_TORCH_TRANSFORMERS and torch.cuda.is_available() and device == "cuda") else "cpu"
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_id: str, device: str = "cuda") -> "HybridFeatureExtractor":
        """Factory method to load model and tokenizer safely."""
        if not HAS_TORCH_TRANSFORMERS:
            return cls(model=None, tokenizer=None, model_id=model_id, device="cpu")

        actual_device = "cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu"
        logger.info("Loading model %s on %s...", model_id, actual_device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=True,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "᠀"

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if actual_device == "cuda" else torch.float32,
                device_map="auto" if actual_device == "cuda" else None,
                local_files_only=True,
                trust_remote_code=True,
            )
            model.eval()
            return cls(model=model, tokenizer=tokenizer, model_id=model_id, device=actual_device)
        except Exception as e:
            logger.info("Local checkpoint unavailable for %s (%s). Using structural mode.", model_id, e)
            return cls(model=None, tokenizer=None, model_id=model_id, device="cpu")

    @torch.inference_mode()
    def extract_scp_features(self, problem: str) -> Dict[str, float]:
        """Extract SCSP contrastive features: CDI_latent, CDI_digits, spectral_rank, digit_margin."""
        if not HAS_TORCH_TRANSFORMERS or self.model is None or self.tokenizer is None:
            return {
                "cdi_latent": 0.0,
                "cdi_digits": 0.0,
                "spectral_rank": 0.60,
                "digit_margin": 0.18,
                "premise_sensitivity": 0.05,
                "preserve_drift": 0.05,
            }

        # Build 4-prompt contrastive quad
        prompts = self._build_contrastive_quad(problem)
        orig_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"

        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        finally:
            self.tokenizer.padding_side = orig_padding_side

        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states) - 1
        idx_mid = max(1, int(n_layers * 0.55))
        idx_late = max(1, int(n_layers * 0.82))
        idx_final = -1

        mid_h = hidden_states[idx_mid]
        late_h = hidden_states[idx_late]
        final_h = hidden_states[idx_final]

        # 1. Spectral Rank on clean prompt tokens
        clean_mask = inputs["attention_mask"][0].bool()
        spectral_rank = compute_spectral_rank(mid_h[0, clean_mask, :])

        # 2. Whitened Latent Drift (last 8 tokens)
        w_window = min(8, late_h.size(1))
        cone_late = late_h[0, clean_mask, :].float().mean(dim=0, keepdim=True)
        cone_final = final_h[0, clean_mask, :].float().mean(dim=0, keepdim=True)

        def pair_drift(a: int, b: int) -> float:
            va_l = (late_h[a, -w_window:, :].float() - cone_late).mean(dim=0)
            vb_l = (late_h[b, -w_window:, :].float() - cone_late).mean(dim=0)
            d_l = 1.0 - F.cosine_similarity(va_l, vb_l, dim=0).item()
            va_f = (final_h[a, -w_window:, :].float() - cone_final).mean(dim=0)
            vb_f = (final_h[b, -w_window:, :].float() - cone_final).mean(dim=0)
            d_f = 1.0 - F.cosine_similarity(va_f, vb_f, dim=0).item()
            return float(np.clip(0.5 * (d_l + d_f), 0.0, 2.0))

        d_pres_1 = pair_drift(0, 1)
        d_pres_2 = pair_drift(0, 2)
        d_pres = 0.5 * (d_pres_1 + d_pres_2)
        d_dest = pair_drift(0, 3)

        cdi_latent = (d_dest - d_pres) / (d_dest + d_pres + 1e-5)

        # 3. Forced-Answer Digit Distribution JSD & Epistemic Margin
        logits_last = outputs.logits[:, -1, :].float()
        # Get digit token IDs
        digit_token_ids = []
        for d in "0123456789":
            toks = self.tokenizer.encode(d, add_special_tokens=False)
            if toks:
                digit_token_ids.append(toks[-1])
        if not digit_token_ids:
            digit_token_ids = list(range(10))
        digit_token_ids = torch.tensor(digit_token_ids, dtype=torch.long, device=self.device)

        digit_logits = logits_last[:, digit_token_ids]
        digit_probs = F.softmax(digit_logits, dim=-1)

        def js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
            p = p.clamp(min=eps)
            q = q.clamp(min=eps)
            m = 0.5 * (p + q)
            kl_pm = torch.sum(p * torch.log(p / m)).item()
            kl_qm = torch.sum(q * torch.log(q / m)).item()
            return float(np.clip(0.5 * (kl_pm + kl_qm) / math.log(2.0), 0.0, 1.0))

        jsd_pres = 0.5 * (
            js_div(digit_probs[0], digit_probs[1]) +
            js_div(digit_probs[0], digit_probs[2])
        )
        jsd_dest = js_div(digit_probs[0], digit_probs[3])
        cdi_digits = (jsd_dest - jsd_pres) / (jsd_dest + jsd_pres + 1e-5)

        sorted_p, _ = torch.sort(digit_probs[0], descending=True)
        digit_margin = float((sorted_p[0] - sorted_p[1]).item())

        del outputs, hidden_states, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "cdi_latent": float(np.clip(cdi_latent, -1.0, 1.0)),
            "cdi_digits": float(np.clip(cdi_digits, -1.0, 1.0)),
            "spectral_rank": float(spectral_rank),
            "digit_margin": float(digit_margin),
            "premise_sensitivity": float(jsd_dest),
            "preserve_drift": float(d_pres),
        }

    @torch.inference_mode()
    def extract_synthetic_tree_features(self, problem: str) -> Dict[str, float]:
        """Extract expert_ablate and expert_mutate drift features."""
        if not HAS_TORCH_TRANSFORMERS or self.model is None or self.tokenizer is None:
            return {
                "drift_expert_ablate": 0.24,
                "drift_expert_mutate": 0.25,
            }

        branches = build_synthetic_tree_branches(problem)
        prompts = [problem] + [branches["expert_ablate"], branches["expert_mutate"]]

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        mask = enc.get("attention_mask")

        out = self.model(**enc, output_hidden_states=True, use_cache=False)
        all_hidden = out.hidden_states
        num_layers = len(all_hidden)

        # Use mid and late layers
        l_mid = int(num_layers * 0.50)
        l_late = int(num_layers * 0.75)
        h_mid = all_hidden[l_mid]
        h_late = all_hidden[l_late]
        h_combined = 0.50 * h_mid + 0.50 * h_late

        # Pool on math-grounded tokens
        MATH_TOKEN_CHARS = set("0123456789+-*/=^_{}\\$")
        pooled_vectors = []
        for i in range(3):
            input_ids = enc["input_ids"][i]
            seq_len = input_ids.size(0)
            math_token_indices = []
            for t_idx in range(seq_len):
                if mask is not None and mask[i, t_idx] == 0:
                    continue
                try:
                    tok_text = self.tokenizer.decode([input_ids[t_idx]])
                    if any(c in MATH_TOKEN_CHARS for c in tok_text):
                        math_token_indices.append(t_idx)
                except Exception:
                    pass
            if not math_token_indices:
                if mask is not None:
                    nonzeros = mask[i].nonzero().squeeze(-1)
                    math_token_indices = nonzeros.tolist() if nonzeros.numel() > 0 else [seq_len - 1]
                else:
                    math_token_indices = list(range(seq_len))

            seq_vecs = h_combined[i, math_token_indices].float()
            pooled = seq_vecs.mean(dim=0)
            pooled_norm = pooled / (torch.norm(pooled) + 1e-8)
            pooled_vectors.append(pooled_norm)

        v_root = pooled_vectors[0]
        drifts = {}
        for idx, b_name in enumerate(["expert_ablate", "expert_mutate"], start=1):
            v_b = pooled_vectors[idx]
            cos_sim = torch.dot(v_root, v_b).item()
            drift = float(max(0.0, min(1.0, 1.0 - cos_sim)))
            drifts[f"drift_{b_name}"] = drift

        del out, all_hidden, h_mid, h_late, h_combined, enc
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return drifts

    @torch.inference_mode()
    def extract_early_cot_features(self, problem: str) -> Dict[str, float]:
        """Extract early CoT dynamics: entropy, margin, volatility (first 64 tokens)."""
        if not HAS_TORCH_TRANSFORMERS or self.model is None or self.tokenizer is None:
            return {
                "cot_mean_entropy": 1.50,
                "cot_std_entropy": 0.40,
                "cot_entropy_volatility": 0.23,
                "cot_mean_margin": 0.45,
                "cot_min_margin": 0.15,
            }

        prompt = problem.strip() + "\nLet's reason step by step:\n1."
        enc = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        mean_entropy = 1.50
        std_entropy = 0.40
        entropy_volatility = 0.23
        mean_margin = 0.45
        min_margin = 0.15

        try:
            gen_out = self.model.generate(
                **enc,
                max_new_tokens=64,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            scores = gen_out.scores
            if scores and len(scores) > 0:
                step_ents = []
                step_margs = []
                for s in scores:
                    logits = s[0].float()
                    probs = F.softmax(logits, dim=-1)
                    ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                    step_ents.append(ent)
                    top2 = torch.topk(logits, k=2).values
                    step_margs.append((top2[0] - top2[1]).item())

                mean_entropy = float(np.mean(step_ents))
                std_entropy = float(np.std(step_ents))
                entropy_volatility = float(std_entropy / max(1e-5, mean_entropy))
                mean_margin = float(np.mean(step_margs))
                min_margin = float(np.min(step_margs))

            del gen_out
        except Exception as e:
            logger.debug("Early CoT generation skipped: %s", e)

        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "cot_mean_entropy": mean_entropy,
            "cot_std_entropy": std_entropy,
            "cot_entropy_volatility": entropy_volatility,
            "cot_mean_margin": mean_margin,
            "cot_min_margin": min_margin,
        }

    def _build_contrastive_quad(self, problem_text: str) -> List[str]:
        """Generate [clean, preserve1, preserve2, destroy] with forced answer trigger."""
        text = (problem_text or "").strip()
        probe_suffix = "\n\nTherefore, the exact numerical answer is \\boxed{"

        # Preserve 1: Harmless layout/framing shift
        p_pres_1 = "Consider the following mathematical problem and express the result in base 10:\n" + text.replace(". ", ".\n")

        # Preserve 2: Safe isomorphic variable relabeling
        p_pres_2 = _SAFE_VAR_RE.sub(lambda m: _VAR_SWAP_MAP.get(m.group(1), m.group(1)), text)
        if p_pres_2 == text:
            p_pres_2 = text + "\n(Note: All quantities are well-defined.)"

        # Destroy: Permute numerical premises and invert operations
        nums = _INT_NUMBER_RE.findall(text)
        if len(nums) >= 2:
            mutated_nums = [str(int(n) + 3) for n in (nums[1:] + nums[:1])]
            it = iter(mutated_nums)
            p_destroy = _INT_NUMBER_RE.sub(lambda _: next(it), text)
        elif len(nums) == 1:
            p_destroy = _INT_NUMBER_RE.sub(lambda m: str(int(m.group(1)) + 7), text)
        else:
            p_destroy = text.replace("+", "-").replace("=", " != ") + " Assume no solution exists."

        return [
            text + probe_suffix,
            p_pres_1 + probe_suffix,
            p_pres_2 + probe_suffix,
            p_destroy + probe_suffix,
        ]

    def extract_all_features(
        self,
        problem: str,
        model_id: str,
        reasoning_effort: str,
    ) -> Dict[str, float]:
        """Extract complete feature set combining all signal sources."""
        features = {}

        # 1. Structural features (always available, zero GPU)
        features.update(extract_structural_features(problem, model_id, reasoning_effort))

        # 2. SCSP contrastive features
        features.update(self.extract_scp_features(problem))

        # 3. Synthetic tree expert sensitivity
        features.update(self.extract_synthetic_tree_features(problem))

        # 4. Early CoT dynamics
        features.update(self.extract_early_cot_features(problem))

        return features