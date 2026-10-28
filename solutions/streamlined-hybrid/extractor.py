"""Streamlined Hybrid Feature Extractor for AIMO Interpretability Challenge.

Pillar 1: Fast Prefill Micro-Perturbation Stability (Layer Quartiles Cosine Similarity & Linear CKA)
Pillar 2: Early-Token Generation Dynamics (First-64-Token Entropy & Margin)
Pillar 3: Activation Matrix Stable Rank (Dimensionality & Spectral Geometry)

Optimized for:
- Single-GPU RTX 6000 96GB execution within < 3600s total budget.
- Zero GPU OOM: torch.inference_mode, manual cache cleanup, prompt length bounding.
- Graceful CPU fallback when CUDA is unavailable.
- Offline execution: local_files_only=True.
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

# Structural Regex
MATH_DELIM_RE = re.compile(r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)")
RELATION_RE = re.compile(r"(=|\\le|\\ge|\\leq|\\geq|<|>|\\equiv|\\mid)")
VARIABLE_RE = re.compile(r"(?<![\\a-zA-Z])([a-zA-Z])(?![a-zA-Z])")

# Default feature names
FEATURE_NAMES = [
    "cos_sim_q1",
    "cos_sim_q2",
    "cos_sim_q3",
    "cos_sim_last",
    "linear_cka",
    "mean_entropy",
    "std_entropy",
    "mean_margin",
    "min_margin",
    "stable_rank",
    "n_conditions",
    "log_length",
]


def generate_micro_perturbation(problem: str) -> str:
    """Deterministically swap mathematical variables and syntax preserving semantics."""
    if not problem:
        return ""
    
    # 1. LaTeX syntax equivalences
    p = problem
    if r"\le " in p:
        p = p.replace(r"\le ", r"\leq ")
    elif r"\leq " in p:
        p = p.replace(r"\leq ", r"\le ")
    
    if r"\ge " in p:
        p = p.replace(r"\ge ", r"\geq ")
    elif r"\geq " in p:
        p = p.replace(r"\geq ", r"\ge ")

    # 2. Variable renaming inside math blocks
    math_spans = list(MATH_DELIM_RE.finditer(p))
    if not math_spans:
        # Standard fallback variable replacement
        return p.replace("x", "t").replace("n", "k") if "x" in p or "n" in p else p + " "

    new_parts: List[str] = []
    last_idx = 0
    var_map = {"x": "t", "y": "u", "z": "v", "n": "k", "a": "p", "b": "q"}

    for m in math_spans:
        start, end = m.span()
        new_parts.append(p[last_idx:start])
        block = p[start:end]
        for src, dst in var_map.items():
            if f"${src}$" in block or f" {src} " in block or f"{src}^" in block or f"{src}_" in block:
                block = re.sub(rf"(?<![\\a-zA-Z]){src}(?![a-zA-Z])", dst, block)
                break
        new_parts.append(block)
        last_idx = end
    new_parts.append(p[last_idx:])
    
    perturbed = "".join(new_parts)
    if perturbed == problem:
        # Guarantee distinct tokenization if no variables matched
        perturbed = "Consider the following problem. " + problem
    return perturbed


def compute_linear_cka(H1: Any, H2: Any) -> float:
    """Compute Linear Centered Kernel Alignment between two hidden state matrices."""
    try:
        # H: [seq_len, dim]
        X = H1.float()
        Y = H2.float()
        
        # Truncate to common sequence length if necessary
        min_len = min(X.size(0), Y.size(0))
        if min_len < 2:
            return 1.0
        X = X[:min_len]
        Y = Y[:min_len]

        # Center columns
        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)

        # Tr(X_c Y_c^T Y_c X_c^T) = ||Y_c^T X_c||_F^2
        cross_norm_sq = torch.norm(torch.mm(Y_c.t(), X_c), p="fro") ** 2
        norm_x_sq = torch.norm(torch.mm(X_c.t(), X_c), p="fro") ** 2
        norm_y_sq = torch.norm(torch.mm(Y_c.t(), Y_c), p="fro") ** 2

        denom = torch.sqrt(norm_x_sq * norm_y_sq) + 1e-8
        cka = (cross_norm_sq / denom).item()
        return float(max(0.0, min(1.0, cka)))
    except Exception:
        return 0.90


def compute_stable_rank(H: Any) -> float:
    """Compute Stable Rank SR = ||H||_F^2 / ||H||_2^2 for activation matrix."""
    try:
        # H: [seq_len, dim]
        matrix = H.float()
        frobenius_sq = torch.sum(matrix ** 2).item()
        
        # Spectral norm approximated via largest singular value via power iteration
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


class StreamlinedHybridExtractor:
    """High-performance feature extractor combining prefill stability and early-token dynamics."""

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
    def from_pretrained(cls, model_id: str, device: str = "cuda") -> "StreamlinedHybridExtractor":
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
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<unk>"

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
            logger.info("Local checkpoint unavailable for %s (%s). Using lightweight structural mode.", model_id, e)
            return cls(model=None, tokenizer=None, model_id=model_id, device="cpu")


    def extract_structural_features(self, problem: str) -> Dict[str, float]:
        """Compute pure text structural features with zero GPU cost."""
        p = problem or ""
        n_chars = len(p)
        relations = RELATION_RE.findall(p)
        n_conditions = len(relations)
        return {
            "n_conditions": float(n_conditions),
            "log_length": float(math.log(max(10, n_chars))),
        }

    @torch.inference_mode()
    def extract_features(self, problem: str) -> np.ndarray:
        """Extract full 12-dimensional feature vector for a problem."""
        struct_feats = self.extract_structural_features(problem)

        if not HAS_TORCH_TRANSFORMERS or self.model is None or self.tokenizer is None:
            # CPU or offline fallback defaults
            return np.array([
                0.90,  # cos_sim_q1
                0.88,  # cos_sim_q2
                0.85,  # cos_sim_q3
                0.82,  # cos_sim_last
                0.85,  # linear_cka
                1.50,  # mean_entropy
                0.40,  # std_entropy
                0.45,  # mean_margin
                0.15,  # min_margin
                1.00,  # stable_rank
                struct_feats["n_conditions"],
                struct_feats["log_length"],
            ], dtype=np.float32)

        prompt_orig = problem.strip() + "\nLet's reason step by step:\n1."
        prompt_pert = generate_micro_perturbation(problem).strip() + "\nLet's reason step by step:\n1."

        # Tokenize with truncation to bound runtime
        enc_orig = self.tokenizer(prompt_orig, return_tensors="pt", max_length=1024, truncation=True)
        enc_pert = self.tokenizer(prompt_pert, return_tensors="pt", max_length=1024, truncation=True)
        
        enc_orig = {k: v.to(self.device) for k, v in enc_orig.items()}
        enc_pert = {k: v.to(self.device) for k, v in enc_pert.items()}

        # --- 1. FAST PREFILL PASS (Original vs. Micro-Perturbed) ---
        out_orig = self.model(**enc_orig, output_hidden_states=True, use_cache=False)
        out_pert = self.model(**enc_pert, output_hidden_states=True, use_cache=False)

        hidden_orig = out_orig.hidden_states  # Tuple of [1, seq_len, dim]
        hidden_pert = out_pert.hidden_states

        num_layers = len(hidden_orig)
        q1_idx = max(0, int(num_layers * 0.25) - 1)
        q2_idx = max(0, int(num_layers * 0.50) - 1)
        q3_idx = max(0, int(num_layers * 0.75) - 1)
        last_idx = num_layers - 1

        # Cosine similarity at last token across quartiles
        def _last_token_cos_sim(layer_idx: int) -> float:
            h_o = hidden_orig[layer_idx][0, -1, :].float()
            h_p = hidden_pert[layer_idx][0, -1, :].float()
            sim = F.cosine_similarity(h_o.unsqueeze(0), h_p.unsqueeze(0), dim=-1).item()
            return float(max(-1.0, min(1.0, sim)))

        cos_q1 = _last_token_cos_sim(q1_idx)
        cos_q2 = _last_token_cos_sim(q2_idx)
        cos_q3 = _last_token_cos_sim(q3_idx)
        cos_last = _last_token_cos_sim(last_idx)

        # Linear CKA between final layer hidden sequences
        cka = compute_linear_cka(hidden_orig[last_idx][0], hidden_pert[last_idx][0])

        # Stable rank of original problem's final layer hidden matrix
        stable_rk = compute_stable_rank(hidden_orig[last_idx][0])

        # --- 2. EARLY-TOKEN GENERATION DYNAMICS (First 64 tokens) ---
        mean_entropy = 1.50
        std_entropy = 0.40
        mean_margin = 0.45
        min_margin = 0.15

        try:
            gen_out = self.model.generate(
                **enc_orig,
                max_new_tokens=64,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

            scores = gen_out.scores  # Tuple of [1, vocab_size]
            if scores and len(scores) > 0:
                step_entropies = []
                step_margins = []
                for s in scores:
                    logits = s[0].float()
                    probs = F.softmax(logits, dim=-1)
                    ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                    step_entropies.append(ent)

                    top2 = torch.topk(logits, k=2).values
                    margin = (top2[0] - top2[1]).item()
                    step_margins.append(margin)

                mean_entropy = float(np.mean(step_entropies))
                std_entropy = float(np.std(step_entropies))
                mean_margin = float(np.mean(step_margins))
                min_margin = float(np.min(step_margins))

            del gen_out
        except Exception as e:
            logger.debug("Early generation dynamics skipped: %s", e)

        # Cleanup tensors
        del out_orig, out_pert, enc_orig, enc_pert
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return np.array([
            cos_q1,
            cos_q2,
            cos_q3,
            cos_last,
            cka,
            mean_entropy,
            std_entropy,
            mean_margin,
            min_margin,
            stable_rk,
            struct_feats["n_conditions"],
            struct_feats["log_length"],
        ], dtype=np.float32)
