#!/usr/bin/env python3
"""
Perceptors & Feature Extractors for AIMO Interpretability Challenge.
Fuses:
  - Tier 1: Prefill Contrastive Drift & Forced-Answer Logit Probes (k in {0%, 35%, 70%, 100%})
  - Tier 2: Subspace Geometry (Stable Rank ||H||_F^2 / ||H||_2^2, Spectral Rank, Layer-Wise Velocity, Attention Grounding)
  - Tier 3: Conditional Micro-Rollout Verification for Boundary Instances
  - Nearest Neighbor (NN) & Kernelized Retrieval Routing (Faiss / PyTorch Cosine)
"""

from __future__ import annotations
import gc
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from triton_kernels import FlashLinearAttention, compute_linear_attention_drift

# Regex for safe standalone single-letter math variables outside LaTeX macros
_SAFE_VAR_RE = re.compile(r"(?<![\\a-zA-Z])([xyznmabABCXYZPQR])(?![a-zA-Z])")
_VAR_SWAP_MAP = {
    "x": "u", "y": "v", "z": "w", "n": "k", "m": "p", "a": "r", "b": "s",
    "A": "P", "B": "Q", "C": "R", "X": "U", "Y": "V", "Z": "W", "P": "A", "Q": "B", "R": "C",
}
_INT_NUMBER_RE = re.compile(r"(?<!\\)\b(\d+)\b")
_EQUATION_REL_RE = re.compile(r"(=|\\le|\\ge|\\leq|\\geq|\\neq|<|>|\\equiv)")


# ==============================================================================
# 1. FAST SYMBOLIC / LEXICAL PERTURBATION GENERATOR (DELTA-AWARE)
# ==============================================================================
class SymbolicPerturber:
    """
    Generates deterministic counterfactuals X_tilde conditioned on Delta
    covering all 7 official distribution shift categories:
    Rephrase, Rename, Domain, Distract, Typos, Expert, Expert No-Solution.
    """

    @classmethod
    def perturb_variable_names(cls, text: str) -> str:
        """Isomorphic variable renaming (Rename shift)."""
        return _SAFE_VAR_RE.sub(lambda m: _VAR_SWAP_MAP.get(m.group(1), m.group(1)), text)

    @classmethod
    def perturb_clause_framing(cls, text: str, delta_desc: str = "") -> str:
        """Stylistic and domain rephrasing (Rephrase & Domain shift)."""
        prefix = (
            "Consider the following mathematical problem under standard domain "
            "assumptions. Verify all boundary conditions before concluding:\n"
        )
        if delta_desc:
            prefix = f"[Shift Specification: {delta_desc[:120]}]\n" + prefix
        return prefix + text.replace(". ", ".\n")

    @classmethod
    def perturb_numerical_premises(cls, text: str) -> str:
        """Cyclic numerical and relation mutation (Expert No-Solution shift)."""
        nums = _INT_NUMBER_RE.findall(text)
        if len(nums) >= 2:
            mutated = [str(int(n) + 3) for n in (nums[1:] + nums[:1])]
            it = iter(mutated)
            return _INT_NUMBER_RE.sub(lambda _: next(it), text)
        elif len(nums) == 1:
            return _INT_NUMBER_RE.sub(lambda m: str(int(m.group(1)) + 7), text)
        return text.replace("+", "-").replace("=", " != ") + " Assume no real solution exists."

    @classmethod
    def generate_contrastive_quad(
        cls, problem_text: str, delta_desc: str = ""
    ) -> Tuple[List[str], List[str]]:
        """
        Builds a contrastive quad [X_clean, X_pres_1, X_pres_2, X_dest] with
        both raw and forced-answer probe forms.
        """
        text = (problem_text or "").strip()
        p_pres_1 = cls.perturb_clause_framing(text, delta_desc)
        p_pres_2 = cls.perturb_variable_names(text)
        if p_pres_2 == text:
            p_pres_2 = text + "\n(Note: All mathematical variables represent positive integers.)"
        p_dest = cls.perturb_numerical_premises(text)

        raw_prompts = [text, p_pres_1, p_pres_2, p_dest]
        probe_suffix = "\n\nTherefore, the exact numerical answer is \\boxed{"
        probed_prompts = [p + probe_suffix for p in raw_prompts]
        return raw_prompts, probed_prompts


# ==============================================================================
# 2. SUBSPACE GEOMETRY & INTERNAL DYNAMICS
# ==============================================================================
class SubspaceGeometrySensor:
    """
    Computes subspace stable rank, spectral effective rank, and layer velocity
    to evaluate whether mathematical deduction traverses orthogonal subspaces.
    """

    @staticmethod
    def compute_stable_rank(H: torch.Tensor) -> float:
        """
        Computes subspace stable rank: ||H||_F^2 / ||H||_2^2 = sum(sigma_i^2) / sigma_max^2.
        Higher stable rank = orthogonal multi-step deduction.
        Lower stable rank = low-rank repetitive shortcut collapse.
        """
        if H.ndim != 2 or H.size(0) < 2:
            return 1.0

        H_fp32 = H.detach().to(torch.float32)
        H_centered = H_fp32 - H_fp32.mean(dim=0, keepdim=True)
        frob_sq = torch.sum(H_centered ** 2).item()
        if frob_sq < 1e-12:
            return 1.0

        # Fast power iteration (5 iterations) for largest singular value squared
        d = H_centered.shape[1]
        v = torch.randn(d, 1, device=H_centered.device, dtype=torch.float32)
        v = v / (torch.norm(v) + 1e-12)
        for _ in range(5):
            u = H_centered @ v
            u = u / (torch.norm(u) + 1e-12)
            v = H_centered.T @ u
            v = v / (torch.norm(v) + 1e-12)
        sigma_max_sq = torch.sum((H_centered @ v) ** 2).item()
        srank = float(frob_sq / max(sigma_max_sq, 1e-12))
        return float(np.clip(srank / math.sqrt(float(H.size(0))), 0.0, 1.0))

    @staticmethod
    def compute_spectral_effective_rank(H: torch.Tensor) -> float:
        """
        Computes normalized von Neumann spectral entropy of token Gram matrix in [0, 1].
        """
        if H.ndim != 2 or H.size(0) < 4:
            return 0.58

        h = H.detach().to(torch.float32)
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

    @staticmethod
    def compute_cross_layer_velocity(h_mid: torch.Tensor, h_late: torch.Tensor) -> float:
        """Computes representation rotation distance across transformer depth."""
        v_mid = h_mid.detach().to(torch.float32).mean(dim=0)
        v_late = h_late.detach().to(torch.float32).mean(dim=0)
        cos_sim = F.cosine_similarity(v_mid.unsqueeze(0), v_late.unsqueeze(0), dim=-1).item()
        return float(np.clip(1.0 - cos_sim, 0.0, 2.0))


# ==============================================================================
# 3. NEAREST NEIGHBOR (NN) & KERNELIZED RETRIEVAL ROUTING
# ==============================================================================
class KernelizedRetrievalRouter:
    """
    Nearest-Neighbor embedding lookup (Faiss IndexFlatIP with PyTorch fallback).
    Computes invariant distance features between perturbed representations Delta X
    and reference base representations X.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.reference_embeddings: List[torch.Tensor] = []
        self.reference_ids: List[str] = []
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

    def add_reference(self, embedding: torch.Tensor, ref_id: str) -> None:
        emb_norm = F.normalize(embedding.detach().to(torch.float32).cpu(), p=2, dim=-1)
        self.reference_embeddings.append(emb_norm)
        self.reference_ids.append(ref_id)
        if HAS_FAISS and self.index is not None:
            self.index.add(emb_norm.numpy().astype(np.float32).reshape(1, -1))

    def query_similarity(self, query_emb: torch.Tensor, top_k: int = 1) -> float:
        """Returns maximum cosine similarity against reference embeddings."""
        if not self.reference_embeddings:
            return 1.0

        q_norm = F.normalize(query_emb.detach().to(torch.float32).cpu(), p=2, dim=-1)

        if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
            D, _ = self.index.search(q_norm.numpy().astype(np.float32).reshape(1, -1), min(top_k, self.index.ntotal))
            return float(D[0, 0])

        # Vectorized PyTorch fallback
        ref_matrix = torch.stack(self.reference_embeddings, dim=0)
        sims = torch.mv(ref_matrix, q_norm.squeeze())
        return float(torch.max(sims).item())

    def query_perturbation_severity(self, query: Union[str, torch.Tensor], top_k: int = 1) -> float:
        """Score perturbation severity against canonical reference manifolds in [0, 1]."""
        if isinstance(query, str):
            # Fast deterministic hash-embedding of problem string
            chars = [ord(c) for c in query[:256]]
            vec = torch.zeros(self.dim, dtype=torch.float32)
            for idx, ch in enumerate(chars):
                pos = (ch * 31 + idx * 17) % self.dim
                vec[pos] += 1.0
            query_emb = vec
        else:
            query_emb = query

        if not self.reference_embeddings:
            # Deterministic lexical complexity proxy when no references are pre-populated
            if isinstance(query, str):
                return float(np.clip(len(query.split()) / 250.0, 0.0, 1.0))
            return 0.0

        sim = self.query_similarity(query_emb, top_k=top_k)
        return float(np.clip(1.0 - sim, 0.0, 1.0))


# ==============================================================================
# 4. WHITE-BOX MULTI-TIER FEATURE EXTRACTOR
# ==============================================================================
class WhiteBoxReasoningProber:
    """
    Unified 3-Tier Feature Extractor:
      - Tier 1: Truncation Curve Forced-Answer Probes + Contrastive Quad CDI
      - Tier 2: Subspace Stable Rank, Spectral Rank, Layer Velocity, Attention Mass
      - Tier 3: Conditional Micro-Rollout Verification for Boundary Instances
    """

    def __init__(self, model_name_or_path: str, model_instance: Any = None, tokenizer_instance: Any = None):
        self.model_name = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if tokenizer_instance is not None:
            self.tokenizer = tokenizer_instance
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=True,
                trust_remote_code=True,
            )

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or getattr(self.tokenizer, "bos_token", "<pad>")

        if model_instance is not None:
            self.model = model_instance
        else:
            from transformers import AutoModelForCausalLM
            dtype = "auto" if "gpt-oss-120b" in model_name_or_path.lower() else (
                torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported())
                else (torch.float16 if self.device == "cuda" else torch.float32)
            )
            load_kwargs: Dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": True,
                "torch_dtype": dtype,
            }
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **load_kwargs,
            )
            if self.device != "cuda":
                self.model.to(self.device)
            self.model.eval()

        self.num_layers = getattr(self.model.config, "num_hidden_layers", 32)
        self.probe_layer_mid = max(1, int(self.num_layers * 0.55))
        self.probe_layer_late = max(1, int(self.num_layers * 0.82))
        self.digit_token_ids = self._resolve_digit_tokens()
        self.router = KernelizedRetrievalRouter()

    def _resolve_digit_tokens(self) -> torch.Tensor:
        ids: List[int] = []
        for d in "0123456789":
            toks = self.tokenizer.encode(d, add_special_tokens=False)
            if toks:
                ids.append(toks[-1])
        if not ids:
            ids = list(range(10))
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * torch.log(p / m)).item()
        kl_qm = torch.sum(q * torch.log(q / m)).item()
        return float(np.clip(0.5 * (kl_pm + kl_qm) / math.log(2.0), 0.0, 1.0))

    @torch.inference_mode()
    def extract_instance_features(
        self,
        problem: str,
        cot: str = "",
        target_answer: str = "0",
        delta_desc: str = "",
        reasoning_effort: str = "default",
    ) -> Dict[str, float]:
        """
        Executes single-pass whitebox contrastive probing and returns unified feature dict.
        """
        # 1. Structural surface analysis
        char_len = len(problem)
        num_sentences = max(1, len([s for s in re.split(r"[.?!]+", problem) if s.strip()]))
        num_relations = len(_EQUATION_REL_RE.findall(problem))
        num_numbers = len(_INT_NUMBER_RE.findall(problem))

        len_norm = math.tanh((char_len - 320.0) / 300.0)
        rel_norm = math.tanh((num_relations - 3.5) / 3.0)
        sent_norm = math.tanh((num_sentences - 3.0) / 2.5)
        structural_complexity = float(np.clip(0.45 * len_norm + 0.35 * rel_norm + 0.20 * sent_norm, -1.0, 1.0))

        feats: Dict[str, float] = {
            "structural_complexity": structural_complexity,
            "char_len": float(char_len),
            "num_relations": float(num_relations),
            "num_numbers": float(num_numbers),
        }

        # 2. Contrastive Quad Prefill Pass (Tier 1 + Tier 2)
        _, probed_prompts = SymbolicPerturber.generate_contrastive_quad(problem, delta_desc)
        orig_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"

        try:
            inputs = self.tokenizer(
                probed_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        finally:
            self.tokenizer.padding_side = orig_padding_side

        hidden_states = outputs.hidden_states
        mid_h = hidden_states[self.probe_layer_mid]
        late_h = hidden_states[self.probe_layer_late]
        final_h = hidden_states[-1]

        # Tier 2: Subspace Geometry (Stable Rank & SVD Spectral Rank)
        clean_mask = inputs["attention_mask"][0].bool()
        stable_rank = SubspaceGeometrySensor.compute_stable_rank(mid_h[0, clean_mask, :])
        spectral_rank = SubspaceGeometrySensor.compute_spectral_effective_rank(mid_h[0, clean_mask, :])
        layer_vel = SubspaceGeometrySensor.compute_cross_layer_velocity(
            mid_h[0, clean_mask, :], late_h[0, clean_mask, :]
        )

        # Tier 1: Latent Drift across final 8 tokens
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

        # Tier 1: Forced-Answer Digit Distribution JSD & Confidence Margin
        logits_last = outputs.logits[:, -1, :].float()
        digit_logits = logits_last[:, self.digit_token_ids]
        digit_probs = F.softmax(digit_logits, dim=-1)

        jsd_pres = 0.5 * (
            self._js_divergence(digit_probs[0], digit_probs[1])
            + self._js_divergence(digit_probs[0], digit_probs[2])
        )
        jsd_dest = self._js_divergence(digit_probs[0], digit_probs[3])
        cdi_digits = (jsd_dest - jsd_pres) / (jsd_dest + jsd_pres + 1e-5)

        sorted_p, _ = torch.sort(digit_probs[0], descending=True)
        digit_margin = float((sorted_p[0] - sorted_p[1]).item())
        target_token_prob = float(sorted_p[0].item())

        # Nearest-Neighbor retrieval routing feature
        retrieval_sim = self.router.query_similarity(final_h[0, -1, :])

        del outputs, hidden_states, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        feats.update({
            "cdi_latent": float(np.clip(cdi_latent, -1.0, 1.0)),
            "cdi_digits": float(np.clip(cdi_digits, -1.0, 1.0)),
            "premise_sensitivity": float(jsd_dest),
            "preserve_drift": float(d_pres),
            "spectral_rank": float(spectral_rank),
            "stable_rank": float(stable_rank),
            "layer_velocity": float(layer_vel),
            "digit_margin": float(digit_margin),
            "target_token_prob": target_token_prob,
            "retrieval_similarity": retrieval_sim,
        })
        return feats
