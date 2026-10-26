"""Contrastive latent representation and multi-layer prefill invariance detector.

Implements batched prefill extraction across dynamic fractional layers, anisotropy centering,
math-token subspace pooling, temperature-smoothed bounded JSD, and conditional 32-token micro-rollout.
"""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False

RECOMMENDED_STRATEGY = "best_layer_mean_margin"

# Module-level model cache to avoid re-loading weights for each batch problem
_LOADED_MODEL_ID: str | None = None
_LOADED_TOKENIZER: Any = None
_LOADED_MODEL: Any = None

MATH_TOKEN_RE = re.compile(r"[0-9+\-*/=^_{}\\]")


@dataclass
class ContrastiveMetrics:
    latent_drift: float = 0.0          # Anisotropy-centered multi-layer cosine distance
    kl_div: float = 0.0                # Temperature-smoothed (T=2.0) Jensen-Shannon divergence
    baseline_entropy: float = 0.0      # Next-token Shannon entropy on P_0
    margin_disagreement: float = 0.0   # Mean |margin(P_0) - margin(P_j)|
    math_span_drift: float = 0.0       # Cosine shift pooled strictly over math/digit tokens
    rel_l2_drift: float = 0.0          # Relative L2 drift ||h(P_j) - h(P_0)||_2 / ||h(P_0)||_2
    entropy_shift: float = 0.0         # Mean |H(P_0) - H(P_j)|
    rollout_divergence: Optional[float] = None
    rollout_convergence_ratio: Optional[float] = None
    rollout_min_margin: Optional[float] = None
    used_micro_rollout: bool = False
    # Backward compatibility attributes:
    cosine_sim: float = 1.0
    rel_l2_dist: float = 0.0
    margin_orig: float = 0.0
    margin_variant: float = 0.0
    margin_agreement: bool = True


@dataclass(frozen=True)
class ProbeArtifact:
    model_id: str
    system_prompt: str
    data: dict[str, Any]
    kind: str


def _safe_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def _artifact_candidates(model_id: str) -> list[Path]:
    root_dir = Path(__file__).resolve().parents[1]
    safe_id = _safe_model_id(model_id)
    candidates = []
    if safe_id:
        candidates.extend([
            root_dir / "probe_artifacts" / f"{safe_id}.pkl",
            root_dir / "probe_artifacts" / f"{safe_id}.npz",
        ])
    candidates.extend([
        root_dir / "probe_artifact.pkl",
        root_dir / "probe_artifacts" / "probe_artifact.pkl",
        root_dir / "probe_artifact.npz",
    ])
    return candidates


def load_probe_artifact(model_id: str) -> ProbeArtifact | None:
    path = next((c for c in _artifact_candidates(model_id) if c.is_file()), None)
    if path is None:
        return None
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict) or data.get("schema_version") != 2:
            return None
        return ProbeArtifact(
            model_id=str(data.get("model_id") or ""),
            system_prompt=str(data.get("system_prompt") or ""),
            data=data,
            kind="pickle",
        )
    except Exception:
        return None


def _layer_index(artifact: ProbeArtifact) -> int:
    strategy = artifact.data.get("recommended_strategy") or {}
    return int(strategy.get("layer_index", artifact.data["best_layer_index"]))


def get_or_load_model(model_id: str) -> tuple[Any, Any] | None:
    """Safely load or retrieve cached model and tokenizer offline."""
    global _LOADED_MODEL_ID, _LOADED_TOKENIZER, _LOADED_MODEL

    if not HAS_TORCH_TRANSFORMERS:
        return None

    if _LOADED_MODEL_ID == model_id and _LOADED_MODEL is not None:
        return _LOADED_TOKENIZER, _LOADED_MODEL

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device == "cuda" else torch.float32
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=True,
        )
        model.to(device)
        model.eval()

        _LOADED_MODEL_ID = model_id
        _LOADED_TOKENIZER = tokenizer
        _LOADED_MODEL = model
        return tokenizer, model
    except Exception:
        return None


def extract_prefill_metrics(
    prompts: List[str],
    tokenizer: Any,
    model: Any,
) -> ContrastiveMetrics:
    """Stage 1 — Batched Prefill Extraction across dynamic fractional layers.

    Tokenizes [P_0, P_1, ..., P_k] in a single batch, performs anisotropy centering,
    math-token subspace pooling, and bounded temperature-smoothed JSD.
    """
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = "right"
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        # Dynamic Fractional Layers: {0.50*L, 0.75*L, L}
        total_layers = getattr(model.config, "num_hidden_layers", len(out.hidden_states) - 1)
        layer_indices = sorted({
            max(1, round(0.50 * total_layers)),
            max(1, round(0.75 * total_layers)),
            total_layers,
        })

        mask = enc["attention_mask"]  # [B, S]
        batch_size = mask.size(0)
        last_valid_indices = [int(mask[i].nonzero()[-1].item()) for i in range(batch_size)]

        # Anisotropy Centering & Multi-layer Latent Drift
        layer_drifts: list[float] = []
        for l_idx in layer_indices:
            hs = out.hidden_states[l_idx].float()  # [B, S, D]
            h_centered_list = []
            for i in range(batch_size):
                last_idx = last_valid_indices[i]
                seq_mean = hs[i, : last_idx + 1].mean(dim=0)
                h_last = hs[i, last_idx]
                h_centered_list.append(h_last - seq_mean)
            h_centered = torch.stack(h_centered_list, dim=0)  # [B, D]
            sims = F.cosine_similarity(h_centered[0:1], h_centered[1:], dim=-1)
            dist = (1.0 - sims).mean().item()
            layer_drifts.append(dist)

        latent_drift = float(np.mean(layer_drifts)) if layer_drifts else 0.0

        # Relative L2 drift on top layer
        hs_top = out.hidden_states[-1].float()
        h0_top = hs_top[0, last_valid_indices[0]]
        h_others = torch.stack([hs_top[i, last_valid_indices[i]] for i in range(1, batch_size)], dim=0)
        diff_norm = torch.norm(h_others - h0_top.unsqueeze(0), dim=-1)
        base_norm = torch.norm(h0_top) + 1e-8
        rel_l2_drift = float((diff_norm / base_norm).mean().item())

        # Math-Token Subspace Pooling
        input_ids = enc["input_ids"]
        math_pooled_list = []
        for i in range(batch_size):
            last_idx = last_valid_indices[i]
            seq_tokens = input_ids[i, : last_idx + 1].tolist()
            math_indices = [
                idx for idx, tok_id in enumerate(seq_tokens)
                if MATH_TOKEN_RE.search(tokenizer.decode([tok_id]))
            ]
            if math_indices:
                math_pooled = hs_top[i, math_indices].mean(dim=0)
            else:
                math_pooled = hs_top[i, last_idx]
            math_pooled_list.append(math_pooled)

        math_pooled_tensor = torch.stack(math_pooled_list, dim=0)
        math_sims = F.cosine_similarity(math_pooled_tensor[0:1], math_pooled_tensor[1:], dim=-1)
        math_span_drift = float((1.0 - math_sims).mean().item())

        # Final-position Logits & Temperature-Smoothed Bounded JSD
        logits_list = [out.logits[i, last_valid_indices[i]].float() for i in range(batch_size)]
        logits_tensor = torch.stack(logits_list, dim=0)  # [B, V]

        # Temperature-smoothed (T=2.0)
        p_smooth = F.softmax(logits_tensor / 2.0, dim=-1).clamp(min=1e-8)
        p_0_s = p_smooth[0:1]
        p_k_s = p_smooth[1:]
        m_s = 0.5 * (p_0_s + p_k_s)

        kl_p0 = (p_0_s * (p_0_s.log() - m_s.log())).sum(dim=-1)
        kl_pk = (p_k_s * (p_k_s.log() - m_s.log())).sum(dim=-1)
        jsd = 0.5 * (kl_p0 + kl_pk).mean().item()
        kl_div = max(0.0, float(jsd))

        # Unsmoothed probabilities for Shannon Entropy & Probability Margins
        p_unsmoothed = F.softmax(logits_tensor, dim=-1).clamp(min=1e-12)
        h_0 = -(p_unsmoothed[0] * p_unsmoothed[0].log()).sum().item()
        baseline_entropy = float(h_0)

        h_k = -(p_unsmoothed[1:] * p_unsmoothed[1:].log()).sum(dim=-1)
        entropy_shift = float((h_k - h_0).abs().mean().item())

        top2 = torch.topk(p_unsmoothed, k=2, dim=-1).values
        margins = top2[:, 0] - top2[:, 1]
        margin_disagreement = float((margins[1:] - margins[0]).abs().mean().item())

        margin_orig = float(margins[0].item())
        margin_variant = float(margins[1:].mean().item())
        margin_agreement = bool((margin_orig > 0.10) == (margin_variant > 0.10))

        return ContrastiveMetrics(
            latent_drift=latent_drift,
            kl_div=kl_div,
            baseline_entropy=baseline_entropy,
            margin_disagreement=margin_disagreement,
            math_span_drift=math_span_drift,
            rel_l2_drift=rel_l2_drift,
            entropy_shift=entropy_shift,
            cosine_sim=max(-1.0, min(1.0, 1.0 - latent_drift)),
            rel_l2_dist=rel_l2_drift,
            margin_orig=margin_orig,
            margin_variant=margin_variant,
            margin_agreement=margin_agreement,
        )
    finally:
        tokenizer.padding_side = orig_padding_side


def run_micro_rollout_verification(
    prompts: List[str],
    prefill_metrics: ContrastiveMetrics,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 32,
) -> ContrastiveMetrics:
    """Stage 2 — Conditional 32-Token Micro-Rollout for borderline cases.

    Generates max_new_tokens for the top pair [P_0, P_1], measures minimum probability
    margin over all 32 steps, and assesses hidden-state trajectory convergence.
    """
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        pair = prompts[:2]
        enc = tokenizer(pair, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # 10th-percentile probability margin across all generated steps for P_0
        step_margins: list[float] = []
        if hasattr(gen, "scores") and gen.scores:
            for step_logits in gen.scores:
                p_step = F.softmax(step_logits[0:1].float(), dim=-1)
                top2_step = torch.topk(p_step, k=2, dim=-1).values
                step_margins.append(float((top2_step[0, 0] - top2_step[0, 1]).item()))

        rollout_min_margin = float(np.percentile(step_margins, 10)) if step_margins else 0.50

        # Step-32 hidden state drift between P_0 and P_1
        if hasattr(gen, "hidden_states") and gen.hidden_states:
            last_step_layers = gen.hidden_states[-1]
            top_hs = last_step_layers[-1].float()  # [B, 1, D]
            h0_step32 = top_hs[0:1, -1, :]
            h1_step32 = top_hs[1:2, -1, :]
            step32_sim = F.cosine_similarity(h0_step32, h1_step32, dim=-1).item()
            step32_drift = max(0.0, 1.0 - step32_sim)
        else:
            step32_drift = prefill_metrics.latent_drift

        rollout_convergence_ratio = step32_drift / (prefill_metrics.latent_drift + 1e-4)

        return ContrastiveMetrics(
            latent_drift=prefill_metrics.latent_drift,
            kl_div=prefill_metrics.kl_div,
            baseline_entropy=prefill_metrics.baseline_entropy,
            margin_disagreement=prefill_metrics.margin_disagreement,
            math_span_drift=prefill_metrics.math_span_drift,
            rel_l2_drift=prefill_metrics.rel_l2_drift,
            entropy_shift=prefill_metrics.entropy_shift,
            rollout_divergence=step32_drift,
            rollout_convergence_ratio=rollout_convergence_ratio,
            rollout_min_margin=rollout_min_margin,
            used_micro_rollout=True,
            cosine_sim=prefill_metrics.cosine_sim,
            rel_l2_dist=prefill_metrics.rel_l2_dist,
            margin_orig=prefill_metrics.margin_orig,
            margin_variant=prefill_metrics.margin_variant,
            margin_agreement=prefill_metrics.margin_agreement,
        )
    finally:
        tokenizer.padding_side = orig_padding_side


def extract_contrastive_batch(
    model_id: str,
    original_problems: list[str],
    variant_problems: list[str],
) -> list[ContrastiveMetrics] | None:
    """Extract contrastive white-box metrics for a batch of problems."""
    artifact = load_probe_artifact(model_id)
    if artifact is None:
        return None

    target_model_id = artifact.model_id or model_id
    model_bundle = get_or_load_model(target_model_id)
    if model_bundle is None:
        return None

    tokenizer, model = model_bundle
    try:
        results = []
        for p_orig, p_var in zip(original_problems, variant_problems):
            metrics = extract_prefill_metrics([p_orig, p_var], tokenizer, model)
            results.append(metrics)
        return results
    except Exception:
        return None
