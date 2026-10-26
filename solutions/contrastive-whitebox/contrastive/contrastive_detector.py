"""Contrastive latent representation and multi-layer prefill invariance detector.

Implements batched prefill extraction across dynamic fractional layers, anisotropy centering,
math-token subspace pooling, candidate-support temperature-smoothed JSD, dual-variant max drift,
and late-layer self-consistency depth verification.

NO-CRASH PROTOCOL:
- Zero pickle loads (probe_artifact.pkl is NOT loaded — avoids version mismatch)
- VRAM Size Gate: Never loads models > 32B (prevents CPU offload + OS SIGKILL 137)
- Fast GPU Proxy Substitution: Reuses active GPU model for oversized checkpoints
- VRAM Headroom Guard: Immediately frees full forward pass output graph
- Model loading uses local_files_only=True, device_map="auto", trust_remote_code=True
- Safe device resolution via _get_device(model) instead of model.device
"""

from __future__ import annotations

import gc
import math
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
_LAST_SAFE_MODEL_ID: str | None = None

MATH_TOKEN_RE = re.compile(r"[0-9+\-*/=^_{}\\]")

OVERSIZED_TAGS = (
    "32b", "70b", "72b", "110b", "120b", "235b", "397b", "405b", "671b", "gpt-oss", "frontier"
)

SAFE_OFFLINE_PROXIES = (
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "Qwen/Qwen3.5-4B",
)


@dataclass
class ContrastiveMetrics:
    latent_drift: float = 0.0          # Multi-layer centered cosine + math-span + top drift
    kl_div: float = 0.0                # Temperature-smoothed (T=1.5) Jensen-Shannon divergence
    baseline_entropy: float = 0.0      # Next-token Shannon entropy on P_0
    margin_disagreement: float = 0.0   # Mean |margin(P_0) - margin(P_j)|
    math_span_drift: float = 0.0       # Cosine shift pooled strictly over math/digit tokens
    rel_l2_drift: float = 0.0          # Relative L2 drift ||h(P_j) - h(P_0)||_2 / ||h(P_0)||_2
    entropy_shift: float = 0.0         # Mean |H(P_0) - H(P_j)|
    late_layer_agreement: float = 1.0  # Cosine similarity between 0.75L and 1.00L on P_0
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


def _get_device(model: Any) -> Any:
    """Safely resolve the device a model's parameters live on.

    With device_map="auto", model.device does not exist on nn.Module.
    We fall back to inspecting the first parameter's device.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    except Exception:
        pass
    return getattr(model, "device", torch.device("cpu") if HAS_TORCH_TRANSFORMERS else "cpu")


def unload_current_model() -> None:
    """Explicitly free GPU VRAM from the currently cached model."""
    global _LOADED_MODEL_ID, _LOADED_TOKENIZER, _LOADED_MODEL
    if _LOADED_MODEL is not None:
        del _LOADED_MODEL
    if _LOADED_TOKENIZER is not None:
        del _LOADED_TOKENIZER
    _LOADED_MODEL = None
    _LOADED_TOKENIZER = None
    _LOADED_MODEL_ID = None
    gc.collect()
    if HAS_TORCH_TRANSFORMERS and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oversized_model(model_id: str) -> bool:
    """Check if model exceeds safe single-GPU VRAM (prevents CPU offload + SIGKILL 137)."""
    m = (model_id or "").lower()
    if any(tag in m for tag in OVERSIZED_TAGS):
        return True
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
        h = getattr(cfg, "hidden_size", 0)
        L = getattr(cfg, "num_hidden_layers", 0)
        if h * L > 200_000:
            return True
    except Exception:
        pass
    return False


def _load_model_direct(model_id: str) -> tuple[Any, Any] | None:
    """Internal direct loader for a specific safe model_id."""
    global _LOADED_MODEL_ID, _LOADED_TOKENIZER, _LOADED_MODEL, _LAST_SAFE_MODEL_ID

    if not HAS_TORCH_TRANSFORMERS:
        return None

    if _LOADED_MODEL_ID == model_id and _LOADED_MODEL is not None:
        return _LOADED_TOKENIZER, _LOADED_MODEL

    if _LOADED_MODEL is not None:
        print(f"[contrastive] Unloading {_LOADED_MODEL_ID} to free VRAM...")
        unload_current_model()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device == "cuda" else torch.float32
        )

        print(f"[contrastive] Loading {model_id} (local_files_only=True, device_map={'auto' if device == 'cuda' else device})...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=True,
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs,
        )
        if device != "cuda":
            model.to(device)
        model.eval()

        _LOADED_MODEL_ID = model_id
        _LOADED_TOKENIZER = tokenizer
        _LOADED_MODEL = model
        _LAST_SAFE_MODEL_ID = model_id
        print(f"[contrastive] Successfully loaded {model_id}")
        return tokenizer, model
    except Exception as exc:
        print(f"[contrastive] Failed to load {model_id}: {exc}")
        return None


def get_or_load_model(model_id: str) -> tuple[Any, Any] | None:
    """Safely load or retrieve cached model and tokenizer offline.

    Intercepts oversized models (>32B) to prevent CPU offload and Linux SIGKILL (OOM).
    Reuses the active GPU safe model or loads a fast offline 4B-8B proxy.
    """
    global _LOADED_MODEL_ID, _LOADED_TOKENIZER, _LOADED_MODEL, _LAST_SAFE_MODEL_ID

    if not HAS_TORCH_TRANSFORMERS:
        return None

    base_id = model_id.split(":")[0].strip() if ":" in model_id else model_id.strip()

    # Oversized Model Gate: NEVER load models > 32B onto GPU/CPU
    if _is_oversized_model(base_id):
        if _LOADED_MODEL is not None:
            print(f"[contrastive] Oversized model {base_id} intercepted! Reusing active GPU proxy {_LOADED_MODEL_ID}.")
            return _LOADED_TOKENIZER, _LOADED_MODEL
        for candidate in SAFE_OFFLINE_PROXIES:
            bundle = _load_model_direct(candidate)
            if bundle is not None:
                print(f"[contrastive] Loaded safe offline proxy {candidate} for oversized target {base_id}.")
                return bundle
        print(f"[contrastive] No offline proxy found for {base_id}. Using symbolic fallback.")
        return None

    return _load_model_direct(base_id)


def extract_prefill_metrics(
    prompts: List[str],
    tokenizer: Any,
    model: Any,
) -> ContrastiveMetrics:
    """Stage 1 — Batched Prefill Extraction across dynamic fractional layers.

    Supports prompts = [P_orig, P_v1, P_v2] (or [P_orig, P_v1]).
    Extracts target layers {0.50L, 0.75L, 1.00L}, calculates late-layer agreement on P_orig,
    and returns maximum drift across variants.
    """
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = "right"
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        device = _get_device(model)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        # Dynamic Fractional Layers: {0.50*L, 0.75*L, L}
        total_layers = getattr(model.config, "num_hidden_layers", len(out.hidden_states) - 1)
        l_50 = max(1, round(0.50 * total_layers))
        l_75 = max(1, round(0.75 * total_layers))
        l_100 = total_layers
        target_layer_indices = sorted({l_50, l_75, l_100})

        mask = enc["attention_mask"]  # [B, S]
        batch_size = mask.size(0)
        last_valid_indices = [int(mask[i].nonzero()[-1].item()) for i in range(batch_size)]

        # Immediately extract target layer hidden states and logits to float32, then free out
        target_hs = {l_idx: out.hidden_states[l_idx].float() for l_idx in target_layer_indices}
        hs_top = target_hs[l_100]
        logits_tensor = torch.stack(
            [out.logits[i, last_valid_indices[i]].float() for i in range(batch_size)], dim=0
        )
        del out  # VRAM headroom guard: free full intermediate graph immediately

        # Late-Layer Internal Agreement on P_0 (Self-Consistency Depth Signal)
        hs_0_75 = target_hs[l_75][0, last_valid_indices[0]]
        hs_0_100 = hs_top[0, last_valid_indices[0]]
        late_layer_sim = F.cosine_similarity(hs_0_75.unsqueeze(0), hs_0_100.unsqueeze(0), dim=-1).item()
        late_layer_agreement = float(max(-1.0, min(1.0, late_layer_sim)))

        # Math-Token Subspace Pooling on P_0 and variants
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

        # Candidate token set for JSD (Union of top-64 from P_0 and all variants)
        top_cand_indices = set(torch.topk(logits_tensor[0], k=min(64, logits_tensor.size(-1))).indices.tolist())
        for k in range(1, batch_size):
            top_cand_indices.update(torch.topk(logits_tensor[k], k=min(64, logits_tensor.size(-1))).indices.tolist())
        candidate_tensor = torch.tensor(sorted(top_cand_indices), device=logits_tensor.device, dtype=torch.long)
        sliced_logits = logits_tensor[:, candidate_tensor]
        p_smooth = F.softmax(sliced_logits / 1.5, dim=-1).clamp(min=1e-8)

        # Unsmoothed probabilities for Shannon Entropy & Probability Margins
        p_unsmoothed = F.softmax(logits_tensor, dim=-1).clamp(min=1e-12)
        h_0 = float(-(p_unsmoothed[0] * p_unsmoothed[0].log()).sum().item())
        baseline_entropy = h_0

        top2 = torch.topk(p_unsmoothed, k=2, dim=-1).values
        margins = top2[:, 0] - top2[:, 1]
        margin_orig = float(margins[0].item())

        # Measure drift across each variant k in range(1, batch_size), then take max
        variant_latent_drifts: list[float] = []
        variant_math_drifts: list[float] = []
        variant_rel_l2_drifts: list[float] = []
        variant_jsds: list[float] = []
        variant_ent_shifts: list[float] = []
        variant_margins: list[float] = []

        h0_top = hs_top[0, last_valid_indices[0]]
        base_norm = torch.norm(h0_top) + 1e-8

        for k in range(1, batch_size):
            # Centered multi-layer cosine drift between P_0 and P_k
            layer_drifts: list[float] = []
            for l_idx in target_layer_indices:
                hs_l = target_hs[l_idx]
                mean_0 = hs_l[0, : last_valid_indices[0] + 1].mean(dim=0)
                mean_k = hs_l[k, : last_valid_indices[k] + 1].mean(dim=0)
                c_0 = hs_l[0, last_valid_indices[0]] - mean_0
                c_k = hs_l[k, last_valid_indices[k]] - mean_k
                sim = F.cosine_similarity(c_0.unsqueeze(0), c_k.unsqueeze(0), dim=-1).item()
                layer_drifts.append(1.0 - sim)
            last_tok_drift = float(np.mean(layer_drifts)) if layer_drifts else 0.0

            # Math span drift between P_0 and P_k
            m_sim = F.cosine_similarity(math_pooled_tensor[0:1], math_pooled_tensor[k:k+1], dim=-1).item()
            m_drift = float(1.0 - m_sim)
            variant_math_drifts.append(m_drift)

            # Rel L2 drift
            diff_norm = torch.norm(hs_top[k, last_valid_indices[k]] - h0_top).item()
            rel_l2 = float(diff_norm / base_norm)
            variant_rel_l2_drifts.append(rel_l2)

            # Latent drift formula: 0.45 * last_token + 0.35 * math_span + 0.20 * rel_l2
            k_latent = float(0.45 * last_tok_drift + 0.35 * m_drift + 0.20 * rel_l2)
            variant_latent_drifts.append(k_latent)

            # JSD
            p0_s = p_smooth[0:1]
            pk_s = p_smooth[k:k+1]
            m_s = 0.5 * (p0_s + pk_s)
            kl_0 = (p0_s * (p0_s.log() - m_s.log())).sum(dim=-1).item()
            kl_k = (pk_s * (pk_s.log() - m_s.log())).sum(dim=-1).item()
            variant_jsds.append(max(0.0, 0.5 * (kl_0 + kl_k)))

            # Entropy shift
            h_k = float(-(p_unsmoothed[k] * p_unsmoothed[k].log()).sum().item())
            variant_ent_shifts.append(abs(h_k - h_0))

            # Margin
            variant_margins.append(float(margins[k].item()))

        # Take MAXIMUM drift across all variants (captures fragile dependency without averaging down)
        latent_drift = max(variant_latent_drifts) if variant_latent_drifts else 0.0
        math_span_drift = max(variant_math_drifts) if variant_math_drifts else 0.0
        rel_l2_drift = max(variant_rel_l2_drifts) if variant_rel_l2_drifts else 0.0
        kl_div = max(variant_jsds) if variant_jsds else 0.0
        entropy_shift = max(variant_ent_shifts) if variant_ent_shifts else 0.0
        margin_variant = float(np.mean(variant_margins)) if variant_margins else 0.0
        margin_disagreement = float(np.mean([abs(m - margin_orig) for m in variant_margins])) if variant_margins else 0.0
        margin_agreement = bool((margin_orig > 0.10) == (margin_variant > 0.10))

        return ContrastiveMetrics(
            latent_drift=latent_drift,
            kl_div=kl_div,
            baseline_entropy=baseline_entropy,
            margin_disagreement=margin_disagreement,
            math_span_drift=math_span_drift,
            rel_l2_drift=rel_l2_drift,
            entropy_shift=entropy_shift,
            late_layer_agreement=late_layer_agreement,
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
        device = _get_device(model)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        step_margins: list[float] = []
        if hasattr(gen, "scores") and gen.scores:
            for step_logits in gen.scores:
                p_step = F.softmax(step_logits[0:1].float(), dim=-1)
                top2_step = torch.topk(p_step, k=2, dim=-1).values
                step_margins.append(float((top2_step[0, 0] - top2_step[0, 1]).item()))

        rollout_min_margin = float(np.percentile(step_margins, 10)) if step_margins else 0.50

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
            late_layer_agreement=prefill_metrics.late_layer_agreement,
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


REASONING_TRIGGER = "\nLet's reason step by step:\n1."


def extract_contrastive_batch(
    model_id: str,
    original_problems: list[str],
    variant_problems: list[str] | None = None,
) -> list[ContrastiveMetrics] | None:
    """Extract contrastive white-box metrics for a batch of problems using 2 distinct variants.

    Loads model_id directly from the Codabench HF cache with persistent caching across
    effort levels (:low, :medium, :high, :default) and safe proxy substitution for oversized models.
    """
    model_bundle = get_or_load_model(model_id)
    if model_bundle is None:
        return None

    tokenizer, model = model_bundle
    try:
        from contrastive.perturbation_engine import generate_variants

        results = []
        for p_orig in original_problems:
            # Generate 2 distinct variants: v1 (algebraic/syntactic), v2 (clause/reframe/flip)
            vars_list = generate_variants(p_orig, n=2)
            v1 = vars_list[0] if len(vars_list) > 0 else f"Solve: {p_orig}"
            v2 = vars_list[1] if len(vars_list) > 1 else f"Carefully determine: {p_orig}"

            p_orig_prompt = p_orig + REASONING_TRIGGER
            p_v1_prompt = v1 + REASONING_TRIGGER
            p_v2_prompt = v2 + REASONING_TRIGGER

            metrics = extract_prefill_metrics(
                [p_orig_prompt, p_v1_prompt, p_v2_prompt], tokenizer, model
            )
            results.append(metrics)
        return results
    except Exception as exc:
        print(f"[contrastive] extract_contrastive_batch failed: {exc}")
        return None
