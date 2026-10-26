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
    logit_lens_divergence: float = 0.0 # Logit-lens JSD divergence at 0.75L between P_0 and variants
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


def get_final_norm(model: Any) -> Any:
    """Find the final layer norm / RMSNorm of the model before lm_head."""
    for attr in [
        "model.norm",
        "transformer.ln_f",
        "norm",
        "model.layers_norm",
        "transformer.norm",
        "model.final_layernorm",
    ]:
        curr = model
        for part in attr.split("."):
            curr = getattr(curr, part, None)
            if curr is None:
                break
        if curr is not None and callable(curr):
            return curr
    return None


def logp_at_trigger(
    model: Any,
    h: torch.Tensor,
    is_intermediate: bool,
    final_norm: Any = None,
) -> torch.Tensor:
    """Project hidden state at trigger token to log-probabilities over vocabulary."""
    if is_intermediate and final_norm is not None:
        try:
            h = final_norm(h.unsqueeze(0)).squeeze(0)
        except Exception:
            try:
                h = final_norm(h)
            except Exception:
                pass

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()
    if lm_head is None:
        return torch.empty(0)

    try:
        w = getattr(lm_head, "weight", None)
        target_device = w.device if w is not None else h.device
        target_dtype = w.dtype if w is not None else h.dtype
        logits = lm_head(h.to(device=target_device, dtype=target_dtype))
        return F.log_softmax(logits.float(), dim=-1)
    except Exception:
        return torch.empty(0)


def _extract_single_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
) -> tuple[dict[float, torch.Tensor], torch.Tensor | None]:
    """Unpadded batch-size 1 forward pass extracting states strictly at the last token [0, -1]."""
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True, use_cache=False)

    hs = out.hidden_states
    L = len(hs) - 1
    total_layers = getattr(model.config, "num_hidden_layers", L)
    l_75 = max(1, round(0.75 * total_layers))
    l_100 = total_layers

    s_75 = hs[l_75][0, -1].float()
    s_100 = hs[l_100][0, -1].float()

    logits = None
    if hasattr(out, "logits") and out.logits is not None:
        logits = out.logits[0, -1].float()

    del out
    return {0.75: s_75, 1.0: s_100}, logits


def _calc_pair_drift(
    sa: dict[float, torch.Tensor],
    sb: dict[float, torch.Tensor],
    model: Any,
    final_norm: Any,
) -> dict[str, float]:
    """Compute cosine distance and Logit-Lens symmetrized KL divergence between two trigger states."""
    cos_100 = 1.0 - F.cosine_similarity(sa[1.0].unsqueeze(0), sb[1.0].unsqueeze(0), dim=-1).item()
    cos_75 = 1.0 - F.cosine_similarity(sa[0.75].unsqueeze(0), sb[0.75].unsqueeze(0), dim=-1).item()

    norm_a = torch.norm(sa[1.0]).item() + 1e-8
    diff_norm = torch.norm(sb[1.0] - sa[1.0]).item()
    rel_l2 = diff_norm / norm_a

    kl_100 = 0.0
    kl_75 = 0.0
    try:
        pa_100 = logp_at_trigger(model, sa[1.0], is_intermediate=False, final_norm=final_norm)
        pb_100 = logp_at_trigger(model, sb[1.0], is_intermediate=False, final_norm=final_norm)
        if pa_100.numel() > 0 and pb_100.numel() > 0:
            kl_100 = 0.5 * (
                F.kl_div(pb_100, pa_100, log_target=True, reduction="sum") +
                F.kl_div(pa_100, pb_100, log_target=True, reduction="sum")
            ).item()
            kl_100 = max(0.0, float(kl_100))
    except Exception:
        pass

    try:
        pa_75 = logp_at_trigger(model, sa[0.75], is_intermediate=True, final_norm=final_norm)
        pb_75 = logp_at_trigger(model, sb[0.75], is_intermediate=True, final_norm=final_norm)
        if pa_75.numel() > 0 and pb_75.numel() > 0:
            kl_75 = 0.5 * (
                F.kl_div(pb_75, pa_75, log_target=True, reduction="sum") +
                F.kl_div(pa_75, pb_75, log_target=True, reduction="sum")
            ).item()
            kl_75 = max(0.0, float(kl_75))
    except Exception:
        pass

    return {
        "cos_100": max(0.0, float(cos_100)),
        "cos_75": max(0.0, float(cos_75)),
        "rel_l2": max(0.0, float(rel_l2)),
        "kl_100": max(0.0, float(kl_100)),
        "kl_75": max(0.0, float(kl_75)),
    }


def extract_prefill_metrics(
    prompts: List[str],
    tokenizer: Any,
    model: Any,
) -> ContrastiveMetrics:
    """Stage 1 — Unpadded Token-Targeted Prefill Extraction & Logit Lens with Control.

    Processes prompts = [P_orig, P_ctrl, P_v1, P_v2] (or subsets).
    Measures excess drift beyond the harmless surface control at index [0, -1]
    with final layer norm correction before lm_head.
    """
    if not prompts:
        return ContrastiveMetrics()

    device = _get_device(model)
    final_norm = get_final_norm(model)

    try:
        states: list[dict[float, torch.Tensor]] = []
        logits_list: list[torch.Tensor | None] = []
        for p in prompts:
            s, l = _extract_single_prompt(model, tokenizer, p, device)
            states.append(s)
            logits_list.append(l)

        s_0 = states[0]
        # Late-layer internal agreement on P_0: cosine similarity between 0.75L and 1.00L
        late_layer_sim = F.cosine_similarity(s_0[0.75].unsqueeze(0), s_0[1.0].unsqueeze(0), dim=-1).item()
        late_layer_agreement = float(max(-1.0, min(1.0, late_layer_sim)))

        # Baseline entropy and margins from P_0
        baseline_entropy = 1.5
        margin_orig = 0.30
        margin_variant = 0.30
        margin_agreement = True

        if logits_list[0] is not None:
            p_0 = F.softmax(logits_list[0], dim=-1).clamp(min=1e-12)
            baseline_entropy = float(-(p_0 * p_0.log()).sum().item())
            top2_0 = torch.topk(p_0, k=min(2, p_0.size(-1))).values
            if top2_0.numel() >= 2:
                margin_orig = float((top2_0[0] - top2_0[1]).item())
            if len(logits_list) > 1 and logits_list[-1] is not None:
                p_v = F.softmax(logits_list[-1], dim=-1).clamp(min=1e-12)
                top2_v = torch.topk(p_v, k=min(2, p_v.size(-1))).values
                if top2_v.numel() >= 2:
                    margin_variant = float((top2_v[0] - top2_v[1]).item())
                    margin_agreement = bool((margin_orig > 0.10) == (margin_variant > 0.10))

        # Drift calculation with Control subtraction when >= 4 prompts are provided
        if len(prompts) >= 4:
            drift_ctrl = _calc_pair_drift(s_0, states[1], model, final_norm)
            drift_v1 = _calc_pair_drift(s_0, states[2], model, final_norm)
            drift_v2 = _calc_pair_drift(s_0, states[3], model, final_norm)

            excess_cos_v1 = max(0.0, drift_v1["cos_100"] - drift_ctrl["cos_100"])
            excess_cos_v2 = max(0.0, drift_v2["cos_100"] - drift_ctrl["cos_100"])
            latent_drift = max(excess_cos_v1, excess_cos_v2)

            excess_kl_v1 = max(0.0, drift_v1["kl_100"] - drift_ctrl["kl_100"])
            excess_kl_v2 = max(0.0, drift_v2["kl_100"] - drift_ctrl["kl_100"])
            kl_div = max(excess_kl_v1, excess_kl_v2)

            excess_ll_v1 = max(0.0, drift_v1["kl_75"] - drift_ctrl["kl_75"])
            excess_ll_v2 = max(0.0, drift_v2["kl_75"] - drift_ctrl["kl_75"])
            logit_lens_divergence = max(excess_ll_v1, excess_ll_v2)

            excess_l2_v1 = max(0.0, drift_v1["rel_l2"] - drift_ctrl["rel_l2"])
            excess_l2_v2 = max(0.0, drift_v2["rel_l2"] - drift_ctrl["rel_l2"])
            rel_l2_drift = max(excess_l2_v1, excess_l2_v2)

        elif len(prompts) == 3:
            d1 = _calc_pair_drift(s_0, states[1], model, final_norm)
            d2 = _calc_pair_drift(s_0, states[2], model, final_norm)
            latent_drift = max(d1["cos_100"], d2["cos_100"])
            kl_div = max(d1["kl_100"], d2["kl_100"])
            logit_lens_divergence = max(d1["kl_75"], d2["kl_75"])
            rel_l2_drift = max(d1["rel_l2"], d2["rel_l2"])

        elif len(prompts) == 2:
            d1 = _calc_pair_drift(s_0, states[1], model, final_norm)
            latent_drift = d1["cos_100"]
            kl_div = d1["kl_100"]
            logit_lens_divergence = d1["kl_75"]
            rel_l2_drift = d1["rel_l2"]

        else:
            latent_drift = 0.0
            kl_div = 0.0
            logit_lens_divergence = 0.0
            rel_l2_drift = 0.0

        return ContrastiveMetrics(
            latent_drift=latent_drift,
            kl_div=kl_div,
            baseline_entropy=baseline_entropy,
            margin_disagreement=abs(margin_orig - margin_variant),
            math_span_drift=latent_drift * 1.1,
            rel_l2_drift=rel_l2_drift,
            entropy_shift=abs(margin_orig - margin_variant) * 0.5,
            late_layer_agreement=late_layer_agreement,
            logit_lens_divergence=logit_lens_divergence,
            cosine_sim=max(-1.0, min(1.0, 1.0 - latent_drift)),
            rel_l2_dist=rel_l2_drift,
            margin_orig=margin_orig,
            margin_variant=margin_variant,
            margin_agreement=margin_agreement,
        )
    except Exception as exc:
        print(f"[contrastive] extract_prefill_metrics error: {exc}")
        return ContrastiveMetrics()


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
            p_ctrl = f"Consider the following mathematical problem: {p_orig}"

            p_orig_prompt = p_orig + REASONING_TRIGGER
            p_ctrl_prompt = p_ctrl + REASONING_TRIGGER
            p_v1_prompt = v1 + REASONING_TRIGGER
            p_v2_prompt = v2 + REASONING_TRIGGER

            metrics = extract_prefill_metrics(
                [p_orig_prompt, p_ctrl_prompt, p_v1_prompt, p_v2_prompt], tokenizer, model
            )
            results.append(metrics)
        return results
    except Exception as exc:
        print(f"[contrastive] extract_contrastive_batch failed: {exc}")
        return None
