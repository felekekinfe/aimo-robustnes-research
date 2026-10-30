#!/usr/bin/env python3
"""
Flash Linear Attention (FLA) & Chunkwise Recurrent Triton Kernels.
Fuses feature mapping phi(x), recurrent state accumulation S_t = S_{t-1} + phi(K_t)^T V_t,
and output projection with O(N) memory complexity to prevent VRAM bottlenecks on long Olympiad contexts.

Includes seamless high-performance PyTorch fallback for CPU / environments without Triton.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


# ==============================================================================
# 1. TRITON CHUNKWISE LINEAR ATTENTION KERNELS
# ==============================================================================
if HAS_TRITON:
    @triton.jit
    def _fused_chunkwise_linear_attn_fwd_kernel(
        Q, K, V, Out, InitialState, FinalState,
        stride_qz, stride_qh, stride_qn, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_on, stride_od,
        stride_sz, stride_sh, stride_sk, stride_sv,
        Z, H, N, D_K: tl.constexpr, D_V: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
    ):
        """
        Fused Chunkwise Linear Attention Forward Kernel:
        Accumulates recurrent memory state S in on-chip SRAM:
            S_c = S_{c-1} + K_chunk^T @ V_chunk
            Out_chunk = Q_chunk @ S_{c-1} + (Q_chunk @ K_chunk^T)_causal @ V_chunk
        """
        pid_z = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_k = tl.arange(0, D_K)
        offs_v = tl.arange(0, D_V)

        # Allocate on-chip state accumulator in SRAM
        state = tl.zeros([D_K, D_V], dtype=tl.float32)

        # Load initial state if provided
        if InitialState is not None:
            state_ptr = (
                InitialState
                + pid_z * stride_sz
                + pid_h * stride_sh
                + offs_k[:, None] * stride_sk
                + offs_v[None, :] * stride_sv
            )
            state += tl.load(state_ptr)

        num_chunks = tl.cdiv(N, CHUNK_SIZE)

        for c in range(num_chunks):
            chunk_start = c * CHUNK_SIZE
            chunk_offs = chunk_start + tl.arange(0, CHUNK_SIZE)
            mask_chunk = chunk_offs < N

            # Pointers for this chunk
            q_ptr = (
                Q
                + pid_z * stride_qz
                + pid_h * stride_qh
                + chunk_offs[:, None] * stride_qn
                + offs_k[None, :] * stride_qd
            )
            k_ptr = (
                K
                + pid_z * stride_kz
                + pid_h * stride_kh
                + chunk_offs[:, None] * stride_kn
                + offs_k[None, :] * stride_kd
            )
            v_ptr = (
                V
                + pid_z * stride_vz
                + pid_h * stride_vh
                + chunk_offs[:, None] * stride_vn
                + offs_v[None, :] * stride_vd
            )
            out_ptr = (
                Out
                + pid_z * stride_oz
                + pid_h * stride_oh
                + chunk_offs[:, None] * stride_on
                + offs_v[None, :] * stride_od
            )

            # Load Q, K, V
            q = tl.load(q_ptr, mask=mask_chunk[:, None], other=0.0).to(tl.float32)
            k = tl.load(k_ptr, mask=mask_chunk[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=mask_chunk[:, None], other=0.0).to(tl.float32)

            # Feature map phi(x) = elu(x) + 1.0
            phi_q = tl.where(q > 0.0, q + 1.0, tl.exp(q))
            phi_k = tl.where(k > 0.0, k + 1.0, tl.exp(k))

            # 1. Inter-chunk projection: phi(Q) @ S_{c-1}
            out_inter = tl.dot(phi_q, state)

            # 2. Intra-chunk causal self-attention
            sim = tl.dot(phi_q, tl.trans(phi_k))
            causal_mask = (chunk_offs[:, None] >= chunk_offs[None, :]) & mask_chunk[:, None] & mask_chunk[None, :]
            sim_causal = tl.where(causal_mask, sim, 0.0)
            out_intra = tl.dot(sim_causal, v)

            # Total chunk output
            out = out_inter + out_intra
            tl.store(out_ptr, out.to(Q.dtype.element_ty), mask=mask_chunk[:, None])

            # Update recurrent state in SRAM for next chunk: S_c = S_{c-1} + phi(K)^T @ V
            state += tl.dot(tl.trans(phi_k), v)

        # Store final accumulated state if buffer provided
        if FinalState is not None:
            final_ptr = (
                FinalState
                + pid_z * stride_sz
                + pid_h * stride_sh
                + offs_k[:, None] * stride_sk
                + offs_v[None, :] * stride_sv
            )
            tl.store(final_ptr, state.to(FinalState.dtype.element_ty))


# ==============================================================================
# 2. VECTORIZED PYTORCH FALLBACK (MATHEMATICALLY IDENTICAL)
# ==============================================================================
def _torch_chunkwise_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized Chunkwise Linear Attention in Pure PyTorch.
    Operates in O(N * D_k * D_v) time and O(chunk_size * D_k) memory.
    """
    # Feature map phi(x) = elu(x) + 1.0 (strictly non-negative kernel)
    phi_q = F.elu(q) + 1.0
    phi_k = F.elu(k) + 1.0

    batch_size, num_heads, seq_len, d_k = q.shape
    d_v = v.shape[-1]

    if initial_state is not None:
        state = initial_state.clone().to(torch.float32)
    else:
        state = torch.zeros(
            (batch_size, num_heads, d_k, d_v),
            device=q.device,
            dtype=torch.float32,
        )

    out = torch.zeros(
        (batch_size, num_heads, seq_len, d_v),
        device=q.device,
        dtype=q.dtype,
    )

    num_chunks = math.ceil(seq_len / chunk_size)

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, seq_len)
        cur_chunk_len = end - start

        q_c = phi_q[:, :, start:end, :].to(torch.float32)  # [B, H, C, d_k]
        k_c = phi_k[:, :, start:end, :].to(torch.float32)  # [B, H, C, d_k]
        v_c = v[:, :, start:end, :].to(torch.float32)      # [B, H, C, d_v]

        # 1. Inter-chunk output: Q_c @ S_{c-1}
        # [B, H, C, d_k] @ [B, H, d_k, d_v] -> [B, H, C, d_v]
        out_inter = torch.matmul(q_c, state)

        # 2. Intra-chunk causal attention: (Q_c @ K_c^T)_causal @ V_c
        sim_c = torch.matmul(q_c, k_c.transpose(-1, -2))  # [B, H, C, C]
        causal_mask = torch.tril(torch.ones((cur_chunk_len, cur_chunk_len), device=q.device, dtype=torch.bool))
        sim_causal = sim_c.masked_fill(~causal_mask, 0.0)
        out_intra = torch.matmul(sim_causal, v_c)  # [B, H, C, d_v]

        out[:, :, start:end, :] = (out_inter + out_intra).to(q.dtype)

        # 3. Update recurrent state: S_c = S_{c-1} + K_c^T @ V_c
        # [B, H, d_k, C] @ [B, H, C, d_v] -> [B, H, d_k, d_v]
        state = state + torch.matmul(k_c.transpose(-1, -2), v_c)

    return out, state


# ==============================================================================
# 3. HIGH-LEVEL FLASH LINEAR ATTENTION MODULE
# ==============================================================================
class FlashLinearAttention(nn.Module):
    """
    Flash Linear Attention Module with Automatic Hardware Dispatch.
    Switches to Triton GPU kernels when CUDA is available, otherwise uses
    vectorized chunkwise PyTorch tensor computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        chunk_size: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            initial_state: Optional [batch_size, num_heads, head_dim, head_dim]
        Returns:
            out: [batch_size, seq_len, d_model]
            final_state: [batch_size, num_heads, head_dim, head_dim]
        """
        B, N, D = hidden_states.shape
        H = self.num_heads
        d_k = self.head_dim

        q = self.q_proj(hidden_states).view(B, N, H, d_k).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, H, d_k).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, H, d_k).transpose(1, 2)

        # Hardware-aware execution path
        if (
            HAS_TRITON
            and hidden_states.is_cuda
            and d_k in {16, 32, 64, 128}
            and d_k <= 128
        ):
            out = torch.empty_like(v)
            final_state = torch.zeros((B, H, d_k, d_k), device=hidden_states.device, dtype=torch.float32)

            grid = (B, H)
            _fused_chunkwise_linear_attn_fwd_kernel[grid](
                q, k, v, out, initial_state, final_state,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                final_state.stride(0), final_state.stride(1), final_state.stride(2), final_state.stride(3),
                B, H, N,
                D_K=d_k, D_V=d_k,
                CHUNK_SIZE=self.chunk_size,
            )
        else:
            out, final_state = _torch_chunkwise_linear_attention(
                q=q,
                k=k,
                v=v,
                chunk_size=self.chunk_size,
                initial_state=initial_state,
            )

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out), final_state


def compute_linear_attention_drift(
    h_orig: torch.Tensor,
    h_perturbed: torch.Tensor,
    fla_layer: Optional[FlashLinearAttention] = None,
) -> float:
    """
    Computes linear attention recurrent state divergence between original and perturbed inputs.
    Measures manifold stability under O(N) memory complexity.
    """
    if h_orig.ndim == 2:
        h_orig = h_orig.unsqueeze(0)
    if h_perturbed.ndim == 2:
        h_perturbed = h_perturbed.unsqueeze(0)

    if fla_layer is None:
        d_model = h_orig.shape[-1]
        fla_layer = FlashLinearAttention(d_model=d_model, num_heads=4, chunk_size=32, device=h_orig.device, dtype=h_orig.dtype)

    with torch.no_grad():
        _, state_orig = fla_layer(h_orig)
        _, state_pert = fla_layer(h_perturbed)

        # Cosine similarity between accumulated recurrent memory states
        s_orig_flat = state_orig.flatten()
        s_pert_flat = state_pert.flatten()
        cos_sim = F.cosine_similarity(s_orig_flat.unsqueeze(0), s_pert_flat.unsqueeze(0), dim=-1).item()
        return float(np.clip(1.0 - cos_sim, 0.0, 2.0))
