"""Fused attention with streamed introspection stats (attention stats). No autotune,
fixed (BLOCK_M=64, BLOCK_N=32, num_warps=4) - keeps the wrapper runnable under
TRITON_INTERPRET=1; the canonical `attention` untouched."""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl

__all__ = ["attention_with_stats", "AttnStats"]

_BLOCK_M = 64
_BLOCK_N = 32


class AttnStats(NamedTuple):
    entropy: torch.Tensor      # (B, H, seq_q) fp32, nats; 0 for padded rows
    max_weight: torch.Tensor   # (B, H, seq_q) fp32; 0 for padded rows
    top_idx: torch.Tensor      # (B, H, seq_q, K) int32, sorted desc; -1 pads
    top_p: torch.Tensor        # (B, H, seq_q, K) fp32, sorted desc; 0 pads


@triton.jit
def _attention_stats_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    ent_ptr, maxw_ptr, topi_ptr, topp_ptr,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    H, SEQ_Q, SEQ_K,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WITH_STATS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    off_z = pid_zh // H
    off_h = pid_zh % H

    q_base = q_ptr + off_z * stride_qz + off_h * stride_qh
    k_base = k_ptr + off_z * stride_kz + off_h * stride_kh
    v_base = v_ptr + off_z * stride_vz + off_h * stride_vh
    o_base = o_ptr + off_z * stride_oz + off_h * stride_oh

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)
    m_mask = m_offs < SEQ_Q

    q = tl.load(
        q_base + m_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd,
        mask=m_mask[:, None], other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    r_i = tl.zeros((BLOCK_M,), dtype=tl.float32)              # sum exp(s-shift)*s
    top_s = tl.full((BLOCK_M, TOPK), float("-inf"), dtype=tl.float32)
    top_i = tl.full((BLOCK_M, TOPK), -1, dtype=tl.int32)

    if IS_CAUSAL:
        hi = tl.minimum(SEQ_K, (pid_m + 1) * BLOCK_M)
    else:
        hi = SEQ_K

    for n_start in range(0, hi, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < SEQ_K

        k = tl.load(
            k_base + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
            mask=n_mask[:, None], other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * sm_scale

        valid = m_mask[:, None] & n_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (m_offs[:, None] >= n_offs[None, :])
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        shift = tl.where(m_new == float("-inf"), 0.0, m_new)  # NaN guard
        alpha = tl.exp(m_i - shift)
        p = tl.exp(s - shift[:, None])

        v = tl.load(
            v_base + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=n_mask[:, None], other=0.0,
        )

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q_ptr.dtype.element_ty), v)

        if WITH_STATS:
            # r-recurrence: same alpha rescale as l.
            r_i = r_i * alpha + tl.sum(p * tl.where(valid, s, 0.0), axis=1)
            s_work = s
            col = tl.arange(0, BLOCK_N)
            slot = tl.arange(0, TOPK)
            for _t in range(TOPK):
                bv = tl.max(s_work, axis=1)
                bi = tl.argmax(s_work, axis=1).to(tl.int32)
                mv = tl.min(top_s, axis=1)
                mi_slot = tl.argmin(top_s, axis=1).to(tl.int32)
                do_ins = bv > mv
                ins = do_ins[:, None] & (slot[None, :] == mi_slot[:, None])
                top_s = tl.where(ins, bv[:, None], top_s)
                top_i = tl.where(ins, (bi + n_start)[:, None], top_i)
                s_work = tl.where(col[None, :] == bi[:, None],
                                  float("-inf"), s_work)

        m_i = m_new

    denom = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / denom[:, None]
    tl.store(
        o_base + m_offs[:, None] * stride_om + d_offs[None, :] * stride_od,
        out.to(o_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )
    if WITH_STATS:
        row_valid = l_i > 0.0
        # H = log l + m - r/l  (see module docstring); clamp fp roundoff.
        ent = tl.log(denom) + tl.where(row_valid, m_i, 0.0) - r_i / denom
        ent = tl.maximum(tl.where(row_valid, ent, 0.0), 0.0)
        maxw = tl.where(row_valid, 1.0 / denom, 0.0)

        row_base = pid_zh * SEQ_Q + m_offs                     # contiguous outs
        tl.store(ent_ptr + row_base, ent, mask=m_mask)
        tl.store(maxw_ptr + row_base, maxw, mask=m_mask)

        shift_f = tl.where(m_i == float("-inf"), 0.0, m_i)
        slot = tl.arange(0, TOPK)
        top_valid = top_s > float("-inf")
        top_p = tl.where(top_valid,
                         tl.exp(top_s - shift_f[:, None]) / denom[:, None], 0.0)
        top_i_out = tl.where(top_valid, top_i, -1)
        top_base = row_base[:, None] * TOPK + slot[None, :]
        tl.store(topp_ptr + top_base, top_p, mask=m_mask[:, None])
        tl.store(topi_ptr + top_base, top_i_out, mask=m_mask[:, None])


def _device_ok(t: torch.Tensor) -> bool:
    return t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


def attention_with_stats(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool = False, scale: float | None = None,
    topk: int = 4, collect_stats: bool = True,
):
    """Fused attention returning (out, AttnStats | None)."""
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.ndim != 4:
            raise ValueError(f"{name} must be 4D (B,H,seq,D), got {t.ndim}D")
        if not _device_ok(t):
            raise RuntimeError("attention_with_stats requires CUDA tensors "
                               "(or TRITON_INTERPRET=1 for CPU runs)")
    B, H, SQ, D = q.shape
    Bk, Hk, SK, Dk = k.shape
    if (B, H, D) != (Bk, Hk, Dk) or v.shape != k.shape:
        raise ValueError("q/k/v batch, head, and head-dim must agree; "
                         "k and v shapes must match")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) \
            or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must share one dtype in {fp16, bf16, fp32}")
    if D < 16 or D > 128 or (D & (D - 1)):
        raise ValueError(f"HEAD_DIM must be a power of two in [16,128], got {D}")
    if causal and SQ != SK:
        raise ValueError("causal requires seq_q == seq_k")
    if not (1 <= topk <= 8) or topk > _BLOCK_N:
        raise ValueError(f"topk must be in [1, 8], got {topk}")

    sm_scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    dev = q.device
    out = torch.empty((B, H, SQ, D), dtype=q.dtype, device=dev)
    if collect_stats:
        ent = torch.empty((B, H, SQ), dtype=torch.float32, device=dev)
        maxw = torch.empty((B, H, SQ), dtype=torch.float32, device=dev)
        topi = torch.empty((B, H, SQ, topk), dtype=torch.int32, device=dev)
        topp = torch.empty((B, H, SQ, topk), dtype=torch.float32, device=dev)
    else:  # dummy 1-elem buffers; kernel never touches them (WITH_STATS=0)
        ent = maxw = torch.empty(1, dtype=torch.float32, device=dev)
        topi = torch.empty(1, dtype=torch.int32, device=dev)
        topp = torch.empty(1, dtype=torch.float32, device=dev)

    grid = (triton.cdiv(SQ, _BLOCK_M), B * H)
    _attention_stats_kernel[grid](
        q, k, v, out, ent, maxw, topi, topp,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, SQ, SK, sm_scale,
        HEAD_DIM=D, IS_CAUSAL=causal,
        WITH_STATS=collect_stats, TOPK=topk,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
        num_warps=4,
    )
    if not collect_stats:
        return out, None

    # In-kernel top set is unsorted; K elements per row, sort on host.
    order = torch.argsort(topp, dim=-1, descending=True, stable=True)
    topp = torch.gather(topp, -1, order)
    topi = torch.gather(topi, -1, order)
    return out, AttnStats(ent, maxw, topi, topp)
