from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_CONFIGS, key=["SEQ_Q", "SEQ_K", "HEAD_DIM"])
@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    H, SEQ_Q, SEQ_K,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
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
        mask=m_mask[:, None],
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum(SEQ_K, (pid_m + 1) * BLOCK_M)
    else:
        hi = SEQ_K

    for n_start in range(0, hi, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < SEQ_K

        k = tl.load(
            k_base + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
            mask=n_mask[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * sm_scale  # fp32 (BLOCK_M, BLOCK_N)

        valid = m_mask[:, None] & n_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (m_offs[:, None] >= n_offs[None, :])
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # NaN guard: rows with every score masked have m_new == -inf.
        shift = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - shift)
        p = tl.exp(s - shift[:, None])

        v = tl.load(
            v_base + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=n_mask[:, None],
            other=0.0,
        )

        l_i = l_i * alpha + tl.sum(p, axis=1)
        # Cast p to the input dtype so the PV product runs on tensor cores.
        acc = acc * alpha[:, None] + tl.dot(p.to(q_ptr.dtype.element_ty), v)
        m_i = m_new

    # 0/0 guard for fully-masked (padded) rows; they are never stored.
    denom = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / denom[:, None]

    tl.store(
        o_base + m_offs[:, None] * stride_om + d_offs[None, :] * stride_od,
        out.to(o_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )
