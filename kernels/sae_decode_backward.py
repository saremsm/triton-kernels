from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from kernels.sae_decode import _BLOCK_D, _device_ok, sae_decode

__all__ = ["SparseSAEDecode", "sae_decode_fn", "sae_decode_backward"]


@triton.jit
def _grad_val_kernel(
    idx_ptr, g_ptr, w_ptr, val_ptr, gval_ptr,
    N, K, D,
    stride_it, stride_ik,
    stride_gn, stride_gd,
    stride_wf, stride_wd,
    stride_vt, stride_vk,
    stride_ot, stride_ok,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_n >= N or pid_k >= K:
        return

    v = tl.load(val_ptr + pid_n * stride_vt + pid_k * stride_vk)
    is_pad = v == 0.0

    f = tl.load(idx_ptr + pid_n * stride_it + pid_k * stride_ik)
    acc = tl.zeros((), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        g = tl.load(g_ptr + pid_n * stride_gn + d * stride_gd,
                    mask=dm, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + f * stride_wf + d * stride_wd,
                    mask=dm, other=0.0).to(tl.float32)
        acc += tl.sum(g * w)

    out = tl.where(is_pad, 0.0, acc)
    tl.store(gval_ptr + pid_n * stride_ot + pid_k * stride_ok,
             out.to(gval_ptr.dtype.element_ty))
