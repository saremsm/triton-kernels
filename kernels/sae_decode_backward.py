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


@triton.jit
def _grad_w_atomic_kernel(
    idx_ptr, val_ptr, g_ptr, gw_ptr,
    N, K, D,
    stride_it, stride_ik,
    stride_vt, stride_vk,
    stride_gn, stride_gd,
    stride_wf, stride_wd,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_n >= N or pid_k >= K:
        return

    v = tl.load(val_ptr + pid_n * stride_vt + pid_k * stride_vk).to(tl.float32)
    if v == 0.0:                       # pad contributes nothing to grad_W
        return
    f = tl.load(idx_ptr + pid_n * stride_it + pid_k * stride_ik)

    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        g = tl.load(g_ptr + pid_n * stride_gn + d * stride_gd,
                    mask=dm, other=0.0).to(tl.float32)
        tl.atomic_add(gw_ptr + f * stride_wf + d * stride_wd, v * g, mask=dm)


@triton.jit
def _grad_w_segmented_kernel(
    seg_ptr, contrib_val_ptr, contrib_row_ptr, g_ptr, gw_ptr,
    F, D,
    stride_gn, stride_gd,
    stride_wf, stride_wd,
    BLOCK_D: tl.constexpr,
):
    f = tl.program_id(0)
    if f >= F:
        return
    start = tl.load(seg_ptr + f)
    end = tl.load(seg_ptr + f + 1)
    if start == end:                   # feature never fired
        return

    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        # Fixed-order reduction: iterate contributions in sorted position.
        for i in range(start, end):
            v = tl.load(contrib_val_ptr + i).to(tl.float32)
            n = tl.load(contrib_row_ptr + i)
            g = tl.load(g_ptr + n * stride_gn + d * stride_gd,
                        mask=dm, other=0.0).to(tl.float32)
            acc += v * g
        tl.store(gw_ptr + f * stride_wf + d * stride_wd,
                 acc.to(gw_ptr.dtype.element_ty), mask=dm)


def _prep_segments(idx: torch.Tensor, val: torch.Tensor):
    """Flatten active (n, k) contributions and stable-sort by feature id."""
    N, K = idx.shape
    device = idx.device
    flat_f = idx.reshape(-1).long()
    flat_v = val.reshape(-1)
    flat_n = (torch.arange(N, device=device)
              .repeat_interleave(K))                 # source token per slot

    active = flat_v != 0.0                           # drop pads
    flat_f, flat_v, flat_n = flat_f[active], flat_v[active], flat_n[active]

    # Stable sort by feature id; equal keys keep their (n, k) row-major order.
    order = torch.argsort(flat_f, stable=True)
    sf = flat_f[order]
    contrib_val = flat_v[order].contiguous()
    contrib_row = flat_n[order].to(torch.int32).contiguous()
    return sf, contrib_val, contrib_row


def _segments_to_offsets(sorted_feat: torch.Tensor, F: int, device):
    """seg[f]..seg[f+1] runs, via searchsorted on the sorted feature ids."""
    boundaries = torch.arange(F + 1, device=device)
    seg = torch.searchsorted(sorted_feat, boundaries, right=False)
    return seg.to(torch.int32).contiguous()


def sae_decode_backward(grad_out: torch.Tensor, idx: torch.Tensor,
                        val: torch.Tensor, W_dec: torch.Tensor,
                        backend: str = "atomic"):
    """Return (grad_val, grad_W_dec, grad_b_dec). backend in {atomic, deterministic}."""
    if backend not in ("atomic", "deterministic"):
        raise ValueError(f"backend must be 'atomic' or 'deterministic', got {backend!r}")
    N, K = idx.shape
    F, D = W_dec.shape
    dev = W_dec.device
    g = grad_out.contiguous()

    grad_b = g.to(torch.float32).sum(0).to(W_dec.dtype)

    grad_val = torch.empty((N, K), dtype=W_dec.dtype, device=dev)
    grad_w = torch.zeros((F, D), dtype=torch.float32, device=dev)  # fp32 accum
    if N == 0 or K == 0:
        return grad_val, grad_w.to(W_dec.dtype), grad_b

    # grad_val (shared by both backends)
    _grad_val_kernel[(N, K)](
        idx, g, W_dec, val, grad_val,
        N, K, D,
        idx.stride(0), idx.stride(1),
        g.stride(0), g.stride(1),
        W_dec.stride(0), W_dec.stride(1),
        val.stride(0), val.stride(1),
        grad_val.stride(0), grad_val.stride(1),
        BLOCK_D=_BLOCK_D,
    )

    if backend == "atomic":
        _grad_w_atomic_kernel[(N, K)](
            idx, val, g, grad_w,
            N, K, D,
            idx.stride(0), idx.stride(1),
            val.stride(0), val.stride(1),
            g.stride(0), g.stride(1),
            grad_w.stride(0), grad_w.stride(1),
            BLOCK_D=_BLOCK_D,
        )
    else:
        sorted_feat, contrib_val, contrib_row = _prep_segments(idx, val)
        seg = _segments_to_offsets(sorted_feat, F, dev)
        _grad_w_segmented_kernel[(F,)](
            seg, contrib_val, contrib_row, g, grad_w,
            F, D,
            g.stride(0), g.stride(1),
            grad_w.stride(0), grad_w.stride(1),
            BLOCK_D=_BLOCK_D,
        )

    return grad_val, grad_w.to(W_dec.dtype), grad_b


class SparseSAEDecode(torch.autograd.Function):
    """Autograd-wrapped sparse decode; trains the decoder end to end."""

    @staticmethod
    def forward(ctx, idx, val, W_dec, b_dec, backend="atomic"):
        ctx.save_for_backward(idx, val, W_dec)
        ctx.backend = backend
        return sae_decode(idx, val, W_dec, b_dec)

    @staticmethod
    def backward(ctx, grad_out):
        idx, val, W_dec = ctx.saved_tensors
        grad_val, grad_W, grad_b = sae_decode_backward(
            grad_out, idx, val, W_dec, backend=ctx.backend)
        # grads match forward args: (idx, val, W_dec, b_dec, backend)
        return None, grad_val, grad_W, grad_b, None


def sae_decode_fn(idx, val, W_dec, b_dec, backend="atomic"):
    """Functional entry point with autograd support."""
    return SparseSAEDecode.apply(idx, val, W_dec, b_dec, backend)
