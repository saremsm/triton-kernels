from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

__all__ = ["sae_decode", "sparsify", "densify"]

_BLOCK_T = 16
_BLOCK_D = 256


@triton.jit
def _sae_decode_kernel(
    idx_ptr, val_ptr, w_ptr, b_ptr, out_ptr,
    N, K, D,
    stride_it, stride_ik,
    stride_vt, stride_vk,
    stride_wf, stride_wd,
    stride_ot, stride_od,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_offs < N
    d_mask = d_offs < D
    tile_mask = t_mask[:, None] & d_mask[None, :]

    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    for k in range(0, K):
        # One (feature-id, value) pair per token at position k.
        f = tl.load(idx_ptr + t_offs * stride_it + k * stride_ik,
                    mask=t_mask, other=0)
        v = tl.load(val_ptr + t_offs * stride_vt + k * stride_vk,
                    mask=t_mask, other=0.0).to(tl.float32)
        w_ptrs = w_ptr + f[:, None] * stride_wf + d_offs[None, :] * stride_wd
        w = tl.load(w_ptrs, mask=tile_mask, other=0.0).to(tl.float32)
        acc += v[:, None] * w

    bias = tl.load(b_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    out_ptrs = out_ptr + t_offs[:, None] * stride_ot + d_offs[None, :] * stride_od
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=tile_mask)


def _device_ok(t: torch.Tensor) -> bool:
    return t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


def sae_decode(idx: torch.Tensor, val: torch.Tensor,
               W_dec: torch.Tensor, b_dec: torch.Tensor) -> torch.Tensor:
    """Sparse decode. idx/val: (N, K_pad); W_dec: (F, D); b_dec: (D,)."""
    if idx.ndim != 2 or val.shape != idx.shape:
        raise ValueError(f"idx/val must be matching 2D, got {tuple(idx.shape)} "
                         f"and {tuple(val.shape)}")
    if idx.dtype != torch.int32:
        raise ValueError(f"idx must be int32 (sparsify() emits it), got {idx.dtype}")
    if W_dec.ndim != 2:
        raise ValueError(f"W_dec must be (F, D), got {tuple(W_dec.shape)}")
    F_, D = W_dec.shape
    if b_dec.shape != (D,):
        raise ValueError(f"b_dec must be ({D},), got {tuple(b_dec.shape)}")
    if val.dtype != W_dec.dtype or b_dec.dtype != W_dec.dtype:
        raise ValueError("val, W_dec, b_dec must share one floating dtype")
    if W_dec.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported dtype {W_dec.dtype}")
    for t in (idx, val, W_dec, b_dec):
        if not _device_ok(t):
            raise RuntimeError("sae_decode requires CUDA tensors "
                               "(or TRITON_INTERPRET=1 for CPU interpreter runs)")

    N, K = idx.shape
    out = torch.empty((N, D), dtype=W_dec.dtype, device=W_dec.device)
    if N == 0:
        return out
    if K == 0:
        out[:] = b_dec
        return out

    grid = (triton.cdiv(N, _BLOCK_T), triton.cdiv(D, _BLOCK_D))
    _sae_decode_kernel[grid](
        idx, val, W_dec, b_dec, out,
        N, K, D,
        idx.stride(0), idx.stride(1),
        val.stride(0), val.stride(1),
        W_dec.stride(0), W_dec.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=_BLOCK_T, BLOCK_D=_BLOCK_D,
        num_warps=4,
    )
    return out


def sparsify(h: torch.Tensor, pad_multiple: int = 8):
    """Dense post-ReLU activations (N, F) -> padded sparse (idx, val)."""
    if h.ndim != 2:
        raise ValueError(f"h must be (N, F), got {tuple(h.shape)}")
    if h.numel() and h.min() < 0:
        raise ValueError("sparsify expects post-ReLU (nonnegative) activations")
    nnz = (h > 0).sum(dim=-1)
    k_max = int(nnz.max().item()) if h.numel() else 0
    K = max(1, -(-max(k_max, 1) // pad_multiple) * pad_multiple)
    K = min(K, h.shape[1])
    val, idx = torch.topk(h, K, dim=-1)
    return idx.to(torch.int32), val


def densify(idx: torch.Tensor, val: torch.Tensor, n_features: int) -> torch.Tensor:
    """Inverse of sparsify (reference/testing helper)."""
    h = torch.zeros((idx.shape[0], n_features), dtype=val.dtype, device=val.device)
    h.scatter_(1, idx.long(), val)
    return h
