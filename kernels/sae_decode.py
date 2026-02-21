from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

__all__ = ["sae_decode", "sparsify", "densify"]


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
