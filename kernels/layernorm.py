from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_BLOCK: int = 16384


@triton.jit
def _layernorm_kernel(
    out_ptr,
    x_ptr,
    w_ptr,
    b_ptr,
    row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)  # re-zero padding BEFORE variance
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b

    tl.store(
        out_ptr + row * row_stride + cols,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm over the last dimension, matching F.layer_norm semantics."""
    if not x.is_cuda:
        raise ValueError("triton layernorm requires a CUDA tensor")
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    if weight.shape != (n_cols,) or bias.shape != (n_cols,):
        raise ValueError("weight/bias must have shape (last_dim,)")

    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK:
        raise ValueError(
            f"layernorm width {n_cols} exceeds single-block limit {MAX_BLOCK}"
        )

    x2d = x.reshape(-1, n_cols).contiguous()
    n_rows = x2d.shape[0]

    num_warps = 4
    if block >= 2048:
        num_warps = 8
    if block >= 8192:
        num_warps = 16

    out = torch.empty_like(x2d)
    _layernorm_kernel[(n_rows,)](
        out,
        x2d,
        weight,
        bias,
        x2d.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )
    return out.reshape(orig_shape)
