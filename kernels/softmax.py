from __future__ import annotations

import warnings

import torch
import triton
import triton.language as tl

# Rows wider than this fall back to torch.softmax.
MAX_BLOCK: int = 16384


@triton.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # -inf placeholder: masked lanes must not affect the max.
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)

    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(
        out_ptr + row * out_row_stride + cols,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension. CUDA only; fp16/bf16/fp32."""
    if not x.is_cuda:
        raise ValueError("triton softmax requires a CUDA tensor")
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x2d.shape

    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK:
        warnings.warn(
            f"softmax row width {n_cols} exceeds single-block limit "
            f"{MAX_BLOCK}; falling back to torch.softmax",
            stacklevel=2,
        )
        return torch.softmax(x, dim=-1)

    num_warps = 4
    if block >= 2048:
        num_warps = 8
    if block >= 8192:
        num_warps = 16

    out = torch.empty_like(x2d)
    _softmax_kernel[(n_rows,)](
        out,
        x2d,
        x2d.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )
    return out.reshape(orig_shape)
