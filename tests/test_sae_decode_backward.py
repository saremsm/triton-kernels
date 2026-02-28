"""Tests for the sparse SAE decoder backward (kernel #2). Run on CUDA when
available; on CPU via `TRITON_INTERPRET=1 pytest ...`. Atomic nondeterminism is a
GPU-scale timing property, demonstrated (not asserted) in the benchmark."""
import os

import pytest
import torch

from kernels.sae_decode_backward import sae_decode_backward
from kernels.sae_decode import densify, sparsify

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif INTERPRET:
    DEVICE = "cpu"
else:
    pytest.skip("requires a CUDA GPU or TRITON_INTERPRET=1",
                allow_module_level=True)

D_MODEL, N_FEATURES, MEASURED_L0 = 768, 3072, 27
BACKENDS = ["atomic", "deterministic"]


def make_sparse_h(n, n_features, l0, device, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.zeros(n, n_features, dtype=dtype)
    for row in range(n):
        cols = torch.randperm(n_features, generator=g)[:l0]
        h[row, cols] = (torch.rand(l0, generator=g) + 0.1).to(dtype)
    return h.to(device)


def make_weights(device, dtype=torch.float32, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(N_FEATURES, D_MODEL, generator=g, dtype=torch.float32)
    W = W / W.norm(dim=1, keepdim=True)
    b = torch.randn(D_MODEL, generator=g, dtype=torch.float32)
    return W.to(device, dtype), b.to(device, dtype)


def autograd_reference(idx, val, W, b, grad_out):
    """Dense autograd: densify to h, run h@W+b, backprop grad_out."""
    n_features = W.shape[0]
    h = densify(idx, val, n_features).clone().requires_grad_(True)
    W_ = W.clone().requires_grad_(True)
    b_ = b.clone().requires_grad_(True)
    out = h @ W_ + b_
    out.backward(grad_out)
    grad_h = h.grad                                  # (N, F)
    # gather grad_val at the active positions
    grad_val = torch.gather(grad_h, 1, idx.long())
    grad_val = grad_val * (val != 0)                 # pads -> 0
    return grad_val, W_.grad, b_.grad


def test_grad_val_pad_positions_are_zero():
    """grad_val at pad slots must be masked to 0, not grad_out . W_dec[0]."""
    W, _ = make_weights(DEVICE)
    h = make_sparse_h(20, N_FEATURES, 5, DEVICE, seed=3)      # L0=5
    idx, val = sparsify(h, pad_multiple=8)                    # K_pad=8 -> 3 pads/row
    grad_out = torch.randn(20, D_MODEL, device=DEVICE)
    gv, _, _ = sae_decode_backward(grad_out, idx, val, W, backend="atomic")
    pad_mask = (val == 0)
    assert pad_mask.any()                                     # test is meaningful
    assert torch.equal(gv[pad_mask], torch.zeros_like(gv[pad_mask]))
