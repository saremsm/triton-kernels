"""Tests for the sparse SAE decoder backward (kernel #2). Run on CUDA when
available; on CPU via `TRITON_INTERPRET=1 pytest ...`. Atomic nondeterminism is a
GPU-scale timing property, demonstrated (not asserted) in the benchmark."""
import os

import pytest
import torch

from kernels.sae_decode import densify, sparsify
from kernels.sae_decode_backward import (sae_decode_backward, sae_decode_fn)

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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [1, 5, 257])
@pytest.mark.parametrize("l0", [1, MEASURED_L0])
def test_backward_matches_autograd(backend, n, l0):
    W, b = make_weights(DEVICE)
    h = make_sparse_h(n, N_FEATURES, l0, DEVICE, seed=n * 10 + l0)
    idx, val = sparsify(h)
    grad_out = torch.randn(n, D_MODEL, device=DEVICE)

    gv, gW, gb = sae_decode_backward(grad_out, idx, val, W, backend=backend)
    rv, rW, rb = autograd_reference(idx, val, W, b, grad_out)

    torch.testing.assert_close(gv, rv, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gW, rW, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(gb, rb, atol=1e-4, rtol=1e-4)


def test_checkpoint_dims_l0_27_backward():
    """Checkpoint config: 768->3072 at L0=27, both backends vs autograd."""
    W, b = make_weights(DEVICE)
    h = make_sparse_h(512, N_FEATURES, MEASURED_L0, DEVICE, seed=42)
    idx, val = sparsify(h)
    grad_out = torch.randn(512, D_MODEL, device=DEVICE)
    rv, rW, rb = autograd_reference(idx, val, W, b, grad_out)
    for backend in BACKENDS:
        gv, gW, gb = sae_decode_backward(grad_out, idx, val, W, backend=backend)
        torch.testing.assert_close(gW, rW, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(gv, rv, atol=1e-4, rtol=1e-4)


def test_deterministic_backend_bitwise_stable():
    """The deterministic grad_W_dec must be byte-identical across >=3 runs."""
    W, _ = make_weights(DEVICE)
    h = make_sparse_h(256, N_FEATURES, MEASURED_L0, DEVICE, seed=7)
    idx, val = sparsify(h)
    grad_out = torch.randn(256, D_MODEL, device=DEVICE)
    runs = [sae_decode_backward(grad_out, idx, val, W, backend="deterministic")[1]
            for _ in range(3)]
    assert torch.equal(runs[0], runs[1])
    assert torch.equal(runs[1], runs[2])


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


def test_both_backends_agree():
    """Atomic and deterministic must agree to fp32 tol (same math, diff order)."""
    W, _ = make_weights(DEVICE)
    h = make_sparse_h(128, N_FEATURES, MEASURED_L0, DEVICE, seed=11)
    idx, val = sparsify(h)
    grad_out = torch.randn(128, D_MODEL, device=DEVICE)
    _, gW_a, _ = sae_decode_backward(grad_out, idx, val, W, backend="atomic")
    _, gW_d, _ = sae_decode_backward(grad_out, idx, val, W, backend="deterministic")
    torch.testing.assert_close(gW_a, gW_d, atol=1e-3, rtol=1e-3)


def test_high_collision_matches_autograd():
    """All tokens fire the SAME feature: max write contention for grad_W."""
    W, b = make_weights(DEVICE)
    n = 200
    idx = torch.full((n, 1), 100, dtype=torch.int32, device=DEVICE)
    val = torch.rand(n, 1, device=DEVICE) + 0.1
    grad_out = torch.randn(n, D_MODEL, device=DEVICE)
    rv, rW, rb = autograd_reference(idx, val, W, b, grad_out)
    for backend in BACKENDS:
        gv, gW, gb = sae_decode_backward(grad_out, idx, val, W, backend=backend)
        torch.testing.assert_close(gW, rW, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_autograd_function_end_to_end(backend):
    """SparseSAEDecode.apply backprops correctly through the sparse path."""
    W, b = make_weights(DEVICE)
    h = make_sparse_h(64, N_FEATURES, MEASURED_L0, DEVICE, seed=5)
    idx, val = sparsify(h)
    val = val.clone().requires_grad_(True)
    W_ = W.clone().requires_grad_(True)
    b_ = b.clone().requires_grad_(True)
    out = sae_decode_fn(idx, val, W_, b_, backend=backend)
    loss = (out ** 2).sum()
    loss.backward()
    assert val.grad is not None and W_.grad is not None and b_.grad is not None
    assert val.grad.shape == val.shape
    assert W_.grad.shape == W_.shape
    # cross-check W grad against the functional backward
    grad_out = 2 * sae_decode_fn(idx, val.detach(), W, b, backend=backend)
    _, gW_ref, _ = sae_decode_backward(grad_out, idx, val.detach(), W, backend=backend)
    torch.testing.assert_close(W_.grad, gW_ref, atol=1e-3, rtol=1e-3)


def test_rejects_bad_backend():
    W, _ = make_weights(DEVICE)
    h = make_sparse_h(4, N_FEATURES, 5, DEVICE, seed=1)
    idx, val = sparsify(h)
    grad_out = torch.randn(4, D_MODEL, device=DEVICE)
    with pytest.raises(ValueError):
        sae_decode_backward(grad_out, idx, val, W, backend="nope")
