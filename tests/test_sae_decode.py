import os

import pytest
import torch
from kernels.sae_decode import sae_decode, sparsify, densify


INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif INTERPRET:
    DEVICE = "cpu"
else:
    pytest.skip("requires a CUDA GPU or TRITON_INTERPRET=1",
                allow_module_level=True)

# Checkpoint dims from sae-gpt2-small (layer-8 checkpoint, measured L0=27).
D_MODEL, N_FEATURES, MEASURED_L0 = 768, 3072, 27


def make_sparse_h(n, n_features, l0, device, dtype=torch.float32, seed=0):
    """Exact-L0 synthetic post-ReLU activations: l0 positive entries per row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.zeros(n, n_features, dtype=dtype)
    for row in range(n):
        cols = torch.randperm(n_features, generator=g)[:l0]
        h[row, cols] = torch.rand(l0, generator=g) + 0.1
    return h.to(device)


def dense_reference(h, W, b):
    return (h.float() @ W.float() + b.float()).to(W.dtype)


def make_weights(device, dtype=torch.float32, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(N_FEATURES, D_MODEL, generator=g, dtype=torch.float32)
    W = W / W.norm(dim=1, keepdim=True)          # unit-norm rows, like the SAE
    b = torch.randn(D_MODEL, generator=g, dtype=torch.float32)
    return W.to(device, dtype), b.to(device, dtype)


@pytest.mark.parametrize("n", [1, 5, 257, 1024])
@pytest.mark.parametrize("l0", [1, MEASURED_L0])
def test_matches_dense_reference(n, l0):
    W, b = make_weights(DEVICE)
    h = make_sparse_h(n, N_FEATURES, l0, DEVICE, seed=n * 100 + l0)
    idx, val = sparsify(h)
    out = sae_decode(idx, val, W, b)
    ref = dense_reference(h, W, b)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_checkpoint_dims_l0_27():
    """The checkpoint configuration: 768 -> 3072 at the measured L0 = 27."""
    W, b = make_weights(DEVICE)
    h = make_sparse_h(2048, N_FEATURES, MEASURED_L0, DEVICE, seed=42)
    idx, val = sparsify(h)
    assert idx.shape[1] >= MEASURED_L0
    out = sae_decode(idx, val, W, b)
    ref = dense_reference(h, W, b)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(DEVICE != "cuda", reason="fp16 path is GPU-only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_matches_fp32_reference(dtype):
    W, b = make_weights(DEVICE, dtype)
    h = make_sparse_h(512, N_FEATURES, MEASURED_L0, DEVICE, dtype=dtype, seed=7)
    idx, val = sparsify(h)
    out = sae_decode(idx, val, W, b)
    ref = dense_reference(h, W, b)     # fp32 compute, cast back
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_zero_active_rows_equal_bias():
    """Rows with no active features must decode to exactly b_dec."""
    W, b = make_weights(DEVICE)
    h = make_sparse_h(64, N_FEATURES, 5, DEVICE, seed=3)
    h[10] = 0.0
    h[63] = 0.0
    idx, val = sparsify(h)
    out = sae_decode(idx, val, W, b)
    torch.testing.assert_close(out[10], b, atol=1e-6, rtol=0)
    torch.testing.assert_close(out[63], b, atol=1e-6, rtol=0)


def test_extra_padding_is_noop():
    """Widening K_pad with (idx=0, val=0) columns must not change output."""
    W, b = make_weights(DEVICE)
    h = make_sparse_h(33, N_FEATURES, 9, DEVICE, seed=5)
    idx, val = sparsify(h)
    out1 = sae_decode(idx, val, W, b)
    pad_i = torch.zeros(33, 24, dtype=torch.int32, device=DEVICE)
    pad_v = torch.zeros(33, 24, dtype=val.dtype, device=DEVICE)
    out2 = sae_decode(torch.cat([idx, pad_i], 1), torch.cat([val, pad_v], 1), W, b)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_sparsify_roundtrip():
    h = make_sparse_h(50, N_FEATURES, 13, DEVICE, seed=9)
    idx, val = sparsify(h)
    torch.testing.assert_close(densify(idx, val, N_FEATURES), h, atol=0, rtol=0)


def test_sparsify_exact_l0_padding_contract():
    h = make_sparse_h(20, N_FEATURES, MEASURED_L0, DEVICE, seed=11)
    idx, val = sparsify(h)
    assert idx.dtype == torch.int32
    assert val.shape[1] % 8 == 0                       # pad_multiple honored
    nnz_kept = (val > 0).sum(dim=-1)
    assert (nnz_kept == MEASURED_L0).all()             # nothing lost
    assert (val[:, MEASURED_L0:] == 0).all()           # pads exactly zero


def test_checkpoint_format_loads_and_matches(tmp_path):
    """Synthetic checkpoint in sae-gpt2-small's exact dict format."""
    W, b = make_weights("cpu")
    ckpt = {
        "config": {"d_model": D_MODEL, "n_features": N_FEATURES,
                   "l1_coefficient": 5e-3, "lr": 2e-4},
        "sae_state_dict": {
            "W_enc": W.T.clone(), "b_enc": torch.zeros(N_FEATURES),
            "W_dec": W.clone(), "b_dec": b.clone(),
            "feature_activation_counts": torch.zeros(N_FEATURES,
                                                     dtype=torch.long),
        },
        "layer": 8,
        "training_history": {"step": [50], "loss": [0.5]},
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    W_l = loaded["sae_state_dict"]["W_dec"].to(DEVICE)
    b_l = loaded["sae_state_dict"]["b_dec"].to(DEVICE)
    assert W_l.shape == (loaded["config"]["n_features"],
                         loaded["config"]["d_model"])

    h = make_sparse_h(128, N_FEATURES, MEASURED_L0, DEVICE, seed=13)
    idx, val = sparsify(h)
    out = sae_decode(idx, val, W_l, b_l)
    torch.testing.assert_close(out, dense_reference(h, W_l, b_l),
                               atol=1e-4, rtol=1e-4)


def test_forward_bitwise_deterministic():
    """No atomics in this forward: two runs must be bit-identical."""
    W, b = make_weights(DEVICE)
    h = make_sparse_h(300, N_FEATURES, MEASURED_L0, DEVICE, seed=17)
    idx, val = sparsify(h)
    out1 = sae_decode(idx, val, W, b)
    out2 = sae_decode(idx, val, W, b)
    assert torch.equal(out1, out2)


def test_noncontiguous_inputs():
    W, b = make_weights(DEVICE)
    h = make_sparse_h(40, N_FEATURES, 6, DEVICE, seed=19)
    idx, val = sparsify(h)
    idx_nc = idx.repeat_interleave(2, dim=0)[::2]      # strided view
    val_nc = val.repeat_interleave(2, dim=0)[::2]
    assert not idx_nc.is_contiguous()
    out = sae_decode(idx_nc, val_nc, W, b)
    torch.testing.assert_close(out, dense_reference(h, W, b),
                               atol=1e-4, rtol=1e-4)


def test_rejects_bad_inputs():
    W, b = make_weights(DEVICE)
    idx = torch.zeros(4, 8, dtype=torch.int32, device=DEVICE)
    val = torch.zeros(4, 8, dtype=torch.float32, device=DEVICE)
    with pytest.raises(ValueError):
        sae_decode(idx[:, :4], val, W, b)               # shape mismatch
    with pytest.raises(ValueError):
        sae_decode(idx.long(), val, W, b)               # wrong idx dtype
    with pytest.raises(ValueError):
        sae_decode(idx, val, W, b[:100])                # bias/D mismatch
    with pytest.raises(ValueError):
        sae_decode(idx, val.half(), W, b)               # dtype mismatch
    with pytest.raises(ValueError):
        sparsify(-torch.ones(2, 8, device=DEVICE))      # negative input
