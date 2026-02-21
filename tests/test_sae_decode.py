import os

import pytest
import torch
from kernels.sae_decode import sparsify, densify


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
