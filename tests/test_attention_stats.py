"""Tests for fused attention with streamed stats (attention stats). CUDA when
available; CPU via TRITON_INTERPRET=1 (fp32 there). Top-k checks are value-based
(gather reference p at our indices) so argmax tie-breaking can't fail falsely."""

import math
import os

import pytest
import torch

from kernels.attention_stats import attention_with_stats

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
if torch.cuda.is_available():
    DEVICE, DTYPE, ATOL = "cuda", torch.float16, 2e-2
elif INTERPRET:
    DEVICE, DTYPE, ATOL = "cpu", torch.float32, 1e-4
else:
    pytest.skip("requires a CUDA GPU or TRITON_INTERPRET=1",
                allow_module_level=True)

SHAPES = [  # (causal, B, H, seq_q, seq_k, D)
    (False, 1, 2, 77, 53, 32),    # cross-attention, padded query rows
    (True, 1, 2, 77, 77, 32),     # causal, padded rows
    (False, 1, 1, 64, 64, 16),    # exact block fit
]


def make_qkv(B, H, SQ, SK, D, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, SQ, D, generator=g).to(DEVICE, DTYPE)
    k = torch.randn(B, H, SK, D, generator=g).to(DEVICE, DTYPE)
    v = torch.randn(B, H, SK, D, generator=g).to(DEVICE, DTYPE)
    return q, k, v


def reference(q, k, v, causal, topk):
    """Materialized fp32 softmax + explicit stats."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        SQ, SK = s.shape[-2], s.shape[-1]
        mask = torch.triu(torch.ones(SQ, SK, dtype=torch.bool,
                                     device=s.device), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    out = p @ v.float()
    plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
    ent = -plogp.sum(-1)
    maxw = p.max(-1).values
    top_p, top_i = torch.topk(p, topk, dim=-1)
    return out, ent, maxw, p, top_p, top_i


@pytest.mark.parametrize("causal,B,H,SQ,SK,D", SHAPES)
def test_out_matches_reference(causal, B, H, SQ, SK, D):
    q, k, v = make_qkv(B, H, SQ, SK, D, seed=SQ + SK)
    out, _ = attention_with_stats(q, k, v, causal=causal)
    ref_out = reference(q, k, v, causal, 4)[0]
    torch.testing.assert_close(out.float(), ref_out, atol=ATOL, rtol=ATOL)


@pytest.mark.parametrize("causal,B,H,SQ,SK,D", SHAPES)
def test_entropy_matches_reference(causal, B, H, SQ, SK, D):
    q, k, v = make_qkv(B, H, SQ, SK, D, seed=SQ + SK + 1)
    _, stats = attention_with_stats(q, k, v, causal=causal)
    ent_ref = reference(q, k, v, causal, 4)[1]
    torch.testing.assert_close(stats.entropy, ent_ref, atol=5 * ATOL,
                               rtol=5 * ATOL)


@pytest.mark.parametrize("causal,B,H,SQ,SK,D", SHAPES)
def test_max_weight_matches_reference(causal, B, H, SQ, SK, D):
    q, k, v = make_qkv(B, H, SQ, SK, D, seed=SQ + SK + 2)
    _, stats = attention_with_stats(q, k, v, causal=causal)
    maxw_ref = reference(q, k, v, causal, 4)[2]
    torch.testing.assert_close(stats.max_weight, maxw_ref, atol=5 * ATOL,
                               rtol=5 * ATOL)


@pytest.mark.parametrize("causal,B,H,SQ,SK,D", SHAPES)
def test_topk_values_and_index_consistency(causal, B, H, SQ, SK, D):
    q, k, v = make_qkv(B, H, SQ, SK, D, seed=SQ + SK + 3)
    _, stats = attention_with_stats(q, k, v, causal=causal, topk=4)
    _, _, _, p_ref, top_p_ref, _ = reference(q, k, v, causal, 4)
    # (a) our sorted top weights match the reference top-k values
    torch.testing.assert_close(stats.top_p, top_p_ref, atol=5 * ATOL,
                               rtol=5 * ATOL)
    # (b) index/value consistency.
    idx = stats.top_idx.clamp(min=0).long()
    p_at_idx = torch.gather(p_ref, -1, idx)
    valid = stats.top_idx >= 0
    torch.testing.assert_close(stats.top_p[valid], p_at_idx[valid],
                               atol=5 * ATOL, rtol=5 * ATOL)
    assert (stats.top_p[~valid] == 0).all()


def test_uniform_scores_analytic():
    """q = 0 -> uniform attention: H = log(n_valid), maxw = 1/n_valid."""
    B, H, S, D = 1, 1, 48, 16
    q = torch.zeros(B, H, S, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    _, st = attention_with_stats(q, k, v, causal=True)
    n_valid = torch.arange(1, S + 1, device=DEVICE, dtype=torch.float32)
    torch.testing.assert_close(st.entropy[0, 0], n_valid.log(),
                               atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(st.max_weight[0, 0], 1.0 / n_valid,
                               atol=5e-3, rtol=5e-3)


def test_single_key_row_exact():
    """seq_k = ... row 0 under causal sees exactly one key: H=0, maxw=1."""
    q, k, v = make_qkv(1, 1, 33, 33, 16, seed=9)
    _, st = attention_with_stats(q, k, v, causal=True)
    assert st.entropy[0, 0, 0].item() == 0.0
    assert abs(st.max_weight[0, 0, 0].item() - 1.0) < 1e-6
    assert st.top_idx[0, 0, 0, 0].item() == 0        # only key 0 visible
    assert st.top_idx[0, 0, 0, 1].item() == -1       # rest invalid


def test_one_hot_row():
    """One dominant key: H ~ 0, maxw ~ 1, top-1 index = that key."""
    B, H, S, D = 1, 1, 8, 16
    q = torch.zeros(B, H, S, D); q[..., 0] = 1.0
    k = torch.randn(B, H, S, D) * 0.01
    k[0, 0, 5, 0] = 50.0                              # key 5 dominates
    v = torch.randn(B, H, S, D)
    q, k, v = (t.to(DEVICE, DTYPE) for t in (q, k, v))
    _, st = attention_with_stats(q, k, v, causal=False)
    assert (st.top_idx[0, 0, :, 0] == 5).all()
    assert (st.max_weight[0, 0] > 0.99).all()
    assert (st.entropy[0, 0] < 0.05).all()


def test_padded_rows_stats_zeroed():
    """Rows beyond seq_q must never poison stats (kernel masks stores), and stats
    buffers for real rows are finite."""
    q, k, v = make_qkv(1, 2, 77, 53, 32, seed=13)     # 77 -> 51 padded rows
    _, st = attention_with_stats(q, k, v, causal=False)
    assert torch.isfinite(st.entropy).all()
    assert torch.isfinite(st.max_weight).all()
    assert st.entropy.shape == (1, 2, 77)
    assert (st.max_weight > 0).all()                  # every real row attends


def test_no_stats_control_arm_matches():
    """collect_stats=False (the benchmark control) must return identical out."""
    q, k, v = make_qkv(1, 2, 64, 64, 16, seed=17)
    out1, st = attention_with_stats(q, k, v, causal=False, collect_stats=True)
    out0, none = attention_with_stats(q, k, v, causal=False,
                                      collect_stats=False)
    assert none is None and st is not None
    torch.testing.assert_close(out1, out0, atol=0, rtol=0)


@pytest.mark.skipif(DEVICE != "cuda", reason="autotuned attention kernel is CUDA-only")
def test_out_matches_autotuned_attention():
    from kernels.attention import attention
    q, k, v = make_qkv(2, 4, 128, 128, 64, seed=21)
    out_stats, _ = attention_with_stats(q, k, v, causal=True)
    out_base = attention(q, k, v, causal=True)
    torch.testing.assert_close(out_stats, out_base, atol=2e-2, rtol=2e-2)


def test_rejects_bad_inputs():
    q, k, v = make_qkv(1, 1, 8, 8, 16, seed=1)
    with pytest.raises(ValueError):
        attention_with_stats(q[:, :, :, :8], k, v)         # D not in [16,128]
    with pytest.raises(ValueError):
        attention_with_stats(q, k[:, :, :4], v)            # k/v mismatch
    with pytest.raises(ValueError):
        attention_with_stats(q, k, v, topk=0)              # bad topk
    with pytest.raises(ValueError):
        attention_with_stats(q, k, v, topk=9)
    with pytest.raises(ValueError):
        attention_with_stats(q, k[:, :, :7], v[:, :, :7], causal=True)
