from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

torch.manual_seed(0)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

ATOL, RTOL = 2e-2, 2e-2  # fp16 output vs fp32 reference, FA-style tolerance


def ref_attention(q, k, v, causal: bool, scale: float | None = None):
    """Manual fp32 reference: materialized scores, exact softmax."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    s = (q.float() @ k.float().transpose(-2, -1)) * scale
    if causal:
        m, n = s.shape[-2], s.shape[-1]
        mask = torch.triu(
            torch.ones(m, n, device=s.device, dtype=torch.bool), diagonal=1
        )
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return (p @ v.float()).to(q.dtype)


def make_qkv(B, H, M, N, D, dtype=torch.float16):
    q = torch.randn(B, H, M, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "B,H,M,D",
    [
        (2, 4, 128, 64),
        (1, 2, 515, 64),    # odd length: partial blocks on both loop axes
        (2, 2, 1000, 128),  # max head_dim, non-power-of-2 seq
        (1, 1, 64, 16),     # min head_dim
        (1, 3, 1, 64),      # single query/key
    ],
)
def test_matches_manual_reference_self_attention(causal, B, H, M, D):
    from kernels import attention

    q, k, v = make_qkv(B, H, M, M, D)
    out = attention(q, k, v, causal=causal)
    ref = ref_attention(q, k, v, causal=causal)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert not torch.isnan(out).any(), "NaNs in output (padded-row guard failed?)"
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("M,N", [(333, 257), (128, 1024), (1, 777)])
def test_cross_attention_noncausal(M, N):
    from kernels import attention

    q, k, v = make_qkv(2, 4, M, N, 64)
    out = attention(q, k, v, causal=False)
    ref = ref_attention(q, k, v, causal=False)
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa(causal):
    """Second, independent reference: PyTorch SDPA."""
    from kernels import attention

    q, k, v = make_qkv(2, 8, 512, 512, 64)
    out = attention(q, k, v, causal=causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_scale_override():
    from kernels import attention

    q, k, v = make_qkv(1, 2, 256, 256, 64)
    out = attention(q, k, v, causal=False, scale=0.3)
    ref = ref_attention(q, k, v, causal=False, scale=0.3)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_strided_views_no_copy():
    """(B, M, H, D) storage viewed as (B, H, M, D) - the layout most checkpoint
    formats use; the kernel consumes strides directly."""
    from kernels import attention

    B, H, M, D = 2, 4, 320, 64
    q_s = torch.randn(B, M, H, D, device="cuda", dtype=torch.float16)
    k_s = torch.randn(B, M, H, D, device="cuda", dtype=torch.float16)
    v_s = torch.randn(B, M, H, D, device="cuda", dtype=torch.float16)
    q, k, v = (t.transpose(1, 2) for t in (q_s, k_s, v_s))
    assert not q.is_contiguous()
    out = attention(q, k, v, causal=True)
    ref = ref_attention(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="bf16 requires Ampere+",
)
def test_bf16():
    from kernels import attention

    q, k, v = make_qkv(2, 4, 384, 384, 64, dtype=torch.bfloat16)
    out = attention(q, k, v, causal=True)
    ref = ref_attention(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


def test_rejects_bad_inputs():
    from kernels import attention

    q, k, v = make_qkv(1, 2, 64, 32, 64)
    with pytest.raises(ValueError, match="causal attention requires"):
        attention(q, k, v, causal=True)  # M != N
    q2, k2, v2 = make_qkv(1, 2, 64, 64, 48)
    with pytest.raises(ValueError, match="power of two"):
        attention(q2, k2, v2)  # head_dim 48
    q3 = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="dtypes"):
        attention(q3, q3, q3)  # fp32 unsupported
