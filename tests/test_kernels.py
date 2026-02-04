from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

torch.manual_seed(0)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

TOL = {
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}

SOFTMAX_SHAPES = [(8, 128), (1823, 781), (4096, 4096), (1, 1), (5, 2049), (64, 1)]


@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_softmax_matches_torch(shape, dtype):
    from kernels import softmax

    x = torch.randn(*shape, device="cuda", dtype=dtype)
    ref = torch.softmax(x.float(), dim=-1).to(dtype)
    out = softmax(x)
    assert out.shape == x.shape and out.dtype == dtype
    torch.testing.assert_close(out, ref, **TOL[dtype])


def test_softmax_noncontiguous_input():
    from kernels import softmax

    x = torch.randn(512, 384, device="cuda", dtype=torch.float16).t()  # view
    assert not x.is_contiguous()
    ref = torch.softmax(x.float(), dim=-1).to(x.dtype)
    torch.testing.assert_close(softmax(x), ref, **TOL[torch.float16])


def test_softmax_3d_input():
    from kernels import softmax

    x = torch.randn(4, 37, 129, device="cuda", dtype=torch.float16)
    ref = torch.softmax(x.float(), dim=-1).to(x.dtype)
    torch.testing.assert_close(softmax(x), ref, **TOL[torch.float16])


def test_softmax_oversize_row_falls_back():
    from kernels import softmax
    from kernels.softmax import MAX_BLOCK

    x = torch.randn(2, MAX_BLOCK + 1, device="cuda", dtype=torch.float16)
    with pytest.warns(UserWarning, match="falling back"):
        out = softmax(x)
    ref = torch.softmax(x.float(), dim=-1).to(x.dtype)
    torch.testing.assert_close(out, ref, **TOL[torch.float16])
