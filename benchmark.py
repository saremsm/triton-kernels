from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from kernels import layernorm, matmul, softmax


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax as separate ops: ~5 kernels, ~8 HBM trips."""
    x_max = x.max(dim=-1, keepdim=True).values
    z = x - x_max
    num = torch.exp(z)
    den = num.sum(dim=-1, keepdim=True)
    return num / den


def naive_layernorm(x, w, b, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm as separate ops: ~6 kernels over the same tensor."""
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    return (x - mu) * torch.rsqrt(var + eps) * w + b


def _gbps(nbytes_moved: int, ms: float) -> float:
    return nbytes_moved / (ms * 1e-3) / 1e9


def bench_softmax(rows: int = 4096) -> None:
    print(f"\n## Fused softmax ({rows} rows, fp16)\n")
    print("| n_cols | naive (ms) | torch (ms) | triton (ms) | vs naive | vs torch | triton GB/s |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n_cols in (512, 1024, 2048, 4096, 8192):
        x = torch.randn(rows, n_cols, device="cuda", dtype=torch.float16)
        t_naive = triton.testing.do_bench(lambda: naive_softmax(x))
        t_torch = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
        t_triton = triton.testing.do_bench(lambda: softmax(x))
        moved = 2 * x.numel() * x.element_size()  # one read + one write
        print(
            f"| {n_cols} | {t_naive:.3f} | {t_torch:.3f} | {t_triton:.3f} "
            f"| {t_naive / t_triton:.2f}x | {t_torch / t_triton:.2f}x "
            f"| {_gbps(moved, t_triton):.0f} |"
        )


def bench_layernorm(rows: int = 4096) -> None:
    print(f"\n## Fused LayerNorm ({rows} rows, fp16)\n")
    print("| n_cols | naive (ms) | torch (ms) | triton (ms) | vs naive | vs torch | triton GB/s |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n_cols in (512, 1024, 2048, 4096, 8192):
        x = torch.randn(rows, n_cols, device="cuda", dtype=torch.float16)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float16)
        b = torch.randn(n_cols, device="cuda", dtype=torch.float16)
        t_naive = triton.testing.do_bench(lambda: naive_layernorm(x, w, b))
        t_torch = triton.testing.do_bench(lambda: F.layer_norm(x, (n_cols,), w, b))
        t_triton = triton.testing.do_bench(lambda: layernorm(x, w, b))
        moved = 2 * x.numel() * x.element_size()
        print(
            f"| {n_cols} | {t_naive:.3f} | {t_torch:.3f} | {t_triton:.3f} "
            f"| {t_naive / t_triton:.2f}x | {t_torch / t_triton:.2f}x "
            f"| {_gbps(moved, t_triton):.0f} |"
        )


def bench_matmul() -> None:
    print("\n## Tiled matmul (fp16, fp32 accumulation)\n")
    print("| M x N x K | cuBLAS TFLOPS | triton TFLOPS | % of cuBLAS |")
    print("|---:|---:|---:|---:|")
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 4096, 2048),
    ]
    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        matmul(a, b)  # trigger autotuning outside the timed region
        t_cublas = triton.testing.do_bench(lambda: torch.matmul(a, b))
        t_triton = triton.testing.do_bench(lambda: matmul(a, b))
        flops = 2 * M * N * K
        tf_cublas = flops / (t_cublas * 1e-3) / 1e12
        tf_triton = flops / (t_triton * 1e-3) / 1e12
        print(
            f"| {M}x{N}x{K} | {tf_cublas:.1f} | {tf_triton:.1f} "
            f"| {100 * tf_triton / tf_cublas:.0f}% |"
        )


def naive_attention(q, k, v, causal: bool) -> torch.Tensor:
    """Materialized attention: the O(N^2)-memory baseline this kernel exists to
    avoid."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    s = (q @ k.transpose(-2, -1)) * scale
    if causal:
        m, n = s.shape[-2], s.shape[-1]
        mask = torch.triu(torch.ones(m, n, device=s.device, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def bench_attention(B: int = 4, H: int = 8, D: int = 64) -> None:
    from kernels import attention

    for causal in (False, True):
        label = "causal" if causal else "non-causal"
        print(f"\n## Fused attention ({label}; B={B}, H={H}, D={D}, fp16)\n")
        print("| seq | naive (ms) | SDPA (ms) | triton (ms) | vs naive | vs SDPA | triton TFLOPS |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for seq in (512, 1024, 2048, 4096):
            q = torch.randn(B, H, seq, D, device="cuda", dtype=torch.float16)
            k = torch.randn(B, H, seq, D, device="cuda", dtype=torch.float16)
            v = torch.randn(B, H, seq, D, device="cuda", dtype=torch.float16)
            attention(q, k, v, causal=causal)  # autotune outside timing

            try:
                t_naive = triton.testing.do_bench(lambda: naive_attention(q, k, v, causal))
                naive_col = f"{t_naive:.3f}"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                t_naive = None
                naive_col = "OOM"
            t_sdpa = triton.testing.do_bench(
                lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=causal
                )
            )
            t_triton = triton.testing.do_bench(lambda: attention(q, k, v, causal=causal))

            flops = 4 * B * H * seq * seq * D * (0.5 if causal else 1.0)
            tf = flops / (t_triton * 1e-3) / 1e12
            vs_naive = f"{t_naive / t_triton:.2f}x" if t_naive else "-"
            print(
                f"| {seq} | {naive_col} | {t_sdpa:.3f} | {t_triton:.3f} "
                f"| {vs_naive} | {t_sdpa / t_triton:.2f}x | {tf:.1f} |"
            )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("benchmark requires a CUDA GPU")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch {torch.__version__}, triton {triton.__version__}")
    bench_softmax()
    bench_layernorm()
    bench_matmul()
    bench_attention()
    print(
        "\nNotes: 'vs naive' is the HBM-round-trip-elimination comparison; "
        "'vs torch'/'vs SDPA' compares against PyTorch's already-fused "
        "implementations, where parity is the realistic target. On Ampere+ "
        "GPUs SDPA dispatches to FlashAttention/cuDNN, so the attention "
        "'vs SDPA' column competes with the real thing - read it "
        "accordingly. GB/s counts one read + one write of the tensor (the "
        "minimum traffic), so it is a lower bound on achieved bandwidth."
    )


if __name__ == "__main__":
    main()
