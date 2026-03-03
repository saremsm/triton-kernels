from __future__ import annotations

import torch
import triton

from kernels.sae_decode import sparsify
from kernels.sae_decode_backward import sae_decode_backward

D_MODEL, N_FEATURES = 768, 3072
SWEEP_L0 = (4, 8, 16, 27, 32, 64, 128, 256, 512, 1024)


def make_sparse_h(n, n_features, l0, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.arange(n).repeat_interleave(l0)
    cols = torch.stack([torch.randperm(n_features, generator=g)[:l0]
                        for _ in range(n)]).reshape(-1)
    h = torch.zeros(n, n_features, dtype=torch.float32)
    h[rows, cols] = torch.rand(n * l0, generator=g) + 0.1
    return h.to(device=device, dtype=dtype)


def _safe_bench(fn):
    try:
        t = triton.testing.do_bench(fn)
        torch.cuda.synchronize()
        return t
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def dense_autograd_backward(idx, val, W, grad_out, n_features):
    from kernels.sae_decode import densify
    h = densify(idx, val, n_features).clone().requires_grad_(True)
    W_ = W.clone().requires_grad_(True)
    (h @ W_).backward(grad_out)
    return W_.grad


def index_add_backward(idx, val, W, grad_out):
    F, D = W.shape
    gW = torch.zeros(F, D, dtype=torch.float32, device=W.device)
    feat = idx.reshape(-1).long()
    v = val.reshape(-1).float()
    contrib = v[:, None] * grad_out[
        torch.arange(idx.shape[0], device=W.device).repeat_interleave(idx.shape[1])
    ].float()
    gW.index_add_(0, feat, contrib)
    return gW


def _fmt(t):
    return f"{t:.3f}" if t is not None else "OOM"


def bench_tax(n_tokens: int, dtype=torch.float16) -> None:
    W = torch.randn(N_FEATURES, D_MODEL, device="cuda", dtype=dtype)
    W = W / W.norm(dim=1, keepdim=True)
    print(f"\n## Determinism tax: sparse decoder backward "
          f"({n_tokens} tokens, F={N_FEATURES}, D={D_MODEL}, "
          f"{str(dtype).split('.')[-1]})\n")
    print("| L0 | K_pad | atomic (ms) | deterministic (ms) | **tax** "
          "| dense autograd (ms) | index_add (ms) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for l0 in SWEEP_L0:
        h = make_sparse_h(n_tokens, N_FEATURES, l0, "cuda", dtype, seed=l0)
        idx, val = sparsify(h)
        K = idx.shape[1]
        grad_out = torch.randn(n_tokens, D_MODEL, device="cuda", dtype=dtype)
        # warmups / compile outside timing
        sae_decode_backward(grad_out, idx, val, W, backend="atomic")
        sae_decode_backward(grad_out, idx, val, W, backend="deterministic")

        t_atom = _safe_bench(
            lambda: sae_decode_backward(grad_out, idx, val, W, backend="atomic"))
        t_det = _safe_bench(
            lambda: sae_decode_backward(grad_out, idx, val, W, backend="deterministic"))
        t_dense = _safe_bench(
            lambda: dense_autograd_backward(idx, val, W, grad_out, N_FEATURES))
        t_idx = _safe_bench(lambda: index_add_backward(idx, val, W, grad_out))

        tax = (f"{t_det / t_atom:.2f}x"
               if (t_atom and t_det) else "n/a")
        print(f"| {l0} | {K} | {_fmt(t_atom)} | {_fmt(t_det)} | **{tax}** "
              f"| {_fmt(t_dense)} | {_fmt(t_idx)} |")


def demo_atomic_nondeterminism(n_tokens: int, repeats: int = 8,
                               dtype=torch.float16) -> None:
    """High-collision input: all tokens fire feature 0."""
    W = torch.randn(N_FEATURES, D_MODEL, device="cuda", dtype=dtype)
    idx = torch.zeros(n_tokens, 1, dtype=torch.int32, device="cuda")
    val = torch.rand(n_tokens, 1, device="cuda", dtype=dtype) + 0.1
    grad_out = torch.randn(n_tokens, D_MODEL, device="cuda", dtype=dtype)

    ref = sae_decode_backward(grad_out, idx, val, W, backend="atomic")[1]
    max_delta = 0.0
    for _ in range(repeats):
        gW = sae_decode_backward(grad_out, idx, val, W, backend="atomic")[1]
        max_delta = max(max_delta, (gW - ref).abs().max().item())
    det = sae_decode_backward(grad_out, idx, val, W, backend="deterministic")[1]
    det_delta = max((sae_decode_backward(grad_out, idx, val, W,
                                         backend="deterministic")[1] - det)
                    .abs().max().item() for _ in range(repeats))
    print(f"\n## Atomic non-determinism demo (high collision: "
          f"{n_tokens} tokens -> feature 0)\n")
    print(f"- atomic backend: max run-to-run |Δ| over {repeats} runs = "
          f"{max_delta:.3e}  (nonzero => observed nondeterminism)")
    print(f"- deterministic backend: max run-to-run |Δ| = {det_delta:.3e}  "
          f"(expected exactly 0)")
    print("\n(Under TRITON_INTERPRET both read 0.0 -- serial execution "
          "cannot exhibit atomic reordering. GPU-only claim.)")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("benchmark requires a CUDA GPU")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch {torch.__version__}, triton {triton.__version__}")
    try:
        bench_tax(8192)
    except Exception as e:
        print(f"\n[tax sweep aborted: {type(e).__name__}: {e}]")
        torch.cuda.empty_cache()
    try:
        demo_atomic_nondeterminism(8192)
    except Exception as e:
        print(f"\n[nondeterminism demo aborted: {type(e).__name__}: {e}]")

    print("\nNotes: tax = deterministic/atomic, sort cost included in the "
          "deterministic timing. 'dense autograd' keeps its tensor cores; "
          "'index_add' is the torch-ops sparse baseline. This is the "
          "kernel-level form of the eval-harness reproducibility/throughput "
          "trade -- determinism-by-construction has a measurable price.")


if __name__ == "__main__":
    main()
