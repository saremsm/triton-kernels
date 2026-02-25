"""Sparse SAE decode benchmark: gather vs. dense tensor-core matmul. At production
SAE widths (F ~ 131072, W_dec ~ 200 MB) it would not hold; the crossover is
regime-dependent and this table only claims the regime it measures."""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import triton

from kernels.sae_decode import sae_decode, sparsify

D_MODEL, N_FEATURES = 768, 3072
SWEEP_L0 = (4, 8, 16, 27, 32, 64, 128, 256, 512, 1024)


def make_sparse_h(n, n_features, l0, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.arange(n).repeat_interleave(l0)
    cols = torch.stack([torch.randperm(n_features, generator=g)[:l0]
                        for _ in range(n)]).reshape(-1)
    h = torch.zeros(n, n_features, dtype=dtype)
    h[rows, cols] = torch.rand(n * l0, generator=g) + 0.1
    return h.to(device)


def naive_sparse(idx, val, W, b):
    gathered = F.embedding(idx.long(), W)          # (N, K, D) in HBM
    return (gathered * val.unsqueeze(-1)).sum(dim=1) + b


def _safe_bench(fn):
    """do_bench, but returns None on OOM instead of killing the process."""
    try:
        t = triton.testing.do_bench(fn)
        torch.cuda.synchronize()
        return t
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def _bench_row(h, W, b, label):
    idx, val = sparsify(h)
    K = idx.shape[1]
    sae_decode(idx, val, W, b)                     # JIT compile outside timing
    t_dense = triton.testing.do_bench(lambda: h @ W + b)
    t_naive = _safe_bench(lambda: naive_sparse(idx, val, W, b))
    t_triton = triton.testing.do_bench(lambda: sae_decode(idx, val, W, b))
    traffic = h.shape[0] * K * (D_MODEL * W.element_size() + 6)
    gbps = traffic / (t_triton * 1e-3) / 1e9
    if t_naive is None:
        naive_ms, vs_naive = "OOM", "n/a"
    else:
        naive_ms, vs_naive = f"{t_naive:.3f}", f"{t_naive / t_triton:.2f}x"
    print(f"| {label} | {K} | {t_dense:.3f} | {naive_ms} | {t_triton:.3f} "
          f"| {t_dense / t_triton:.2f}x | {vs_naive} | {gbps:.0f} |")
    return t_dense / t_triton


def bench_sweep(n_tokens: int, dtype=torch.float16) -> None:
    W = torch.randn(N_FEATURES, D_MODEL, device="cuda", dtype=dtype)
    W = W / W.norm(dim=1, keepdim=True)
    b = torch.randn(D_MODEL, device="cuda", dtype=dtype)
    print(f"\n## Sparse SAE decode vs dense ({n_tokens} tokens, "
          f"F={N_FEATURES}, D={D_MODEL}, {str(dtype).split('.')[-1]})\n")
    print("| L0 | K_pad | dense (ms) | naive gather (ms) | triton (ms) "
          "| vs dense | vs naive | gather GB/s* |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    speedups = {}
    for l0 in SWEEP_L0:
        h = make_sparse_h(n_tokens, N_FEATURES, l0, "cuda", dtype, seed=l0)
        speedups[l0] = _bench_row(h, W, b, str(l0))
    wins = [l0 for l0, s in speedups.items() if s > 1.0]
    if wins and len(wins) < len(speedups):
        print(f"\nCrossover: sparse beats dense up to L0 = {max(wins)}; "
              f"dense wins from L0 = {min(l0 for l0 in speedups if l0 not in wins)}.")
    elif wins:
        print("\nSparse beats dense across the entire swept range.")
    else:
        print("\nDense wins across the entire swept range - a finding to "
              "investigate, not to hide (check L2 residency assumptions).")


def bench_checkpoint(path: str, acts: str | None, l0: int,
                     n_tokens: int) -> None:
    ckpt = torch.load(path, map_location="cuda", weights_only=True)
    W = ckpt["sae_state_dict"]["W_dec"]
    b = ckpt["sae_state_dict"]["b_dec"]
    cfg = ckpt["config"]
    assert W.shape == (cfg["n_features"], cfg["d_model"]), "layout drift"

    if acts:
        h = torch.load(acts, map_location="cuda", weights_only=True).float()
        src = f"real activations ({acts})"
    else:
        h = make_sparse_h(n_tokens, cfg["n_features"], l0, "cuda",
                          torch.float32, seed=42)
        src = f"synthetic at L0={l0}"
    mean_l0 = (h > 0).float().sum(-1).mean().item()

    # fp16 on both sides: the dense opponent gets its tensor cores.
    W16, b16, h16 = W.half(), b.half(), h.half().clamp(min=0)
    print(f"\n## Checkpoint decode: layer {ckpt.get('layer', '?')}, "
          f"{h.shape[0]} tokens, {src}, measured mean L0 = {mean_l0:.1f}\n")
    print("| source | K_pad | dense (ms) | naive gather (ms) | triton (ms) "
          "| vs dense | vs naive | gather GB/s* |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    speedup = _bench_row(h16, W16, b16, "ckpt fp16")
    print(f"\nSAE decode at L0={mean_l0:.0f}: "
          f"{speedup:.2f}x vs dense tensor-core matmul.")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("benchmark requires a CUDA GPU")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="sae-gpt2-small checkpoint .pt")
    ap.add_argument("--acts", help="dumped post-ReLU activations .pt (N, F)")
    ap.add_argument("--l0", type=int, default=27)
    ap.add_argument("--n-tokens", type=int, default=8192)
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch {torch.__version__}, triton {triton.__version__}")
    try:
        bench_sweep(args.n_tokens)
    except Exception as e:                      # sweep is diagnostic; the checkpoint-L0 number is what matters
        print(f"\n[sweep aborted: {type(e).__name__}: {e}]")
        torch.cuda.empty_cache()
    if args.checkpoint:
        bench_checkpoint(args.checkpoint, args.acts, args.l0, args.n_tokens)
    print("\nNotes: 'vs dense' is the honest fight - the dense side keeps "
          "its tensor cores. 'vs naive' is the HBM-round-trip baseline. "
          "*gather GB/s uses the worst-case no-reuse traffic model; above-"
          "HBM figures indicate L2 residency (W_dec fits in L2 at these "
          "dims), not error. See module docstring for the regime caveat.")


if __name__ == "__main__":
    main()
