"""Marginal-cost benchmark for streamed attention stats (attention stats). The
materialized baseline should lose by a factor growing with seq (its fp32 scores
are ~2 GB per step at seq 4096, B=4, H=8)."""

from __future__ import annotations

import math

import torch
import triton

from kernels.attention_stats import attention_with_stats

SEQS = (512, 1024, 2048, 4096)
B, H, D = 4, 8, 64
TOPK = 4


def _safe_bench(fn):
    try:
        t = triton.testing.do_bench(fn)
        torch.cuda.synchronize()
        return t
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def materialized_stats(q, k, v, causal):
    """SDPA for the output + explicit fp32 score materialization for stats."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal)
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        S = s.shape[-1]
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool,
                                     device=s.device), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
    ent = -plogp.sum(-1)
    maxw = p.max(-1).values
    top_p, top_i = torch.topk(p, TOPK, dim=-1)
    return out, ent, maxw, top_p, top_i


def _fmt(t):
    return f"{t:.3f}" if t is not None else "OOM"


def bench(causal: bool) -> None:
    mode = "causal" if causal else "non-causal"
    print(f"\n## Streamed attention stats ({mode}; B={B}, H={H}, D={D}, "
          f"fp16, TOPK={TOPK})\n")
    print("| seq | fused stats OFF (ms) | fused stats ON (ms) | **marginal** "
          "| SDPA + materialized (ms) | vs materialized |")
    print("|---:|---:|---:|---:|---:|---:|")
    for seq in SEQS:
        q = torch.randn(B, H, seq, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, seq, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, seq, D, device="cuda", dtype=torch.float16)
        # compile both variants outside timing
        attention_with_stats(q, k, v, causal=causal, collect_stats=False)
        attention_with_stats(q, k, v, causal=causal, collect_stats=True)

        t_off = _safe_bench(lambda: attention_with_stats(
            q, k, v, causal=causal, collect_stats=False))
        t_on = _safe_bench(lambda: attention_with_stats(
            q, k, v, causal=causal, collect_stats=True))
        t_mat = _safe_bench(lambda: materialized_stats(q, k, v, causal))

        marginal = (f"{100.0 * (t_on - t_off) / t_off:+.1f}%"
                    if (t_on and t_off) else "n/a")
        vs_mat = (f"{t_mat / t_on:.2f}x" if (t_mat and t_on) else "n/a")
        print(f"| {seq} | {_fmt(t_off)} | {_fmt(t_on)} | **{marginal}** "
              f"| {_fmt(t_mat)} | {vs_mat} |")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("benchmark requires a CUDA GPU")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch {torch.__version__}, triton {triton.__version__}")
    for causal in (False, True):
        try:
            bench(causal)
        except Exception as e:
            print(f"\n[{'causal' if causal else 'non-causal'} sweep aborted: "
                  f"{type(e).__name__}: {e}]")
            torch.cuda.empty_cache()
    print("\nNotes: 'marginal' is stats-on vs stats-off in the SAME kernel "
          "and config (WITH_STATS constexpr) - the isolated price of "
          "observability. 'vs materialized' is the honest alternative: "
          "production SDPA plus an fp32 score-matrix recompute for the "
          "stats. The fused stats-off column is a fixed-config kernel and "
          "is NOT the autotuned attention number; compare marginal within "
          "this table, absolute throughput in the canonical table's.")


if __name__ == "__main__":
    main()
