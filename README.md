# triton-kernels

Four Triton kernels - fused softmax, fused LayerNorm, an autotuned tiled
matmul, and a **FlashAttention-style fused attention** (forward) - with
correctness tests against independent references and a benchmark harness
that compares against both baselines.

## running

```
pip install -r requirements.txt
pytest -v                # requires a CUDA GPU
python benchmark.py      # canonical kernels: prints markdown tables
python benchmark_sae.py  # sparse decode crossover sweep (+ --checkpoint)
```

The sparse-decode suite also runs without a GPU via Triton's interpreter
(`TRITON_INTERPRET=1 pytest tests/test_sae_decode.py`) - CPU logic
verification for CI; performance claims still require the GPU run.

### SAE decode numbers

Measured on **1x A10 (24 GB), torch 2.10.0+cu126, triton 3.6.0**, 8192
tokens, F=3072, D=768, fp16. Sweep of gather vs. dense tensor-core matmul
across sparsity:

| L0 | K_pad | dense (ms) | naive gather (ms) | triton (ms) | vs dense | vs naive | gather GB/s* |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 0.573 | 1.010 | 0.067 | **8.55×** | 15.08× | 1509 |
| 8 | 8 | 0.576 | 1.055 | 0.085 | 6.81× | 12.48× | 1195 |
| 16 | 16 | 0.575 | 2.009 | 0.133 | 4.31× | 15.07× | 1516 |
| **27** | 32 | 0.579 | 3.862 | 0.215 | **2.70×** | 17.99× | 1883 |
| 32 | 32 | 0.575 | 3.919 | 0.241 | 2.38× | 16.23× | 1674 |
| 64 | 64 | 0.579 | 7.738 | 0.488 | 1.19× | 15.86× | 1657 |
| 128 | 128 | 0.576 | 15.388 | 0.989 | 0.58× | 15.56× | 1635 |
| 256 | 256 | 0.581 | 30.718 | 2.027 | 0.29× | 15.15× | 1595 |
| 512 | 512 | 0.577 | 61.592 | 4.102 | 0.14× | 15.01× | 1577 |
| 1024 | 1024 | 0.572 | OOM | 8.332 | 0.07× | n/a | 1552 |

**At the checkpoint's measured L0 = 27, sparse decode is 2.61×
faster than the dense tensor-core matmul** (measured in `--checkpoint`
mode on the real layer-8 weights; the L0=27 sweep row above reads 2.70×  - 
same config, independent timing, normal run-to-run jitter). The crossover
sits between L0 = 64 (1.19×, sparse still ahead) and L0 = 128 (0.58×, dense
ahead) - so at trained SAE sparsity the gather wins comfortably, and the
point where dense reclaims the lead is measured. The naive
gather column hits `OOM` at L0 = 1024 (its (8192, 1024, 768) fp16
intermediate is 12 GB): a property of that baseline, not the kernel, which
handled the row in 8.3 ms. Gather bandwidth of 1200-1900 GB/s against the
A10's ~600 GB/s HBM peak confirms the mechanism: W_dec (4.5 MB fp16) is
L2-resident, so the "traffic" is served from L2, not HBM. The `vs naive`
column (15-18× throughout) is the round-trip-elimination story against the
torch-ops gather; it holds flat across the whole range because the naive
path is HBM-bound at every density.

Regime caveat: this crossover is for
GPT-2-small dims where W_dec fits in L2. At production SAE widths
(F ≈ 131072, W_dec ≈ 200 MB) the gather goes to HBM and the crossover
moves left; the kernel and benchmark are unchanged, but the winning
sparsity regime narrows. The number above holds for the checkpoint it
names and should not be extrapolated to larger SAEs without re-measuring.

## The canonical four

Fused softmax, fused LayerNorm, tiled matmul, fused attention - the
teaching set, kept because the discipline around it (adversarial tests,
two-baseline benchmarks, honest limitations) is the point, and because
The SAE kernels reuse that discipline wholesale.

### canonical-kernel numbers

Measured on **1x A10 (24 GB), torch 2.10.0+cu126, triton 3.6.0**, fp16.

**Fused softmax** (4096 rows): 3.4-4.3× vs naive multi-op, 1.1-1.3× vs
`torch.softmax`, 385-483 GB/s. **Fused LayerNorm** (4096 rows): 6.9-7.3×
vs naive, 1.1-1.5× vs `F.layer_norm`, 402-481 GB/s. Both memory-bound and
landing in the expected fusion band, edging PyTorch's own fused paths.

**Tiled matmul** (fp16, fp32 accum): 84-104% of cuBLAS across
1024³ → 8192×4096×2048 (88% / 84% / 102% / 104%) - inside the 70-90% target
at the small shapes and matching-to-beating cuBLAS at the large ones.

**Fused attention** (B=4, H=8, D=64): vs SDPA 0.87-0.93× non-causal,
0.87-1.22× causal; vs naive materialized attention 5.7-7.0× (non-causal)
and 10.6-20.0× (causal, growing with seq as the naive baseline's memory
blows up). SDPA on Ampere dispatches to FlashAttention/cuDNN, so
0.87-0.93× against the production kernel is the honest and expected result
 -  parity, not a win, and the README says so.

