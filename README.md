# triton-kernels

Four Triton kernels - fused softmax, fused LayerNorm, an autotuned tiled
matmul, and a **FlashAttention-style fused attention** (forward) - with
correctness tests against independent references and a benchmark harness
that compares against both baselines.

## running

```
pip install -r requirements.txt
pytest -v            # requires a CUDA GPU
python benchmark.py  # paste-ready markdown tables
```

## numbers

Measured on **1x A10 (24 GB), torch 2.10.0+cu126, triton 3.6.0**, fp16.

**Fused softmax** (4096 rows): 3.4-4.3× vs naive multi-op, 1.1-1.3× vs
`torch.softmax`, 385-483 GB/s. 

**Fused LayerNorm** (4096 rows): 6.9-7.3× vs naive, 1.1-1.5× vs `F.layer_norm`, 
402-481 GB/s. Both memory-bound and landing in the expected fusion band, 
edging PyTorch's own fused paths.

**Tiled matmul** (fp16, fp32 accum): 84-104% of cuBLAS across
1024³ → 8192×4096×2048 (88% / 84% / 102% / 104%) - inside the 70-90% target
at the small shapes and matching-to-beating cuBLAS at the large ones.

**Fused attention** (B=4, H=8, D=64): vs SDPA 0.87-0.93× non-causal,
0.87-1.22× causal; vs naive materialized attention 5.7-7.0× (non-causal)
and 10.6-20.0× (causal, growing with seq as the naive baseline's memory
blows up). SDPA on Ampere dispatches to FlashAttention/cuDNN, so
0.87-0.93× against the production kernel is the honest and expected result
 -  parity, not a win, and the README says so.
