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

_Not yet measured.

