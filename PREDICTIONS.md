# Predictions and outcomes

Pre-run estimates in this repo are recorded in commit messages, next to the
session that measured them. This file collects them so the misses are as easy
to find as the wins. The measured tables live in the README; nothing here is a
result, only what was expected before the result and where each is recorded.

## Marginal cost of fused attention stats

- Estimate, before the run-3 numbers came back: streaming per-row stats out of
  the attention kernel would cost **at most roughly 15%** over the same kernel
  with stats off.
- Measured: **+153% to +346%** (README, attention-stats tables). A miss.
- Recorded in commit `1c8772d` ("Record A10 run 3: marginal-cost miss and
  diagnosis."), together with the diagnosis: the first explanation (top-k
  register pressure) died against the profiler; the one that held is that
  compute-only work is not cheap when it lengthens the inner loop's dependency
  chain. The vs-materialized claim (13.6-34.5x) survives on the O(N) memory
  half of the argument.

## Determinism delta of the atomic SAE backward

- Expectation, set when the demo was written: deterministic run-to-run delta
  **0.0**, stated in advance as "not observed in 8 runs" rather than assumed.
- Measured: atomic max run-to-run |delta| = 6.25e-2 (exactly 2 fp16 ULPs at the
  contested row's magnitude); deterministic max |delta| = **0.000e+00** across
  8 runs on real parallel hardware.
- Recorded in commit `f8cd33b` ("Add determinism-tax benchmark and delta
  demo."). The tax estimate in the same session - deterministic predicted
  1.5-4x slower - also missed, in the good direction: measured 0.42-0.48x
  (faster), and the default flipped on the measurement.
