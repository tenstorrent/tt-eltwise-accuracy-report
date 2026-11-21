# Findings

**Status: measured.** Every row below is a published result that changed.

_d80d9482a41 → bfc3f70fd49_

## Improvements (9)

A scored metric got better.

| Variant | Moved |
|---|---|
| `bh/bf16/atan2/default` | max_ulp 200 → 1.91, mean_ulp 3.56 → 1.7 |
| `bh/fp32/atan2/default` | max_ulp 1.32e+07 → 3.67, mean_ulp 1.26e+05 → 2.39, ulp_clipped 3594 → 0, rounded_frac 0.857 → 0.858 |
| `bh/fp32/logaddexp2_/default` | mean_ulp 1.23e+05 → 1.22e+05, ulp_clipped 5609 → 4192, rounded_frac 0.959 → 0.963 |
| `wh/bf16/asin/default` | defects 2 → 0, usable_to — → 1 |
| `wh/bf16/atan/default` | max_ulp 0.527 → 0.5, mean_ulp 0.513 → 0, defects 2 → 0, usable_to — → 3.39e+38 |
| `wh/bf16/atan2/default` | max_ulp 200 → 0.644, mean_ulp 2.51 → 0.615 |
| `wh/fp32/atan2/default` | max_ulp 1.32e+07 → 3.35, mean_ulp 1.26e+05 → 2.26, ulp_clipped 3594 → 0 |
| `wh/fp32/logaddexp2/default` | mean_ulp 1.23e+05 → 1.22e+05, ulp_clipped 5609 → 4192, rounded_frac 0.959 → 0.963 |
| `wh/fp32/logaddexp2_/default` | mean_ulp 1.23e+05 → 1.22e+05, ulp_clipped 5609 → 4192, rounded_frac 0.959 → 0.963 |

## Expected (7)

No scored metric got worse on comparable measurements. Either `usable_to` moved, because a different kernel puts the 2 ULP boundary on a neighbouring group; or `n_inputs` moved, which means the swept domain changed and the two runs measure different populations; or a metric is reported here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/digamma_bw/default` | usable_to 1 → 0.996 |
| `bh/bf16/expm1_bw/default` | usable_to 33.2 → 33.5 |
| `bh/bf16/hardswish_bw/default` | usable_to 1.12 → 1.13 |
| `bh/bf16/multigammaln_bw/default` | usable_to 5.55e-17 → 5.59e-17 |
| `bh/bf16/sigmoid_bw/default` | usable_to 12.4 → 12.5 |
| `bh/bf16/silu_bw/default` | usable_to 0.867 → 0.863 |
| `bh/fp32/silu_bw/default` | usable_to 2.96e-07 → 2.98e-07 |

## New coverage (32)

Measured here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/bias_gelu/default` | — |
| `bh/bf16/eq/default` | — |
| `bh/bf16/ge/default` | — |
| `bh/bf16/gt/default` | — |
| `bh/bf16/ldexp/default` | — |
| `bh/bf16/le/default` | — |
| `bh/bf16/logaddexp/default` | — |
| `bh/bf16/logaddexp2/default` | — |
| `bh/bf16/logical_and/default` | — |
| `bh/bf16/logical_or/default` | — |
| `bh/bf16/lt/default` | — |
| `bh/bf16/maximum/default` | — |
| `bh/bf16/minimum/default` | — |
| `bh/bf16/ne/default` | — |
| `bh/bf16/remainder/default` | — |
| `bh/bf16/xlogy/default` | — |
| `bh/fp32/bias_gelu/default` | — |
| `bh/fp32/eq/default` | — |
| `bh/fp32/ge/default` | — |
| `bh/fp32/gt/default` | — |
| `bh/fp32/ldexp/default` | — |
| `bh/fp32/le/default` | — |
| `bh/fp32/logaddexp/default` | — |
| `bh/fp32/logaddexp2/default` | — |
| `bh/fp32/logical_and/default` | — |
| `bh/fp32/logical_or/default` | — |
| `bh/fp32/lt/default` | — |
| `bh/fp32/maximum/default` | — |
| `bh/fp32/minimum/default` | — |
| `bh/fp32/ne/default` | — |
| `bh/fp32/remainder/default` | — |
| `bh/fp32/xlogy/default` | — |
