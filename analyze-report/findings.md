# Findings

**Status: measured.** Every row below is a published result that changed.

_5b8d933, f6deef232f7, fbf7d27db93 → 9f9cd4fd590, fbf7d27db93_

## Regressions (5)

A scored metric got worse.

| Variant | Moved |
|---|---|
| `wh/bf16/mac/default` | max_ulp 64 → 250, mean_ulp 26.4 → 38.5 |
| `wh/fp32/lgamma/default` | ulp_clipped 313 → 314 |
| `wh/fp32/mac/default` | max_ulp 4.22e+06 → 8.33e+06, mean_ulp 7.31e+05 → 1.45e+06, ulp_clipped 32158 → 32408 |
| `wh/fp32/multigammaln/default` | ulp_clipped 2987 → 2989, usable_to 5.55e-17 → 5.59e-17 |
| `wh/fp32/tanhshrink/default` | mean_ulp 1.23e+05 → 9.11e+04, ulp_clipped 1798 → 1984 |

## Expected (13)

No scored metric got worse on comparable measurements. Either `usable_to` moved, because a different kernel puts the 2 ULP boundary on a neighbouring group; or `n_inputs` moved, which means the swept domain changed and the two runs measure different populations; or a metric is reported here for the first time.

| Variant | Moved |
|---|---|
| `wh/bf16/digamma_bw/default` | usable_to 0.996 → 1 |
| `wh/bf16/expm1_bw/default` | usable_to 2.08 → 2.09 |
| `wh/bf16/gelu_bw/default` | usable_to 8.38 → 8.44 |
| `wh/bf16/hardswish_bw/default` | usable_to 1.13 → 1.12 |
| `wh/bf16/multigammaln/default` | usable_to 5.59e-17 → 5.55e-17 |
| `wh/bf16/multigammaln_bw/default` | usable_to 5.59e-17 → 5.55e-17 |
| `wh/bf16/polygamma/k1` | mean_ulp 1.32e+04 → 1.43e+05, defects 263 → 7, n_inputs 64769 → 49922 |
| `wh/bf16/polygamma/k2` | n_inputs 64769 → 49922 |
| `wh/bf16/polygamma/k4` | n_inputs 64769 → 49922 |
| `wh/bf16/selu_bw/default` | usable_to 0.582 → 0.586 |
| `wh/bf16/sigmoid_bw/default` | usable_to 1.86 → 1.85 |
| `wh/bf16/silu_bw/default` | usable_to 0.887 → 0.883 |
| `wh/fp32/polygamma/k1` | mean_ulp 3.25e+29 → 4.68e+29, ulp_clipped 15848 → 1000, defects 255 → 0, n_inputs 64769 → 49922 |

## New coverage (28)

Measured here for the first time.

| Variant | Moved |
|---|---|
| `wh/bf16/hardswish/default` | — |
| `wh/bf16/isfinite/default` | — |
| `wh/bf16/isinf/default` | — |
| `wh/bf16/isnan/default` | — |
| `wh/bf16/isneginf/default` | — |
| `wh/bf16/isposinf/default` | — |
| `wh/fp32/clamp/min0.0_max0.0` | — |
| `wh/fp32/clamp/min1.0_max-1.0` | — |
| `wh/fp32/clip/min0.0_max0.0` | — |
| `wh/fp32/clip/min1.0_max-1.0` | — |
| `wh/fp32/hardswish/default` | — |
| `wh/fp32/heaviside/value0.0` | — |
| `wh/fp32/heaviside/value1.0` | — |
| `wh/fp32/isfinite/default` | — |
| `wh/fp32/isinf/default` | — |
| `wh/fp32/isnan/default` | — |
| `wh/fp32/isneginf/default` | — |
| `wh/fp32/isposinf/default` | — |
| `wh/fp32/leaky_relu/negative_slope0.0` | — |
| `wh/fp32/leaky_relu/negative_slope1.0` | — |
| `wh/fp32/polygamma/k2` | — |
| `wh/fp32/polygamma/k4` | — |
| `wh/fp32/relu_max/upper_limit0.0` | — |
| `wh/fp32/relu_max/upper_limit6.0` | — |
| `wh/fp32/relu_min/lower_limit0.0` | — |
| `wh/fp32/rpow/exponent0.5` | — |
| `wh/fp32/rpow/exponent1.0` | — |
| `wh/fp32/threshold/threshold0.0_value1.0` | — |

## No longer measured (4)

Present in the baseline, absent now.

| Variant | Moved |
|---|---|
| `bh/bf16/i1/default` | — |
| `bh/fp32/i1/default` | — |
| `wh/bf16/i1/default` | — |
| `wh/fp32/i1/default` | — |
