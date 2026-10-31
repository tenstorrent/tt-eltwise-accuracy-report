# Findings

**Status: measured.** Every row below is a published result that changed.

_978f93b56e3 → d80d9482a41_

## Regressions (3)

A scored metric got worse.

| Variant | Moved |
|---|---|
| `bh/fp32/floor_div/default` | mean_ulp 1.56e+03 → 1.62e+03, ulp_clipped 13703 → 13600 |
| `bh/fp32/logit/default` | rounded_frac 0.941 → 0.94 |
| `wh/fp32/logit/default` | rounded_frac 0.94 → 0.939 |

## Improvements (21)

A scored metric got better.

| Variant | Moved |
|---|---|
| `bh/bf16/polygamma/k2` | defects 75 → 1 |
| `bh/bf16/polygamma/k4` | mean_ulp 1.66e+07 → 1.65e+07, defects 142 → 0, usable_to 3.5 → 3.48 |
| `bh/bf16/polygamma_bw/n1` | defects 75 → 1 |
| `bh/fp32/div/default` | max_ulp 2.44 → 0.751, mean_ulp 1.44 → 0.662, rounded_frac 0.894 → 0.937 |
| `bh/fp32/div_no_nan/default` | max_ulp 9.24e+04 → 0.751, mean_ulp 4.51e+04 → 0.662, ulp_clipped 35970 → 0, rounded_frac 0.702 → 0.937 |
| `bh/fp32/divide/default` | max_ulp 2.44 → 0.751, mean_ulp 1.44 → 0.662, rounded_frac 0.894 → 0.937 |
| `bh/fp32/divide_/default` | max_ulp 2.44 → 0.751, mean_ulp 1.44 → 0.662, rounded_frac 0.894 → 0.937 |
| `bh/fp32/polygamma/k2` | mean_ulp 4.42e+26 → 4.4e+26, defects 75 → 0, rounded_frac 0.183 → 0.18 |
| `bh/fp32/polygamma/k4` | mean_ulp 6.51e+21 → 6.43e+21, defects 141 → 0, rounded_frac 0.138 → 0.141 |
| `bh/fp32/polygamma_bw/n1` | mean_ulp 4.42e+26 → 4.4e+26, defects 75 → 0, rounded_frac 0.183 → 0.18 |
| `wh/bf16/polygamma/k2` | mean_ulp 6.81e+05 → 6.79e+05, defects 75 → 1 |
| `wh/bf16/polygamma/k4` | mean_ulp 1.09e+07 → 1.08e+07, defects 142 → 0 |
| `wh/bf16/polygamma_bw/n1` | mean_ulp 6.81e+05 → 6.79e+05, defects 75 → 1 |
| `wh/fp32/div/default` | max_ulp 1.78 → 0.9, mean_ulp 1.22 → 0.789, rounded_frac 0.862 → 0.92 |
| `wh/fp32/div_no_nan/default` | max_ulp 1.79 → 0.9, mean_ulp 1.52 → 0.789, rounded_frac 0.714 → 0.92 |
| `wh/fp32/divide/default` | max_ulp 1.78 → 0.9, mean_ulp 1.22 → 0.789, rounded_frac 0.862 → 0.92 |
| `wh/fp32/divide_/default` | max_ulp 1.78 → 0.9, mean_ulp 1.22 → 0.789, rounded_frac 0.862 → 0.92 |
| `wh/fp32/floor_div/default` | mean_ulp 1.77e+03 → 1.7e+03, ulp_clipped 13707 → 13618, rounded_frac 0.955 → 0.968 |
| `wh/fp32/polygamma/k2` | mean_ulp 4.42e+26 → 4.4e+26, defects 75 → 0, rounded_frac 0.199 → 0.191 |
| `wh/fp32/polygamma/k4` | mean_ulp 6.51e+21 → 6.43e+21, defects 141 → 0, rounded_frac 0.191 → 0.194 |
| `wh/fp32/polygamma_bw/n1` | mean_ulp 4.42e+26 → 4.4e+26, defects 75 → 0, rounded_frac 0.199 → 0.191 |

## Expected (11)

No scored metric got worse on comparable measurements. Either `usable_to` moved, because a different kernel puts the 2 ULP boundary on a neighbouring group; or `n_inputs` moved, which means the swept domain changed and the two runs measure different populations; or a metric is reported here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/digamma_bw/default` | usable_to 0.996 → 1 |
| `bh/bf16/expm1_bw/default` | usable_to 33.5 → 33.2 |
| `bh/bf16/hardswish_bw/default` | usable_to 1.13 → 1.12 |
| `bh/bf16/lgamma/default` | usable_to 0.412 → 0.414 |
| `bh/bf16/log_sigmoid/default` | usable_to 0.428 → 0.426 |
| `bh/bf16/multigammaln/default` | usable_to 5.59e-17 → 5.55e-17 |
| `bh/bf16/multigammaln_bw/default` | usable_to 5.59e-17 → 5.55e-17 |
| `bh/bf16/sigmoid_bw/default` | usable_to 12.5 → 12.4 |
| `bh/bf16/silu_bw/default` | usable_to 0.863 → 0.867 |
| `bh/fp32/multigammaln/default` | usable_to 5.55e-17 → 5.59e-17 |
| `bh/fp32/silu_bw/default` | usable_to 2.98e-07 → 2.96e-07 |

## No longer measured (32)

Present in the baseline, absent now.

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
