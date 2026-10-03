# Findings

**Status: measured.** Every row below is a published result that changed.

_a3a9fb4229a → de546d3b146_

## Regressions (27)

A scored metric got worse.

| Variant | Moved |
|---|---|
| `bh/bf16/cos_bw/default` | max_ulp 7.45e+42 → 2.09e+42, mean_ulp 3.63e+39 → 2.59e+39, ulp_clipped 4648 → 4604, defects 21275 → 21281, rounded_frac 0.85 → 0.867, usable_to 2.62e+05 → 2.15e+06 |
| `bh/bf16/erfinv/default` | mean_ulp 6.4 → 6.34, rounded_frac 0.394 → 0.391 |
| `bh/bf16/expm1_bw/default` | max_ulp 125 → 34, mean_ulp 5.58 → 0.695, defects 331 → 0, rounded_frac 0.987 → 0.984, usable_to 2.08 → 33.5 |
| `bh/bf16/sin_bw/default` | max_ulp 1.94e+42 → 2.98e+41, mean_ulp 1.95e+39 → 1.39e+39, ulp_clipped 4661 → 4677, defects 21274 → 21284, rounded_frac 0.84 → 0.857, usable_to 1.31e+05 → 1.07e+06 |
| `bh/bf16/tan_bw/default` | max_ulp 3.1e+40 → 3.57e+40, mean_ulp 4.91e+37 → 1.01e+38, ulp_clipped 1282 → 1278, defects 22759 → 22782, rounded_frac 0.841 → 0.855 |
| `bh/fp32/cos_bw/default` | max_ulp 2.68e+51 → 8.59e+51, mean_ulp 2.8e+48 → 3.23e+48, ulp_clipped 9970 → 9113, defects 18190 → 18643, rounded_frac 0.835 → 0.841 |
| `bh/fp32/gelu_bw/default` | max_ulp 6.59e+08 → 1.47e+08, mean_ulp 1.24e+05 → 2.3e+04, ulp_clipped 321 → 224, rounded_frac 0.962 → 0.929, usable_to 0.0296 → 0.0298 |
| `bh/fp32/log2/default` | mean_ulp 0.511 → 0.517, rounded_frac 0.995 → 0.993 |
| `bh/fp32/logaddexp2/default` | rounded_frac 0.716 → 0.715 |
| `bh/fp32/logaddexp2_/default` | rounded_frac 0.716 → 0.715 |
| `bh/fp32/sin_bw/default` | max_ulp 1.56e+53 → 1.24e+52, mean_ulp 2.03e+49 → 3.44e+48, ulp_clipped 10226 → 9582, defects 18190 → 18430, rounded_frac 0.802 → 0.808 |
| `bh/fp32/tan_bw/default` | mean_ulp 5.11e+44 → 4.74e+44, ulp_clipped 8039 → 7389, defects 20376 → 20623, rounded_frac 0.865 → 0.87 |
| `bh/fp32/tanh_bw/default` | max_ulp 4.19e+06 → 6.59e+04, mean_ulp 9.01e+03 → 7.51e+03, ulp_clipped 32866 → 32756, rounded_frac 0.485 → 0.482 |
| `wh/bf16/cos_bw/default` | max_ulp 7.45e+42 → 2.09e+42, mean_ulp 3.63e+39 → 2.59e+39, ulp_clipped 4648 → 4604, defects 21275 → 21281, rounded_frac 0.85 → 0.867, usable_to 2.62e+05 → 2.15e+06 |
| `wh/bf16/erfinv/default` | mean_ulp 6.4 → 6.34, rounded_frac 0.394 → 0.391 |
| `wh/bf16/expm1_bw/default` | max_ulp 125 → 34, mean_ulp 5.58 → 0.695, defects 331 → 0, rounded_frac 0.987 → 0.984, usable_to 2.09 → 33.2 |
| `wh/bf16/sin_bw/default` | max_ulp 1.94e+42 → 2.98e+41, mean_ulp 1.95e+39 → 1.39e+39, ulp_clipped 4661 → 4677, defects 21274 → 21284, rounded_frac 0.84 → 0.857, usable_to 1.31e+05 → 1.07e+06 |
| `wh/bf16/tan_bw/default` | max_ulp 3.1e+40 → 3.57e+40, mean_ulp 1.95e+37 → 3.81e+37, ulp_clipped 1282 → 1278, defects 12178 → 12184, rounded_frac 0.68 → 0.691 |
| `wh/fp32/cos_bw/default` | max_ulp 2.68e+51 → 8.59e+51, mean_ulp 2.8e+48 → 3.23e+48, ulp_clipped 9970 → 9113, defects 18190 → 18643, rounded_frac 0.835 → 0.841 |
| `wh/fp32/gelu_bw/default` | max_ulp 6.59e+08 → 1.47e+08, mean_ulp 1.24e+05 → 2.3e+04, ulp_clipped 321 → 224, rounded_frac 0.962 → 0.929, usable_to 0.0296 → 0.0298 |
| `wh/fp32/log2/default` | mean_ulp 0.511 → 0.517, rounded_frac 0.995 → 0.993 |
| `wh/fp32/logaddexp2/default` | rounded_frac 0.716 → 0.715 |
| `wh/fp32/logaddexp2_/default` | rounded_frac 0.716 → 0.715 |
| `wh/fp32/sin_bw/default` | max_ulp 1.56e+53 → 1.24e+52, mean_ulp 2.03e+49 → 3.44e+48, ulp_clipped 10226 → 9582, defects 18190 → 18430, rounded_frac 0.802 → 0.808 |
| `wh/fp32/tan_bw/default` | mean_ulp 5.11e+44 → 4.74e+44, ulp_clipped 8039 → 7389, defects 20376 → 20623, rounded_frac 0.864 → 0.87 |
| `wh/fp32/tanh_bw/default` | max_ulp 4.19e+06 → 6.59e+04, mean_ulp 9.01e+03 → 7.51e+03, ulp_clipped 32866 → 32756, rounded_frac 0.485 → 0.482 |
| `wh/fp32/tanhshrink/default` | mean_ulp 9.35e+04 → 1.23e+05, ulp_clipped 1984 → 1798, rounded_frac 0.866 → 0.91 |

## Improvements (40)

A scored metric got better.

| Variant | Moved |
|---|---|
| `bh/bf16/cos/default` | max_ulp 4.1e+03 → 0.552, mean_ulp 26.5 → 0.552, ulp_clipped 6 → 0, rounded_frac 0.983 → 1, usable_to 1.31e+05 → 9.99e+05 |
| `bh/bf16/gelu_bw/default` | max_ulp 3.5 → 0.939, mean_ulp 1.37 → 0.681, rounded_frac 0.998 → 0.999, usable_to 8.38 → 3.39e+38 |
| `bh/bf16/sin/default` | max_ulp 4.1e+03 → 0.552, mean_ulp 73.5 → 0.526, ulp_clipped 6 → 0, rounded_frac 0.992 → 1, usable_to 2.62e+05 → 9.99e+05 |
| `bh/bf16/tan/default` | max_ulp 4.1e+03 → 0.551, mean_ulp 49.1 → 0.509, ulp_clipped 6 → 0, rounded_frac 0.985 → 0.999, usable_to 1.31e+05 → 9.99e+05 |
| `bh/bf16/tanh_bw/default` | max_ulp 55.5 → 1.16, mean_ulp 1.93 → 0.525, rounded_frac 0.996 → 0.997, usable_to 17.2 → 3.39e+38 |
| `bh/bf16/xlogy/default` | max_ulp 897 → 23.5, mean_ulp 655 → 17.6, rounded_frac 0.11 → 0.139 |
| `bh/fp32/asinh/default` | mean_ulp 0.774 → 0.773 |
| `bh/fp32/atanh/default` | max_ulp 3.04 → 2.97 |
| `bh/fp32/cos/default` | max_ulp 3.3e+12 → 1.34e+08, mean_ulp 3.33e+09 → 1.33e+05, ulp_clipped 746 → 342, rounded_frac 0.919 → 0.927 |
| `bh/fp32/erfinv/default` | mean_ulp 2.62e+05 → 2.59e+05, rounded_frac 0.0002 → 0.000202 |
| `bh/fp32/erfinv_bw/default` | mean_ulp 1.68e+03 → 1.67e+03, ulp_clipped 334 → 332, rounded_frac 0.913 → 0.914 |
| `bh/fp32/expm1_bw/default` | max_ulp 4.91e+06 → 4.19e+06, mean_ulp 1.14e+04 → 3.83e+03, ulp_clipped 149 → 54, defects 138 → 0, rounded_frac 0.985 → 0.995, usable_to 1.39 → 22 |
| `bh/fp32/lgamma/default` | ulp_clipped 314 → 313 |
| `bh/fp32/log1p/default` | max_ulp 0.984 → 0.967 |
| `bh/fp32/multigammaln/default` | ulp_clipped 2989 → 2987 |
| `bh/fp32/pow/default` | max_ulp 5.92e+03 → 333, mean_ulp 3.17 → 2.35, ulp_clipped 5 → 0, rounded_frac 0.992 → 0.993 |
| `bh/fp32/rsqrt_bw/default` | max_ulp 7.19 → 6.21, mean_ulp 4.49 → 4.15, rounded_frac 0.293 → 0.303 |
| `bh/fp32/sin/default` | max_ulp 2.2e+12 → 3.36e+07, mean_ulp 2.68e+09 → 3.07e+04, ulp_clipped 490 → 86, rounded_frac 0.955 → 0.96 |
| `bh/fp32/softplus/default` | defects 30 → 5 |
| `bh/fp32/tan/default` | max_ulp 2.2e+12 → 2.26e+08, mean_ulp 5.61e+09 → 2.32e+05, ulp_clipped 746 → 342, rounded_frac 0.938 → 0.944 |
| `bh/fp32/xlogy/default` | max_ulp 8.84e+07 → 2.39e+06, mean_ulp 6.07e+07 → 1.68e+06, rounded_frac 1.15e-05 → 0.00256 |
| `wh/bf16/cos/default` | max_ulp 4.1e+03 → 0.552, mean_ulp 26.5 → 0.552, ulp_clipped 6 → 0, rounded_frac 0.983 → 1, usable_to 1.31e+05 → 9.99e+05 |
| `wh/bf16/gelu_bw/default` | max_ulp 3.5 → 0.939, mean_ulp 1.37 → 0.681, rounded_frac 0.998 → 0.999, usable_to 8.44 → 3.39e+38 |
| `wh/bf16/sin/default` | max_ulp 4.1e+03 → 0.552, mean_ulp 73.5 → 0.526, ulp_clipped 6 → 0, rounded_frac 0.992 → 1, usable_to 2.62e+05 → 9.99e+05 |
| `wh/bf16/tan/default` | max_ulp 4.1e+03 → 0.551, mean_ulp 47.6 → 0.5, ulp_clipped 6 → 0, rounded_frac 0.985 → 0.999, usable_to 1.31e+05 → 9.99e+05 |
| `wh/bf16/tanh_bw/default` | max_ulp 55.5 → 1.16, mean_ulp 1.93 → 0.525, rounded_frac 0.996 → 0.997, usable_to 17.2 → 3.39e+38 |
| `wh/bf16/xlogy/default` | max_ulp 897 → 23.5, mean_ulp 655 → 17.6, rounded_frac 0.11 → 0.139 |
| `wh/fp32/cos/default` | max_ulp 3.3e+12 → 1.34e+08, mean_ulp 3.33e+09 → 1.33e+05, ulp_clipped 746 → 342, rounded_frac 0.919 → 0.927 |
| `wh/fp32/erfinv/default` | mean_ulp 2.62e+05 → 2.59e+05, rounded_frac 0.0002 → 0.000202 |
| `wh/fp32/erfinv_bw/default` | ulp_clipped 334 → 332, rounded_frac 0.913 → 0.914 |
| `wh/fp32/expm1_bw/default` | max_ulp 4.91e+06 → 4.19e+06, mean_ulp 1.14e+04 → 3.83e+03, ulp_clipped 149 → 54, defects 138 → 0, rounded_frac 0.985 → 0.995, usable_to 1.39 → 22 |
| `wh/fp32/lgamma/default` | ulp_clipped 314 → 313 |
| `wh/fp32/log1p/default` | max_ulp 0.984 → 0.967 |
| `wh/fp32/multigammaln/default` | ulp_clipped 2989 → 2987 |
| `wh/fp32/pow/default` | max_ulp 5.92e+03 → 324, mean_ulp 3.17 → 2.42, ulp_clipped 5 → 0, rounded_frac 0.992 → 0.993 |
| `wh/fp32/rsqrt_bw/default` | max_ulp 7.19 → 6.76, mean_ulp 4.49 → 4.26, rounded_frac 0.293 → 0.3 |
| `wh/fp32/sin/default` | max_ulp 2.2e+12 → 3.36e+07, mean_ulp 2.68e+09 → 3.07e+04, ulp_clipped 490 → 86, rounded_frac 0.955 → 0.96 |
| `wh/fp32/softplus/default` | defects 30 → 5 |
| `wh/fp32/tan/default` | max_ulp 2.2e+12 → 2.26e+08, mean_ulp 5.61e+09 → 2.32e+05, ulp_clipped 746 → 342, rounded_frac 0.938 → 0.944 |
| `wh/fp32/xlogy/default` | max_ulp 8.84e+07 → 2.39e+06, mean_ulp 6.07e+07 → 1.68e+06, rounded_frac 1.15e-05 → 0.00256 |

## Expected (6)

No scored metric got worse on comparable measurements. Either `usable_to` moved, because a different kernel puts the 2 ULP boundary on a neighbouring group; or `n_inputs` moved, which means the swept domain changed and the two runs measure different populations; or a metric is reported here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/i1/default` | max_ulp 218 → 0.858, mean_ulp 8.58 → 0.589, rounded_frac 0.993 → 0.516, n_inputs 33904 → 65024 |
| `bh/fp32/i0/default` | usable_to 2.43 → 2.45 |
| `bh/fp32/i1/default` | max_ulp 1.56e+07 → 8.02, mean_ulp 4.1e+04 → 0.731, ulp_clipped 14 → 0, rounded_frac 0.928 → 0.482, n_inputs 33904 → 65024 |
| `wh/bf16/i1/default` | max_ulp 218 → 0.858, mean_ulp 8.58 → 0.589, rounded_frac 0.993 → 0.516, n_inputs 33904 → 65024 |
| `wh/fp32/i0/default` | usable_to 2.43 → 2.45 |
| `wh/fp32/i1/default` | max_ulp 1.56e+07 → 8.02, mean_ulp 4.11e+04 → 0.722, ulp_clipped 14 → 0, rounded_frac 0.929 → 0.483, n_inputs 33904 → 65024 |
