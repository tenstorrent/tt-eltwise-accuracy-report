# Findings

**Status: measured.** Every row below is a published result that changed.

_a3a9fb4229a, fbf7d27db93 → a3a9fb4229a_

## Regressions (114)

A scored metric got worse.

| Variant | Moved |
|---|---|
| `bh/bf16/acos/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/acosh/default` | max_ulp 1 → 1.41, mean_ulp 0.994 → 0.596, unflushed None → 0 |
| `bh/bf16/acosh_bw/default` | max_ulp 3 → 3.26, mean_ulp 1.01 → 0.688, unflushed None → 0, usable_to 1.05 → 1.03 |
| `bh/bf16/addcdiv/default` | mean_ulp 2.21e+03 → 2.35e+03, unflushed None → 0 |
| `bh/bf16/asin/default` | max_ulp 0 → 0.499, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/asinh/default` | max_ulp 1 → 1.32, mean_ulp 0.998 → 0.615, unflushed None → 0 |
| `bh/bf16/asinh_bw/default` | max_ulp 1 → 1.33, mean_ulp 1 → 0.681, unflushed None → 0 |
| `bh/bf16/atan_bw/default` | max_ulp 2 → 2.23, mean_ulp 1.04 → 0.867, unflushed None → 0, usable_to 9.19e+18 → 0.23 |
| `bh/bf16/atanh/default` | max_ulp 2 → 2.08, mean_ulp 0.997 → 0.995, unflushed None → 0 |
| `bh/bf16/atanh_bw/default` | max_ulp 8 → 8.17, mean_ulp 1.13 → 1, unflushed None → 0 |
| `bh/bf16/celu/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/cos_bw/default` | max_ulp 3.38e+38 → 7.45e+42, mean_ulp 2.53e+36 → 3.63e+39, ulp_clipped 4323 → 4648, unflushed None → 0 |
| `bh/bf16/cosh_bw/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/div/default` | max_ulp 0 → 0.498, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/div_bw/default` | max_ulp 0 → 0.498, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/div_no_nan/default` | max_ulp 0 → 0.498, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/divide/default` | max_ulp 0 → 0.498, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/divide_/default` | max_ulp 0 → 0.498, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/elu/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/erf/default` | max_ulp 1 → 1.06, mean_ulp 1 → 0.771, unflushed None → 0 |
| `bh/bf16/exp/fast_approx` | max_ulp 6 → 6.08, unflushed None → 0 |
| `bh/bf16/expm1/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/gelu_bw/default` | max_ulp 3 → 3.5, mean_ulp 1.52 → 1.37, unflushed None → 0 |
| `bh/bf16/hardsigmoid_bw/default` | max_ulp 0 → 0.333, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/hardswish/default` | max_ulp 2 → 2.14, mean_ulp 1.01 → 0.94, unflushed None → 0 |
| `bh/bf16/hardswish_bw/default` | max_ulp 85 → 85.3, mean_ulp 2.36 → 2.1, unflushed None → 0 |
| `bh/bf16/hypot/default` | max_ulp 53 → 52.7, mean_ulp 1.46 → 1.58, unflushed None → 0 |
| `bh/bf16/leaky_relu_bw/default` | max_ulp 0 → 0.16, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/log1p_bw/default` | max_ulp 1 → 1.42, mean_ulp 1 → 0.744, unflushed None → 0 |
| `bh/bf16/log_bw/default` | max_ulp 0 → 0.498, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/log_sigmoid/default` | max_ulp 6 → 6.39, mean_ulp 1.51 → 1.38, unflushed None → 0, usable_to 4.12 → 0.428 |
| `bh/bf16/logaddexp2_bw/default` | max_ulp 41 → 41.4, mean_ulp 3.82 → 3.6, unflushed None → 0 |
| `bh/bf16/logaddexp_bw/default` | max_ulp 62 → 62.5, mean_ulp 4.88 → 4.9, unflushed None → 0 |
| `bh/bf16/mish/default` | max_ulp 1 → 1.49, mean_ulp 0.997 → 0.666, unflushed None → 0 |
| `bh/bf16/mse_loss/default` | max_ulp 2 → 2.03, mean_ulp 1.71 → 1.65, unflushed None → 0 |
| `bh/bf16/multiply/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/multiply_/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/rdiv/value2.0` | max_ulp 0 → 0.498, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/reciprocal/default` | max_ulp 0 → 0.498, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/rsqrt/default` | max_ulp 0 → 0.499, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/rsqrt_bw/default` | max_ulp 3 → 3.23, mean_ulp 1.28 → 1.19, unflushed None → 0 |
| `bh/bf16/selu/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/selu_bw/default` | max_ulp 3 → 2.62, mean_ulp 1.04 → 1.06, unflushed None → 0, usable_to 0.582 → 0.00443 |
| `bh/bf16/sin_bw/default` | max_ulp 2.96e+38 → 1.94e+42, mean_ulp 2.71e+36 → 1.95e+39, ulp_clipped 4349 → 4661, unflushed None → 0 |
| `bh/bf16/sinh/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/sqrt/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/bf16/sqrt_bw/default` | max_ulp 1 → 1.14, mean_ulp 1 → 0.699, unflushed None → 0 |
| `bh/bf16/squared_difference/default` | max_ulp 2 → 2.03, mean_ulp 1.71 → 1.65, unflushed None → 0 |
| `bh/bf16/squared_difference_/default` | max_ulp 2 → 2.03, mean_ulp 1.71 → 1.65, unflushed None → 0 |
| `bh/bf16/tan_bw/default` | max_ulp 2.53e+38 → 3.1e+40, mean_ulp 7.9e+35 → 4.91e+37, ulp_clipped 1232 → 1282, unflushed None → 0, usable_to 1.31e+05 → 1.13 |
| `bh/fp32/acosh/default` | max_ulp 2 → 2.47, mean_ulp 1 → 0.76, unflushed None → 0, usable_to 3.39e+38 → — |
| `bh/fp32/acosh_bw/default` | mean_ulp 1.12 → 1.13, unflushed None → 0 |
| `bh/fp32/add/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/add_/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/addalpha/alpha2.0` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/asinh_bw/default` | max_ulp 2 → 1.98, mean_ulp 1.03 → 1.06, unflushed None → 0 |
| `bh/fp32/atan/default` | max_ulp 2 → 2.34, mean_ulp 1.02 → 0.806, unflushed None → 0, usable_to 3.39e+38 → 0.902 |
| `bh/fp32/atanh/default` | max_ulp 3 → 3.04, unflushed None → 0, usable_to 0.00772 → 0.000229 |
| `bh/fp32/bias_gelu/default` | max_ulp 2.91e+38 → 7.45e+40, mean_ulp 4.94e+36 → 6.51e+39, ulp_clipped 30425 → 35234, unflushed None → 0 |
| `bh/fp32/bias_gelu_/default` | max_ulp 2.91e+38 → 7.45e+40, mean_ulp 4.94e+36 → 6.51e+39, ulp_clipped 30425 → 35234, unflushed None → 0 |
| `bh/fp32/celu/default` | max_ulp 1 → 1.36, mean_ulp 1 → 0.799, unflushed None → 0 |
| `bh/fp32/cos_bw/default` | max_ulp 3.32e+38 → 2.68e+51, mean_ulp 1.86e+36 → 2.8e+48, ulp_clipped 4514 → 9970, unflushed None → 0, usable_to 1.04e+05 → 28 |
| `bh/fp32/cosh/default` | max_ulp 1 → 1.35, mean_ulp 1 → 0.791, unflushed None → 0, usable_to 89 → 89.1 |
| `bh/fp32/cosh_bw/default` | max_ulp 2 → 2.2, mean_ulp 1.12 → 1.16, unflushed None → 0, usable_to 88.5 → 0.0155 |
| `bh/fp32/div/default` | max_ulp 2 → 2.44, mean_ulp 1.58 → 1.44, unflushed None → 0 |
| `bh/fp32/divide/default` | max_ulp 2 → 2.44, mean_ulp 1.58 → 1.44, unflushed None → 0 |
| `bh/fp32/divide_/default` | max_ulp 2 → 2.44, mean_ulp 1.58 → 1.44, unflushed None → 0 |
| `bh/fp32/elu/default` | max_ulp 1 → 1.36, mean_ulp 1 → 0.799, unflushed None → 0 |
| `bh/fp32/erf/default` | max_ulp 7 → 7.38, mean_ulp 1.23 → 1.36, unflushed None → 0 |
| `bh/fp32/erf_bw/default` | max_ulp 65 → 65.5, mean_ulp 3.72 → 3.73, unflushed None → 0, usable_to 0.996 → 0.348 |
| `bh/fp32/erfc/default` | ulp_clipped 29690 → 29691, unflushed None → 198 |
| `bh/fp32/erfc_bw/default` | max_ulp 65 → 65.5, mean_ulp 3.72 → 3.73, unflushed None → 0, usable_to 0.996 → 0.348 |
| `bh/fp32/fmod/default` | max_ulp 3.19e+38 → 1.61e+45, mean_ulp 4e+36 → 2.55e+41, ulp_clipped 29398 → 61698, unflushed None → 0 |
| `bh/fp32/gelu/fast_approx` | max_ulp 2.91e+38 → 7.45e+40, mean_ulp 4.9e+36 → 1.18e+39, ulp_clipped 30383 → 32433, unflushed None → 254, usable_to 6e-36 → 2.33e-38 |
| `bh/fp32/hardsigmoid/default` | mean_ulp 1.25e+03 → 3e+03, unflushed None → 0, usable_to 2.61 → 2.25 |
| `bh/fp32/l1_loss/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/ldexp/default` | max_ulp 2 → 1.92, mean_ulp 1.5 → 1.52, unflushed None → 0 |
| `bh/fp32/ldexp_/default` | max_ulp 2 → 1.92, mean_ulp 1.5 → 1.52, unflushed None → 0 |
| `bh/fp32/leaky_relu_bw/default` | max_ulp 0 → 0.24, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/lerp_bw/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/lgamma/default` | ulp_clipped 313 → 314, unflushed None → 0, usable_to 0.108 → 0.0181 |
| `bh/fp32/log10/default` | max_ulp 2 → 2.13, mean_ulp 1.22 → 1.36, unflushed None → 0, usable_to 3.39e+38 → 0.336 |
| `bh/fp32/log2/default` | max_ulp 2 → 2.43, mean_ulp 1 → 0.511, unflushed None → 0, usable_to 3.39e+38 → 0.704 |
| `bh/fp32/log_sigmoid_bw/default` | max_ulp 3 → 2.93, mean_ulp 1.25 → 1.28, unflushed None → 0, usable_to 1.09 → 0.163 |
| `bh/fp32/logaddexp/default` | ulp_clipped 46909 → 46910, unflushed None → 0 |
| `bh/fp32/logaddexp_/default` | ulp_clipped 46909 → 46910, unflushed None → 0 |
| `bh/fp32/logit_bw/default` | max_ulp 2 → 2.28, mean_ulp 1.05 → 0.789, unflushed None → 0, usable_to 0.998 → 2.97e-08 |
| `bh/fp32/logiteps_bw/default` | max_ulp 2 → 2.28, mean_ulp 1.05 → 0.789, unflushed None → 0, usable_to 0.998 → 2.97e-08 |
| `bh/fp32/multigammaln/default` | ulp_clipped 2987 → 2989, unflushed None → 0 |
| `bh/fp32/multiply/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/multiply_/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/prelu/weight0.25` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/rdiv/value2.0` | max_ulp 1 → 1.49, mean_ulp 1 → 1.19, unflushed None → 0 |
| `bh/fp32/remainder/default` | max_ulp 3.19e+38 → 8.92e+44, mean_ulp 4.03e+36 → 2.49e+41, ulp_clipped 29402 → 61699, unflushed None → 0 |
| `bh/fp32/rsqrt/default` | max_ulp 1 → 1.12, mean_ulp 1 → 0.834, unflushed None → 0 |
| `bh/fp32/rsqrt_bw/default` | max_ulp 7 → 7.19, mean_ulp 4.5 → 4.49, unflushed None → 0 |
| `bh/fp32/rsub/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/rsub_/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/selu/default` | max_ulp 51 → 51.3, mean_ulp 26.2 → 26.3, unflushed None → 0 |
| `bh/fp32/sigmoid/default` | max_ulp 3 → 3.31, mean_ulp 1.65 → 1.59, unflushed None → 0, usable_to 0.000475 → 1.59e-05 |
| `bh/fp32/sigmoid_accurate/default` | max_ulp 3 → 3.31, mean_ulp 1.65 → 1.59, unflushed None → 0, usable_to 0.000475 → 1.59e-05 |
| `bh/fp32/sin_bw/default` | max_ulp 3.21e+38 → 1.56e+53, mean_ulp 1.45e+36 → 2.03e+49, ulp_clipped 4770 → 10226, unflushed None → 0, usable_to 5.23e+04 → 92.4 |
| `bh/fp32/sinh/default` | max_ulp 2 → 2.2, mean_ulp 1.12 → 1.16, unflushed None → 0, usable_to 89 → 0.0155 |
| `bh/fp32/sinh_bw/default` | max_ulp 1 → 1.35, mean_ulp 1 → 0.791, unflushed None → 0 |
| `bh/fp32/softplus/default` | max_ulp 8.2e+03 → 8.21e+03, unflushed None → 0 |
| `bh/fp32/softshrink/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/sqrt_bw/default` | max_ulp 2 → 2.19, mean_ulp 1.38 → 1.4, unflushed None → 0, usable_to 3.39e+38 → — |
| `bh/fp32/square/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/squared_difference_bw/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/subalpha/alpha2.0` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/subtract/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/subtract_/default` | max_ulp 0 → 0.5, mean_ulp — → 0, unflushed None → 0 |
| `bh/fp32/tan_bw/default` | max_ulp 3.16e+38 → 2.85e+45, mean_ulp 9.23e+35 → 5.11e+44, ulp_clipped 3944 → 8039, unflushed None → 0, usable_to 1.01 → 0.852 |
| `bh/fp32/tanhshrink_bw/default` | mean_ulp 9.94e+04 → 1.02e+05, unflushed None → 0 |

## Improvements (111)

A scored metric got better.

| Variant | Moved |
|---|---|
| `bh/bf16/acos_bw/default` | max_ulp 3 → 2.7, mean_ulp 1.1 → 0.744, unflushed None → 0, usable_to 0.953 → 0.949 |
| `bh/bf16/add/default` | max_ulp 1 → 0.621, mean_ulp 1 → 0.571, unflushed None → 0 |
| `bh/bf16/add_/default` | max_ulp 1 → 0.621, mean_ulp 1 → 0.571, unflushed None → 0 |
| `bh/bf16/addalpha/alpha2.0` | mean_ulp 2.42 → 2, unflushed None → 0 |
| `bh/bf16/addcmul/default` | mean_ulp 19.6 → 19.4, unflushed None → 0 |
| `bh/bf16/asin_bw/default` | max_ulp 3 → 2.76, mean_ulp 1.1 → 0.787, unflushed None → 0 |
| `bh/bf16/atan/default` | max_ulp 1 → 0.527, mean_ulp 1 → 0.513, unflushed None → 0 |
| `bh/bf16/atan2/default` | mean_ulp 3.85 → 3.56, unflushed None → 0 |
| `bh/bf16/atan2_bw/default` | mean_ulp 4.92 → 4.89, unflushed None → 0 |
| `bh/bf16/cbrt/default` | max_ulp 1 → 0.507, mean_ulp 1 → 0.502, unflushed None → 0 |
| `bh/bf16/celu_bw/default` | max_ulp 1 → 0.892, mean_ulp 0.776 → 0.52, unflushed None → 0 |
| `bh/bf16/cosh/default` | max_ulp 1 → 0.501, mean_ulp 1 → 0.501, unflushed None → 0 |
| `bh/bf16/deg2rad/default` | max_ulp 1 → 0.998, mean_ulp 0.992 → 0.744, unflushed None → 0 |
| `bh/bf16/elu_bw/default` | max_ulp 1 → 0.892, mean_ulp 0.776 → 0.52, unflushed None → 0 |
| `bh/bf16/erf_bw/default` | mean_ulp 5 → 4.97, unflushed None → 0, usable_to 1.75 → 1.5 |
| `bh/bf16/erfc_bw/default` | mean_ulp 5 → 4.97, unflushed None → 0, usable_to 1.75 → 1.5 |
| `bh/bf16/erfinv/default` | mean_ulp 6.53 → 6.4, unflushed None → 0 |
| `bh/bf16/erfinv_bw/default` | max_ulp 9 → 8.91, mean_ulp 1.39 → 1.28, unflushed None → 0 |
| `bh/bf16/exp/default` | max_ulp 1 → 0.892, mean_ulp 0.856 → 0.556, unflushed None → 0 |
| `bh/bf16/exp2/default` | max_ulp 1 → 0.898, mean_ulp 0.843 → 0.564, unflushed None → 0 |
| `bh/bf16/exp2_bw/default` | max_ulp 2 → 1.89, mean_ulp 1.05 → 0.942, unflushed None → 0 |
| `bh/bf16/exp_bw/default` | max_ulp 1 → 0.892, mean_ulp 0.856 → 0.556, unflushed None → 0 |
| `bh/bf16/expm1_bw/default` | mean_ulp 5.83 → 5.58, unflushed None → 0 |
| `bh/bf16/gelu/default` | mean_ulp 9.42 → 9.18, unflushed None → 0 |
| `bh/bf16/hardmish/default` | mean_ulp 0.993 → 0.913, unflushed None → 0 |
| `bh/bf16/hardsigmoid/default` | mean_ulp 0.661 → 0.557, unflushed None → 0 |
| `bh/bf16/hypot_bw/default` | max_ulp 75 → 74.6, mean_ulp 2.44 → 2.43, unflushed None → 0 |
| `bh/bf16/i0/default` | mean_ulp 56.4 → 56.3, unflushed None → 0, usable_to 13.9 → 13.6 |
| `bh/bf16/i0_bw/default` | mean_ulp 8.95 → 8.58, unflushed None → 0 |
| `bh/bf16/l1_loss/default` | max_ulp 1 → 0.621, mean_ulp 1 → 0.571, unflushed None → 0 |
| `bh/bf16/ldexp/default` | mean_ulp 94.5 → 94.4, unflushed None → 0 |
| `bh/bf16/ldexp_/default` | mean_ulp 94.5 → 94.4, unflushed None → 0 |
| `bh/bf16/ldexp_bw/default` | max_ulp 1 → 0.898, mean_ulp 1 → 0.898, unflushed None → 0 |
| `bh/bf16/leaky_relu/negative_slope0.01` | max_ulp 1 → 0.96, mean_ulp 1 → 0.741, unflushed None → 0 |
| `bh/bf16/lerp/default` | mean_ulp 40 → 39.9, unflushed None → 0 |
| `bh/bf16/lerp_bw/default` | max_ulp 1 → 0.506, mean_ulp 1 → 0.506, unflushed None → 0 |
| `bh/bf16/lgamma/default` | mean_ulp 1.17 → 0.846, unflushed None → 0, usable_to 0.439 → 0.412 |
| `bh/bf16/log/default` | max_ulp 1 → 0.78, mean_ulp 1 → 0.534, unflushed None → 0 |
| `bh/bf16/log10/default` | max_ulp 1 → 0.785, mean_ulp 1 → 0.543, unflushed None → 0 |
| `bh/bf16/log10_bw/default` | max_ulp 2 → 1.74, mean_ulp 1.02 → 0.803, unflushed None → 0 |
| `bh/bf16/log1p/default` | max_ulp 1 → 0.931, mean_ulp 1 → 0.57, unflushed None → 0 |
| `bh/bf16/log2/default` | max_ulp 1 → 0.766, mean_ulp 1 → 0.542, unflushed None → 0 |
| `bh/bf16/log2_bw/default` | max_ulp 2 → 1.51, mean_ulp 1.02 → 0.847, unflushed None → 0 |
| `bh/bf16/log_sigmoid_bw/default` | max_ulp 2 → 1.71, mean_ulp 0.895 → 0.722, unflushed None → 0 |
| `bh/bf16/logit/default` | mean_ulp 1.95 → 1.67, unflushed None → 0 |
| `bh/bf16/logit_bw/default` | max_ulp 2 → 1.65, mean_ulp 1 → 0.729, unflushed None → 0 |
| `bh/bf16/logiteps_bw/default` | max_ulp 2 → 1.65, mean_ulp 1 → 0.729, unflushed None → 0 |
| `bh/bf16/mac/default` | max_ulp 64 → 63.5, unflushed None → 0 |
| `bh/bf16/multigammaln/default` | mean_ulp 1.81 → 1.67, unflushed None → 0 |
| `bh/bf16/multigammaln_bw/default` | mean_ulp 294 → 293, unflushed None → 0 |
| `bh/bf16/rad2deg/default` | max_ulp 1 → 0.992, mean_ulp 1 → 0.75, unflushed None → 0 |
| `bh/bf16/rdiv_bw/scalar2.0` | max_ulp 2 → 1.51, mean_ulp 1.02 → 0.837, unflushed None → 0 |
| `bh/bf16/reciprocal_bw/default` | max_ulp 2 → 1.51, mean_ulp 1.02 → 0.837, unflushed None → 0 |
| `bh/bf16/rpow/exponent2.0` | max_ulp 1 → 0.898, mean_ulp 0.843 → 0.564, unflushed None → 0 |
| `bh/bf16/rsub/default` | max_ulp 1 → 0.621, mean_ulp 1 → 0.571, unflushed None → 0 |
| `bh/bf16/rsub_/default` | max_ulp 1 → 0.621, mean_ulp 1 → 0.571, unflushed None → 0 |
| `bh/bf16/sigmoid/default` | max_ulp 1 → 0.873, mean_ulp 0.886 → 0.522, unflushed None → 0 |
| `bh/bf16/sigmoid_accurate/default` | max_ulp 1 → 0.873, mean_ulp 0.886 → 0.522, unflushed None → 0 |
| `bh/bf16/sigmoid_bw/default` | mean_ulp 4.97 → 4.81, unflushed None → 0, usable_to 1.86 → 1.76 |
| `bh/bf16/silu/default` | max_ulp 1 → 0.888, mean_ulp 0.998 → 0.582, unflushed None → 0 |
| `bh/bf16/silu_bw/default` | mean_ulp 1.54 → 1.3, unflushed None → 0, usable_to 0.887 → 0.863 |
| `bh/bf16/sinh_bw/default` | max_ulp 1 → 0.501, mean_ulp 1 → 0.501, unflushed None → 0 |
| `bh/bf16/softcap/beta50.0` | max_ulp 1 → 0.718, mean_ulp 1 → 0.574, unflushed None → 0 |
| `bh/bf16/softplus/default` | max_ulp 1 → 0.775, mean_ulp 1 → 0.556, unflushed None → 0 |
| `bh/bf16/softplus_bw/default` | max_ulp 2 → 1.71, mean_ulp 0.861 → 0.678, unflushed None → 0 |
| `bh/bf16/softshrink/default` | mean_ulp 0.996 → 0.948, unflushed None → 0 |
| `bh/bf16/softsign/default` | mean_ulp 0.642 → 0.605, unflushed None → 0 |
| `bh/bf16/softsign_bw/default` | mean_ulp 3.34 → 3.24, unflushed None → 0, usable_to 0.00491 → 0.00391 |
| `bh/bf16/square/default` | max_ulp 1 → 0.973, mean_ulp 0.991 → 0.688, unflushed None → 0 |
| `bh/bf16/squared_difference_bw/default` | max_ulp 1 → 0.621, mean_ulp 1 → 0.571, unflushed None → 0 |
| `bh/bf16/subalpha/alpha2.0` | mean_ulp 2.42 → 2, unflushed None → 0 |
| `bh/bf16/subtract/default` | max_ulp 1 → 0.621, mean_ulp 1 → 0.571, unflushed None → 0 |
| `bh/bf16/subtract_/default` | max_ulp 1 → 0.621, mean_ulp 1 → 0.571, unflushed None → 0 |
| `bh/bf16/swish/default` | max_ulp 1 → 0.888, mean_ulp 0.998 → 0.582, unflushed None → 0 |
| `bh/bf16/tanh/default` | max_ulp 1 → 0.811, mean_ulp 1 → 0.586, unflushed None → 0 |
| `bh/bf16/tanh_bw/default` | mean_ulp 2.23 → 1.93, unflushed None → 0 |
| `bh/bf16/tanhshrink/default` | mean_ulp 9.5 → 9.38, unflushed None → 0 |
| `bh/bf16/tanhshrink_bw/default` | mean_ulp 3.71 → 3.52, unflushed None → 0 |
| `bh/bf16/xielu/default` | max_ulp 1 → 0.5, mean_ulp 1 → 0.5, unflushed None → 0 |
| `bh/bf16/xlogy_bw/default` | max_ulp 1 → 0.78, mean_ulp 1 → 0.78, unflushed None → 0 |
| `bh/fp32/acos/default` | max_ulp 2 → 1.55, mean_ulp 1 → 0.865, unflushed None → 0 |
| `bh/fp32/acos_bw/default` | mean_ulp 1.61 → 1.34, unflushed None → 0, usable_to 0.938 → 0.926 |
| `bh/fp32/asin/default` | max_ulp 2 → 1.77, mean_ulp 1.08 → 0.795, unflushed None → 0 |
| `bh/fp32/asin_bw/default` | mean_ulp 1.61 → 1.34, unflushed None → 0, usable_to 0.938 → 0.926 |
| `bh/fp32/asinh/default` | max_ulp 2 → 1.64, mean_ulp 1 → 0.774, unflushed None → 0 |
| `bh/fp32/cbrt/default` | max_ulp 3 → 2.55, mean_ulp 1.82 → 1.76, unflushed None → 0, usable_to 5.77e-38 → 2.26e-38 |
| `bh/fp32/celu_bw/default` | max_ulp 1 → 0.866, mean_ulp 1 → 0.578, unflushed None → 0 |
| `bh/fp32/deg2rad/default` | max_ulp 1 → 0.63, mean_ulp 1 → 0.595, unflushed None → 0, usable_to 3.39e+38 → 3.4e+38 |
| `bh/fp32/elu_bw/default` | max_ulp 1 → 0.866, mean_ulp 1 → 0.578, unflushed None → 0 |
| `bh/fp32/exp/default` | max_ulp 1 → 0.866, mean_ulp 1 → 0.578, unflushed None → 0 |
| `bh/fp32/exp2/default` | max_ulp 1 → 0.965, mean_ulp 1 → 0.626, unflushed None → 0 |
| `bh/fp32/exp2_bw/default` | max_ulp 2 → 1.85, mean_ulp 1.17 → 1.09, unflushed None → 0 |
| `bh/fp32/exp_bw/default` | max_ulp 1 → 0.866, mean_ulp 1 → 0.578, unflushed None → 0 |
| `bh/fp32/expm1/default` | max_ulp 1 → 0.997, mean_ulp 1 → 0.565, unflushed None → 0 |
| `bh/fp32/ldexp_bw/default` | max_ulp 1 → 0.742, mean_ulp 1 → 0.742, unflushed None → 0 |
| `bh/fp32/leaky_relu/negative_slope0.01` | max_ulp 1 → 0.84, mean_ulp 1 → 0.747, unflushed None → 0 |
| `bh/fp32/log/default` | max_ulp 1 → 0.961, mean_ulp 1 → 0.562, unflushed None → 0 |
| `bh/fp32/log1p/default` | max_ulp 1 → 0.984, mean_ulp 1 → 0.558, unflushed None → 0 |
| `bh/fp32/mse_loss/default` | max_ulp 2 → 1.91, mean_ulp 1.94 → 1.78, unflushed None → 0 |
| `bh/fp32/pow/default` | mean_ulp 3.21 → 3.17, unflushed None → 0 |
| `bh/fp32/rad2deg/default` | max_ulp 1 → 0.696, mean_ulp 1 → 0.643, unflushed None → 0, usable_to 5.9e+36 → 5.93e+36 |
| `bh/fp32/rpow/exponent2.0` | max_ulp 1 → 0.879, mean_ulp 1 → 0.609, unflushed None → 0 |
| `bh/fp32/selu_bw/default` | max_ulp 51 → 50.7, mean_ulp 20.6 → 20.4, unflushed None → 0 |
| `bh/fp32/silu/default` | max_ulp 4 → 3.61, mean_ulp 1.82 → 1.78, unflushed None → 0, usable_to 2.63e-05 → 2.01e-05 |
| `bh/fp32/softplus_bw/default` | max_ulp 3 → 2.93, mean_ulp 1.36 → 1.33, unflushed None → 0, usable_to 1.09 → 0.164 |
| `bh/fp32/sqrt/default` | max_ulp 1 → 0.867, mean_ulp 1 → 0.827, unflushed None → 0 |
| `bh/fp32/squared_difference/default` | max_ulp 2 → 1.91, mean_ulp 1.94 → 1.78, unflushed None → 0 |
| `bh/fp32/squared_difference_/default` | max_ulp 2 → 1.91, mean_ulp 1.94 → 1.78, unflushed None → 0 |
| `bh/fp32/swish/default` | max_ulp 4 → 3.61, mean_ulp 1.82 → 1.78, unflushed None → 0, usable_to 2.63e-05 → 2.01e-05 |
| `bh/fp32/tanh/default` | max_ulp 3 → 2.6, mean_ulp 1.54 → 1.45, unflushed None → 0, usable_to 0.0154 → 0.000946 |
| `bh/fp32/xlogy_bw/default` | max_ulp 1 → 0.766, mean_ulp 1 → 0.766, unflushed None → 0 |

## Expected (26)

No scored metric got worse on comparable measurements. Either `usable_to` moved, because a different kernel puts the 2 ULP boundary on a neighbouring group; or `n_inputs` moved, which means the swept domain changed and the two runs measure different populations; or a metric is reported here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/cos/default` | max_ulp 2.96e+38 → 4.1e+03, mean_ulp 2.71e+36 → 26.5, ulp_clipped 4349 → 6, defects 21274 → 0, unflushed None → 0, n_inputs 65024 → 37354 |
| `bh/bf16/erfc/default` | unflushed None → 198 |
| `bh/bf16/gelu/fast_approx` | unflushed None → 254 |
| `bh/bf16/polygamma/k1` | mean_ulp 1.36e+04 → 2.13e+05, defects 263 → 7, unflushed None → 0, n_inputs 64769 → 49922 |
| `bh/bf16/sin/default` | max_ulp 3.38e+38 → 4.1e+03, mean_ulp 2.53e+36 → 73.5, ulp_clipped 4323 → 6, defects 21275 → 0, unflushed None → 0, n_inputs 65024 → 37354 |
| `bh/bf16/tan/default` | max_ulp 3.3e+38 → 4.1e+03, mean_ulp 1.16e+36 → 49.1, ulp_clipped 2345 → 6, defects 21634 → 0, unflushed None → 0, n_inputs 65024 → 37354 |
| `bh/fp32/atan_bw/default` | unflushed None → 0, usable_to 4.08e+03 → 1.02 |
| `bh/fp32/atanh_bw/default` | unflushed None → 0, usable_to 0.855 → 0.681 |
| `bh/fp32/cos/default` | max_ulp 3.21e+38 → 3.3e+12, mean_ulp 1.45e+36 → 3.33e+09, ulp_clipped 4770 → 746, defects 18190 → 0, unflushed None → 0, usable_to 5.23e+04 → 92.4, n_inputs 65024 → 37354 |
| `bh/fp32/erfinv_bw/default` | unflushed None → 0, usable_to 0.000496 → 0.000456 |
| `bh/fp32/expm1_bw/default` | unflushed None → 0, usable_to 2.08 → 1.39 |
| `bh/fp32/gelu/default` | unflushed None → 0, usable_to 0.301 → 0.208 |
| `bh/fp32/gelu_bw/default` | unflushed None → 0, usable_to 0.0393 → 0.0296 |
| `bh/fp32/hardswish_bw/default` | unflushed None → 0, usable_to 0.00179 → 0.00133 |
| `bh/fp32/i0/default` | unflushed None → 0, usable_to 2.56 → 2.43 |
| `bh/fp32/i0_bw/default` | unflushed None → 0, usable_to 0.00664 → 0.00362 |
| `bh/fp32/log10_bw/default` | unflushed None → 0, usable_to 8.81e+30 → 4.42e+30 |
| `bh/fp32/logit/default` | unflushed None → 0, usable_to 0.375 → 0.266 |
| `bh/fp32/polygamma/k1` | mean_ulp 3.25e+29 → 4.68e+29, ulp_clipped 15847 → 1000, defects 257 → 1, unflushed None → 0, n_inputs 64769 → 49922 |
| `bh/fp32/sigmoid_bw/default` | unflushed None → 0, usable_to 0.617 → 0.283 |
| `bh/fp32/silu_bw/default` | unflushed None → 0, usable_to 1.59e-05 → 2.98e-07 |
| `bh/fp32/sin/default` | max_ulp 3.32e+38 → 2.2e+12, mean_ulp 1.86e+36 → 2.68e+09, ulp_clipped 4514 → 490, defects 18190 → 0, unflushed None → 0, usable_to 1.04e+05 → 28, n_inputs 65024 → 37354 |
| `bh/fp32/softsign/default` | unflushed None → 0, usable_to 0.000243 → 1.21e-05 |
| `bh/fp32/softsign_bw/default` | unflushed None → 0, usable_to 6.44e-05 → 1.79e-07 |
| `bh/fp32/tan/default` | max_ulp 3.34e+38 → 2.2e+12, mean_ulp 1.43e+36 → 5.61e+09, ulp_clipped 3698 → 746, defects 20110 → 0, unflushed None → 0, usable_to 20.3 → 3.92, n_inputs 65024 → 37354 |
| `bh/fp32/xielu/default` | unflushed None → 0, usable_to 0.691 → 1.71e-13 |

## New coverage (86)

Measured here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/clamp/min0.0_max0.0` | — |
| `bh/bf16/clamp/min1.0_max-1.0` | — |
| `bh/bf16/clip/min0.0_max0.0` | — |
| `bh/bf16/clip/min1.0_max-1.0` | — |
| `bh/bf16/eq/default` | — |
| `bh/bf16/eq_/default` | — |
| `bh/bf16/eqz/default` | — |
| `bh/bf16/ge/default` | — |
| `bh/bf16/gez/default` | — |
| `bh/bf16/gt/default` | — |
| `bh/bf16/gtz/default` | — |
| `bh/bf16/heaviside/value0.0` | — |
| `bh/bf16/heaviside/value1.0` | — |
| `bh/bf16/i1/default` | — |
| `bh/bf16/isclose/default` | — |
| `bh/bf16/isfinite/default` | — |
| `bh/bf16/isinf/default` | — |
| `bh/bf16/isnan/default` | — |
| `bh/bf16/isneginf/default` | — |
| `bh/bf16/isposinf/default` | — |
| `bh/bf16/le/default` | — |
| `bh/bf16/leaky_relu/negative_slope0.0` | — |
| `bh/bf16/leaky_relu/negative_slope1.0` | — |
| `bh/bf16/lez/default` | — |
| `bh/bf16/logical_and/default` | — |
| `bh/bf16/logical_and_/default` | — |
| `bh/bf16/logical_not/default` | — |
| `bh/bf16/logical_or/default` | — |
| `bh/bf16/logical_or_/default` | — |
| `bh/bf16/lt/default` | — |
| `bh/bf16/ltz/default` | — |
| `bh/bf16/ne/default` | — |
| `bh/bf16/ne_/default` | — |
| `bh/bf16/nez/default` | — |
| `bh/bf16/polygamma/k2` | — |
| `bh/bf16/polygamma/k4` | — |
| `bh/bf16/relu_max/upper_limit0.0` | — |
| `bh/bf16/relu_max/upper_limit6.0` | — |
| `bh/bf16/relu_min/lower_limit0.0` | — |
| `bh/bf16/rpow/exponent0.5` | — |
| `bh/bf16/rpow/exponent1.0` | — |
| `bh/bf16/signbit/default` | — |
| `bh/bf16/threshold/threshold0.0_value1.0` | — |
| `bh/fp32/clamp/min0.0_max0.0` | — |
| `bh/fp32/clamp/min1.0_max-1.0` | — |
| `bh/fp32/clip/min0.0_max0.0` | — |
| `bh/fp32/clip/min1.0_max-1.0` | — |
| `bh/fp32/eq/default` | — |
| `bh/fp32/eq_/default` | — |
| `bh/fp32/eqz/default` | — |
| `bh/fp32/ge/default` | — |
| `bh/fp32/gez/default` | — |
| `bh/fp32/gt/default` | — |
| `bh/fp32/gtz/default` | — |
| `bh/fp32/heaviside/value0.0` | — |
| `bh/fp32/heaviside/value1.0` | — |
| `bh/fp32/i1/default` | — |
| `bh/fp32/isclose/default` | — |
| `bh/fp32/isfinite/default` | — |
| `bh/fp32/isinf/default` | — |
| `bh/fp32/isnan/default` | — |
| `bh/fp32/isneginf/default` | — |
| `bh/fp32/isposinf/default` | — |
| `bh/fp32/le/default` | — |
| `bh/fp32/leaky_relu/negative_slope0.0` | — |
| `bh/fp32/leaky_relu/negative_slope1.0` | — |
| `bh/fp32/lez/default` | — |
| `bh/fp32/logical_and/default` | — |
| `bh/fp32/logical_and_/default` | — |
| `bh/fp32/logical_not/default` | — |
| `bh/fp32/logical_or/default` | — |
| `bh/fp32/logical_or_/default` | — |
| `bh/fp32/lt/default` | — |
| `bh/fp32/ltz/default` | — |
| `bh/fp32/ne/default` | — |
| `bh/fp32/ne_/default` | — |
| `bh/fp32/nez/default` | — |
| `bh/fp32/polygamma/k2` | — |
| `bh/fp32/polygamma/k4` | — |
| `bh/fp32/relu_max/upper_limit0.0` | — |
| `bh/fp32/relu_max/upper_limit6.0` | — |
| `bh/fp32/relu_min/lower_limit0.0` | — |
| `bh/fp32/rpow/exponent0.5` | — |
| `bh/fp32/rpow/exponent1.0` | — |
| `bh/fp32/signbit/default` | — |
| `bh/fp32/threshold/threshold0.0_value1.0` | — |
