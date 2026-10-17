# Findings

**Status: measured.** Every row below is a published result that changed.

_de546d3b146 → 978f93b56e3_

## Regressions (17)

A scored metric got worse.

| Variant | Moved |
|---|---|
| `bh/bf16/erf_bw/default` | max_ulp 112 → 70.5, mean_ulp 4.97 → 3.04, rounded_frac 0.976 → 0.975, usable_to 1.5 → 1.38 |
| `bh/bf16/erfc_bw/default` | max_ulp 112 → 70.5, mean_ulp 4.97 → 3.04, rounded_frac 0.976 → 0.975, usable_to 1.5 → 1.38 |
| `bh/bf16/logaddexp2_/default` | max_ulp 3.03e+06 → 6.87e+10, mean_ulp 4.08e+03 → 2.05e+06, ulp_clipped 16684 → 119, defects 15488 → 0, rounded_frac 0.871 → 0.988 |
| `bh/bf16/nextafter/default` | max_ulp 1.3e+33 → 1, mean_ulp 2.32e+31 → 1, ulp_clipped 25856 → 0, rounded_frac 0.56 → 0.00585 |
| `bh/bf16/sigmoid_bw/default` | max_ulp 125 → 127, mean_ulp 4.81 → 12.7, defects 331 → 141, rounded_frac 0.988 → 0.999, usable_to 1.77 → 12.5 |
| `bh/bf16/tan_bw/default` | mean_ulp 1.01e+38 → 1e+38, rounded_frac 0.855 → 0.853, usable_to 1.13 → 940 |
| `bh/fp32/nextafter/default` | max_ulp 8.51e+37 → 1, mean_ulp 1.34e+36 → 1, ulp_clipped 29952 → 0, rounded_frac 0.5 → 1.54e-05 |
| `bh/fp32/softplus_bw/default` | max_ulp 2.93 → 3.31, mean_ulp 1.33 → 1.59, rounded_frac 0.954 → 0.953, usable_to 0.164 → 1.59e-05 |
| `wh/bf16/erf_bw/default` | max_ulp 112 → 70.5, mean_ulp 4.97 → 3.04, rounded_frac 0.976 → 0.975, usable_to 1.5 → 1.38 |
| `wh/bf16/erfc_bw/default` | max_ulp 112 → 70.5, mean_ulp 4.97 → 3.04, rounded_frac 0.976 → 0.975, usable_to 1.5 → 1.38 |
| `wh/bf16/log10_bw/default` | mean_ulp 0.796 → 0.803, rounded_frac 0.524 → 0.508 |
| `wh/bf16/log2_bw/default` | mean_ulp 0.842 → 0.847, rounded_frac 0.578 → 0.57 |
| `wh/bf16/logaddexp2/default` | max_ulp 3.03e+06 → 6.87e+10, mean_ulp 4.08e+03 → 2.05e+06, ulp_clipped 16684 → 119, defects 15488 → 0, rounded_frac 0.871 → 0.988 |
| `wh/bf16/logaddexp2_/default` | max_ulp 3.03e+06 → 6.87e+10, mean_ulp 4.08e+03 → 2.05e+06, ulp_clipped 16684 → 119, defects 15488 → 0, rounded_frac 0.871 → 0.988 |
| `wh/bf16/nextafter/default` | max_ulp 1.3e+33 → 1, mean_ulp 2.32e+31 → 1, ulp_clipped 25856 → 0, rounded_frac 0.56 → 0.00588 |
| `wh/bf16/tan_bw/default` | mean_ulp 3.81e+37 → 3.8e+37, rounded_frac 0.691 → 0.69, usable_to 1.13 → 940 |
| `wh/fp32/nextafter/default` | max_ulp 8.51e+37 → 1, mean_ulp 1.34e+36 → 1, ulp_clipped 29952 → 0, rounded_frac 0.5 → 4.06e-05 |

## Improvements (65)

A scored metric got better.

| Variant | Moved |
|---|---|
| `bh/bf16/acosh_bw/default` | max_ulp 3.26 → 3.07, mean_ulp 0.688 → 0.64, rounded_frac 0.781 → 0.884 |
| `bh/bf16/asin_bw/default` | max_ulp 2.76 → 2.7, mean_ulp 0.787 → 0.702, usable_to 0.938 → 0.949 |
| `bh/bf16/asinh_bw/default` | max_ulp 1.33 → 1.2, mean_ulp 0.681 → 0.638, rounded_frac 0.918 → 0.954 |
| `bh/bf16/atan_bw/default` | mean_ulp 0.867 → 0.767, rounded_frac 0.849 → 0.896 |
| `bh/bf16/atanh_bw/default` | max_ulp 8.17 → 7.76, mean_ulp 1 → 0.885, rounded_frac 0.79 → 0.835, usable_to 0.82 → 0.918 |
| `bh/bf16/bias_gelu_/default` | max_ulp 1.14e+36 → 179, mean_ulp 4.5e+34 → 86.3, ulp_clipped 30210 → 0, rounded_frac 0.749 → 0.996 |
| `bh/bf16/cosh_bw/default` | defects 2 → 0, usable_to 88.5 → 89 |
| `bh/bf16/erfinv_bw/default` | mean_ulp 1.28 → 1.2, usable_to 0.777 → 0.859 |
| `bh/bf16/hypot/default` | mean_ulp 1.58 → 1.64, rounded_frac 0.947 → 0.992 |
| `bh/bf16/logaddexp_/default` | max_ulp 8.39e+06 → 2.22e+05, mean_ulp 4.85e+03 → 44.7, ulp_clipped 16583 → 108, defects 15566 → 0, rounded_frac 0.858 → 0.986 |
| `bh/bf16/mse_loss/default` | mean_ulp 1.65 → 1.79, rounded_frac 0.571 → 0.956 |
| `bh/bf16/reciprocal_bw/default` | max_ulp 1.51 → 1.23, mean_ulp 0.837 → 0.714, rounded_frac 0.602 → 0.734 |
| `bh/bf16/sinh_bw/default` | defects 2 → 0, usable_to 88.5 → 89 |
| `bh/bf16/softplus_bw/default` | max_ulp 1.71 → 0.873, mean_ulp 0.678 → 0.522, rounded_frac 0.983 → 0.994 |
| `bh/bf16/softsign_bw/default` | max_ulp 81 → 79, mean_ulp 3.24 → 3.74, rounded_frac 0.856 → 0.887 |
| `bh/bf16/square/default` | max_ulp 0.973 → 0.5, mean_ulp 0.688 → 0, rounded_frac 0.578 → 1 |
| `bh/bf16/squared_difference/default` | mean_ulp 1.65 → 1.79, rounded_frac 0.571 → 0.956 |
| `bh/bf16/squared_difference_/default` | mean_ulp 1.65 → 1.79, rounded_frac 0.571 → 0.956 |
| `bh/bf16/tanh_bw/default` | max_ulp 1.16 → 0.5, mean_ulp 0.525 → 0, rounded_frac 0.997 → 1 |
| `bh/bf16/tanhshrink_bw/default` | mean_ulp 3.52 → 6.1, rounded_frac 0.937 → 0.967 |
| `bh/fp32/bias_gelu_/default` | max_ulp 7.45e+40 → 1.27e+08, mean_ulp 6.51e+39 → 1.48e+07, ulp_clipped 35234 → 34793, rounded_frac 0.746 → 0.964 |
| `bh/fp32/cosh_bw/default` | defects 2 → 0 |
| `bh/fp32/logaddexp2_/default` | max_ulp 1.54e+09 → 1.06e+07, mean_ulp 1.05e+07 → 1.23e+05, ulp_clipped 46998 → 5609, defects 15488 → 0, rounded_frac 0.715 → 0.959 |
| `bh/fp32/logaddexp_/default` | max_ulp 5.23e+08 → 8e+06, mean_ulp 7.28e+06 → 9.77e+04, ulp_clipped 46910 → 3924, defects 15566 → 0, rounded_frac 0.726 → 0.975 |
| `bh/fp32/sinh_bw/default` | defects 2 → 0, usable_to 88.5 → 89.1 |
| `bh/fp32/tanh_bw/default` | max_ulp 6.59e+04 → 5, mean_ulp 7.51e+03 → 1.67, ulp_clipped 32756 → 0, rounded_frac 0.482 → 0.935, usable_to — → 0.199 |
| `wh/bf16/acosh_bw/default` | max_ulp 3.26 → 3.07, mean_ulp 0.688 → 0.64, rounded_frac 0.781 → 0.884 |
| `wh/bf16/asin_bw/default` | max_ulp 2.76 → 2.7, mean_ulp 0.787 → 0.702, usable_to 0.938 → 0.949 |
| `wh/bf16/asinh_bw/default` | max_ulp 1.33 → 1.2, mean_ulp 0.681 → 0.638, rounded_frac 0.918 → 0.954 |
| `wh/bf16/atan_bw/default` | mean_ulp 0.867 → 0.767, defects 2 → 0, rounded_frac 0.854 → 0.896 |
| `wh/bf16/atanh_bw/default` | max_ulp 8.17 → 7.76, mean_ulp 1.02 → 0.885, defects 2 → 0, rounded_frac 0.84 → 0.835, usable_to 0.82 → 0.918 |
| `wh/bf16/bias_gelu/default` | max_ulp 1.14e+36 → 179, mean_ulp 4.5e+34 → 86.3, ulp_clipped 30210 → 0, rounded_frac 0.749 → 0.996 |
| `wh/bf16/bias_gelu_/default` | max_ulp 1.14e+36 → 179, mean_ulp 4.5e+34 → 86.3, ulp_clipped 30210 → 0, rounded_frac 0.749 → 0.996 |
| `wh/bf16/cosh_bw/default` | defects 2 → 0, usable_to 88.5 → 89 |
| `wh/bf16/div_bw/default` | max_ulp 0.512 → 0.498, mean_ulp 0.512 → 0, rounded_frac 0.984 → 1 |
| `wh/bf16/erfinv_bw/default` | mean_ulp 1.28 → 1.2, usable_to 0.777 → 0.859 |
| `wh/bf16/hypot/default` | mean_ulp 1.58 → 1.64, rounded_frac 0.947 → 0.992 |
| `wh/bf16/log1p_bw/default` | mean_ulp 0.68 → 0.744, defects 2 → 0, rounded_frac 0.975 → 0.982, usable_to 8.47e+37 → 8.51e+37 |
| `wh/bf16/log_bw/default` | max_ulp 0.512 → 0.498, mean_ulp 0.507 → 0, defects 2 → 0, rounded_frac 0.984 → 1, usable_to 8.47e+37 → 8.51e+37 |
| `wh/bf16/log_sigmoid_bw/default` | max_ulp 1.91 → 1.71, mean_ulp 0.785 → 0.722, rounded_frac 0.972 → 0.976 |
| `wh/bf16/logaddexp/default` | max_ulp 8.39e+06 → 2.22e+05, mean_ulp 4.85e+03 → 44.7, ulp_clipped 16583 → 108, defects 15566 → 0, rounded_frac 0.858 → 0.986 |
| `wh/bf16/logaddexp_/default` | max_ulp 8.39e+06 → 2.22e+05, mean_ulp 4.85e+03 → 44.7, ulp_clipped 16583 → 108, defects 15566 → 0, rounded_frac 0.858 → 0.986 |
| `wh/bf16/logit_bw/default` | mean_ulp 0.649 → 0.729, rounded_frac 0.961 → 0.975 |
| `wh/bf16/logiteps_bw/default` | mean_ulp 0.649 → 0.729, rounded_frac 0.961 → 0.975 |
| `wh/bf16/mse_loss/default` | mean_ulp 1.65 → 1.79, rounded_frac 0.571 → 0.956 |
| `wh/bf16/reciprocal/default` | max_ulp 0.512 → 0.498, mean_ulp 0.507 → 0, defects 2 → 0, rounded_frac 0.984 → 1, usable_to 8.47e+37 → 8.51e+37 |
| `wh/bf16/reciprocal_bw/default` | max_ulp 1.51 → 1.23, mean_ulp 0.835 → 0.714, defects 256 → 254, rounded_frac 0.617 → 0.734 |
| `wh/bf16/sigmoid_bw/default` | max_ulp 266 → 127, mean_ulp 5.5 → 14.2, defects 329 → 141, rounded_frac 0.988 → 0.999, usable_to 1.77 → 12.4 |
| `wh/bf16/sinh_bw/default` | defects 2 → 0, usable_to 88.5 → 89 |
| `wh/bf16/softplus_bw/default` | max_ulp 1.91 → 0.857, mean_ulp 0.746 → 0.517, rounded_frac 0.982 → 0.994 |
| `wh/bf16/softsign_bw/default` | max_ulp 81 → 79, mean_ulp 3.29 → 3.74, rounded_frac 0.859 → 0.887 |
| `wh/bf16/sqrt_bw/default` | mean_ulp 0.695 → 0.699, rounded_frac 0.719 → 0.723 |
| `wh/bf16/square/default` | max_ulp 0.973 → 0.5, mean_ulp 0.688 → 0, rounded_frac 0.578 → 1 |
| `wh/bf16/squared_difference/default` | mean_ulp 1.65 → 1.79, rounded_frac 0.571 → 0.956 |
| `wh/bf16/squared_difference_/default` | mean_ulp 1.65 → 1.79, rounded_frac 0.571 → 0.956 |
| `wh/bf16/tanhshrink_bw/default` | mean_ulp 3.52 → 6.1, rounded_frac 0.937 → 0.967 |
| `wh/fp32/bias_gelu/default` | max_ulp 7.45e+40 → 1.22e+08, mean_ulp 6.51e+39 → 1.48e+07, ulp_clipped 35234 → 34797, rounded_frac 0.746 → 0.964 |
| `wh/fp32/bias_gelu_/default` | max_ulp 7.45e+40 → 1.22e+08, mean_ulp 6.51e+39 → 1.48e+07, ulp_clipped 35234 → 34797, rounded_frac 0.746 → 0.964 |
| `wh/fp32/cosh_bw/default` | defects 2 → 0 |
| `wh/fp32/logaddexp/default` | max_ulp 5.23e+08 → 8e+06, mean_ulp 7.28e+06 → 9.77e+04, ulp_clipped 46910 → 3924, defects 15566 → 0, rounded_frac 0.726 → 0.975 |
| `wh/fp32/logaddexp2/default` | max_ulp 1.54e+09 → 1.06e+07, mean_ulp 1.05e+07 → 1.23e+05, ulp_clipped 46998 → 5609, defects 15488 → 0, rounded_frac 0.715 → 0.959 |
| `wh/fp32/logaddexp2_/default` | max_ulp 1.54e+09 → 1.06e+07, mean_ulp 1.05e+07 → 1.23e+05, ulp_clipped 46998 → 5609, defects 15488 → 0, rounded_frac 0.715 → 0.959 |
| `wh/fp32/logaddexp_/default` | max_ulp 5.23e+08 → 8e+06, mean_ulp 7.28e+06 → 9.77e+04, ulp_clipped 46910 → 3924, defects 15566 → 0, rounded_frac 0.726 → 0.975 |
| `wh/fp32/sinh_bw/default` | defects 2 → 0, usable_to 88.5 → 89.1 |
| `wh/fp32/softplus_bw/default` | max_ulp 2.95 → 2.64, mean_ulp 1.34 → 1.31, rounded_frac 0.954 → 0.96, usable_to 0.00163 → 0.000345 |

## Expected (6)

No scored metric got worse on comparable measurements. Either `usable_to` moved, because a different kernel puts the 2 ULP boundary on a neighbouring group; or `n_inputs` moved, which means the swept domain changed and the two runs measure different populations; or a metric is reported here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/digamma_bw/default` | usable_to 1 → 0.996 |
| `bh/bf16/expm1_bw/default` | usable_to 33.2 → 33.5 |
| `bh/bf16/hardswish_bw/default` | usable_to 1.12 → 1.13 |
| `bh/bf16/multigammaln_bw/default` | usable_to 5.55e-17 → 5.59e-17 |
| `bh/bf16/silu_bw/default` | usable_to 0.867 → 0.863 |
| `bh/fp32/silu_bw/default` | usable_to 2.96e-07 → 2.98e-07 |

## No longer measured (84)

Present in the baseline, absent now.

| Variant | Moved |
|---|---|
| `bh/bf16/atan2_bw/default` | — |
| `bh/bf16/bias_gelu/default` | — |
| `bh/bf16/eq/default` | — |
| `bh/bf16/ge/default` | — |
| `bh/bf16/gt/default` | — |
| `bh/bf16/hypot_bw/default` | — |
| `bh/bf16/ldexp/default` | — |
| `bh/bf16/ldexp_bw/default` | — |
| `bh/bf16/le/default` | — |
| `bh/bf16/lerp_bw/default` | — |
| `bh/bf16/logaddexp/default` | — |
| `bh/bf16/logaddexp2/default` | — |
| `bh/bf16/logaddexp2_bw/default` | — |
| `bh/bf16/logaddexp_bw/default` | — |
| `bh/bf16/logical_and/default` | — |
| `bh/bf16/logical_or/default` | — |
| `bh/bf16/lt/default` | — |
| `bh/bf16/max_bw/default` | — |
| `bh/bf16/maximum/default` | — |
| `bh/bf16/min_bw/default` | — |
| `bh/bf16/minimum/default` | — |
| `bh/bf16/mul_bw/default` | — |
| `bh/bf16/ne/default` | — |
| `bh/bf16/rdiv_bw/scalar2.0` | — |
| `bh/bf16/remainder/default` | — |
| `bh/bf16/rpow_bw/exponent2.0` | — |
| `bh/bf16/squared_difference_bw/default` | — |
| `bh/bf16/xlogy/default` | — |
| `bh/bf16/xlogy_bw/default` | — |
| `bh/fp32/atan2_bw/default` | — |
| `bh/fp32/bias_gelu/default` | — |
| `bh/fp32/eq/default` | — |
| `bh/fp32/ge/default` | — |
| `bh/fp32/gt/default` | — |
| `bh/fp32/hypot_bw/default` | — |
| `bh/fp32/ldexp/default` | — |
| `bh/fp32/ldexp_bw/default` | — |
| `bh/fp32/le/default` | — |
| `bh/fp32/lerp_bw/default` | — |
| `bh/fp32/logaddexp/default` | — |
| `bh/fp32/logaddexp2/default` | — |
| `bh/fp32/logaddexp2_bw/default` | — |
| `bh/fp32/logaddexp_bw/default` | — |
| `bh/fp32/logical_and/default` | — |
| `bh/fp32/logical_or/default` | — |
| `bh/fp32/lt/default` | — |
| `bh/fp32/max_bw/default` | — |
| `bh/fp32/maximum/default` | — |
| `bh/fp32/min_bw/default` | — |
| `bh/fp32/minimum/default` | — |
| `bh/fp32/mul_bw/default` | — |
| `bh/fp32/ne/default` | — |
| `bh/fp32/rdiv_bw/scalar2.0` | — |
| `bh/fp32/remainder/default` | — |
| `bh/fp32/rpow_bw/exponent2.0` | — |
| `bh/fp32/squared_difference_bw/default` | — |
| `bh/fp32/xlogy/default` | — |
| `bh/fp32/xlogy_bw/default` | — |
| `wh/bf16/atan2_bw/default` | — |
| `wh/bf16/hypot_bw/default` | — |
| `wh/bf16/ldexp_bw/default` | — |
| `wh/bf16/lerp_bw/default` | — |
| `wh/bf16/logaddexp2_bw/default` | — |
| `wh/bf16/logaddexp_bw/default` | — |
| `wh/bf16/max_bw/default` | — |
| `wh/bf16/min_bw/default` | — |
| `wh/bf16/mul_bw/default` | — |
| `wh/bf16/rdiv_bw/scalar2.0` | — |
| `wh/bf16/rpow_bw/exponent2.0` | — |
| `wh/bf16/squared_difference_bw/default` | — |
| `wh/bf16/xlogy_bw/default` | — |
| `wh/fp32/atan2_bw/default` | — |
| `wh/fp32/hypot_bw/default` | — |
| `wh/fp32/ldexp_bw/default` | — |
| `wh/fp32/lerp_bw/default` | — |
| `wh/fp32/logaddexp2_bw/default` | — |
| `wh/fp32/logaddexp_bw/default` | — |
| `wh/fp32/max_bw/default` | — |
| `wh/fp32/min_bw/default` | — |
| `wh/fp32/mul_bw/default` | — |
| `wh/fp32/rdiv_bw/scalar2.0` | — |
| `wh/fp32/rpow_bw/exponent2.0` | — |
| `wh/fp32/squared_difference_bw/default` | — |
| `wh/fp32/xlogy_bw/default` | — |
