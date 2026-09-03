# Findings

_812427269cb, f9377427c7f → 72023534b03, 812427269cb_

## Improvements (2)

A scored metric got better.

| Variant | Moved |
|---|---|
| `bh/fp32/lgamma/default` | ulp_clipped 314 → 313, defects None → 0 |
| `bh/fp32/multigammaln/default` | ulp_clipped 2989 → 2987, defects None → 7422, usable_to 2.98e-08 → 5.55e-17 |

## Expected (98)

Only `usable_to` moved: a different kernel puts the 2 ULP boundary on a neighbouring group. No scored metric changed.

| Variant | Moved |
|---|---|
| `bh/bf16/acosh_bw/default` | defects None → 15874 |
| `bh/bf16/asin/default` | defects None → 2, usable_to 1 → — |
| `bh/bf16/asinh_bw/default` | defects None → 15874, usable_to 8.51e+37 → 1.84e+19 |
| `bh/bf16/atan/default` | defects None → 2, usable_to 3.39e+38 → — |
| `bh/bf16/atan_bw/default` | defects None → 2, usable_to 9.22e+18 → 9.19e+18 |
| `bh/bf16/atanh/default` | defects None → 2, usable_to 0.996 → — |
| `bh/bf16/atanh_bw/default` | defects None → 2 |
| `bh/bf16/cos/default` | defects None → 21274 |
| `bh/bf16/cos_bw/default` | defects None → 21275 |
| `bh/bf16/cosh_bw/default` | defects None → 2, usable_to 89 → 88.5 |
| `bh/bf16/deg2rad/default` | defects None → 2, usable_to 3.39e+38 → 6.7e-37 |
| `bh/bf16/digamma_bw/default` | defects None → 263, usable_to 5.31 → 0.996 |
| `bh/bf16/erfinv/default` | defects None → 29424, usable_to 0.000515 → 1.31e-38 |
| `bh/bf16/exp2_bw/default` | defects None → 1, usable_to 3.39e+38 → 128 |
| `bh/bf16/expm1_bw/default` | defects None → 331 |
| `bh/bf16/gelu/default` | defects None → 86, usable_to 4.81 → 2.33e-38 |
| `bh/bf16/gelu/fast_approx` | defects None → 198 |
| `bh/bf16/hardswish/default` | defects None → 2, usable_to 3.39e+38 → 2.33e-38 |
| `bh/bf16/hypot/default` | defects None → 16384 |
| `bh/bf16/i0_bw/default` | defects None → 4, usable_to 88.5 → 2.33e-38 |
| `bh/bf16/i1/default` | defects None → 4, usable_to 88.5 → 2.33e-38 |
| `bh/bf16/log1p_bw/default` | defects None → 2, usable_to 8.51e+37 → 8.47e+37 |
| `bh/bf16/log2_bw/default` | defects None → 116, usable_to 1.22e+38 → — |
| `bh/bf16/log_bw/default` | defects None → 2, usable_to 8.51e+37 → 8.47e+37 |
| `bh/bf16/logaddexp/default` | defects None → 15566 |
| `bh/bf16/logaddexp2/default` | defects None → 15488 |
| `bh/bf16/logaddexp2_/default` | defects None → 15488 |
| `bh/bf16/logaddexp_/default` | defects None → 15566 |
| `bh/bf16/mish/default` | defects None → 11, usable_to 3.39e+38 → 1.95e-38 |
| `bh/bf16/multigammaln/default` | defects None → 11536, usable_to 0.00243 → 5.59e-17 |
| `bh/bf16/polygamma/k1` | defects None → 263, usable_to 5.31 → 0.996 |
| `bh/bf16/polygamma_bw/n1` | defects None → 75 |
| `bh/bf16/prelu/weight0.25` | defects None → 1, usable_to 3.39e+38 → 4.68e-38 |
| `bh/bf16/rdiv/value2.0` | defects None → 258, usable_to 1.7e+38 → 8.47e+37 |
| `bh/bf16/rdiv_bw/scalar2.0` | defects None → 256, usable_to 1.3e+19 → 7.67e-20 |
| `bh/bf16/reciprocal/default` | defects None → 2, usable_to 8.51e+37 → 8.47e+37 |
| `bh/bf16/reciprocal_bw/default` | defects None → 256, usable_to 9.22e+18 → 5.42e-20 |
| `bh/bf16/rpow_bw/exponent2.0` | defects None → 32384, usable_to 1.69e+38 → 1.18e-38 |
| `bh/bf16/rsqrt_bw/default` | defects None → 75, usable_to 3.43e-26 → — |
| `bh/bf16/selu_bw/default` | defects None → 1 |
| `bh/bf16/sigmoid_bw/default` | defects None → 331 |
| `bh/bf16/silu/default` | defects None → 13, usable_to 3.39e+38 → 2.33e-38 |
| `bh/bf16/silu_bw/default` | defects None → 9 |
| `bh/bf16/sin/default` | defects None → 21275 |
| `bh/bf16/sin_bw/default` | defects None → 21274 |
| `bh/bf16/sinh_bw/default` | defects None → 2, usable_to 89 → 88.5 |
| `bh/bf16/softcap/beta50.0` | defects None → 1426, usable_to 3.39e+38 → — |
| `bh/bf16/softplus/default` | defects None → 526, usable_to 3.39e+38 → 5.03 |
| `bh/bf16/softsign/default` | defects None → 512, usable_to 3.39e+38 → 8.47e+37 |
| `bh/bf16/swish/default` | defects None → 13, usable_to 3.39e+38 → 2.33e-38 |
| `bh/bf16/tan/default` | defects None → 21634 |
| `bh/bf16/tan_bw/default` | defects None → 22759 |
| `bh/bf16/tanh/default` | defects None → 2, usable_to 3.39e+38 → — |
| `bh/bf16/xielu/default` | defects None → 1, usable_to 3.39e+38 → 2.33e-38 |
| `bh/fp32/acosh_bw/default` | defects None → 15874 |
| `bh/fp32/asinh_bw/default` | defects None → 15874, usable_to 8.51e+37 → 1.84e+19 |
| `bh/fp32/atan_bw/default` | defects None → 2 |
| `bh/fp32/atanh_bw/default` | defects None → 2 |
| `bh/fp32/cos/default` | defects None → 18190 |
| `bh/fp32/cos_bw/default` | defects None → 18190 |
| `bh/fp32/cosh_bw/default` | defects None → 2, usable_to 89 → 88.5 |
| `bh/fp32/digamma_bw/default` | defects None → 257 |
| `bh/fp32/erfinv/default` | defects None → 29420, usable_to 0.000511 → 1.32e-38 |
| `bh/fp32/exp2_bw/default` | defects None → 1, usable_to 3.39e+38 → 128 |
| `bh/fp32/expm1_bw/default` | defects None → 138 |
| `bh/fp32/gelu/default` | defects None → 83 |
| `bh/fp32/gelu/fast_approx` | defects None → 197 |
| `bh/fp32/hypot/default` | defects None → 16384 |
| `bh/fp32/hypot_bw/default` | defects None → 16384 |
| `bh/fp32/log1p_bw/default` | defects None → 2 |
| `bh/fp32/log2_bw/default` | defects None → 112, usable_to 1.6e+31 → — |
| `bh/fp32/log_bw/default` | defects None → 2 |
| `bh/fp32/logaddexp/default` | defects None → 15566 |
| `bh/fp32/logaddexp2/default` | defects None → 15488 |
| `bh/fp32/logaddexp2_/default` | defects None → 15488 |
| `bh/fp32/logaddexp_/default` | defects None → 15566 |
| `bh/fp32/mish/default` | defects None → 9 |
| `bh/fp32/polygamma/k1` | defects None → 257 |
| `bh/fp32/polygamma_bw/n1` | defects None → 75 |
| `bh/fp32/rdiv/value2.0` | defects None → 258, usable_to 1.7e+38 → 8.47e+37 |
| `bh/fp32/rdiv_bw/scalar2.0` | defects None → 254, usable_to 4.49e+15 → 7.67e-20 |
| `bh/fp32/reciprocal/default` | defects None → 2 |
| `bh/fp32/reciprocal_bw/default` | defects None → 256, usable_to 4.49e+15 → 5.42e-20 |
| `bh/fp32/rpow/exponent2.0` | defects None → 1536, usable_to 3.39e+38 → 8.28e+34 |
| `bh/fp32/rpow_bw/exponent2.0` | defects None → 32384, usable_to 1.69e+38 → 1.18e-38 |
| `bh/fp32/rsqrt_bw/default` | defects None → 75, usable_to 2.04e-26 → — |
| `bh/fp32/selu_bw/default` | defects None → 1 |
| `bh/fp32/sigmoid_bw/default` | defects None → 140 |
| `bh/fp32/silu/default` | defects None → 9 |
| `bh/fp32/silu_bw/default` | defects None → 9 |
| `bh/fp32/sin/default` | defects None → 18190 |
| `bh/fp32/sin_bw/default` | defects None → 18190 |
| `bh/fp32/sinh_bw/default` | defects None → 2, usable_to 89 → 88.5 |
| `bh/fp32/softplus/default` | defects None → 30 |
| `bh/fp32/softsign/default` | defects None → 512 |
| `bh/fp32/swish/default` | defects None → 9 |
| `bh/fp32/tan/default` | defects None → 20110 |
| `bh/fp32/tan_bw/default` | defects None → 20376 |
