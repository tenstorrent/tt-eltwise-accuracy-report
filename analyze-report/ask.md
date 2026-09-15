# Ask an assistant about these measurements

Give your assistant this page's URL, or paste the file, then ask in plain language —
"is `exp` usable on Wormhole in bfloat16?", "which ops are worse than 2 ULP in fp32?".
Everything needed to answer is below: the definitions, then every measured result.

Answering rules, in force for whoever reads this: quote the `Verdict` column rather than
judging the numbers yourself, name the architecture and dtype in every answer, and if a
variant is not in the table say so instead of extrapolating from a neighbouring one. A timing
is only comparable to another taken on the same host, which the "Measured against" table names.


## Measured against

| Arch | Dtype | tt-metal | ttnn | Timed on |
|---|---|---|---|---|
| bh | bf16 | `fbf7d27db93` | 0.75.0rc10.dev916+ga5cd86212e1 | yyzo-bh-26-special-ijankowski-for-reservation-207161 |
| bh | fp32 | `fbf7d27db93` | 0.75.0rc10.dev916+ga5cd86212e1 | yyzo-bh-26-special-ijankowski-for-reservation-207161 |
| wh | bf16 | `9f9cd4fd590` | 0.77.0 | c38e6ad1d8cf |
| wh | fp32 | `9f9cd4fd590` | 0.77.0 | c38e6ad1d8cf |

**Status: in force.** Every field below exists in `report_index.json` today.

The definitions behind every number in the report. An assistant answering from it uses
these and no others. The verdict rules have a code twin in `report/charts.py::verdict`;
the two change together.

## Fields, per arch/dtype/op/variant

| Field | Definition |
|---|---|
| `max_ulp` | worst defined ULP error: `\|y_ref − y\| / ULP(y_ref)`, reference in fp64, ULP by tt-metal's definition at the compute dtype |
| `mean_ulp` | mean over points with a defined, non-zero ULP. `—` means every point was exact |
| `usable_to` | largest \|x\| below which no point exceeds 2 ULP and none is a defect. Unary only; `—` elsewhere, where x alone does not determine the output |
| `max_abs` | worst absolute error |
| `ulp_clipped` | points past the chart clamp of 1000 ULP |
| `n_inputs` | measured points. An fp32 row is the worst point of 2¹⁶ consecutive codes |
| `outcomes` | count per label, below |
| `specials` | device vs golden at ±0, ±inf, NaN, ±min-normal. Printed values, no ULP |
| `defects` | device returned inf or zero where the reference is representable. These carry no ULP, so no other field counts them. A `mismatch` against a NaN reference is a domain disagreement, not a defect |
| `verdict` | one of the seven phrases below, precomputed |
| `perf` | `us_median` and `melem_per_s` for one dispatch over 2²⁴ resident elements, with the `host` that took them. `spread_pct` appears only when the row is not trustworthy. Never scored |
| `rationale` | why this variant has these parameters, from `ops/overrides.py` |
| `_runs.{arch}.{dtype}` | tt-metal commit, device, versions. Every claim is per build |

## Outcome labels

| Label | Meaning |
|---|---|
| `exact` | bit-identical to the reference |
| `inexact` | differs, ULP defined |
| `flushed` | reference is subnormal, the dtype cannot hold it |
| `zeroed` | hardware returned 0 for a representable reference. Real, but ULP cannot express it |
| `overflow` | true result exceeds the dtype |
| `undefined` | reference is NaN, outside the mathematical domain |
| `mismatch` | one side finite, the other not. A defect |
| `special` | a special-values row, excluded from every statistic |

## Verdict rules

Computed, never inferred. First match wins.

| Verdict | Rule |
|---|---|
| `N of M points returned inf or zero where a value exists` | defects > 0. Takes precedence, because those points carry no ULP and the figures below exclude them |
| `bit-exact` | max_ulp = 0 |
| `within 2 ULP everywhere` | max_ulp ≤ 2 |
| `accurate to \|x\| <= B; up to M ULP beyond` | unary, usable_to = B defined |
| `never within 2 ULP; mean X, worst M` | unary, no point within 2 ULP |
| `worst pairing M ULP; mean X` | multi-operand, maximum over sampled pairings |
| `no scorable points` | no defined ULP anywhere |

## Answering rules

1. Cite the entry as arch/dtype/op/variant, with its tt-metal commit.
2. Never average across architectures or dtypes. Never extrapolate to an unmeasured cell.
3. A sampled maximum is a lower bound. Binary fp32 and every ternary sweep are sampled, and
   the page says so. `ttnn-accuracy refine` sweeps the cell the sample drew from:
   `fp32/pow` publishes 5,918 ULP and holds 8,300.
4. Never compare a timing across hosts. A timing difference is not a regression.
5. Some wrongness carries no ULP and is in no field here. `ttnn-accuracy check` also requires
   that the answer not change with the tiling, that an in-place op write to its own operand,
   that aliased operands agree with distinct ones, and that an op commute wherever its
   reference does. Only a check run reports these.
6. Not in the index means not measured. Read the reason from the manifest's exclusions and
   refusals, or from [uncovered.md](uncovered.md).

## Results — 865 variants

| Arch | Dtype | Op | Parameters | Verdict | Max ULP | Mean ULP | Usable to | µs | Melem/s |
|---|---|---|---|---|---|---|---|---|---|
| bh | bf16 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | 199 | 84392 |
| bh | bf16 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 648 | 25882 |
| bh | bf16 | `acos` | `default` | bit-exact | 0 | — | 1 | 280 | 59879 |
| bh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.953; up to 3 ULP beyond | 3 | 1.1 | 0.953 | 3902 | 4300 |
| bh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1 | 0.994 | 3.39e+38 | 355 | 47240 |
| bh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 3 | 1.01 | 1.05 | 4484 | 3741 |
| bh | bf16 | `add` | `default` | within 2 ULP everywhere | 1 | 1 | — | 272 | 61765 |
| bh | bf16 | `add_` | `default` | within 2 ULP everywhere | 1 | 1 | — | 272 | 61775 |
| bh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | 272 | 61648 |
| bh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.21e+03 | 2.11e+06 | 2.21e+03 | — | 360 | 46581 |
| bh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.6 | 126 | 19.6 | — | 358 | 46810 |
| bh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists | 0 | — | — | 277 | 60638 |
| bh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 3 ULP beyond | 3 | 1.1 | 0.938 | 3795 | 4421 |
| bh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1 | 0.998 | 3.39e+38 | 482 | 34774 |
| bh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1 | 1 | 1.84e+19 | 752 | 22312 |
| bh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | 201 | 83381 |
| bh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 3.85 | 200 | 3.85 | — | 278 | 60398 |
| bh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.92 | 248 | 4.92 | — | 2807 | 5977 |
| bh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 2 | 1.04 | 9.19e+18 | 656 | 25581 |
| bh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists | 2 | 0.997 | — | 260 | 64573 |
| bh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 8 | 1.13 | 0.82 | 4286 | 3914 |
| bh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 272 | 61673 |
| bh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 272 | 61651 |
| bh | bf16 | `cbrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 219 | 76682 |
| bh | bf16 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | 199 | 84101 |
| bh | bf16 | `celu` | `default` | bit-exact | 0 | — | 3.39e+38 | 189 ±8% | 88581 |
| bh | bf16 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | 1396 | 12022 |
| bh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 199 | 84212 |
| bh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | — | — | 1246 | 13464 |
| bh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 199 | 84127 |
| bh | bf16 | `clip_bw` | `default` | bit-exact | 0 | — | — | 1248 | 13444 |
| bh | bf16 | `cos` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | 259 | 64722 |
| bh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | 875 | 19163 |
| bh | bf16 | `cosh` | `default` | within 2 ULP everywhere | 1 | 1 | 89 | 280 | 59823 |
| bh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0 | — | 88.5 | 3482 | 4818 |
| bh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 0.992 | 6.7e-37 | 200 | 84008 |
| bh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 383 | 43826 |
| bh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.36e+04 | 0.996 | 3486 | 4813 |
| bh | bf16 | `div` | `default` | bit-exact | 0 | — | — | 277 | 60625 |
| bh | bf16 | `div_bw` | `default` | bit-exact | 0 | — | — | 5436 | 3086 |
| bh | bf16 | `div_no_nan` | `default` | bit-exact | 0 | — | — | 725 | 23149 |
| bh | bf16 | `divide` | `default` | bit-exact | 0 | — | — | 277 | 60551 |
| bh | bf16 | `divide_` | `default` | bit-exact | 0 | — | — | 276 | 60748 |
| bh | bf16 | `elu` | `default` | bit-exact | 0 | — | 3.39e+38 | 193 ±5% | 87149 |
| bh | bf16 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | 1394 | 12036 |
| bh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 198 ±7% | 84606 |
| bh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | 1236 | 13569 |
| bh | bf16 | `erfc` | `default` | accurate to |x| <= 2.5; up to 3.2e+28 ULP beyond | 3.2e+28 | 3.23e+24 | 2.5 | 303 | 55436 |
| bh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | 1234 | 13599 |
| bh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists | 121 | 6.53 | 1.31e-38 | 369 | 45498 |
| bh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 9 ULP beyond | 9 | 1.39 | 0.777 | 3994 | 4200 |
| bh | bf16 | `exp` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | 206 | 81458 |
| bh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6 | 6 | 3.05 | — | 204 | 82142 |
| bh | bf16 | `exp2` | `default` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | 235 | 71501 |
| bh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.05 | 128 | 891 | 18829 |
| bh | bf16 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | 660 ±9% | 25415 |
| bh | bf16 | `expm1` | `default` | bit-exact | 0 | — | 3.39e+38 | 288 ±6% | 58281 |
| bh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists | 125 | 5.83 | 2.08 | 943 | 17794 |
| bh | bf16 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | 204 | 82195 |
| bh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | — | 1284 | 13062 |
| bh | bf16 | `fmod` | `default` | worst pairing 9.14e+35 ULP; mean 8.94e+34 | 9.14e+35 | 8.94e+34 | — | 292 ±5% | 57369 |
| bh | bf16 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | 204 | 82291 |
| bh | bf16 | `ge_` | `default` | bit-exact | 0 | — | — | 277 ±7% | 60561 |
| bh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists | 165 | 9.42 | 2.33e-38 | 331 | 50628 |
| bh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists | 1.14e+36 | 1.82e+34 | 2.33e-38 | 204 ±9% | 82171 |
| bh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.38; up to 3 ULP beyond | 3 | 1.52 | 8.38 | 709 | 23668 |
| bh | bf16 | `gt_` | `default` | bit-exact | 0 | — | — | 277 | 60516 |
| bh | bf16 | `hardmish` | `default` | within 2 ULP everywhere | 1 | 0.993 | 3.39e+38 | 204 | 82265 |
| bh | bf16 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | 205 | 82008 |
| bh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 849 | 19768 |
| bh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.661 | 3.39e+38 | 205 | 81951 |
| bh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1311 | 12801 |
| bh | bf16 | `hardswish` | `default` | 2 of 65024 points returned inf or zero where a value exists | 2 | 1.01 | 2.33e-38 | 211 | 79366 |
| bh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.13; up to 85 ULP beyond | 85 | 2.36 | 1.13 | 1854 | 9052 |
| bh | bf16 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | 205 | 81750 |
| bh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1114 | 15063 |
| bh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | 206 | 81459 |
| bh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 53 | 1.46 | — | 290 | 57939 |
| bh | bf16 | `hypot_bw` | `default` | worst pairing 75 ULP; mean 2.44 | 75 | 2.44 | — | 1714 | 9788 |
| bh | bf16 | `i0` | `default` | accurate to |x| <= 13.9; up to 255 ULP beyond | 255 | 56.4 | 13.9 | 297 | 56572 |
| bh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.95 | 2.33e-38 | 891 | 18833 |
| bh | bf16 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | 205 | 81986 |
| bh | bf16 | `l1_loss` | `default` | within 2 ULP everywhere | 1 | 1 | — | 277 | 60555 |
| bh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | 281 | 59685 |
| bh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | 281 | 59693 |
| bh | bf16 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 1300 | 12908 |
| bh | bf16 | `le_` | `default` | bit-exact | 0 | — | — | 278 | 60405 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 206 | 81608 |
| bh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 950 | 17662 |
| bh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 40 | 1.77e+03 | 40 | — | 365 | 45930 |
| bh | bf16 | `lerp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 1438 | 11669 |
| bh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.439; up to 324 ULP beyond | 324 | 1.17 | 0.439 | 835 | 20094 |
| bh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 853 | 19660 |
| bh | bf16 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 242 | 69214 |
| bh | bf16 | `log10` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 255 | 65871 |
| bh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.69e+37 | 2694 | 6227 |
| bh | bf16 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 260 | 64404 |
| bh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 2685 | 6248 |
| bh | bf16 | `log2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 256 | 65586 |
| bh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists | 2 | 1.02 | — | 2702 | 6208 |
| bh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0 | — | 8.47e+37 | 2056 | 8161 |
| bh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 4.12; up to 6 ULP beyond | 6 | 1.51 | 4.12 | 218 | 76929 |
| bh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 2 | 0.895 | 3.39e+38 | 2957 | 5674 |
| bh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 304 | 55102 |
| bh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 372 | 45081 |
| bh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 377 ±12% | 44451 |
| bh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41 ULP; mean 3.82 | 41 | 3.82 | — | 2733 | 6140 |
| bh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 303 | 55352 |
| bh | bf16 | `logaddexp_bw` | `default` | worst pairing 62 ULP; mean 4.88 | 62 | 4.88 | — | 2391 | 7016 |
| bh | bf16 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | 202 | 83030 |
| bh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | — | — | 280 | 59935 |
| bh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.95 | 0.395 | 323 ±8% | 51872 |
| bh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | 3574 | 4694 |
| bh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | 4247 | 3951 |
| bh | bf16 | `lt_` | `default` | bit-exact | 0 | — | — | 277 | 60542 |
| bh | bf16 | `mac` | `default` | worst pairing 64 ULP; mean 26.4 | 64 | 26.4 | — | 366 | 45831 |
| bh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 2806 | 5979 |
| bh | bf16 | `maximum` | `default` | bit-exact | 0 | — | — | 277 | 60555 |
| bh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 2803 | 5985 |
| bh | bf16 | `minimum` | `default` | bit-exact | 0 | — | — | 273 | 61525 |
| bh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists | 1 | 0.997 | 1.95e-38 | 249 | 67461 |
| bh | bf16 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | 273 | 61553 |
| bh | bf16 | `mul_bw` | `default` | bit-exact | 0 | — | — | 734 | 22862 |
| bh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists | 724 | 1.81 | 5.59e-17 | 4819 | 3482 |
| bh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.59e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 294 | 5.59e-17 | 4094 | 4098 |
| bh | bf16 | `multiply` | `default` | bit-exact | 0 | — | — | 278 | 60244 |
| bh | bf16 | `multiply_` | `default` | bit-exact | 0 | — | — | 278 | 60288 |
| bh | bf16 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | 205 | 81791 |
| bh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | — | 1629 | 10300 |
| bh | bf16 | `polygamma` | `k=1` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.36e+04 | 0.996 | 333 | 50401 |
| bh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 1.21e+06 | 4.47 | 3553 | 4722 |
| bh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | — | 386 | 43429 |
| bh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | 1313 | 12775 |
| bh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists | 0 | — | 4.68e-38 | 206 | 81630 |
| bh | bf16 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | 205 | 81868 |
| bh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 0 | — | 8.47e+37 | 204 | 82073 |
| bh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists | 2 | 1.02 | 7.67e-20 | 3510 | 4780 |
| bh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0 | — | 8.47e+37 | 205 | 81642 |
| bh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 2 | 1.02 | 5.42e-20 | 2629 | 6382 |
| bh | bf16 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | 206 | 81565 |
| bh | bf16 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | 205 | 81756 |
| bh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2031 | 8260 |
| bh | bf16 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 663 ±11% | 25306 |
| bh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 205 | 81734 |
| bh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 205 | 81755 |
| bh | bf16 | `remainder` | `default` | worst pairing 1.08e+36 ULP; mean 1.22e+35 | 1.08e+36 | 1.22e+35 | — | 292 ±20% | 57469 |
| bh | bf16 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | 204 | 82260 |
| bh | bf16 | `rpow` | `exponent=2.0` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | 381 ±19% | 44020 |
| bh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | 1510 | 11108 |
| bh | bf16 | `rsqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | 251 ±20% | 66954 |
| bh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists | 3 | 1.28 | — | 3088 | 5433 |
| bh | bf16 | `rsub` | `default` | within 2 ULP everywhere | 1 | 1 | — | 277 | 60604 |
| bh | bf16 | `rsub_` | `default` | within 2 ULP everywhere | 1 | 1 | — | 277 | 60469 |
| bh | bf16 | `selu` | `default` | bit-exact | 0 | — | 3.39e+38 | 197 | 85220 |
| bh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 3 | 1.04 | 0.582 | 1605 | 10452 |
| bh | bf16 | `sigmoid` | `default` | within 2 ULP everywhere | 1 | 0.886 | 3.39e+38 | 288 ±5% | 58300 |
| bh | bf16 | `sigmoid_accurate` | `default` | within 2 ULP everywhere | 1 | 0.886 | 3.39e+38 | 287 ±12% | 58400 |
| bh | bf16 | `sigmoid_bw` | `default` | 331 of 65024 points returned inf or zero where a value exists | 125 | 4.97 | 1.86 | 1196 | 14032 |
| bh | bf16 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | 205 | 81902 |
| bh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | 298 | 56247 |
| bh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 285 | 1.54 | 0.887 | 1635 | 10261 |
| bh | bf16 | `sin` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | 241 | 69654 |
| bh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | 716 | 23418 |
| bh | bf16 | `sinh` | `default` | bit-exact | 0 | — | 89 | 282 ±8% | 59436 |
| bh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1 | 1 | 88.5 | 3119 | 5379 |
| bh | bf16 | `softcap` | `beta=50.0` | 1426 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | 235 | 71369 |
| bh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists | 1 | 1 | 5.03 | 251 ±18% | 66964 |
| bh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 2 | 0.861 | 3.39e+38 | 2239 | 7495 |
| bh | bf16 | `softshrink` | `default` | within 2 ULP everywhere | 1 | 0.996 | 3.39e+38 | 210 | 80061 |
| bh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1110 | 15110 |
| bh | bf16 | `softsign` | `default` | 512 of 65024 points returned inf or zero where a value exists | 1 | 0.642 | 8.47e+37 | 205 | 81998 |
| bh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00491; up to 81 ULP beyond | 81 | 3.34 | 0.00491 | 685 | 24482 |
| bh | bf16 | `sqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | 236 | 71217 |
| bh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 3365 | 4986 |
| bh | bf16 | `square` | `default` | within 2 ULP everywhere | 1 | 0.991 | 1.84e+19 | 205 | 81817 |
| bh | bf16 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | 678 | 24752 |
| bh | bf16 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | 278 | 60389 |
| bh | bf16 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | 278 | 60368 |
| bh | bf16 | `squared_difference_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 1140 | 14721 |
| bh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | 277 | 60565 |
| bh | bf16 | `subtract` | `default` | within 2 ULP everywhere | 1 | 1 | — | 275 | 60898 |
| bh | bf16 | `subtract_` | `default` | within 2 ULP everywhere | 1 | 1 | — | 277 | 60481 |
| bh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | 299 ±16% | 56165 |
| bh | bf16 | `tan` | `default` | 21634 of 65024 points returned inf or zero where a value exists | 3.3e+38 | 1.16e+36 | 1.31e+05 | 293 | 57168 |
| bh | bf16 | `tan_bw` | `default` | 22759 of 65024 points returned inf or zero where a value exists | 2.53e+38 | 7.9e+35 | 1.31e+05 | 1128 | 14879 |
| bh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | 204 | 82077 |
| bh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 2.23 | 17.2 | 635 | 26420 |
| bh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.5 | 1.35e-08 | 301 ±9% | 55734 |
| bh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.71 | 7.45e-09 | 862 | 19473 |
| bh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | 205 | 81871 |
| bh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 851 | 19713 |
| bh | bf16 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | 200 | 83894 |
| bh | bf16 | `where` | `default` | bit-exact | 0 | — | — | 360 | 46544 |
| bh | bf16 | `xielu` | `default` | 1 of 56847 points returned inf or zero where a value exists | 1 | 1 | 2.33e-38 | 417 | 40189 |
| bh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | — | 280 | 59909 |
| bh | bf16 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 4589 | 3656 |
| bh | fp32 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | 379 | 44237 |
| bh | fp32 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1258 | 13333 |
| bh | fp32 | `acos` | `default` | within 2 ULP everywhere | 2 | 1 | 1 | 465 | 36066 |
| bh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | 7575 | 2215 |
| bh | fp32 | `acosh` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | 441 | 38036 |
| bh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 724 | 1.12 | 0.996 | 8665 | 1936 |
| bh | fp32 | `add` | `default` | bit-exact | 0 | — | — | 520 | 32284 |
| bh | fp32 | `add_` | `default` | bit-exact | 0 | — | — | 518 | 32379 |
| bh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0 | — | — | 521 | 32225 |
| bh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | — | 700 | 23982 |
| bh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | — | 698 | 24024 |
| bh | fp32 | `asin` | `default` | within 2 ULP everywhere | 2 | 1.08 | 1 | 447 | 37498 |
| bh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | 7432 | 2257 |
| bh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | 583 | 28798 |
| bh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 2 | 1.03 | 1.84e+19 | 1318 | 12730 |
| bh | fp32 | `atan` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.39e+38 | 455 | 36875 |
| bh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | — | 534 | 31425 |
| bh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 1.66e+05 | 1.64e+07 | 1.66e+05 | — | 5439 | 3085 |
| bh | fp32 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 9.02e+04 | 3.25e+03 | 4.08e+03 | 1266 | 13249 |
| bh | fp32 | `atanh` | `default` | accurate to |x| <= 0.00772; up to 3 ULP beyond | 3 | 1.51 | 0.00772 | 603 | 27803 |
| bh | fp32 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 9.02e+04 | 3.31e+03 | 0.855 | 8381 | 2002 |
| bh | fp32 | `bias_gelu` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | 521 | 32204 |
| bh | fp32 | `bias_gelu_` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | 519 | 32306 |
| bh | fp32 | `cbrt` | `default` | accurate to |x| <= 5.77e-38; up to 3 ULP beyond | 3 | 1.82 | 5.77e-38 | 411 | 40842 |
| bh | fp32 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | 379 | 44289 |
| bh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 427 | 39248 |
| bh | fp32 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 2764 | 6071 |
| bh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 379 | 44305 |
| bh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | — | — | 2418 | 6939 |
| bh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 379 | 44273 |
| bh | fp32 | `clip_bw` | `default` | bit-exact | 0 | — | — | 2415 | 6948 |
| bh | fp32 | `cos` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | 419 | 40069 |
| bh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | 1658 | 10117 |
| bh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1 | 1 | 89 | 439 | 38180 |
| bh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 2 | 1.12 | 88.5 | 6744 | 2488 |
| bh | fp32 | `deg2rad` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 379 | 44240 |
| bh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 712 | 23554 |
| bh | fp32 | `digamma_bw` | `default` | 257 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 5.4e-20 | 7188 | 2334 |
| bh | fp32 | `div` | `default` | within 2 ULP everywhere | 2 | 1.58 | — | 524 | 31998 |
| bh | fp32 | `div_bw` | `default` | worst pairing 7.95e+04 ULP; mean 7.95e+04 | 7.95e+04 | 7.95e+04 | — | 10503 | 1597 |
| bh | fp32 | `div_no_nan` | `default` | worst pairing 9.24e+04 ULP; mean 4.51e+04 | 9.24e+04 | 4.51e+04 | — | 1764 | 9514 |
| bh | fp32 | `divide` | `default` | within 2 ULP everywhere | 2 | 1.58 | — | 526 | 31885 |
| bh | fp32 | `divide_` | `default` | within 2 ULP everywhere | 2 | 1.58 | — | 523 | 32082 |
| bh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 426 ±6% | 39372 |
| bh | fp32 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 2758 | 6083 |
| bh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 7 ULP beyond | 7 | 1.23 | 0.000334 | 533 | 31501 |
| bh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | 2438 | 6882 |
| bh | fp32 | `erfc` | `default` | never within 2 ULP; mean 1.71e+29, worst 2.1e+33 | 2.1e+33 | 1.71e+29 | — | 591 | 28372 |
| bh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | 2447 | 6856 |
| bh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists | 7.96e+06 | 2.62e+05 | 1.32e-38 | 380 | 44131 |
| bh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000496; up to 1.03e+06 ULP beyond | 1.03e+06 | 1.68e+03 | 0.000496 | 7486 | 2241 |
| bh | fp32 | `exp` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 454 | 36986 |
| bh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | — | 385 | 43624 |
| bh | fp32 | `exp2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 422 | 39749 |
| bh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.17 | 128 | 1673 | 10029 |
| bh | fp32 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 1334 | 12580 |
| bh | fp32 | `expm1` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 455 | 36840 |
| bh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists | 4.91e+06 | 1.14e+04 | 2.08 | 1721 | 9749 |
| bh | fp32 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | 383 ±10% | 43792 |
| bh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.56e+03 | 4.19e+06 | 1.56e+03 | — | 2481 | 6761 |
| bh | fp32 | `fmod` | `default` | worst pairing 3.19e+38 ULP; mean 4e+36 | 3.19e+38 | 4e+36 | — | 545 ±9% | 30796 |
| bh | fp32 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | 393 | 42741 |
| bh | fp32 | `ge_` | `default` | bit-exact | 0 | — | — | 542 | 30958 |
| bh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists | 1.71e+08 | 1.7e+05 | 0.301 | 402 ±20% | 41772 |
| bh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists | 2.91e+38 | 4.9e+36 | 6e-36 | 395 ±5% | 42524 |
| bh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0393; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.0393 | 928 | 18086 |
| bh | fp32 | `gt_` | `default` | bit-exact | 0 | — | — | 530 | 31661 |
| bh | fp32 | `hardmish` | `default` | bit-exact | 0 | — | 3.39e+38 | 384 ±9% | 43746 |
| bh | fp32 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | 385 | 43589 |
| bh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1643 | 10211 |
| bh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.61; up to 4.89e+06 ULP beyond | 4.89e+06 | 1.25e+03 | 2.61 | 394 ±8% | 42562 |
| bh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2549 | 6582 |
| bh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36; up to 7.34e+06 ULP beyond | 7.34e+06 | 1.41e+03 | 2.36 | 405 ±16% | 41408 |
| bh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00179; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.00179 | 3605 | 4654 |
| bh | fp32 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | 386 | 43514 |
| bh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2165 | 7751 |
| bh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | 392 | 42826 |
| bh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | — | 545 | 30771 |
| bh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 4.86e+06 | 4.48e+04 | — | 3312 | 5065 |
| bh | fp32 | `i0` | `default` | accurate to |x| <= 2.56; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 2.56 | 448 | 37468 |
| bh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00664; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.1e+04 | 0.00664 | 1436 | 11680 |
| bh | fp32 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | 388 | 43266 |
| bh | fp32 | `l1_loss` | `default` | bit-exact | 0 | — | — | 537 | 31214 |
| bh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | 539 | 31109 |
| bh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | 546 | 30716 |
| bh | fp32 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 2380 | 7048 |
| bh | fp32 | `le_` | `default` | bit-exact | 0 | — | — | 538 | 31156 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 385 | 43579 |
| bh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1835 | 9143 |
| bh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | — | 710 | 23633 |
| bh | fp32 | `lerp_bw` | `default` | bit-exact | 0 | — | — | 2794 | 6005 |
| bh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.108; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.108 | 1337 | 12551 |
| bh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 1611 | 10412 |
| bh | fp32 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 443 | 37895 |
| bh | fp32 | `log10` | `default` | within 2 ULP everywhere | 2 | 1.22 | 3.39e+38 | 451 ±10% | 37178 |
| bh | fp32 | `log10_bw` | `default` | accurate to |x| <= 8.81e+30; up to 9.02e+04 ULP beyond | 9.02e+04 | 1.73e+03 | 8.81e+30 | 5216 | 3216 |
| bh | fp32 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 446 ±5% | 37581 |
| bh | fp32 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 9.02e+04 | 2.96e+03 | 2.02e+31 | 5220 | 3214 |
| bh | fp32 | `log2` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | 452 ±10% | 37136 |
| bh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists | 9.02e+04 | 1.76e+03 | — | 5214 | 3218 |
| bh | fp32 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 9.02e+04 | 1.72e+03 | 2.02e+31 | 3952 | 4245 |
| bh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | — | 477 ±10% | 35153 |
| bh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 1.09; up to 3 ULP beyond | 3 | 1.25 | 1.09 | 5833 | 2876 |
| bh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 586 | 28642 |
| bh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 549 | 30538 |
| bh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 554 | 30258 |
| bh | fp32 | `logaddexp2_bw` | `default` | worst pairing 9.02e+04 ULP; mean 5.6e+04 | 9.02e+04 | 5.6e+04 | — | 5132 | 3269 |
| bh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 589 ±12% | 28483 |
| bh | fp32 | `logaddexp_bw` | `default` | worst pairing 8.98e+04 ULP; mean 4.46e+04 | 8.98e+04 | 4.46e+04 | — | 4791 | 3502 |
| bh | fp32 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | 383 | 43814 |
| bh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | — | — | 535 | 31374 |
| bh | fp32 | `logit` | `default` | accurate to |x| <= 0.375; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.375 | 425 ±9% | 39487 |
| bh | fp32 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1.05 | 0.998 | 7023 | 2389 |
| bh | fp32 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1.05 | 0.998 | 8319 | 2017 |
| bh | fp32 | `lt_` | `default` | bit-exact | 0 | — | — | 529 | 31725 |
| bh | fp32 | `mac` | `default` | worst pairing 4.22e+06 ULP; mean 7.31e+05 | 4.22e+06 | 7.31e+05 | — | 709 | 23676 |
| bh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 5430 | 3090 |
| bh | fp32 | `maximum` | `default` | bit-exact | 0 | — | — | 530 | 31633 |
| bh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 5438 | 3085 |
| bh | fp32 | `minimum` | `default` | bit-exact | 0 | — | — | 531 | 31604 |
| bh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 7 | 2.58 | 1.49e-07 | 592 | 28358 |
| bh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | 531 ±9% | 31584 |
| bh | fp32 | `mul_bw` | `default` | bit-exact | 0 | — | — | 1409 | 11904 |
| bh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists | 3.51e+07 | 9.16e+03 | 5.55e-17 | 8312 | 2018 |
| bh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 5.55e-17 | 7880 | 2129 |
| bh | fp32 | `multiply` | `default` | bit-exact | 0 | — | — | 533 | 31483 |
| bh | fp32 | `multiply_` | `default` | bit-exact | 0 | — | — | 531 | 31621 |
| bh | fp32 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | 386 | 43462 |
| bh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | — | 3172 | 5290 |
| bh | fp32 | `polygamma` | `k=1` | 257 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 5.4e-20 | 1025 | 16366 |
| bh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 7253 | 2313 |
| bh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.21 | 5.92e+03 | 3.21 | — | 650 | 25804 |
| bh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | 2525 | 6645 |
| bh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | — | 3.39e+38 | 389 | 43178 |
| bh | fp32 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | 385 ±5% | 43594 |
| bh | fp32 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 392 | 42786 |
| bh | fp32 | `rdiv_bw` | `scalar=2.0` | 254 of 48490 points returned inf or zero where a value exists | 9.02e+04 | 1.92e+03 | 7.67e-20 | 6798 | 2468 |
| bh | fp32 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 9.02e+04 | 1.72e+03 | 2.02e+31 | 388 | 43263 |
| bh | fp32 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 9.02e+04 | 1.92e+03 | 5.42e-20 | 5072 | 3308 |
| bh | fp32 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | 385 | 43612 |
| bh | fp32 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | 386 | 43475 |
| bh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 3995 | 4200 |
| bh | fp32 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1275 | 13158 |
| bh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 384 | 43686 |
| bh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 384 | 43688 |
| bh | fp32 | `remainder` | `default` | worst pairing 3.19e+38 ULP; mean 4.03e+36 | 3.19e+38 | 4.03e+36 | — | 543 | 30874 |
| bh | fp32 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | 386 | 43510 |
| bh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists | 1 | 1 | 8.28e+34 | 629 ±9% | 26666 |
| bh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | 2891 | 5803 |
| bh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 422 | 39750 |
| bh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists | 7 | 4.5 | — | 5764 | 2911 |
| bh | fp32 | `rsub` | `default` | bit-exact | 0 | — | — | 531 | 31589 |
| bh | fp32 | `rsub_` | `default` | bit-exact | 0 | — | — | 536 | 31290 |
| bh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.2, worst 51 | 51 | 26.2 | — | 433 | 38750 |
| bh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 51 | 20.6 | — | 3136 | 5350 |
| bh | fp32 | `sigmoid` | `default` | accurate to |x| <= 0.000475; up to 3 ULP beyond | 3 | 1.65 | 0.000475 | 430 ±5% | 38972 |
| bh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 0.000475; up to 3 ULP beyond | 3 | 1.65 | 0.000475 | 435 ±9% | 38543 |
| bh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists | 8.39e+06 | 2.54e+04 | 0.617 | 2196 | 7639 |
| bh | fp32 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | 393 ±12% | 42699 |
| bh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists | 4 | 1.82 | 2.63e-05 | 434 | 38653 |
| bh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 5.72e+06 | 842 | 1.59e-05 | 3080 | 5447 |
| bh | fp32 | `sin` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | 424 ±10% | 39534 |
| bh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | 1305 | 12855 |
| bh | fp32 | `sinh` | `default` | within 2 ULP everywhere | 2 | 1.12 | 89 | 462 | 36311 |
| bh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1 | 1 | 88.5 | 5988 | 2802 |
| bh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists | 8.2e+03 | 656 | — | 410 | 40885 |
| bh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 1.09; up to 3 ULP beyond | 3 | 1.36 | 1.09 | 4401 | 3812 |
| bh | fp32 | `softshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | 394 | 42529 |
| bh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2162 | 7760 |
| bh | fp32 | `softsign` | `default` | 512 of 65024 points returned inf or zero where a value exists | 3 | 1.28 | 0.000243 | 385 | 43552 |
| bh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 6.44e-05; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 6.44e-05 | 1290 | 13010 |
| bh | fp32 | `sqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 412 | 40715 |
| bh | fp32 | `sqrt_bw` | `default` | within 2 ULP everywhere | 2 | 1.38 | 3.39e+38 | 6449 | 2601 |
| bh | fp32 | `square` | `default` | bit-exact | 0 | — | 1.84e+19 | 396 ±5% | 42319 |
| bh | fp32 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | 1270 | 13209 |
| bh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | 541 | 30988 |
| bh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | 541 | 31039 |
| bh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0 | — | — | 2165 | 7750 |
| bh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0 | — | — | 532 | 31518 |
| bh | fp32 | `subtract` | `default` | bit-exact | 0 | — | — | 542 | 30928 |
| bh | fp32 | `subtract_` | `default` | bit-exact | 0 | — | — | 541 | 30995 |
| bh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 4 | 1.82 | 2.63e-05 | 446 ±6% | 37657 |
| bh | fp32 | `tan` | `default` | 20110 of 65024 points returned inf or zero where a value exists | 3.34e+38 | 1.43e+36 | 20.3 | 479 ±8% | 35050 |
| bh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists | 3.16e+38 | 9.23e+35 | 1.01 | 2097 | 8001 |
| bh | fp32 | `tanh` | `default` | accurate to |x| <= 0.0154; up to 3 ULP beyond | 3 | 1.54 | 0.0154 | 449 | 37405 |
| bh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | — | 936 | 17932 |
| bh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.23e+05 | 1.34e-08 | 545 ±11% | 30764 |
| bh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.94e+04 | 7.42e-09 | 1709 | 9816 |
| bh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | 385 | 43596 |
| bh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 1640 | 10227 |
| bh | fp32 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | 379 | 44230 |
| bh | fp32 | `where` | `default` | bit-exact | 0 | — | — | 699 | 24018 |
| bh | fp32 | `xielu` | `default` | accurate to |x| <= 0.691; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 0.691 | 415 | 40414 |
| bh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | — | 526 | 31911 |
| bh | fp32 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 8937 | 1877 |
| wh | bf16 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | 395 ±6% | 42447 |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1304 | 12863 |
| wh | bf16 | `acos` | `default` | bit-exact | 0 | — | 1 | 667 | 25167 |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.953; up to 3 ULP beyond | 3 | 1.1 | 0.953 | 7495 | 2239 |
| wh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1 | 0.994 | 3.39e+38 | 927 | 18105 |
| wh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 3 | 1.01 | 1.05 | 8640 | 1942 |
| wh | bf16 | `add` | `default` | within 2 ULP everywhere | 1 | 1 | — | 526 | 31913 |
| wh | bf16 | `add_` | `default` | within 2 ULP everywhere | 1 | 1 | — | 540 | 31084 |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | 530 | 31662 |
| wh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.32e+03 | 2.11e+06 | 2.32e+03 | — | 693 ±7% | 24210 |
| wh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.6 | 126 | 19.6 | — | 684 | 24517 |
| wh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists | 0 | — | — | 675 | 24844 |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 3 ULP beyond | 3 | 1.1 | 0.938 | 7405 | 2266 |
| wh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1 | 0.996 | 3.39e+38 | 1469 | 11418 |
| wh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1 | 1 | 1.84e+19 | 1343 | 12492 |
| wh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | 609 ±158% | 27562 |
| wh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 2.88 | 200 | 2.88 | — | 546 ±7% | 30700 |
| wh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.92 | 248 | 4.92 | — | 5476 | 3064 |
| wh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 2 | 1.04 | 9.19e+18 | 1335 | 12568 |
| wh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists | 2 | 0.997 | — | 686 | 24442 |
| wh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 8 | 1.16 | 0.82 | 8427 | 1991 |
| wh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 526 | 31884 |
| wh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 541 | 31028 |
| wh | bf16 | `cbrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 430 ±11% | 38978 |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | 416 ±7% | 40355 |
| wh | bf16 | `celu` | `default` | bit-exact | 0 | — | 3.39e+38 | 490 ±357% | 34260 |
| wh | bf16 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | 2697 | 6221 |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 409 ±7% | 40972 |
| wh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 403 ±6% | 41658 |
| wh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | — | 3.39e+38 | 402 ±6% | 41703 |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | — | — | 2447 | 6855 |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 414 ±15% | 40537 |
| wh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 399 ±5% | 42050 |
| wh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | — | 3.39e+38 | 407 ±13% | 41188 |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | — | — | 2437 | 6885 |
| wh | bf16 | `cos` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | 429 ±6% | 39152 |
| wh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | 1720 | 9755 |
| wh | bf16 | `cosh` | `default` | bit-exact | 0 | — | 89 | 452 ±11% | 37083 |
| wh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0 | — | 88.5 | 6702 | 2503 |
| wh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 0.992 | 6.7e-37 | 411 ±6% | 40775 |
| wh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 1181 | 14211 |
| wh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.32e+04 | 1 | 11530 | 1455 |
| wh | bf16 | `div` | `default` | bit-exact | 0 | — | — | 554 ±7% | 30272 |
| wh | bf16 | `div_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 10530 ±21% | 1593 |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0 | — | — | 1455 | 11532 |
| wh | bf16 | `divide` | `default` | bit-exact | 0 | — | — | 552 | 30380 |
| wh | bf16 | `divide_` | `default` | bit-exact | 0 | — | — | 567 | 29568 |
| wh | bf16 | `elu` | `default` | bit-exact | 0 | — | 3.39e+38 | 478 ±13% | 35094 |
| wh | bf16 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | 2727 | 6152 |
| wh | bf16 | `eq` | `default` | bit-exact | 0 | — | — | 534 ±16% | 31400 |
| wh | bf16 | `eq_` | `default` | bit-exact | 0 | — | — | 541 | 30989 |
| wh | bf16 | `eqz` | `default` | bit-exact | 0 | — | 3.39e+38 | 397 ±8% | 42283 |
| wh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 589 ±9% | 28496 |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | 2462 | 6815 |
| wh | bf16 | `erfc` | `default` | accurate to |x| <= 2.5; up to 3.19e+28 ULP beyond | 3.19e+28 | 3.23e+24 | 2.5 | 834 | 20126 |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | 2450 | 6848 |
| wh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists | 121 | 6.53 | 1.31e-38 | 901 | 18616 |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 9 ULP beyond | 9 | 1.39 | 0.777 | 7886 | 2127 |
| wh | bf16 | `exp` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | 424 ±9% | 39603 |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6 | 6 | 3.05 | — | 396 ±5% | 42360 |
| wh | bf16 | `exp2` | `default` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | 432 ±6% | 38832 |
| wh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.05 | 128 | 1699 | 9872 |
| wh | bf16 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | 1319 | 12717 |
| wh | bf16 | `expm1` | `default` | bit-exact | 0 | — | 3.39e+38 | 421 ±16% | 39847 |
| wh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists | 125 | 5.83 | 2.09 | 1703 | 9852 |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | 419 ±10% | 40076 |
| wh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | — | 2530 | 6631 |
| wh | bf16 | `fmod` | `default` | worst pairing 6.65e+35 ULP; mean 7.8e+34 | 6.65e+35 | 7.8e+34 | — | 707 | 23720 |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | 408 | 41091 |
| wh | bf16 | `ge` | `default` | bit-exact | 0 | — | — | 531 | 31586 |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | — | — | 543 | 30885 |
| wh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists | 165 | 9.42 | 2.33e-38 | 806 | 20809 |
| wh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists | 1.14e+36 | 1.82e+34 | 2.33e-38 | 412 ±8% | 40722 |
| wh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.44; up to 3 ULP beyond | 3 | 1.52 | 8.44 | 1773 | 9462 |
| wh | bf16 | `gez` | `default` | bit-exact | 0 | — | 3.39e+38 | 408 ±6% | 41147 |
| wh | bf16 | `gt` | `default` | bit-exact | 0 | — | — | 530 | 31679 |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | — | — | 538 | 31157 |
| wh | bf16 | `gtz` | `default` | bit-exact | 0 | — | 3.39e+38 | 397 ±6% | 42284 |
| wh | bf16 | `hardmish` | `default` | within 2 ULP everywhere | 1 | 0.993 | 3.39e+38 | 411 ±6% | 40844 |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | 402 | 41687 |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1676 | 10009 |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.661 | 3.39e+38 | 415 ±6% | 40439 |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2532 | 6626 |
| wh | bf16 | `hardswish` | `default` | 2 of 65024 points returned inf or zero where a value exists | 2 | 1.01 | 2.33e-38 | 431 ±8% | 38964 |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12; up to 85 ULP beyond | 85 | 2.36 | 1.12 | 3532 | 4750 |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | 396 ±7% | 42380 |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2146 | 7819 |
| wh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | — | 3.39e+38 | 415 ±7% | 40420 |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | 418 ±10% | 40102 |
| wh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | — | 3.39e+38 | 403 ±8% | 41585 |
| wh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 53 | 1.46 | — | 545 | 30800 |
| wh | bf16 | `hypot_bw` | `default` | worst pairing 75 ULP; mean 2.44 | 75 | 2.44 | — | 3369 | 4980 |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.9; up to 255 ULP beyond | 255 | 56.4 | 13.9 | 440 | 38133 |
| wh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.95 | 2.33e-38 | 2085 | 8047 |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | 394 | 42534 |
| wh | bf16 | `isclose` | `default` | bit-exact | 0 | — | — | 546 | 30735 |
| wh | bf16 | `isfinite` | `default` | bit-exact | 0 | — | 3.39e+38 | 400 | 41901 |
| wh | bf16 | `isinf` | `default` | bit-exact | 0 | — | 3.39e+38 | 397 ±6% | 42222 |
| wh | bf16 | `isnan` | `default` | bit-exact | 0 | — | 3.39e+38 | 401 | 41806 |
| wh | bf16 | `isneginf` | `default` | bit-exact | 0 | — | 3.39e+38 | 404 ±6% | 41514 |
| wh | bf16 | `isposinf` | `default` | bit-exact | 0 | — | 3.39e+38 | 414 ±9% | 40549 |
| wh | bf16 | `l1_loss` | `default` | within 2 ULP everywhere | 1 | 1 | — | 526 | 31912 |
| wh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | 540 | 31085 |
| wh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | 558 | 30060 |
| wh | bf16 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 2708 | 6196 |
| wh | bf16 | `le` | `default` | bit-exact | 0 | — | — | 537 ±6% | 31254 |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | — | — | 543 | 30907 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | — | 3.39e+38 | 390 ±6% | 43061 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 405 ±12% | 41375 |
| wh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | — | 3.39e+38 | 398 ±7% | 42141 |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1812 | 9259 |
| wh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 40 | 1.77e+03 | 40 | — | 682 ±7% | 24598 |
| wh | bf16 | `lerp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 2796 ±7% | 6001 |
| wh | bf16 | `lez` | `default` | bit-exact | 0 | — | 3.39e+38 | 399 ±8% | 42005 |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.439; up to 324 ULP beyond | 324 | 1.17 | 0.439 | 2429 | 6908 |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 2064 | 8128 |
| wh | bf16 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 423 ±11% | 39708 |
| wh | bf16 | `log10` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 419 | 40016 |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.69e+37 | 5261 | 3189 |
| wh | bf16 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 417 ±9% | 40213 |
| wh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 5240 | 3202 |
| wh | bf16 | `log2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 436 ±15% | 38458 |
| wh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists | 2 | 1.02 | — | 5264 | 3187 |
| wh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 3993 | 4202 |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 4.12; up to 6 ULP beyond | 6 | 1.51 | 4.12 | 588 ±6% | 28511 |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 2 | 0.929 | 3.39e+38 | 5908 | 2840 |
| wh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 816 ±68% | 20567 |
| wh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 898 | 18688 |
| wh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 897 | 18696 |
| wh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41 ULP; mean 3.82 | 41 | 3.82 | — | 5768 | 2908 |
| wh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 792 ±5% | 21178 |
| wh | bf16 | `logaddexp_bw` | `default` | worst pairing 62 ULP; mean 4.88 | 62 | 4.88 | — | 4782 | 3508 |
| wh | bf16 | `logical_and` | `default` | bit-exact | 0 | — | — | 546 | 30754 |
| wh | bf16 | `logical_and_` | `default` | bit-exact | 0 | — | — | 557 | 30122 |
| wh | bf16 | `logical_not` | `default` | bit-exact | 0 | — | 3.39e+38 | 410 ±22% | 40961 |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | 411 ±13% | 40835 |
| wh | bf16 | `logical_or` | `default` | bit-exact | 0 | — | — | 548 | 30595 |
| wh | bf16 | `logical_or_` | `default` | bit-exact | 0 | — | — | 555 ±8% | 30251 |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | — | — | 559 | 30017 |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.95 | 0.395 | 895 | 18744 |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | 6993 | 2399 |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | 8248 | 2034 |
| wh | bf16 | `lt` | `default` | bit-exact | 0 | — | — | 536 | 31281 |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | — | — | 549 ±8% | 30537 |
| wh | bf16 | `ltz` | `default` | bit-exact | 0 | — | 3.39e+38 | 405 | 41375 |
| wh | bf16 | `mac` | `default` | worst pairing 250 ULP; mean 38.5 | 250 | 38.5 | — | 1038 | 16162 |
| wh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 5443 | 3082 |
| wh | bf16 | `maximum` | `default` | bit-exact | 0 | — | — | 526 | 31890 |
| wh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 5442 | 3083 |
| wh | bf16 | `minimum` | `default` | bit-exact | 0 | — | — | 533 ±19% | 31469 |
| wh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists | 1 | 0.997 | 1.95e-38 | 607 ±10% | 27634 |
| wh | bf16 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | 533 | 31503 |
| wh | bf16 | `mul_bw` | `default` | bit-exact | 0 | — | — | 1413 | 11876 |
| wh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists | 724 | 1.81 | 5.55e-17 | 12590 | 1333 |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 294 | 5.55e-17 | 9634 | 1742 |
| wh | bf16 | `multiply` | `default` | bit-exact | 0 | — | — | 531 | 31598 |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0 | — | — | 546 | 30756 |
| wh | bf16 | `ne` | `default` | bit-exact | 0 | — | — | 526 | 31904 |
| wh | bf16 | `ne_` | `default` | bit-exact | 0 | — | — | 539 | 31106 |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | 402 ±6% | 41755 |
| wh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | — | 3105 | 5402 |
| wh | bf16 | `nez` | `default` | bit-exact | 0 | — | 3.39e+38 | 401 ±7% | 41852 |
| wh | bf16 | `polygamma` | `k=1` | 7 of 49922 points returned inf or zero where a value exists | 1.02e+08 | 1.43e+05 | 0.996 | 5295 | 3168 |
| wh | bf16 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 4.47 | 5493 | 3054 |
| wh | bf16 | `polygamma` | `k=4` | 142 of 49922 points returned inf or zero where a value exists | 3.22e+09 | 1.09e+07 | 4.47 | 6043 | 2776 |
| wh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 4.47 | 11691 | 1435 |
| wh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | — | 960 ±616% | 17472 |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | 2536 | 6615 |
| wh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists | 0 | — | 4.68e-38 | 394 | 42604 |
| wh | bf16 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | 399 | 42039 |
| wh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 418 ±9% | 40090 |
| wh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists | 2 | 1.02 | 7.67e-20 | 6906 | 2429 |
| wh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 408 ±8% | 41143 |
| wh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 2 | 1.02 | 5.42e-20 | 5106 | 3286 |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | 398 | 42196 |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | 404 ±7% | 41503 |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 3941 | 4257 |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1290 | 13005 |
| wh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | — | 3.39e+38 | 405 ±7% | 41458 |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 408 ±5% | 41144 |
| wh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | — | 3.39e+38 | 416 ±5% | 40298 |
| wh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | — | 3.39e+38 | 403 | 41598 |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 392 ±7% | 42782 |
| wh | bf16 | `remainder` | `default` | worst pairing 6.65e+35 ULP; mean 1.06e+35 | 6.65e+35 | 1.06e+35 | — | 794 | 21139 |
| wh | bf16 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | 404 ±16% | 41550 |
| wh | bf16 | `rpow` | `exponent=0.5` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | 944 | 17770 |
| wh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | — | 3.39e+38 | 954 | 17583 |
| wh | bf16 | `rpow` | `exponent=2.0` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | 937 | 17913 |
| wh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | 2905 | 5775 |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | 423 ±12% | 39694 |
| wh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists | 3 | 1.28 | — | 6227 | 2694 |
| wh | bf16 | `rsub` | `default` | within 2 ULP everywhere | 1 | 1 | — | 531 | 31578 |
| wh | bf16 | `rsub_` | `default` | within 2 ULP everywhere | 1 | 1 | — | 536 | 31299 |
| wh | bf16 | `selu` | `default` | bit-exact | 0 | — | 3.39e+38 | 530 ±13% | 31647 |
| wh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 3 | 1.04 | 0.586 | 3098 | 5415 |
| wh | bf16 | `sigmoid` | `default` | within 2 ULP everywhere | 1 | 0.876 | 3.39e+38 | 500 ±12% | 33570 |
| wh | bf16 | `sigmoid_accurate` | `default` | within 2 ULP everywhere | 1 | 0.876 | 3.39e+38 | 477 ±12% | 35206 |
| wh | bf16 | `sigmoid_bw` | `default` | 329 of 65024 points returned inf or zero where a value exists | 266 | 5.66 | 1.85 | 2228 | 7529 |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | 404 ±10% | 41560 |
| wh | bf16 | `signbit` | `default` | bit-exact | 0 | — | 3.39e+38 | 392 ±8% | 42763 |
| wh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | 503 ±11% | 33372 |
| wh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 285 | 1.54 | 0.883 | 3107 | 5399 |
| wh | bf16 | `sin` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | 423 ±14% | 39708 |
| wh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | 1322 | 12687 |
| wh | bf16 | `sinh` | `default` | bit-exact | 0 | — | 89 | 571 ±14% | 29404 |
| wh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0 | — | 88.5 | 5828 | 2879 |
| wh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists | 1 | 1 | 5.03 | 471 ±15% | 35616 |
| wh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 2 | 0.897 | 3.39e+38 | 4335 | 3870 |
| wh | bf16 | `softshrink` | `default` | within 2 ULP everywhere | 1 | 0.996 | 3.39e+38 | 410 | 40889 |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2160 | 7768 |
| wh | bf16 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 1 | 0.69 | 8.51e+37 | 446 ±10% | 37615 |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00491; up to 81 ULP beyond | 81 | 3.38 | 0.00491 | 1314 | 12768 |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | 452 ±8% | 37080 |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 6578 | 2551 |
| wh | bf16 | `square` | `default` | within 2 ULP everywhere | 1 | 0.991 | 1.84e+19 | 399 ±9% | 42040 |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | 1281 | 13098 |
| wh | bf16 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | 526 | 31911 |
| wh | bf16 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | 537 | 31252 |
| wh | bf16 | `squared_difference_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 2162 | 7761 |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | 535 | 31337 |
| wh | bf16 | `subtract` | `default` | within 2 ULP everywhere | 1 | 1 | — | 525 | 31979 |
| wh | bf16 | `subtract_` | `default` | within 2 ULP everywhere | 1 | 1 | — | 539 | 31126 |
| wh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | 495 ±12% | 33878 |
| wh | bf16 | `tan` | `default` | 21634 of 65024 points returned inf or zero where a value exists | 3.3e+38 | 1.16e+36 | 1.31e+05 | 574 ±15% | 29248 |
| wh | bf16 | `tan_bw` | `default` | 12178 of 65024 points returned inf or zero where a value exists | 2.53e+38 | 3.12e+35 | 1.31e+05 | 2206 | 7605 |
| wh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | 417 ±6% | 40197 |
| wh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 2.23 | 17.2 | 1470 | 11409 |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.5 | 1.35e-08 | 428 ±11% | 39219 |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.71 | 7.45e-09 | 1685 | 9959 |
| wh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | — | 3.39e+38 | 399 ±6% | 42045 |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | 416 ±12% | 40313 |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 1650 | 10167 |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | 402 ±5% | 41757 |
| wh | bf16 | `where` | `default` | bit-exact | 0 | — | — | 678 | 24734 |
| wh | bf16 | `xielu` | `default` | 2 of 56847 points returned inf or zero where a value exists | 1.87e+38 | 3.67e+36 | 2.33e-38 | 972 | 17259 |
| wh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | — | 551 | 30464 |
| wh | bf16 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 8887 ±42% | 1888 |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | 774 | 21682 |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2483 | 6758 |
| wh | fp32 | `acos` | `default` | within 2 ULP everywhere | 2 | 1 | 1 | 1073 | 15639 |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | 14942 | 1123 |
| wh | fp32 | `acosh` | `default` | never within 2 ULP; mean 1, worst 3 | 3 | 1 | — | 1431 ±8% | 11726 |
| wh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 724 | 1.12 | 0.996 | 17134 | 979 |
| wh | fp32 | `add` | `default` | bit-exact | 0 | — | — | 982 | 17092 |
| wh | fp32 | `add_` | `default` | bit-exact | 0 | — | — | 997 | 16833 |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0 | — | — | 988 | 16987 |
| wh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | — | 1333 ±141% | 12585 |
| wh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | — | 1328 | 12634 |
| wh | fp32 | `asin` | `default` | within 2 ULP everywhere | 2 | 1.08 | 1 | 1049 | 15989 |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | 14717 | 1140 |
| wh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | 1795 | 9349 |
| wh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 2 | 1.03 | 1.84e+19 | 2598 | 6456 |
| wh | fp32 | `atan` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.39e+38 | 908 | 18469 |
| wh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | — | 1029 | 16306 |
| wh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 9.5e+04 | 1.64e+07 | 9.5e+04 | — | 10585 | 1585 |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 4.08e+03; up to 3 ULP beyond | 3 | 1.19 | 4.08e+03 | 2563 | 6546 |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000852; up to 3 ULP beyond | 3 | 1.56 | 0.000852 | 1353 ±6% | 12402 |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.852; up to 2.05e+03 ULP beyond | 2.05e+03 | 1.51 | 0.852 | 16372 | 1025 |
| wh | fp32 | `bias_gelu` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | 990 ±8% | 16954 |
| wh | fp32 | `bias_gelu_` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | 993 | 16896 |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 5.77e-38; up to 3 ULP beyond | 3 | 1.82 | 5.77e-38 | 844 ±6% | 19870 |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | 770 | 21789 |
| wh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 940 | 17855 |
| wh | fp32 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 5439 | 3084 |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 773 | 21715 |
| wh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 773 | 21705 |
| wh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | — | 3.39e+38 | 772 | 21734 |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | — | — | 4619 | 3632 |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 781 ±12% | 21471 |
| wh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 767 | 21877 |
| wh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | — | 3.39e+38 | 768 | 21853 |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | — | — | 4623 | 3629 |
| wh | fp32 | `cos` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | 859 | 19529 |
| wh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | 3306 | 5075 |
| wh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1 | 1 | 89 | 979 | 17137 |
| wh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 2 | 1.14 | 88.5 | 13493 | 1243 |
| wh | fp32 | `deg2rad` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 773 | 21701 |
| wh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 2651 | 6328 |
| wh | fp32 | `digamma_bw` | `default` | 255 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 8.58e-05 | 18064 | 929 |
| wh | fp32 | `div` | `default` | within 2 ULP everywhere | 2 | 1.2 | — | 1006 | 16670 |
| wh | fp32 | `div_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 20695 | 811 |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP everywhere | 2 | 1.53 | — | 3466 | 4840 |
| wh | fp32 | `divide` | `default` | within 2 ULP everywhere | 2 | 1.2 | — | 1001 | 16756 |
| wh | fp32 | `divide_` | `default` | within 2 ULP everywhere | 2 | 1.2 | — | 1007 ±7% | 16666 |
| wh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 934 | 17970 |
| wh | fp32 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 5430 | 3090 |
| wh | fp32 | `eq` | `default` | bit-exact | 0 | — | — | 983 | 17061 |
| wh | fp32 | `eq_` | `default` | bit-exact | 0 | — | — | 1003 | 16734 |
| wh | fp32 | `eqz` | `default` | bit-exact | 0 | — | 3.39e+38 | 770 | 21803 |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 6 ULP beyond | 6 | 1.22 | 0.000334 | 1198 ±11% | 14000 |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | 4874 ±38% | 3442 |
| wh | fp32 | `erfc` | `default` | never within 2 ULP; mean 1.71e+29, worst 2.1e+33 | 2.1e+33 | 1.71e+29 | — | 1352 ±7% | 12411 |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | 4865 | 3449 |
| wh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists | 7.96e+06 | 2.62e+05 | 1.32e-38 | 1407 ±12% | 11923 |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000496; up to 4.98e+05 ULP beyond | 4.98e+05 | 1.32e+03 | 0.000496 | 15403 | 1089 |
| wh | fp32 | `exp` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 896 | 18735 |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | — | 766 | 21905 |
| wh | fp32 | `exp2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 840 | 19979 |
| wh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.17 | 128 | 3312 | 5066 |
| wh | fp32 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 2603 | 6444 |
| wh | fp32 | `expm1` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 1034 ±6% | 16219 |
| wh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists | 4.91e+06 | 1.14e+04 | 2.08 | 3496 ±56% | 4799 |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | 781 | 21492 |
| wh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.77e+03 | 4.19e+06 | 1.77e+03 | — | 4788 | 3504 |
| wh | fp32 | `fmod` | `default` | worst pairing 1.7e+38 ULP; mean 2.72e+36 | 1.7e+38 | 2.72e+36 | — | 1022 | 16417 |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | 766 | 21908 |
| wh | fp32 | `ge` | `default` | bit-exact | 0 | — | — | 982 | 17078 |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | — | — | 994 | 16874 |
| wh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists | 1.51e+08 | 1.55e+05 | 0.309 | 1390 ±6% | 12067 |
| wh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists | 2.91e+38 | 4.9e+36 | 6e-36 | 769 ±27% | 21814 |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0393; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.0393 | 2093 | 8015 |
| wh | fp32 | `gez` | `default` | bit-exact | 0 | — | 3.39e+38 | 769 ±5% | 21831 |
| wh | fp32 | `gt` | `default` | bit-exact | 0 | — | — | 990 | 16945 |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | — | — | 1002 | 16747 |
| wh | fp32 | `gtz` | `default` | bit-exact | 0 | — | 3.39e+38 | 765 ±6% | 21928 |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | — | 3.39e+38 | 766 | 21891 |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | 776 ±7% | 21619 |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 3239 | 5179 |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.61; up to 4.89e+06 ULP beyond | 4.89e+06 | 1.25e+03 | 2.61 | 777 | 21591 |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 5045 | 3326 |
| wh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36; up to 7.34e+06 ULP beyond | 7.34e+06 | 1.41e+03 | 2.36 | 828 | 20266 |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00179; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.00179 | 7079 | 2370 |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | 776 | 21622 |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 4258 | 3940 |
| wh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | — | 3.39e+38 | 776 | 21612 |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | 773 | 21708 |
| wh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | — | 3.39e+38 | 784 | 21399 |
| wh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | — | 1020 | 16448 |
| wh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 4.86e+06 | 4.48e+04 | — | 6371 | 2633 |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.56; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 2.56 | 888 | 18900 |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00774; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.00774 | 3453 | 4858 |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | 768 | 21848 |
| wh | fp32 | `isclose` | `default` | bit-exact | 0 | — | — | 1002 | 16747 |
| wh | fp32 | `isfinite` | `default` | bit-exact | 0 | — | 3.39e+38 | 765 | 21930 |
| wh | fp32 | `isinf` | `default` | bit-exact | 0 | — | 3.39e+38 | 773 | 21691 |
| wh | fp32 | `isnan` | `default` | bit-exact | 0 | — | 3.39e+38 | 773 | 21704 |
| wh | fp32 | `isneginf` | `default` | bit-exact | 0 | — | 3.39e+38 | 779 ±8% | 21533 |
| wh | fp32 | `isposinf` | `default` | bit-exact | 0 | — | 3.39e+38 | 771 | 21754 |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0 | — | — | 990 | 16952 |
| wh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | 1001 | 16762 |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | 1010 | 16615 |
| wh | fp32 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 5237 | 3203 |
| wh | fp32 | `le` | `default` | bit-exact | 0 | — | — | 991 | 16930 |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | — | — | 998 | 16808 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | — | 3.39e+38 | 779 ±9% | 21547 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 772 | 21738 |
| wh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | — | 3.39e+38 | 767 | 21873 |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 3580 | 4686 |
| wh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | — | 1322 | 12688 |
| wh | fp32 | `lerp_bw` | `default` | bit-exact | 0 | — | — | 5381 | 3118 |
| wh | fp32 | `lez` | `default` | bit-exact | 0 | — | 3.39e+38 | 777 | 21605 |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.108; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.108 | 3735 | 4492 |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 4364 | 3845 |
| wh | fp32 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 865 ±213% | 19388 |
| wh | fp32 | `log10` | `default` | within 2 ULP everywhere | 2 | 1.22 | 3.39e+38 | 877 | 19138 |
| wh | fp32 | `log10_bw` | `default` | within 2 ULP everywhere | 2 | 1.41 | 3.69e+37 | 10258 | 1635 |
| wh | fp32 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 898 | 18678 |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP everywhere | 2 | 1.03 | 8.51e+37 | 10259 | 1635 |
| wh | fp32 | `log2` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | 874 | 19191 |
| wh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists | 2 | 1.33 | — | 10222 | 1641 |
| wh | fp32 | `log_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 8.51e+37 | 7763 | 2161 |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | — | 1070 | 15682 |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 1.09; up to 3 ULP beyond | 3 | 1.25 | 1.09 | 11496 | 1459 |
| wh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 1509 ±38% | 11118 |
| wh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 1294 | 12970 |
| wh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 1288 | 13030 |
| wh | fp32 | `logaddexp2_bw` | `default` | worst pairing 46 ULP; mean 6.56 | 46 | 6.56 | — | 11227 | 1494 |
| wh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 1496 | 11214 |
| wh | fp32 | `logaddexp_bw` | `default` | worst pairing 66 ULP; mean 8.57 | 66 | 8.57 | — | 9406 | 1784 |
| wh | fp32 | `logical_and` | `default` | bit-exact | 0 | — | — | 1003 | 16722 |
| wh | fp32 | `logical_and_` | `default` | bit-exact | 0 | — | — | 1016 | 16514 |
| wh | fp32 | `logical_not` | `default` | bit-exact | 0 | — | 3.39e+38 | 768 | 21843 |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | 772 | 21718 |
| wh | fp32 | `logical_or` | `default` | bit-exact | 0 | — | — | 1003 ±13% | 16730 |
| wh | fp32 | `logical_or_` | `default` | bit-exact | 0 | — | — | 1014 | 16547 |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | — | — | 1022 | 16421 |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.332; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.332 | 1047 | 16024 |
| wh | fp32 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1.06 | 0.998 | 13846 | 1212 |
| wh | fp32 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1.06 | 0.998 | 16426 | 1021 |
| wh | fp32 | `lt` | `default` | bit-exact | 0 | — | — | 998 | 16818 |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | — | — | 996 | 16838 |
| wh | fp32 | `ltz` | `default` | bit-exact | 0 | — | 3.39e+38 | 766 | 21910 |
| wh | fp32 | `mac` | `default` | worst pairing 8.33e+06 ULP; mean 1.45e+06 | 8.33e+06 | 1.45e+06 | — | 1951 | 8601 |
| wh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 10463 | 1604 |
| wh | fp32 | `maximum` | `default` | bit-exact | 0 | — | — | 989 | 16959 |
| wh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 10462 | 1604 |
| wh | fp32 | `minimum` | `default` | bit-exact | 0 | — | — | 989 | 16958 |
| wh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 7 | 2.49 | 8.61e-06 | 1176 ±8% | 14268 |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | 989 | 16961 |
| wh | fp32 | `mul_bw` | `default` | bit-exact | 0 | — | — | 2692 | 6231 |
| wh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists | 3.51e+07 | 9.16e+03 | 5.59e-17 | 20737 | 809 |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 5.55e-17 | 20237 | 829 |
| wh | fp32 | `multiply` | `default` | bit-exact | 0 | — | — | 983 | 17059 |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0 | — | — | 998 | 16805 |
| wh | fp32 | `ne` | `default` | bit-exact | 0 | — | — | 1002 | 16747 |
| wh | fp32 | `ne_` | `default` | bit-exact | 0 | — | — | 1001 | 16767 |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | 765 ±6% | 21926 |
| wh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | — | 6082 | 2759 |
| wh | fp32 | `nez` | `default` | bit-exact | 0 | — | 3.39e+38 | 765 | 21918 |
| wh | fp32 | `polygamma` | `k=1` | accurate to |x| <= 8.58e-05; up to 7.91e+33 ULP beyond | 7.91e+33 | 4.68e+29 | 8.58e-05 | 6014 | 2790 |
| wh | fp32 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 6205 | 2704 |
| wh | fp32 | `polygamma` | `k=4` | 141 of 49922 points returned inf or zero where a value exists | 3.77e+25 | 6.51e+21 | 3.68e-08 | 6661 | 2519 |
| wh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 18131 | 925 |
| wh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.21 | 5.92e+03 | 3.21 | — | 2098 | 7996 |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | 4955 | 3386 |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | — | 3.39e+38 | 768 | 21856 |
| wh | fp32 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | 769 | 21806 |
| wh | fp32 | `rdiv` | `value=2.0` | 256 of 64770 points returned inf or zero where a value exists | 1 | 1 | 8.51e+37 | 801 | 20950 |
| wh | fp32 | `rdiv_bw` | `scalar=2.0` | 252 of 48490 points returned inf or zero where a value exists | 2 | 1.14 | 7.67e-20 | 13337 | 1258 |
| wh | fp32 | `reciprocal` | `default` | within 2 ULP everywhere | 1 | 1 | 8.51e+37 | 788 | 21278 |
| wh | fp32 | `reciprocal_bw` | `default` | 254 of 48386 points returned inf or zero where a value exists | 2 | 1.14 | 5.42e-20 | 10013 ±37% | 1676 |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | 764 | 21965 |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | 773 | 21702 |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 7863 | 2134 |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2476 | 6776 |
| wh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | — | 3.39e+38 | 766 | 21910 |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 767 | 21875 |
| wh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | — | 3.39e+38 | 768 | 21852 |
| wh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | — | 3.39e+38 | 766 | 21908 |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 766 | 21893 |
| wh | fp32 | `remainder` | `default` | worst pairing 1.7e+38 ULP; mean 2.68e+36 | 1.7e+38 | 2.68e+36 | — | 1039 | 16154 |
| wh | fp32 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | 771 | 21754 |
| wh | fp32 | `rpow` | `exponent=0.5` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 1814 | 9248 |
| wh | fp32 | `rpow` | `exponent=1.0` | 1536 of 49536 points returned inf or zero where a value exists | 0 | — | 8.28e+34 | 1813 | 9254 |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists | 1 | 1 | 8.28e+34 | 1799 | 9326 |
| wh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | 5695 | 2946 |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 844 | 19874 |
| wh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists | 7 | 4.51 | — | 11833 | 1418 |
| wh | fp32 | `rsub` | `default` | bit-exact | 0 | — | — | 978 | 17149 |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0 | — | — | 1008 | 16647 |
| wh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.2, worst 51 | 51 | 26.2 | — | 968 | 17333 |
| wh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 51 | 20.6 | — | 6170 | 2719 |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 16.6; up to 3 ULP beyond | 3 | 1.41 | 16.6 | 1070 ±119% | 15676 |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 16.6; up to 3 ULP beyond | 3 | 1.41 | 16.6 | 1056 | 15881 |
| wh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists | 8.39e+06 | 2.71e+04 | 0.859 | 4491 | 3736 |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | 775 | 21643 |
| wh | fp32 | `signbit` | `default` | bit-exact | 0 | — | 3.39e+38 | 764 | 21964 |
| wh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists | 3 | 1.58 | 0.000928 | 1092 | 15360 |
| wh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 5.72e+06 | 841 | 0.000244 | 6223 | 2696 |
| wh | fp32 | `sin` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | 833 | 20144 |
| wh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | 2565 | 6540 |
| wh | fp32 | `sinh` | `default` | within 2 ULP everywhere | 2 | 1.14 | 89 | 1092 | 15357 |
| wh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1 | 1 | 88.5 | 11883 | 1412 |
| wh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists | 8.2e+03 | 656 | — | 1416 ±9% | 11846 |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 1.1; up to 3 ULP beyond | 3 | 1.37 | 1.1 | 8669 | 1935 |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | 772 | 21744 |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 4256 | 3942 |
| wh | fp32 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 3 | 0.89 | 2.53e+07 | 812 | 20654 |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 9.88e-05; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 9.88e-05 | 2580 | 6503 |
| wh | fp32 | `sqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 829 | 20239 |
| wh | fp32 | `sqrt_bw` | `default` | within 2 ULP everywhere | 2 | 1.41 | 3.39e+38 | 12662 | 1325 |
| wh | fp32 | `square` | `default` | bit-exact | 0 | — | 1.84e+19 | 769 | 21826 |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | 2498 | 6715 |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | 990 | 16952 |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | 1000 | 16782 |
| wh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0 | — | — | 4218 | 3978 |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0 | — | — | 995 ±277% | 16860 |
| wh | fp32 | `subtract` | `default` | bit-exact | 0 | — | — | 989 | 16972 |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0 | — | — | 995 | 16855 |
| wh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 3 | 1.58 | 0.000928 | 1076 | 15595 |
| wh | fp32 | `tan` | `default` | 20110 of 65024 points returned inf or zero where a value exists | 3.34e+38 | 1.43e+36 | 20.3 | 1155 ±9% | 14524 |
| wh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists | 3.16e+38 | 9.23e+35 | 1.03 | 4374 | 3835 |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000485; up to 3 ULP beyond | 3 | 1.55 | 0.000485 | 927 | 18103 |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | — | 1773 | 9461 |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.11e+04 | 1.34e-08 | 1415 ±6% | 11860 |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.85e+04 | 7.42e-09 | 3395 | 4942 |
| wh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | — | 3.39e+38 | 771 | 21769 |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | 772 | 21719 |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 3248 | 5165 |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | 768 | 21834 |
| wh | fp32 | `where` | `default` | bit-exact | 0 | — | — | 1331 | 12604 |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 0.691; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 0.691 | 1469 ±10% | 11423 |
| wh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | — | 996 | 16846 |
| wh | fp32 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 17458 | 961 |
