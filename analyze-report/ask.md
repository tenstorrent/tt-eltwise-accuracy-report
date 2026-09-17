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
| wh | bf16 | `a3a9fb4229a` | 0.78.0 | 2d7f04074d58 |
| wh | fp32 | `a3a9fb4229a` | 0.78.0 | 2d7f04074d58 |

**Status: in force.** Every field below exists in `report_index.json` today, except
`max_rel`, `median_rel`, `bits_worst`, `bits_median`, `unflushed`, `offenders`,
`monotonic` and `nonfinite`, which appear on entries measured after they were added.

The definitions behind every number in the report. An assistant answering from it uses
these and no others. The verdict rules have a code twin in `report/score.py::verdict`;
the two change together.

## Fields, per arch/dtype/op/variant

| Field | Definition |
|---|---|
| `max_ulp` | worst defined ULP error: `\|reference − y\| / ULP(y_ref)`, ULP by tt-metal's definition at the compute dtype. **The reference is one step wider than the measurement** — fp32 for a bf16 sweep, fp64 for fp32 — so the error is fractional: at or below **0.5** the device returned the dtype's correctly rounded answer, and only past 0.5 did it pick the wrong neighbour. Rounding the reference to the dtype first would put both sides on one grid and quantise every error to a whole ULP |
| `mean_ulp` | mean over the points whose outcome is `faithful` or `inexact`, so it answers how wrong it is when it is not the correctly rounded answer; `0` when every defined point was exact, `—` when none was scorable |
| `p50_ulp`, `p95_ulp`, `p99_ulp` | percentiles over every point with a defined ULP, exact ones included. An fp32 row is a group maximum, so an fp32 percentile is a percentile of group maxima |
| `exact_frac` | share of defined points whose outcome is `exact`, meaning the device returned the dtype's correctly rounded answer. `flushed`, `zeroed` and `unflushed` points carry no ULP and are in neither the numerator nor the denominator |
| `rounded_frac` | share of **inputs** within half a ULP, counted before the group reduction, so unlike every other fp32 field it is a per-point rate rather than a statistic over group maxima. On bf16 a row is one input and it equals the share of rows within half a ULP |
| `usable_to` | largest \|x\| below which no point exceeds 2 ULP and none is a defect. Unary only; `—` elsewhere, where x alone does not determine the output |
| `max_abs` | worst absolute error |
| `max_rel` | worst relative error `\|y_ref − y\| / \|y_ref\|`, in fp64, over the points ULP is defined for. The only figure comparable between bf16 and fp32 |
| `median_rel` | median relative error over points where it is non-zero, like `mean_ulp`. Read it beside `exact_frac`, which is the rest of the distribution. A bit-exact variant reads `0` here and `—` for its bits, because an error of zero has no finite precision to report |
| `bits_worst`, `bits_median` | `−log2` of the two above, so bits and relative error always agree. Negative means the error exceeds the value itself |
| `unflushed` | device returned a normal value where the reference underflows to zero: `exp(-100)` answering −4.3e33. Carries no ULP, so no other field counts it |
| `ulp_clipped` | points past the chart clamp of 1000 ULP |
| `n_inputs` | measured points. An fp32 row is the worst point of 2¹⁶ consecutive codes |
| `outcomes` | count per label, below |
| `specials` | device vs golden at ±0, ±inf, NaN, ±min-normal. Printed at the 9 digits that round-trip fp32, no ULP. The reference is rounded to the measured dtype first, as the sweep does, or `acos(0)` would differ by the rounding of pi/2 alone. `agree` is decided on the values, and is stricter than the sweep in one way: two NaNs agree, but `-0` and `0` do not, so `ceil(-min_normal)` reads `differ` here while the sweep scores it exact on IEEE equality |
| `offenders` | the ten worst points by ULP, with their inputs, reference and device value. The maximum says how bad; these say where |
| `monotonic` | `pairs`, `violations`, `rate`, `worst_dy` and the five worst, for neighbouring inputs whose device outputs move against the reference's own direction. Present only where the reference is itself ordered across a domain interval, so a non-monotonic op (`sin`, `gelu` below zero) has no entry. Intervals end where x crosses zero, so `reciprocal` is never compared across its pole, and no pair spans an unscorable point. Ordering is not accuracy: an op can sit within 1 ULP and still step backwards |
| `nonfinite` | `total` and the split by side — `both`, `device_only`, `golden_only`, `device_inf`, `device_nan`, `golden_inf`, `golden_nan` — plus up to ten points. These carry no ULP, so no other field counts them. Absent when every point was finite |
| `defects` | device returned inf or zero where the reference is representable. These carry no ULP, so no other field counts them. A `mismatch` against a NaN reference is a domain disagreement, not a defect |
| `verdict` | one of the seven phrases below, precomputed |
| `perf` | `us_median` and `melem_per_s` for one dispatch over 2²⁴ resident elements, with the `host` that took them. `spread_pct` appears only when the row is not trustworthy. Never scored |
| `rationale` | why this variant has these parameters, from `ops/overrides.py` |
| `_runs.{arch}.{dtype}` | tt-metal commit, device, versions. Every claim is per build |

## Outcome labels

| Label | Meaning |
|---|---|
| `exact` | bit-identical to the reference |
| `faithful` | differs by exactly one representable step, within 1 ULP: the device took the reference's other neighbour. Both are within half a ULP of the true value, so this is a tie-breaking rule that differs from the reference's, not an accuracy shortfall. Wormhole and Blackhole break ties away from zero (`RoundMode::Nearest` is `NearestAway`), PyTorch breaks them to even |
| `inexact` | differs, ULP defined |
| `flushed` | reference is subnormal, the dtype cannot hold it |
| `zeroed` | hardware returned 0 for a representable reference. Real, but ULP cannot express it |
| `unflushed` | reference underflows to zero, hardware returned a normal value. ULP over the subnormal gap would read 1e73 |
| `overflow` | true result exceeds the dtype |
| `undefined` | reference is NaN, outside the mathematical domain |
| `mismatch` | one side finite, the other not. A defect |
| `special` | a special-values row, excluded from every statistic |

## Verdict rules

Computed, never inferred. First match wins.

| Verdict | Rule |
|---|---|
| `N of M points returned inf or zero where a value exists` | defects > 0. Takes precedence, because those points carry no ULP and the figures below exclude them |
| `N of M points returned a value where the reference is zero` | unflushed > 0. Also unscorable, and otherwise invisible: those points would read bit-exact |
| `bit-exact` | no point is `faithful` or `inexact`, so every scorable one is the dtype's correctly rounded answer. Not `max_ulp = 0`: against a wider reference such a variant still reads up to 0.5 ULP |
| `faithfully rounded; N of M points took the other neighbour` | no point is `inexact`, but N are `faithful`. The device is never more than one step out and both steps are within half a ULP of the truth; it breaks ties the other way |
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

## Results — 867 variants

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
| wh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 399 | 42064 |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1312 | 12783 |
| wh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 679 | 24697 |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949; up to 2.7 ULP beyond | 2.7 | 0.744 | 0.949 | 7532 | 2228 |
| wh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1.41 | 0.596 | 3.39e+38 | 923 | 18178 |
| wh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 3.26 | 0.688 | 1.03 | 8665 | 1936 |
| wh | bf16 | `add` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 536 | 31321 |
| wh | bf16 | `add_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 548 | 30596 |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | — | 545 ±6% | 30797 |
| wh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.35e+03 | 2.11e+06 | 2.35e+03 | — | 693 | 24218 |
| wh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.4 | 126 | 19.4 | — | 684 | 24540 |
| wh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists | 0.499 | 0 | — | 675 | 24864 |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 2.76 ULP beyond | 2.76 | 0.787 | 0.938 | 7448 | 2252 |
| wh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1.32 | 0.619 | 3.39e+38 | 1411 | 11890 |
| wh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1.33 | 0.681 | 1.84e+19 | 1352 | 12411 |
| wh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.527 | 0.513 | — | 602 ±17% | 27872 |
| wh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 2.51 | 200 | 2.51 | — | 563 ±6% | 29823 |
| wh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.88 | 248 | 4.88 | — | 5519 | 3040 |
| wh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 2.23 | 0.867 | 0.23 | 1342 | 12501 |
| wh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists | 2.08 | 0.995 | — | 686 | 24462 |
| wh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 8.17 | 1.02 | 0.82 | 8471 | 1980 |
| wh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 543 | 30908 |
| wh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 547 | 30655 |
| wh | bf16 | `cbrt` | `default` | faithfully rounded; 846 of 65024 points took the other neighbour | 0.507 | 0.502 | 3.39e+38 | 423 ±9% | 39645 |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 411 ±8% | 40840 |
| wh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 479 ±13% | 34992 |
| wh | bf16 | `celu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 3.39e+38 | 2717 | 6174 |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 401 ±7% | 41809 |
| wh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 410 ±8% | 40922 |
| wh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 401 | 41789 |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | — | 2476 | 6777 |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 405 ±8% | 41421 |
| wh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 406 ±8% | 41359 |
| wh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 400 | 41933 |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | — | 2464 | 6808 |
| wh | bf16 | `cos` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 26.5 | 1.31e+05 | 448 ±12% | 37451 |
| wh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 7.45e+42 | 3.63e+39 | 2.62e+05 | 1733 | 9683 |
| wh | bf16 | `cosh` | `default` | bit-exact | 0.5 | 0 | 89 | 445 ±7% | 37712 |
| wh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0.5 | 0 | 88.5 | 6717 | 2498 |
| wh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.998 | 0.744 | 6.7e-37 | 399 | 42019 |
| wh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 1183 | 14180 |
| wh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.32e+04 | 1 | 12084 | 1388 |
| wh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | — | 544 | 30863 |
| wh | bf16 | `div_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.512 | 0.512 | — | 10548 | 1591 |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | — | 1449 | 11577 |
| wh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | — | 551 | 30445 |
| wh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | — | 562 | 29849 |
| wh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 474 ±9% | 35412 |
| wh | bf16 | `elu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 3.39e+38 | 2742 | 6119 |
| wh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | — | 539 | 31146 |
| wh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | — | 545 | 30763 |
| wh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 400 ±6% | 41923 |
| wh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1.06 | 0.771 | 3.39e+38 | 584 ±15% | 28739 |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 1.5 | 2457 ±5% | 6829 |
| wh | bf16 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero | 3.19e+28 | 3.23e+24 | 2.5 | 833 | 20130 |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 1.5 | 2458 | 6826 |
| wh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists | 121 | 6.4 | 1.31e-38 | 890 | 18861 |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 8.91 ULP beyond | 8.91 | 1.28 | 0.777 | 7879 | 2129 |
| wh | bf16 | `exp` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 3.39e+38 | 414 ±10% | 40514 |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6.08 | 6.08 | 3.05 | — | 399 ±5% | 42085 |
| wh | bf16 | `exp2` | `default` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 3.39e+38 | 443 ±12% | 37889 |
| wh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 1.89 | 0.942 | 128 | 1711 | 9803 |
| wh | bf16 | `exp_bw` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 3.39e+38 | 1313 | 12782 |
| wh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 422 ±7% | 39720 |
| wh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists | 125 | 5.58 | 2.09 | 1718 | 9768 |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 413 ±6% | 40595 |
| wh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | — | 2515 | 6670 |
| wh | bf16 | `fmod` | `default` | worst pairing 6.65e+35 ULP; mean 7.8e+34 | 6.65e+35 | 7.8e+34 | — | 669 | 25083 |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 408 | 41111 |
| wh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | — | 535 | 31346 |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | — | 547 | 30670 |
| wh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists | 165 | 9.18 | 2.33e-38 | 813 | 20646 |
| wh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists | 1.14e+36 | 1.82e+34 | 2.33e-38 | 402 ±8% | 41727 |
| wh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.44; up to 3.5 ULP beyond | 3.5 | 1.37 | 8.44 | 1760 | 9534 |
| wh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 403 ±7% | 41648 |
| wh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | — | 545 ±5% | 30786 |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | — | 549 | 30543 |
| wh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 396 ±6% | 42388 |
| wh | bf16 | `hardmish` | `default` | faithfully rounded; 2587 of 65024 points took the other neighbour | 1 | 0.913 | 3.39e+38 | 408 ±6% | 41088 |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 398 | 42127 |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1681 | 9983 |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.557 | 3.39e+38 | 411 | 40829 |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 3.39e+38 | 2550 | 6580 |
| wh | bf16 | `hardswish` | `default` | 2 of 65024 points returned inf or zero where a value exists | 2.14 | 0.94 | 2.33e-38 | 431 ±15% | 38902 |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12; up to 85.3 ULP beyond | 85.3 | 2.1 | 1.12 | 3557 | 4716 |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 402 ±8% | 41769 |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 2172 | 7726 |
| wh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 411 ±10% | 40792 |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 3.39e+38 | 422 | 39788 |
| wh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 413 ±11% | 40621 |
| wh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 52.7 | 1.58 | — | 549 | 30559 |
| wh | bf16 | `hypot_bw` | `default` | worst pairing 74.6 ULP; mean 2.42 | 74.6 | 2.42 | — | 3382 | 4961 |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.6; up to 255 ULP beyond | 255 | 56.3 | 13.6 | 449 ±13% | 37378 |
| wh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.58 | 2.33e-38 | 2088 | 8037 |
| wh | bf16 | `i1` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.58 | 2.33e-38 | 1197 | 14020 |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 397 | 42285 |
| wh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | — | 557 ±5% | 30104 |
| wh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 398 ±8% | 42183 |
| wh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 397 ±6% | 42274 |
| wh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 418 ±10% | 40126 |
| wh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 407 | 41260 |
| wh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 405 ±6% | 41440 |
| wh | bf16 | `l1_loss` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 542 | 30966 |
| wh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | — | 548 | 30625 |
| wh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | — | 559 | 30028 |
| wh | bf16 | `ldexp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.898 | 0.898 | — | 2734 | 6137 |
| wh | bf16 | `le` | `default` | bit-exact | 0 | 0 | — | 533 | 31495 |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | — | 551 | 30447 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 413 ±8% | 40606 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 15341 of 65024 points took the other neighbour | 0.96 | 0.741 | 3.39e+38 | 402 ±6% | 41757 |
| wh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 401 ±10% | 41816 |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 3.39e+38 | 1837 | 9132 |
| wh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 39.9 | 1.77e+03 | 39.9 | — | 684 | 24529 |
| wh | bf16 | `lerp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.506 | 0.506 | — | 2805 | 5981 |
| wh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 410 ±6% | 40924 |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.414; up to 324 ULP beyond | 324 | 0.846 | 0.414 | 2438 | 6883 |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 2069 | 8108 |
| wh | bf16 | `log` | `default` | faithfully rounded; 63 of 32512 points took the other neighbour | 0.78 | 0.534 | 3.39e+38 | 447 ±12% | 37515 |
| wh | bf16 | `log10` | `default` | faithfully rounded; 58 of 32512 points took the other neighbour | 0.785 | 0.543 | 3.39e+38 | 419 ±7% | 40034 |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 1.74 | 0.796 | 3.69e+37 | 5278 | 3178 |
| wh | bf16 | `log1p` | `default` | faithfully rounded; 109 of 48640 points took the other neighbour | 0.931 | 0.57 | 3.39e+38 | 432 ±8% | 38835 |
| wh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1.42 | 0.68 | 8.47e+37 | 5289 | 3172 |
| wh | bf16 | `log2` | `default` | faithfully rounded; 59 of 32512 points took the other neighbour | 0.766 | 0.542 | 3.39e+38 | 436 ±6% | 38506 |
| wh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists | 1.51 | 0.842 | — | 5280 | 3177 |
| wh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0.512 | 0.507 | 8.47e+37 | 4000 | 4195 |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.426; up to 6.39 ULP beyond | 6.39 | 1.38 | 0.426 | 592 | 28361 |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 1.91 | 0.785 | 3.39e+38 | 5925 | 2832 |
| wh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 804 | 20857 |
| wh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 914 | 18361 |
| wh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 907 | 18504 |
| wh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41.4 ULP; mean 3.6 | 41.4 | 3.6 | — | 5832 | 2877 |
| wh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 782 | 21460 |
| wh | bf16 | `logaddexp_bw` | `default` | worst pairing 62.5 ULP; mean 4.9 | 62.5 | 4.9 | — | 4761 | 3524 |
| wh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | — | 552 | 30404 |
| wh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | — | 559 | 30039 |
| wh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 399 ±13% | 42064 |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 406 | 41295 |
| wh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | — | 542 ±11% | 30952 |
| wh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | — | 560 | 29978 |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | — | 562 | 29861 |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.67 | 0.395 | 891 | 18820 |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.649 | 0.996 | 7002 | 2396 |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.649 | 0.996 | 8260 | 2031 |
| wh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | — | 551 | 30458 |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | — | 550 ±7% | 30519 |
| wh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 396 ±5% | 42415 |
| wh | bf16 | `mac` | `default` | worst pairing 63.5 ULP; mean 26.4 | 63.5 | 26.4 | — | 683 | 24559 |
| wh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 5471 | 3067 |
| wh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | — | 550 | 30482 |
| wh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 5473 | 3065 |
| wh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | — | 548 | 30603 |
| wh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists | 1.49 | 0.667 | 1.95e-38 | 607 ±8% | 27659 |
| wh | bf16 | `mse_loss` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | — | 541 | 30994 |
| wh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | — | 1424 | 11783 |
| wh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists | 724 | 1.67 | 5.55e-17 | 12636 | 1328 |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 293 | 5.55e-17 | 9666 | 1736 |
| wh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | — | 538 ±8% | 31185 |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | — | 557 ±7% | 30120 |
| wh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | — | 544 ±6% | 30849 |
| wh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | — | 552 | 30419 |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 395 | 42464 |
| wh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | — | 3127 | 5366 |
| wh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 401 ±6% | 41877 |
| wh | bf16 | `polygamma` | `k=1` | 7 of 49922 points returned inf or zero where a value exists | 1.02e+08 | 1.43e+05 | 0.996 | 5560 | 3018 |
| wh | bf16 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 4.47 | 5627 | 2981 |
| wh | bf16 | `polygamma` | `k=4` | 142 of 49922 points returned inf or zero where a value exists | 3.22e+09 | 1.09e+07 | 3.48 | 5731 | 2928 |
| wh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 4.47 | 12071 | 1390 |
| wh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | — | 1001 | 16766 |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1.69e+38 | 2565 | 6541 |
| wh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists | 0 | 0 | 4.68e-38 | 406 ±6% | 41370 |
| wh | bf16 | `rad2deg` | `default` | faithfully rounded; 31760 of 63518 points took the other neighbour | 0.992 | 0.75 | 5.9e+36 | 403 ±7% | 41604 |
| wh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 0.512 | 0.507 | 8.47e+37 | 412 ±8% | 40725 |
| wh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists | 1.51 | 0.835 | 7.67e-20 | 6915 | 2426 |
| wh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0.512 | 0.507 | 8.47e+37 | 411 ±9% | 40828 |
| wh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 1.51 | 0.835 | 5.42e-20 | 5138 | 3265 |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 413 ±6% | 40611 |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 400 ±7% | 41993 |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 3962 | 4235 |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1311 ±20% | 12796 |
| wh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 406 ±5% | 41338 |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 411 ±8% | 40850 |
| wh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 3.39e+38 | 400 ±6% | 41906 |
| wh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 401 | 41825 |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 400 | 41894 |
| wh | bf16 | `remainder` | `default` | worst pairing 6.65e+35 ULP; mean 1.06e+35 | 6.65e+35 | 1.06e+35 | — | 668 | 25112 |
| wh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 404 ±13% | 41556 |
| wh | bf16 | `rpow` | `exponent=0.5` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 3.39e+38 | 964 | 17411 |
| wh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 976 | 17196 |
| wh | bf16 | `rpow` | `exponent=2.0` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 3.39e+38 | 956 | 17540 |
| wh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | 0 | 1.18e-38 | 2933 | 5719 |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 3.39e+38 | 443 ±10% | 37902 |
| wh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists | 3.23 | 1.19 | — | 6328 | 2652 |
| wh | bf16 | `rsub` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 547 | 30655 |
| wh | bf16 | `rsub_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 562 ±11% | 29860 |
| wh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 499 ±13% | 33624 |
| wh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 2.62 | 1.06 | 0.00443 | 3113 | 5389 |
| wh | bf16 | `sigmoid` | `default` | faithfully rounded; 505 of 65024 points took the other neighbour | 0.857 | 0.517 | 3.39e+38 | 465 ±6% | 36046 |
| wh | bf16 | `sigmoid_accurate` | `default` | faithfully rounded; 505 of 65024 points took the other neighbour | 0.857 | 0.517 | 3.39e+38 | 467 ±9% | 35903 |
| wh | bf16 | `sigmoid_bw` | `default` | 329 of 65024 points returned inf or zero where a value exists | 266 | 5.5 | 1.77 | 2252 | 7450 |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 396 ±7% | 42375 |
| wh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 396 | 42372 |
| wh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists | 0.914 | 0.585 | 2.33e-38 | 475 ±12% | 35336 |
| wh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 285 | 1.31 | 0.867 | 3100 | 5412 |
| wh | bf16 | `sin` | `default` | accurate to |x| <= 2.62e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 73.5 | 2.62e+05 | 433 ±9% | 38735 |
| wh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 1.94e+42 | 1.95e+39 | 1.31e+05 | 1344 | 12478 |
| wh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 89 | 546 ±25% | 30719 |
| wh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0.5 | 0 | 88.5 | 5864 | 2861 |
| wh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists | 0.775 | 0.556 | 5.03 | 489 ±7% | 34303 |
| wh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 1.91 | 0.746 | 3.39e+38 | 4373 | 3837 |
| wh | bf16 | `softshrink` | `default` | faithfully rounded; 3968 of 65024 points took the other neighbour | 1 | 0.948 | 3.39e+38 | 408 ±8% | 41072 |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 2172 | 7724 |
| wh | bf16 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 1 | 0.641 | 8.51e+37 | 437 ±7% | 38378 |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391; up to 81 ULP beyond | 81 | 3.29 | 0.00391 | 1342 | 12500 |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 439 ±10% | 38231 |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1.14 | 0.695 | 3.39e+38 | 6593 | 2545 |
| wh | bf16 | `square` | `default` | faithfully rounded; 13970 of 48640 points took the other neighbour | 0.973 | 0.688 | 1.84e+19 | 399 ±5% | 42090 |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1.69e+38 | 1295 | 12957 |
| wh | bf16 | `squared_difference` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | — | 543 ±8% | 30884 |
| wh | bf16 | `squared_difference_` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | — | 548 | 30608 |
| wh | bf16 | `squared_difference_bw` | `default` | faithfully rounded; 64842 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 2177 | 7706 |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | — | 543 | 30910 |
| wh | bf16 | `subtract` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 531 | 31585 |
| wh | bf16 | `subtract_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 560 | 29958 |
| wh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists | 0.914 | 0.585 | 2.33e-38 | 480 ±10% | 34941 |
| wh | bf16 | `tan` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 47.6 | 1.31e+05 | 577 ±14% | 29070 |
| wh | bf16 | `tan_bw` | `default` | 12178 of 65024 points returned inf or zero where a value exists | 3.1e+40 | 1.95e+37 | 1.13 | 2183 | 7686 |
| wh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.811 | 0.586 | — | 406 | 41280 |
| wh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 1.93 | 17.2 | 1468 | 11430 |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.38 | 1.35e-08 | 442 ±8% | 37998 |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.52 | 7.45e-09 | 1690 | 9929 |
| wh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 415 ±11% | 40451 |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 404 | 41532 |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 1670 | 10044 |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 396 ±5% | 42375 |
| wh | bf16 | `where` | `default` | bit-exact | 0 | 0 | — | 679 | 24691 |
| wh | bf16 | `xielu` | `default` | 1 of 56847 points returned inf or zero where a value exists | 0.5 | 0.5 | 2.33e-38 | 1013 | 16568 |
| wh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | — | 548 | 30627 |
| wh | bf16 | `xlogy_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.78 | 0.78 | — | 8907 | 1884 |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 782 | 21453 |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 2497 | 6718 |
| wh | fp32 | `acos` | `default` | within 2 ULP everywhere | 1.55 | 0.865 | 1 | 1103 | 15204 |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.926 | 14951 | 1122 |
| wh | fp32 | `acosh` | `default` | never within 2 ULP; mean 0.764, worst 2.76 | 2.76 | 0.764 | — | 1412 ±14% | 11886 |
| wh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 724 | 1.13 | 0.996 | 17197 | 976 |
| wh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | — | 984 | 17054 |
| wh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | — | 1019 | 16466 |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | — | 998 | 16809 |
| wh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | — | 1315 | 12760 |
| wh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | — | 1314 | 12772 |
| wh | fp32 | `asin` | `default` | within 2 ULP everywhere | 1.77 | 0.795 | 1 | 1064 | 15773 |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.926 | 14786 | 1135 |
| wh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 1.72 | 0.774 | 3.39e+38 | 1795 | 9345 |
| wh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1.98 | 1.06 | 1.84e+19 | 2615 | 6416 |
| wh | fp32 | `atan` | `default` | accurate to |x| <= 0.902; up to 2.34 ULP beyond | 2.34 | 0.807 | 0.902 | 922 | 18194 |
| wh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | — | 1009 | 16628 |
| wh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 9.5e+04 | 1.64e+07 | 9.5e+04 | — | 10596 | 1583 |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 1.01; up to 2.8 ULP beyond | 2.8 | 1.23 | 1.01 | 2569 | 6531 |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 4.28e-07; up to 3.11 ULP beyond | 3.11 | 1.55 | 4.28e-07 | 1300 ±10% | 12908 |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.678; up to 2.05e+03 ULP beyond | 2.05e+03 | 1.55 | 0.678 | 16427 | 1021 |
| wh | fp32 | `bias_gelu` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | — | 989 | 16956 |
| wh | fp32 | `bias_gelu_` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | — | 1012 | 16585 |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38; up to 2.55 ULP beyond | 2.55 | 1.76 | 2.26e-38 | 852 | 19696 |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 784 | 21400 |
| wh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 3.39e+38 | 951 ±6% | 17636 |
| wh | fp32 | `celu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 5451 | 3078 |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 779 | 21534 |
| wh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 789 | 21257 |
| wh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 780 | 21498 |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | — | 4636 | 3619 |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 776 | 21632 |
| wh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 777 ±5% | 21581 |
| wh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 779 | 21542 |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | — | 4628 | 3626 |
| wh | fp32 | `cos` | `default` | accurate to |x| <= 92.4; up to 3.3e+12 ULP beyond | 3.3e+12 | 3.33e+09 | 92.4 | 870 | 19287 |
| wh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 2.68e+51 | 2.8e+48 | 28 | 3323 | 5049 |
| wh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1.35 | 0.798 | 89.1 | 975 | 17204 |
| wh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 2.21 | 1.17 | 0.0155 | 13573 | 1236 |
| wh | fp32 | `deg2rad` | `default` | faithfully rounded; 63542 of 65024 points took the other neighbour | 0.63 | 0.595 | 3.4e+38 | 775 | 21657 |
| wh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 2626 | 6389 |
| wh | fp32 | `digamma_bw` | `default` | 255 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 5.4e-20 | 17892 | 938 |
| wh | fp32 | `div` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | — | 1003 | 16722 |
| wh | fp32 | `div_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.84 | 0.84 | — | 20768 | 808 |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP everywhere | 1.79 | 1.52 | — | 3465 | 4841 |
| wh | fp32 | `divide` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | — | 998 | 16805 |
| wh | fp32 | `divide_` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | — | 1025 | 16366 |
| wh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 3.39e+38 | 951 | 17646 |
| wh | fp32 | `elu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 5427 | 3091 |
| wh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | — | 993 | 16891 |
| wh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | — | 1011 | 16591 |
| wh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 774 | 21670 |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 6.48 ULP beyond | 6.48 | 1.35 | 0.000334 | 1226 | 13689 |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.348 | 4887 | 3433 |
| wh | fp32 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero | 2.1e+33 | 1.71e+29 | — | 1312 ±9% | 12787 |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.348 | 4890 | 3431 |
| wh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists | 7.96e+06 | 2.62e+05 | 1.32e-38 | 1309 ±11% | 12814 |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456; up to 4.98e+05 ULP beyond | 4.98e+05 | 1.32e+03 | 0.000456 | 15329 | 1094 |
| wh | fp32 | `exp` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 894 | 18764 |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | — | 778 ±7% | 21557 |
| wh | fp32 | `exp2` | `default` | faithfully rounded; 6044 of 49536 points took the other neighbour | 0.965 | 0.626 | 3.39e+38 | 843 | 19896 |
| wh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 1.85 | 1.09 | 128 | 3336 | 5028 |
| wh | fp32 | `exp_bw` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 2610 ±10% | 6428 |
| wh | fp32 | `expm1` | `default` | faithfully rounded; 5257 of 49458 points took the other neighbour | 0.997 | 0.565 | 3.39e+38 | 1041 | 16119 |
| wh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists | 4.91e+06 | 1.14e+04 | 1.39 | 3515 | 4774 |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 781 | 21479 |
| wh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.77e+03 | 4.19e+06 | 1.77e+03 | — | 4773 | 3515 |
| wh | fp32 | `fmod` | `default` | worst pairing 1.43e+45 ULP; mean 1.86e+41 | 1.43e+45 | 1.86e+41 | — | 1011 | 16596 |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 778 | 21574 |
| wh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | — | 1001 | 16766 |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | — | 1014 | 16554 |
| wh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists | 1.51e+08 | 1.55e+05 | 0.253 | 1315 ±11% | 12761 |
| wh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists | 7.45e+40 | 1.18e+39 | 2.33e-38 | 782 | 21459 |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0296; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.0296 | 2093 | 8015 |
| wh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 779 | 21542 |
| wh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | — | 990 | 16946 |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | — | 1012 | 16578 |
| wh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 773 | 21696 |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 782 | 21451 |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 779 | 21525 |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 3251 | 5161 |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25; up to 4.89e+06 ULP beyond | 4.89e+06 | 3e+03 | 2.25 | 791 | 21220 |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 5064 ±6% | 3313 |
| wh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36; up to 7.34e+06 ULP beyond | 7.34e+06 | 1.41e+03 | 2.36 | 827 | 20291 |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.00133 | 7091 | 2366 |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 780 | 21508 |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 4281 | 3919 |
| wh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 784 | 21399 |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 3.39e+38 | 789 | 21269 |
| wh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 783 ±7% | 21421 |
| wh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | — | 1018 | 16488 |
| wh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 4.86e+06 | 4.48e+04 | — | 6385 ±8% | 2628 |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.43; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 2.43 | 902 | 18601 |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00626; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.00626 | 3447 | 4868 |
| wh | fp32 | `i1` | `default` | accurate to |x| <= 0.00626; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.00626 | 1733 | 9682 |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 777 | 21600 |
| wh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | — | 1008 | 16644 |
| wh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 776 | 21617 |
| wh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 772 | 21730 |
| wh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 782 | 21449 |
| wh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 776 | 21633 |
| wh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 780 | 21522 |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | — | 1005 | 16700 |
| wh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | — | 1002 | 16749 |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | — | 1018 | 16478 |
| wh | fp32 | `ldexp_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.742 | 0.742 | — | 5270 | 3184 |
| wh | fp32 | `le` | `default` | bit-exact | 0 | 0 | — | 982 | 17089 |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | — | 1019 | 16468 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 772 | 21726 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 31672 of 65024 points took the other neighbour | 0.84 | 0.747 | 3.39e+38 | 773 | 21694 |
| wh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 776 | 21616 |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 3.39e+38 | 3584 | 4681 |
| wh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | — | 1309 | 12820 |
| wh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | — | 5413 | 3100 |
| wh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 781 | 21468 |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.0181 | 3740 | 4486 |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 4339 | 3867 |
| wh | fp32 | `log` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.961 | 0.562 | 3.39e+38 | 882 | 19026 |
| wh | fp32 | `log10` | `default` | accurate to |x| <= 0.336; up to 2.13 ULP beyond | 2.13 | 1.36 | 0.336 | 883 | 19008 |
| wh | fp32 | `log10_bw` | `default` | accurate to |x| <= 2.04e-38; up to 2.1 ULP beyond | 2.1 | 1.41 | 2.04e-38 | 10262 | 1635 |
| wh | fp32 | `log1p` | `default` | faithfully rounded; 20111 of 48640 points took the other neighbour | 0.984 | 0.558 | 3.39e+38 | 911 | 18408 |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP everywhere | 1.89 | 0.769 | 8.51e+37 | 10297 | 1629 |
| wh | fp32 | `log2` | `default` | accurate to |x| <= 0.704; up to 2.43 ULP beyond | 2.43 | 0.511 | 0.704 | 888 | 18902 |
| wh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists | 1.9 | 1.3 | — | 10257 | 1636 |
| wh | fp32 | `log_bw` | `default` | faithfully rounded; 64512 of 64514 points took the other neighbour | 0.892 | 0.714 | 8.51e+37 | 7789 | 2154 |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | — | 1097 | 15292 |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.00164; up to 2.95 ULP beyond | 2.95 | 1.28 | 0.00164 | 11556 | 1452 |
| wh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 1498 | 11201 |
| wh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 1301 | 12900 |
| wh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 1296 | 12942 |
| wh | fp32 | `logaddexp2_bw` | `default` | worst pairing 45.7 ULP; mean 6.68 | 45.7 | 6.68 | — | 11300 | 1485 |
| wh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 1488 | 11272 |
| wh | fp32 | `logaddexp_bw` | `default` | worst pairing 65.7 ULP; mean 8.66 | 65.7 | 8.66 | — | 9454 | 1775 |
| wh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | — | 1004 | 16718 |
| wh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | — | 1026 | 16358 |
| wh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 773 | 21694 |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 773 | 21718 |
| wh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | — | 1002 | 16737 |
| wh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | — | 1024 | 16390 |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | — | 1030 | 16292 |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.266; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.266 | 1044 ±5% | 16071 |
| wh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.98e-08; up to 2.36 ULP beyond | 2.36 | 0.839 | 2.98e-08 | 13857 | 1211 |
| wh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.98e-08; up to 2.36 ULP beyond | 2.36 | 0.839 | 2.98e-08 | 16440 | 1020 |
| wh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | — | 990 | 16941 |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | — | 1003 | 16724 |
| wh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 772 ±6% | 21729 |
| wh | fp32 | `mac` | `default` | worst pairing 4.22e+06 ULP; mean 7.31e+05 | 4.22e+06 | 7.31e+05 | — | 1315 | 12760 |
| wh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 10514 | 1596 |
| wh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | — | 999 | 16790 |
| wh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 10519 | 1595 |
| wh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | — | 987 | 16998 |
| wh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 7 | 2.49 | 8.61e-06 | 1221 | 13738 |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | — | 1000 | 16782 |
| wh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | — | 2709 | 6193 |
| wh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists | 3.51e+07 | 9.16e+03 | 5.59e-17 | 20823 | 806 |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 5.55e-17 | 20192 | 831 |
| wh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | — | 1000 | 16784 |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | — | 1019 | 16466 |
| wh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | — | 993 | 16896 |
| wh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | — | 1015 | 16530 |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 772 | 21726 |
| wh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | — | 6071 | 2764 |
| wh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 773 | 21698 |
| wh | fp32 | `polygamma` | `k=1` | accurate to |x| <= 5.4e-20; up to 7.91e+33 ULP beyond | 7.91e+33 | 4.68e+29 | 5.4e-20 | 5712 | 2937 |
| wh | fp32 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 5899 | 2844 |
| wh | fp32 | `polygamma` | `k=4` | 141 of 49922 points returned inf or zero where a value exists | 3.77e+25 | 6.51e+21 | 3.68e-08 | 6299 | 2664 |
| wh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 18137 | 925 |
| wh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.17 | 5.92e+03 | 3.17 | — | 2154 | 7788 |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1.69e+38 | 4989 | 3363 |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | 0 | 3.39e+38 | 787 | 21306 |
| wh | fp32 | `rad2deg` | `default` | faithfully rounded; 63518 of 63518 points took the other neighbour | 0.696 | 0.643 | 5.93e+36 | 780 | 21519 |
| wh | fp32 | `rdiv` | `value=2.0` | 256 of 64770 points returned inf or zero where a value exists | 0.892 | 0.714 | 8.51e+37 | 806 | 20814 |
| wh | fp32 | `rdiv_bw` | `scalar=2.0` | 252 of 48490 points returned inf or zero where a value exists | 1.84 | 1.22 | 7.67e-20 | 13408 | 1251 |
| wh | fp32 | `reciprocal` | `default` | faithfully rounded; 64512 of 64514 points took the other neighbour | 0.892 | 0.714 | 8.51e+37 | 791 | 21204 |
| wh | fp32 | `reciprocal_bw` | `default` | 254 of 48386 points returned inf or zero where a value exists | 1.84 | 1.22 | 5.42e-20 | 10055 | 1668 |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 783 | 21437 |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 771 | 21768 |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 7884 | 2128 |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 2502 | 6705 |
| wh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 780 | 21496 |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 782 | 21443 |
| wh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 3.39e+38 | 773 | 21702 |
| wh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 776 | 21620 |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 773 | 21717 |
| wh | fp32 | `remainder` | `default` | worst pairing 1.43e+45 ULP; mean 1.68e+41 | 1.43e+45 | 1.68e+41 | — | 999 | 16799 |
| wh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 778 ±6% | 21563 |
| wh | fp32 | `rpow` | `exponent=0.5` | faithfully rounded; 5624 of 49536 points took the other neighbour | 0.879 | 0.609 | 3.39e+38 | 1831 | 9164 |
| wh | fp32 | `rpow` | `exponent=1.0` | 1536 of 49536 points returned inf or zero where a value exists | 0 | 0 | 8.28e+34 | 1840 | 9119 |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists | 0.879 | 0.609 | 8.28e+34 | 1830 | 9167 |
| wh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | 0 | 1.18e-38 | 5722 | 2932 |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1.12 | 0.834 | 3.39e+38 | 847 | 19812 |
| wh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists | 7.19 | 4.49 | — | 11962 | 1403 |
| wh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | — | 990 | 16945 |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | — | 1008 | 16645 |
| wh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.3, worst 51.3 | 51.3 | 26.3 | — | 974 | 17227 |
| wh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 50.7 | 20.4 | — | 6201 | 2705 |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 0.000345; up to 2.64 ULP beyond | 2.64 | 1.31 | 0.000345 | 1079 | 15547 |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 0.000345; up to 2.64 ULP beyond | 2.64 | 1.31 | 0.000345 | 1078 | 15557 |
| wh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists | 8.39e+06 | 2.71e+04 | 0.447 | 4530 | 3704 |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 774 | 21675 |
| wh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 778 | 21554 |
| wh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists | 2.94 | 1.55 | 0.000121 | 1098 | 15275 |
| wh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 5.72e+06 | 841 | 2.98e-07 | 6255 | 2682 |
| wh | fp32 | `sin` | `default` | accurate to |x| <= 28; up to 2.2e+12 ULP beyond | 2.2e+12 | 2.68e+09 | 28 | 852 | 19698 |
| wh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 1.56e+53 | 2.03e+49 | 92.4 | 2588 | 6484 |
| wh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155; up to 2.21 ULP beyond | 2.21 | 1.17 | 0.0155 | 1117 | 15014 |
| wh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1.35 | 0.798 | 88.5 | 11917 | 1408 |
| wh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists | 8.21e+03 | 656 | — | 1336 ±15% | 12560 |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.00163; up to 2.95 ULP beyond | 2.95 | 1.34 | 0.00163 | 8699 | 1929 |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 780 | 21512 |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 4267 | 3932 |
| wh | fp32 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 2.66 | 0.897 | 0.000462 | 825 | 20336 |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 1.79e-07 | 2591 | 6475 |
| wh | fp32 | `sqrt` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.867 | 0.827 | 3.39e+38 | 831 | 20186 |
| wh | fp32 | `sqrt_bw` | `default` | never within 2 ULP; mean 1.44, worst 2.24 | 2.24 | 1.44 | — | 12726 | 1318 |
| wh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1.84e+19 | 776 | 21606 |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1.69e+38 | 2507 | 6692 |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | — | 989 | 16957 |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | — | 1010 | 16612 |
| wh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | — | 4228 | 3968 |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | — | 1002 | 16741 |
| wh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | — | 984 | 17042 |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | — | 1018 | 16488 |
| wh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 2.94 | 1.55 | 0.000121 | 1099 | 15267 |
| wh | fp32 | `tan` | `default` | accurate to |x| <= 3.92; up to 2.2e+12 ULP beyond | 2.2e+12 | 5.61e+09 | 3.92 | 1203 ±6% | 13947 |
| wh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists | 2.85e+45 | 5.11e+44 | 0.882 | 4434 | 3784 |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000439; up to 2.79 ULP beyond | 2.79 | 1.49 | 0.000439 | 938 | 17877 |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | — | 1778 | 9434 |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.35e+04 | 1.34e-08 | 1329 ±14% | 12623 |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.01e+05 | 7.42e-09 | 3409 | 4921 |
| wh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 777 ±6% | 21579 |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 777 | 21590 |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 3261 | 5145 |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 772 | 21728 |
| wh | fp32 | `where` | `default` | bit-exact | 0 | 0 | — | 1307 | 12836 |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 1.71e-13 | 1346 ±15% | 12460 |
| wh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | — | 998 | 16811 |
| wh | fp32 | `xlogy_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.766 | 0.766 | — | 17519 | 958 |
