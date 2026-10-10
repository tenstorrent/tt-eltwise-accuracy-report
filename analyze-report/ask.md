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
| bh | bf16 | `de546d3b146` | 0.79.0 | 25a8d03fd5da |
| bh | fp32 | `de546d3b146` | 0.79.0 | 25a8d03fd5da |
| wh | bf16 | `de546d3b146` | 0.79.0 | 312b06a002e7 |
| wh | fp32 | `de546d3b146` | 0.79.0 | 312b06a002e7 |

**Status: in force.** The committed `report_index.json` carries `max_ulp`, `mean_ulp`,
`usable_to`, `max_abs`, `ulp_clipped`, `n_inputs`, `outcomes`, `specials`, `defects`,
`verdict`, `perf`, and `rationale`. Fields below that are missing from an entry appear
after the next published sweep — the CSVs currently on disk predate `n_rounded` /
`ulp_signed` / `faithful`, so `charts` cannot backfill them.

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
| `verdict` | one of the nine phrases below, precomputed |
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
| `N/M defects; rest X ULP` | defects > 0. Leads, because those points carry no ULP and every figure below excludes them — but it does not replace the accuracy of the points that do score. Two defects once hid 9.02e+04 ULP on the other 64,512 |
| `N/M unflushed; rest X ULP` | unflushed > 0. Also unscorable, and otherwise invisible: those points would read bit-exact |
| `bit-exact` | no point is `faithful` or `inexact`, so every scorable one is the dtype's correctly rounded answer. Not `max_ulp = 0`: against a wider reference such a variant still reads up to 0.5 ULP |
| `faithful; N/M tie-breaks` | no point is `inexact`, but N are `faithful`. The device is never more than one step out and both steps are within half a ULP of the truth; it breaks ties the other way |
| `within 2 ULP` | max_ulp ≤ 2 |
| `accurate to \|x\| <= B` | unary, usable_to = B defined |
| `never within 2 ULP` | unary, no point within 2 ULP |
| `worst sampled pairing` | multi-operand: the maximum is over sampled pairings, so it is a lower bound |
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
7. Do not subtract `max_ulp` or `mean_ulp` from an index that lacks `rounded_frac` against
   one that has it. Those ULP figures are different quantities. `ttnn-accuracy compare`
   already refuses that; a nightly of two pre-`rounded_frac` indexes still scores ULP.

## Results — 953 variants

| Arch | Dtype | Op | Parameters | Verdict | Max ULP | Mean ULP | Rounded | Usable to | µs | Melem/s |
|---|---|---|---|---|---|---|---|---|---|---|
| bh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88313 |
| bh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 595 | 28179 |
| bh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 271 | 61891 |
| bh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.744 | 0.996 | 0.949 | 3518 | 4770 |
| bh | bf16 | `acosh` | `default` | within 2 ULP | 1.41 | 0.596 | 0.973 | 3.39e+38 | 320 ±7% | 52434 |
| bh | bf16 | `acosh_bw` | `default` | 15874/64514 defects; rest 3.26 ULP | 3.26 | 0.688 | 0.781 | 1.03 | 4043 | 4149 |
| bh | bf16 | `add` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 243 | 68967 |
| bh | bf16 | `add_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 243 ±7% | 69138 |
| bh | bf16 | `addalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 245 | 68347 |
| bh | bf16 | `addcdiv` | `default` | worst sampled pairing | 2.11e+06 | 2.35e+03 | 0.999 | — | 349 | 48136 |
| bh | bf16 | `addcmul` | `default` | worst sampled pairing | 126 | 19.4 | 1 | — | 347 | 48416 |
| bh | bf16 | `asin` | `default` | 2/32258 defects; rest 0.499 ULP | 0.499 | 0 | 1 | — | 263 | 63780 |
| bh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938 | 2.76 | 0.787 | 0.992 | 0.938 | 3449 | 4864 |
| bh | bf16 | `asinh` | `default` | within 2 ULP | 1.32 | 0.615 | 0.991 | 3.39e+38 | 448 | 37444 |
| bh | bf16 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.33 ULP | 1.33 | 0.681 | 0.918 | 1.84e+19 | 575 | 29184 |
| bh | bf16 | `atan` | `default` | 2/65024 defects; rest 0.527 ULP | 0.527 | 0.513 | 1 | — | 212 ±23% | 79022 |
| bh | bf16 | `atan2` | `default` | worst sampled pairing | 200 | 3.56 | 0.901 | — | 256 ±6% | 65436 |
| bh | bf16 | `atan2_bw` | `default` | worst sampled pairing | 248 | 4.89 | 0.508 | — | 2510 | 6683 |
| bh | bf16 | `atan_bw` | `default` | 2/48386 defects; rest 2.23 ULP | 2.23 | 0.867 | 0.849 | 0.23 | 614 | 27316 |
| bh | bf16 | `atanh` | `default` | 2/32256 defects; rest 2.08 ULP | 2.08 | 0.995 | 0.175 | — | 248 | 67560 |
| bh | bf16 | `atanh_bw` | `default` | 2/48386 defects; rest 8.17 ULP | 8.17 | 1 | 0.79 | 0.82 | 3849 | 4359 |
| bh | bf16 | `bias_gelu` | `default` | worst sampled pairing | 1.14e+36 | 4.5e+34 | 0.749 | — | 240 | 69810 |
| bh | bf16 | `bias_gelu_` | `default` | worst sampled pairing | 1.14e+36 | 4.5e+34 | 0.749 | — | 244 ±5% | 68683 |
| bh | bf16 | `cbrt` | `default` | faithful; 846/65024 tie-breaks | 0.507 | 0.502 | 0.987 | 3.39e+38 | 213 | 78612 |
| bh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 206 ±18% | 81511 |
| bh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 192 | 87400 |
| bh | bf16 | `celu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1264 | 13272 |
| bh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 ±12% | 85501 |
| bh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 186 | 90114 |
| bh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±10% | 87415 |
| bh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1105 | 15182 |
| bh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89088 |
| bh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88819 |
| bh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88807 |
| bh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1100 | 15251 |
| bh | bf16 | `cos` | `default` | faithful; 2/37354 tie-breaks | 0.552 | 0.552 | 1 | 9.99e+05 | 242 | 69190 |
| bh | bf16 | `cos_bw` | `default` | 21281/65024 defects; rest 2.09e+42 ULP | 2.09e+42 | 2.59e+39 | 0.867 | 2.15e+06 | 798 | 21016 |
| bh | bf16 | `cosh` | `default` | faithful; 4/33894 tie-breaks | 0.501 | 0.501 | 1 | 89 | 260 ±7% | 64605 |
| bh | bf16 | `cosh_bw` | `default` | 2/33894 defects; rest 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 3064 | 5476 |
| bh | bf16 | `deg2rad` | `default` | 2/65024 defects; rest 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 194 | 86447 |
| bh | bf16 | `digamma` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 373 | 45029 |
| bh | bf16 | `digamma_bw` | `default` | 263/64769 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.36e+04 | 0.674 | 1 | 3160 | 5310 |
| bh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 276 | 60786 |
| bh | bf16 | `div_bw` | `default` | bit-exact | 0.498 | 0 | 1 | — | 1573 | 10669 |
| bh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 660 ±5% | 25419 |
| bh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 251 | 66929 |
| bh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 246 | 68108 |
| bh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 182 ±5% | 92386 |
| bh | bf16 | `elu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1271 | 13200 |
| bh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 267 | 62933 |
| bh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 243 | 69028 |
| bh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 87925 |
| bh | bf16 | `erf` | `default` | within 2 ULP | 1.06 | 0.771 | 0.729 | 3.39e+38 | 199 ±6% | 84462 |
| bh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 1136 ±6% | 14765 |
| bh | bf16 | `erfc` | `default` | 198/65024 unflushed; rest 3.2e+28 ULP | 3.2e+28 | 3.23e+24 | 0.949 | 2.5 | 306 | 54779 |
| bh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 1136 | 14766 |
| bh | bf16 | `erfinv` | `default` | 29424/32256 defects; rest 121 ULP | 121 | 6.34 | 0.391 | 1.31e-38 | 323 | 51959 |
| bh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777 | 8.91 | 1.28 | 0.98 | 0.777 | 3574 | 4695 |
| bh | bf16 | `exp` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 194 | 86693 |
| bh | bf16 | `exp` | `fast_approx` | never within 2 ULP | 6.08 | 3.05 | 0.314 | — | 199 | 84131 |
| bh | bf16 | `exp2` | `default` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 232 | 72464 |
| bh | bf16 | `exp2_bw` | `default` | 1/49537 defects; rest 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 821 | 20430 |
| bh | bf16 | `exp_bw` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 599 | 27995 |
| bh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 266 ±6% | 63119 |
| bh | bf16 | `expm1_bw` | `default` | accurate to |x| <= 33.2 | 34 | 0.695 | 0.984 | 33.2 | 598 | 28033 |
| bh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±18% | 87604 |
| bh | bf16 | `floor_div` | `default` | worst sampled pairing | 64 | 42 | 0.996 | — | 1146 | 14637 |
| bh | bf16 | `fmod` | `default` | worst sampled pairing | 9.14e+35 | 8.94e+34 | 0.673 | — | 279 ±6% | 60105 |
| bh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 203 | 82616 |
| bh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 243 | 69050 |
| bh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 240 | 69838 |
| bh | bf16 | `gelu` | `default` | 86/65024 defects; rest 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 297 | 56429 |
| bh | bf16 | `gelu` | `fast_approx` | 198/65024 defects; rest 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 195 ±6% | 86187 |
| bh | bf16 | `gelu_bw` | `default` | faithful; 38/65024 tie-breaks | 0.939 | 0.681 | 0.999 | 3.39e+38 | 667 ±6% | 25144 |
| bh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88484 |
| bh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 267 ±7% | 62931 |
| bh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 244 | 68863 |
| bh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 ±10% | 87702 |
| bh | bf16 | `hardmish` | `default` | faithful; 2587/65024 tie-breaks | 1 | 0.913 | 0.961 | 3.39e+38 | 190 ±11% | 88176 |
| bh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 187 | 89525 |
| bh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21907 |
| bh | bf16 | `hardsigmoid` | `default` | within 2 ULP | 1 | 0.557 | 0.986 | 3.39e+38 | 193 ±12% | 87035 |
| bh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 1172 | 14318 |
| bh | bf16 | `hardswish` | `default` | 2/65024 defects; rest 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 191 ±9% | 87986 |
| bh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12 | 85.3 | 2.1 | 0.993 | 1.12 | 1645 | 10201 |
| bh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88586 |
| bh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 978 | 17149 |
| bh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88625 |
| bh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88573 |
| bh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88849 |
| bh | bf16 | `hypot` | `default` | 16384/65026 defects; rest 52.7 ULP | 52.7 | 1.58 | 0.947 | — | 259 | 64850 |
| bh | bf16 | `hypot_bw` | `default` | worst sampled pairing | 74.6 | 2.43 | 0.835 | — | 1505 | 11150 |
| bh | bf16 | `i0` | `default` | accurate to |x| <= 13.6 | 255 | 56.3 | 0.952 | 13.6 | 246 ±5% | 68320 |
| bh | bf16 | `i0_bw` | `default` | 4/33904 defects; rest 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 828 | 20263 |
| bh | bf16 | `i1` | `default` | 4/65024 defects; rest 0.858 ULP | 0.858 | 0.589 | 0.516 | 2.33e-38 | 420 | 39940 |
| bh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 187 | 89769 |
| bh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 271 ±51% | 61814 |
| bh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 ±7% | 85837 |
| bh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±6% | 88367 |
| bh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88591 |
| bh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 203 ±13% | 82462 |
| bh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 ±12% | 87769 |
| bh | bf16 | `l1_loss` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 264 ±14% | 63447 |
| bh | bf16 | `ldexp` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 273 | 61440 |
| bh | bf16 | `ldexp_` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 249 | 67296 |
| bh | bf16 | `ldexp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.898 | 0.898 | 0.983 | — | 1191 | 14085 |
| bh | bf16 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 245 | 68455 |
| bh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 243 | 68961 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88295 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithful; 15341/65024 tie-breaks | 0.96 | 0.741 | 0.761 | 3.39e+38 | 190 | 88228 |
| bh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 ±7% | 87717 |
| bh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 866 | 19365 |
| bh | bf16 | `lerp` | `default` | worst sampled pairing | 1.77e+03 | 39.9 | 1 | — | 345 | 48664 |
| bh | bf16 | `lerp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.506 | 0.506 | 0.992 | — | 1258 | 13337 |
| bh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±6% | 88403 |
| bh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.412 | 324 | 0.846 | 0.76 | 0.412 | 846 | 19842 |
| bh | bf16 | `lgamma_bw` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 775 | 21646 |
| bh | bf16 | `log` | `default` | faithful; 63/32512 tie-breaks | 0.78 | 0.534 | 0.998 | 3.39e+38 | 225 | 74647 |
| bh | bf16 | `log10` | `default` | faithful; 58/32512 tie-breaks | 0.785 | 0.543 | 0.998 | 3.39e+38 | 255 ±6% | 65692 |
| bh | bf16 | `log10_bw` | `default` | within 2 ULP | 1.74 | 0.803 | 0.508 | 3.69e+37 | 2446 | 6858 |
| bh | bf16 | `log1p` | `default` | faithful; 109/48640 tie-breaks | 0.931 | 0.57 | 0.998 | 3.39e+38 | 244 ±18% | 68844 |
| bh | bf16 | `log1p_bw` | `default` | 2/64514 defects; rest 1.42 ULP | 1.42 | 0.744 | 0.982 | 8.47e+37 | 2398 | 6996 |
| bh | bf16 | `log2` | `default` | faithful; 59/32512 tie-breaks | 0.766 | 0.542 | 0.998 | 3.39e+38 | 250 | 67209 |
| bh | bf16 | `log2_bw` | `default` | 116/64626 defects; rest 1.51 ULP | 1.51 | 0.847 | 0.57 | — | 2428 | 6910 |
| bh | bf16 | `log_bw` | `default` | 2/64514 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 1849 | 9073 |
| bh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.428 | 6.39 | 1.38 | 0.958 | 0.428 | 223 | 75137 |
| bh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP | 1.71 | 0.722 | 0.976 | 3.39e+38 | 2678 | 6266 |
| bh | bf16 | `logaddexp` | `default` | 15566/65026 defects; rest 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 330 ±6% | 50907 |
| bh | bf16 | `logaddexp2` | `default` | 15488/65026 defects; rest 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 413 ±6% | 40628 |
| bh | bf16 | `logaddexp2_` | `default` | 15488/65026 defects; rest 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 383 | 43783 |
| bh | bf16 | `logaddexp2_bw` | `default` | worst sampled pairing | 41.4 | 3.6 | 0.978 | — | 2505 | 6696 |
| bh | bf16 | `logaddexp_` | `default` | 15566/65026 defects; rest 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 299 | 56034 |
| bh | bf16 | `logaddexp_bw` | `default` | worst sampled pairing | 62.5 | 4.9 | 0.979 | — | 2142 | 7834 |
| bh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 252 | 66625 |
| bh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 249 ±15% | 67443 |
| bh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85450 |
| bh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 187 | 89752 |
| bh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 249 | 67471 |
| bh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 247 | 67878 |
| bh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 270 ±5% | 62169 |
| bh | bf16 | `logit` | `default` | accurate to |x| <= 0.395 | 64 | 1.67 | 0.985 | 0.395 | 324 | 51712 |
| bh | bf16 | `logit_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3044 | 5512 |
| bh | bf16 | `logiteps_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3830 | 4380 |
| bh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 267 ±8% | 62829 |
| bh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 242 | 69324 |
| bh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88149 |
| bh | bf16 | `mac` | `default` | worst sampled pairing | 63.5 | 26.4 | 1 | — | 349 ±10% | 48123 |
| bh | bf16 | `max_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 2489 | 6741 |
| bh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 264 | 63517 |
| bh | bf16 | `min_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 2483 | 6757 |
| bh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 265 ±8% | 63341 |
| bh | bf16 | `mish` | `default` | 11/65024 defects; rest 1.49 ULP | 1.49 | 0.666 | 0.988 | 1.95e-38 | 237 | 70712 |
| bh | bf16 | `mse_loss` | `default` | worst sampled pairing | 2.03 | 1.65 | 0.571 | — | 265 | 63223 |
| bh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 643 | 26101 |
| bh | bf16 | `multigammaln` | `default` | 11536/48328 defects; rest 724 ULP | 724 | 1.67 | 0.717 | 5.59e-17 | 4689 | 3578 |
| bh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 2.38e+06 | 293 | 0.249 | 5.55e-17 | 3723 | 4507 |
| bh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 248 | 67642 |
| bh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 245 ±8% | 68512 |
| bh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 270 ±7% | 62217 |
| bh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 240 | 69784 |
| bh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88573 |
| bh | bf16 | `nextafter` | `default` | worst sampled pairing | 1.3e+33 | 2.32e+31 | 0.56 | — | 1433 | 11711 |
| bh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89023 |
| bh | bf16 | `polygamma` | `k=1` | 7/49922 defects; rest 1.02e+08 ULP | 1.02e+08 | 2.13e+05 | 0.97 | 0.996 | 331 ±7% | 50622 |
| bh | bf16 | `polygamma` | `k=2` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 378 | 44405 |
| bh | bf16 | `polygamma` | `k=4` | 142/49922 defects; rest 3.22e+09 ULP | 3.22e+09 | 1.66e+07 | 0.958 | 3.5 | 422 | 39780 |
| bh | bf16 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 3196 | 5249 |
| bh | bf16 | `pow` | `default` | worst sampled pairing | 4.86e+18 | 2.08e+14 | 0.982 | — | 413 | 40589 |
| bh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1182 | 14195 |
| bh | bf16 | `prelu` | `weight=0.25` | 1/65024 defects; rest 0 ULP | 0 | 0 | 1 | 4.68e-38 | 194 | 86388 |
| bh | bf16 | `rad2deg` | `default` | faithful; 31760/63518 tie-breaks | 0.992 | 0.75 | 0.5 | 5.9e+36 | 191 ±74% | 87881 |
| bh | bf16 | `rdiv` | `value=2.0` | 258/64770 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 193 | 86882 |
| bh | bf16 | `rdiv_bw` | `scalar=2.0` | 256/48492 defects; rest 1.51 ULP | 1.51 | 0.837 | 0.602 | 7.67e-20 | 3155 ±18% | 5318 |
| bh | bf16 | `reciprocal` | `default` | 2/64514 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 190 | 88335 |
| bh | bf16 | `reciprocal_bw` | `default` | 256/48386 defects; rest 1.51 ULP | 1.51 | 0.837 | 0.602 | 5.42e-20 | 2372 | 7074 |
| bh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 88060 |
| bh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 87956 |
| bh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1820 | 9220 |
| bh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 594 ±28% | 28228 |
| bh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88792 |
| bh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±6% | 88443 |
| bh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 | 87389 |
| bh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89254 |
| bh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 ±6% | 87643 |
| bh | bf16 | `remainder` | `default` | worst sampled pairing | 1.08e+36 | 1.22e+35 | 0.672 | — | 265 ±6% | 63269 |
| bh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±14% | 87182 |
| bh | bf16 | `rpow` | `exponent=0.5` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.975 | 3.39e+38 | 377 | 44462 |
| bh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 373 | 44964 |
| bh | bf16 | `rpow` | `exponent=2.0` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 378 | 44379 |
| bh | bf16 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 1353 | 12396 |
| bh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 243 | 68973 |
| bh | bf16 | `rsqrt_bw` | `default` | 75/21665 defects; rest 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 2824 | 5942 |
| bh | bf16 | `rsub` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 243 | 68910 |
| bh | bf16 | `rsub_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 241 ±6% | 69588 |
| bh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 193 | 86879 |
| bh | bf16 | `selu_bw` | `default` | 1/65024 defects; rest 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 1458 | 11506 |
| bh | bf16 | `sigmoid` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 260 | 64406 |
| bh | bf16 | `sigmoid_accurate` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 260 | 64492 |
| bh | bf16 | `sigmoid_bw` | `default` | 331/65024 defects; rest 125 ULP | 125 | 4.81 | 0.988 | 1.77 | 1059 | 15836 |
| bh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88353 |
| bh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89034 |
| bh | bf16 | `silu` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 218 | 76900 |
| bh | bf16 | `silu_bw` | `default` | 9/65024 defects; rest 285 ULP | 285 | 1.3 | 0.979 | 0.867 | 1450 | 11568 |
| bh | bf16 | `sin` | `default` | faithful; 4/37354 tie-breaks | 0.552 | 0.526 | 1 | 9.99e+05 | 222 | 75422 |
| bh | bf16 | `sin_bw` | `default` | 21284/65024 defects; rest 2.98e+41 ULP | 2.98e+41 | 1.39e+39 | 0.857 | 1.07e+06 | 648 | 25892 |
| bh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 174 ±1928% | 96150 |
| bh | bf16 | `sinh_bw` | `default` | 2/33894 defects; rest 0.501 ULP | 0.501 | 0.501 | 1 | 88.5 | 2773 | 6051 |
| bh | bf16 | `softcap` | `beta=50.0` | 1426/65024 defects; rest 0.718 ULP | 0.718 | 0.574 | 0.997 | — | 228 | 73698 |
| bh | bf16 | `softplus` | `default` | 526/65024 defects; rest 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 165 | 101970 |
| bh | bf16 | `softplus_bw` | `default` | within 2 ULP | 1.71 | 0.678 | 0.983 | 3.39e+38 | 2010 | 8348 |
| bh | bf16 | `softshrink` | `default` | faithful; 3968/65024 tie-breaks | 1 | 0.948 | 0.941 | 3.39e+38 | 189 | 88726 |
| bh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 984 | 17046 |
| bh | bf16 | `softsign` | `default` | 512/65024 defects; rest 1 ULP | 1 | 0.605 | 0.907 | 8.47e+37 | 190 ±10% | 88104 |
| bh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391 | 81 | 3.24 | 0.856 | 0.00391 | 641 | 26181 |
| bh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 228 ±9% | 73527 |
| bh | bf16 | `sqrt_bw` | `default` | within 2 ULP | 1.14 | 0.699 | 0.723 | 3.39e+38 | 3008 | 5578 |
| bh | bf16 | `square` | `default` | faithful; 13970/48640 tie-breaks | 0.973 | 0.688 | 0.578 | 1.84e+19 | 189 | 88543 |
| bh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 600 | 27981 |
| bh | bf16 | `squared_difference` | `default` | worst sampled pairing | 2.03 | 1.65 | 0.571 | — | 243 | 68992 |
| bh | bf16 | `squared_difference_` | `default` | worst sampled pairing | 2.03 | 1.65 | 0.571 | — | 240 | 70017 |
| bh | bf16 | `squared_difference_bw` | `default` | faithful; 64842/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 1013 | 16563 |
| bh | bf16 | `subalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 243 | 69018 |
| bh | bf16 | `subtract` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 244 | 68683 |
| bh | bf16 | `subtract_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 248 ±17% | 67692 |
| bh | bf16 | `swish` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 218 ±10% | 76794 |
| bh | bf16 | `tan` | `default` | faithful; 30/37354 tie-breaks | 0.551 | 0.509 | 0.999 | 9.99e+05 | 259 | 64777 |
| bh | bf16 | `tan_bw` | `default` | 22782/65024 defects; rest 3.57e+40 ULP | 3.57e+40 | 1.01e+38 | 0.855 | 1.13 | 1009 | 16636 |
| bh | bf16 | `tanh` | `default` | 2/65024 defects; rest 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 196 | 85753 |
| bh | bf16 | `tanh_bw` | `default` | within 2 ULP | 1.16 | 0.525 | 0.997 | 3.39e+38 | 619 | 27101 |
| bh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08 | 63.5 | 9.38 | 0.981 | 1.35e-08 | 221 ±8% | 75765 |
| bh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09 | 63 | 3.52 | 0.937 | 7.45e-09 | 768 | 21832 |
| bh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 | 87478 |
| bh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88217 |
| bh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 763 ±23% | 21974 |
| bh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88181 |
| bh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 329 ±12% | 51005 |
| bh | bf16 | `xielu` | `default` | 1/56847 defects; rest 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 400 | 41944 |
| bh | bf16 | `xlogy` | `default` | worst sampled pairing | 23.5 | 17.6 | 0.139 | — | 273 | 61546 |
| bh | bf16 | `xlogy_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.78 | 0.78 | 0.998 | — | 4100 | 4092 |
| bh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48715 |
| bh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1150 | 14589 |
| bh | fp32 | `acos` | `default` | within 2 ULP | 1.55 | 0.865 | 0.923 | 1 | 465 | 36046 |
| bh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 6875 ±56% | 2440 |
| bh | fp32 | `acosh` | `default` | never within 2 ULP | 2.47 | 0.76 | 0.782 | — | 397 ±8% | 42250 |
| bh | fp32 | `acosh_bw` | `default` | 15874/64514 defects; rest 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 7782 | 2156 |
| bh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 472 | 35549 |
| bh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 473 ±10% | 35470 |
| bh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 487 ±10% | 34446 |
| bh | fp32 | `addcdiv` | `default` | worst sampled pairing | 2.19e+12 | 8.82e+07 | 0.852 | — | 673 | 24943 |
| bh | fp32 | `addcmul` | `default` | worst sampled pairing | 2.13e+06 | 1.59e+04 | 0.985 | — | 651 ±9% | 25763 |
| bh | fp32 | `asin` | `default` | within 2 ULP | 1.77 | 0.795 | 0.985 | 1 | 451 | 37198 |
| bh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 6680 | 2512 |
| bh | fp32 | `asinh` | `default` | within 2 ULP | 1.64 | 0.773 | 0.881 | 3.39e+38 | 531 | 31571 |
| bh | fp32 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 1202 | 13961 |
| bh | fp32 | `atan` | `default` | accurate to |x| <= 0.902 | 2.34 | 0.806 | 0.955 | 0.902 | 398 | 42184 |
| bh | fp32 | `atan2` | `default` | worst sampled pairing | 1.32e+07 | 1.26e+05 | 0.857 | — | 515 | 32547 |
| bh | fp32 | `atan2_bw` | `default` | worst sampled pairing | 1.64e+07 | 1.66e+05 | 0.587 | — | 4921 | 3409 |
| bh | fp32 | `atan_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.25e+03 | 0.859 | 1.02 | 1139 ±156% | 14727 |
| bh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000229 | 2.97 | 1.51 | 0.912 | 0.000229 | 507 ±6% | 33116 |
| bh | fp32 | `atanh_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.31e+03 | 0.876 | 0.681 | 7551 ±36% | 2222 |
| bh | fp32 | `bias_gelu` | `default` | worst sampled pairing | 7.45e+40 | 6.51e+39 | 0.746 | — | 477 | 35207 |
| bh | fp32 | `bias_gelu_` | `default` | worst sampled pairing | 7.45e+40 | 6.51e+39 | 0.746 | — | 473 | 35450 |
| bh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38 | 2.55 | 1.76 | 0.521 | 2.26e-38 | 386 ±7% | 43513 |
| bh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 | 47660 |
| bh | fp32 | `celu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 392 ±7% | 42752 |
| bh | fp32 | `celu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2497 | 6718 |
| bh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 ±5% | 47706 |
| bh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48418 |
| bh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 ±5% | 48298 |
| bh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2192 | 7654 |
| bh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48546 |
| bh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48437 |
| bh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48556 |
| bh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2170 | 7731 |
| bh | fp32 | `cos` | `default` | accurate to |x| <= 92.4 | 1.34e+08 | 1.33e+05 | 0.927 | 92.4 | 385 | 43582 |
| bh | fp32 | `cos_bw` | `default` | 18643/65024 defects; rest 8.59e+51 ULP | 8.59e+51 | 3.23e+48 | 0.841 | 28 | 1484 | 11309 |
| bh | fp32 | `cosh` | `default` | within 2 ULP | 1.35 | 0.791 | 0.98 | 89.1 | 385 | 43529 |
| bh | fp32 | `cosh_bw` | `default` | 2/33894 defects; rest 2.2 ULP | 2.2 | 1.16 | 0.943 | 0.0155 | 6088 | 2756 |
| bh | fp32 | `deg2rad` | `default` | faithful; 63542/65024 tie-breaks | 0.63 | 0.595 | 0.906 | 3.4e+38 | 347 ±9% | 48299 |
| bh | fp32 | `digamma` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00543 | — | 720 | 23313 |
| bh | fp32 | `digamma_bw` | `default` | 257/64769 defects; rest 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.324 | 5.4e-20 | 6595 | 2544 |
| bh | fp32 | `div` | `default` | worst sampled pairing | 2.44 | 1.44 | 0.894 | — | 509 | 32965 |
| bh | fp32 | `div_bw` | `default` | worst sampled pairing | 7.95e+04 | 7.95e+04 | 0.89 | — | 3028 | 5541 |
| bh | fp32 | `div_no_nan` | `default` | worst sampled pairing | 9.24e+04 | 4.51e+04 | 0.702 | — | 1584 | 10591 |
| bh | fp32 | `divide` | `default` | worst sampled pairing | 2.44 | 1.44 | 0.894 | — | 481 | 34877 |
| bh | fp32 | `divide_` | `default` | worst sampled pairing | 2.44 | 1.44 | 0.894 | — | 481 ±8% | 34885 |
| bh | fp32 | `elu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 387 | 43311 |
| bh | fp32 | `elu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2480 | 6764 |
| bh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 498 ±6% | 33691 |
| bh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 474 | 35422 |
| bh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48197 |
| bh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334 | 7.38 | 1.36 | 0.672 | 0.000334 | 528 | 31775 |
| bh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 2175 | 7712 |
| bh | fp32 | `erfc` | `default` | 198/65024 unflushed; rest 2.1e+33 ULP | 2.1e+33 | 1.71e+29 | 0.329 | — | 473 ±10% | 35500 |
| bh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 2174 | 7717 |
| bh | fp32 | `erfinv` | `default` | 29420/32256 defects; rest 7.96e+06 ULP | 7.96e+06 | 2.59e+05 | 0.000202 | 1.32e-38 | 334 | 50273 |
| bh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456 | 1.03e+06 | 1.67e+03 | 0.914 | 0.000456 | 6706 | 2502 |
| bh | fp32 | `exp` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 393 | 42638 |
| bh | fp32 | `exp` | `fast_approx` | never within 2 ULP | 3.79e+05 | 2e+05 | 0.309 | — | 345 ±10% | 48663 |
| bh | fp32 | `exp2` | `default` | faithful; 6044/49536 tie-breaks | 0.965 | 0.626 | 0.992 | 3.39e+38 | 386 ±6% | 43501 |
| bh | fp32 | `exp2_bw` | `default` | 1/49537 defects; rest 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 1504 | 11152 |
| bh | fp32 | `exp_bw` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 1177 | 14252 |
| bh | fp32 | `expm1` | `default` | faithful; 5257/49458 tie-breaks | 0.997 | 0.565 | 0.997 | 3.39e+38 | 432 | 38821 |
| bh | fp32 | `expm1_bw` | `default` | accurate to |x| <= 22 | 4.19e+06 | 3.83e+03 | 0.995 | 22 | 1177 | 14252 |
| bh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48178 |
| bh | fp32 | `floor_div` | `default` | worst sampled pairing | 4.19e+06 | 1.56e+03 | 0.974 | — | 2251 | 7455 |
| bh | fp32 | `fmod` | `default` | worst sampled pairing | 1.61e+45 | 2.55e+41 | 0.596 | — | 516 | 32505 |
| bh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 | 47715 |
| bh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 475 | 35287 |
| bh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 472 | 35560 |
| bh | fp32 | `gelu` | `default` | 83/65024 defects; rest 1.71e+08 ULP | 1.71e+08 | 1.7e+05 | 0.963 | 0.208 | 347 | 48304 |
| bh | fp32 | `gelu` | `fast_approx` | 197/65024 defects; rest 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 347 | 48412 |
| bh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0298 | 1.47e+08 | 2.3e+04 | 0.929 | 0.0298 | 829 | 20244 |
| bh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48210 |
| bh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 474 | 35399 |
| bh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 472 | 35518 |
| bh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48607 |
| bh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48298 |
| bh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 ±7% | 48784 |
| bh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1463 | 11470 |
| bh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25 | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 348 | 48238 |
| bh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2287 | 7336 |
| bh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36 | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 357 | 47005 |
| bh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133 | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 3239 | 5180 |
| bh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48621 |
| bh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1935 | 8672 |
| bh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 ±8% | 47653 |
| bh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 ±13% | 48016 |
| bh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48500 |
| bh | fp32 | `hypot` | `default` | 16384/65024 defects; rest 3.44e+06 ULP | 3.44e+06 | 3.28e+04 | 0.965 | — | 495 | 33870 |
| bh | fp32 | `hypot_bw` | `default` | 16384/65024 defects; rest 4.86e+06 ULP | 4.86e+06 | 4.48e+04 | 0.826 | — | 2970 | 5649 |
| bh | fp32 | `i0` | `default` | accurate to |x| <= 2.45 | 1.68e+07 | 2.19e+06 | 0.959 | 2.45 | 403 | 41679 |
| bh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00362 | 1.56e+07 | 4.1e+04 | 0.928 | 0.00362 | 1325 | 12658 |
| bh | fp32 | `i1` | `default` | accurate to |x| <= 0.00362 | 8.02 | 0.731 | 0.482 | 0.00362 | 538 | 31194 |
| bh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48658 |
| bh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 504 | 33286 |
| bh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 343 | 48933 |
| bh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48505 |
| bh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48690 |
| bh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48628 |
| bh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 ±37% | 48152 |
| bh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 475 | 35318 |
| bh | fp32 | `ldexp` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 509 | 32964 |
| bh | fp32 | `ldexp_` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 485 | 34610 |
| bh | fp32 | `ldexp_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.742 | 0.742 | 0.996 | — | 2219 | 7559 |
| bh | fp32 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 476 | 35239 |
| bh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 475 | 35338 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 353 | 47483 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithful; 31672/65024 tie-breaks | 0.84 | 0.747 | 0.868 | 3.39e+38 | 346 | 48423 |
| bh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48335 |
| bh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 1646 | 10193 |
| bh | fp32 | `lerp` | `default` | worst sampled pairing | 1.46e+08 | 9.53e+04 | 0.962 | — | 681 ±5% | 24625 |
| bh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 2521 | 6656 |
| bh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47927 |
| bh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181 | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 1296 | 12949 |
| bh | fp32 | `lgamma_bw` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00543 | — | 1483 | 11315 |
| bh | fp32 | `log` | `default` | faithful; 32512/32512 tie-breaks | 0.961 | 0.562 | 0.952 | 3.39e+38 | 399 | 42012 |
| bh | fp32 | `log10` | `default` | accurate to |x| <= 0.336 | 2.13 | 1.36 | 0.641 | 0.336 | 421 ±7% | 39873 |
| bh | fp32 | `log10_bw` | `default` | accurate to |x| <= 4.42e+30 | 9.02e+04 | 1.73e+03 | 0.667 | 4.42e+30 | 4699 | 3570 |
| bh | fp32 | `log1p` | `default` | faithful; 20111/48640 tie-breaks | 0.967 | 0.558 | 0.982 | 3.39e+38 | 393 | 42740 |
| bh | fp32 | `log1p_bw` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 2.96e+03 | 0.89 | 2.02e+31 | 4704 | 3567 |
| bh | fp32 | `log2` | `default` | accurate to |x| <= 0.704 | 2.43 | 0.517 | 0.993 | 0.704 | 393 ±6% | 42670 |
| bh | fp32 | `log2_bw` | `default` | 112/64626 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.76e+03 | 0.684 | — | 4692 | 3576 |
| bh | fp32 | `log_bw` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 3557 | 4717 |
| bh | fp32 | `log_sigmoid` | `default` | never within 2 ULP | 4.4e+05 | 1.82e+04 | 0.481 | — | 459 | 36538 |
| bh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.163 | 2.93 | 1.28 | 0.953 | 0.163 | 5203 | 3225 |
| bh | fp32 | `logaddexp` | `default` | 15566/65024 defects; rest 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 608 | 27608 |
| bh | fp32 | `logaddexp2` | `default` | 15488/65024 defects; rest 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.715 | — | 524 | 32018 |
| bh | fp32 | `logaddexp2_` | `default` | 15488/65024 defects; rest 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.715 | — | 521 | 32188 |
| bh | fp32 | `logaddexp2_bw` | `default` | worst sampled pairing | 9.02e+04 | 5.6e+04 | 0.959 | — | 4743 | 3537 |
| bh | fp32 | `logaddexp_` | `default` | 15566/65024 defects; rest 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 581 | 28860 |
| bh | fp32 | `logaddexp_bw` | `default` | worst sampled pairing | 8.98e+04 | 4.46e+04 | 0.962 | — | 4261 | 3937 |
| bh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 487 | 34434 |
| bh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 486 ±9% | 34487 |
| bh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48289 |
| bh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 343 | 48942 |
| bh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 486 | 34552 |
| bh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 484 | 34695 |
| bh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 500 | 33526 |
| bh | fp32 | `logit` | `default` | accurate to |x| <= 0.266 | 4.19e+06 | 261 | 0.941 | 0.266 | 362 | 46357 |
| bh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 5975 | 2808 |
| bh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 7495 | 2238 |
| bh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 499 | 33625 |
| bh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 471 | 35590 |
| bh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 ±8% | 48151 |
| bh | fp32 | `mac` | `default` | worst sampled pairing | 4.22e+06 | 7.31e+05 | 0.974 | — | 647 | 25918 |
| bh | fp32 | `max_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 4861 | 3452 |
| bh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 499 | 33644 |
| bh | fp32 | `min_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 4865 | 3448 |
| bh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 500 | 33528 |
| bh | fp32 | `mish` | `default` | 9/65024 defects; rest 7 ULP | 7 | 2.58 | 0.953 | 1.49e-07 | 521 | 32199 |
| bh | fp32 | `mse_loss` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 508 ±6% | 33058 |
| bh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1252 | 13398 |
| bh | fp32 | `multigammaln` | `default` | 7422/50376 defects; rest 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.55e-17 | 7856 | 2136 |
| bh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 1.36e+15 | 2.04e+11 | 0.0102 | 5.55e-17 | 7261 | 2310 |
| bh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 478 | 35120 |
| bh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 476 ±6% | 35238 |
| bh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 499 ±11% | 33650 |
| bh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 475 | 35326 |
| bh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 ±6% | 48763 |
| bh | fp32 | `nextafter` | `default` | worst sampled pairing | 8.51e+37 | 1.34e+36 | 0.5 | — | 2861 | 5865 |
| bh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48206 |
| bh | fp32 | `polygamma` | `k=1` | 1/49922 defects; rest 7.91e+33 ULP | 7.91e+33 | 4.68e+29 | 0.467 | 5.4e-20 | 1026 | 16344 |
| bh | fp32 | `polygamma` | `k=2` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 1055 | 15907 |
| bh | fp32 | `polygamma` | `k=4` | 141/49922 defects; rest 3.77e+25 ULP | 3.77e+25 | 6.51e+21 | 0.138 | 3.68e-08 | 1080 | 15530 |
| bh | fp32 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 6599 ±18% | 2542 |
| bh | fp32 | `pow` | `default` | worst sampled pairing | 333 | 2.35 | 0.993 | — | 689 | 24336 |
| bh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2260 | 7423 |
| bh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 347 | 48407 |
| bh | fp32 | `rad2deg` | `default` | faithful; 63518/63518 tie-breaks | 0.696 | 0.643 | 0.858 | 5.93e+36 | 348 | 48275 |
| bh | fp32 | `rdiv` | `value=2.0` | 258/64770 defects; rest 1.49 ULP | 1.49 | 1.19 | 0.664 | 8.47e+37 | 346 | 48513 |
| bh | fp32 | `rdiv_bw` | `scalar=2.0` | 254/48490 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.92e+03 | 0.705 | 7.67e-20 | 6094 | 2753 |
| bh | fp32 | `reciprocal` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 348 | 48182 |
| bh | fp32 | `reciprocal_bw` | `default` | 256/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.92e+03 | 0.705 | 5.42e-20 | 4566 | 3675 |
| bh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48693 |
| bh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48364 |
| bh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3592 | 4671 |
| bh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1130 | 14845 |
| bh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48500 |
| bh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48525 |
| bh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48231 |
| bh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48451 |
| bh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48680 |
| bh | fp32 | `remainder` | `default` | worst sampled pairing | 8.92e+44 | 2.49e+41 | 0.596 | — | 513 | 32710 |
| bh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 | 48130 |
| bh | fp32 | `rpow` | `exponent=0.5` | faithful; 5624/49536 tie-breaks | 0.879 | 0.609 | 0.995 | 3.39e+38 | 632 | 26554 |
| bh | fp32 | `rpow` | `exponent=1.0` | 1536/49536 defects; rest 0 ULP | 0 | 0 | 1 | 8.28e+34 | 621 | 27026 |
| bh | fp32 | `rpow` | `exponent=2.0` | 1536/49536 defects; rest 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 634 ±6% | 26483 |
| bh | fp32 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 2585 | 6489 |
| bh | fp32 | `rsqrt` | `default` | within 2 ULP | 1.12 | 0.834 | 0.864 | 3.39e+38 | 381 | 44064 |
| bh | fp32 | `rsqrt_bw` | `default` | 75/21666 defects; rest 6.21 ULP | 6.21 | 4.15 | 0.303 | — | 5209 | 3221 |
| bh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 503 | 33331 |
| bh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 484 | 34659 |
| bh | fp32 | `selu` | `default` | never within 2 ULP | 51.3 | 26.3 | 0 | — | 399 | 42095 |
| bh | fp32 | `selu_bw` | `default` | 1/65024 defects; rest 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 2818 | 5954 |
| bh | fp32 | `sigmoid` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 416 | 40304 |
| bh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 418 ±6% | 40133 |
| bh | fp32 | `sigmoid_bw` | `default` | 140/65024 defects; rest 8.39e+06 ULP | 8.39e+06 | 2.54e+04 | 0.973 | 0.283 | 1985 | 8451 |
| bh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48543 |
| bh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48206 |
| bh | fp32 | `silu` | `default` | 9/65024 defects; rest 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 424 | 39594 |
| bh | fp32 | `silu_bw` | `default` | 9/65024 defects; rest 5.72e+06 ULP | 5.72e+06 | 842 | 0.934 | 2.96e-07 | 2782 | 6031 |
| bh | fp32 | `sin` | `default` | accurate to |x| <= 28 | 3.36e+07 | 3.07e+04 | 0.96 | 28 | 370 | 45400 |
| bh | fp32 | `sin_bw` | `default` | 18430/65024 defects; rest 1.24e+52 ULP | 1.24e+52 | 3.44e+48 | 0.808 | 92.4 | 1169 | 14355 |
| bh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155 | 2.2 | 1.16 | 0.943 | 0.0155 | 429 | 39076 |
| bh | fp32 | `sinh_bw` | `default` | 2/33894 defects; rest 1.35 ULP | 1.35 | 0.791 | 0.98 | 88.5 | 5383 | 3116 |
| bh | fp32 | `softplus` | `default` | 5/65024 defects; rest 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 370 | 45397 |
| bh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.164 | 2.93 | 1.33 | 0.954 | 0.164 | 3952 | 4246 |
| bh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 347 ±5% | 48342 |
| bh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1939 | 8652 |
| bh | fp32 | `softsign` | `default` | 512/65024 defects; rest 3 ULP | 3 | 1.28 | 0.863 | 1.21e-05 | 374 ±6% | 44817 |
| bh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07 | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 1164 | 14409 |
| bh | fp32 | `sqrt` | `default` | faithful; 32512/32512 tie-breaks | 0.867 | 0.827 | 0.869 | 3.39e+38 | 370 ±94% | 45371 |
| bh | fp32 | `sqrt_bw` | `default` | never within 2 ULP | 2.19 | 1.4 | 0.708 | — | 5766 | 2910 |
| bh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 344 | 48815 |
| bh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1132 | 14816 |
| bh | fp32 | `squared_difference` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 473 | 35471 |
| bh | fp32 | `squared_difference_` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 474 | 35395 |
| bh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1923 | 8725 |
| bh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 500 | 33538 |
| bh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 476 ±8% | 35275 |
| bh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 470 ±6% | 35690 |
| bh | fp32 | `swish` | `default` | 9/65024 defects; rest 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 422 | 39711 |
| bh | fp32 | `tan` | `default` | accurate to |x| <= 3.92 | 2.26e+08 | 2.32e+05 | 0.944 | 3.92 | 449 | 37389 |
| bh | fp32 | `tan_bw` | `default` | 20623/65024 defects; rest 2.85e+45 ULP | 2.85e+45 | 4.74e+44 | 0.87 | 0.852 | 1907 | 8800 |
| bh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000946 | 2.6 | 1.45 | 0.98 | 0.000946 | 392 ±6% | 42786 |
| bh | fp32 | `tanh_bw` | `default` | never within 2 ULP | 6.59e+04 | 7.51e+03 | 0.482 | — | 855 | 19623 |
| bh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08 | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 343 | 48924 |
| bh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09 | 4.19e+06 | 1.02e+05 | 0.901 | 7.42e-09 | 1507 | 11134 |
| bh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 | 48133 |
| bh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 ±6% | 48277 |
| bh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1468 | 11431 |
| bh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 ±7% | 48240 |
| bh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 649 | 25836 |
| bh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13 | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 397 ±6% | 42269 |
| bh | fp32 | `xlogy` | `default` | worst sampled pairing | 2.39e+06 | 1.68e+06 | 0.00256 | — | 481 | 34894 |
| bh | fp32 | `xlogy_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.766 | 0.766 | 0.951 | — | 8039 | 2087 |
| wh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 406 ±14% | 41366 |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1328 ±47% | 12634 |
| wh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 675 | 24861 |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.744 | 0.996 | 0.949 | 7486 | 2241 |
| wh | bf16 | `acosh` | `default` | within 2 ULP | 1.41 | 0.596 | 0.973 | 3.39e+38 | 924 | 18159 |
| wh | bf16 | `acosh_bw` | `default` | 15874/64514 defects; rest 3.26 ULP | 3.26 | 0.688 | 0.781 | 1.03 | 8628 | 1945 |
| wh | bf16 | `add` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 544 ±8% | 30851 |
| wh | bf16 | `add_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 541 ±49% | 31014 |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 536 | 31298 |
| wh | bf16 | `addcdiv` | `default` | worst sampled pairing | 2.11e+06 | 2.35e+03 | 0.999 | — | 698 | 24040 |
| wh | bf16 | `addcmul` | `default` | worst sampled pairing | 126 | 19.4 | 1 | — | 688 | 24382 |
| wh | bf16 | `asin` | `default` | 2/32258 defects; rest 0.499 ULP | 0.499 | 0 | 1 | — | 673 | 24917 |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938 | 2.76 | 0.787 | 0.992 | 0.938 | 7435 | 2256 |
| wh | bf16 | `asinh` | `default` | within 2 ULP | 1.32 | 0.619 | 0.992 | 3.39e+38 | 1408 | 11916 |
| wh | bf16 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.33 ULP | 1.33 | 0.681 | 0.918 | 1.84e+19 | 1358 ±10% | 12353 |
| wh | bf16 | `atan` | `default` | 2/65024 defects; rest 0.527 ULP | 0.527 | 0.513 | 1 | — | 591 ±11% | 28370 |
| wh | bf16 | `atan2` | `default` | worst sampled pairing | 200 | 2.51 | 0.998 | — | 547 | 30688 |
| wh | bf16 | `atan2_bw` | `default` | worst sampled pairing | 248 | 4.88 | 0.522 | — | 5514 | 3043 |
| wh | bf16 | `atan_bw` | `default` | 2/48386 defects; rest 2.23 ULP | 2.23 | 0.867 | 0.854 | 0.23 | 1329 ±6% | 12627 |
| wh | bf16 | `atanh` | `default` | 2/32256 defects; rest 2.08 ULP | 2.08 | 0.995 | 0.081 | — | 689 | 24356 |
| wh | bf16 | `atanh_bw` | `default` | 2/48386 defects; rest 8.17 ULP | 8.17 | 1.02 | 0.84 | 0.82 | 8440 | 1988 |
| wh | bf16 | `bias_gelu` | `default` | worst sampled pairing | 1.14e+36 | 4.5e+34 | 0.749 | — | 533 | 31472 |
| wh | bf16 | `bias_gelu_` | `default` | worst sampled pairing | 1.14e+36 | 4.5e+34 | 0.749 | — | 543 | 30920 |
| wh | bf16 | `cbrt` | `default` | faithful; 846/65024 tie-breaks | 0.507 | 0.502 | 0.987 | 3.39e+38 | 435 ±10% | 38607 |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 409 ±5% | 41048 |
| wh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 493 ±7% | 34054 |
| wh | bf16 | `celu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2705 | 6202 |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 406 ±5% | 41341 |
| wh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 411 ±7% | 40868 |
| wh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±7% | 41485 |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2451 | 6845 |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 412 ±12% | 40762 |
| wh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 ±8% | 42345 |
| wh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 409 ±8% | 41042 |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2445 | 6862 |
| wh | bf16 | `cos` | `default` | faithful; 2/37354 tie-breaks | 0.552 | 0.552 | 1 | 9.99e+05 | 431 ±11% | 38921 |
| wh | bf16 | `cos_bw` | `default` | 21281/65024 defects; rest 2.09e+42 ULP | 2.09e+42 | 2.59e+39 | 0.867 | 2.15e+06 | 1702 | 9859 |
| wh | bf16 | `cosh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 434 ±455% | 38660 |
| wh | bf16 | `cosh_bw` | `default` | 2/33894 defects; rest 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 6711 | 2500 |
| wh | bf16 | `deg2rad` | `default` | 2/65024 defects; rest 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 395 ±8% | 42507 |
| wh | bf16 | `digamma` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 1172 | 14313 |
| wh | bf16 | `digamma_bw` | `default` | 263/64769 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.32e+04 | 0.664 | 1 | 12083 | 1388 |
| wh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 546 | 30737 |
| wh | bf16 | `div_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.512 | 0.512 | 0.984 | — | 10534 | 1593 |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 1447 | 11595 |
| wh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 548 | 30621 |
| wh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 573 | 29286 |
| wh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 479 ±7% | 35012 |
| wh | bf16 | `elu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2744 | 6114 |
| wh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 532 | 31541 |
| wh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 552 | 30375 |
| wh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 406 | 41344 |
| wh | bf16 | `erf` | `default` | within 2 ULP | 1.06 | 0.771 | 0.729 | 3.39e+38 | 563 ±8% | 29815 |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 2455 | 6833 |
| wh | bf16 | `erfc` | `default` | 198/65024 unflushed; rest 3.19e+28 ULP | 3.19e+28 | 3.23e+24 | 0.969 | 2.5 | 823 | 20396 |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 2452 | 6843 |
| wh | bf16 | `erfinv` | `default` | 29424/32256 defects; rest 121 ULP | 121 | 6.34 | 0.391 | 1.31e-38 | 901 | 18631 |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777 | 8.91 | 1.28 | 0.98 | 0.777 | 7854 | 2136 |
| wh | bf16 | `exp` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 416 ±7% | 40313 |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP | 6.08 | 3.05 | 0.314 | — | 392 | 42840 |
| wh | bf16 | `exp2` | `default` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 432 ±16% | 38865 |
| wh | bf16 | `exp2_bw` | `default` | 1/49537 defects; rest 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 1704 | 9843 |
| wh | bf16 | `exp_bw` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 1317 | 12744 |
| wh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 421 ±10% | 39830 |
| wh | bf16 | `expm1_bw` | `default` | accurate to |x| <= 33.2 | 34 | 0.695 | 0.984 | 33.2 | 1706 ±6% | 9834 |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 422 ±10% | 39725 |
| wh | bf16 | `floor_div` | `default` | worst sampled pairing | 64 | 42 | 0.996 | — | 2531 | 6630 |
| wh | bf16 | `fmod` | `default` | worst sampled pairing | 6.65e+35 | 7.8e+34 | 0.853 | — | 651 | 25774 |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±5% | 42082 |
| wh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 535 ±9% | 31363 |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 548 | 30642 |
| wh | bf16 | `gelu` | `default` | 86/65024 defects; rest 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 799 | 21001 |
| wh | bf16 | `gelu` | `fast_approx` | 198/65024 defects; rest 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 397 ±6% | 42212 |
| wh | bf16 | `gelu_bw` | `default` | faithful; 38/65024 tie-breaks | 0.939 | 0.681 | 0.999 | 3.39e+38 | 1759 | 9541 |
| wh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 409 ±8% | 41050 |
| wh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 536 | 31328 |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 550 ±19% | 30482 |
| wh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 409 ±9% | 41052 |
| wh | bf16 | `hardmish` | `default` | faithful; 2587/65024 tie-breaks | 1 | 0.913 | 0.961 | 3.39e+38 | 407 ±6% | 41235 |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 ±6% | 42140 |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1682 | 9974 |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP | 1 | 0.557 | 0.986 | 3.39e+38 | 413 ±21% | 40648 |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 2542 | 6600 |
| wh | bf16 | `hardswish` | `default` | 2/65024 defects; rest 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 427 | 39249 |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12 | 85.3 | 2.1 | 0.993 | 1.12 | 3542 | 4736 |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 | 42324 |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2154 | 7789 |
| wh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 416 ±17% | 40300 |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 | 41698 |
| wh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 413 ±11% | 40636 |
| wh | bf16 | `hypot` | `default` | 16384/65026 defects; rest 52.7 ULP | 52.7 | 1.58 | 0.947 | — | 551 | 30424 |
| wh | bf16 | `hypot_bw` | `default` | worst sampled pairing | 74.6 | 2.42 | 0.833 | — | 3375 | 4971 |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.6 | 255 | 56.3 | 0.952 | 13.6 | 459 ±6% | 36547 |
| wh | bf16 | `i0_bw` | `default` | 4/33904 defects; rest 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 2090 | 8027 |
| wh | bf16 | `i1` | `default` | 4/65024 defects; rest 0.858 ULP | 0.858 | 0.589 | 0.516 | 2.33e-38 | 1203 | 13948 |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 392 ±6% | 42810 |
| wh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 559 | 30007 |
| wh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 ±9% | 42552 |
| wh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 ±8% | 42186 |
| wh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 ±7% | 42202 |
| wh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 ±7% | 41881 |
| wh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 ±6% | 42465 |
| wh | bf16 | `l1_loss` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 538 ±9% | 31156 |
| wh | bf16 | `ldexp` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 553 ±5% | 30318 |
| wh | bf16 | `ldexp_` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 556 ±8% | 30157 |
| wh | bf16 | `ldexp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.898 | 0.898 | 0.983 | — | 2725 | 6157 |
| wh | bf16 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 537 | 31246 |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 546 ±6% | 30721 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±7% | 41533 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithful; 15341/65024 tie-breaks | 0.96 | 0.741 | 0.761 | 3.39e+38 | 397 | 42287 |
| wh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 ±6% | 42380 |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 1834 | 9146 |
| wh | bf16 | `lerp` | `default` | worst sampled pairing | 1.77e+03 | 39.9 | 1 | — | 692 ±8% | 24244 |
| wh | bf16 | `lerp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.506 | 0.506 | 0.992 | — | 2807 | 5977 |
| wh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 412 ±12% | 40766 |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.414 | 324 | 0.846 | 0.76 | 0.414 | 2433 | 6895 |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 2083 | 8055 |
| wh | bf16 | `log` | `default` | faithful; 63/32512 tie-breaks | 0.78 | 0.534 | 0.998 | 3.39e+38 | 421 ±14% | 39866 |
| wh | bf16 | `log10` | `default` | faithful; 58/32512 tie-breaks | 0.785 | 0.543 | 0.998 | 3.39e+38 | 426 ±10% | 39360 |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP | 1.74 | 0.796 | 0.524 | 3.69e+37 | 5284 | 3175 |
| wh | bf16 | `log1p` | `default` | faithful; 109/48640 tie-breaks | 0.931 | 0.57 | 0.998 | 3.39e+38 | 430 ±6% | 39041 |
| wh | bf16 | `log1p_bw` | `default` | 2/64514 defects; rest 1.42 ULP | 1.42 | 0.68 | 0.975 | 8.47e+37 | 5254 | 3193 |
| wh | bf16 | `log2` | `default` | faithful; 59/32512 tie-breaks | 0.766 | 0.542 | 0.998 | 3.39e+38 | 428 ±40% | 39158 |
| wh | bf16 | `log2_bw` | `default` | 116/64626 defects; rest 1.51 ULP | 1.51 | 0.842 | 0.578 | — | 5280 | 3177 |
| wh | bf16 | `log_bw` | `default` | 2/64514 defects; rest 0.512 ULP | 0.512 | 0.507 | 0.984 | 8.47e+37 | 3993 | 4202 |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.426 | 6.39 | 1.38 | 0.958 | 0.426 | 592 ±6% | 28318 |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP | 1.91 | 0.785 | 0.972 | 3.39e+38 | 5922 | 2833 |
| wh | bf16 | `logaddexp` | `default` | 15566/65026 defects; rest 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 792 | 21197 |
| wh | bf16 | `logaddexp2` | `default` | 15488/65026 defects; rest 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 904 ±13% | 18566 |
| wh | bf16 | `logaddexp2_` | `default` | 15488/65026 defects; rest 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 901 | 18612 |
| wh | bf16 | `logaddexp2_bw` | `default` | worst sampled pairing | 41.4 | 3.6 | 0.978 | — | 5833 | 2876 |
| wh | bf16 | `logaddexp_` | `default` | 15566/65026 defects; rest 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 815 ±5% | 20588 |
| wh | bf16 | `logaddexp_bw` | `default` | worst sampled pairing | 62.5 | 4.9 | 0.979 | — | 4795 | 3499 |
| wh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 551 ±20% | 30424 |
| wh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 560 | 29967 |
| wh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 ±6% | 42143 |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 396 | 42329 |
| wh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 546 | 30715 |
| wh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 563 ±6% | 29794 |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 561 | 29885 |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395 | 64 | 1.67 | 0.985 | 0.395 | 895 | 18738 |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP | 1.65 | 0.649 | 0.961 | 0.996 | 6964 | 2409 |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP | 1.65 | 0.649 | 0.961 | 0.996 | 8230 | 2038 |
| wh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 531 | 31614 |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 544 | 30847 |
| wh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 ±6% | 42571 |
| wh | bf16 | `mac` | `default` | worst sampled pairing | 63.5 | 26.4 | 1 | — | 685 ±6% | 24478 |
| wh | bf16 | `max_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 5479 ±66% | 3062 |
| wh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 541 ±8% | 30999 |
| wh | bf16 | `min_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 5486 | 3058 |
| wh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 533 | 31472 |
| wh | bf16 | `mish` | `default` | 11/65024 defects; rest 1.49 ULP | 1.49 | 0.667 | 0.988 | 1.95e-38 | 594 ±5% | 28251 |
| wh | bf16 | `mse_loss` | `default` | worst sampled pairing | 2.03 | 1.65 | 0.571 | — | 538 ±5% | 31166 |
| wh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1423 | 11789 |
| wh | bf16 | `multigammaln` | `default` | 11536/48328 defects; rest 724 ULP | 724 | 1.67 | 0.717 | 5.55e-17 | 12650 | 1326 |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 2.38e+06 | 293 | 0.249 | 5.55e-17 | 9659 | 1737 |
| wh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 537 | 31248 |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 560 ±6% | 29942 |
| wh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 533 | 31458 |
| wh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 547 | 30674 |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 ±5% | 42630 |
| wh | bf16 | `nextafter` | `default` | worst sampled pairing | 1.3e+33 | 2.32e+31 | 0.56 | — | 3114 | 5388 |
| wh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 391 ±6% | 42950 |
| wh | bf16 | `polygamma` | `k=1` | 7/49922 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.43e+05 | 0.956 | 0.996 | 5579 | 3007 |
| wh | bf16 | `polygamma` | `k=2` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 6.81e+05 | 0.945 | 4.47 | 5683 ±7% | 2952 |
| wh | bf16 | `polygamma` | `k=4` | 142/49922 defects; rest 3.22e+09 ULP | 3.22e+09 | 1.09e+07 | 0.936 | 3.48 | 5768 | 2908 |
| wh | bf16 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 6.81e+05 | 0.945 | 4.47 | 12120 | 1384 |
| wh | bf16 | `pow` | `default` | worst sampled pairing | 4.86e+18 | 2.08e+14 | 0.982 | — | 987 ±12% | 17000 |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2549 | 6582 |
| wh | bf16 | `prelu` | `weight=0.25` | 1/65024 defects; rest 0 ULP | 0 | 0 | 1 | 4.68e-38 | 396 ±49% | 42370 |
| wh | bf16 | `rad2deg` | `default` | faithful; 31760/63518 tie-breaks | 0.992 | 0.75 | 0.5 | 5.9e+36 | 398 | 42196 |
| wh | bf16 | `rdiv` | `value=2.0` | 258/64770 defects; rest 0.512 ULP | 0.512 | 0.507 | 0.984 | 8.47e+37 | 426 ±15% | 39386 |
| wh | bf16 | `rdiv_bw` | `scalar=2.0` | 256/48492 defects; rest 1.51 ULP | 1.51 | 0.835 | 0.617 | 7.67e-20 | 6904 | 2430 |
| wh | bf16 | `reciprocal` | `default` | 2/64514 defects; rest 0.512 ULP | 0.512 | 0.507 | 0.984 | 8.47e+37 | 420 ±11% | 39948 |
| wh | bf16 | `reciprocal_bw` | `default` | 256/48386 defects; rest 1.51 ULP | 1.51 | 0.835 | 0.617 | 5.42e-20 | 5127 | 3272 |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±8% | 41522 |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 407 ±10% | 41187 |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3956 | 4241 |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1311 ±13% | 12799 |
| wh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 ±6% | 41798 |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 | 41756 |
| wh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 ±6% | 41777 |
| wh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 | 42314 |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 ±6% | 42126 |
| wh | bf16 | `remainder` | `default` | worst sampled pairing | 6.65e+35 | 1.06e+35 | 0.853 | — | 675 | 24867 |
| wh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 409 ±140% | 41070 |
| wh | bf16 | `rpow` | `exponent=0.5` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.975 | 3.39e+38 | 964 | 17406 |
| wh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 973 | 17247 |
| wh | bf16 | `rpow` | `exponent=2.0` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 964 | 17400 |
| wh | bf16 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 2914 | 5757 |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 425 ±8% | 39483 |
| wh | bf16 | `rsqrt_bw` | `default` | 75/21665 defects; rest 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 6312 | 2658 |
| wh | bf16 | `rsub` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 542 | 30966 |
| wh | bf16 | `rsub_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 548 | 30611 |
| wh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 514 ±17% | 32638 |
| wh | bf16 | `selu_bw` | `default` | 1/65024 defects; rest 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 3104 | 5405 |
| wh | bf16 | `sigmoid` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 472 ±12% | 35517 |
| wh | bf16 | `sigmoid_accurate` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 468 ±7% | 35860 |
| wh | bf16 | `sigmoid_bw` | `default` | 329/65024 defects; rest 266 ULP | 266 | 5.5 | 0.988 | 1.77 | 2255 ±19% | 7440 |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 406 ±17% | 41312 |
| wh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 849 ±142% | 19770 |
| wh | bf16 | `silu` | `default` | 13/65024 defects; rest 0.914 ULP | 0.914 | 0.585 | 0.993 | 2.33e-38 | 489 ±10% | 34280 |
| wh | bf16 | `silu_bw` | `default` | 9/65024 defects; rest 285 ULP | 285 | 1.31 | 0.979 | 0.867 | 3125 | 5369 |
| wh | bf16 | `sin` | `default` | faithful; 4/37354 tie-breaks | 0.552 | 0.526 | 1 | 9.99e+05 | 434 ±12% | 38641 |
| wh | bf16 | `sin_bw` | `default` | 21284/65024 defects; rest 2.98e+41 ULP | 2.98e+41 | 1.39e+39 | 0.857 | 1.07e+06 | 1320 | 12708 |
| wh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 541 ±9% | 31035 |
| wh | bf16 | `sinh_bw` | `default` | 2/33894 defects; rest 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 5844 | 2871 |
| wh | bf16 | `softplus` | `default` | 526/65024 defects; rest 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 473 ±7% | 35454 |
| wh | bf16 | `softplus_bw` | `default` | within 2 ULP | 1.91 | 0.746 | 0.982 | 3.39e+38 | 4352 | 3855 |
| wh | bf16 | `softshrink` | `default` | faithful; 3968/65024 tie-breaks | 1 | 0.948 | 0.941 | 3.39e+38 | 399 ±10% | 42003 |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2158 | 7773 |
| wh | bf16 | `softsign` | `default` | 510/65024 defects; rest 1 ULP | 1 | 0.641 | 0.907 | 8.51e+37 | 443 ±9% | 37862 |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391 | 81 | 3.29 | 0.859 | 0.00391 | 1326 | 12656 |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 423 ±12% | 39634 |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP | 1.14 | 0.695 | 0.719 | 3.39e+38 | 6567 | 2555 |
| wh | bf16 | `square` | `default` | faithful; 13970/48640 tie-breaks | 0.973 | 0.688 | 0.578 | 1.84e+19 | 394 ±9% | 42574 |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1295 | 12958 |
| wh | bf16 | `squared_difference` | `default` | worst sampled pairing | 2.03 | 1.65 | 0.571 | — | 535 | 31374 |
| wh | bf16 | `squared_difference_` | `default` | worst sampled pairing | 2.03 | 1.65 | 0.571 | — | 548 ±12% | 30597 |
| wh | bf16 | `squared_difference_bw` | `default` | faithful; 64842/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 2175 | 7712 |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 549 ±10% | 30564 |
| wh | bf16 | `subtract` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 533 | 31458 |
| wh | bf16 | `subtract_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 546 ±6% | 30736 |
| wh | bf16 | `swish` | `default` | 13/65024 defects; rest 0.914 ULP | 0.914 | 0.585 | 0.993 | 2.33e-38 | 496 ±11% | 33814 |
| wh | bf16 | `tan` | `default` | faithful; 48/37354 tie-breaks | 0.551 | 0.5 | 0.999 | 9.99e+05 | 557 ±8% | 30133 |
| wh | bf16 | `tan_bw` | `default` | 12184/65024 defects; rest 3.57e+40 ULP | 3.57e+40 | 3.81e+37 | 0.691 | 1.13 | 2185 | 7678 |
| wh | bf16 | `tanh` | `default` | 2/65024 defects; rest 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 417 ±948% | 40269 |
| wh | bf16 | `tanh_bw` | `default` | within 2 ULP | 1.16 | 0.525 | 0.997 | 3.39e+38 | 1473 | 11386 |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08 | 63.5 | 9.38 | 0.981 | 1.35e-08 | 432 ±9% | 38848 |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09 | 63 | 3.52 | 0.937 | 7.45e-09 | 1683 | 9967 |
| wh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 ±5% | 42164 |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 | 42482 |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1655 | 10136 |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 | 41803 |
| wh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 689 ±8% | 24361 |
| wh | bf16 | `xielu` | `default` | 1/56847 defects; rest 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 1008 | 16648 |
| wh | bf16 | `xlogy` | `default` | worst sampled pairing | 23.5 | 17.6 | 0.139 | — | 553 ±26% | 30328 |
| wh | bf16 | `xlogy_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.78 | 0.78 | 0.998 | — | 8884 | 1888 |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21897 |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2489 | 6741 |
| wh | fp32 | `acos` | `default` | within 2 ULP | 1.55 | 0.865 | 0.923 | 1 | 1095 | 15321 |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 14977 | 1120 |
| wh | fp32 | `acosh` | `default` | never within 2 ULP | 2.76 | 0.764 | 0.782 | — | 1433 ±7% | 11708 |
| wh | fp32 | `acosh_bw` | `default` | 15874/64514 defects; rest 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 17161 | 978 |
| wh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 977 | 17168 |
| wh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 997 | 16828 |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 988 | 16984 |
| wh | fp32 | `addcdiv` | `default` | worst sampled pairing | 2.19e+12 | 8.82e+07 | 0.901 | — | 1345 | 12473 |
| wh | fp32 | `addcmul` | `default` | worst sampled pairing | 2.13e+06 | 1.59e+04 | 0.985 | — | 1343 | 12494 |
| wh | fp32 | `asin` | `default` | within 2 ULP | 1.77 | 0.795 | 0.985 | 1 | 1047 | 16026 |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 14787 ±39% | 1135 |
| wh | fp32 | `asinh` | `default` | within 2 ULP | 1.72 | 0.774 | 0.881 | 3.39e+38 | 1791 | 9367 |
| wh | fp32 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 2598 | 6458 |
| wh | fp32 | `atan` | `default` | accurate to |x| <= 0.902 | 2.34 | 0.807 | 0.955 | 0.902 | 907 | 18498 |
| wh | fp32 | `atan2` | `default` | worst sampled pairing | 1.32e+07 | 1.26e+05 | 0.882 | — | 1009 | 16628 |
| wh | fp32 | `atan2_bw` | `default` | worst sampled pairing | 1.64e+07 | 9.5e+04 | 0.61 | — | 10571 | 1587 |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 1.01 | 2.8 | 1.23 | 0.874 | 1.01 | 2569 | 6531 |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 4.28e-07 | 3.11 | 1.55 | 0.91 | 4.28e-07 | 1313 ±6% | 12778 |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.678 | 2.05e+03 | 1.55 | 0.891 | 0.678 | 16361 ±16% | 1025 |
| wh | fp32 | `bias_gelu` | `default` | worst sampled pairing | 7.45e+40 | 6.51e+39 | 0.746 | — | 978 | 17150 |
| wh | fp32 | `bias_gelu_` | `default` | worst sampled pairing | 7.45e+40 | 6.51e+39 | 0.746 | — | 1004 | 16704 |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38 | 2.55 | 1.76 | 0.521 | 2.26e-38 | 840 | 19977 |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 777 | 21580 |
| wh | fp32 | `celu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 949 | 17674 |
| wh | fp32 | `celu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5473 | 3066 |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21820 |
| wh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21811 |
| wh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21788 |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4613 | 3637 |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 ±18% | 21804 |
| wh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 792 ±19% | 21175 |
| wh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21884 |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4607 | 3642 |
| wh | fp32 | `cos` | `default` | accurate to |x| <= 92.4 | 1.34e+08 | 1.33e+05 | 0.927 | 92.4 | 861 | 19478 |
| wh | fp32 | `cos_bw` | `default` | 18643/65024 defects; rest 8.59e+51 ULP | 8.59e+51 | 3.23e+48 | 0.841 | 28 | 3300 | 5084 |
| wh | fp32 | `cosh` | `default` | within 2 ULP | 1.35 | 0.798 | 0.98 | 89.1 | 971 | 17283 |
| wh | fp32 | `cosh_bw` | `default` | 2/33894 defects; rest 2.21 ULP | 2.21 | 1.17 | 0.943 | 0.0155 | 13572 | 1236 |
| wh | fp32 | `deg2rad` | `default` | faithful; 63542/65024 tie-breaks | 0.63 | 0.595 | 0.906 | 3.4e+38 | 770 | 21796 |
| wh | fp32 | `digamma` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00566 | — | 2633 | 6371 |
| wh | fp32 | `digamma_bw` | `default` | 255/64769 defects; rest 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.429 | 5.4e-20 | 17876 | 938 |
| wh | fp32 | `div` | `default` | within 2 ULP | 1.78 | 1.22 | 0.862 | — | 989 | 16970 |
| wh | fp32 | `div_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.84 | 0.84 | 0.902 | — | 20707 | 810 |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP | 1.79 | 1.52 | 0.714 | — | 3475 | 4828 |
| wh | fp32 | `divide` | `default` | within 2 ULP | 1.78 | 1.22 | 0.862 | — | 997 | 16828 |
| wh | fp32 | `divide_` | `default` | within 2 ULP | 1.78 | 1.22 | 0.862 | — | 1018 | 16479 |
| wh | fp32 | `elu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 949 | 17687 |
| wh | fp32 | `elu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5462 | 3071 |
| wh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 987 | 17002 |
| wh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 1016 | 16511 |
| wh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21790 |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334 | 6.48 | 1.35 | 0.671 | 0.000334 | 1233 | 13608 |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 4864 | 3449 |
| wh | fp32 | `erfc` | `default` | 198/65024 unflushed; rest 2.1e+33 ULP | 2.1e+33 | 1.71e+29 | 0.33 | — | 1330 ±8% | 12617 |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 4859 | 3453 |
| wh | fp32 | `erfinv` | `default` | 29420/32256 defects; rest 7.96e+06 ULP | 7.96e+06 | 2.59e+05 | 0.000202 | 1.32e-38 | 1382 ±7% | 12136 |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456 | 4.98e+05 | 1.32e+03 | 0.914 | 0.000456 | 15440 | 1087 |
| wh | fp32 | `exp` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 901 | 18618 |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP | 3.79e+05 | 2e+05 | 0.309 | — | 765 | 21934 |
| wh | fp32 | `exp2` | `default` | faithful; 6044/49536 tie-breaks | 0.965 | 0.626 | 0.992 | 3.39e+38 | 844 ±9% | 19876 |
| wh | fp32 | `exp2_bw` | `default` | 1/49537 defects; rest 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 3327 | 5043 |
| wh | fp32 | `exp_bw` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 2615 ±146% | 6416 |
| wh | fp32 | `expm1` | `default` | faithful; 5257/49458 tie-breaks | 0.997 | 0.565 | 0.997 | 3.39e+38 | 1031 | 16280 |
| wh | fp32 | `expm1_bw` | `default` | accurate to |x| <= 22 | 4.19e+06 | 3.83e+03 | 0.995 | 22 | 3512 | 4777 |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 782 | 21457 |
| wh | fp32 | `floor_div` | `default` | worst sampled pairing | 4.19e+06 | 1.77e+03 | 0.955 | — | 4806 | 3491 |
| wh | fp32 | `fmod` | `default` | worst sampled pairing | 1.43e+45 | 1.86e+41 | 0.762 | — | 1011 | 16594 |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21940 |
| wh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 982 | 17078 |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 1026 ±5% | 16351 |
| wh | fp32 | `gelu` | `default` | 83/65024 defects; rest 1.51e+08 ULP | 1.51e+08 | 1.55e+05 | 0.963 | 0.253 | 1381 ±8% | 12148 |
| wh | fp32 | `gelu` | `fast_approx` | 197/65024 defects; rest 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 774 | 21677 |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0298 | 1.47e+08 | 2.3e+04 | 0.929 | 0.0298 | 2082 | 8059 |
| wh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 778 | 21558 |
| wh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 991 | 16935 |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1006 ±5% | 16681 |
| wh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21845 |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21893 |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 778 ±6% | 21552 |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3221 | 5208 |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25 | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 768 | 21852 |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 5058 | 3317 |
| wh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36 | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 828 ±9% | 20250 |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133 | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 7117 | 2357 |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21828 |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4273 | 3927 |
| wh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 775 | 21638 |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 776 | 21632 |
| wh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 ±6% | 21666 |
| wh | fp32 | `hypot` | `default` | 16384/65024 defects; rest 3.44e+06 ULP | 3.44e+06 | 3.28e+04 | 0.965 | — | 1011 | 16588 |
| wh | fp32 | `hypot_bw` | `default` | 16384/65024 defects; rest 4.86e+06 ULP | 4.86e+06 | 4.48e+04 | 0.824 | — | 6348 | 2643 |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.45 | 1.68e+07 | 2.19e+06 | 0.959 | 2.45 | 890 ±8% | 18857 |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00626 | 1.56e+07 | 4.11e+04 | 0.929 | 0.00626 | 3448 | 4866 |
| wh | fp32 | `i1` | `default` | accurate to |x| <= 0.00626 | 8.02 | 0.722 | 0.483 | 0.00626 | 1737 | 9657 |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21687 |
| wh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 996 | 16837 |
| wh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21864 |
| wh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21888 |
| wh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21767 |
| wh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21861 |
| wh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 787 ±6% | 21305 |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 986 | 17023 |
| wh | fp32 | `ldexp` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 999 | 16795 |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 1014 | 16544 |
| wh | fp32 | `ldexp_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.742 | 0.742 | 0.996 | — | 5258 | 3191 |
| wh | fp32 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 987 | 16994 |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 1002 | 16749 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 ±6% | 21690 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithful; 31672/65024 tie-breaks | 0.84 | 0.747 | 0.868 | 3.39e+38 | 771 | 21753 |
| wh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 775 | 21652 |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 3598 | 4664 |
| wh | fp32 | `lerp` | `default` | worst sampled pairing | 1.46e+08 | 9.53e+04 | 0.962 | — | 1335 | 12563 |
| wh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 5380 | 3119 |
| wh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 776 ±11% | 21609 |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181 | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 3752 | 4471 |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00566 | — | 4337 | 3869 |
| wh | fp32 | `log` | `default` | faithful; 32512/32512 tie-breaks | 0.961 | 0.562 | 0.952 | 3.39e+38 | 864 | 19420 |
| wh | fp32 | `log10` | `default` | accurate to |x| <= 0.336 | 2.13 | 1.36 | 0.641 | 0.336 | 883 | 19000 |
| wh | fp32 | `log10_bw` | `default` | accurate to |x| <= 2.04e-38 | 2.1 | 1.41 | 0.674 | 2.04e-38 | 10249 | 1637 |
| wh | fp32 | `log1p` | `default` | faithful; 20111/48640 tie-breaks | 0.967 | 0.558 | 0.982 | 3.39e+38 | 910 | 18436 |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP | 1.89 | 0.769 | 0.913 | 8.51e+37 | 10272 | 1633 |
| wh | fp32 | `log2` | `default` | accurate to |x| <= 0.704 | 2.43 | 0.517 | 0.993 | 0.704 | 889 | 18868 |
| wh | fp32 | `log2_bw` | `default` | 112/64626 defects; rest 1.9 ULP | 1.9 | 1.3 | 0.701 | — | 10260 | 1635 |
| wh | fp32 | `log_bw` | `default` | faithful; 64512/64514 tie-breaks | 0.892 | 0.714 | 0.902 | 8.51e+37 | 7820 | 2145 |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP | 4.4e+05 | 1.82e+04 | 0.481 | — | 1095 | 15326 |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.00164 | 2.95 | 1.28 | 0.954 | 0.00164 | 11491 | 1460 |
| wh | fp32 | `logaddexp` | `default` | 15566/65024 defects; rest 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 1486 | 11292 |
| wh | fp32 | `logaddexp2` | `default` | 15488/65024 defects; rest 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.715 | — | 1314 ±149% | 12768 |
| wh | fp32 | `logaddexp2_` | `default` | 15488/65024 defects; rest 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.715 | — | 1305 ±6% | 12857 |
| wh | fp32 | `logaddexp2_bw` | `default` | worst sampled pairing | 45.7 | 6.68 | 0.956 | — | 11267 | 1489 |
| wh | fp32 | `logaddexp_` | `default` | 15566/65024 defects; rest 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 1490 | 11259 |
| wh | fp32 | `logaddexp_bw` | `default` | worst sampled pairing | 65.7 | 8.66 | 0.962 | — | 9418 | 1781 |
| wh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 1043 ±84% | 16086 |
| wh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 1026 ±6% | 16359 |
| wh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 775 ±6% | 21641 |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 770 | 21793 |
| wh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 1007 | 16663 |
| wh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 1018 | 16482 |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 1022 | 16412 |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.266 | 4.19e+06 | 261 | 0.94 | 0.266 | 1057 | 15872 |
| wh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.98e-08 | 2.36 | 0.839 | 0.857 | 2.98e-08 | 13834 | 1213 |
| wh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.98e-08 | 2.36 | 0.839 | 0.857 | 2.98e-08 | 16461 | 1019 |
| wh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 987 | 17003 |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1004 ±6% | 16706 |
| wh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21830 |
| wh | fp32 | `mac` | `default` | worst sampled pairing | 4.22e+06 | 7.31e+05 | 0.974 | — | 1342 | 12503 |
| wh | fp32 | `max_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 10455 | 1605 |
| wh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 984 | 17057 |
| wh | fp32 | `min_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 10459 | 1604 |
| wh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 990 | 16953 |
| wh | fp32 | `mish` | `default` | 9/65024 defects; rest 7 ULP | 7 | 2.49 | 0.954 | 8.61e-06 | 1216 ±6% | 13801 |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 997 ±8% | 16825 |
| wh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2720 | 6168 |
| wh | fp32 | `multigammaln` | `default` | 7422/50376 defects; rest 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.59e-17 | 20810 | 806 |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 1.36e+15 | 2.04e+11 | 0.0104 | 5.55e-17 | 20183 | 831 |
| wh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 989 | 16967 |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1001 | 16765 |
| wh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 997 ±10% | 16820 |
| wh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 1002 | 16743 |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21684 |
| wh | fp32 | `nextafter` | `default` | worst sampled pairing | 8.51e+37 | 1.34e+36 | 0.5 | — | 6130 ±14% | 2737 |
| wh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21823 |
| wh | fp32 | `polygamma` | `k=1` | accurate to |x| <= 5.4e-20 | 7.91e+33 | 4.68e+29 | 0.617 | 5.4e-20 | 5750 | 2918 |
| wh | fp32 | `polygamma` | `k=2` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.199 | 1.79e-13 | 5924 | 2832 |
| wh | fp32 | `polygamma` | `k=4` | 141/49922 defects; rest 3.77e+25 ULP | 3.77e+25 | 6.51e+21 | 0.191 | 3.68e-08 | 6330 ±14% | 2650 |
| wh | fp32 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.199 | 1.79e-13 | 18107 | 926 |
| wh | fp32 | `pow` | `default` | worst sampled pairing | 324 | 2.42 | 0.993 | — | 2174 | 7716 |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 4951 | 3389 |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 775 ±6% | 21658 |
| wh | fp32 | `rad2deg` | `default` | faithful; 63518/63518 tie-breaks | 0.696 | 0.643 | 0.858 | 5.93e+36 | 770 | 21792 |
| wh | fp32 | `rdiv` | `value=2.0` | 256/64770 defects; rest 0.892 ULP | 0.892 | 0.714 | 0.902 | 8.51e+37 | 799 ±7% | 20988 |
| wh | fp32 | `rdiv_bw` | `scalar=2.0` | 252/48490 defects; rest 1.84 ULP | 1.84 | 1.22 | 0.726 | 7.67e-20 | 13361 ±11% | 1256 |
| wh | fp32 | `reciprocal` | `default` | faithful; 64512/64514 tie-breaks | 0.892 | 0.714 | 0.902 | 8.51e+37 | 800 | 20970 |
| wh | fp32 | `reciprocal_bw` | `default` | 254/48386 defects; rest 1.84 ULP | 1.84 | 1.22 | 0.726 | 5.42e-20 | 10052 | 1669 |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 ±6% | 21741 |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 776 | 21621 |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 7915 | 2120 |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2489 | 6740 |
| wh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21825 |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21805 |
| wh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21819 |
| wh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21775 |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 ±16% | 21917 |
| wh | fp32 | `remainder` | `default` | worst sampled pairing | 1.43e+45 | 1.68e+41 | 0.761 | — | 1011 | 16597 |
| wh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 ±52% | 21701 |
| wh | fp32 | `rpow` | `exponent=0.5` | faithful; 5624/49536 tie-breaks | 0.879 | 0.609 | 0.995 | 3.39e+38 | 1834 | 9148 |
| wh | fp32 | `rpow` | `exponent=1.0` | 1536/49536 defects; rest 0 ULP | 0 | 0 | 1 | 8.28e+34 | 1831 | 9160 |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536/49536 defects; rest 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 1835 | 9141 |
| wh | fp32 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 5694 | 2946 |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP | 1.12 | 0.834 | 0.864 | 3.39e+38 | 844 | 19879 |
| wh | fp32 | `rsqrt_bw` | `default` | 75/21666 defects; rest 6.76 ULP | 6.76 | 4.26 | 0.3 | — | 11899 | 1410 |
| wh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 987 | 17005 |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1003 | 16723 |
| wh | fp32 | `selu` | `default` | never within 2 ULP | 51.3 | 26.3 | 0 | — | 961 | 17462 |
| wh | fp32 | `selu_bw` | `default` | 1/65024 defects; rest 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 6208 | 2703 |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 1077 | 15574 |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 1070 ±5% | 15675 |
| wh | fp32 | `sigmoid_bw` | `default` | 140/65024 defects; rest 8.39e+06 ULP | 8.39e+06 | 2.71e+04 | 0.975 | 0.447 | 4519 | 3712 |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 782 ±6% | 21466 |
| wh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21897 |
| wh | fp32 | `silu` | `default` | 9/65024 defects; rest 2.94 ULP | 2.94 | 1.55 | 0.95 | 0.000121 | 1085 | 15467 |
| wh | fp32 | `silu_bw` | `default` | 9/65024 defects; rest 5.72e+06 ULP | 5.72e+06 | 841 | 0.937 | 2.98e-07 | 6232 | 2692 |
| wh | fp32 | `sin` | `default` | accurate to |x| <= 28 | 3.36e+07 | 3.07e+04 | 0.96 | 28 | 841 ±30% | 19943 |
| wh | fp32 | `sin_bw` | `default` | 18430/65024 defects; rest 1.24e+52 ULP | 1.24e+52 | 3.44e+48 | 0.808 | 92.4 | 2573 | 6520 |
| wh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155 | 2.21 | 1.17 | 0.943 | 0.0155 | 1094 | 15333 |
| wh | fp32 | `sinh_bw` | `default` | 2/33894 defects; rest 1.35 ULP | 1.35 | 0.798 | 0.98 | 88.5 | 11919 | 1408 |
| wh | fp32 | `softplus` | `default` | 5/65024 defects; rest 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 1439 ±93% | 11656 |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.00163 | 2.95 | 1.34 | 0.954 | 0.00163 | 8683 | 1932 |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 774 | 21665 |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4269 | 3930 |
| wh | fp32 | `softsign` | `default` | 510/65024 defects; rest 2.66 ULP | 2.66 | 0.897 | 0.92 | 0.000462 | 818 | 20499 |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07 | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 2568 | 6532 |
| wh | fp32 | `sqrt` | `default` | faithful; 32512/32512 tie-breaks | 0.867 | 0.827 | 0.869 | 3.39e+38 | 818 | 20519 |
| wh | fp32 | `sqrt_bw` | `default` | never within 2 ULP | 2.24 | 1.44 | 0.702 | — | 12640 | 1327 |
| wh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 765 | 21929 |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2481 | 6763 |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 986 | 17018 |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 1003 | 16721 |
| wh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 4218 | 3977 |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 991 | 16925 |
| wh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 992 | 16919 |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 996 | 16850 |
| wh | fp32 | `swish` | `default` | 9/65024 defects; rest 2.94 ULP | 2.94 | 1.55 | 0.95 | 0.000121 | 1097 ±6% | 15289 |
| wh | fp32 | `tan` | `default` | accurate to |x| <= 3.92 | 2.26e+08 | 2.32e+05 | 0.944 | 3.92 | 1190 | 14095 |
| wh | fp32 | `tan_bw` | `default` | 20623/65024 defects; rest 2.85e+45 ULP | 2.85e+45 | 4.74e+44 | 0.87 | 0.882 | 4414 | 3801 |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000439 | 2.79 | 1.49 | 0.979 | 0.000439 | 940 ±17% | 17844 |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP | 6.59e+04 | 7.51e+03 | 0.482 | — | 1772 ±59% | 9469 |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08 | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 1388 ±8% | 12084 |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09 | 4.19e+06 | 1.01e+05 | 0.901 | 7.42e-09 | 3409 | 4921 |
| wh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21678 |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21728 |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3252 | 5160 |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21836 |
| wh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 1348 ±6% | 12443 |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13 | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 1412 ±8% | 11884 |
| wh | fp32 | `xlogy` | `default` | worst sampled pairing | 2.39e+06 | 1.68e+06 | 0.00256 | — | 992 | 16915 |
| wh | fp32 | `xlogy_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.766 | 0.766 | 0.951 | — | 17481 | 960 |
