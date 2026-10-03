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
| bh | bf16 | `de546d3b146` | 0.79.0 | 66444ba711c0 |
| bh | fp32 | `de546d3b146` | 0.79.0 | 66444ba711c0 |
| wh | bf16 | `de546d3b146` | 0.79.0 | 468273cc8653 |
| wh | fp32 | `de546d3b146` | 0.79.0 | 468273cc8653 |

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
| bh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 | 86846 |
| bh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 598 ±6% | 28059 |
| bh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 294 ±11% | 57128 |
| bh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.744 | 0.996 | 0.949 | 3536 | 4744 |
| bh | bf16 | `acosh` | `default` | within 2 ULP | 1.41 | 0.596 | 0.973 | 3.39e+38 | 366 | 45788 |
| bh | bf16 | `acosh_bw` | `default` | 15874/64514 defects; rest 3.26 ULP | 3.26 | 0.688 | 0.781 | 1.03 | 4053 | 4140 |
| bh | bf16 | `add` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 253 | 66410 |
| bh | bf16 | `add_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 242 | 69386 |
| bh | bf16 | `addalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 247 | 68027 |
| bh | bf16 | `addcdiv` | `default` | worst sampled pairing | 2.11e+06 | 2.35e+03 | 0.999 | — | 327 | 51271 |
| bh | bf16 | `addcmul` | `default` | worst sampled pairing | 126 | 19.4 | 1 | — | 330 | 50903 |
| bh | bf16 | `asin` | `default` | 2/32258 defects; rest 0.499 ULP | 0.499 | 0 | 1 | — | 282 | 59569 |
| bh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938 | 2.76 | 0.787 | 0.992 | 0.938 | 3446 | 4869 |
| bh | bf16 | `asinh` | `default` | within 2 ULP | 1.32 | 0.615 | 0.991 | 3.39e+38 | 501 | 33490 |
| bh | bf16 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.33 ULP | 1.33 | 0.681 | 0.918 | 1.84e+19 | 583 ±37% | 28774 |
| bh | bf16 | `atan` | `default` | 2/65024 defects; rest 0.527 ULP | 0.527 | 0.513 | 1 | — | 209 | 80165 |
| bh | bf16 | `atan2` | `default` | worst sampled pairing | 200 | 3.56 | 0.901 | — | 261 | 64357 |
| bh | bf16 | `atan2_bw` | `default` | worst sampled pairing | 248 | 4.89 | 0.508 | — | 2515 ±11% | 6671 |
| bh | bf16 | `atan_bw` | `default` | 2/48386 defects; rest 2.23 ULP | 2.23 | 0.867 | 0.849 | 0.23 | 639 ±16% | 26242 |
| bh | bf16 | `atanh` | `default` | 2/32256 defects; rest 2.08 ULP | 2.08 | 0.995 | 0.175 | — | 276 ±9% | 60751 |
| bh | bf16 | `atanh_bw` | `default` | 2/48386 defects; rest 8.17 ULP | 8.17 | 1 | 0.79 | 0.82 | 3846 | 4362 |
| bh | bf16 | `bias_gelu` | `default` | worst sampled pairing | 1.14e+36 | 4.5e+34 | 0.749 | — | 247 | 67970 |
| bh | bf16 | `bias_gelu_` | `default` | worst sampled pairing | 1.14e+36 | 4.5e+34 | 0.749 | — | 248 | 67735 |
| bh | bf16 | `cbrt` | `default` | faithful; 846/65024 tie-breaks | 0.507 | 0.502 | 0.987 | 3.39e+38 | 221 ±8% | 76035 |
| bh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 197 ±10% | 85283 |
| bh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 201 ±11% | 83596 |
| bh | bf16 | `celu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1275 | 13160 |
| bh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 205 ±25% | 81810 |
| bh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 202 ±6% | 83163 |
| bh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 | 86171 |
| bh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1117 ±10% | 15019 |
| bh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 | 84699 |
| bh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 88034 |
| bh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 ±6% | 86663 |
| bh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1107 | 15153 |
| bh | bf16 | `cos` | `default` | faithful; 2/37354 tie-breaks | 0.552 | 0.552 | 1 | 9.99e+05 | 272 ±23% | 61636 |
| bh | bf16 | `cos_bw` | `default` | 21281/65024 defects; rest 2.09e+42 ULP | 2.09e+42 | 2.59e+39 | 0.867 | 2.15e+06 | 813 | 20631 |
| bh | bf16 | `cosh` | `default` | faithful; 4/33894 tie-breaks | 0.501 | 0.501 | 1 | 89 | 258 ±8% | 65062 |
| bh | bf16 | `cosh_bw` | `default` | 2/33894 defects; rest 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 3080 ±8% | 5447 |
| bh | bf16 | `deg2rad` | `default` | 2/65024 defects; rest 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 195 | 86254 |
| bh | bf16 | `digamma` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 399 ±12% | 42078 |
| bh | bf16 | `digamma_bw` | `default` | 263/64769 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.36e+04 | 0.674 | 0.996 | 3179 | 5277 |
| bh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 253 ±19% | 66203 |
| bh | bf16 | `div_bw` | `default` | bit-exact | 0.498 | 0 | 1 | — | 4894 | 3428 |
| bh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 666 | 25187 |
| bh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 261 ±28% | 64200 |
| bh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 255 ±9% | 65891 |
| bh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 192 | 87194 |
| bh | bf16 | `elu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1285 | 13052 |
| bh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 253 ±6% | 66342 |
| bh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 246 | 68091 |
| bh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 ±16% | 85580 |
| bh | bf16 | `erf` | `default` | within 2 ULP | 1.06 | 0.771 | 0.729 | 3.39e+38 | 193 ±12% | 86780 |
| bh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 1144 | 14661 |
| bh | bf16 | `erfc` | `default` | 198/65024 unflushed; rest 3.2e+28 ULP | 3.2e+28 | 3.23e+24 | 0.949 | 2.5 | 319 | 52525 |
| bh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 1140 | 14722 |
| bh | bf16 | `erfinv` | `default` | 29424/32256 defects; rest 121 ULP | 121 | 6.34 | 0.391 | 1.31e-38 | 356 ±29% | 47085 |
| bh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777 | 8.91 | 1.28 | 0.98 | 0.777 | 3621 | 4634 |
| bh | bf16 | `exp` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 204 ±8% | 82440 |
| bh | bf16 | `exp` | `fast_approx` | never within 2 ULP | 6.08 | 3.05 | 0.314 | — | 198 | 84532 |
| bh | bf16 | `exp2` | `default` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 231 ±13% | 72519 |
| bh | bf16 | `exp2_bw` | `default` | 1/49537 defects; rest 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 820 | 20458 |
| bh | bf16 | `exp_bw` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 600 ±6% | 27973 |
| bh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 251 ±7% | 66798 |
| bh | bf16 | `expm1_bw` | `default` | accurate to |x| <= 33.5 | 34 | 0.695 | 0.984 | 33.5 | 814 | 20605 |
| bh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 ±21% | 86140 |
| bh | bf16 | `floor_div` | `default` | worst sampled pairing | 64 | 42 | 0.996 | — | 1178 | 14236 |
| bh | bf16 | `fmod` | `default` | worst sampled pairing | 9.14e+35 | 8.94e+34 | 0.673 | — | 261 ±5% | 64206 |
| bh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 199 | 84219 |
| bh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 246 | 68185 |
| bh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 246 ±18% | 68129 |
| bh | bf16 | `gelu` | `default` | 86/65024 defects; rest 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 351 ±116% | 47761 |
| bh | bf16 | `gelu` | `fast_approx` | 198/65024 defects; rest 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 186 ±8% | 90218 |
| bh | bf16 | `gelu_bw` | `default` | faithful; 38/65024 tie-breaks | 0.939 | 0.681 | 0.999 | 3.39e+38 | 706 | 23778 |
| bh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88311 |
| bh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 254 | 66125 |
| bh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 243 | 68902 |
| bh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 ±9% | 84824 |
| bh | bf16 | `hardmish` | `default` | faithful; 2587/65024 tie-breaks | 1 | 0.913 | 0.961 | 3.39e+38 | 194 ±11% | 86593 |
| bh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 ±8% | 87027 |
| bh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21718 |
| bh | bf16 | `hardsigmoid` | `default` | within 2 ULP | 1 | 0.557 | 0.986 | 3.39e+38 | 191 ±13% | 87925 |
| bh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 1183 | 14188 |
| bh | bf16 | `hardswish` | `default` | 2/65024 defects; rest 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 212 ±5% | 79254 |
| bh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.13 | 85.3 | 2.1 | 0.993 | 1.13 | 1655 | 10138 |
| bh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 200 ±6% | 83867 |
| bh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 988 | 16974 |
| bh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 203 ±646% | 82516 |
| bh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85792 |
| bh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 199 ±5% | 84237 |
| bh | bf16 | `hypot` | `default` | 16384/65026 defects; rest 52.7 ULP | 52.7 | 1.58 | 0.947 | — | 271 ±10% | 61844 |
| bh | bf16 | `hypot_bw` | `default` | worst sampled pairing | 74.6 | 2.43 | 0.835 | — | 1515 | 11074 |
| bh | bf16 | `i0` | `default` | accurate to |x| <= 13.6 | 255 | 56.3 | 0.952 | 13.6 | 173 ±6% | 96712 |
| bh | bf16 | `i0_bw` | `default` | 4/33904 defects; rest 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 849 | 19764 |
| bh | bf16 | `i1` | `default` | 4/65024 defects; rest 0.858 ULP | 0.858 | 0.589 | 0.516 | 2.33e-38 | 442 ±6% | 37985 |
| bh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 ±8% | 84805 |
| bh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 262 | 64147 |
| bh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 ±13% | 86987 |
| bh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 200 ±6% | 83817 |
| bh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 202 ±49% | 83150 |
| bh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 | 84856 |
| bh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±7% | 87520 |
| bh | bf16 | `l1_loss` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 245 | 68466 |
| bh | bf16 | `ldexp` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 261 | 64170 |
| bh | bf16 | `ldexp_` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 251 ±6% | 66914 |
| bh | bf16 | `ldexp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.898 | 0.898 | 0.983 | — | 1203 | 13942 |
| bh | bf16 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 258 ±32% | 64978 |
| bh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 244 ±15% | 68641 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 | 86042 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithful; 15341/65024 tie-breaks | 0.96 | 0.741 | 0.761 | 3.39e+38 | 193 ±1010% | 86884 |
| bh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 | 87254 |
| bh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 883 | 18998 |
| bh | bf16 | `lerp` | `default` | worst sampled pairing | 1.77e+03 | 39.9 | 1 | — | 328 ±58% | 51077 |
| bh | bf16 | `lerp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.506 | 0.506 | 0.992 | — | 1267 | 13242 |
| bh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 | 86416 |
| bh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.412 | 324 | 0.846 | 0.76 | 0.412 | 837 | 20042 |
| bh | bf16 | `lgamma_bw` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 800 | 20982 |
| bh | bf16 | `log` | `default` | faithful; 63/32512 tie-breaks | 0.78 | 0.534 | 0.998 | 3.39e+38 | 239 | 70150 |
| bh | bf16 | `log10` | `default` | faithful; 58/32512 tie-breaks | 0.785 | 0.543 | 0.998 | 3.39e+38 | 252 | 66664 |
| bh | bf16 | `log10_bw` | `default` | within 2 ULP | 1.74 | 0.803 | 0.508 | 3.69e+37 | 2427 | 6913 |
| bh | bf16 | `log1p` | `default` | faithful; 109/48640 tie-breaks | 0.931 | 0.57 | 0.998 | 3.39e+38 | 257 | 65212 |
| bh | bf16 | `log1p_bw` | `default` | 2/64514 defects; rest 1.42 ULP | 1.42 | 0.744 | 0.982 | 8.47e+37 | 2421 | 6929 |
| bh | bf16 | `log2` | `default` | faithful; 59/32512 tie-breaks | 0.766 | 0.542 | 0.998 | 3.39e+38 | 256 | 65504 |
| bh | bf16 | `log2_bw` | `default` | 116/64626 defects; rest 1.51 ULP | 1.51 | 0.847 | 0.57 | — | 2434 ±17% | 6894 |
| bh | bf16 | `log_bw` | `default` | 2/64514 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 1849 | 9072 |
| bh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.428 | 6.39 | 1.38 | 0.958 | 0.428 | 235 | 71413 |
| bh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP | 1.71 | 0.722 | 0.976 | 3.39e+38 | 2708 | 6196 |
| bh | bf16 | `logaddexp` | `default` | 15566/65026 defects; rest 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 326 ±10% | 51470 |
| bh | bf16 | `logaddexp2` | `default` | 15488/65026 defects; rest 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 385 | 43581 |
| bh | bf16 | `logaddexp2_` | `default` | 15488/65026 defects; rest 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 385 ±9% | 43559 |
| bh | bf16 | `logaddexp2_bw` | `default` | worst sampled pairing | 41.4 | 3.6 | 0.978 | — | 2510 | 6685 |
| bh | bf16 | `logaddexp_` | `default` | 15566/65026 defects; rest 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 307 | 54690 |
| bh | bf16 | `logaddexp_bw` | `default` | worst sampled pairing | 62.5 | 4.9 | 0.979 | — | 2168 ±89% | 7740 |
| bh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 266 ±91% | 63007 |
| bh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 258 ±30% | 65009 |
| bh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 ±5% | 86403 |
| bh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 194 ±11% | 86272 |
| bh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 261 ±19% | 64248 |
| bh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 264 ±17% | 63518 |
| bh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 254 ±7% | 66041 |
| bh | bf16 | `logit` | `default` | accurate to |x| <= 0.395 | 64 | 1.67 | 0.985 | 0.395 | 337 | 49856 |
| bh | bf16 | `logit_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3204 | 5237 |
| bh | bf16 | `logiteps_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3822 | 4390 |
| bh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 249 ±5% | 67430 |
| bh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 253 | 66195 |
| bh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 | 86051 |
| bh | bf16 | `mac` | `default` | worst sampled pairing | 63.5 | 26.4 | 1 | — | 331 | 50713 |
| bh | bf16 | `max_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 2501 | 6707 |
| bh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 254 ±21% | 65976 |
| bh | bf16 | `min_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 2538 | 6612 |
| bh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 256 ±34% | 65470 |
| bh | bf16 | `mish` | `default` | 11/65024 defects; rest 1.49 ULP | 1.49 | 0.666 | 0.988 | 1.95e-38 | 268 ±7% | 62685 |
| bh | bf16 | `mse_loss` | `default` | worst sampled pairing | 2.03 | 1.65 | 0.571 | — | 254 ±12% | 65975 |
| bh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 650 | 25819 |
| bh | bf16 | `multigammaln` | `default` | 11536/48328 defects; rest 724 ULP | 724 | 1.67 | 0.717 | 5.59e-17 | 4668 | 3594 |
| bh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.59e-17 | 2.38e+06 | 293 | 0.249 | 5.59e-17 | 3815 | 4398 |
| bh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 248 ±7% | 67666 |
| bh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 246 ±6% | 68279 |
| bh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 246 | 68245 |
| bh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 259 ±45% | 64729 |
| bh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 ±8% | 86983 |
| bh | bf16 | `nextafter` | `default` | worst sampled pairing | 1.3e+33 | 2.32e+31 | 0.56 | — | 1457 | 11513 |
| bh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±11% | 87241 |
| bh | bf16 | `polygamma` | `k=1` | 7/49922 defects; rest 1.02e+08 ULP | 1.02e+08 | 2.13e+05 | 0.97 | 0.996 | 347 | 48288 |
| bh | bf16 | `polygamma` | `k=2` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 409 | 41002 |
| bh | bf16 | `polygamma` | `k=4` | 142/49922 defects; rest 3.22e+09 ULP | 3.22e+09 | 1.66e+07 | 0.958 | 3.5 | 443 | 37848 |
| bh | bf16 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 3236 | 5185 |
| bh | bf16 | `pow` | `default` | worst sampled pairing | 4.86e+18 | 2.08e+14 | 0.982 | — | 407 | 41207 |
| bh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1188 | 14128 |
| bh | bf16 | `prelu` | `weight=0.25` | 1/65024 defects; rest 0 ULP | 0 | 0 | 1 | 4.68e-38 | 195 | 86194 |
| bh | bf16 | `rad2deg` | `default` | faithful; 31760/63518 tie-breaks | 0.992 | 0.75 | 0.5 | 5.9e+36 | 198 ±13% | 84644 |
| bh | bf16 | `rdiv` | `value=2.0` | 258/64770 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 195 ±16% | 86062 |
| bh | bf16 | `rdiv_bw` | `scalar=2.0` | 256/48492 defects; rest 1.51 ULP | 1.51 | 0.837 | 0.602 | 7.67e-20 | 3158 | 5312 |
| bh | bf16 | `reciprocal` | `default` | 2/64514 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 194 ±18% | 86429 |
| bh | bf16 | `reciprocal_bw` | `default` | 256/48386 defects; rest 1.51 ULP | 1.51 | 0.837 | 0.602 | 5.42e-20 | 2392 | 7015 |
| bh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 ±7% | 84759 |
| bh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85475 |
| bh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1850 ±6% | 9071 |
| bh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 603 | 27822 |
| bh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 | 87446 |
| bh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 200 | 84036 |
| bh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 197 ±12% | 85279 |
| bh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 | 84905 |
| bh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 87992 |
| bh | bf16 | `remainder` | `default` | worst sampled pairing | 1.08e+36 | 1.22e+35 | 0.672 | — | 268 | 62673 |
| bh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 ±9% | 85498 |
| bh | bf16 | `rpow` | `exponent=0.5` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.975 | 3.39e+38 | 384 ±20% | 43686 |
| bh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 387 ±14% | 43325 |
| bh | bf16 | `rpow` | `exponent=2.0` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 380 ±7% | 44184 |
| bh | bf16 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 1366 | 12282 |
| bh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 248 ±6% | 67538 |
| bh | bf16 | `rsqrt_bw` | `default` | 75/21665 defects; rest 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 2822 | 5946 |
| bh | bf16 | `rsub` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 248 | 67568 |
| bh | bf16 | `rsub_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 255 ±84% | 65857 |
| bh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 208 ±250% | 80723 |
| bh | bf16 | `selu_bw` | `default` | 1/65024 defects; rest 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 1470 | 11410 |
| bh | bf16 | `sigmoid` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 251 ±7% | 66835 |
| bh | bf16 | `sigmoid_accurate` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 253 ±5% | 66316 |
| bh | bf16 | `sigmoid_bw` | `default` | 331/65024 defects; rest 125 ULP | 125 | 4.81 | 0.988 | 1.76 | 1053 | 15935 |
| bh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 ±7% | 85897 |
| bh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 ±5% | 86304 |
| bh | bf16 | `silu` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 168 ±14% | 99808 |
| bh | bf16 | `silu_bw` | `default` | 9/65024 defects; rest 285 ULP | 285 | 1.3 | 0.979 | 0.863 | 1449 | 11578 |
| bh | bf16 | `sin` | `default` | faithful; 4/37354 tie-breaks | 0.552 | 0.526 | 1 | 9.99e+05 | 235 ±5% | 71320 |
| bh | bf16 | `sin_bw` | `default` | 21284/65024 defects; rest 2.98e+41 ULP | 2.98e+41 | 1.39e+39 | 0.857 | 1.07e+06 | 678 | 24745 |
| bh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 186 | 90191 |
| bh | bf16 | `sinh_bw` | `default` | 2/33894 defects; rest 0.501 ULP | 0.501 | 0.501 | 1 | 88.5 | 2806 | 5979 |
| bh | bf16 | `softcap` | `beta=50.0` | 1426/65024 defects; rest 0.718 ULP | 0.718 | 0.574 | 0.997 | — | 236 | 71143 |
| bh | bf16 | `softplus` | `default` | 526/65024 defects; rest 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 192 ±7% | 87300 |
| bh | bf16 | `softplus_bw` | `default` | within 2 ULP | 1.71 | 0.678 | 0.983 | 3.39e+38 | 2015 | 8328 |
| bh | bf16 | `softshrink` | `default` | faithful; 3968/65024 tie-breaks | 1 | 0.948 | 0.941 | 3.39e+38 | 203 | 82652 |
| bh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 988 | 16984 |
| bh | bf16 | `softsign` | `default` | 512/65024 defects; rest 1 ULP | 1 | 0.605 | 0.907 | 8.47e+37 | 196 | 85457 |
| bh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391 | 81 | 3.24 | 0.856 | 0.00391 | 622 | 26977 |
| bh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 233 ±11% | 72068 |
| bh | bf16 | `sqrt_bw` | `default` | within 2 ULP | 1.14 | 0.699 | 0.723 | 3.39e+38 | 3022 | 5552 |
| bh | bf16 | `square` | `default` | faithful; 13970/48640 tie-breaks | 0.973 | 0.688 | 0.578 | 1.84e+19 | 196 | 85576 |
| bh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 613 ±25% | 27381 |
| bh | bf16 | `squared_difference` | `default` | worst sampled pairing | 2.03 | 1.65 | 0.571 | — | 248 ±7% | 67751 |
| bh | bf16 | `squared_difference_` | `default` | worst sampled pairing | 2.03 | 1.65 | 0.571 | — | 245 | 68584 |
| bh | bf16 | `squared_difference_bw` | `default` | faithful; 64842/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 1024 | 16381 |
| bh | bf16 | `subalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 249 | 67352 |
| bh | bf16 | `subtract` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 249 ±5% | 67366 |
| bh | bf16 | `subtract_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 245 ±15% | 68454 |
| bh | bf16 | `swish` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 176 ±7% | 95084 |
| bh | bf16 | `tan` | `default` | faithful; 30/37354 tie-breaks | 0.551 | 0.509 | 0.999 | 9.99e+05 | 213 ±10% | 78910 |
| bh | bf16 | `tan_bw` | `default` | 22782/65024 defects; rest 3.57e+40 ULP | 3.57e+40 | 1.01e+38 | 0.855 | 1.13 | 970 | 17300 |
| bh | bf16 | `tanh` | `default` | 2/65024 defects; rest 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 198 | 84781 |
| bh | bf16 | `tanh_bw` | `default` | within 2 ULP | 1.16 | 0.525 | 0.997 | 3.39e+38 | 618 | 27158 |
| bh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08 | 63.5 | 9.38 | 0.981 | 1.35e-08 | 180 ±6% | 93293 |
| bh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09 | 63 | 3.52 | 0.937 | 7.45e-09 | 770 | 21775 |
| bh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 | 84597 |
| bh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 199 ±28% | 84436 |
| bh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 ±21% | 21798 |
| bh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 201 ±14% | 83302 |
| bh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 325 | 51656 |
| bh | bf16 | `xielu` | `default` | 1/56847 defects; rest 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 436 | 38480 |
| bh | bf16 | `xlogy` | `default` | worst sampled pairing | 23.5 | 17.6 | 0.139 | — | 257 ±7% | 65373 |
| bh | bf16 | `xlogy_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.78 | 0.78 | 0.998 | — | 4112 | 4080 |
| bh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47732 |
| bh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1134 | 14790 |
| bh | fp32 | `acos` | `default` | within 2 ULP | 1.55 | 0.865 | 0.923 | 1 | 473 ±5% | 35483 |
| bh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 6841 | 2452 |
| bh | fp32 | `acosh` | `default` | never within 2 ULP | 2.47 | 0.76 | 0.782 | — | 453 | 37023 |
| bh | fp32 | `acosh_bw` | `default` | 15874/64514 defects; rest 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 7846 | 2138 |
| bh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 482 ±6% | 34838 |
| bh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 487 ±6% | 34416 |
| bh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 487 ±6% | 34455 |
| bh | fp32 | `addcdiv` | `default` | worst sampled pairing | 2.19e+12 | 8.82e+07 | 0.852 | — | 650 | 25800 |
| bh | fp32 | `addcmul` | `default` | worst sampled pairing | 2.13e+06 | 1.59e+04 | 0.985 | — | 649 | 25844 |
| bh | fp32 | `asin` | `default` | within 2 ULP | 1.77 | 0.795 | 0.985 | 1 | 455 ±7% | 36900 |
| bh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 6698 | 2505 |
| bh | fp32 | `asinh` | `default` | within 2 ULP | 1.64 | 0.773 | 0.881 | 3.39e+38 | 602 | 27856 |
| bh | fp32 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 1187 | 14140 |
| bh | fp32 | `atan` | `default` | accurate to |x| <= 0.902 | 2.34 | 0.806 | 0.955 | 0.902 | 400 | 41949 |
| bh | fp32 | `atan2` | `default` | worst sampled pairing | 1.32e+07 | 1.26e+05 | 0.857 | — | 505 | 33237 |
| bh | fp32 | `atan2_bw` | `default` | worst sampled pairing | 1.64e+07 | 1.66e+05 | 0.587 | — | 4896 | 3426 |
| bh | fp32 | `atan_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.25e+03 | 0.859 | 1.02 | 1146 | 14636 |
| bh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000229 | 2.97 | 1.51 | 0.912 | 0.000229 | 343 | 48876 |
| bh | fp32 | `atanh_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.31e+03 | 0.876 | 0.681 | 7523 | 2230 |
| bh | fp32 | `bias_gelu` | `default` | worst sampled pairing | 7.45e+40 | 6.51e+39 | 0.746 | — | 480 ±7% | 34956 |
| bh | fp32 | `bias_gelu_` | `default` | worst sampled pairing | 7.45e+40 | 6.51e+39 | 0.746 | — | 480 ±7% | 34985 |
| bh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38 | 2.55 | 1.76 | 0.521 | 2.26e-38 | 398 ±7% | 42190 |
| bh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 | 47614 |
| bh | fp32 | `celu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 416 ±60% | 40282 |
| bh | fp32 | `celu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2496 | 6722 |
| bh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 359 ±6% | 46763 |
| bh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47944 |
| bh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 | 48014 |
| bh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2191 | 7656 |
| bh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 | 48009 |
| bh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48557 |
| bh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 | 47729 |
| bh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2184 | 7682 |
| bh | fp32 | `cos` | `default` | accurate to |x| <= 92.4 | 1.34e+08 | 1.33e+05 | 0.927 | 92.4 | 405 ±7% | 41474 |
| bh | fp32 | `cos_bw` | `default` | 18643/65024 defects; rest 8.59e+51 ULP | 8.59e+51 | 3.23e+48 | 0.841 | 28 | 1496 | 11214 |
| bh | fp32 | `cosh` | `default` | within 2 ULP | 1.35 | 0.791 | 0.98 | 89.1 | 390 | 43022 |
| bh | fp32 | `cosh_bw` | `default` | 2/33894 defects; rest 2.2 ULP | 2.2 | 1.16 | 0.943 | 0.0155 | 6128 | 2738 |
| bh | fp32 | `deg2rad` | `default` | faithful; 63542/65024 tie-breaks | 0.63 | 0.595 | 0.906 | 3.4e+38 | 354 ±12% | 47338 |
| bh | fp32 | `digamma` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00543 | — | 737 | 22772 |
| bh | fp32 | `digamma_bw` | `default` | 257/64769 defects; rest 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.324 | 5.4e-20 | 6580 | 2550 |
| bh | fp32 | `div` | `default` | worst sampled pairing | 2.44 | 1.44 | 0.894 | — | 490 ±6% | 34229 |
| bh | fp32 | `div_bw` | `default` | worst sampled pairing | 7.95e+04 | 7.95e+04 | 0.89 | — | 9437 | 1778 |
| bh | fp32 | `div_no_nan` | `default` | worst sampled pairing | 9.24e+04 | 4.51e+04 | 0.702 | — | 1603 | 10467 |
| bh | fp32 | `divide` | `default` | worst sampled pairing | 2.44 | 1.44 | 0.894 | — | 492 | 34109 |
| bh | fp32 | `divide_` | `default` | worst sampled pairing | 2.44 | 1.44 | 0.894 | — | 489 | 34282 |
| bh | fp32 | `elu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 410 | 40918 |
| bh | fp32 | `elu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2486 | 6749 |
| bh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 482 ±8% | 34824 |
| bh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 486 | 34494 |
| bh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47871 |
| bh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334 | 7.38 | 1.36 | 0.672 | 0.000334 | 529 | 31714 |
| bh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 2197 | 7636 |
| bh | fp32 | `erfc` | `default` | 198/65024 unflushed; rest 2.1e+33 ULP | 2.1e+33 | 1.71e+29 | 0.329 | — | 369 ±10% | 45407 |
| bh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 2180 | 7696 |
| bh | fp32 | `erfinv` | `default` | 29420/32256 defects; rest 7.96e+06 ULP | 7.96e+06 | 2.59e+05 | 0.000202 | 1.32e-38 | 360 | 46656 |
| bh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456 | 1.03e+06 | 1.67e+03 | 0.914 | 0.000456 | 6749 | 2486 |
| bh | fp32 | `exp` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 402 ±6% | 41751 |
| bh | fp32 | `exp` | `fast_approx` | never within 2 ULP | 3.79e+05 | 2e+05 | 0.309 | — | 345 | 48604 |
| bh | fp32 | `exp2` | `default` | faithful; 6044/49536 tie-breaks | 0.965 | 0.626 | 0.992 | 3.39e+38 | 384 | 43706 |
| bh | fp32 | `exp2_bw` | `default` | 1/49537 defects; rest 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 1506 | 11143 |
| bh | fp32 | `exp_bw` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 1181 | 14202 |
| bh | fp32 | `expm1` | `default` | faithful; 5257/49458 tie-breaks | 0.997 | 0.565 | 0.997 | 3.39e+38 | 458 | 36594 |
| bh | fp32 | `expm1_bw` | `default` | accurate to |x| <= 22 | 4.19e+06 | 3.83e+03 | 0.995 | 22 | 1578 | 10629 |
| bh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 | 48070 |
| bh | fp32 | `floor_div` | `default` | worst sampled pairing | 4.19e+06 | 1.56e+03 | 0.974 | — | 2266 | 7404 |
| bh | fp32 | `fmod` | `default` | worst sampled pairing | 1.61e+45 | 2.55e+41 | 0.596 | — | 497 | 33756 |
| bh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 359 ±12% | 46776 |
| bh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 479 | 35010 |
| bh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 481 | 34898 |
| bh | fp32 | `gelu` | `default` | 83/65024 defects; rest 1.71e+08 ULP | 1.71e+08 | 1.7e+05 | 0.963 | 0.208 | 368 | 45597 |
| bh | fp32 | `gelu` | `fast_approx` | 197/65024 defects; rest 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 348 | 48204 |
| bh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0298 | 1.47e+08 | 2.3e+04 | 0.929 | 0.0298 | 832 | 20175 |
| bh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 353 ±10% | 47576 |
| bh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 485 | 34618 |
| bh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 480 ±8% | 34942 |
| bh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 355 | 47321 |
| bh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 354 ±47% | 47357 |
| bh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48144 |
| bh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1477 | 11358 |
| bh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25 | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 352 | 47601 |
| bh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2311 | 7259 |
| bh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36 | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 372 ±6% | 45047 |
| bh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133 | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 3253 | 5158 |
| bh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 355 | 47267 |
| bh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1970 | 8517 |
| bh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 357 | 46977 |
| bh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 356 ±19% | 47081 |
| bh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47867 |
| bh | fp32 | `hypot` | `default` | 16384/65024 defects; rest 3.44e+06 ULP | 3.44e+06 | 3.28e+04 | 0.965 | — | 512 | 32773 |
| bh | fp32 | `hypot_bw` | `default` | 16384/65024 defects; rest 4.86e+06 ULP | 4.86e+06 | 4.48e+04 | 0.826 | — | 2985 | 5620 |
| bh | fp32 | `i0` | `default` | accurate to |x| <= 2.45 | 1.68e+07 | 2.19e+06 | 0.959 | 2.45 | 403 | 41623 |
| bh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00362 | 1.56e+07 | 4.1e+04 | 0.928 | 0.00362 | 1357 | 12367 |
| bh | fp32 | `i1` | `default` | accurate to |x| <= 0.00362 | 8.02 | 0.731 | 0.482 | 0.00362 | 568 ±7% | 29552 |
| bh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 353 | 47486 |
| bh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 493 ±6% | 34065 |
| bh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48290 |
| bh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 356 | 47120 |
| bh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47770 |
| bh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 356 | 47178 |
| bh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47817 |
| bh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 478 | 35120 |
| bh | fp32 | `ldexp` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 498 | 33682 |
| bh | fp32 | `ldexp_` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 491 | 34175 |
| bh | fp32 | `ldexp_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.742 | 0.742 | 0.996 | — | 2214 | 7578 |
| bh | fp32 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 482 ±6% | 34771 |
| bh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 482 ±9% | 34810 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 354 | 47376 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithful; 31672/65024 tie-breaks | 0.84 | 0.747 | 0.868 | 3.39e+38 | 347 | 48330 |
| bh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47947 |
| bh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 1668 | 10055 |
| bh | fp32 | `lerp` | `default` | worst sampled pairing | 1.46e+08 | 9.53e+04 | 0.962 | — | 646 | 25964 |
| bh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 2515 | 6670 |
| bh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 355 ±6% | 47228 |
| bh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181 | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 1336 | 12555 |
| bh | fp32 | `lgamma_bw` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00543 | — | 1517 | 11060 |
| bh | fp32 | `log` | `default` | faithful; 32512/32512 tie-breaks | 0.961 | 0.562 | 0.952 | 3.39e+38 | 407 | 41222 |
| bh | fp32 | `log10` | `default` | accurate to |x| <= 0.336 | 2.13 | 1.36 | 0.641 | 0.336 | 398 | 42152 |
| bh | fp32 | `log10_bw` | `default` | accurate to |x| <= 4.42e+30 | 9.02e+04 | 1.73e+03 | 0.667 | 4.42e+30 | 4689 | 3578 |
| bh | fp32 | `log1p` | `default` | faithful; 20111/48640 tie-breaks | 0.967 | 0.558 | 0.982 | 3.39e+38 | 392 | 42774 |
| bh | fp32 | `log1p_bw` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 2.96e+03 | 0.89 | 2.02e+31 | 4709 | 3563 |
| bh | fp32 | `log2` | `default` | accurate to |x| <= 0.704 | 2.43 | 0.517 | 0.993 | 0.704 | 401 ±6% | 41859 |
| bh | fp32 | `log2_bw` | `default` | 112/64626 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.76e+03 | 0.684 | — | 4700 | 3570 |
| bh | fp32 | `log_bw` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 3572 | 4696 |
| bh | fp32 | `log_sigmoid` | `default` | never within 2 ULP | 4.4e+05 | 1.82e+04 | 0.481 | — | 481 | 34856 |
| bh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.163 | 2.93 | 1.28 | 0.953 | 0.163 | 5210 | 3220 |
| bh | fp32 | `logaddexp` | `default` | 15566/65024 defects; rest 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 592 ±7% | 28356 |
| bh | fp32 | `logaddexp2` | `default` | 15488/65024 defects; rest 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.715 | — | 511 | 32843 |
| bh | fp32 | `logaddexp2_` | `default` | 15488/65024 defects; rest 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.715 | — | 510 | 32869 |
| bh | fp32 | `logaddexp2_bw` | `default` | worst sampled pairing | 9.02e+04 | 5.6e+04 | 0.959 | — | 4729 | 3548 |
| bh | fp32 | `logaddexp_` | `default` | 15566/65024 defects; rest 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 592 | 28344 |
| bh | fp32 | `logaddexp_bw` | `default` | worst sampled pairing | 8.98e+04 | 4.46e+04 | 0.962 | — | 4281 | 3919 |
| bh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 514 ±6% | 32620 |
| bh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 500 ±6% | 33530 |
| bh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 | 47680 |
| bh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 374 ±13% | 44904 |
| bh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 493 | 34036 |
| bh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 488 | 34389 |
| bh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 496 | 33824 |
| bh | fp32 | `logit` | `default` | accurate to |x| <= 0.266 | 4.19e+06 | 261 | 0.941 | 0.266 | 367 | 45777 |
| bh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 6326 | 2652 |
| bh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 7487 | 2241 |
| bh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 498 ±6% | 33691 |
| bh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 472 | 35529 |
| bh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 356 | 47106 |
| bh | fp32 | `mac` | `default` | worst sampled pairing | 4.22e+06 | 7.31e+05 | 0.974 | — | 654 | 25672 |
| bh | fp32 | `max_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 4874 | 3442 |
| bh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 491 ±6% | 34199 |
| bh | fp32 | `min_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 4881 | 3437 |
| bh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 489 ±5% | 34341 |
| bh | fp32 | `mish` | `default` | 9/65024 defects; rest 7 ULP | 7 | 2.58 | 0.953 | 1.49e-07 | 460 ±8% | 36511 |
| bh | fp32 | `mse_loss` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 484 ±8% | 34690 |
| bh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1262 | 13290 |
| bh | fp32 | `multigammaln` | `default` | 7422/50376 defects; rest 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.55e-17 | 7980 | 2102 |
| bh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 1.36e+15 | 2.04e+11 | 0.0102 | 5.55e-17 | 7375 | 2275 |
| bh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 482 | 34783 |
| bh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 498 ±5% | 33667 |
| bh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 485 ±5% | 34598 |
| bh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 473 | 35489 |
| bh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47759 |
| bh | fp32 | `nextafter` | `default` | worst sampled pairing | 8.51e+37 | 1.34e+36 | 0.5 | — | 2892 | 5801 |
| bh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47743 |
| bh | fp32 | `polygamma` | `k=1` | 1/49922 defects; rest 7.91e+33 ULP | 7.91e+33 | 4.68e+29 | 0.467 | 5.4e-20 | 1032 | 16254 |
| bh | fp32 | `polygamma` | `k=2` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 1079 | 15543 |
| bh | fp32 | `polygamma` | `k=4` | 141/49922 defects; rest 3.77e+25 ULP | 3.77e+25 | 6.51e+21 | 0.138 | 3.68e-08 | 1092 | 15365 |
| bh | fp32 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 6621 | 2534 |
| bh | fp32 | `pow` | `default` | worst sampled pairing | 333 | 2.35 | 0.993 | — | 662 ±9% | 25328 |
| bh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2279 ±26% | 7363 |
| bh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 364 ±39% | 46062 |
| bh | fp32 | `rad2deg` | `default` | faithful; 63518/63518 tie-breaks | 0.696 | 0.643 | 0.858 | 5.93e+36 | 347 | 48298 |
| bh | fp32 | `rdiv` | `value=2.0` | 258/64770 defects; rest 1.49 ULP | 1.49 | 1.19 | 0.664 | 8.47e+37 | 350 | 47930 |
| bh | fp32 | `rdiv_bw` | `scalar=2.0` | 254/48490 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.92e+03 | 0.705 | 7.67e-20 | 6089 | 2756 |
| bh | fp32 | `reciprocal` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 349 | 48126 |
| bh | fp32 | `reciprocal_bw` | `default` | 256/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.92e+03 | 0.705 | 5.42e-20 | 4561 | 3678 |
| bh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 355 | 47314 |
| bh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 355 | 47295 |
| bh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3617 | 4639 |
| bh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1148 | 14614 |
| bh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48428 |
| bh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 356 | 47074 |
| bh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47976 |
| bh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 353 ±10% | 47461 |
| bh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 355 | 47255 |
| bh | fp32 | `remainder` | `default` | worst sampled pairing | 8.92e+44 | 2.49e+41 | 0.596 | — | 503 | 33370 |
| bh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 359 ±8% | 46744 |
| bh | fp32 | `rpow` | `exponent=0.5` | faithful; 5624/49536 tie-breaks | 0.879 | 0.609 | 0.995 | 3.39e+38 | 632 | 26535 |
| bh | fp32 | `rpow` | `exponent=1.0` | 1536/49536 defects; rest 0 ULP | 0 | 0 | 1 | 8.28e+34 | 630 | 26629 |
| bh | fp32 | `rpow` | `exponent=2.0` | 1536/49536 defects; rest 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 631 ±6% | 26568 |
| bh | fp32 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 2598 | 6457 |
| bh | fp32 | `rsqrt` | `default` | within 2 ULP | 1.12 | 0.834 | 0.864 | 3.39e+38 | 388 | 43246 |
| bh | fp32 | `rsqrt_bw` | `default` | 75/21666 defects; rest 6.21 ULP | 6.21 | 4.15 | 0.303 | — | 5238 | 3203 |
| bh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 481 | 34855 |
| bh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 479 | 34990 |
| bh | fp32 | `selu` | `default` | never within 2 ULP | 51.3 | 26.3 | 0 | — | 433 | 38786 |
| bh | fp32 | `selu_bw` | `default` | 1/65024 defects; rest 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 2817 | 5955 |
| bh | fp32 | `sigmoid` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 425 ±7% | 39502 |
| bh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 420 | 39938 |
| bh | fp32 | `sigmoid_bw` | `default` | 140/65024 defects; rest 8.39e+06 ULP | 8.39e+06 | 2.54e+04 | 0.973 | 0.283 | 2013 | 8333 |
| bh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 ±6% | 48068 |
| bh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48582 |
| bh | fp32 | `silu` | `default` | 9/65024 defects; rest 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 426 ±9% | 39421 |
| bh | fp32 | `silu_bw` | `default` | 9/65024 defects; rest 5.72e+06 ULP | 5.72e+06 | 842 | 0.934 | 2.98e-07 | 2804 | 5984 |
| bh | fp32 | `sin` | `default` | accurate to |x| <= 28 | 3.36e+07 | 3.07e+04 | 0.96 | 28 | 379 | 44290 |
| bh | fp32 | `sin_bw` | `default` | 18430/65024 defects; rest 1.24e+52 ULP | 1.24e+52 | 3.44e+48 | 0.808 | 92.4 | 1200 | 13987 |
| bh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155 | 2.2 | 1.16 | 0.943 | 0.0155 | 459 | 36542 |
| bh | fp32 | `sinh_bw` | `default` | 2/33894 defects; rest 1.35 ULP | 1.35 | 0.791 | 0.98 | 88.5 | 5392 | 3111 |
| bh | fp32 | `softplus` | `default` | 5/65024 defects; rest 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 414 | 40544 |
| bh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.164 | 2.93 | 1.33 | 0.954 | 0.164 | 3947 ±13% | 4250 |
| bh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 367 ±104% | 45762 |
| bh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1962 | 8551 |
| bh | fp32 | `softsign` | `default` | 512/65024 defects; rest 3 ULP | 3 | 1.28 | 0.863 | 1.21e-05 | 350 ±9% | 47893 |
| bh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07 | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 1167 | 14379 |
| bh | fp32 | `sqrt` | `default` | faithful; 32512/32512 tie-breaks | 0.867 | 0.827 | 0.869 | 3.39e+38 | 376 ±7% | 44668 |
| bh | fp32 | `sqrt_bw` | `default` | never within 2 ULP | 2.19 | 1.4 | 0.708 | — | 5785 | 2900 |
| bh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 351 | 47731 |
| bh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1145 | 14649 |
| bh | fp32 | `squared_difference` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 486 ±20% | 34489 |
| bh | fp32 | `squared_difference_` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 479 | 35054 |
| bh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1943 | 8635 |
| bh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 486 | 34537 |
| bh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 478 | 35106 |
| bh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 477 ±8% | 35177 |
| bh | fp32 | `swish` | `default` | 9/65024 defects; rest 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 434 | 38638 |
| bh | fp32 | `tan` | `default` | accurate to |x| <= 3.92 | 2.26e+08 | 2.32e+05 | 0.944 | 3.92 | 469 | 35767 |
| bh | fp32 | `tan_bw` | `default` | 20623/65024 defects; rest 2.85e+45 ULP | 2.85e+45 | 4.74e+44 | 0.87 | 0.852 | 1932 | 8682 |
| bh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000946 | 2.6 | 1.45 | 0.98 | 0.000946 | 395 | 42456 |
| bh | fp32 | `tanh_bw` | `default` | never within 2 ULP | 6.59e+04 | 7.51e+03 | 0.482 | — | 860 ±44% | 19510 |
| bh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08 | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 360 | 46647 |
| bh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09 | 4.19e+06 | 1.02e+05 | 0.901 | 7.42e-09 | 1514 | 11082 |
| bh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 358 | 46849 |
| bh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 358 ±7% | 46910 |
| bh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1486 | 11292 |
| bh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 353 | 47558 |
| bh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 650 | 25798 |
| bh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13 | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 433 | 38753 |
| bh | fp32 | `xlogy` | `default` | worst sampled pairing | 2.39e+06 | 1.68e+06 | 0.00256 | — | 495 ±8% | 33889 |
| bh | fp32 | `xlogy_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.766 | 0.766 | 0.951 | — | 8029 | 2090 |
| wh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 392 ±6% | 42799 |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1318 | 12730 |
| wh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 671 | 25007 |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.744 | 0.996 | 0.949 | 7515 | 2233 |
| wh | bf16 | `acosh` | `default` | within 2 ULP | 1.41 | 0.596 | 0.973 | 3.39e+38 | 871 | 19271 |
| wh | bf16 | `acosh_bw` | `default` | 15874/64514 defects; rest 3.26 ULP | 3.26 | 0.688 | 0.781 | 1.03 | 8691 | 1930 |
| wh | bf16 | `add` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 532 | 31515 |
| wh | bf16 | `add_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 549 | 30586 |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | 0.997 | — | 539 | 31134 |
| wh | bf16 | `addcdiv` | `default` | worst sampled pairing | 2.11e+06 | 2.35e+03 | 0.999 | — | 692 | 24258 |
| wh | bf16 | `addcmul` | `default` | worst sampled pairing | 126 | 19.4 | 1 | — | 681 | 24625 |
| wh | bf16 | `asin` | `default` | 2/32258 defects; rest 0.499 ULP | 0.499 | 0 | 1 | — | 660 | 25416 |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938 | 2.76 | 0.787 | 0.992 | 0.938 | 7449 | 2252 |
| wh | bf16 | `asinh` | `default` | within 2 ULP | 1.32 | 0.619 | 0.992 | 3.39e+38 | 1330 | 12610 |
| wh | bf16 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.33 ULP | 1.33 | 0.681 | 0.918 | 1.84e+19 | 1346 | 12466 |
| wh | bf16 | `atan` | `default` | 2/65024 defects; rest 0.527 ULP | 0.527 | 0.513 | 1 | — | 571 ±8% | 29368 |
| wh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 2.51 | 200 | 2.51 | 0.998 | — | 542 | 30960 |
| wh | bf16 | `atan2_bw` | `default` | worst sampled pairing | 248 | 4.88 | 0.522 | — | 5521 | 3039 |
| wh | bf16 | `atan_bw` | `default` | 2/48386 defects; rest 2.23 ULP | 2.23 | 0.867 | 0.854 | 0.23 | 1346 | 12465 |
| wh | bf16 | `atanh` | `default` | 2/32256 defects; rest 2.08 ULP | 2.08 | 0.995 | 0.081 | — | 651 | 25784 |
| wh | bf16 | `atanh_bw` | `default` | 2/48386 defects; rest 8.17 ULP | 8.17 | 1.02 | 0.84 | 0.82 | 8466 | 1982 |
| wh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | 0.749 | — | 535 | 31369 |
| wh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | 0.749 | — | 546 ±7% | 30718 |
| wh | bf16 | `cbrt` | `default` | faithful; 846/65024 tie-breaks | 0.507 | 0.502 | 0.987 | 3.39e+38 | 425 ±10% | 39477 |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 ±6% | 41772 |
| wh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 446 ±7% | 37624 |
| wh | bf16 | `celu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2712 | 6186 |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 ±6% | 42227 |
| wh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 ±5% | 41659 |
| wh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 | 42036 |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2461 | 6816 |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 | 42300 |
| wh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 | 41786 |
| wh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 ±11% | 41759 |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2451 | 6846 |
| wh | bf16 | `cos` | `default` | faithful; 2/37354 tie-breaks | 0.552 | 0.552 | 1 | 9.99e+05 | 431 ±7% | 38903 |
| wh | bf16 | `cos_bw` | `default` | 21281/65024 defects; rest 2.09e+42 ULP | 2.09e+42 | 2.59e+39 | 0.867 | 2.15e+06 | 1700 | 9870 |
| wh | bf16 | `cosh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 429 ±8% | 39074 |
| wh | bf16 | `cosh_bw` | `default` | 2/33894 defects; rest 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 6670 ±18% | 2516 |
| wh | bf16 | `deg2rad` | `default` | 2/65024 defects; rest 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 400 ±7% | 41993 |
| wh | bf16 | `digamma` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 1152 | 14561 |
| wh | bf16 | `digamma_bw` | `default` | 263/64769 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.32e+04 | 0.664 | 1 | 10926 | 1536 |
| wh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 549 | 30576 |
| wh | bf16 | `div_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.512 | 0.512 | 0.984 | — | 3473 | 4831 |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 1448 | 11584 |
| wh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 546 | 30700 |
| wh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 561 | 29925 |
| wh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 453 ±8% | 37018 |
| wh | bf16 | `elu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2734 | 6137 |
| wh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 537 | 31252 |
| wh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 548 | 30627 |
| wh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 ±6% | 42237 |
| wh | bf16 | `erf` | `default` | within 2 ULP | 1.06 | 0.771 | 0.729 | 3.39e+38 | 571 ±12% | 29394 |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 2444 | 6866 |
| wh | bf16 | `erfc` | `default` | 198/65024 unflushed; rest 3.19e+28 ULP | 3.19e+28 | 3.23e+24 | 0.969 | 2.5 | 811 | 20678 |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 2445 | 6863 |
| wh | bf16 | `erfinv` | `default` | 29424/32256 defects; rest 121 ULP | 121 | 6.34 | 0.391 | 1.31e-38 | 802 | 20929 |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777 | 8.91 | 1.28 | 0.98 | 0.777 | 7795 | 2152 |
| wh | bf16 | `exp` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 420 ±7% | 39919 |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP | 6.08 | 3.05 | 0.314 | — | 401 | 41834 |
| wh | bf16 | `exp2` | `default` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 435 ±8% | 38565 |
| wh | bf16 | `exp2_bw` | `default` | 1/49537 defects; rest 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 1694 | 9903 |
| wh | bf16 | `exp_bw` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 1301 | 12895 |
| wh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 422 ±10% | 39760 |
| wh | bf16 | `expm1_bw` | `default` | accurate to |x| <= 33.2 | 34 | 0.695 | 0.984 | 33.2 | 1309 | 12819 |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 | 41736 |
| wh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | 0.996 | — | 2524 | 6648 |
| wh | bf16 | `fmod` | `default` | worst pairing 6.65e+35 ULP; mean 7.8e+34 | 6.65e+35 | 7.8e+34 | 0.853 | — | 660 | 25433 |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±8% | 42051 |
| wh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 533 | 31495 |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 551 | 30436 |
| wh | bf16 | `gelu` | `default` | 86/65024 defects; rest 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 742 | 22601 |
| wh | bf16 | `gelu` | `fast_approx` | 198/65024 defects; rest 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 398 ±8% | 42179 |
| wh | bf16 | `gelu_bw` | `default` | faithful; 38/65024 tie-breaks | 0.939 | 0.681 | 0.999 | 3.39e+38 | 1725 | 9728 |
| wh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±6% | 41561 |
| wh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 535 ±5% | 31372 |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 553 ±6% | 30323 |
| wh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 | 42097 |
| wh | bf16 | `hardmish` | `default` | faithful; 2587/65024 tie-breaks | 1 | 0.913 | 0.961 | 3.39e+38 | 398 ±5% | 42194 |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 | 42275 |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1688 | 9937 |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP | 1 | 0.557 | 0.986 | 3.39e+38 | 397 | 42289 |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 2545 | 6592 |
| wh | bf16 | `hardswish` | `default` | 2/65024 defects; rest 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 412 ±9% | 40739 |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12 | 85.3 | 2.1 | 0.993 | 1.12 | 3545 | 4733 |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 | 42553 |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2154 | 7789 |
| wh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 | 42093 |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 ±6% | 41650 |
| wh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±9% | 42071 |
| wh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists; the rest reach 52.7 ULP | 52.7 | 1.58 | 0.947 | — | 545 | 30787 |
| wh | bf16 | `hypot_bw` | `default` | worst sampled pairing | 74.6 | 2.42 | 0.833 | — | 3373 | 4974 |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.6 | 255 | 56.3 | 0.952 | 13.6 | 425 ±10% | 39512 |
| wh | bf16 | `i0_bw` | `default` | 4/33904 defects; rest 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 2055 | 8164 |
| wh | bf16 | `i1` | `default` | 4/65024 defects; rest 0.858 ULP | 0.858 | 0.589 | 0.516 | 2.33e-38 | 1164 | 14414 |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 | 42466 |
| wh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 549 | 30579 |
| wh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 ±6% | 42548 |
| wh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 | 41599 |
| wh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 ±7% | 41801 |
| wh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 | 42504 |
| wh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 ±6% | 41584 |
| wh | bf16 | `l1_loss` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 529 | 31688 |
| wh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | 0.955 | — | 544 | 30853 |
| wh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | 0.955 | — | 556 | 30159 |
| wh | bf16 | `ldexp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.898 | 0.898 | 0.983 | — | 2745 | 6112 |
| wh | bf16 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 533 | 31449 |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 547 | 30652 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 | 42343 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithful; 15341/65024 tie-breaks | 0.96 | 0.741 | 0.761 | 3.39e+38 | 395 | 42439 |
| wh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 ±5% | 42283 |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 1827 | 9184 |
| wh | bf16 | `lerp` | `default` | worst sampled pairing | 1.77e+03 | 39.9 | 1 | — | 688 | 24374 |
| wh | bf16 | `lerp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.506 | 0.506 | 0.992 | — | 2807 | 5978 |
| wh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 | 42045 |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.414 | 324 | 0.846 | 0.76 | 0.414 | 2439 | 6880 |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 2037 | 8237 |
| wh | bf16 | `log` | `default` | faithful; 63/32512 tie-breaks | 0.78 | 0.534 | 0.998 | 3.39e+38 | 425 ±9% | 39450 |
| wh | bf16 | `log10` | `default` | faithful; 58/32512 tie-breaks | 0.785 | 0.543 | 0.998 | 3.39e+38 | 433 ±12% | 38760 |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP | 1.74 | 0.796 | 0.524 | 3.69e+37 | 5286 | 3174 |
| wh | bf16 | `log1p` | `default` | faithful; 109/48640 tie-breaks | 0.931 | 0.57 | 0.998 | 3.39e+38 | 439 ±11% | 38175 |
| wh | bf16 | `log1p_bw` | `default` | 2/64514 defects; rest 1.42 ULP | 1.42 | 0.68 | 0.975 | 8.47e+37 | 5287 | 3174 |
| wh | bf16 | `log2` | `default` | faithful; 59/32512 tie-breaks | 0.766 | 0.542 | 0.998 | 3.39e+38 | 425 ±15% | 39512 |
| wh | bf16 | `log2_bw` | `default` | 116/64626 defects; rest 1.51 ULP | 1.51 | 0.842 | 0.578 | — | 5276 | 3180 |
| wh | bf16 | `log_bw` | `default` | 2/64514 defects; rest 0.512 ULP | 0.512 | 0.507 | 0.984 | 8.47e+37 | 3994 | 4201 |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.426 | 6.39 | 1.38 | 0.958 | 0.426 | 595 ±6% | 28207 |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP | 1.91 | 0.785 | 0.972 | 3.39e+38 | 5920 | 2834 |
| wh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists; the rest reach 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 788 | 21292 |
| wh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists; the rest reach 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 918 | 18270 |
| wh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists; the rest reach 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 914 | 18362 |
| wh | bf16 | `logaddexp2_bw` | `default` | worst sampled pairing | 41.4 | 3.6 | 0.978 | — | 5868 | 2859 |
| wh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists; the rest reach 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 767 | 21882 |
| wh | bf16 | `logaddexp_bw` | `default` | worst sampled pairing | 62.5 | 4.9 | 0.979 | — | 4764 | 3521 |
| wh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 544 | 30814 |
| wh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 560 | 29963 |
| wh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 ±8% | 42437 |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 409 | 40995 |
| wh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 547 | 30648 |
| wh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 558 | 30063 |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 556 ±6% | 30163 |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395 | 64 | 1.67 | 0.985 | 0.395 | 877 | 19120 |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP | 1.65 | 0.649 | 0.961 | 0.996 | 6639 | 2527 |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP | 1.65 | 0.649 | 0.961 | 0.996 | 8263 | 2030 |
| wh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 533 | 31465 |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 547 | 30680 |
| wh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 | 42345 |
| wh | bf16 | `mac` | `default` | worst sampled pairing | 63.5 | 26.4 | 1 | — | 687 | 24418 |
| wh | bf16 | `max_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 5470 | 3067 |
| wh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 560 | 29969 |
| wh | bf16 | `min_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 5488 | 3057 |
| wh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 538 | 31197 |
| wh | bf16 | `mish` | `default` | 11/65024 defects; rest 1.49 ULP | 1.49 | 0.667 | 0.988 | 1.95e-38 | 589 ±8% | 28483 |
| wh | bf16 | `mse_loss` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 538 | 31212 |
| wh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1420 | 11812 |
| wh | bf16 | `multigammaln` | `default` | 11536/48328 defects; rest 724 ULP | 724 | 1.67 | 0.717 | 5.55e-17 | 12716 | 1319 |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 2.38e+06 | 293 | 0.249 | 5.55e-17 | 9578 | 1752 |
| wh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 534 | 31413 |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 550 ±6% | 30522 |
| wh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 535 | 31372 |
| wh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 548 | 30604 |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 ±8% | 42222 |
| wh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | 0.56 | — | 3114 | 5388 |
| wh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 | 41586 |
| wh | bf16 | `polygamma` | `k=1` | 7/49922 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.43e+05 | 0.956 | 0.996 | 4649 | 3608 |
| wh | bf16 | `polygamma` | `k=2` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 6.81e+05 | 0.945 | 4.47 | 5536 | 3030 |
| wh | bf16 | `polygamma` | `k=4` | 142/49922 defects; rest 3.22e+09 ULP | 3.22e+09 | 1.09e+07 | 0.936 | 3.48 | 5722 | 2932 |
| wh | bf16 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 6.81e+05 | 0.945 | 4.47 | 11837 | 1417 |
| wh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | 0.982 | — | 995 | 16857 |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2549 | 6581 |
| wh | bf16 | `prelu` | `weight=0.25` | 1/65024 defects; rest 0 ULP | 0 | 0 | 1 | 4.68e-38 | 395 ±5% | 42490 |
| wh | bf16 | `rad2deg` | `default` | faithful; 31760/63518 tie-breaks | 0.992 | 0.75 | 0.5 | 5.9e+36 | 398 ±6% | 42127 |
| wh | bf16 | `rdiv` | `value=2.0` | 258/64770 defects; rest 0.512 ULP | 0.512 | 0.507 | 0.984 | 8.47e+37 | 407 ±8% | 41201 |
| wh | bf16 | `rdiv_bw` | `scalar=2.0` | 256/48492 defects; rest 1.51 ULP | 1.51 | 0.835 | 0.617 | 7.67e-20 | 6919 | 2425 |
| wh | bf16 | `reciprocal` | `default` | 2/64514 defects; rest 0.512 ULP | 0.512 | 0.507 | 0.984 | 8.47e+37 | 406 ±7% | 41312 |
| wh | bf16 | `reciprocal_bw` | `default` | 256/48386 defects; rest 1.51 ULP | 1.51 | 0.835 | 0.617 | 5.42e-20 | 5119 | 3278 |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 | 42320 |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 405 | 41450 |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3949 | 4249 |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1295 | 12957 |
| wh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 | 42087 |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 | 41807 |
| wh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 | 42349 |
| wh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 | 42176 |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 ±6% | 42232 |
| wh | bf16 | `remainder` | `default` | worst pairing 6.65e+35 ULP; mean 1.06e+35 | 6.65e+35 | 1.06e+35 | 0.853 | — | 677 | 24769 |
| wh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 405 | 41429 |
| wh | bf16 | `rpow` | `exponent=0.5` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.975 | 3.39e+38 | 970 | 17304 |
| wh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 966 | 17366 |
| wh | bf16 | `rpow` | `exponent=2.0` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 967 | 17356 |
| wh | bf16 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 2928 | 5730 |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 430 ±15% | 39033 |
| wh | bf16 | `rsqrt_bw` | `default` | 75/21665 defects; rest 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 6323 | 2654 |
| wh | bf16 | `rsub` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 536 | 31302 |
| wh | bf16 | `rsub_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 549 | 30564 |
| wh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 471 ±8% | 35647 |
| wh | bf16 | `selu_bw` | `default` | 1/65024 defects; rest 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 3095 ±6% | 5420 |
| wh | bf16 | `sigmoid` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 449 ±10% | 37380 |
| wh | bf16 | `sigmoid_accurate` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 443 ±7% | 37875 |
| wh | bf16 | `sigmoid_bw` | `default` | 329/65024 defects; rest 266 ULP | 266 | 5.5 | 0.988 | 1.77 | 2247 | 7467 |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±6% | 41520 |
| wh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 ±5% | 42611 |
| wh | bf16 | `silu` | `default` | 13/65024 defects; rest 0.914 ULP | 0.914 | 0.585 | 0.993 | 2.33e-38 | 474 ±10% | 35365 |
| wh | bf16 | `silu_bw` | `default` | 9/65024 defects; rest 285 ULP | 285 | 1.31 | 0.979 | 0.867 | 3092 | 5425 |
| wh | bf16 | `sin` | `default` | faithful; 4/37354 tie-breaks | 0.552 | 0.526 | 1 | 9.99e+05 | 429 ±10% | 39120 |
| wh | bf16 | `sin_bw` | `default` | 21284/65024 defects; rest 2.98e+41 ULP | 2.98e+41 | 1.39e+39 | 0.857 | 1.07e+06 | 1334 | 12573 |
| wh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 475 ±14% | 35326 |
| wh | bf16 | `sinh_bw` | `default` | 2/33894 defects; rest 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 5856 | 2865 |
| wh | bf16 | `softplus` | `default` | 526/65024 defects; rest 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 453 ±9% | 37059 |
| wh | bf16 | `softplus_bw` | `default` | within 2 ULP | 1.91 | 0.746 | 0.982 | 3.39e+38 | 4350 | 3857 |
| wh | bf16 | `softshrink` | `default` | faithful; 3968/65024 tie-breaks | 1 | 0.948 | 0.941 | 3.39e+38 | 404 | 41530 |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2160 | 7766 |
| wh | bf16 | `softsign` | `default` | 510/65024 defects; rest 1 ULP | 1 | 0.641 | 0.907 | 8.51e+37 | 430 ±18% | 39041 |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391 | 81 | 3.29 | 0.859 | 0.00391 | 1329 | 12626 |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 434 ±9% | 38666 |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP | 1.14 | 0.695 | 0.719 | 3.39e+38 | 6628 | 2531 |
| wh | bf16 | `square` | `default` | faithful; 13970/48640 tie-breaks | 0.973 | 0.688 | 0.578 | 1.84e+19 | 396 | 42412 |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1285 | 13059 |
| wh | bf16 | `squared_difference` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 529 | 31699 |
| wh | bf16 | `squared_difference_` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 545 | 30774 |
| wh | bf16 | `squared_difference_bw` | `default` | faithful; 64842/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 2175 | 7714 |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | 0.997 | — | 562 ±13% | 29879 |
| wh | bf16 | `subtract` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 533 | 31480 |
| wh | bf16 | `subtract_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 546 | 30744 |
| wh | bf16 | `swish` | `default` | 13/65024 defects; rest 0.914 ULP | 0.914 | 0.585 | 0.993 | 2.33e-38 | 474 ±11% | 35388 |
| wh | bf16 | `tan` | `default` | faithful; 48/37354 tie-breaks | 0.551 | 0.5 | 0.999 | 9.99e+05 | 545 ±16% | 30799 |
| wh | bf16 | `tan_bw` | `default` | 12184/65024 defects; rest 3.57e+40 ULP | 3.57e+40 | 3.81e+37 | 0.691 | 1.13 | 2163 | 7756 |
| wh | bf16 | `tanh` | `default` | 2/65024 defects; rest 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 400 ±7% | 41940 |
| wh | bf16 | `tanh_bw` | `default` | within 2 ULP | 1.16 | 0.525 | 0.997 | 3.39e+38 | 1465 | 11454 |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08 | 63.5 | 9.38 | 0.981 | 1.35e-08 | 419 ±8% | 40010 |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09 | 63 | 3.52 | 0.937 | 7.45e-09 | 1687 | 9948 |
| wh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 ±7% | 41720 |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 400 ±6% | 41967 |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1664 | 10085 |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 ±6% | 41769 |
| wh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 684 | 24540 |
| wh | bf16 | `xielu` | `default` | 1/56847 defects; rest 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 986 | 17009 |
| wh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | 0.11 | — | 547 ±5% | 30689 |
| wh | bf16 | `xlogy_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.78 | 0.78 | 0.998 | — | 8897 | 1886 |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21733 |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2494 | 6727 |
| wh | fp32 | `acos` | `default` | within 2 ULP | 1.55 | 0.865 | 0.923 | 1 | 1094 | 15332 |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 14954 | 1122 |
| wh | fp32 | `acosh` | `default` | never within 2 ULP | 2.76 | 0.764 | 0.782 | — | 1378 ±14% | 12175 |
| wh | fp32 | `acosh_bw` | `default` | 15874/64514 defects; rest 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 17191 | 976 |
| wh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 983 | 17067 |
| wh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1007 | 16666 |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 988 | 16980 |
| wh | fp32 | `addcdiv` | `default` | worst sampled pairing | 2.19e+12 | 8.82e+07 | 0.901 | — | 1312 | 12786 |
| wh | fp32 | `addcmul` | `default` | worst sampled pairing | 2.13e+06 | 1.59e+04 | 0.985 | — | 1309 | 12815 |
| wh | fp32 | `asin` | `default` | within 2 ULP | 1.77 | 0.795 | 0.985 | 1 | 1052 | 15948 |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 14770 | 1136 |
| wh | fp32 | `asinh` | `default` | within 2 ULP | 1.72 | 0.774 | 0.881 | 3.39e+38 | 1785 ±6% | 9398 |
| wh | fp32 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 2605 | 6439 |
| wh | fp32 | `atan` | `default` | accurate to |x| <= 0.902 | 2.34 | 0.807 | 0.955 | 0.902 | 911 | 18426 |
| wh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | 0.882 | — | 1000 | 16770 |
| wh | fp32 | `atan2_bw` | `default` | worst sampled pairing | 1.64e+07 | 9.5e+04 | 0.61 | — | 10612 | 1581 |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 1.01 | 2.8 | 1.23 | 0.874 | 1.01 | 2564 | 6544 |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 4.28e-07 | 3.11 | 1.55 | 0.91 | 4.28e-07 | 1256 ±11% | 13356 |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.678 | 2.05e+03 | 1.55 | 0.891 | 0.678 | 16432 | 1021 |
| wh | fp32 | `bias_gelu` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | 0.746 | — | 987 | 16992 |
| wh | fp32 | `bias_gelu_` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | 0.746 | — | 1012 | 16579 |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38 | 2.55 | 1.76 | 0.521 | 2.26e-38 | 850 | 19734 |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 777 | 21605 |
| wh | fp32 | `celu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 903 | 18581 |
| wh | fp32 | `celu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5433 | 3088 |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21679 |
| wh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21711 |
| wh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21746 |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4630 | 3623 |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21741 |
| wh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21695 |
| wh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21744 |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4619 | 3632 |
| wh | fp32 | `cos` | `default` | accurate to |x| <= 92.4 | 1.34e+08 | 1.33e+05 | 0.927 | 92.4 | 849 | 19755 |
| wh | fp32 | `cos_bw` | `default` | 18643/65024 defects; rest 8.59e+51 ULP | 8.59e+51 | 3.23e+48 | 0.841 | 28 | 3306 | 5075 |
| wh | fp32 | `cosh` | `default` | within 2 ULP | 1.35 | 0.798 | 0.98 | 89.1 | 978 | 17153 |
| wh | fp32 | `cosh_bw` | `default` | 2/33894 defects; rest 2.21 ULP | 2.21 | 1.17 | 0.943 | 0.0155 | 13537 | 1239 |
| wh | fp32 | `deg2rad` | `default` | faithful; 63542/65024 tie-breaks | 0.63 | 0.595 | 0.906 | 3.4e+38 | 776 | 21633 |
| wh | fp32 | `digamma` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00566 | — | 2607 | 6437 |
| wh | fp32 | `digamma_bw` | `default` | 255/64769 defects; rest 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.429 | 5.4e-20 | 17305 | 970 |
| wh | fp32 | `div` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | 0.862 | — | 994 | 16871 |
| wh | fp32 | `div_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.84 | 0.84 | 0.902 | — | 6720 | 2497 |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP everywhere | 1.79 | 1.52 | 0.714 | — | 3469 | 4836 |
| wh | fp32 | `divide` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | 0.862 | — | 990 | 16938 |
| wh | fp32 | `divide_` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | 0.862 | — | 1022 | 16415 |
| wh | fp32 | `elu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 911 | 18426 |
| wh | fp32 | `elu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5430 | 3090 |
| wh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 991 | 16924 |
| wh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 1011 | 16600 |
| wh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21733 |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334 | 6.48 | 1.35 | 0.671 | 0.000334 | 1224 | 13710 |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 4885 | 3435 |
| wh | fp32 | `erfc` | `default` | 198/65024 unflushed; rest 2.1e+33 ULP | 2.1e+33 | 1.71e+29 | 0.33 | — | 1295 ±12% | 12958 |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 4889 | 3432 |
| wh | fp32 | `erfinv` | `default` | 29420/32256 defects; rest 7.96e+06 ULP | 7.96e+06 | 2.59e+05 | 0.000202 | 1.32e-38 | 1274 ±9% | 13170 |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456 | 4.98e+05 | 1.32e+03 | 0.914 | 0.000456 | 15308 | 1096 |
| wh | fp32 | `exp` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 901 | 18610 |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP | 3.79e+05 | 2e+05 | 0.309 | — | 771 | 21750 |
| wh | fp32 | `exp2` | `default` | faithful; 6044/49536 tie-breaks | 0.965 | 0.626 | 0.992 | 3.39e+38 | 844 | 19873 |
| wh | fp32 | `exp2_bw` | `default` | 1/49537 defects; rest 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 3320 | 5053 |
| wh | fp32 | `exp_bw` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 2614 ±54% | 6418 |
| wh | fp32 | `expm1` | `default` | faithful; 5257/49458 tie-breaks | 0.997 | 0.565 | 0.997 | 3.39e+38 | 996 | 16847 |
| wh | fp32 | `expm1_bw` | `default` | accurate to |x| <= 22 | 4.19e+06 | 3.83e+03 | 0.995 | 22 | 2608 | 6432 |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 777 | 21594 |
| wh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.77e+03 | 4.19e+06 | 1.77e+03 | 0.955 | — | 4771 | 3517 |
| wh | fp32 | `fmod` | `default` | worst pairing 1.43e+45 ULP; mean 1.86e+41 | 1.43e+45 | 1.86e+41 | 0.762 | — | 1016 | 16519 |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21716 |
| wh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 984 | 17053 |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 1010 | 16608 |
| wh | fp32 | `gelu` | `default` | 83/65024 defects; rest 1.51e+08 ULP | 1.51e+08 | 1.55e+05 | 0.963 | 0.253 | 1329 ±16% | 12627 |
| wh | fp32 | `gelu` | `fast_approx` | 197/65024 defects; rest 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 771 | 21753 |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0298 | 1.47e+08 | 2.3e+04 | 0.929 | 0.0298 | 2071 | 8100 |
| wh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21715 |
| wh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 991 | 16934 |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1006 | 16669 |
| wh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21791 |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21727 |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21730 |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3245 | 5171 |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25 | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 776 | 21628 |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 5061 | 3315 |
| wh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36 | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 810 | 20722 |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133 | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 7088 | 2367 |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 ±7% | 21771 |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4275 | 3925 |
| wh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21760 |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21689 |
| wh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21677 |
| wh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists; the rest reach 3.44e+06 ULP | 3.44e+06 | 3.28e+04 | 0.965 | — | 997 | 16830 |
| wh | fp32 | `hypot_bw` | `default` | 16384/65024 defects; rest 4.86e+06 ULP | 4.86e+06 | 4.48e+04 | 0.824 | — | 6364 | 2636 |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.45 | 1.68e+07 | 2.19e+06 | 0.959 | 2.45 | 873 | 19226 |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00626 | 1.56e+07 | 4.11e+04 | 0.929 | 0.00626 | 3425 | 4898 |
| wh | fp32 | `i1` | `default` | accurate to |x| <= 0.00626 | 8.02 | 0.722 | 0.483 | 0.00626 | 1714 | 9789 |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21758 |
| wh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 993 | 16903 |
| wh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 778 | 21555 |
| wh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21820 |
| wh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21721 |
| wh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21706 |
| wh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21679 |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 987 | 16997 |
| wh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | 0.953 | — | 993 | 16903 |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | 0.953 | — | 1011 | 16595 |
| wh | fp32 | `ldexp_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.742 | 0.742 | 0.996 | — | 5351 | 3135 |
| wh | fp32 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 987 | 16994 |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 1008 | 16650 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21678 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithful; 31672/65024 tie-breaks | 0.84 | 0.747 | 0.868 | 3.39e+38 | 776 | 21634 |
| wh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21791 |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 3592 | 4671 |
| wh | fp32 | `lerp` | `default` | worst sampled pairing | 1.46e+08 | 9.53e+04 | 0.962 | — | 1312 | 12784 |
| wh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 5413 | 3100 |
| wh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21673 |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181 | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 3778 | 4441 |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00566 | — | 4319 | 3884 |
| wh | fp32 | `log` | `default` | faithful; 32512/32512 tie-breaks | 0.961 | 0.562 | 0.952 | 3.39e+38 | 873 | 19220 |
| wh | fp32 | `log10` | `default` | accurate to |x| <= 0.336 | 2.13 | 1.36 | 0.641 | 0.336 | 879 | 19091 |
| wh | fp32 | `log10_bw` | `default` | accurate to |x| <= 2.04e-38 | 2.1 | 1.41 | 0.674 | 2.04e-38 | 10269 | 1634 |
| wh | fp32 | `log1p` | `default` | faithful; 20111/48640 tie-breaks | 0.967 | 0.558 | 0.982 | 3.39e+38 | 883 | 18995 |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP | 1.89 | 0.769 | 0.913 | 8.51e+37 | 10292 | 1630 |
| wh | fp32 | `log2` | `default` | accurate to |x| <= 0.704 | 2.43 | 0.517 | 0.993 | 0.704 | 889 | 18873 |
| wh | fp32 | `log2_bw` | `default` | 112/64626 defects; rest 1.9 ULP | 1.9 | 1.3 | 0.701 | — | 10272 ±13% | 1633 |
| wh | fp32 | `log_bw` | `default` | faithful; 64512/64514 tie-breaks | 0.892 | 0.714 | 0.902 | 8.51e+37 | 7790 | 2154 |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP | 4.4e+05 | 1.82e+04 | 0.481 | — | 1110 | 15109 |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.00164 | 2.95 | 1.28 | 0.954 | 0.00164 | 11559 | 1452 |
| wh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists; the rest reach 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 1484 | 11303 |
| wh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists; the rest reach 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.716 | — | 1301 | 12895 |
| wh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists; the rest reach 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.716 | — | 1303 | 12878 |
| wh | fp32 | `logaddexp2_bw` | `default` | worst sampled pairing | 45.7 | 6.68 | 0.956 | — | 11431 | 1468 |
| wh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists; the rest reach 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 1477 | 11357 |
| wh | fp32 | `logaddexp_bw` | `default` | worst sampled pairing | 65.7 | 8.66 | 0.962 | — | 9455 | 1774 |
| wh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 993 | 16901 |
| wh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 1018 | 16486 |
| wh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21683 |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 776 | 21630 |
| wh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 993 | 16889 |
| wh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 1014 | 16538 |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 1020 | 16445 |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.266 | 4.19e+06 | 261 | 0.94 | 0.266 | 1216 ±10% | 13796 |
| wh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.98e-08 | 2.36 | 0.839 | 0.857 | 2.98e-08 | 13094 | 1281 |
| wh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.98e-08 | 2.36 | 0.839 | 0.857 | 2.98e-08 | 16437 | 1021 |
| wh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 988 | 16985 |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1011 | 16595 |
| wh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21716 |
| wh | fp32 | `mac` | `default` | worst sampled pairing | 4.22e+06 | 7.31e+05 | 0.974 | — | 1310 | 12807 |
| wh | fp32 | `max_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 10507 | 1597 |
| wh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 1014 | 16549 |
| wh | fp32 | `min_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 10524 | 1594 |
| wh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 1013 ±7% | 16556 |
| wh | fp32 | `mish` | `default` | 9/65024 defects; rest 7 ULP | 7 | 2.49 | 0.954 | 8.61e-06 | 1208 ±7% | 13889 |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 983 | 17059 |
| wh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2694 | 6228 |
| wh | fp32 | `multigammaln` | `default` | 7422/50376 defects; rest 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.59e-17 | 20988 | 799 |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 1.36e+15 | 2.04e+11 | 0.0104 | 5.55e-17 | 20161 | 832 |
| wh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 984 | 17049 |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1007 | 16668 |
| wh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 986 | 17008 |
| wh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 1006 | 16673 |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21691 |
| wh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | 0.5 | — | 6071 | 2763 |
| wh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21760 |
| wh | fp32 | `polygamma` | `k=1` | accurate to |x| <= 5.4e-20 | 7.91e+33 | 4.68e+29 | 0.617 | 5.4e-20 | 5558 | 3019 |
| wh | fp32 | `polygamma` | `k=2` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.199 | 1.79e-13 | 5970 | 2810 |
| wh | fp32 | `polygamma` | `k=4` | 141/49922 defects; rest 3.77e+25 ULP | 3.77e+25 | 6.51e+21 | 0.191 | 3.68e-08 | 6316 | 2656 |
| wh | fp32 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.199 | 1.79e-13 | 17992 | 932 |
| wh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.17 | 5.92e+03 | 3.17 | 0.992 | — | 2022 | 8299 |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 4972 | 3374 |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21732 |
| wh | fp32 | `rad2deg` | `default` | faithful; 63518/63518 tie-breaks | 0.696 | 0.643 | 0.858 | 5.93e+36 | 772 | 21740 |
| wh | fp32 | `rdiv` | `value=2.0` | 256/64770 defects; rest 0.892 ULP | 0.892 | 0.714 | 0.902 | 8.51e+37 | 803 | 20889 |
| wh | fp32 | `rdiv_bw` | `scalar=2.0` | 252/48490 defects; rest 1.84 ULP | 1.84 | 1.22 | 0.726 | 7.67e-20 | 13398 | 1252 |
| wh | fp32 | `reciprocal` | `default` | faithful; 64512/64514 tie-breaks | 0.892 | 0.714 | 0.902 | 8.51e+37 | 789 | 21266 |
| wh | fp32 | `reciprocal_bw` | `default` | 254/48386 defects; rest 1.84 ULP | 1.84 | 1.22 | 0.726 | 5.42e-20 | 10034 | 1672 |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 775 | 21658 |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21720 |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 7897 | 2125 |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2483 | 6756 |
| wh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21714 |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21851 |
| wh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21680 |
| wh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21767 |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21809 |
| wh | fp32 | `remainder` | `default` | worst pairing 1.43e+45 ULP; mean 1.68e+41 | 1.43e+45 | 1.68e+41 | 0.761 | — | 997 | 16827 |
| wh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21686 |
| wh | fp32 | `rpow` | `exponent=0.5` | faithful; 5624/49536 tie-breaks | 0.879 | 0.609 | 0.995 | 3.39e+38 | 1892 | 8869 |
| wh | fp32 | `rpow` | `exponent=1.0` | 1536/49536 defects; rest 0 ULP | 0 | 0 | 1 | 8.28e+34 | 1888 | 8886 |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536/49536 defects; rest 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 1889 | 8881 |
| wh | fp32 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 5720 | 2933 |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP | 1.12 | 0.834 | 0.864 | 3.39e+38 | 843 | 19908 |
| wh | fp32 | `rsqrt_bw` | `default` | 75/21666 defects; rest 6.76 ULP | 6.76 | 4.26 | 0.3 | — | 11977 | 1401 |
| wh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 981 | 17104 |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1009 | 16629 |
| wh | fp32 | `selu` | `default` | never within 2 ULP | 51.3 | 26.3 | 0 | — | 936 | 17921 |
| wh | fp32 | `selu_bw` | `default` | 1/65024 defects; rest 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 6193 | 2709 |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 1070 | 15674 |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 1070 | 15674 |
| wh | fp32 | `sigmoid_bw` | `default` | 140/65024 defects; rest 8.39e+06 ULP | 8.39e+06 | 2.71e+04 | 0.975 | 0.447 | 4560 | 3679 |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 | 21151 |
| wh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 797 | 21054 |
| wh | fp32 | `silu` | `default` | 9/65024 defects; rest 2.94 ULP | 2.94 | 1.55 | 0.95 | 0.000121 | 1096 | 15312 |
| wh | fp32 | `silu_bw` | `default` | 9/65024 defects; rest 5.72e+06 ULP | 5.72e+06 | 841 | 0.937 | 2.98e-07 | 6264 | 2678 |
| wh | fp32 | `sin` | `default` | accurate to |x| <= 28 | 3.36e+07 | 3.07e+04 | 0.96 | 28 | 832 | 20166 |
| wh | fp32 | `sin_bw` | `default` | 18430/65024 defects; rest 1.24e+52 ULP | 1.24e+52 | 3.44e+48 | 0.808 | 92.4 | 2565 | 6540 |
| wh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155 | 2.21 | 1.17 | 0.943 | 0.0155 | 1067 | 15719 |
| wh | fp32 | `sinh_bw` | `default` | 2/33894 defects; rest 1.35 ULP | 1.35 | 0.798 | 0.98 | 88.5 | 11921 | 1407 |
| wh | fp32 | `softplus` | `default` | 5/65024 defects; rest 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 1307 ±13% | 12837 |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.00163 | 2.95 | 1.34 | 0.954 | 0.00163 | 8709 | 1926 |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 767 | 21860 |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4276 | 3924 |
| wh | fp32 | `softsign` | `default` | 510/65024 defects; rest 2.66 ULP | 2.66 | 0.897 | 0.92 | 0.000462 | 816 | 20557 |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07 | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 2582 | 6498 |
| wh | fp32 | `sqrt` | `default` | faithful; 32512/32512 tie-breaks | 0.867 | 0.827 | 0.869 | 3.39e+38 | 825 | 20331 |
| wh | fp32 | `sqrt_bw` | `default` | never within 2 ULP | 2.24 | 1.44 | 0.702 | — | 12736 | 1317 |
| wh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 773 | 21705 |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2504 | 6700 |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 985 | 17030 |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 1009 | 16622 |
| wh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 4228 | 3968 |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 985 | 17039 |
| wh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 980 | 17113 |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1005 | 16688 |
| wh | fp32 | `swish` | `default` | 9/65024 defects; rest 2.94 ULP | 2.94 | 1.55 | 0.95 | 0.000121 | 1090 | 15392 |
| wh | fp32 | `tan` | `default` | accurate to |x| <= 3.92 | 2.26e+08 | 2.32e+05 | 0.944 | 3.92 | 1195 ±7% | 14035 |
| wh | fp32 | `tan_bw` | `default` | 20623/65024 defects; rest 2.85e+45 ULP | 2.85e+45 | 4.74e+44 | 0.87 | 0.882 | 4436 | 3782 |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000439 | 2.79 | 1.49 | 0.979 | 0.000439 | 918 | 18270 |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP | 6.59e+04 | 7.51e+03 | 0.482 | — | 1767 | 9497 |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08 | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 1288 ±11% | 13030 |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09 | 4.19e+06 | 1.01e+05 | 0.901 | 7.42e-09 | 3383 | 4959 |
| wh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21844 |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 776 | 21632 |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3261 | 5145 |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21754 |
| wh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 1311 | 12800 |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13 | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 1304 ±20% | 12870 |
| wh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | 1.15e-05 | — | 992 | 16916 |
| wh | fp32 | `xlogy_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.766 | 0.766 | 0.951 | — | 17513 | 958 |
