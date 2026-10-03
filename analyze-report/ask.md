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
| bh | bf16 | `de546d3b146` | 0.79.0 | 956d60646612 |
| bh | fp32 | `de546d3b146` | 0.79.0 | 956d60646612 |
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
| bh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88770 |
| bh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 593 | 28291 |
| bh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 270 | 62076 |
| bh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.744 | 0.996 | 0.949 | 3520 | 4766 |
| bh | bf16 | `acosh` | `default` | within 2 ULP | 1.41 | 0.596 | 0.973 | 3.39e+38 | 317 ±10% | 52911 |
| bh | bf16 | `acosh_bw` | `default` | 15874/64514 defects; rest 3.26 ULP | 3.26 | 0.688 | 0.781 | 1.03 | 4023 | 4170 |
| bh | bf16 | `add` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 246 | 68218 |
| bh | bf16 | `add_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 242 | 69412 |
| bh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | 0.997 | — | 267 | 62884 |
| bh | bf16 | `addcdiv` | `default` | worst sampled pairing | 2.11e+06 | 2.35e+03 | 0.999 | — | 356 ±9% | 47130 |
| bh | bf16 | `addcmul` | `default` | worst sampled pairing | 126 | 19.4 | 1 | — | 344 | 48840 |
| bh | bf16 | `asin` | `default` | 2/32258 defects; rest 0.499 ULP | 0.499 | 0 | 1 | — | 267 | 62830 |
| bh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938 | 2.76 | 0.787 | 0.992 | 0.938 | 3442 | 4874 |
| bh | bf16 | `asinh` | `default` | within 2 ULP | 1.32 | 0.615 | 0.991 | 3.39e+38 | 439 | 38176 |
| bh | bf16 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.33 ULP | 1.33 | 0.681 | 0.918 | 1.84e+19 | 595 | 28177 |
| bh | bf16 | `atan` | `default` | 2/65024 defects; rest 0.527 ULP | 0.527 | 0.513 | 1 | — | 203 | 82457 |
| bh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 3.56 | 200 | 3.56 | 0.901 | — | 276 ±6% | 60798 |
| bh | bf16 | `atan2_bw` | `default` | worst sampled pairing | 248 | 4.89 | 0.508 | — | 2512 | 6678 |
| bh | bf16 | `atan_bw` | `default` | 2/48386 defects; rest 2.23 ULP | 2.23 | 0.867 | 0.849 | 0.23 | 616 | 27234 |
| bh | bf16 | `atanh` | `default` | 2/32256 defects; rest 2.08 ULP | 2.08 | 0.995 | 0.175 | — | 255 | 65812 |
| bh | bf16 | `atanh_bw` | `default` | 2/48386 defects; rest 8.17 ULP | 8.17 | 1 | 0.79 | 0.82 | 3853 | 4355 |
| bh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | 0.749 | — | 255 ±12% | 65793 |
| bh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | 0.749 | — | 241 | 69510 |
| bh | bf16 | `cbrt` | `default` | faithful; 846/65024 tie-breaks | 0.507 | 0.502 | 0.987 | 3.39e+38 | 214 | 78494 |
| bh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 ±17% | 86460 |
| bh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 182 | 92223 |
| bh | bf16 | `celu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1262 | 13292 |
| bh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 ±5% | 85982 |
| bh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 202 ±15% | 83250 |
| bh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±11% | 87505 |
| bh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1093 | 15353 |
| bh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 ±10% | 88620 |
| bh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88236 |
| bh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±12% | 88310 |
| bh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1093 | 15354 |
| bh | bf16 | `cos` | `default` | faithful; 2/37354 tie-breaks | 0.552 | 0.552 | 1 | 9.99e+05 | 242 | 69195 |
| bh | bf16 | `cos_bw` | `default` | 21281/65024 defects; rest 2.09e+42 ULP | 2.09e+42 | 2.59e+39 | 0.867 | 2.15e+06 | 805 | 20849 |
| bh | bf16 | `cosh` | `default` | faithful; 4/33894 tie-breaks | 0.501 | 0.501 | 1 | 89 | 257 ±5% | 65356 |
| bh | bf16 | `cosh_bw` | `default` | 2/33894 defects; rest 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 3061 | 5482 |
| bh | bf16 | `deg2rad` | `default` | 2/65024 defects; rest 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 191 | 87895 |
| bh | bf16 | `digamma` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 373 | 44926 |
| bh | bf16 | `digamma_bw` | `default` | 263/64769 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.36e+04 | 0.674 | 0.996 | 3155 | 5317 |
| bh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 271 ±8% | 61902 |
| bh | bf16 | `div_bw` | `default` | bit-exact | 0.498 | 0 | 1 | — | 1570 | 10689 |
| bh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 653 | 25711 |
| bh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 252 | 66538 |
| bh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 251 | 66964 |
| bh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 195 | 86204 |
| bh | bf16 | `elu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1269 | 13225 |
| bh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 283 ±43% | 59348 |
| bh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 241 | 69542 |
| bh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89106 |
| bh | bf16 | `erf` | `default` | within 2 ULP | 1.06 | 0.771 | 0.729 | 3.39e+38 | 190 ±30% | 88276 |
| bh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 1138 | 14748 |
| bh | bf16 | `erfc` | `default` | 198/65024 unflushed; rest 3.2e+28 ULP | 3.2e+28 | 3.23e+24 | 0.949 | 2.5 | 309 | 54321 |
| bh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5 | 112 | 4.97 | 0.976 | 1.5 | 1138 | 14744 |
| bh | bf16 | `erfinv` | `default` | 29424/32256 defects; rest 121 ULP | 121 | 6.34 | 0.391 | 1.31e-38 | 325 | 51602 |
| bh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777 | 8.91 | 1.28 | 0.98 | 0.777 | 3578 | 4689 |
| bh | bf16 | `exp` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 193 | 86954 |
| bh | bf16 | `exp` | `fast_approx` | never within 2 ULP | 6.08 | 3.05 | 0.314 | — | 187 | 89669 |
| bh | bf16 | `exp2` | `default` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 227 | 73951 |
| bh | bf16 | `exp2_bw` | `default` | 1/49537 defects; rest 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 816 | 20559 |
| bh | bf16 | `exp_bw` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 598 | 28074 |
| bh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 263 | 63799 |
| bh | bf16 | `expm1_bw` | `default` | accurate to |x| <= 33.5 | 34 | 0.695 | 0.984 | 33.5 | 595 | 28174 |
| bh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85636 |
| bh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | 0.996 | — | 1155 | 14520 |
| bh | bf16 | `fmod` | `default` | worst pairing 9.14e+35 ULP; mean 8.94e+34 | 9.14e+35 | 8.94e+34 | 0.673 | — | 259 | 64690 |
| bh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±17% | 87299 |
| bh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 246 | 68155 |
| bh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 242 | 69346 |
| bh | bf16 | `gelu` | `default` | 86/65024 defects; rest 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 292 | 57437 |
| bh | bf16 | `gelu` | `fast_approx` | 198/65024 defects; rest 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 188 | 89215 |
| bh | bf16 | `gelu_bw` | `default` | faithful; 38/65024 tie-breaks | 0.939 | 0.681 | 0.999 | 3.39e+38 | 669 | 25093 |
| bh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 203 | 82777 |
| bh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 266 ±5% | 63131 |
| bh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 243 | 69068 |
| bh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88868 |
| bh | bf16 | `hardmish` | `default` | faithful; 2587/65024 tie-breaks | 1 | 0.913 | 0.961 | 3.39e+38 | 190 | 88215 |
| bh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88493 |
| bh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21888 |
| bh | bf16 | `hardsigmoid` | `default` | within 2 ULP | 1 | 0.557 | 0.986 | 3.39e+38 | 189 | 88848 |
| bh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 1169 | 14349 |
| bh | bf16 | `hardswish` | `default` | 2/65024 defects; rest 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 191 | 88057 |
| bh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.13 | 85.3 | 2.1 | 0.993 | 1.13 | 1642 | 10219 |
| bh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±12% | 88111 |
| bh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 977 | 17170 |
| bh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88586 |
| bh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 | 86338 |
| bh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88786 |
| bh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists; the rest reach 52.7 ULP | 52.7 | 1.58 | 0.947 | — | 255 | 65817 |
| bh | bf16 | `hypot_bw` | `default` | worst sampled pairing | 74.6 | 2.43 | 0.835 | — | 1506 | 11138 |
| bh | bf16 | `i0` | `default` | accurate to |x| <= 13.6 | 255 | 56.3 | 0.952 | 13.6 | 247 ±6% | 67819 |
| bh | bf16 | `i0_bw` | `default` | 4/33904 defects; rest 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 826 | 20307 |
| bh | bf16 | `i1` | `default` | 4/65024 defects; rest 0.858 ULP | 0.858 | 0.589 | 0.516 | 2.33e-38 | 421 | 39853 |
| bh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 ±13% | 89339 |
| bh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 247 | 67836 |
| bh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±5% | 88425 |
| bh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88836 |
| bh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88452 |
| bh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85758 |
| bh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±6% | 88520 |
| bh | bf16 | `l1_loss` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 266 | 63121 |
| bh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | 0.955 | — | 274 ±6% | 61335 |
| bh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | 0.955 | — | 247 | 67982 |
| bh | bf16 | `ldexp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.898 | 0.898 | 0.983 | — | 1192 | 14071 |
| bh | bf16 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 246 | 68237 |
| bh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 242 | 69437 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88643 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithful; 15341/65024 tie-breaks | 0.96 | 0.741 | 0.761 | 3.39e+38 | 190 | 88244 |
| bh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 | 86798 |
| bh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 861 | 19477 |
| bh | bf16 | `lerp` | `default` | worst sampled pairing | 1.77e+03 | 39.9 | 1 | — | 358 ±6% | 46898 |
| bh | bf16 | `lerp_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.506 | 0.506 | 0.992 | — | 1270 | 13207 |
| bh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88898 |
| bh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.412 | 324 | 0.846 | 0.76 | 0.412 | 847 | 19803 |
| bh | bf16 | `lgamma_bw` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 775 | 21646 |
| bh | bf16 | `log` | `default` | faithful; 63/32512 tie-breaks | 0.78 | 0.534 | 0.998 | 3.39e+38 | 231 | 72745 |
| bh | bf16 | `log10` | `default` | faithful; 58/32512 tie-breaks | 0.785 | 0.543 | 0.998 | 3.39e+38 | 240 | 69793 |
| bh | bf16 | `log10_bw` | `default` | within 2 ULP | 1.74 | 0.803 | 0.508 | 3.69e+37 | 2451 | 6846 |
| bh | bf16 | `log1p` | `default` | faithful; 109/48640 tie-breaks | 0.931 | 0.57 | 0.998 | 3.39e+38 | 240 | 69841 |
| bh | bf16 | `log1p_bw` | `default` | 2/64514 defects; rest 1.42 ULP | 1.42 | 0.744 | 0.982 | 8.47e+37 | 2406 | 6973 |
| bh | bf16 | `log2` | `default` | faithful; 59/32512 tie-breaks | 0.766 | 0.542 | 0.998 | 3.39e+38 | 251 | 66754 |
| bh | bf16 | `log2_bw` | `default` | 116/64626 defects; rest 1.51 ULP | 1.51 | 0.847 | 0.57 | — | 2436 | 6886 |
| bh | bf16 | `log_bw` | `default` | 2/64514 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 1854 | 9048 |
| bh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.428 | 6.39 | 1.38 | 0.958 | 0.428 | 220 | 76316 |
| bh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP | 1.71 | 0.722 | 0.976 | 3.39e+38 | 2675 | 6273 |
| bh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists; the rest reach 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 315 | 53343 |
| bh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists; the rest reach 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 406 | 41323 |
| bh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists; the rest reach 3.03e+06 ULP | 3.03e+06 | 4.08e+03 | 0.871 | — | 385 | 43600 |
| bh | bf16 | `logaddexp2_bw` | `default` | worst sampled pairing | 41.4 | 3.6 | 0.978 | — | 2505 | 6698 |
| bh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists; the rest reach 8.39e+06 ULP | 8.39e+06 | 4.85e+03 | 0.858 | — | 300 | 55954 |
| bh | bf16 | `logaddexp_bw` | `default` | worst sampled pairing | 62.5 | 4.9 | 0.979 | — | 2139 | 7844 |
| bh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 250 | 67120 |
| bh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 248 | 67708 |
| bh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88864 |
| bh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 186 | 90240 |
| bh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 252 | 66547 |
| bh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 246 | 68174 |
| bh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 268 | 62638 |
| bh | bf16 | `logit` | `default` | accurate to |x| <= 0.395 | 64 | 1.67 | 0.985 | 0.395 | 324 | 51742 |
| bh | bf16 | `logit_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3040 | 5518 |
| bh | bf16 | `logiteps_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3825 | 4386 |
| bh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 243 | 69003 |
| bh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 242 | 69222 |
| bh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88262 |
| bh | bf16 | `mac` | `default` | worst sampled pairing | 63.5 | 26.4 | 1 | — | 334 | 50251 |
| bh | bf16 | `max_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 2480 | 6765 |
| bh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 244 | 68631 |
| bh | bf16 | `min_bw` | `default` | worst sampled pairing | 64 | 64 | 1 | — | 2485 | 6750 |
| bh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 264 ±9% | 63616 |
| bh | bf16 | `mish` | `default` | 11/65024 defects; rest 1.49 ULP | 1.49 | 0.666 | 0.988 | 1.95e-38 | 238 | 70455 |
| bh | bf16 | `mse_loss` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 267 | 62754 |
| bh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 641 ±49% | 26157 |
| bh | bf16 | `multigammaln` | `default` | 11536/48328 defects; rest 724 ULP | 724 | 1.67 | 0.717 | 5.59e-17 | 4683 | 3583 |
| bh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.59e-17 | 2.38e+06 | 293 | 0.249 | 5.59e-17 | 3723 | 4507 |
| bh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 250 ±6% | 67015 |
| bh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 243 | 68976 |
| bh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 267 | 62759 |
| bh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 242 | 69343 |
| bh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88345 |
| bh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | 0.56 | — | 1428 | 11748 |
| bh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88998 |
| bh | bf16 | `polygamma` | `k=1` | 7/49922 defects; rest 1.02e+08 ULP | 1.02e+08 | 2.13e+05 | 0.97 | 0.996 | 326 | 51465 |
| bh | bf16 | `polygamma` | `k=2` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 375 | 44771 |
| bh | bf16 | `polygamma` | `k=4` | 142/49922 defects; rest 3.22e+09 ULP | 3.22e+09 | 1.66e+07 | 0.958 | 3.5 | 423 | 39662 |
| bh | bf16 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 3203 | 5239 |
| bh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | 0.982 | — | 414 | 40484 |
| bh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1201 | 13968 |
| bh | bf16 | `prelu` | `weight=0.25` | 1/65024 defects; rest 0 ULP | 0 | 0 | 1 | 4.68e-38 | 193 ±6% | 86933 |
| bh | bf16 | `rad2deg` | `default` | faithful; 31760/63518 tie-breaks | 0.992 | 0.75 | 0.5 | 5.9e+36 | 188 | 89179 |
| bh | bf16 | `rdiv` | `value=2.0` | 258/64770 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 189 | 88606 |
| bh | bf16 | `rdiv_bw` | `scalar=2.0` | 256/48492 defects; rest 1.51 ULP | 1.51 | 0.837 | 0.602 | 7.67e-20 | 3146 | 5332 |
| bh | bf16 | `reciprocal` | `default` | 2/64514 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 190 | 88428 |
| bh | bf16 | `reciprocal_bw` | `default` | 256/48386 defects; rest 1.51 ULP | 1.51 | 0.837 | 0.602 | 5.42e-20 | 2378 | 7056 |
| bh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±5% | 87353 |
| bh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88643 |
| bh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1817 | 9231 |
| bh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 595 | 28195 |
| bh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 ±14% | 86789 |
| bh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88655 |
| bh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±9% | 88173 |
| bh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88565 |
| bh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88640 |
| bh | bf16 | `remainder` | `default` | worst pairing 1.08e+36 ULP; mean 1.22e+35 | 1.08e+36 | 1.22e+35 | 0.672 | — | 293 ±12% | 57304 |
| bh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 | 86636 |
| bh | bf16 | `rpow` | `exponent=0.5` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.975 | 3.39e+38 | 377 | 44501 |
| bh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 372 | 45045 |
| bh | bf16 | `rpow` | `exponent=2.0` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 376 | 44564 |
| bh | bf16 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 1351 | 12422 |
| bh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 243 | 69068 |
| bh | bf16 | `rsqrt_bw` | `default` | 75/21665 defects; rest 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 2824 ±126% | 5940 |
| bh | bf16 | `rsub` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 243 | 69127 |
| bh | bf16 | `rsub_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 242 | 69366 |
| bh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 193 | 87059 |
| bh | bf16 | `selu_bw` | `default` | 1/65024 defects; rest 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 1456 | 11520 |
| bh | bf16 | `sigmoid` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 262 | 63930 |
| bh | bf16 | `sigmoid_accurate` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 261 | 64306 |
| bh | bf16 | `sigmoid_bw` | `default` | 331/65024 defects; rest 125 ULP | 125 | 4.81 | 0.988 | 1.76 | 1053 | 15927 |
| bh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89082 |
| bh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88940 |
| bh | bf16 | `silu` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 215 ±7% | 78153 |
| bh | bf16 | `silu_bw` | `default` | 9/65024 defects; rest 285 ULP | 285 | 1.3 | 0.979 | 0.863 | 1446 | 11602 |
| bh | bf16 | `sin` | `default` | faithful; 4/37354 tie-breaks | 0.552 | 0.526 | 1 | 9.99e+05 | 220 | 76424 |
| bh | bf16 | `sin_bw` | `default` | 21284/65024 defects; rest 2.98e+41 ULP | 2.98e+41 | 1.39e+39 | 0.857 | 1.07e+06 | 660 ±5% | 25425 |
| bh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 171 ±17% | 97932 |
| bh | bf16 | `sinh_bw` | `default` | 2/33894 defects; rest 0.501 ULP | 0.501 | 0.501 | 1 | 88.5 | 2788 | 6018 |
| bh | bf16 | `softcap` | `beta=50.0` | 1426/65024 defects; rest 0.718 ULP | 0.718 | 0.574 | 0.997 | — | 230 ±13% | 72904 |
| bh | bf16 | `softplus` | `default` | 526/65024 defects; rest 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 165 ±6% | 101668 |
| bh | bf16 | `softplus_bw` | `default` | within 2 ULP | 1.71 | 0.678 | 0.983 | 3.39e+38 | 2019 | 8309 |
| bh | bf16 | `softshrink` | `default` | faithful; 3968/65024 tie-breaks | 1 | 0.948 | 0.941 | 3.39e+38 | 189 | 88941 |
| bh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 982 | 17085 |
| bh | bf16 | `softsign` | `default` | 512/65024 defects; rest 1 ULP | 1 | 0.605 | 0.907 | 8.47e+37 | 191 | 87949 |
| bh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391 | 81 | 3.24 | 0.856 | 0.00391 | 628 | 26719 |
| bh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 223 ±15% | 75213 |
| bh | bf16 | `sqrt_bw` | `default` | within 2 ULP | 1.14 | 0.699 | 0.723 | 3.39e+38 | 3023 | 5550 |
| bh | bf16 | `square` | `default` | faithful; 13970/48640 tie-breaks | 0.973 | 0.688 | 0.578 | 1.84e+19 | 185 ±18% | 90591 |
| bh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 600 | 27959 |
| bh | bf16 | `squared_difference` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 242 ±22% | 69381 |
| bh | bf16 | `squared_difference_` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 245 ±14% | 68346 |
| bh | bf16 | `squared_difference_bw` | `default` | faithful; 64842/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 1022 ±8% | 16410 |
| bh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | 0.997 | — | 265 | 63349 |
| bh | bf16 | `subtract` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 241 | 69477 |
| bh | bf16 | `subtract_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 240 | 69801 |
| bh | bf16 | `swish` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 215 ±6% | 77941 |
| bh | bf16 | `tan` | `default` | faithful; 30/37354 tie-breaks | 0.551 | 0.509 | 0.999 | 9.99e+05 | 259 | 64855 |
| bh | bf16 | `tan_bw` | `default` | 22782/65024 defects; rest 3.57e+40 ULP | 3.57e+40 | 1.01e+38 | 0.855 | 1.13 | 1005 | 16700 |
| bh | bf16 | `tanh` | `default` | 2/65024 defects; rest 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 189 | 88641 |
| bh | bf16 | `tanh_bw` | `default` | within 2 ULP | 1.16 | 0.525 | 0.997 | 3.39e+38 | 612 | 27394 |
| bh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08 | 63.5 | 9.38 | 0.981 | 1.35e-08 | 218 ±10% | 76849 |
| bh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09 | 63 | 3.52 | 0.937 | 7.45e-09 | 769 | 21813 |
| bh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88911 |
| bh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±5% | 88452 |
| bh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 758 | 22134 |
| bh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88486 |
| bh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 327 | 51343 |
| bh | bf16 | `xielu` | `default` | 1/56847 defects; rest 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 399 | 42027 |
| bh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | 0.11 | — | 275 | 61079 |
| bh | bf16 | `xlogy_bw` | `default` | faithful; 65026/65026 tie-breaks | 0.78 | 0.78 | 0.998 | — | 4104 | 4088 |
| bh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48172 |
| bh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1146 | 14636 |
| bh | fp32 | `acos` | `default` | within 2 ULP | 1.55 | 0.865 | 0.923 | 1 | 469 ±6% | 35739 |
| bh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 6842 | 2452 |
| bh | fp32 | `acosh` | `default` | never within 2 ULP | 2.47 | 0.76 | 0.782 | — | 395 | 42525 |
| bh | fp32 | `acosh_bw` | `default` | 15874/64514 defects; rest 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 7811 | 2148 |
| bh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 475 | 35311 |
| bh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 473 | 35489 |
| bh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 501 ±6% | 33457 |
| bh | fp32 | `addcdiv` | `default` | worst sampled pairing | 2.19e+12 | 8.82e+07 | 0.852 | — | 667 | 25138 |
| bh | fp32 | `addcmul` | `default` | worst sampled pairing | 2.13e+06 | 1.59e+04 | 0.985 | — | 667 | 25161 |
| bh | fp32 | `asin` | `default` | within 2 ULP | 1.77 | 0.795 | 0.985 | 1 | 448 | 37456 |
| bh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 6693 | 2507 |
| bh | fp32 | `asinh` | `default` | within 2 ULP | 1.64 | 0.773 | 0.881 | 3.39e+38 | 535 | 31332 |
| bh | fp32 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 1204 | 13929 |
| bh | fp32 | `atan` | `default` | accurate to |x| <= 0.902 | 2.34 | 0.806 | 0.955 | 0.902 | 400 ±823% | 41983 |
| bh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | 0.857 | — | 508 ±6% | 33023 |
| bh | fp32 | `atan2_bw` | `default` | worst sampled pairing | 1.64e+07 | 1.66e+05 | 0.587 | — | 4895 | 3428 |
| bh | fp32 | `atan_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.25e+03 | 0.859 | 1.02 | 1139 | 14732 |
| bh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000229 | 2.97 | 1.51 | 0.912 | 0.000229 | 503 | 33374 |
| bh | fp32 | `atanh_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.31e+03 | 0.876 | 0.681 | 7525 | 2230 |
| bh | fp32 | `bias_gelu` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | 0.746 | — | 495 | 33904 |
| bh | fp32 | `bias_gelu_` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | 0.746 | — | 471 | 35611 |
| bh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38 | 2.55 | 1.76 | 0.521 | 2.26e-38 | 391 ±89% | 42880 |
| bh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48823 |
| bh | fp32 | `celu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 386 | 43441 |
| bh | fp32 | `celu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2499 | 6713 |
| bh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48299 |
| bh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48616 |
| bh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48830 |
| bh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2172 | 7723 |
| bh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48315 |
| bh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48680 |
| bh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48758 |
| bh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2184 | 7683 |
| bh | fp32 | `cos` | `default` | accurate to |x| <= 92.4 | 1.34e+08 | 1.33e+05 | 0.927 | 92.4 | 383 | 43810 |
| bh | fp32 | `cos_bw` | `default` | 18643/65024 defects; rest 8.59e+51 ULP | 8.59e+51 | 3.23e+48 | 0.841 | 28 | 1504 | 11156 |
| bh | fp32 | `cosh` | `default` | within 2 ULP | 1.35 | 0.791 | 0.98 | 89.1 | 388 | 43232 |
| bh | fp32 | `cosh_bw` | `default` | 2/33894 defects; rest 2.2 ULP | 2.2 | 1.16 | 0.943 | 0.0155 | 6103 | 2749 |
| bh | fp32 | `deg2rad` | `default` | faithful; 63542/65024 tie-breaks | 0.63 | 0.595 | 0.906 | 3.4e+38 | 347 | 48374 |
| bh | fp32 | `digamma` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00543 | — | 702 | 23887 |
| bh | fp32 | `digamma_bw` | `default` | 257/64769 defects; rest 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.324 | 5.4e-20 | 6569 | 2554 |
| bh | fp32 | `div` | `default` | worst pairing 2.44 ULP; mean 1.44 | 2.44 | 1.44 | 0.894 | — | 505 ±5% | 33191 |
| bh | fp32 | `div_bw` | `default` | worst sampled pairing | 7.95e+04 | 7.95e+04 | 0.89 | — | 3047 | 5506 |
| bh | fp32 | `div_no_nan` | `default` | worst pairing 9.24e+04 ULP; mean 4.51e+04 | 9.24e+04 | 4.51e+04 | 0.702 | — | 1583 | 10596 |
| bh | fp32 | `divide` | `default` | worst pairing 2.44 ULP; mean 1.44 | 2.44 | 1.44 | 0.894 | — | 481 | 34900 |
| bh | fp32 | `divide_` | `default` | worst pairing 2.44 ULP; mean 1.44 | 2.44 | 1.44 | 0.894 | — | 479 | 35052 |
| bh | fp32 | `elu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 393 | 42730 |
| bh | fp32 | `elu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2481 | 6762 |
| bh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 477 | 35191 |
| bh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 471 | 35637 |
| bh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48569 |
| bh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334 | 7.38 | 1.36 | 0.672 | 0.000334 | 531 | 31613 |
| bh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 2185 | 7679 |
| bh | fp32 | `erfc` | `default` | 198/65024 unflushed; rest 2.1e+33 ULP | 2.1e+33 | 1.71e+29 | 0.329 | — | 461 ±9% | 36382 |
| bh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 2188 | 7669 |
| bh | fp32 | `erfinv` | `default` | 29420/32256 defects; rest 7.96e+06 ULP | 7.96e+06 | 2.59e+05 | 0.000202 | 1.32e-38 | 338 ±7% | 49607 |
| bh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456 | 1.03e+06 | 1.67e+03 | 0.914 | 0.000456 | 6719 | 2497 |
| bh | fp32 | `exp` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 385 | 43540 |
| bh | fp32 | `exp` | `fast_approx` | never within 2 ULP | 3.79e+05 | 2e+05 | 0.309 | — | 354 | 47337 |
| bh | fp32 | `exp2` | `default` | faithful; 6044/49536 tie-breaks | 0.965 | 0.626 | 0.992 | 3.39e+38 | 379 | 44215 |
| bh | fp32 | `exp2_bw` | `default` | 1/49537 defects; rest 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 1501 | 11177 |
| bh | fp32 | `exp_bw` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 1176 | 14262 |
| bh | fp32 | `expm1` | `default` | faithful; 5257/49458 tie-breaks | 0.997 | 0.565 | 0.997 | 3.39e+38 | 429 | 39149 |
| bh | fp32 | `expm1_bw` | `default` | accurate to |x| <= 22 | 4.19e+06 | 3.83e+03 | 0.995 | 22 | 1178 | 14238 |
| bh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48545 |
| bh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.56e+03 | 4.19e+06 | 1.56e+03 | 0.974 | — | 2257 | 7432 |
| bh | fp32 | `fmod` | `default` | worst pairing 1.61e+45 ULP; mean 2.55e+41 | 1.61e+45 | 2.55e+41 | 0.596 | — | 489 | 34292 |
| bh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48544 |
| bh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 474 | 35388 |
| bh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 476 | 35236 |
| bh | fp32 | `gelu` | `default` | 83/65024 defects; rest 1.71e+08 ULP | 1.71e+08 | 1.7e+05 | 0.963 | 0.208 | 349 | 48065 |
| bh | fp32 | `gelu` | `fast_approx` | 197/65024 defects; rest 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 346 | 48502 |
| bh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0298 | 1.47e+08 | 2.3e+04 | 0.929 | 0.0298 | 828 | 20272 |
| bh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48792 |
| bh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 499 | 33588 |
| bh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 472 | 35522 |
| bh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48669 |
| bh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48649 |
| bh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48765 |
| bh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1471 | 11402 |
| bh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25 | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 346 | 48482 |
| bh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2287 | 7334 |
| bh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36 | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 355 | 47281 |
| bh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133 | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 3241 | 5177 |
| bh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48604 |
| bh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1938 | 8655 |
| bh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48452 |
| bh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48612 |
| bh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48716 |
| bh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists; the rest reach 3.44e+06 ULP | 3.44e+06 | 3.28e+04 | 0.965 | — | 496 | 33837 |
| bh | fp32 | `hypot_bw` | `default` | 16384/65024 defects; rest 4.86e+06 ULP | 4.86e+06 | 4.48e+04 | 0.826 | — | 2971 | 5646 |
| bh | fp32 | `i0` | `default` | accurate to |x| <= 2.45 | 1.68e+07 | 2.19e+06 | 0.959 | 2.45 | 403 | 41617 |
| bh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00362 | 1.56e+07 | 4.1e+04 | 0.928 | 0.00362 | 1324 | 12675 |
| bh | fp32 | `i1` | `default` | accurate to |x| <= 0.00362 | 8.02 | 0.731 | 0.482 | 0.00362 | 540 | 31088 |
| bh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 343 | 48888 |
| bh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 502 | 33399 |
| bh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48674 |
| bh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 343 | 48928 |
| bh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 ±17% | 47684 |
| bh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48764 |
| bh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 343 | 48849 |
| bh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 496 ±6% | 33809 |
| bh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | 0.953 | — | 510 ±6% | 32867 |
| bh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | 0.953 | — | 484 | 34695 |
| bh | fp32 | `ldexp_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.742 | 0.742 | 0.996 | — | 2214 | 7579 |
| bh | fp32 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 474 | 35397 |
| bh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 473 | 35472 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 ±6% | 47976 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithful; 31672/65024 tie-breaks | 0.84 | 0.747 | 0.868 | 3.39e+38 | 346 | 48508 |
| bh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48718 |
| bh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 1640 | 10231 |
| bh | fp32 | `lerp` | `default` | worst sampled pairing | 1.46e+08 | 9.53e+04 | 0.962 | — | 653 | 25699 |
| bh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 2515 | 6672 |
| bh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 343 | 48875 |
| bh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181 | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 1291 | 12995 |
| bh | fp32 | `lgamma_bw` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00543 | — | 1480 | 11335 |
| bh | fp32 | `log` | `default` | faithful; 32512/32512 tie-breaks | 0.961 | 0.562 | 0.952 | 3.39e+38 | 401 | 41883 |
| bh | fp32 | `log10` | `default` | accurate to |x| <= 0.336 | 2.13 | 1.36 | 0.641 | 0.336 | 400 | 41892 |
| bh | fp32 | `log10_bw` | `default` | accurate to |x| <= 4.42e+30 | 9.02e+04 | 1.73e+03 | 0.667 | 4.42e+30 | 4702 | 3568 |
| bh | fp32 | `log1p` | `default` | faithful; 20111/48640 tie-breaks | 0.967 | 0.558 | 0.982 | 3.39e+38 | 397 | 42248 |
| bh | fp32 | `log1p_bw` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 2.96e+03 | 0.89 | 2.02e+31 | 4705 | 3566 |
| bh | fp32 | `log2` | `default` | accurate to |x| <= 0.704 | 2.43 | 0.517 | 0.993 | 0.704 | 391 | 42896 |
| bh | fp32 | `log2_bw` | `default` | 112/64626 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.76e+03 | 0.684 | — | 4694 | 3574 |
| bh | fp32 | `log_bw` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 3541 | 4738 |
| bh | fp32 | `log_sigmoid` | `default` | never within 2 ULP | 4.4e+05 | 1.82e+04 | 0.481 | — | 457 | 36750 |
| bh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.163 | 2.93 | 1.28 | 0.953 | 0.163 | 5207 | 3222 |
| bh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists; the rest reach 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 614 ±5% | 27338 |
| bh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists; the rest reach 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.716 | — | 542 | 30942 |
| bh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists; the rest reach 1.54e+09 ULP | 1.54e+09 | 1.05e+07 | 0.716 | — | 519 | 32338 |
| bh | fp32 | `logaddexp2_bw` | `default` | worst sampled pairing | 9.02e+04 | 5.6e+04 | 0.959 | — | 4729 | 3548 |
| bh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists; the rest reach 5.23e+08 ULP | 5.23e+08 | 7.28e+06 | 0.726 | — | 579 | 28983 |
| bh | fp32 | `logaddexp_bw` | `default` | worst sampled pairing | 8.98e+04 | 4.46e+04 | 0.962 | — | 4255 | 3943 |
| bh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 487 | 34484 |
| bh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 486 | 34548 |
| bh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48155 |
| bh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 343 | 48948 |
| bh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 485 | 34592 |
| bh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 485 | 34566 |
| bh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 509 | 32976 |
| bh | fp32 | `logit` | `default` | accurate to |x| <= 0.266 | 4.19e+06 | 261 | 0.941 | 0.266 | 355 | 47284 |
| bh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 5987 | 2802 |
| bh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 7485 | 2242 |
| bh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 500 | 33586 |
| bh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 472 | 35562 |
| bh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48682 |
| bh | fp32 | `mac` | `default` | worst sampled pairing | 4.22e+06 | 7.31e+05 | 0.974 | — | 669 | 25073 |
| bh | fp32 | `max_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 4855 | 3456 |
| bh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 501 | 33520 |
| bh | fp32 | `min_bw` | `default` | worst sampled pairing | 4.19e+06 | 4.19e+06 | 1 | — | 4837 | 3469 |
| bh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 500 | 33585 |
| bh | fp32 | `mish` | `default` | 9/65024 defects; rest 7 ULP | 7 | 2.58 | 0.953 | 1.49e-07 | 519 | 32327 |
| bh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 502 | 33443 |
| bh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1254 | 13378 |
| bh | fp32 | `multigammaln` | `default` | 7422/50376 defects; rest 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.55e-17 | 7855 | 2136 |
| bh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 1.36e+15 | 2.04e+11 | 0.0102 | 5.55e-17 | 7265 | 2309 |
| bh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 475 | 35338 |
| bh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 475 | 35295 |
| bh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 486 | 34556 |
| bh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 474 ±6% | 35415 |
| bh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48365 |
| bh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | 0.5 | — | 2880 | 5825 |
| bh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 ±9% | 48108 |
| bh | fp32 | `polygamma` | `k=1` | 1/49922 defects; rest 7.91e+33 ULP | 7.91e+33 | 4.68e+29 | 0.467 | 5.4e-20 | 1021 | 16436 |
| bh | fp32 | `polygamma` | `k=2` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 1063 | 15788 |
| bh | fp32 | `polygamma` | `k=4` | 141/49922 defects; rest 3.77e+25 ULP | 3.77e+25 | 6.51e+21 | 0.138 | 3.68e-08 | 1078 | 15557 |
| bh | fp32 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 6609 | 2539 |
| bh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.17 | 5.92e+03 | 3.17 | 0.992 | — | 677 | 24778 |
| bh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2275 | 7375 |
| bh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 345 | 48646 |
| bh | fp32 | `rad2deg` | `default` | faithful; 63518/63518 tie-breaks | 0.696 | 0.643 | 0.858 | 5.93e+36 | 345 | 48620 |
| bh | fp32 | `rdiv` | `value=2.0` | 258/64770 defects; rest 1.49 ULP | 1.49 | 1.19 | 0.664 | 8.47e+37 | 345 | 48616 |
| bh | fp32 | `rdiv_bw` | `scalar=2.0` | 254/48490 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.92e+03 | 0.705 | 7.67e-20 | 6095 | 2753 |
| bh | fp32 | `reciprocal` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 344 | 48744 |
| bh | fp32 | `reciprocal_bw` | `default` | 256/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.92e+03 | 0.705 | 5.42e-20 | 4559 | 3680 |
| bh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48467 |
| bh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48642 |
| bh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3596 | 4665 |
| bh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1131 | 14834 |
| bh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48299 |
| bh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 ±6% | 48203 |
| bh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48482 |
| bh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48602 |
| bh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48591 |
| bh | fp32 | `remainder` | `default` | worst pairing 8.92e+44 ULP; mean 2.49e+41 | 8.92e+44 | 2.49e+41 | 0.596 | — | 513 | 32692 |
| bh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48238 |
| bh | fp32 | `rpow` | `exponent=0.5` | faithful; 5624/49536 tie-breaks | 0.879 | 0.609 | 0.995 | 3.39e+38 | 632 | 26547 |
| bh | fp32 | `rpow` | `exponent=1.0` | 1536/49536 defects; rest 0 ULP | 0 | 0 | 1 | 8.28e+34 | 621 | 27005 |
| bh | fp32 | `rpow` | `exponent=2.0` | 1536/49536 defects; rest 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 630 | 26630 |
| bh | fp32 | `rpow_bw` | `exponent=2.0` | 32384/64768 defects; rest 0 ULP | 0 | 0 | 1 | 1.18e-38 | 2592 ±52% | 6474 |
| bh | fp32 | `rsqrt` | `default` | within 2 ULP | 1.12 | 0.834 | 0.864 | 3.39e+38 | 381 | 44090 |
| bh | fp32 | `rsqrt_bw` | `default` | 75/21666 defects; rest 6.21 ULP | 6.21 | 4.15 | 0.303 | — | 5213 | 3219 |
| bh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 501 | 33473 |
| bh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 474 | 35398 |
| bh | fp32 | `selu` | `default` | never within 2 ULP | 51.3 | 26.3 | 0 | — | 423 | 39679 |
| bh | fp32 | `selu_bw` | `default` | 1/65024 defects; rest 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 2823 | 5944 |
| bh | fp32 | `sigmoid` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 418 | 40112 |
| bh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 416 | 40284 |
| bh | fp32 | `sigmoid_bw` | `default` | 140/65024 defects; rest 8.39e+06 ULP | 8.39e+06 | 2.54e+04 | 0.973 | 0.283 | 1991 | 8427 |
| bh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48770 |
| bh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48659 |
| bh | fp32 | `silu` | `default` | 9/65024 defects; rest 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 426 | 39363 |
| bh | fp32 | `silu_bw` | `default` | 9/65024 defects; rest 5.72e+06 ULP | 5.72e+06 | 842 | 0.934 | 2.98e-07 | 2787 | 6020 |
| bh | fp32 | `sin` | `default` | accurate to |x| <= 28 | 3.36e+07 | 3.07e+04 | 0.96 | 28 | 377 | 44464 |
| bh | fp32 | `sin_bw` | `default` | 18430/65024 defects; rest 1.24e+52 ULP | 1.24e+52 | 3.44e+48 | 0.808 | 92.4 | 1165 | 14399 |
| bh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155 | 2.2 | 1.16 | 0.943 | 0.0155 | 432 ±6% | 38811 |
| bh | fp32 | `sinh_bw` | `default` | 2/33894 defects; rest 1.35 ULP | 1.35 | 0.791 | 0.98 | 88.5 | 5366 | 3126 |
| bh | fp32 | `softplus` | `default` | 5/65024 defects; rest 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 368 | 45548 |
| bh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.164 | 2.93 | 1.33 | 0.954 | 0.164 | 3942 ±8% | 4256 |
| bh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 344 | 48700 |
| bh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1939 | 8651 |
| bh | fp32 | `softsign` | `default` | 512/65024 defects; rest 3 ULP | 3 | 1.28 | 0.863 | 1.21e-05 | 346 | 48520 |
| bh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07 | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 1175 | 14280 |
| bh | fp32 | `sqrt` | `default` | faithful; 32512/32512 tie-breaks | 0.867 | 0.827 | 0.869 | 3.39e+38 | 368 | 45573 |
| bh | fp32 | `sqrt_bw` | `default` | never within 2 ULP | 2.19 | 1.4 | 0.708 | — | 5773 | 2906 |
| bh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 350 ±12% | 47874 |
| bh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1131 | 14830 |
| bh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 472 ±7% | 35520 |
| bh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 481 ±10% | 34863 |
| bh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1942 | 8639 |
| bh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 470 | 35695 |
| bh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 471 | 35613 |
| bh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 469 | 35778 |
| bh | fp32 | `swish` | `default` | 9/65024 defects; rest 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 422 ±6% | 39775 |
| bh | fp32 | `tan` | `default` | accurate to |x| <= 3.92 | 2.26e+08 | 2.32e+05 | 0.944 | 3.92 | 450 | 37316 |
| bh | fp32 | `tan_bw` | `default` | 20623/65024 defects; rest 2.85e+45 ULP | 2.85e+45 | 4.74e+44 | 0.87 | 0.852 | 1905 | 8809 |
| bh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000946 | 2.6 | 1.45 | 0.98 | 0.000946 | 387 | 43321 |
| bh | fp32 | `tanh_bw` | `default` | never within 2 ULP | 6.59e+04 | 7.51e+03 | 0.482 | — | 859 | 19538 |
| bh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08 | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 344 | 48708 |
| bh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09 | 4.19e+06 | 1.02e+05 | 0.901 | 7.42e-09 | 1506 | 11140 |
| bh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 | 48094 |
| bh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48634 |
| bh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1461 | 11482 |
| bh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48182 |
| bh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 648 | 25893 |
| bh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13 | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 398 | 42174 |
| bh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | 1.15e-05 | — | 504 | 33272 |
| bh | fp32 | `xlogy_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.766 | 0.766 | 0.951 | — | 8026 | 2090 |
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
