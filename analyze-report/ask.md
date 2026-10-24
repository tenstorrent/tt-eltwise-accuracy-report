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
| bh | bf16 | `978f93b56e3` | 0.80.0 | 2b2a06ae6c16 |
| bh | fp32 | `978f93b56e3` | 0.80.0 | 2b2a06ae6c16 |
| wh | bf16 | `978f93b56e3` | 0.80.0 | 722eff3324c2 |
| wh | fp32 | `978f93b56e3` | 0.80.0 | 722eff3324c2 |

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

## Results — 841 variants

| Arch | Dtype | Op | Parameters | Verdict | Max ULP | Mean ULP | Rounded | Usable to | µs | Melem/s |
|---|---|---|---|---|---|---|---|---|---|---|
| bh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 | 85891 |
| bh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 606 | 27682 |
| bh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 269 ±5% | 62374 |
| bh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.744 | 0.996 | 0.949 | 3536 | 4744 |
| bh | bf16 | `acosh` | `default` | within 2 ULP | 1.41 | 0.596 | 0.973 | 3.39e+38 | 311 | 53905 |
| bh | bf16 | `acosh_bw` | `default` | 15874/64514 defects; rest 3.07 ULP | 3.07 | 0.64 | 0.884 | 1.03 | 4052 | 4140 |
| bh | bf16 | `add` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 246 | 68292 |
| bh | bf16 | `add_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 242 ±9% | 69222 |
| bh | bf16 | `addalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 242 | 69236 |
| bh | bf16 | `addcdiv` | `default` | worst sampled pairing | 2.11e+06 | 2.35e+03 | 0.999 | — | 345 | 48625 |
| bh | bf16 | `addcmul` | `default` | worst sampled pairing | 126 | 19.4 | 1 | — | 348 ±7% | 48216 |
| bh | bf16 | `asin` | `default` | 2/32258 defects; rest 0.499 ULP | 0.499 | 0 | 1 | — | 260 | 64577 |
| bh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.702 | 0.992 | 0.949 | 3474 | 4829 |
| bh | bf16 | `asinh` | `default` | within 2 ULP | 1.32 | 0.615 | 0.991 | 3.39e+38 | 435 | 38577 |
| bh | bf16 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.2 ULP | 1.2 | 0.638 | 0.954 | 1.84e+19 | 608 | 27590 |
| bh | bf16 | `atan` | `default` | 2/65024 defects; rest 0.527 ULP | 0.527 | 0.513 | 1 | — | 206 ±6% | 81570 |
| bh | bf16 | `atan2` | `default` | worst sampled pairing | 200 | 3.56 | 0.901 | — | 273 | 61520 |
| bh | bf16 | `atan_bw` | `default` | 2/48386 defects; rest 2.23 ULP | 2.23 | 0.767 | 0.896 | 0.23 | 627 | 26749 |
| bh | bf16 | `atanh` | `default` | 2/32256 defects; rest 2.08 ULP | 2.08 | 0.995 | 0.175 | — | 236 | 70945 |
| bh | bf16 | `atanh_bw` | `default` | 2/48386 defects; rest 7.76 ULP | 7.76 | 0.885 | 0.835 | 0.918 | 3279 | 5116 |
| bh | bf16 | `bias_gelu_` | `default` | worst sampled pairing | 179 | 86.3 | 0.996 | — | 315 ±10% | 53326 |
| bh | bf16 | `cbrt` | `default` | faithful; 846/65024 tie-breaks | 0.507 | 0.502 | 0.987 | 3.39e+38 | 213 | 78747 |
| bh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 87868 |
| bh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 185 ±9% | 90491 |
| bh | bf16 | `celu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1268 | 13228 |
| bh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 | 86359 |
| bh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88765 |
| bh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88850 |
| bh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1116 | 15035 |
| bh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 ±6% | 87869 |
| bh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±7% | 87607 |
| bh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89230 |
| bh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1097 | 15297 |
| bh | bf16 | `composite_example` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 361 | 46426 |
| bh | bf16 | `cos` | `default` | faithful; 2/37354 tie-breaks | 0.552 | 0.552 | 1 | 9.99e+05 | 248 | 67526 |
| bh | bf16 | `cos_bw` | `default` | 21281/65024 defects; rest 2.09e+42 ULP | 2.09e+42 | 2.59e+39 | 0.867 | 2.15e+06 | 818 | 20502 |
| bh | bf16 | `cosh` | `default` | faithful; 4/33894 tie-breaks | 0.501 | 0.501 | 1 | 89 | 262 | 63986 |
| bh | bf16 | `cosh_bw` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 1369 | 12257 |
| bh | bf16 | `deg2rad` | `default` | 2/65024 defects; rest 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 192 | 87580 |
| bh | bf16 | `digamma` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 371 | 45184 |
| bh | bf16 | `digamma_bw` | `default` | 263/64769 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.36e+04 | 0.674 | 0.996 | 3149 | 5328 |
| bh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 277 ±7% | 60488 |
| bh | bf16 | `div_bw` | `default` | bit-exact | 0.498 | 0 | 1 | — | 1629 | 10296 |
| bh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 658 | 25511 |
| bh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 249 | 67447 |
| bh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 246 | 68064 |
| bh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 184 | 91154 |
| bh | bf16 | `elu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1273 | 13177 |
| bh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 256 | 65663 |
| bh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 87723 |
| bh | bf16 | `erf` | `default` | within 2 ULP | 1.06 | 0.771 | 0.729 | 3.39e+38 | 187 | 89644 |
| bh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.38 | 70.5 | 3.04 | 0.975 | 1.38 | 623 | 26936 |
| bh | bf16 | `erfc` | `default` | 198/65024 unflushed; rest 3.2e+28 ULP | 3.2e+28 | 3.23e+24 | 0.949 | 2.5 | 304 | 55116 |
| bh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.38 | 70.5 | 3.04 | 0.975 | 1.38 | 1137 | 14760 |
| bh | bf16 | `erfinv` | `default` | 29424/32256 defects; rest 121 ULP | 121 | 6.34 | 0.391 | 1.31e-38 | 327 | 51337 |
| bh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.859 | 8.91 | 1.2 | 0.98 | 0.859 | 2054 | 8168 |
| bh | bf16 | `exp` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 193 | 86814 |
| bh | bf16 | `exp` | `fast_approx` | never within 2 ULP | 6.08 | 3.05 | 0.314 | — | 189 | 88958 |
| bh | bf16 | `exp2` | `default` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 223 ±6% | 75344 |
| bh | bf16 | `exp2_bw` | `default` | 1/49537 defects; rest 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 815 | 20580 |
| bh | bf16 | `exp_bw` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 597 | 28081 |
| bh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 240 | 69840 |
| bh | bf16 | `expm1_bw` | `default` | accurate to |x| <= 33.5 | 34 | 0.695 | 0.984 | 33.5 | 597 | 28098 |
| bh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 88050 |
| bh | bf16 | `floor_div` | `default` | worst sampled pairing | 64 | 42 | 0.996 | — | 1149 | 14606 |
| bh | bf16 | `fmod` | `default` | worst sampled pairing | 9.14e+35 | 8.94e+34 | 0.673 | — | 278 | 60289 |
| bh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 185 | 90739 |
| bh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 240 | 69822 |
| bh | bf16 | `gelu` | `default` | 86/65024 defects; rest 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 289 | 58043 |
| bh | bf16 | `gelu` | `fast_approx` | 198/65024 defects; rest 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 188 | 89155 |
| bh | bf16 | `gelu_bw` | `default` | faithful; 38/65024 tie-breaks | 0.939 | 0.681 | 0.999 | 3.39e+38 | 666 | 25195 |
| bh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±6% | 87161 |
| bh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 255 | 65720 |
| bh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88327 |
| bh | bf16 | `hardmish` | `default` | faithful; 2587/65024 tie-breaks | 1 | 0.913 | 0.961 | 3.39e+38 | 190 | 88473 |
| bh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88549 |
| bh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21757 |
| bh | bf16 | `hardsigmoid` | `default` | within 2 ULP | 1 | 0.557 | 0.986 | 3.39e+38 | 191 ±10% | 87627 |
| bh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 1175 | 14275 |
| bh | bf16 | `hardswish` | `default` | 2/65024 defects; rest 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 194 ±16% | 86599 |
| bh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.13 | 85.3 | 2.1 | 0.993 | 1.13 | 1651 | 10160 |
| bh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 187 | 89517 |
| bh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 980 | 17122 |
| bh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88381 |
| bh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88140 |
| bh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88184 |
| bh | bf16 | `hypot` | `default` | 16384/65026 defects; rest 52.7 ULP | 52.7 | 1.64 | 0.992 | — | 263 ±12% | 63801 |
| bh | bf16 | `i0` | `default` | accurate to |x| <= 13.6 | 255 | 56.3 | 0.952 | 13.6 | 258 | 65084 |
| bh | bf16 | `i0_bw` | `default` | 4/33904 defects; rest 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 832 | 20154 |
| bh | bf16 | `i1` | `default` | 4/65024 defects; rest 0.858 ULP | 0.858 | 0.589 | 0.516 | 2.33e-38 | 425 | 39511 |
| bh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±6% | 88281 |
| bh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 273 ±11% | 61525 |
| bh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 | 86015 |
| bh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89131 |
| bh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88990 |
| bh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89110 |
| bh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88888 |
| bh | bf16 | `l1_loss` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 260 | 64404 |
| bh | bf16 | `ldexp_` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 262 ±7% | 64122 |
| bh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 245 ±8% | 68396 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89084 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithful; 15341/65024 tie-breaks | 0.96 | 0.741 | 0.761 | 3.39e+38 | 188 | 89199 |
| bh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 ±7% | 88874 |
| bh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 864 | 19410 |
| bh | bf16 | `lerp` | `default` | worst sampled pairing | 1.77e+03 | 39.9 | 1 | — | 337 ±6% | 49728 |
| bh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 | 86285 |
| bh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.412 | 324 | 0.846 | 0.76 | 0.412 | 834 | 20112 |
| bh | bf16 | `lgamma_bw` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 776 | 21629 |
| bh | bf16 | `log` | `default` | faithful; 63/32512 tie-breaks | 0.78 | 0.534 | 0.998 | 3.39e+38 | 226 | 74192 |
| bh | bf16 | `log10` | `default` | faithful; 58/32512 tie-breaks | 0.785 | 0.543 | 0.998 | 3.39e+38 | 240 | 70000 |
| bh | bf16 | `log10_bw` | `default` | within 2 ULP | 1.74 | 0.803 | 0.508 | 3.69e+37 | 2443 | 6869 |
| bh | bf16 | `log1p` | `default` | faithful; 109/48640 tie-breaks | 0.931 | 0.57 | 0.998 | 3.39e+38 | 239 | 70119 |
| bh | bf16 | `log1p_bw` | `default` | 2/64514 defects; rest 1.42 ULP | 1.42 | 0.744 | 0.982 | 8.47e+37 | 2402 | 6985 |
| bh | bf16 | `log2` | `default` | faithful; 59/32512 tie-breaks | 0.766 | 0.542 | 0.998 | 3.39e+38 | 255 | 65796 |
| bh | bf16 | `log2_bw` | `default` | 116/64626 defects; rest 1.51 ULP | 1.51 | 0.847 | 0.57 | — | 2425 | 6919 |
| bh | bf16 | `log_bw` | `default` | 2/64514 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 1853 | 9056 |
| bh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.428 | 6.39 | 1.38 | 0.958 | 0.428 | 220 | 76348 |
| bh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP | 1.71 | 0.722 | 0.976 | 3.39e+38 | 2701 | 6211 |
| bh | bf16 | `logaddexp2_` | `default` | worst sampled pairing | 6.87e+10 | 2.05e+06 | 0.988 | — | 275 ±6% | 61093 |
| bh | bf16 | `logaddexp_` | `default` | worst sampled pairing | 2.22e+05 | 44.7 | 0.986 | — | 253 | 66208 |
| bh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 248 | 67764 |
| bh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±8% | 88391 |
| bh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 186 | 90015 |
| bh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 246 ±5% | 68321 |
| bh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 245 | 68410 |
| bh | bf16 | `logit` | `default` | accurate to |x| <= 0.395 | 64 | 1.67 | 0.985 | 0.395 | 321 | 52239 |
| bh | bf16 | `logit_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3035 | 5528 |
| bh | bf16 | `logiteps_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3828 | 4383 |
| bh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 255 ±8% | 65724 |
| bh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 | 86767 |
| bh | bf16 | `mac` | `default` | worst sampled pairing | 63.5 | 26.4 | 1 | — | 342 ±6% | 49105 |
| bh | bf16 | `mish` | `default` | 11/65024 defects; rest 1.49 ULP | 1.49 | 0.666 | 0.988 | 1.95e-38 | 239 | 70343 |
| bh | bf16 | `mse_loss` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 261 ±12% | 64232 |
| bh | bf16 | `multigammaln` | `default` | 11536/48328 defects; rest 724 ULP | 724 | 1.67 | 0.717 | 5.59e-17 | 4631 | 3623 |
| bh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.59e-17 | 2.38e+06 | 293 | 0.249 | 5.59e-17 | 3706 | 4527 |
| bh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 249 | 67460 |
| bh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 241 | 69732 |
| bh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 256 | 65485 |
| bh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 | 86658 |
| bh | bf16 | `nextafter` | `default` | within 2 ULP | 1 | 1 | 0.00585 | — | 254 | 65960 |
| bh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88560 |
| bh | bf16 | `polygamma` | `k=1` | 7/49922 defects; rest 1.02e+08 ULP | 1.02e+08 | 2.13e+05 | 0.97 | 0.996 | 326 | 51489 |
| bh | bf16 | `polygamma` | `k=2` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 377 ±7% | 44522 |
| bh | bf16 | `polygamma` | `k=4` | 142/49922 defects; rest 3.22e+09 ULP | 3.22e+09 | 1.66e+07 | 0.958 | 3.5 | 418 | 40159 |
| bh | bf16 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 3205 | 5235 |
| bh | bf16 | `pow` | `default` | worst sampled pairing | 4.86e+18 | 2.08e+14 | 0.982 | — | 411 | 40858 |
| bh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1182 | 14199 |
| bh | bf16 | `prelu` | `weight=0.25` | 1/65024 defects; rest 0 ULP | 0 | 0 | 1 | 4.68e-38 | 188 | 89112 |
| bh | bf16 | `rad2deg` | `default` | faithful; 31760/63518 tie-breaks | 0.992 | 0.75 | 0.5 | 5.9e+36 | 189 | 88914 |
| bh | bf16 | `rdiv` | `value=2.0` | 258/64770 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 189 ±14% | 88861 |
| bh | bf16 | `reciprocal` | `default` | 2/64514 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 189 | 88608 |
| bh | bf16 | `reciprocal_bw` | `default` | 256/48386 defects; rest 1.23 ULP | 1.23 | 0.714 | 0.734 | 5.42e-20 | 1488 | 11278 |
| bh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 | 86484 |
| bh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 187 | 89543 |
| bh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 992 | 16917 |
| bh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 594 | 28268 |
| bh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89344 |
| bh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88321 |
| bh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89016 |
| bh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88676 |
| bh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88624 |
| bh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88748 |
| bh | bf16 | `rpow` | `exponent=0.5` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.975 | 3.39e+38 | 376 | 44606 |
| bh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 372 | 45078 |
| bh | bf16 | `rpow` | `exponent=2.0` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 374 | 44839 |
| bh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 253 | 66283 |
| bh | bf16 | `rsqrt_bw` | `default` | 75/21665 defects; rest 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 2426 | 6917 |
| bh | bf16 | `rsub` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 263 ±8% | 63780 |
| bh | bf16 | `rsub_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 245 ±6% | 68452 |
| bh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 197 ±11% | 85011 |
| bh | bf16 | `selu_bw` | `default` | 1/65024 defects; rest 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 1472 | 11394 |
| bh | bf16 | `sigmoid` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 260 | 64464 |
| bh | bf16 | `sigmoid_accurate` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 264 ±6% | 63453 |
| bh | bf16 | `sigmoid_bw` | `default` | 141/65024 defects; rest 127 ULP | 127 | 12.7 | 0.999 | 12.5 | 461 | 36422 |
| bh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88435 |
| bh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 188 | 89269 |
| bh | bf16 | `silu` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 223 | 75102 |
| bh | bf16 | `silu_bw` | `default` | 9/65024 defects; rest 285 ULP | 285 | 1.3 | 0.979 | 0.863 | 1453 | 11549 |
| bh | bf16 | `sin` | `default` | faithful; 4/37354 tie-breaks | 0.552 | 0.526 | 1 | 9.99e+05 | 224 | 74910 |
| bh | bf16 | `sin_bw` | `default` | 21284/65024 defects; rest 2.98e+41 ULP | 2.98e+41 | 1.39e+39 | 0.857 | 1.07e+06 | 644 | 26035 |
| bh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 173 | 96823 |
| bh | bf16 | `sinh_bw` | `default` | faithful; 4/33894 tie-breaks | 0.501 | 0.501 | 1 | 89 | 1450 | 11572 |
| bh | bf16 | `softcap` | `beta=50.0` | 1426/65024 defects; rest 0.718 ULP | 0.718 | 0.574 | 0.997 | — | 228 | 73586 |
| bh | bf16 | `softplus` | `default` | 526/65024 defects; rest 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 165 ±18% | 101444 |
| bh | bf16 | `softplus_bw` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 1333 | 12586 |
| bh | bf16 | `softshrink` | `default` | faithful; 3968/65024 tie-breaks | 1 | 0.948 | 0.941 | 3.39e+38 | 190 | 88455 |
| bh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 984 | 17048 |
| bh | bf16 | `softsign` | `default` | 512/65024 defects; rest 1 ULP | 1 | 0.605 | 0.907 | 8.47e+37 | 188 | 89049 |
| bh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391 | 79 | 3.74 | 0.887 | 0.00391 | 642 | 26134 |
| bh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 245 | 68583 |
| bh | bf16 | `sqrt_bw` | `default` | within 2 ULP | 1.14 | 0.699 | 0.723 | 3.39e+38 | 3030 | 5538 |
| bh | bf16 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 190 | 88264 |
| bh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 601 | 27898 |
| bh | bf16 | `squared_difference` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 242 | 69358 |
| bh | bf16 | `squared_difference_` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 239 | 70160 |
| bh | bf16 | `subalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 251 ±6% | 66740 |
| bh | bf16 | `subtract` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 241 | 69478 |
| bh | bf16 | `subtract_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 239 | 70261 |
| bh | bf16 | `swish` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 224 ±7% | 74943 |
| bh | bf16 | `tan` | `default` | faithful; 30/37354 tie-breaks | 0.551 | 0.509 | 0.999 | 9.99e+05 | 248 ±6% | 67634 |
| bh | bf16 | `tan_bw` | `default` | 22782/65024 defects; rest 3.57e+40 ULP | 3.57e+40 | 1e+38 | 0.853 | 940 | 998 | 16803 |
| bh | bf16 | `tanh` | `default` | 2/65024 defects; rest 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 190 | 88302 |
| bh | bf16 | `tanh_bw` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 569 | 29496 |
| bh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08 | 63.5 | 9.38 | 0.981 | 1.35e-08 | 229 ±7% | 73232 |
| bh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09 | 63 | 6.1 | 0.967 | 7.45e-09 | 769 | 21817 |
| bh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88212 |
| bh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88964 |
| bh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 759 | 22095 |
| bh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 | 86169 |
| bh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 328 | 51187 |
| bh | bf16 | `xielu` | `default` | 1/56847 defects; rest 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 403 | 41649 |
| bh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 ±5% | 48062 |
| bh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1152 | 14569 |
| bh | fp32 | `acos` | `default` | within 2 ULP | 1.55 | 0.865 | 0.923 | 1 | 472 | 35548 |
| bh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 6867 | 2443 |
| bh | fp32 | `acosh` | `default` | never within 2 ULP | 2.47 | 0.76 | 0.782 | — | 391 | 42957 |
| bh | fp32 | `acosh_bw` | `default` | 15874/64514 defects; rest 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 7834 | 2142 |
| bh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 473 | 35437 |
| bh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 470 | 35661 |
| bh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 474 | 35388 |
| bh | fp32 | `addcdiv` | `default` | worst sampled pairing | 2.19e+12 | 8.82e+07 | 0.852 | — | 663 | 25304 |
| bh | fp32 | `addcmul` | `default` | worst sampled pairing | 2.13e+06 | 1.59e+04 | 0.985 | — | 667 | 25164 |
| bh | fp32 | `asin` | `default` | within 2 ULP | 1.77 | 0.795 | 0.985 | 1 | 452 | 37086 |
| bh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 6706 | 2502 |
| bh | fp32 | `asinh` | `default` | within 2 ULP | 1.64 | 0.773 | 0.881 | 3.39e+38 | 529 | 31703 |
| bh | fp32 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 1204 | 13940 |
| bh | fp32 | `atan` | `default` | accurate to |x| <= 0.902 | 2.34 | 0.806 | 0.955 | 0.902 | 398 | 42158 |
| bh | fp32 | `atan2` | `default` | worst sampled pairing | 1.32e+07 | 1.26e+05 | 0.857 | — | 513 | 32688 |
| bh | fp32 | `atan_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.25e+03 | 0.859 | 1.02 | 1157 | 14504 |
| bh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000229 | 2.97 | 1.51 | 0.912 | 0.000229 | 525 | 31980 |
| bh | fp32 | `atanh_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.31e+03 | 0.876 | 0.681 | 6398 | 2622 |
| bh | fp32 | `bias_gelu_` | `default` | worst sampled pairing | 1.27e+08 | 1.48e+07 | 0.964 | — | 530 | 31631 |
| bh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38 | 2.55 | 1.76 | 0.521 | 2.26e-38 | 383 | 43845 |
| bh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48390 |
| bh | fp32 | `celu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 389 | 43078 |
| bh | fp32 | `celu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2492 | 6733 |
| bh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48627 |
| bh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48684 |
| bh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48564 |
| bh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2190 | 7661 |
| bh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48604 |
| bh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48781 |
| bh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48720 |
| bh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2183 | 7686 |
| bh | fp32 | `composite_example` | `default` | never within 2 ULP | 6.55e+04 | 6.55e+04 | 1.53e-05 | — | 672 | 24984 |
| bh | fp32 | `cos` | `default` | accurate to |x| <= 92.4 | 1.34e+08 | 1.33e+05 | 0.927 | 92.4 | 386 | 43469 |
| bh | fp32 | `cos_bw` | `default` | 18643/65024 defects; rest 8.59e+51 ULP | 8.59e+51 | 3.23e+48 | 0.841 | 28 | 1492 | 11248 |
| bh | fp32 | `cosh` | `default` | within 2 ULP | 1.35 | 0.791 | 0.98 | 89.1 | 386 | 43438 |
| bh | fp32 | `cosh_bw` | `default` | accurate to |x| <= 0.0155 | 2.2 | 1.16 | 0.943 | 0.0155 | 2835 | 5918 |
| bh | fp32 | `deg2rad` | `default` | faithful; 63542/65024 tie-breaks | 0.63 | 0.595 | 0.906 | 3.4e+38 | 344 | 48727 |
| bh | fp32 | `digamma` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00543 | — | 694 | 24176 |
| bh | fp32 | `digamma_bw` | `default` | 257/64769 defects; rest 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.324 | 5.4e-20 | 6560 | 2558 |
| bh | fp32 | `div` | `default` | worst sampled pairing | 2.44 | 1.44 | 0.894 | — | 503 | 33369 |
| bh | fp32 | `div_bw` | `default` | worst sampled pairing | 7.95e+04 | 7.95e+04 | 0.89 | — | 3181 | 5274 |
| bh | fp32 | `div_no_nan` | `default` | worst sampled pairing | 9.24e+04 | 4.51e+04 | 0.702 | — | 1590 | 10551 |
| bh | fp32 | `divide` | `default` | worst sampled pairing | 2.44 | 1.44 | 0.894 | — | 480 | 34946 |
| bh | fp32 | `divide_` | `default` | worst sampled pairing | 2.44 | 1.44 | 0.894 | — | 477 | 35201 |
| bh | fp32 | `elu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 393 | 42697 |
| bh | fp32 | `elu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2479 | 6767 |
| bh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 490 | 34264 |
| bh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48591 |
| bh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334 | 7.38 | 1.36 | 0.672 | 0.000334 | 525 | 31971 |
| bh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 1169 | 14352 |
| bh | fp32 | `erfc` | `default` | 198/65024 unflushed; rest 2.1e+33 ULP | 2.1e+33 | 1.71e+29 | 0.329 | — | 427 ±13% | 39273 |
| bh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 2176 | 7711 |
| bh | fp32 | `erfinv` | `default` | 29420/32256 defects; rest 7.96e+06 ULP | 7.96e+06 | 2.59e+05 | 0.000202 | 1.32e-38 | 336 | 49951 |
| bh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456 | 1.03e+06 | 1.67e+03 | 0.914 | 0.000456 | 3577 | 4690 |
| bh | fp32 | `exp` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 391 | 42934 |
| bh | fp32 | `exp` | `fast_approx` | never within 2 ULP | 3.79e+05 | 2e+05 | 0.309 | — | 344 | 48818 |
| bh | fp32 | `exp2` | `default` | faithful; 6044/49536 tie-breaks | 0.965 | 0.626 | 0.992 | 3.39e+38 | 381 | 44044 |
| bh | fp32 | `exp2_bw` | `default` | 1/49537 defects; rest 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 1501 | 11179 |
| bh | fp32 | `exp_bw` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 1174 | 14288 |
| bh | fp32 | `expm1` | `default` | faithful; 5257/49458 tie-breaks | 0.997 | 0.565 | 0.997 | 3.39e+38 | 440 | 38124 |
| bh | fp32 | `expm1_bw` | `default` | accurate to |x| <= 22 | 4.19e+06 | 3.83e+03 | 0.995 | 22 | 1177 | 14254 |
| bh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48466 |
| bh | fp32 | `floor_div` | `default` | worst sampled pairing | 4.19e+06 | 1.56e+03 | 0.974 | — | 2240 | 7488 |
| bh | fp32 | `fmod` | `default` | worst sampled pairing | 1.61e+45 | 2.55e+41 | 0.596 | — | 488 | 34371 |
| bh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48513 |
| bh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 472 | 35560 |
| bh | fp32 | `gelu` | `default` | 83/65024 defects; rest 1.71e+08 ULP | 1.71e+08 | 1.7e+05 | 0.963 | 0.208 | 343 | 48890 |
| bh | fp32 | `gelu` | `fast_approx` | 197/65024 defects; rest 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 350 | 47960 |
| bh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0298 | 1.47e+08 | 2.3e+04 | 0.929 | 0.0298 | 818 | 20508 |
| bh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48352 |
| bh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 488 | 34363 |
| bh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48682 |
| bh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48521 |
| bh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 ±6% | 48508 |
| bh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1464 | 11458 |
| bh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25 | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 348 | 48214 |
| bh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2293 | 7318 |
| bh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36 | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 350 | 47963 |
| bh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133 | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 3251 | 5160 |
| bh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 | 48064 |
| bh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1948 | 8614 |
| bh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48397 |
| bh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48327 |
| bh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 | 48138 |
| bh | fp32 | `hypot` | `default` | 16384/65024 defects; rest 3.44e+06 ULP | 3.44e+06 | 3.28e+04 | 0.965 | — | 515 | 32580 |
| bh | fp32 | `i0` | `default` | accurate to |x| <= 2.45 | 1.68e+07 | 2.19e+06 | 0.959 | 2.45 | 400 | 41915 |
| bh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00362 | 1.56e+07 | 4.1e+04 | 0.928 | 0.00362 | 1333 | 12586 |
| bh | fp32 | `i1` | `default` | accurate to |x| <= 0.00362 | 8.02 | 0.731 | 0.482 | 0.00362 | 543 | 30881 |
| bh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48402 |
| bh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 503 | 33362 |
| bh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48589 |
| bh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48593 |
| bh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47774 |
| bh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48679 |
| bh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 ±13% | 48564 |
| bh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 494 | 33939 |
| bh | fp32 | `ldexp_` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 503 | 33328 |
| bh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 475 | 35330 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48784 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithful; 31672/65024 tie-breaks | 0.84 | 0.747 | 0.868 | 3.39e+38 | 345 | 48581 |
| bh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48740 |
| bh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 1655 | 10138 |
| bh | fp32 | `lerp` | `default` | worst sampled pairing | 1.46e+08 | 9.53e+04 | 0.962 | — | 680 ±21% | 24667 |
| bh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48556 |
| bh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181 | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 1298 | 12924 |
| bh | fp32 | `lgamma_bw` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00543 | — | 1484 | 11304 |
| bh | fp32 | `log` | `default` | faithful; 32512/32512 tie-breaks | 0.961 | 0.562 | 0.952 | 3.39e+38 | 400 | 41898 |
| bh | fp32 | `log10` | `default` | accurate to |x| <= 0.336 | 2.13 | 1.36 | 0.641 | 0.336 | 399 | 42034 |
| bh | fp32 | `log10_bw` | `default` | accurate to |x| <= 4.42e+30 | 9.02e+04 | 1.73e+03 | 0.667 | 4.42e+30 | 4708 | 3563 |
| bh | fp32 | `log1p` | `default` | faithful; 20111/48640 tie-breaks | 0.967 | 0.558 | 0.982 | 3.39e+38 | 394 | 42540 |
| bh | fp32 | `log1p_bw` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 2.96e+03 | 0.89 | 2.02e+31 | 4709 | 3563 |
| bh | fp32 | `log2` | `default` | accurate to |x| <= 0.704 | 2.43 | 0.517 | 0.993 | 0.704 | 386 | 43412 |
| bh | fp32 | `log2_bw` | `default` | 112/64626 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.76e+03 | 0.684 | — | 4701 | 3569 |
| bh | fp32 | `log_bw` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 3570 | 4699 |
| bh | fp32 | `log_sigmoid` | `default` | never within 2 ULP | 4.4e+05 | 1.82e+04 | 0.481 | — | 456 | 36827 |
| bh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.163 | 2.93 | 1.28 | 0.953 | 0.163 | 5201 | 3226 |
| bh | fp32 | `logaddexp2_` | `default` | worst sampled pairing | 1.06e+07 | 1.23e+05 | 0.959 | — | 516 | 32515 |
| bh | fp32 | `logaddexp_` | `default` | worst sampled pairing | 8e+06 | 9.77e+04 | 0.975 | — | 535 | 31361 |
| bh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 481 | 34873 |
| bh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48708 |
| bh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 342 | 48996 |
| bh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 481 | 34872 |
| bh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 489 | 34279 |
| bh | fp32 | `logit` | `default` | accurate to |x| <= 0.266 | 4.19e+06 | 261 | 0.941 | 0.266 | 353 | 47566 |
| bh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 5986 | 2803 |
| bh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 7509 | 2234 |
| bh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 488 | 34415 |
| bh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48592 |
| bh | fp32 | `mac` | `default` | worst sampled pairing | 4.22e+06 | 7.31e+05 | 0.974 | — | 666 | 25186 |
| bh | fp32 | `mish` | `default` | 9/65024 defects; rest 7 ULP | 7 | 2.58 | 0.953 | 1.49e-07 | 526 | 31866 |
| bh | fp32 | `mse_loss` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 474 ±6% | 35404 |
| bh | fp32 | `multigammaln` | `default` | 7422/50376 defects; rest 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.55e-17 | 7847 | 2138 |
| bh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 1.36e+15 | 2.04e+11 | 0.0102 | 5.55e-17 | 7277 | 2306 |
| bh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 472 | 35570 |
| bh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 470 | 35696 |
| bh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 486 | 34522 |
| bh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48625 |
| bh | fp32 | `nextafter` | `default` | faithful; 65024/65024 tie-breaks | 1 | 1 | 1.54e-05 | — | 514 ±14% | 32633 |
| bh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 ±9% | 48406 |
| bh | fp32 | `polygamma` | `k=1` | 1/49922 defects; rest 7.91e+33 ULP | 7.91e+33 | 4.68e+29 | 0.467 | 5.4e-20 | 1021 | 16439 |
| bh | fp32 | `polygamma` | `k=2` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 1052 | 15942 |
| bh | fp32 | `polygamma` | `k=4` | 141/49922 defects; rest 3.77e+25 ULP | 3.77e+25 | 6.51e+21 | 0.138 | 3.68e-08 | 1069 | 15688 |
| bh | fp32 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 6626 | 2532 |
| bh | fp32 | `pow` | `default` | worst sampled pairing | 333 | 2.35 | 0.993 | — | 660 | 25405 |
| bh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2269 | 7394 |
| bh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 345 | 48565 |
| bh | fp32 | `rad2deg` | `default` | faithful; 63518/63518 tie-breaks | 0.696 | 0.643 | 0.858 | 5.93e+36 | 344 | 48704 |
| bh | fp32 | `rdiv` | `value=2.0` | 258/64770 defects; rest 1.49 ULP | 1.49 | 1.19 | 0.664 | 8.47e+37 | 346 | 48552 |
| bh | fp32 | `reciprocal` | `default` | 2/64514 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 343 | 48915 |
| bh | fp32 | `reciprocal_bw` | `default` | 256/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 1.92e+03 | 0.705 | 5.42e-20 | 2948 | 5692 |
| bh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48597 |
| bh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48506 |
| bh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1945 | 8628 |
| bh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1133 | 14807 |
| bh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48774 |
| bh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48439 |
| bh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48747 |
| bh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48356 |
| bh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48702 |
| bh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48489 |
| bh | fp32 | `rpow` | `exponent=0.5` | faithful; 5624/49536 tie-breaks | 0.879 | 0.609 | 0.995 | 3.39e+38 | 629 | 26688 |
| bh | fp32 | `rpow` | `exponent=1.0` | 1536/49536 defects; rest 0 ULP | 0 | 0 | 1 | 8.28e+34 | 619 | 27109 |
| bh | fp32 | `rpow` | `exponent=2.0` | 1536/49536 defects; rest 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 629 | 26679 |
| bh | fp32 | `rsqrt` | `default` | within 2 ULP | 1.12 | 0.834 | 0.864 | 3.39e+38 | 386 | 43439 |
| bh | fp32 | `rsqrt_bw` | `default` | 75/21666 defects; rest 6.21 ULP | 6.21 | 4.15 | 0.303 | — | 4439 | 3780 |
| bh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 491 ±8% | 34143 |
| bh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 475 | 35305 |
| bh | fp32 | `selu` | `default` | never within 2 ULP | 51.3 | 26.3 | 0 | — | 405 ±8% | 41376 |
| bh | fp32 | `selu_bw` | `default` | 1/65024 defects; rest 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 2828 | 5932 |
| bh | fp32 | `sigmoid` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 416 | 40350 |
| bh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 415 | 40413 |
| bh | fp32 | `sigmoid_bw` | `default` | 140/65024 defects; rest 8.39e+06 ULP | 8.39e+06 | 2.54e+04 | 0.973 | 0.283 | 844 | 19882 |
| bh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48722 |
| bh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48769 |
| bh | fp32 | `silu` | `default` | 9/65024 defects; rest 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 421 | 39858 |
| bh | fp32 | `silu_bw` | `default` | 9/65024 defects; rest 5.72e+06 ULP | 5.72e+06 | 842 | 0.934 | 2.98e-07 | 2797 | 5999 |
| bh | fp32 | `sin` | `default` | accurate to |x| <= 28 | 3.36e+07 | 3.07e+04 | 0.96 | 28 | 368 | 45589 |
| bh | fp32 | `sin_bw` | `default` | 18430/65024 defects; rest 1.24e+52 ULP | 1.24e+52 | 3.44e+48 | 0.808 | 92.4 | 1169 | 14348 |
| bh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155 | 2.2 | 1.16 | 0.943 | 0.0155 | 434 | 38692 |
| bh | fp32 | `sinh_bw` | `default` | within 2 ULP | 1.35 | 0.791 | 0.98 | 89.1 | 2774 | 6047 |
| bh | fp32 | `softplus` | `default` | 5/65024 defects; rest 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 372 | 45126 |
| bh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 2510 | 6684 |
| bh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 345 | 48592 |
| bh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1949 | 8610 |
| bh | fp32 | `softsign` | `default` | 512/65024 defects; rest 3 ULP | 3 | 1.28 | 0.863 | 1.21e-05 | 345 | 48590 |
| bh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07 | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 1159 | 14470 |
| bh | fp32 | `sqrt` | `default` | faithful; 32512/32512 tie-breaks | 0.867 | 0.827 | 0.869 | 3.39e+38 | 378 | 44356 |
| bh | fp32 | `sqrt_bw` | `default` | never within 2 ULP | 2.19 | 1.4 | 0.708 | — | 5806 | 2890 |
| bh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 345 | 48630 |
| bh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1133 | 14806 |
| bh | fp32 | `squared_difference` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 473 | 35462 |
| bh | fp32 | `squared_difference_` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 469 | 35761 |
| bh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 497 | 33770 |
| bh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 472 | 35527 |
| bh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 469 | 35744 |
| bh | fp32 | `swish` | `default` | 9/65024 defects; rest 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 421 ±5% | 39814 |
| bh | fp32 | `tan` | `default` | accurate to |x| <= 3.92 | 2.26e+08 | 2.32e+05 | 0.944 | 3.92 | 453 | 37063 |
| bh | fp32 | `tan_bw` | `default` | 20623/65024 defects; rest 2.85e+45 ULP | 2.85e+45 | 4.74e+44 | 0.87 | 0.852 | 1907 | 8799 |
| bh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000946 | 2.6 | 1.45 | 0.98 | 0.000946 | 389 | 43152 |
| bh | fp32 | `tanh_bw` | `default` | accurate to |x| <= 0.199 | 5 | 1.67 | 0.935 | 0.199 | 856 | 19608 |
| bh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08 | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 339 | 49423 |
| bh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09 | 4.19e+06 | 1.02e+05 | 0.901 | 7.42e-09 | 1511 | 11100 |
| bh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48354 |
| bh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48566 |
| bh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1469 | 11424 |
| bh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48591 |
| bh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 653 | 25701 |
| bh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13 | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 403 | 41638 |
| wh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 ±7% | 41598 |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1305 | 12859 |
| wh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 679 | 24715 |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.744 | 0.996 | 0.949 | 7456 | 2250 |
| wh | bf16 | `acosh` | `default` | within 2 ULP | 1.41 | 0.596 | 0.973 | 3.39e+38 | 901 | 18618 |
| wh | bf16 | `acosh_bw` | `default` | 15874/64514 defects; rest 3.07 ULP | 3.07 | 0.64 | 0.884 | 1.03 | 8606 | 1950 |
| wh | bf16 | `add` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 529 | 31738 |
| wh | bf16 | `add_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 543 | 30904 |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 554 | 30267 |
| wh | bf16 | `addcdiv` | `default` | worst sampled pairing | 2.11e+06 | 2.35e+03 | 0.999 | — | 706 | 23760 |
| wh | bf16 | `addcmul` | `default` | worst sampled pairing | 126 | 19.4 | 1 | — | 712 ±6% | 23570 |
| wh | bf16 | `asin` | `default` | 2/32258 defects; rest 0.499 ULP | 0.499 | 0 | 1 | — | 663 | 25316 |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.702 | 0.992 | 0.949 | 7481 | 2242 |
| wh | bf16 | `asinh` | `default` | within 2 ULP | 1.32 | 0.619 | 0.992 | 3.39e+38 | 1395 | 12026 |
| wh | bf16 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.2 ULP | 1.2 | 0.638 | 0.954 | 1.84e+19 | 1404 | 11948 |
| wh | bf16 | `atan` | `default` | 2/65024 defects; rest 0.527 ULP | 0.527 | 0.513 | 1 | — | 571 ±83% | 29376 |
| wh | bf16 | `atan2` | `default` | worst sampled pairing | 200 | 2.51 | 0.998 | — | 566 | 29655 |
| wh | bf16 | `atan_bw` | `default` | accurate to |x| <= 0.23 | 2.23 | 0.767 | 0.896 | 0.23 | 1326 ±8% | 12649 |
| wh | bf16 | `atanh` | `default` | 2/32256 defects; rest 2.08 ULP | 2.08 | 0.995 | 0.081 | — | 656 | 25592 |
| wh | bf16 | `atanh_bw` | `default` | accurate to |x| <= 0.918 | 7.76 | 0.885 | 0.835 | 0.918 | 7167 | 2341 |
| wh | bf16 | `bias_gelu_` | `default` | worst sampled pairing | 179 | 86.3 | 0.996 | — | 794 | 21137 |
| wh | bf16 | `cbrt` | `default` | faithful; 846/65024 tie-breaks | 0.507 | 0.502 | 0.987 | 3.39e+38 | 424 ±12% | 39556 |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 409 ±8% | 41056 |
| wh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 463 ±9% | 36218 |
| wh | bf16 | `celu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2683 | 6252 |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 ±6% | 42582 |
| wh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 400 | 41941 |
| wh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 | 41889 |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2438 | 6882 |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 ±6% | 41611 |
| wh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 ±6% | 42435 |
| wh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±5% | 42051 |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2450 | 6847 |
| wh | bf16 | `composite_example` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21856 |
| wh | bf16 | `cos` | `default` | faithful; 2/37354 tie-breaks | 0.552 | 0.552 | 1 | 9.99e+05 | 411 ±7% | 40775 |
| wh | bf16 | `cos_bw` | `default` | 21281/65024 defects; rest 2.09e+42 ULP | 2.09e+42 | 2.59e+39 | 0.867 | 2.15e+06 | 1700 | 9869 |
| wh | bf16 | `cosh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 441 ±10% | 38026 |
| wh | bf16 | `cosh_bw` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 3147 | 5331 |
| wh | bf16 | `deg2rad` | `default` | 2/65024 defects; rest 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 404 ±6% | 41496 |
| wh | bf16 | `digamma` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 1160 | 14467 |
| wh | bf16 | `digamma_bw` | `default` | 263/64769 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.32e+04 | 0.664 | 1 | 11051 | 1518 |
| wh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 545 | 30804 |
| wh | bf16 | `div_bw` | `default` | bit-exact | 0.498 | 0 | 1 | — | 3564 | 4708 |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 1457 | 11514 |
| wh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 549 | 30532 |
| wh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 561 | 29908 |
| wh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 468 ±10% | 35881 |
| wh | bf16 | `elu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2718 | 6173 |
| wh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 560 | 29963 |
| wh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 400 ±46% | 41979 |
| wh | bf16 | `erf` | `default` | within 2 ULP | 1.06 | 0.771 | 0.729 | 3.39e+38 | 555 ±9% | 30203 |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.38 | 70.5 | 3.04 | 0.975 | 1.38 | 1407 | 11920 |
| wh | bf16 | `erfc` | `default` | 198/65024 unflushed; rest 3.19e+28 ULP | 3.19e+28 | 3.23e+24 | 0.969 | 2.5 | 821 | 20429 |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.38 | 70.5 | 3.04 | 0.975 | 1.38 | 2459 | 6822 |
| wh | bf16 | `erfinv` | `default` | 29424/32256 defects; rest 121 ULP | 121 | 6.34 | 0.391 | 1.31e-38 | 824 | 20357 |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.859 | 8.91 | 1.2 | 0.98 | 0.859 | 4650 | 3608 |
| wh | bf16 | `exp` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 417 ±8% | 40205 |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP | 6.08 | 3.05 | 0.314 | — | 396 ±6% | 42415 |
| wh | bf16 | `exp2` | `default` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 417 ±8% | 40202 |
| wh | bf16 | `exp2_bw` | `default` | 1/49537 defects; rest 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 1684 | 9961 |
| wh | bf16 | `exp_bw` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 1297 | 12931 |
| wh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 420 ±9% | 39904 |
| wh | bf16 | `expm1_bw` | `default` | accurate to |x| <= 33.2 | 34 | 0.695 | 0.984 | 33.2 | 1311 | 12793 |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 ±8% | 41886 |
| wh | bf16 | `floor_div` | `default` | worst sampled pairing | 64 | 42 | 0.996 | — | 2510 | 6685 |
| wh | bf16 | `fmod` | `default` | worst sampled pairing | 6.65e+35 | 7.8e+34 | 0.853 | — | 678 | 24748 |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 408 ±9% | 41116 |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 551 ±26% | 30447 |
| wh | bf16 | `gelu` | `default` | 86/65024 defects; rest 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 744 | 22550 |
| wh | bf16 | `gelu` | `fast_approx` | 198/65024 defects; rest 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 396 ±7% | 42355 |
| wh | bf16 | `gelu_bw` | `default` | faithful; 38/65024 tie-breaks | 0.939 | 0.681 | 0.999 | 3.39e+38 | 1734 | 9674 |
| wh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 405 ±8% | 41473 |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 560 | 29953 |
| wh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 | 42232 |
| wh | bf16 | `hardmish` | `default` | faithful; 2587/65024 tie-breaks | 1 | 0.913 | 0.961 | 3.39e+38 | 396 | 42345 |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 ±7% | 42307 |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1666 | 10069 |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP | 1 | 0.557 | 0.986 | 3.39e+38 | 410 ±6% | 40942 |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 2528 | 6637 |
| wh | bf16 | `hardswish` | `default` | 2/65024 defects; rest 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 418 ±9% | 40154 |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12 | 85.3 | 2.1 | 0.993 | 1.12 | 3519 | 4768 |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 390 | 42966 |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2145 | 7822 |
| wh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 400 ±8% | 41986 |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 | 41507 |
| wh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 | 42151 |
| wh | bf16 | `hypot` | `default` | 16384/65026 defects; rest 52.7 ULP | 52.7 | 1.64 | 0.992 | — | 636 | 26381 |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.6 | 255 | 56.3 | 0.952 | 13.6 | 413 ±8% | 40613 |
| wh | bf16 | `i0_bw` | `default` | 4/33904 defects; rest 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 2053 | 8173 |
| wh | bf16 | `i1` | `default` | 4/65024 defects; rest 0.858 ULP | 0.858 | 0.589 | 0.516 | 2.33e-38 | 1166 | 14386 |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±10% | 42026 |
| wh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 556 | 30176 |
| wh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 ±6% | 41749 |
| wh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 | 42481 |
| wh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 ±8% | 42103 |
| wh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±9% | 42063 |
| wh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±8% | 41997 |
| wh | bf16 | `l1_loss` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 550 ±6% | 30493 |
| wh | bf16 | `ldexp_` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 577 | 29078 |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 546 | 30733 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 389 | 43123 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithful; 15341/65024 tie-breaks | 0.96 | 0.741 | 0.761 | 3.39e+38 | 396 ±8% | 42407 |
| wh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 390 | 43003 |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 1808 | 9282 |
| wh | bf16 | `lerp` | `default` | worst sampled pairing | 1.77e+03 | 39.9 | 1 | — | 707 ±6% | 23747 |
| wh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±9% | 42001 |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.414 | 324 | 0.846 | 0.76 | 0.414 | 2457 | 6828 |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 2043 | 8214 |
| wh | bf16 | `log` | `default` | faithful; 63/32512 tie-breaks | 0.78 | 0.534 | 0.998 | 3.39e+38 | 408 ±8% | 41149 |
| wh | bf16 | `log10` | `default` | faithful; 58/32512 tie-breaks | 0.785 | 0.543 | 0.998 | 3.39e+38 | 419 ±12% | 39997 |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP | 1.74 | 0.803 | 0.508 | 3.69e+37 | 5258 | 3191 |
| wh | bf16 | `log1p` | `default` | faithful; 109/48640 tie-breaks | 0.931 | 0.57 | 0.998 | 3.39e+38 | 419 ±12% | 40082 |
| wh | bf16 | `log1p_bw` | `default` | within 2 ULP | 1.42 | 0.744 | 0.982 | 8.51e+37 | 5237 | 3203 |
| wh | bf16 | `log2` | `default` | faithful; 59/32512 tie-breaks | 0.766 | 0.542 | 0.998 | 3.39e+38 | 424 ±11% | 39570 |
| wh | bf16 | `log2_bw` | `default` | 116/64626 defects; rest 1.51 ULP | 1.51 | 0.847 | 0.57 | — | 5257 | 3192 |
| wh | bf16 | `log_bw` | `default` | bit-exact | 0.498 | 0 | 1 | 8.51e+37 | 3976 | 4219 |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.426 | 6.39 | 1.38 | 0.958 | 0.426 | 587 | 28575 |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP | 1.71 | 0.722 | 0.976 | 3.39e+38 | 5904 | 2842 |
| wh | bf16 | `logaddexp2_` | `default` | worst sampled pairing | 6.87e+10 | 2.05e+06 | 0.988 | — | 575 | 29182 |
| wh | bf16 | `logaddexp_` | `default` | worst sampled pairing | 2.22e+05 | 44.7 | 0.986 | — | 579 ±16% | 28978 |
| wh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 550 | 30528 |
| wh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 | 42472 |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 400 | 41958 |
| wh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 554 | 30308 |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 572 | 29307 |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395 | 64 | 1.67 | 0.985 | 0.395 | 880 | 19062 |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 6596 | 2544 |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 8206 | 2044 |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 558 | 30075 |
| wh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 ±6% | 42225 |
| wh | bf16 | `mac` | `default` | worst sampled pairing | 63.5 | 26.4 | 1 | — | 701 | 23944 |
| wh | bf16 | `mish` | `default` | 11/65024 defects; rest 1.49 ULP | 1.49 | 0.667 | 0.988 | 1.95e-38 | 585 ±12% | 28692 |
| wh | bf16 | `mse_loss` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 548 | 30620 |
| wh | bf16 | `multigammaln` | `default` | 11536/48328 defects; rest 724 ULP | 724 | 1.67 | 0.717 | 5.55e-17 | 12738 | 1317 |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 2.38e+06 | 293 | 0.249 | 5.55e-17 | 9556 | 1756 |
| wh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 532 ±36% | 31519 |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 545 ±8% | 30783 |
| wh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 564 | 29737 |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 | 42207 |
| wh | bf16 | `nextafter` | `default` | within 2 ULP | 1 | 1 | 0.00588 | — | 668 | 25113 |
| wh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 393 | 42649 |
| wh | bf16 | `polygamma` | `k=1` | 7/49922 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.43e+05 | 0.956 | 0.996 | 4820 | 3481 |
| wh | bf16 | `polygamma` | `k=2` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 6.81e+05 | 0.945 | 4.47 | 5749 | 2918 |
| wh | bf16 | `polygamma` | `k=4` | 142/49922 defects; rest 3.22e+09 ULP | 3.22e+09 | 1.09e+07 | 0.936 | 3.48 | 6006 | 2794 |
| wh | bf16 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 6.81e+05 | 0.945 | 4.47 | 12199 | 1375 |
| wh | bf16 | `pow` | `default` | worst sampled pairing | 4.86e+18 | 2.08e+14 | 0.982 | — | 1017 | 16492 |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2555 | 6568 |
| wh | bf16 | `prelu` | `weight=0.25` | 1/65024 defects; rest 0 ULP | 0 | 0 | 1 | 4.68e-38 | 403 | 41597 |
| wh | bf16 | `rad2deg` | `default` | faithful; 31760/63518 tie-breaks | 0.992 | 0.75 | 0.5 | 5.9e+36 | 393 ±5% | 42667 |
| wh | bf16 | `rdiv` | `value=2.0` | 258/64770 defects; rest 0.512 ULP | 0.512 | 0.507 | 0.984 | 8.47e+37 | 404 ±9% | 41530 |
| wh | bf16 | `reciprocal` | `default` | bit-exact | 0.498 | 0 | 1 | 8.51e+37 | 403 ±6% | 41601 |
| wh | bf16 | `reciprocal_bw` | `default` | 254/48386 defects; rest 1.23 ULP | 1.23 | 0.714 | 0.734 | 5.42e-20 | 3429 | 4892 |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 410 | 40891 |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 400 ±6% | 41979 |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2171 | 7729 |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1289 | 13020 |
| wh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 ±6% | 42365 |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 | 42283 |
| wh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 ±6% | 42331 |
| wh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 393 | 42638 |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 393 ±5% | 42714 |
| wh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 | 42522 |
| wh | bf16 | `rpow` | `exponent=0.5` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.975 | 3.39e+38 | 971 | 17274 |
| wh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 977 | 17168 |
| wh | bf16 | `rpow` | `exponent=2.0` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 972 | 17269 |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 421 ±7% | 39883 |
| wh | bf16 | `rsqrt_bw` | `default` | 75/21665 defects; rest 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 5408 | 3103 |
| wh | bf16 | `rsub` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 549 | 30582 |
| wh | bf16 | `rsub_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 546 | 30706 |
| wh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 484 ±14% | 34630 |
| wh | bf16 | `selu_bw` | `default` | 1/65024 defects; rest 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 3085 | 5438 |
| wh | bf16 | `sigmoid` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 455 ±12% | 36848 |
| wh | bf16 | `sigmoid_accurate` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 453 ±11% | 37012 |
| wh | bf16 | `sigmoid_bw` | `default` | 141/65024 defects; rest 127 ULP | 127 | 14.2 | 0.999 | 12.4 | 1148 | 14614 |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 ±8% | 42291 |
| wh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 390 ±5% | 43055 |
| wh | bf16 | `silu` | `default` | 13/65024 defects; rest 0.914 ULP | 0.914 | 0.585 | 0.993 | 2.33e-38 | 477 ±6% | 35163 |
| wh | bf16 | `silu_bw` | `default` | 9/65024 defects; rest 285 ULP | 285 | 1.31 | 0.979 | 0.867 | 3087 | 5436 |
| wh | bf16 | `sin` | `default` | faithful; 4/37354 tie-breaks | 0.552 | 0.526 | 1 | 9.99e+05 | 415 ±9% | 40458 |
| wh | bf16 | `sin_bw` | `default` | 21284/65024 defects; rest 2.98e+41 ULP | 2.98e+41 | 1.39e+39 | 0.857 | 1.07e+06 | 1314 | 12773 |
| wh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 482 ±7% | 34804 |
| wh | bf16 | `sinh_bw` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 3119 | 5378 |
| wh | bf16 | `softplus` | `default` | 526/65024 defects; rest 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 455 ±8% | 36857 |
| wh | bf16 | `softplus_bw` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 2758 | 6084 |
| wh | bf16 | `softshrink` | `default` | faithful; 3968/65024 tie-breaks | 1 | 0.948 | 0.941 | 3.39e+38 | 395 | 42473 |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2153 | 7792 |
| wh | bf16 | `softsign` | `default` | 510/65024 defects; rest 1 ULP | 1 | 0.641 | 0.907 | 8.51e+37 | 435 ±9% | 38600 |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391 | 79 | 3.74 | 0.887 | 0.00391 | 1319 | 12722 |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 411 ±9% | 40859 |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP | 1.14 | 0.699 | 0.723 | 3.39e+38 | 6531 | 2569 |
| wh | bf16 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 402 ±8% | 41772 |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1280 | 13102 |
| wh | bf16 | `squared_difference` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 530 | 31671 |
| wh | bf16 | `squared_difference_` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 542 | 30934 |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 550 | 30497 |
| wh | bf16 | `subtract` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 534 | 31407 |
| wh | bf16 | `subtract_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 551 ±61% | 30445 |
| wh | bf16 | `swish` | `default` | 13/65024 defects; rest 0.914 ULP | 0.914 | 0.585 | 0.993 | 2.33e-38 | 472 ±8% | 35580 |
| wh | bf16 | `tan` | `default` | faithful; 48/37354 tie-breaks | 0.551 | 0.5 | 0.999 | 9.99e+05 | 538 | 31159 |
| wh | bf16 | `tan_bw` | `default` | 12184/65024 defects; rest 3.57e+40 ULP | 3.57e+40 | 3.8e+37 | 0.69 | 940 | 2164 | 7752 |
| wh | bf16 | `tanh` | `default` | 2/65024 defects; rest 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 397 | 42228 |
| wh | bf16 | `tanh_bw` | `default` | within 2 ULP | 1.16 | 0.525 | 0.997 | 3.39e+38 | 1493 | 11238 |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08 | 63.5 | 9.38 | 0.981 | 1.35e-08 | 422 ±9% | 39718 |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09 | 63 | 6.1 | 0.967 | 7.45e-09 | 1691 | 9922 |
| wh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 | 42393 |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 ±6% | 41854 |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1647 | 10188 |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 ±8% | 41775 |
| wh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 681 | 24637 |
| wh | bf16 | `xielu` | `default` | 1/56847 defects; rest 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 1000 | 16784 |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 776 | 21630 |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2483 | 6756 |
| wh | fp32 | `acos` | `default` | within 2 ULP | 1.55 | 0.865 | 0.923 | 1 | 1084 | 15480 |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 14944 | 1123 |
| wh | fp32 | `acosh` | `default` | never within 2 ULP | 2.76 | 0.764 | 0.782 | — | 1440 ±9% | 11648 |
| wh | fp32 | `acosh_bw` | `default` | 15874/64514 defects; rest 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 17141 | 979 |
| wh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 977 | 17178 |
| wh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 993 | 16901 |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 1007 | 16667 |
| wh | fp32 | `addcdiv` | `default` | worst sampled pairing | 2.19e+12 | 8.82e+07 | 0.901 | — | 1367 | 12270 |
| wh | fp32 | `addcmul` | `default` | worst sampled pairing | 2.13e+06 | 1.59e+04 | 0.985 | — | 1363 | 12309 |
| wh | fp32 | `asin` | `default` | within 2 ULP | 1.77 | 0.795 | 0.985 | 1 | 1062 | 15800 |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 14764 | 1136 |
| wh | fp32 | `asinh` | `default` | within 2 ULP | 1.72 | 0.774 | 0.881 | 3.39e+38 | 1708 | 9825 |
| wh | fp32 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 2605 ±8% | 6440 |
| wh | fp32 | `atan` | `default` | accurate to |x| <= 0.902 | 2.34 | 0.807 | 0.955 | 0.902 | 911 | 18413 |
| wh | fp32 | `atan2` | `default` | worst sampled pairing | 1.32e+07 | 1.26e+05 | 0.882 | — | 1027 | 16336 |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 1.01 | 2.8 | 1.23 | 0.874 | 1.01 | 2550 | 6580 |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 4.28e-07 | 3.11 | 1.55 | 0.91 | 4.28e-07 | 1284 ±9% | 13068 |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.678 | 2.05e+03 | 1.55 | 0.891 | 0.678 | 13852 | 1211 |
| wh | fp32 | `bias_gelu_` | `default` | worst sampled pairing | 1.22e+08 | 1.48e+07 | 0.964 | — | 1057 | 15870 |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38 | 2.55 | 1.76 | 0.521 | 2.26e-38 | 850 | 19733 |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21760 |
| wh | fp32 | `celu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 910 | 18435 |
| wh | fp32 | `celu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5456 | 3075 |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21838 |
| wh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21881 |
| wh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21897 |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4599 | 3648 |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21908 |
| wh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21912 |
| wh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21889 |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4594 | 3652 |
| wh | fp32 | `composite_example` | `default` | never within 2 ULP | 6.55e+04 | 6.55e+04 | 1.53e-05 | — | 1513 | 11088 |
| wh | fp32 | `cos` | `default` | accurate to |x| <= 92.4 | 1.34e+08 | 1.33e+05 | 0.927 | 92.4 | 843 | 19896 |
| wh | fp32 | `cos_bw` | `default` | 18643/65024 defects; rest 8.59e+51 ULP | 8.59e+51 | 3.23e+48 | 0.841 | 28 | 3287 | 5104 |
| wh | fp32 | `cosh` | `default` | within 2 ULP | 1.35 | 0.798 | 0.98 | 89.1 | 980 | 17123 |
| wh | fp32 | `cosh_bw` | `default` | accurate to |x| <= 0.0155 | 2.21 | 1.17 | 0.943 | 0.0155 | 6253 | 2683 |
| wh | fp32 | `deg2rad` | `default` | faithful; 63542/65024 tie-breaks | 0.63 | 0.595 | 0.906 | 3.4e+38 | 767 | 21886 |
| wh | fp32 | `digamma` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00566 | — | 2615 | 6415 |
| wh | fp32 | `digamma_bw` | `default` | 255/64769 defects; rest 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.429 | 5.4e-20 | 17237 | 973 |
| wh | fp32 | `div` | `default` | within 2 ULP | 1.78 | 1.22 | 0.862 | — | 1008 | 16646 |
| wh | fp32 | `div_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.84 | 0.84 | 0.902 | — | 6884 | 2437 |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP | 1.79 | 1.52 | 0.714 | — | 3459 | 4851 |
| wh | fp32 | `divide` | `default` | within 2 ULP | 1.78 | 1.22 | 0.862 | — | 994 | 16884 |
| wh | fp32 | `divide_` | `default` | within 2 ULP | 1.78 | 1.22 | 0.862 | — | 1008 | 16650 |
| wh | fp32 | `elu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 911 | 18413 |
| wh | fp32 | `elu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5425 | 3093 |
| wh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 1017 | 16491 |
| wh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21890 |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334 | 6.48 | 1.35 | 0.671 | 0.000334 | 1184 ±8% | 14166 |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 2511 | 6680 |
| wh | fp32 | `erfc` | `default` | 198/65024 unflushed; rest 2.1e+33 ULP | 2.1e+33 | 1.71e+29 | 0.33 | — | 1321 ±6% | 12699 |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 4865 | 3448 |
| wh | fp32 | `erfinv` | `default` | 29420/32256 defects; rest 7.96e+06 ULP | 7.96e+06 | 2.59e+05 | 0.000202 | 1.32e-38 | 1335 ±6% | 12572 |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456 | 4.98e+05 | 1.32e+03 | 0.914 | 0.000456 | 7955 | 2109 |
| wh | fp32 | `exp` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 892 | 18816 |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP | 3.79e+05 | 2e+05 | 0.309 | — | 769 | 21803 |
| wh | fp32 | `exp2` | `default` | faithful; 6044/49536 tie-breaks | 0.965 | 0.626 | 0.992 | 3.39e+38 | 831 | 20192 |
| wh | fp32 | `exp2_bw` | `default` | 1/49537 defects; rest 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 3305 | 5076 |
| wh | fp32 | `exp_bw` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 2588 | 6482 |
| wh | fp32 | `expm1` | `default` | faithful; 5257/49458 tie-breaks | 0.997 | 0.565 | 0.997 | 3.39e+38 | 1015 | 16522 |
| wh | fp32 | `expm1_bw` | `default` | accurate to |x| <= 22 | 4.19e+06 | 3.83e+03 | 0.995 | 22 | 2595 | 6466 |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 779 ±7% | 21543 |
| wh | fp32 | `floor_div` | `default` | worst sampled pairing | 4.19e+06 | 1.77e+03 | 0.955 | — | 4787 | 3505 |
| wh | fp32 | `fmod` | `default` | worst sampled pairing | 1.43e+45 | 1.86e+41 | 0.762 | — | 1016 | 16518 |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21775 |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 993 | 16893 |
| wh | fp32 | `gelu` | `default` | 83/65024 defects; rest 1.51e+08 ULP | 1.51e+08 | 1.55e+05 | 0.963 | 0.253 | 1356 ±9% | 12372 |
| wh | fp32 | `gelu` | `fast_approx` | 197/65024 defects; rest 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 766 | 21888 |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0298 | 1.47e+08 | 2.3e+04 | 0.929 | 0.0298 | 2056 | 8159 |
| wh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21842 |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1010 | 16613 |
| wh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21807 |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21873 |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21939 |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3226 | 5200 |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25 | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 772 | 21745 |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 5019 | 3342 |
| wh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36 | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 810 | 20724 |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133 | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 7084 | 2368 |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21870 |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4244 | 3953 |
| wh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21840 |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21909 |
| wh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 764 | 21963 |
| wh | fp32 | `hypot` | `default` | 16384/65024 defects; rest 3.44e+06 ULP | 3.44e+06 | 3.28e+04 | 0.965 | — | 1020 | 16454 |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.45 | 1.68e+07 | 2.19e+06 | 0.959 | 2.45 | 865 | 19394 |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00626 | 1.56e+07 | 4.11e+04 | 0.929 | 0.00626 | 3425 | 4899 |
| wh | fp32 | `i1` | `default` | accurate to |x| <= 0.00626 | 8.02 | 0.722 | 0.483 | 0.00626 | 1710 | 9812 |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21917 |
| wh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 1001 | 16755 |
| wh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21785 |
| wh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 764 | 21948 |
| wh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21813 |
| wh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21799 |
| wh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21675 |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 995 | 16867 |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 1026 | 16359 |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 997 | 16826 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21882 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithful; 31672/65024 tie-breaks | 0.84 | 0.747 | 0.868 | 3.39e+38 | 766 | 21894 |
| wh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 763 | 21983 |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 3601 | 4659 |
| wh | fp32 | `lerp` | `default` | worst sampled pairing | 1.46e+08 | 9.53e+04 | 0.962 | — | 1362 | 12322 |
| wh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 ±7% | 21796 |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181 | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 3775 | 4444 |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00566 | — | 4317 | 3887 |
| wh | fp32 | `log` | `default` | faithful; 32512/32512 tie-breaks | 0.961 | 0.562 | 0.952 | 3.39e+38 | 872 | 19247 |
| wh | fp32 | `log10` | `default` | accurate to |x| <= 0.336 | 2.13 | 1.36 | 0.641 | 0.336 | 877 | 19130 |
| wh | fp32 | `log10_bw` | `default` | accurate to |x| <= 2.04e-38 | 2.1 | 1.41 | 0.674 | 2.04e-38 | 10241 | 1638 |
| wh | fp32 | `log1p` | `default` | faithful; 20111/48640 tie-breaks | 0.967 | 0.558 | 0.982 | 3.39e+38 | 877 | 19136 |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP | 1.89 | 0.769 | 0.913 | 8.51e+37 | 10255 | 1636 |
| wh | fp32 | `log2` | `default` | accurate to |x| <= 0.704 | 2.43 | 0.517 | 0.993 | 0.704 | 890 | 18848 |
| wh | fp32 | `log2_bw` | `default` | 112/64626 defects; rest 1.9 ULP | 1.9 | 1.3 | 0.701 | — | 10243 | 1638 |
| wh | fp32 | `log_bw` | `default` | faithful; 64512/64514 tie-breaks | 0.892 | 0.714 | 0.902 | 8.51e+37 | 7780 | 2157 |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP | 4.4e+05 | 1.82e+04 | 0.481 | — | 1106 | 15171 |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.00164 | 2.95 | 1.28 | 0.954 | 0.00164 | 11488 | 1460 |
| wh | fp32 | `logaddexp2_` | `default` | worst sampled pairing | 1.06e+07 | 1.23e+05 | 0.959 | — | 1053 | 15936 |
| wh | fp32 | `logaddexp_` | `default` | worst sampled pairing | 8e+06 | 9.77e+04 | 0.975 | — | 1052 | 15947 |
| wh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 1007 | 16655 |
| wh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21903 |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 768 | 21842 |
| wh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 1010 | 16610 |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 1039 | 16154 |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.266 | 4.19e+06 | 261 | 0.94 | 0.266 | 1262 ±9% | 13293 |
| wh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.98e-08 | 2.36 | 0.839 | 0.857 | 2.98e-08 | 13030 | 1288 |
| wh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.98e-08 | 2.36 | 0.839 | 0.857 | 2.98e-08 | 16432 | 1021 |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1015 | 16531 |
| wh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21850 |
| wh | fp32 | `mac` | `default` | worst sampled pairing | 4.22e+06 | 7.31e+05 | 0.974 | — | 1360 | 12340 |
| wh | fp32 | `mish` | `default` | 9/65024 defects; rest 7 ULP | 7 | 2.49 | 0.954 | 8.61e-06 | 1166 ±8% | 14383 |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 995 | 16856 |
| wh | fp32 | `multigammaln` | `default` | 7422/50376 defects; rest 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.59e-17 | 20931 | 802 |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 1.36e+15 | 2.04e+11 | 0.0104 | 5.55e-17 | 20105 | 834 |
| wh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 977 | 17164 |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 991 | 16933 |
| wh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 1010 | 16609 |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21939 |
| wh | fp32 | `nextafter` | `default` | faithful; 65024/65024 tie-breaks | 1 | 1 | 4.06e-05 | — | 1027 | 16338 |
| wh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21891 |
| wh | fp32 | `polygamma` | `k=1` | accurate to |x| <= 5.4e-20 | 7.91e+33 | 4.68e+29 | 0.617 | 5.4e-20 | 5516 | 3042 |
| wh | fp32 | `polygamma` | `k=2` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.199 | 1.79e-13 | 5802 | 2892 |
| wh | fp32 | `polygamma` | `k=4` | 141/49922 defects; rest 3.77e+25 ULP | 3.77e+25 | 6.51e+21 | 0.191 | 3.68e-08 | 6192 | 2709 |
| wh | fp32 | `polygamma_bw` | `n=1` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.199 | 1.79e-13 | 17876 | 938 |
| wh | fp32 | `pow` | `default` | worst sampled pairing | 324 | 2.42 | 0.993 | — | 2087 | 8038 |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 4953 | 3387 |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21732 |
| wh | fp32 | `rad2deg` | `default` | faithful; 63518/63518 tie-breaks | 0.696 | 0.643 | 0.858 | 5.93e+36 | 766 | 21890 |
| wh | fp32 | `rdiv` | `value=2.0` | 256/64770 defects; rest 0.892 ULP | 0.892 | 0.714 | 0.902 | 8.51e+37 | 796 | 21069 |
| wh | fp32 | `reciprocal` | `default` | faithful; 64512/64514 tie-breaks | 0.892 | 0.714 | 0.902 | 8.51e+37 | 788 | 21297 |
| wh | fp32 | `reciprocal_bw` | `default` | 254/48386 defects; rest 1.84 ULP | 1.84 | 1.22 | 0.726 | 5.42e-20 | 6346 | 2644 |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21885 |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21924 |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4220 | 3975 |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2475 | 6778 |
| wh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21777 |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 768 | 21853 |
| wh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21882 |
| wh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21783 |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21812 |
| wh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21804 |
| wh | fp32 | `rpow` | `exponent=0.5` | faithful; 5624/49536 tie-breaks | 0.879 | 0.609 | 0.995 | 3.39e+38 | 1941 | 8642 |
| wh | fp32 | `rpow` | `exponent=1.0` | 1536/49536 defects; rest 0 ULP | 0 | 0 | 1 | 8.28e+34 | 1944 | 8630 |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536/49536 defects; rest 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 1939 | 8652 |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP | 1.12 | 0.834 | 0.864 | 3.39e+38 | 848 | 19781 |
| wh | fp32 | `rsqrt_bw` | `default` | 75/21666 defects; rest 6.76 ULP | 6.76 | 4.26 | 0.3 | — | 10210 | 1643 |
| wh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1004 | 16702 |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 991 | 16923 |
| wh | fp32 | `selu` | `default` | never within 2 ULP | 51.3 | 26.3 | 0 | — | 932 | 18002 |
| wh | fp32 | `selu_bw` | `default` | 1/65024 defects; rest 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 6197 | 2708 |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 1071 | 15667 |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 1055 | 15904 |
| wh | fp32 | `sigmoid_bw` | `default` | 140/65024 defects; rest 8.39e+06 ULP | 8.39e+06 | 2.71e+04 | 0.975 | 0.447 | 1769 | 9484 |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21935 |
| wh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 763 | 21991 |
| wh | fp32 | `silu` | `default` | 9/65024 defects; rest 2.94 ULP | 2.94 | 1.55 | 0.95 | 0.000121 | 1079 | 15543 |
| wh | fp32 | `silu_bw` | `default` | 9/65024 defects; rest 5.72e+06 ULP | 5.72e+06 | 841 | 0.937 | 2.98e-07 | 6206 | 2703 |
| wh | fp32 | `sin` | `default` | accurate to |x| <= 28 | 3.36e+07 | 3.07e+04 | 0.96 | 28 | 823 | 20377 |
| wh | fp32 | `sin_bw` | `default` | 18430/65024 defects; rest 1.24e+52 ULP | 1.24e+52 | 3.44e+48 | 0.808 | 92.4 | 2549 | 6581 |
| wh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155 | 2.21 | 1.17 | 0.943 | 0.0155 | 1071 | 15672 |
| wh | fp32 | `sinh_bw` | `default` | within 2 ULP | 1.35 | 0.798 | 0.98 | 89.1 | 6161 | 2723 |
| wh | fp32 | `softplus` | `default` | 5/65024 defects; rest 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 1375 ±15% | 12198 |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 5631 | 2979 |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 765 | 21934 |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4243 | 3954 |
| wh | fp32 | `softsign` | `default` | 510/65024 defects; rest 2.66 ULP | 2.66 | 0.897 | 0.92 | 0.000462 | 811 | 20699 |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07 | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 2562 | 6549 |
| wh | fp32 | `sqrt` | `default` | faithful; 32512/32512 tie-breaks | 0.867 | 0.827 | 0.869 | 3.39e+38 | 841 | 19938 |
| wh | fp32 | `sqrt_bw` | `default` | never within 2 ULP | 2.24 | 1.44 | 0.702 | — | 12669 | 1324 |
| wh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 762 | 22016 |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2485 ±6% | 6751 |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 975 | 17200 |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 994 | 16880 |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 1014 ±27% | 16554 |
| wh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 977 | 17175 |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 993 | 16889 |
| wh | fp32 | `swish` | `default` | 9/65024 defects; rest 2.94 ULP | 2.94 | 1.55 | 0.95 | 0.000121 | 1077 | 15579 |
| wh | fp32 | `tan` | `default` | accurate to |x| <= 3.92 | 2.26e+08 | 2.32e+05 | 0.944 | 3.92 | 1161 ±6% | 14456 |
| wh | fp32 | `tan_bw` | `default` | 20623/65024 defects; rest 2.85e+45 ULP | 2.85e+45 | 4.74e+44 | 0.87 | 0.882 | 4368 | 3841 |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000439 | 2.79 | 1.49 | 0.979 | 0.000439 | 927 | 18094 |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP | 6.59e+04 | 7.51e+03 | 0.482 | — | 1780 | 9424 |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08 | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 1354 ±11% | 12393 |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09 | 4.19e+06 | 1.01e+05 | 0.901 | 7.42e-09 | 3390 | 4949 |
| wh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21935 |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21678 |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3239 | 5179 |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21747 |
| wh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 1350 | 12427 |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13 | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 1413 ±6% | 11876 |
