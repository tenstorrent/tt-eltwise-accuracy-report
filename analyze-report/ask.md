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
| bh | bf16 | `d80d9482a41` | 0.81.1 | 66444ba711c0 |
| bh | fp32 | `d80d9482a41` | 0.81.1 | 66444ba711c0 |
| wh | bf16 | `d80d9482a41` | 0.81.1 | e0dceb4541af |
| wh | fp32 | `d80d9482a41` | 0.81.1 | e0dceb4541af |

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

## Results — 839 variants

| Arch | Dtype | Op | Parameters | Verdict | Max ULP | Mean ULP | Rounded | Usable to | µs | Melem/s |
|---|---|---|---|---|---|---|---|---|---|---|
| bh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 | 86846 |
| bh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 598 ±6% | 28059 |
| bh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 294 ±11% | 57128 |
| bh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.744 | 0.996 | 0.949 | 3536 | 4744 |
| bh | bf16 | `acosh` | `default` | within 2 ULP | 1.41 | 0.596 | 0.973 | 3.39e+38 | 366 | 45788 |
| bh | bf16 | `acosh_bw` | `default` | 15874/64514 defects; rest 3.07 ULP | 3.07 | 0.64 | 0.884 | 1.03 | 4053 | 4140 |
| bh | bf16 | `add` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 253 | 66410 |
| bh | bf16 | `add_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 242 | 69386 |
| bh | bf16 | `addalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 247 | 68027 |
| bh | bf16 | `addcdiv` | `default` | worst sampled pairing | 2.11e+06 | 2.35e+03 | 0.999 | — | 327 | 51271 |
| bh | bf16 | `addcmul` | `default` | worst sampled pairing | 126 | 19.4 | 1 | — | 330 | 50903 |
| bh | bf16 | `asin` | `default` | 2/32258 defects; rest 0.499 ULP | 0.499 | 0 | 1 | — | 282 | 59569 |
| bh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.702 | 0.992 | 0.949 | 3446 | 4869 |
| bh | bf16 | `asinh` | `default` | within 2 ULP | 1.32 | 0.615 | 0.991 | 3.39e+38 | 501 | 33490 |
| bh | bf16 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.2 ULP | 1.2 | 0.638 | 0.954 | 1.84e+19 | 583 ±37% | 28774 |
| bh | bf16 | `atan` | `default` | 2/65024 defects; rest 0.527 ULP | 0.527 | 0.513 | 1 | — | 209 | 80165 |
| bh | bf16 | `atan2` | `default` | worst sampled pairing | 200 | 3.56 | 0.901 | — | 261 | 64357 |
| bh | bf16 | `atan_bw` | `default` | 2/48386 defects; rest 2.23 ULP | 2.23 | 0.767 | 0.896 | 0.23 | 639 ±16% | 26242 |
| bh | bf16 | `atanh` | `default` | 2/32256 defects; rest 2.08 ULP | 2.08 | 0.995 | 0.175 | — | 276 ±9% | 60751 |
| bh | bf16 | `atanh_bw` | `default` | 2/48386 defects; rest 7.76 ULP | 7.76 | 0.885 | 0.835 | 0.918 | 3846 | 4362 |
| bh | bf16 | `bias_gelu_` | `default` | worst sampled pairing | 179 | 86.3 | 0.996 | — | 248 | 67735 |
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
| bh | bf16 | `cosh_bw` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 3080 ±8% | 5447 |
| bh | bf16 | `deg2rad` | `default` | 2/65024 defects; rest 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 195 | 86254 |
| bh | bf16 | `digamma` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 399 ±12% | 42078 |
| bh | bf16 | `digamma_bw` | `default` | 263/64769 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.36e+04 | 0.674 | 1 | 3179 | 5277 |
| bh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 253 ±19% | 66203 |
| bh | bf16 | `div_bw` | `default` | bit-exact | 0.498 | 0 | 1 | — | 4894 | 3428 |
| bh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 666 | 25187 |
| bh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 261 ±28% | 64200 |
| bh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 255 ±9% | 65891 |
| bh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 192 | 87194 |
| bh | bf16 | `elu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1285 | 13052 |
| bh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 246 | 68091 |
| bh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 ±16% | 85580 |
| bh | bf16 | `erf` | `default` | within 2 ULP | 1.06 | 0.771 | 0.729 | 3.39e+38 | 193 ±12% | 86780 |
| bh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.38 | 70.5 | 3.04 | 0.975 | 1.38 | 1144 | 14661 |
| bh | bf16 | `erfc` | `default` | 198/65024 unflushed; rest 3.2e+28 ULP | 3.2e+28 | 3.23e+24 | 0.949 | 2.5 | 319 | 52525 |
| bh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.38 | 70.5 | 3.04 | 0.975 | 1.38 | 1140 | 14722 |
| bh | bf16 | `erfinv` | `default` | 29424/32256 defects; rest 121 ULP | 121 | 6.34 | 0.391 | 1.31e-38 | 356 ±29% | 47085 |
| bh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.859 | 8.91 | 1.2 | 0.98 | 0.859 | 3621 | 4634 |
| bh | bf16 | `exp` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 204 ±8% | 82440 |
| bh | bf16 | `exp` | `fast_approx` | never within 2 ULP | 6.08 | 3.05 | 0.314 | — | 198 | 84532 |
| bh | bf16 | `exp2` | `default` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 231 ±13% | 72519 |
| bh | bf16 | `exp2_bw` | `default` | 1/49537 defects; rest 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 820 | 20458 |
| bh | bf16 | `exp_bw` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 600 ±6% | 27973 |
| bh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 251 ±7% | 66798 |
| bh | bf16 | `expm1_bw` | `default` | accurate to |x| <= 33.2 | 34 | 0.695 | 0.984 | 33.2 | 814 | 20605 |
| bh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 ±21% | 86140 |
| bh | bf16 | `floor_div` | `default` | worst sampled pairing | 64 | 42 | 0.996 | — | 1178 | 14236 |
| bh | bf16 | `fmod` | `default` | worst sampled pairing | 9.14e+35 | 8.94e+34 | 0.673 | — | 261 ±5% | 64206 |
| bh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 199 | 84219 |
| bh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 246 ±18% | 68129 |
| bh | bf16 | `gelu` | `default` | 86/65024 defects; rest 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 351 ±116% | 47761 |
| bh | bf16 | `gelu` | `fast_approx` | 198/65024 defects; rest 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 186 ±8% | 90218 |
| bh | bf16 | `gelu_bw` | `default` | faithful; 38/65024 tie-breaks | 0.939 | 0.681 | 0.999 | 3.39e+38 | 706 | 23778 |
| bh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88311 |
| bh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 243 | 68902 |
| bh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 ±9% | 84824 |
| bh | bf16 | `hardmish` | `default` | faithful; 2587/65024 tie-breaks | 1 | 0.913 | 0.961 | 3.39e+38 | 194 ±11% | 86593 |
| bh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 ±8% | 87027 |
| bh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21718 |
| bh | bf16 | `hardsigmoid` | `default` | within 2 ULP | 1 | 0.557 | 0.986 | 3.39e+38 | 191 ±13% | 87925 |
| bh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 1183 | 14188 |
| bh | bf16 | `hardswish` | `default` | 2/65024 defects; rest 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 212 ±5% | 79254 |
| bh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12 | 85.3 | 2.1 | 0.993 | 1.12 | 1655 | 10138 |
| bh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 200 ±6% | 83867 |
| bh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 988 | 16974 |
| bh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 203 ±646% | 82516 |
| bh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85792 |
| bh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 199 ±5% | 84237 |
| bh | bf16 | `hypot` | `default` | 16384/65026 defects; rest 52.7 ULP | 52.7 | 1.64 | 0.992 | — | 271 ±10% | 61844 |
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
| bh | bf16 | `ldexp_` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 251 ±6% | 66914 |
| bh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 244 ±15% | 68641 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 | 86042 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithful; 15341/65024 tie-breaks | 0.96 | 0.741 | 0.761 | 3.39e+38 | 193 ±1010% | 86884 |
| bh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 | 87254 |
| bh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 883 | 18998 |
| bh | bf16 | `lerp` | `default` | worst sampled pairing | 1.77e+03 | 39.9 | 1 | — | 328 ±58% | 51077 |
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
| bh | bf16 | `logaddexp2_` | `default` | worst sampled pairing | 6.87e+10 | 2.05e+06 | 0.988 | — | 385 ±9% | 43559 |
| bh | bf16 | `logaddexp_` | `default` | worst sampled pairing | 2.22e+05 | 44.7 | 0.986 | — | 307 | 54690 |
| bh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 258 ±30% | 65009 |
| bh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 ±5% | 86403 |
| bh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 194 ±11% | 86272 |
| bh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 264 ±17% | 63518 |
| bh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 254 ±7% | 66041 |
| bh | bf16 | `logit` | `default` | accurate to |x| <= 0.395 | 64 | 1.67 | 0.985 | 0.395 | 337 | 49856 |
| bh | bf16 | `logit_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3204 | 5237 |
| bh | bf16 | `logiteps_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 3822 | 4390 |
| bh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 253 | 66195 |
| bh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 | 86051 |
| bh | bf16 | `mac` | `default` | worst sampled pairing | 63.5 | 26.4 | 1 | — | 331 | 50713 |
| bh | bf16 | `mish` | `default` | 11/65024 defects; rest 1.49 ULP | 1.49 | 0.666 | 0.988 | 1.95e-38 | 268 ±7% | 62685 |
| bh | bf16 | `mse_loss` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 254 ±12% | 65975 |
| bh | bf16 | `multigammaln` | `default` | 11536/48328 defects; rest 724 ULP | 724 | 1.67 | 0.717 | 5.59e-17 | 4668 | 3594 |
| bh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 2.38e+06 | 293 | 0.249 | 5.55e-17 | 3815 | 4398 |
| bh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 248 ±7% | 67666 |
| bh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 246 ±6% | 68279 |
| bh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 259 ±45% | 64729 |
| bh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 ±8% | 86983 |
| bh | bf16 | `nextafter` | `default` | within 2 ULP | 1 | 1 | 0.00585 | — | 1457 | 11513 |
| bh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±11% | 87241 |
| bh | bf16 | `polygamma` | `k=1` | 7/49922 defects; rest 1.02e+08 ULP | 1.02e+08 | 2.13e+05 | 0.97 | 0.996 | 347 | 48288 |
| bh | bf16 | `polygamma` | `k=2` | 75/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 409 | 41002 |
| bh | bf16 | `polygamma` | `k=4` | 142/49922 defects; rest 3.22e+09 ULP | 3.22e+09 | 1.66e+07 | 0.958 | 3.5 | 443 | 37848 |
| bh | bf16 | `polygamma_bw` | `n=1` | 1/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 3236 | 5185 |
| bh | bf16 | `pow` | `default` | worst sampled pairing | 4.86e+18 | 2.08e+14 | 0.982 | — | 407 | 41207 |
| bh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1188 | 14128 |
| bh | bf16 | `prelu` | `weight=0.25` | 1/65024 defects; rest 0 ULP | 0 | 0 | 1 | 4.68e-38 | 195 | 86194 |
| bh | bf16 | `rad2deg` | `default` | faithful; 31760/63518 tie-breaks | 0.992 | 0.75 | 0.5 | 5.9e+36 | 198 ±13% | 84644 |
| bh | bf16 | `rdiv` | `value=2.0` | 258/64770 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 195 ±16% | 86062 |
| bh | bf16 | `reciprocal` | `default` | 2/64514 defects; rest 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 194 ±18% | 86429 |
| bh | bf16 | `reciprocal_bw` | `default` | 256/48386 defects; rest 1.23 ULP | 1.23 | 0.714 | 0.734 | 5.42e-20 | 2392 | 7015 |
| bh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 ±7% | 84759 |
| bh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85475 |
| bh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1850 ±6% | 9071 |
| bh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 603 | 27822 |
| bh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 | 87446 |
| bh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 200 | 84036 |
| bh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 197 ±12% | 85279 |
| bh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 | 84905 |
| bh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 87992 |
| bh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 ±9% | 85498 |
| bh | bf16 | `rpow` | `exponent=0.5` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.975 | 3.39e+38 | 384 ±20% | 43686 |
| bh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 387 ±14% | 43325 |
| bh | bf16 | `rpow` | `exponent=2.0` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 380 ±7% | 44184 |
| bh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 248 ±6% | 67538 |
| bh | bf16 | `rsqrt_bw` | `default` | 75/21665 defects; rest 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 2822 | 5946 |
| bh | bf16 | `rsub` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 248 | 67568 |
| bh | bf16 | `rsub_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 255 ±84% | 65857 |
| bh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 208 ±250% | 80723 |
| bh | bf16 | `selu_bw` | `default` | 1/65024 defects; rest 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 1470 | 11410 |
| bh | bf16 | `sigmoid` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 251 ±7% | 66835 |
| bh | bf16 | `sigmoid_accurate` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 253 ±5% | 66316 |
| bh | bf16 | `sigmoid_bw` | `default` | 141/65024 defects; rest 127 ULP | 127 | 12.7 | 0.999 | 12.4 | 1053 | 15935 |
| bh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 ±7% | 85897 |
| bh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 ±5% | 86304 |
| bh | bf16 | `silu` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 168 ±14% | 99808 |
| bh | bf16 | `silu_bw` | `default` | 9/65024 defects; rest 285 ULP | 285 | 1.3 | 0.979 | 0.867 | 1449 | 11578 |
| bh | bf16 | `sin` | `default` | faithful; 4/37354 tie-breaks | 0.552 | 0.526 | 1 | 9.99e+05 | 235 ±5% | 71320 |
| bh | bf16 | `sin_bw` | `default` | 21284/65024 defects; rest 2.98e+41 ULP | 2.98e+41 | 1.39e+39 | 0.857 | 1.07e+06 | 678 | 24745 |
| bh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 186 | 90191 |
| bh | bf16 | `sinh_bw` | `default` | faithful; 4/33894 tie-breaks | 0.501 | 0.501 | 1 | 89 | 2806 | 5979 |
| bh | bf16 | `softcap` | `beta=50.0` | 1426/65024 defects; rest 0.718 ULP | 0.718 | 0.574 | 0.997 | — | 236 | 71143 |
| bh | bf16 | `softplus` | `default` | 526/65024 defects; rest 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 192 ±7% | 87300 |
| bh | bf16 | `softplus_bw` | `default` | faithful; 482/65024 tie-breaks | 0.873 | 0.522 | 0.994 | 3.39e+38 | 2015 | 8328 |
| bh | bf16 | `softshrink` | `default` | faithful; 3968/65024 tie-breaks | 1 | 0.948 | 0.941 | 3.39e+38 | 203 | 82652 |
| bh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 988 | 16984 |
| bh | bf16 | `softsign` | `default` | 512/65024 defects; rest 1 ULP | 1 | 0.605 | 0.907 | 8.47e+37 | 196 | 85457 |
| bh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391 | 79 | 3.74 | 0.887 | 0.00391 | 622 | 26977 |
| bh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 233 ±11% | 72068 |
| bh | bf16 | `sqrt_bw` | `default` | within 2 ULP | 1.14 | 0.699 | 0.723 | 3.39e+38 | 3022 | 5552 |
| bh | bf16 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 196 | 85576 |
| bh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 613 ±25% | 27381 |
| bh | bf16 | `squared_difference` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 248 ±7% | 67751 |
| bh | bf16 | `squared_difference_` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 245 | 68584 |
| bh | bf16 | `subalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 249 | 67352 |
| bh | bf16 | `subtract` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 249 ±5% | 67366 |
| bh | bf16 | `subtract_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 245 ±15% | 68454 |
| bh | bf16 | `swish` | `default` | 13/65024 defects; rest 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 176 ±7% | 95084 |
| bh | bf16 | `tan` | `default` | faithful; 30/37354 tie-breaks | 0.551 | 0.509 | 0.999 | 9.99e+05 | 213 ±10% | 78910 |
| bh | bf16 | `tan_bw` | `default` | 22782/65024 defects; rest 3.57e+40 ULP | 3.57e+40 | 1e+38 | 0.853 | 940 | 970 | 17300 |
| bh | bf16 | `tanh` | `default` | 2/65024 defects; rest 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 198 | 84781 |
| bh | bf16 | `tanh_bw` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 618 | 27158 |
| bh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08 | 63.5 | 9.38 | 0.981 | 1.35e-08 | 180 ±6% | 93293 |
| bh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09 | 63 | 6.1 | 0.967 | 7.45e-09 | 770 | 21775 |
| bh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 | 84597 |
| bh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 199 ±28% | 84436 |
| bh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 ±21% | 21798 |
| bh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 201 ±14% | 83302 |
| bh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 325 | 51656 |
| bh | bf16 | `xielu` | `default` | 1/56847 defects; rest 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 436 | 38480 |
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
| bh | fp32 | `atan_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.25e+03 | 0.859 | 1.02 | 1146 | 14636 |
| bh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000229 | 2.97 | 1.51 | 0.912 | 0.000229 | 343 | 48876 |
| bh | fp32 | `atanh_bw` | `default` | 2/48386 defects; rest 9.02e+04 ULP | 9.02e+04 | 3.31e+03 | 0.876 | 0.681 | 7523 | 2230 |
| bh | fp32 | `bias_gelu_` | `default` | worst sampled pairing | 1.27e+08 | 1.48e+07 | 0.964 | — | 480 ±7% | 34985 |
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
| bh | fp32 | `cosh_bw` | `default` | accurate to |x| <= 0.0155 | 2.2 | 1.16 | 0.943 | 0.0155 | 6128 | 2738 |
| bh | fp32 | `deg2rad` | `default` | faithful; 63542/65024 tie-breaks | 0.63 | 0.595 | 0.906 | 3.4e+38 | 354 ±12% | 47338 |
| bh | fp32 | `digamma` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00543 | — | 737 | 22772 |
| bh | fp32 | `digamma_bw` | `default` | 257/64769 defects; rest 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.324 | 5.4e-20 | 6580 | 2550 |
| bh | fp32 | `div` | `default` | faithful; 65024/65024 tie-breaks | 0.751 | 0.662 | 0.937 | — | 490 ±6% | 34229 |
| bh | fp32 | `div_bw` | `default` | worst sampled pairing | 7.95e+04 | 7.95e+04 | 0.89 | — | 9437 | 1778 |
| bh | fp32 | `div_no_nan` | `default` | faithful; 65024/65024 tie-breaks | 0.751 | 0.662 | 0.937 | — | 1603 | 10467 |
| bh | fp32 | `divide` | `default` | faithful; 65024/65024 tie-breaks | 0.751 | 0.662 | 0.937 | — | 492 | 34109 |
| bh | fp32 | `divide_` | `default` | faithful; 65024/65024 tie-breaks | 0.751 | 0.662 | 0.937 | — | 489 | 34282 |
| bh | fp32 | `elu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 410 | 40918 |
| bh | fp32 | `elu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2486 | 6749 |
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
| bh | fp32 | `floor_div` | `default` | worst sampled pairing | 4.19e+06 | 1.62e+03 | 0.974 | — | 2266 | 7404 |
| bh | fp32 | `fmod` | `default` | worst sampled pairing | 1.61e+45 | 2.55e+41 | 0.596 | — | 497 | 33756 |
| bh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 359 ±12% | 46776 |
| bh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 481 | 34898 |
| bh | fp32 | `gelu` | `default` | 83/65024 defects; rest 1.71e+08 ULP | 1.71e+08 | 1.7e+05 | 0.963 | 0.208 | 368 | 45597 |
| bh | fp32 | `gelu` | `fast_approx` | 197/65024 defects; rest 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 348 | 48204 |
| bh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0298 | 1.47e+08 | 2.3e+04 | 0.929 | 0.0298 | 832 | 20175 |
| bh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 353 ±10% | 47576 |
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
| bh | fp32 | `ldexp_` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 491 | 34175 |
| bh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 482 ±9% | 34810 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 354 | 47376 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithful; 31672/65024 tie-breaks | 0.84 | 0.747 | 0.868 | 3.39e+38 | 347 | 48330 |
| bh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47947 |
| bh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 1668 | 10055 |
| bh | fp32 | `lerp` | `default` | worst sampled pairing | 1.46e+08 | 9.53e+04 | 0.962 | — | 646 | 25964 |
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
| bh | fp32 | `logaddexp2_` | `default` | worst sampled pairing | 1.06e+07 | 1.23e+05 | 0.959 | — | 510 | 32869 |
| bh | fp32 | `logaddexp_` | `default` | worst sampled pairing | 8e+06 | 9.77e+04 | 0.975 | — | 592 | 28344 |
| bh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 500 ±6% | 33530 |
| bh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 | 47680 |
| bh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 374 ±13% | 44904 |
| bh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 488 | 34389 |
| bh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 496 | 33824 |
| bh | fp32 | `logit` | `default` | accurate to |x| <= 0.266 | 4.19e+06 | 261 | 0.941 | 0.266 | 367 | 45777 |
| bh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 6326 | 2652 |
| bh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.97e-08 | 2.28 | 0.789 | 0.874 | 2.97e-08 | 7487 | 2241 |
| bh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 472 | 35529 |
| bh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 356 | 47106 |
| bh | fp32 | `mac` | `default` | worst sampled pairing | 4.22e+06 | 7.31e+05 | 0.974 | — | 654 | 25672 |
| bh | fp32 | `mish` | `default` | 9/65024 defects; rest 7 ULP | 7 | 2.58 | 0.953 | 1.49e-07 | 460 ±8% | 36511 |
| bh | fp32 | `mse_loss` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 484 ±8% | 34690 |
| bh | fp32 | `multigammaln` | `default` | 7422/50376 defects; rest 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.55e-17 | 7980 | 2102 |
| bh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 1.36e+15 | 2.04e+11 | 0.0102 | 5.55e-17 | 7375 | 2275 |
| bh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 482 | 34783 |
| bh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 498 ±5% | 33667 |
| bh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 473 | 35489 |
| bh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47759 |
| bh | fp32 | `nextafter` | `default` | faithful; 65024/65024 tie-breaks | 1 | 1 | 1.54e-05 | — | 2892 | 5801 |
| bh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47743 |
| bh | fp32 | `polygamma` | `k=1` | 1/49922 defects; rest 7.91e+33 ULP | 7.91e+33 | 4.68e+29 | 0.467 | 5.4e-20 | 1032 | 16254 |
| bh | fp32 | `polygamma` | `k=2` | 75/49922 defects; rest 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 1079 | 15543 |
| bh | fp32 | `polygamma` | `k=4` | 141/49922 defects; rest 3.77e+25 ULP | 3.77e+25 | 6.51e+21 | 0.138 | 3.68e-08 | 1092 | 15365 |
| bh | fp32 | `polygamma_bw` | `n=1` | accurate to |x| <= 1.79e-13 | 4.45e+30 | 4.4e+26 | 0.18 | 1.79e-13 | 6621 | 2534 |
| bh | fp32 | `pow` | `default` | worst sampled pairing | 333 | 2.35 | 0.993 | — | 662 ±9% | 25328 |
| bh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2279 ±26% | 7363 |
| bh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 364 ±39% | 46062 |
| bh | fp32 | `rad2deg` | `default` | faithful; 63518/63518 tie-breaks | 0.696 | 0.643 | 0.858 | 5.93e+36 | 347 | 48298 |
| bh | fp32 | `rdiv` | `value=2.0` | 258/64770 defects; rest 1.49 ULP | 1.49 | 1.19 | 0.664 | 8.47e+37 | 350 | 47930 |
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
| bh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 359 ±8% | 46744 |
| bh | fp32 | `rpow` | `exponent=0.5` | faithful; 5624/49536 tie-breaks | 0.879 | 0.609 | 0.995 | 3.39e+38 | 632 | 26535 |
| bh | fp32 | `rpow` | `exponent=1.0` | 1536/49536 defects; rest 0 ULP | 0 | 0 | 1 | 8.28e+34 | 630 | 26629 |
| bh | fp32 | `rpow` | `exponent=2.0` | 1536/49536 defects; rest 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 631 ±6% | 26568 |
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
| bh | fp32 | `silu_bw` | `default` | 9/65024 defects; rest 5.72e+06 ULP | 5.72e+06 | 842 | 0.934 | 2.96e-07 | 2804 | 5984 |
| bh | fp32 | `sin` | `default` | accurate to |x| <= 28 | 3.36e+07 | 3.07e+04 | 0.96 | 28 | 379 | 44290 |
| bh | fp32 | `sin_bw` | `default` | 18430/65024 defects; rest 1.24e+52 ULP | 1.24e+52 | 3.44e+48 | 0.808 | 92.4 | 1200 | 13987 |
| bh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155 | 2.2 | 1.16 | 0.943 | 0.0155 | 459 | 36542 |
| bh | fp32 | `sinh_bw` | `default` | within 2 ULP | 1.35 | 0.791 | 0.98 | 89.1 | 5392 | 3111 |
| bh | fp32 | `softplus` | `default` | 5/65024 defects; rest 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 414 | 40544 |
| bh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 1.59e-05 | 3.31 | 1.59 | 0.953 | 1.59e-05 | 3947 ±13% | 4250 |
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
| bh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 486 | 34537 |
| bh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 478 | 35106 |
| bh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 477 ±8% | 35177 |
| bh | fp32 | `swish` | `default` | 9/65024 defects; rest 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 434 | 38638 |
| bh | fp32 | `tan` | `default` | accurate to |x| <= 3.92 | 2.26e+08 | 2.32e+05 | 0.944 | 3.92 | 469 | 35767 |
| bh | fp32 | `tan_bw` | `default` | 20623/65024 defects; rest 2.85e+45 ULP | 2.85e+45 | 4.74e+44 | 0.87 | 0.852 | 1932 | 8682 |
| bh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000946 | 2.6 | 1.45 | 0.98 | 0.000946 | 395 | 42456 |
| bh | fp32 | `tanh_bw` | `default` | accurate to |x| <= 0.199 | 5 | 1.67 | 0.935 | 0.199 | 860 ±44% | 19510 |
| bh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08 | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 360 | 46647 |
| bh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09 | 4.19e+06 | 1.02e+05 | 0.901 | 7.42e-09 | 1514 | 11082 |
| bh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 358 | 46849 |
| bh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 358 ±7% | 46910 |
| bh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1486 | 11292 |
| bh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 353 | 47558 |
| bh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 650 | 25798 |
| bh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13 | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 433 | 38753 |
| wh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 421 ±9% | 39854 |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1355 | 12380 |
| wh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 712 | 23574 |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.744 | 0.996 | 0.949 | 7567 | 2217 |
| wh | bf16 | `acosh` | `default` | within 2 ULP | 1.41 | 0.596 | 0.973 | 3.39e+38 | 944 | 17779 |
| wh | bf16 | `acosh_bw` | `default` | 15874/64514 defects; rest 3.07 ULP | 3.07 | 0.64 | 0.884 | 1.03 | 8718 | 1924 |
| wh | bf16 | `add` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 529 | 31695 |
| wh | bf16 | `add_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 545 | 30797 |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 553 | 30358 |
| wh | bf16 | `addcdiv` | `default` | worst sampled pairing | 2.11e+06 | 2.35e+03 | 0.999 | — | 713 | 23517 |
| wh | bf16 | `addcmul` | `default` | worst sampled pairing | 126 | 19.4 | 1 | — | 699 | 24002 |
| wh | bf16 | `asin` | `default` | 2/32258 defects; rest 0.499 ULP | 0.499 | 0 | 1 | — | 695 | 24146 |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.949 | 2.7 | 0.702 | 0.992 | 0.949 | 7549 | 2222 |
| wh | bf16 | `asinh` | `default` | within 2 ULP | 1.32 | 0.619 | 0.992 | 3.39e+38 | 1430 | 11729 |
| wh | bf16 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.2 ULP | 1.2 | 0.638 | 0.954 | 1.84e+19 | 1448 | 11590 |
| wh | bf16 | `atan` | `default` | 2/65024 defects; rest 0.527 ULP | 0.527 | 0.513 | 1 | — | 646 ±7% | 25986 |
| wh | bf16 | `atan2` | `default` | worst sampled pairing | 200 | 2.51 | 0.998 | — | 560 | 29969 |
| wh | bf16 | `atan_bw` | `default` | accurate to |x| <= 0.23 | 2.23 | 0.767 | 0.896 | 0.23 | 1396 | 12022 |
| wh | bf16 | `atanh` | `default` | 2/32256 defects; rest 2.08 ULP | 2.08 | 0.995 | 0.081 | — | 669 | 25064 |
| wh | bf16 | `atanh_bw` | `default` | accurate to |x| <= 0.918 | 7.76 | 0.885 | 0.835 | 0.918 | 7228 | 2321 |
| wh | bf16 | `bias_gelu_` | `default` | worst sampled pairing | 179 | 86.3 | 0.996 | — | 788 | 21280 |
| wh | bf16 | `cbrt` | `default` | faithful; 846/65024 tie-breaks | 0.507 | 0.502 | 0.987 | 3.39e+38 | 450 ±11% | 37247 |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 421 ±5% | 39860 |
| wh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 484 ±8% | 34653 |
| wh | bf16 | `celu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2736 | 6132 |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 439 ±7% | 38245 |
| wh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 417 | 40253 |
| wh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 425 | 39441 |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2515 | 6671 |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 423 | 39638 |
| wh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 424 ±11% | 39565 |
| wh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 418 ±5% | 40125 |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2465 | 6807 |
| wh | bf16 | `composite_example` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 780 | 21498 |
| wh | bf16 | `cos` | `default` | faithful; 2/37354 tie-breaks | 0.552 | 0.552 | 1 | 9.99e+05 | 447 ±8% | 37566 |
| wh | bf16 | `cos_bw` | `default` | 21281/65024 defects; rest 2.09e+42 ULP | 2.09e+42 | 2.59e+39 | 0.867 | 2.15e+06 | 1730 | 9697 |
| wh | bf16 | `cosh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 458 ±13% | 36629 |
| wh | bf16 | `cosh_bw` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 3195 | 5250 |
| wh | bf16 | `deg2rad` | `default` | 2/65024 defects; rest 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 424 | 39565 |
| wh | bf16 | `digamma` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 1157 | 14497 |
| wh | bf16 | `digamma_bw` | `default` | 263/64769 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.32e+04 | 0.664 | 1 | 7517 | 2232 |
| wh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 574 | 29223 |
| wh | bf16 | `div_bw` | `default` | bit-exact | 0.498 | 0 | 1 | — | 3629 | 4623 |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 1451 | 11565 |
| wh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 540 | 31045 |
| wh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 556 | 30160 |
| wh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 482 ±6% | 34842 |
| wh | bf16 | `elu_bw` | `default` | faithful; 734/65024 tie-breaks | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2742 | 6118 |
| wh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 562 | 29857 |
| wh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 418 | 40127 |
| wh | bf16 | `erf` | `default` | within 2 ULP | 1.06 | 0.771 | 0.729 | 3.39e+38 | 589 ±8% | 28498 |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.38 | 70.5 | 3.04 | 0.975 | 1.38 | 1446 | 11602 |
| wh | bf16 | `erfc` | `default` | 198/65024 unflushed; rest 3.19e+28 ULP | 3.19e+28 | 3.23e+24 | 0.969 | 2.5 | 878 | 19114 |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.38 | 70.5 | 3.04 | 0.975 | 1.38 | 2482 | 6760 |
| wh | bf16 | `erfinv` | `default` | 29424/32256 defects; rest 121 ULP | 121 | 6.34 | 0.391 | 1.31e-38 | 865 ±5% | 19389 |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.859 | 8.91 | 1.2 | 0.98 | 0.859 | 4695 | 3574 |
| wh | bf16 | `exp` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 453 ±11% | 37010 |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP | 6.08 | 3.05 | 0.314 | — | 416 | 40348 |
| wh | bf16 | `exp2` | `default` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 449 ±8% | 37365 |
| wh | bf16 | `exp2_bw` | `default` | 1/49537 defects; rest 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 1709 | 9815 |
| wh | bf16 | `exp_bw` | `default` | faithful; 1153/49458 tie-breaks | 0.892 | 0.556 | 0.983 | 3.39e+38 | 1322 | 12693 |
| wh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 462 ±7% | 36293 |
| wh | bf16 | `expm1_bw` | `default` | accurate to |x| <= 33.2 | 34 | 0.695 | 0.984 | 33.2 | 1344 | 12480 |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 428 ±6% | 39190 |
| wh | bf16 | `floor_div` | `default` | worst sampled pairing | 64 | 42 | 0.996 | — | 2509 | 6686 |
| wh | bf16 | `fmod` | `default` | worst sampled pairing | 6.65e+35 | 7.8e+34 | 0.853 | — | 672 | 24974 |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 419 | 40056 |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 549 ±8% | 30566 |
| wh | bf16 | `gelu` | `default` | 86/65024 defects; rest 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 764 | 21964 |
| wh | bf16 | `gelu` | `fast_approx` | 198/65024 defects; rest 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 425 ±6% | 39457 |
| wh | bf16 | `gelu_bw` | `default` | faithful; 38/65024 tie-breaks | 0.939 | 0.681 | 0.999 | 3.39e+38 | 1775 | 9451 |
| wh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 429 ±9% | 39118 |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 567 | 29570 |
| wh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 422 ±8% | 39752 |
| wh | bf16 | `hardmish` | `default` | faithful; 2587/65024 tie-breaks | 1 | 0.913 | 0.961 | 3.39e+38 | 419 ±5% | 40078 |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 423 ±8% | 39625 |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1699 | 9876 |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP | 1 | 0.557 | 0.986 | 3.39e+38 | 441 ±7% | 38010 |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 2580 | 6502 |
| wh | bf16 | `hardswish` | `default` | 2/65024 defects; rest 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 437 ±9% | 38428 |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12 | 85.3 | 2.1 | 0.993 | 1.12 | 3583 | 4682 |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 442 ±10% | 37936 |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2175 | 7712 |
| wh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 423 ±7% | 39619 |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 433 | 38762 |
| wh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 421 ±8% | 39821 |
| wh | bf16 | `hypot` | `default` | 16384/65026 defects; rest 52.7 ULP | 52.7 | 1.64 | 0.992 | — | 654 | 25660 |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.6 | 255 | 56.3 | 0.952 | 13.6 | 441 ±9% | 38029 |
| wh | bf16 | `i0_bw` | `default` | 4/33904 defects; rest 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 2088 | 8034 |
| wh | bf16 | `i1` | `default` | 4/65024 defects; rest 0.858 ULP | 0.858 | 0.589 | 0.516 | 2.33e-38 | 1213 | 13834 |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 426 ±5% | 39345 |
| wh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 557 | 30124 |
| wh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 420 ±7% | 39954 |
| wh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 421 ±6% | 39849 |
| wh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 423 | 39679 |
| wh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 426 ±6% | 39356 |
| wh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 420 | 39938 |
| wh | bf16 | `l1_loss` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 550 | 30528 |
| wh | bf16 | `ldexp_` | `default` | worst sampled pairing | 255 | 94.4 | 0.955 | — | 576 | 29106 |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 551 | 30456 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 419 | 40063 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithful; 15341/65024 tie-breaks | 0.96 | 0.741 | 0.761 | 3.39e+38 | 420 ±5% | 39988 |
| wh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 413 | 40627 |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 1845 | 9092 |
| wh | bf16 | `lerp` | `default` | worst sampled pairing | 1.77e+03 | 39.9 | 1 | — | 698 | 24037 |
| wh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 436 ±9% | 38483 |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.414 | 324 | 0.846 | 0.76 | 0.414 | 2468 | 6797 |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP | 5.57e+05 | 177 | 0.173 | — | 2060 | 8144 |
| wh | bf16 | `log` | `default` | faithful; 63/32512 tie-breaks | 0.78 | 0.534 | 0.998 | 3.39e+38 | 453 ±13% | 37018 |
| wh | bf16 | `log10` | `default` | faithful; 58/32512 tie-breaks | 0.785 | 0.543 | 0.998 | 3.39e+38 | 453 ±11% | 37073 |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP | 1.74 | 0.803 | 0.508 | 3.69e+37 | 5351 | 3135 |
| wh | bf16 | `log1p` | `default` | faithful; 109/48640 tie-breaks | 0.931 | 0.57 | 0.998 | 3.39e+38 | 459 ±9% | 36558 |
| wh | bf16 | `log1p_bw` | `default` | within 2 ULP | 1.42 | 0.744 | 0.982 | 8.51e+37 | 5292 | 3170 |
| wh | bf16 | `log2` | `default` | faithful; 59/32512 tie-breaks | 0.766 | 0.542 | 0.998 | 3.39e+38 | 448 ±8% | 37472 |
| wh | bf16 | `log2_bw` | `default` | 116/64626 defects; rest 1.51 ULP | 1.51 | 0.847 | 0.57 | — | 5293 | 3170 |
| wh | bf16 | `log_bw` | `default` | bit-exact | 0.498 | 0 | 1 | 8.51e+37 | 4028 | 4165 |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.426 | 6.39 | 1.38 | 0.958 | 0.426 | 627 ±13% | 26774 |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP | 1.71 | 0.722 | 0.976 | 3.39e+38 | 5952 | 2819 |
| wh | bf16 | `logaddexp2_` | `default` | worst sampled pairing | 6.87e+10 | 2.05e+06 | 0.988 | — | 575 | 29191 |
| wh | bf16 | `logaddexp_` | `default` | worst sampled pairing | 2.22e+05 | 44.7 | 0.986 | — | 583 | 28786 |
| wh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 564 | 29772 |
| wh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 422 ±17% | 39737 |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 434 ±7% | 38696 |
| wh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 554 ±6% | 30307 |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 578 | 29025 |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395 | 64 | 1.67 | 0.985 | 0.395 | 937 | 17914 |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 6672 | 2515 |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP | 1.65 | 0.729 | 0.975 | 0.996 | 8303 | 2020 |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 569 | 29484 |
| wh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 428 ±10% | 39226 |
| wh | bf16 | `mac` | `default` | worst sampled pairing | 63.5 | 26.4 | 1 | — | 697 | 24067 |
| wh | bf16 | `mish` | `default` | 11/65024 defects; rest 1.49 ULP | 1.49 | 0.667 | 0.988 | 1.95e-38 | 615 ±9% | 27278 |
| wh | bf16 | `mse_loss` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 556 | 30152 |
| wh | bf16 | `multigammaln` | `default` | 11536/48328 defects; rest 724 ULP | 724 | 1.67 | 0.717 | 5.55e-17 | 12731 | 1318 |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 2.38e+06 | 293 | 0.249 | 5.55e-17 | 9581 | 1751 |
| wh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 534 ±44% | 31392 |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 546 | 30700 |
| wh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 563 | 29800 |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 415 ±7% | 40392 |
| wh | bf16 | `nextafter` | `default` | within 2 ULP | 1 | 1 | 0.00588 | — | 658 | 25498 |
| wh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 425 ±7% | 39478 |
| wh | bf16 | `polygamma` | `k=1` | 7/49922 defects; rest 1.02e+08 ULP | 1.02e+08 | 1.43e+05 | 0.956 | 0.996 | 1192 | 14072 |
| wh | bf16 | `polygamma` | `k=2` | 1/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 6.79e+05 | 0.945 | 4.47 | 1540 | 10896 |
| wh | bf16 | `polygamma` | `k=4` | accurate to |x| <= 3.48 | 3.22e+09 | 1.08e+07 | 0.936 | 3.48 | 1701 | 9864 |
| wh | bf16 | `polygamma_bw` | `n=1` | 1/49922 defects; rest 3.57e+08 ULP | 3.57e+08 | 6.79e+05 | 0.945 | 4.47 | 7814 | 2147 |
| wh | bf16 | `pow` | `default` | worst sampled pairing | 4.86e+18 | 2.08e+14 | 0.982 | — | 995 | 16866 |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2586 | 6488 |
| wh | bf16 | `prelu` | `weight=0.25` | 1/65024 defects; rest 0 ULP | 0 | 0 | 1 | 4.68e-38 | 420 ±6% | 39926 |
| wh | bf16 | `rad2deg` | `default` | faithful; 31760/63518 tie-breaks | 0.992 | 0.75 | 0.5 | 5.9e+36 | 416 | 40324 |
| wh | bf16 | `rdiv` | `value=2.0` | 258/64770 defects; rest 0.512 ULP | 0.512 | 0.507 | 0.984 | 8.47e+37 | 427 ±7% | 39323 |
| wh | bf16 | `reciprocal` | `default` | bit-exact | 0.498 | 0 | 1 | 8.51e+37 | 434 ±6% | 38652 |
| wh | bf16 | `reciprocal_bw` | `default` | 254/48386 defects; rest 1.23 ULP | 1.23 | 0.714 | 0.734 | 5.42e-20 | 3495 | 4801 |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 435 ±8% | 38531 |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 426 ±6% | 39428 |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2199 | 7629 |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1326 | 12648 |
| wh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 417 ±8% | 40186 |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 426 ±6% | 39405 |
| wh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 416 ±9% | 40357 |
| wh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 418 | 40093 |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 416 | 40365 |
| wh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 425 ±5% | 39467 |
| wh | bf16 | `rpow` | `exponent=0.5` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.975 | 3.39e+38 | 979 | 17131 |
| wh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1010 | 16606 |
| wh | bf16 | `rpow` | `exponent=2.0` | faithful; 1239/49536 tie-breaks | 0.898 | 0.564 | 0.983 | 3.39e+38 | 971 | 17284 |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 462 ±9% | 36318 |
| wh | bf16 | `rsqrt_bw` | `default` | 75/21665 defects; rest 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 5476 | 3064 |
| wh | bf16 | `rsub` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 558 ±41% | 30094 |
| wh | bf16 | `rsub_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 546 | 30712 |
| wh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 505 ±33% | 33231 |
| wh | bf16 | `selu_bw` | `default` | 1/65024 defects; rest 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 3129 ±15% | 5362 |
| wh | bf16 | `sigmoid` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 472 ±8% | 35526 |
| wh | bf16 | `sigmoid_accurate` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 478 ±8% | 35067 |
| wh | bf16 | `sigmoid_bw` | `default` | 141/65024 defects; rest 127 ULP | 127 | 14.2 | 0.999 | 12.4 | 1190 | 14102 |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 426 ±7% | 39386 |
| wh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 413 ±11% | 40660 |
| wh | bf16 | `silu` | `default` | 13/65024 defects; rest 0.914 ULP | 0.914 | 0.585 | 0.993 | 2.33e-38 | 491 ±12% | 34154 |
| wh | bf16 | `silu_bw` | `default` | 9/65024 defects; rest 285 ULP | 285 | 1.31 | 0.979 | 0.867 | 3120 | 5378 |
| wh | bf16 | `sin` | `default` | faithful; 4/37354 tie-breaks | 0.552 | 0.526 | 1 | 9.99e+05 | 441 ±10% | 38037 |
| wh | bf16 | `sin_bw` | `default` | 21284/65024 defects; rest 2.98e+41 ULP | 2.98e+41 | 1.39e+39 | 0.857 | 1.07e+06 | 1356 | 12368 |
| wh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 511 ±8% | 32862 |
| wh | bf16 | `sinh_bw` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 3171 | 5291 |
| wh | bf16 | `softplus` | `default` | 526/65024 defects; rest 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 484 ±12% | 34636 |
| wh | bf16 | `softplus_bw` | `default` | faithful; 505/65024 tie-breaks | 0.857 | 0.517 | 0.994 | 3.39e+38 | 2797 | 5998 |
| wh | bf16 | `softshrink` | `default` | faithful; 3968/65024 tie-breaks | 1 | 0.948 | 0.941 | 3.39e+38 | 436 ±7% | 38461 |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2192 | 7653 |
| wh | bf16 | `softsign` | `default` | 510/65024 defects; rest 1 ULP | 1 | 0.641 | 0.907 | 8.51e+37 | 460 ±8% | 36451 |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391 | 79 | 3.74 | 0.887 | 0.00391 | 1395 | 12026 |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 479 | 34992 |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP | 1.14 | 0.699 | 0.723 | 3.39e+38 | 6626 | 2532 |
| wh | bf16 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 432 ±7% | 38864 |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1298 | 12921 |
| wh | bf16 | `squared_difference` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 533 | 31453 |
| wh | bf16 | `squared_difference_` | `default` | worst sampled pairing | 2.03 | 1.79 | 0.956 | — | 543 | 30899 |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst sampled pairing | 254 | 2 | 0.997 | — | 538 | 31167 |
| wh | bf16 | `subtract` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 527 | 31818 |
| wh | bf16 | `subtract_` | `default` | faithful; 64970/65026 tie-breaks | 0.621 | 0.571 | 0.997 | — | 544 | 30849 |
| wh | bf16 | `swish` | `default` | 13/65024 defects; rest 0.914 ULP | 0.914 | 0.585 | 0.993 | 2.33e-38 | 486 ±13% | 34487 |
| wh | bf16 | `tan` | `default` | faithful; 48/37354 tie-breaks | 0.551 | 0.5 | 0.999 | 9.99e+05 | 575 ±13% | 29198 |
| wh | bf16 | `tan_bw` | `default` | 12184/65024 defects; rest 3.57e+40 ULP | 3.57e+40 | 3.8e+37 | 0.69 | 940 | 2202 | 7620 |
| wh | bf16 | `tanh` | `default` | 2/65024 defects; rest 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 435 ±6% | 38575 |
| wh | bf16 | `tanh_bw` | `default` | within 2 ULP | 1.16 | 0.525 | 0.997 | 3.39e+38 | 1536 | 10925 |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08 | 63.5 | 9.38 | 0.981 | 1.35e-08 | 459 ±6% | 36570 |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09 | 63 | 6.1 | 0.967 | 7.45e-09 | 1702 | 9858 |
| wh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 417 ±6% | 40201 |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 432 ±5% | 38860 |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1678 | 9998 |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 429 ±5% | 39069 |
| wh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 679 | 24724 |
| wh | bf16 | `xielu` | `default` | 1/56847 defects; rest 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 1042 | 16106 |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 ±6% | 21161 |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2541 | 6602 |
| wh | fp32 | `acos` | `default` | within 2 ULP | 1.55 | 0.865 | 0.923 | 1 | 1137 | 14754 |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 15027 | 1116 |
| wh | fp32 | `acosh` | `default` | never within 2 ULP | 2.76 | 0.764 | 0.782 | — | 1402 ±16% | 11970 |
| wh | fp32 | `acosh_bw` | `default` | 15874/64514 defects; rest 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 17264 | 972 |
| wh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 983 | 17068 |
| wh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1002 | 16744 |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 1017 | 16492 |
| wh | fp32 | `addcdiv` | `default` | worst sampled pairing | 2.19e+12 | 8.82e+07 | 0.901 | — | 1325 | 12659 |
| wh | fp32 | `addcmul` | `default` | worst sampled pairing | 2.13e+06 | 1.59e+04 | 0.985 | — | 1319 | 12715 |
| wh | fp32 | `asin` | `default` | within 2 ULP | 1.77 | 0.795 | 0.985 | 1 | 1092 | 15359 |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926 | 609 | 1.34 | 0.988 | 0.926 | 14843 | 1130 |
| wh | fp32 | `asinh` | `default` | within 2 ULP | 1.72 | 0.774 | 0.881 | 3.39e+38 | 1753 | 9571 |
| wh | fp32 | `asinh_bw` | `default` | 15874/64514 defects; rest 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 2665 | 6295 |
| wh | fp32 | `atan` | `default` | accurate to |x| <= 0.902 | 2.34 | 0.807 | 0.955 | 0.902 | 945 ±5% | 17749 |
| wh | fp32 | `atan2` | `default` | worst sampled pairing | 1.32e+07 | 1.26e+05 | 0.882 | — | 1020 | 16444 |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 1.01 | 2.8 | 1.23 | 0.874 | 1.01 | 2615 | 6417 |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 4.28e-07 | 3.11 | 1.55 | 0.91 | 4.28e-07 | 1300 ±7% | 12908 |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.678 | 2.05e+03 | 1.55 | 0.891 | 0.678 | 13979 | 1200 |
| wh | fp32 | `bias_gelu_` | `default` | worst sampled pairing | 1.22e+08 | 1.48e+07 | 0.964 | — | 1051 | 15961 |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38 | 2.55 | 1.76 | 0.521 | 2.26e-38 | 871 ±7% | 19268 |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 799 | 20987 |
| wh | fp32 | `celu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 935 | 17946 |
| wh | fp32 | `celu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5491 | 3056 |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 795 | 21098 |
| wh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 796 | 21089 |
| wh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 792 ±6% | 21171 |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4680 | 3585 |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 792 | 21193 |
| wh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 790 | 21241 |
| wh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 792 | 21186 |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4676 | 3588 |
| wh | fp32 | `composite_example` | `default` | never within 2 ULP | 6.55e+04 | 6.55e+04 | 1.53e-05 | — | 1529 | 10975 |
| wh | fp32 | `cos` | `default` | accurate to |x| <= 92.4 | 1.34e+08 | 1.33e+05 | 0.927 | 92.4 | 870 | 19280 |
| wh | fp32 | `cos_bw` | `default` | 18643/65024 defects; rest 8.59e+51 ULP | 8.59e+51 | 3.23e+48 | 0.841 | 28 | 3344 | 5017 |
| wh | fp32 | `cosh` | `default` | within 2 ULP | 1.35 | 0.798 | 0.98 | 89.1 | 1004 ±37% | 16707 |
| wh | fp32 | `cosh_bw` | `default` | accurate to |x| <= 0.0155 | 2.21 | 1.17 | 0.943 | 0.0155 | 6345 | 2644 |
| wh | fp32 | `deg2rad` | `default` | faithful; 63542/65024 tie-breaks | 0.63 | 0.595 | 0.906 | 3.4e+38 | 792 | 21186 |
| wh | fp32 | `digamma` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00566 | — | 2591 | 6474 |
| wh | fp32 | `digamma_bw` | `default` | 255/64769 defects; rest 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.429 | 5.4e-20 | 13512 | 1242 |
| wh | fp32 | `div` | `default` | within 2 ULP | 1.78 | 1.22 | 0.862 | — | 999 | 16794 |
| wh | fp32 | `div_bw` | `default` | faithful; 65024/65024 tie-breaks | 0.84 | 0.84 | 0.902 | — | 6977 | 2405 |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP | 1.79 | 1.52 | 0.714 | — | 2712 | 6187 |
| wh | fp32 | `divide` | `default` | within 2 ULP | 1.78 | 1.22 | 0.862 | — | 991 | 16936 |
| wh | fp32 | `divide_` | `default` | within 2 ULP | 1.78 | 1.22 | 0.862 | — | 1011 | 16600 |
| wh | fp32 | `elu` | `default` | within 2 ULP | 1.36 | 0.799 | 0.99 | 3.39e+38 | 931 | 18016 |
| wh | fp32 | `elu_bw` | `default` | faithful; 2706/65024 tie-breaks | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5446 | 3081 |
| wh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 1020 | 16448 |
| wh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 787 | 21316 |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334 | 6.48 | 1.35 | 0.671 | 0.000334 | 1278 ±10% | 13126 |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 2579 | 6506 |
| wh | fp32 | `erfc` | `default` | 198/65024 unflushed; rest 2.1e+33 ULP | 2.1e+33 | 1.71e+29 | 0.33 | — | 1342 ±9% | 12497 |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348 | 65.5 | 3.73 | 0.956 | 0.348 | 4935 | 3400 |
| wh | fp32 | `erfinv` | `default` | 29420/32256 defects; rest 7.96e+06 ULP | 7.96e+06 | 2.59e+05 | 0.000202 | 1.32e-38 | 1327 ±9% | 12644 |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456 | 4.98e+05 | 1.32e+03 | 0.914 | 0.000456 | 8006 | 2096 |
| wh | fp32 | `exp` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 918 ±5% | 18273 |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP | 3.79e+05 | 2e+05 | 0.309 | — | 790 | 21250 |
| wh | fp32 | `exp2` | `default` | faithful; 6044/49536 tie-breaks | 0.965 | 0.626 | 0.992 | 3.39e+38 | 859 | 19521 |
| wh | fp32 | `exp2_bw` | `default` | 1/49537 defects; rest 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 3338 | 5027 |
| wh | fp32 | `exp_bw` | `default` | faithful; 5447/49458 tie-breaks | 0.866 | 0.578 | 0.997 | 3.39e+38 | 2651 | 6329 |
| wh | fp32 | `expm1` | `default` | faithful; 5257/49458 tie-breaks | 0.997 | 0.565 | 0.997 | 3.39e+38 | 1041 | 16122 |
| wh | fp32 | `expm1_bw` | `default` | accurate to |x| <= 22 | 4.19e+06 | 3.83e+03 | 0.995 | 22 | 2626 | 6388 |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 796 | 21066 |
| wh | fp32 | `floor_div` | `default` | worst sampled pairing | 4.19e+06 | 1.77e+03 | 0.955 | — | 4758 | 3526 |
| wh | fp32 | `fmod` | `default` | worst sampled pairing | 1.43e+45 | 1.86e+41 | 0.762 | — | 1023 | 16397 |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 789 | 21258 |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 1009 | 16633 |
| wh | fp32 | `gelu` | `default` | 83/65024 defects; rest 1.51e+08 ULP | 1.51e+08 | 1.55e+05 | 0.963 | 0.253 | 1300 ±10% | 12903 |
| wh | fp32 | `gelu` | `fast_approx` | 197/65024 defects; rest 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 796 | 21075 |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0298 | 1.47e+08 | 2.3e+04 | 0.929 | 0.0298 | 2140 | 7839 |
| wh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 795 | 21116 |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1030 | 16295 |
| wh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 788 | 21302 |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 | 21147 |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 792 | 21195 |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3285 | 5108 |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25 | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 798 | 21014 |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 5096 | 3292 |
| wh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36 | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 834 | 20112 |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133 | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 7142 | 2349 |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 802 | 20917 |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4323 | 3881 |
| wh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 | 21153 |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 794 | 21142 |
| wh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 795 ±6% | 21114 |
| wh | fp32 | `hypot` | `default` | 16384/65024 defects; rest 3.44e+06 ULP | 3.44e+06 | 3.28e+04 | 0.965 | — | 1009 | 16626 |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.45 | 1.68e+07 | 2.19e+06 | 0.959 | 2.45 | 898 ±5% | 18687 |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00626 | 1.56e+07 | 4.11e+04 | 0.929 | 0.00626 | 3474 | 4829 |
| wh | fp32 | `i1` | `default` | accurate to |x| <= 0.00626 | 8.02 | 0.722 | 0.483 | 0.00626 | 1751 | 9580 |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 801 | 20939 |
| wh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 1019 | 16461 |
| wh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 ±6% | 21154 |
| wh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 794 | 21133 |
| wh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 | 21164 |
| wh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 | 21162 |
| wh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 794 | 21142 |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1000 | 16783 |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP | 1.92 | 1.52 | 0.953 | — | 1042 | 16102 |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 1009 | 16626 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 791 | 21204 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithful; 31672/65024 tie-breaks | 0.84 | 0.747 | 0.868 | 3.39e+38 | 792 | 21188 |
| wh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 790 | 21226 |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 3616 | 4639 |
| wh | fp32 | `lerp` | `default` | worst sampled pairing | 1.46e+08 | 9.53e+04 | 0.962 | — | 1337 | 12552 |
| wh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 792 | 21183 |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181 | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 3805 | 4409 |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP | 3.66e+17 | 7.77e+12 | 0.00566 | — | 4341 | 3864 |
| wh | fp32 | `log` | `default` | faithful; 32512/32512 tie-breaks | 0.961 | 0.562 | 0.952 | 3.39e+38 | 896 | 18731 |
| wh | fp32 | `log10` | `default` | accurate to |x| <= 0.336 | 2.13 | 1.36 | 0.641 | 0.336 | 912 | 18400 |
| wh | fp32 | `log10_bw` | `default` | accurate to |x| <= 2.04e-38 | 2.1 | 1.41 | 0.674 | 2.04e-38 | 10303 | 1628 |
| wh | fp32 | `log1p` | `default` | faithful; 20111/48640 tie-breaks | 0.967 | 0.558 | 0.982 | 3.39e+38 | 910 | 18435 |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP | 1.89 | 0.769 | 0.913 | 8.51e+37 | 10313 | 1627 |
| wh | fp32 | `log2` | `default` | accurate to |x| <= 0.704 | 2.43 | 0.517 | 0.993 | 0.704 | 915 | 18331 |
| wh | fp32 | `log2_bw` | `default` | 112/64626 defects; rest 1.9 ULP | 1.9 | 1.3 | 0.701 | — | 10307 | 1628 |
| wh | fp32 | `log_bw` | `default` | faithful; 64512/64514 tie-breaks | 0.892 | 0.714 | 0.902 | 8.51e+37 | 7837 | 2141 |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP | 4.4e+05 | 1.82e+04 | 0.481 | — | 1151 | 14580 |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.00164 | 2.95 | 1.28 | 0.954 | 0.00164 | 11595 | 1447 |
| wh | fp32 | `logaddexp2_` | `default` | worst sampled pairing | 1.06e+07 | 1.23e+05 | 0.959 | — | 1092 ±9% | 15363 |
| wh | fp32 | `logaddexp_` | `default` | worst sampled pairing | 8e+06 | 9.77e+04 | 0.975 | — | 1060 | 15824 |
| wh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 1015 | 16525 |
| wh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 ±6% | 21159 |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 795 | 21097 |
| wh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 1016 | 16519 |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 1046 | 16033 |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.266 | 4.19e+06 | 261 | 0.939 | 0.266 | 1306 ±8% | 12846 |
| wh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.98e-08 | 2.36 | 0.839 | 0.857 | 2.98e-08 | 13130 | 1278 |
| wh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.98e-08 | 2.36 | 0.839 | 0.857 | 2.98e-08 | 16506 | 1016 |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1026 | 16352 |
| wh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 788 | 21280 |
| wh | fp32 | `mac` | `default` | worst sampled pairing | 4.22e+06 | 7.31e+05 | 0.974 | — | 1332 | 12598 |
| wh | fp32 | `mish` | `default` | 9/65024 defects; rest 7 ULP | 7 | 2.49 | 0.954 | 8.61e-06 | 1245 ±12% | 13480 |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 1001 | 16758 |
| wh | fp32 | `multigammaln` | `default` | 7422/50376 defects; rest 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.59e-17 | 21050 | 797 |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17 | 1.36e+15 | 2.04e+11 | 0.0104 | 5.55e-17 | 20089 | 835 |
| wh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 986 | 17022 |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1009 | 16630 |
| wh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 1017 | 16492 |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 791 | 21212 |
| wh | fp32 | `nextafter` | `default` | faithful; 65024/65024 tie-breaks | 1 | 1 | 4.06e-05 | — | 1021 ±8% | 16436 |
| wh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 788 | 21284 |
| wh | fp32 | `polygamma` | `k=1` | accurate to |x| <= 5.4e-20 | 7.91e+33 | 4.68e+29 | 0.617 | 5.4e-20 | 1460 ±8% | 11489 |
| wh | fp32 | `polygamma` | `k=2` | accurate to |x| <= 1.79e-13 | 4.45e+30 | 4.4e+26 | 0.191 | 1.79e-13 | 1774 | 9456 |
| wh | fp32 | `polygamma` | `k=4` | accurate to |x| <= 3.68e-08 | 3.77e+25 | 6.43e+21 | 0.194 | 3.68e-08 | 2076 | 8083 |
| wh | fp32 | `polygamma_bw` | `n=1` | accurate to |x| <= 1.79e-13 | 4.45e+30 | 4.4e+26 | 0.191 | 1.79e-13 | 13843 | 1212 |
| wh | fp32 | `pow` | `default` | worst sampled pairing | 324 | 2.42 | 0.993 | — | 2065 | 8124 |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 5038 | 3330 |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 | 21169 |
| wh | fp32 | `rad2deg` | `default` | faithful; 63518/63518 tie-breaks | 0.696 | 0.643 | 0.858 | 5.93e+36 | 791 | 21199 |
| wh | fp32 | `rdiv` | `value=2.0` | 256/64770 defects; rest 0.892 ULP | 0.892 | 0.714 | 0.902 | 8.51e+37 | 820 | 20456 |
| wh | fp32 | `reciprocal` | `default` | faithful; 64512/64514 tie-breaks | 0.892 | 0.714 | 0.902 | 8.51e+37 | 816 | 20564 |
| wh | fp32 | `reciprocal_bw` | `default` | 254/48386 defects; rest 1.84 ULP | 1.84 | 1.22 | 0.726 | 5.42e-20 | 6390 | 2626 |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 791 ±7% | 21204 |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 792 | 21191 |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4257 | 3941 |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2508 | 6688 |
| wh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 | 21160 |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 791 | 21203 |
| wh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 791 | 21220 |
| wh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 790 | 21225 |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 794 | 21119 |
| wh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 794 | 21138 |
| wh | fp32 | `rpow` | `exponent=0.5` | faithful; 5624/49536 tie-breaks | 0.879 | 0.609 | 0.995 | 3.39e+38 | 1980 | 8474 |
| wh | fp32 | `rpow` | `exponent=1.0` | 1536/49536 defects; rest 0 ULP | 0 | 0 | 1 | 8.28e+34 | 1992 | 8421 |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536/49536 defects; rest 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 1955 | 8582 |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP | 1.12 | 0.834 | 0.864 | 3.39e+38 | 875 | 19165 |
| wh | fp32 | `rsqrt_bw` | `default` | 75/21666 defects; rest 6.76 ULP | 6.76 | 4.26 | 0.3 | — | 10329 | 1624 |
| wh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1007 | 16660 |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1004 | 16715 |
| wh | fp32 | `selu` | `default` | never within 2 ULP | 51.3 | 26.3 | 0 | — | 961 | 17453 |
| wh | fp32 | `selu_bw` | `default` | 1/65024 defects; rest 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 6249 | 2685 |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 1111 | 15101 |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 1096 | 15314 |
| wh | fp32 | `sigmoid_bw` | `default` | 140/65024 defects; rest 8.39e+06 ULP | 8.39e+06 | 2.71e+04 | 0.975 | 0.447 | 1848 ±7% | 9078 |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 792 | 21191 |
| wh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 788 | 21279 |
| wh | fp32 | `silu` | `default` | 9/65024 defects; rest 2.94 ULP | 2.94 | 1.55 | 0.95 | 0.000121 | 1126 | 14896 |
| wh | fp32 | `silu_bw` | `default` | 9/65024 defects; rest 5.72e+06 ULP | 5.72e+06 | 841 | 0.937 | 2.98e-07 | 6286 | 2669 |
| wh | fp32 | `sin` | `default` | accurate to |x| <= 28 | 3.36e+07 | 3.07e+04 | 0.96 | 28 | 848 | 19794 |
| wh | fp32 | `sin_bw` | `default` | 18430/65024 defects; rest 1.24e+52 ULP | 1.24e+52 | 3.44e+48 | 0.808 | 92.4 | 2586 | 6488 |
| wh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155 | 2.21 | 1.17 | 0.943 | 0.0155 | 1096 | 15309 |
| wh | fp32 | `sinh_bw` | `default` | within 2 ULP | 1.35 | 0.798 | 0.98 | 89.1 | 6256 | 2682 |
| wh | fp32 | `softplus` | `default` | 5/65024 defects; rest 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 1318 ±12% | 12729 |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.000345 | 2.64 | 1.31 | 0.96 | 0.000345 | 5673 | 2958 |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 792 | 21177 |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4309 | 3893 |
| wh | fp32 | `softsign` | `default` | 510/65024 defects; rest 2.66 ULP | 2.66 | 0.897 | 0.92 | 0.000462 | 846 | 19840 |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07 | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 2632 | 6374 |
| wh | fp32 | `sqrt` | `default` | faithful; 32512/32512 tie-breaks | 0.867 | 0.827 | 0.869 | 3.39e+38 | 882 ±7% | 19016 |
| wh | fp32 | `sqrt_bw` | `default` | never within 2 ULP | 2.24 | 1.44 | 0.702 | — | 12790 | 1312 |
| wh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 790 | 21243 |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2537 | 6614 |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 982 | 17082 |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP | 1.91 | 1.78 | 0.896 | — | 1006 | 16685 |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 1026 | 16347 |
| wh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 994 | 16870 |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1005 ±11% | 16694 |
| wh | fp32 | `swish` | `default` | 9/65024 defects; rest 2.94 ULP | 2.94 | 1.55 | 0.95 | 0.000121 | 1109 | 15132 |
| wh | fp32 | `tan` | `default` | accurate to |x| <= 3.92 | 2.26e+08 | 2.32e+05 | 0.944 | 3.92 | 1222 ±8% | 13731 |
| wh | fp32 | `tan_bw` | `default` | 20623/65024 defects; rest 2.85e+45 ULP | 2.85e+45 | 4.74e+44 | 0.87 | 0.882 | 4468 | 3755 |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000439 | 2.79 | 1.49 | 0.979 | 0.000439 | 969 | 17314 |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP | 6.59e+04 | 7.51e+03 | 0.482 | — | 1833 | 9152 |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08 | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 1332 ±10% | 12598 |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09 | 4.19e+06 | 1.01e+05 | 0.901 | 7.42e-09 | 3449 | 4865 |
| wh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 793 | 21153 |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 796 | 21087 |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3303 | 5080 |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 794 | 21130 |
| wh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 1306 | 12848 |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13 | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 1362 ±12% | 12314 |
