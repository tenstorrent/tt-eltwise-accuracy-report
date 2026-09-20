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
| bh | bf16 | `a3a9fb4229a` | 0.78.0 | 66444ba711c0 |
| bh | fp32 | `a3a9fb4229a` | 0.78.0 | 66444ba711c0 |
| wh | bf16 | `a3a9fb4229a` | 0.78.0 | 57e9901fdca0 |
| wh | fp32 | `a3a9fb4229a` | 0.78.0 | 57e9901fdca0 |

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

## Results — 953 variants

| Arch | Dtype | Op | Parameters | Verdict | Max ULP | Mean ULP | Usable to | µs | Melem/s |
|---|---|---|---|---|---|---|---|---|---|
| bh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 193 | 86846 |
| bh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 598 ±6% | 28059 |
| bh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 294 ±11% | 57128 |
| bh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949; up to 2.7 ULP beyond | 2.7 | 0.744 | 0.949 | 3536 | 4744 |
| bh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1.41 | 0.596 | 3.39e+38 | 366 | 45788 |
| bh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 3.26 | 0.688 | 1.03 | 4053 | 4140 |
| bh | bf16 | `add` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 253 | 66410 |
| bh | bf16 | `add_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 242 | 69386 |
| bh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | — | 247 | 68027 |
| bh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.35e+03 | 2.11e+06 | 2.35e+03 | — | 327 | 51271 |
| bh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.4 | 126 | 19.4 | — | 330 | 50903 |
| bh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists | 0.499 | 0 | — | 282 | 59569 |
| bh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 2.76 ULP beyond | 2.76 | 0.787 | 0.938 | 3446 | 4869 |
| bh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1.32 | 0.615 | 3.39e+38 | 501 | 33490 |
| bh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1.33 | 0.681 | 1.84e+19 | 583 ±37% | 28774 |
| bh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.527 | 0.513 | — | 209 | 80165 |
| bh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 3.56 | 200 | 3.56 | — | 261 | 64357 |
| bh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.89 | 248 | 4.89 | — | 2515 ±11% | 6671 |
| bh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 2.23 | 0.867 | 0.23 | 639 ±16% | 26242 |
| bh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists | 2.08 | 0.995 | — | 276 ±9% | 60751 |
| bh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 8.17 | 1 | 0.82 | 3846 | 4362 |
| bh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 247 | 67970 |
| bh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 248 | 67735 |
| bh | bf16 | `cbrt` | `default` | faithfully rounded; 846 of 65024 points took the other neighbour | 0.507 | 0.502 | 3.39e+38 | 221 ±8% | 76035 |
| bh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 197 ±10% | 85283 |
| bh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 201 ±11% | 83596 |
| bh | bf16 | `celu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 3.39e+38 | 1275 | 13160 |
| bh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 205 ±25% | 81810 |
| bh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 202 ±6% | 83163 |
| bh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 195 | 86171 |
| bh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | — | 1117 ±10% | 15019 |
| bh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 198 | 84699 |
| bh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 191 | 88034 |
| bh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 194 ±6% | 86663 |
| bh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | — | 1107 | 15153 |
| bh | bf16 | `cos` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 26.5 | 1.31e+05 | 272 ±23% | 61636 |
| bh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 7.45e+42 | 3.63e+39 | 2.62e+05 | 813 | 20631 |
| bh | bf16 | `cosh` | `default` | faithfully rounded; 4 of 33894 points took the other neighbour | 0.501 | 0.501 | 89 | 258 ±8% | 65062 |
| bh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0.5 | 0 | 88.5 | 3080 ±8% | 5447 |
| bh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.998 | 0.744 | 6.7e-37 | 195 | 86254 |
| bh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 399 ±12% | 42078 |
| bh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.36e+04 | 0.996 | 3179 | 5277 |
| bh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | — | 253 ±19% | 66203 |
| bh | bf16 | `div_bw` | `default` | bit-exact | 0.498 | 0 | — | 4894 | 3428 |
| bh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | — | 666 | 25187 |
| bh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | — | 261 ±28% | 64200 |
| bh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | — | 255 ±9% | 65891 |
| bh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 192 | 87194 |
| bh | bf16 | `elu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 3.39e+38 | 1285 | 13052 |
| bh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | — | 253 ±6% | 66342 |
| bh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | — | 246 | 68091 |
| bh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 196 ±16% | 85580 |
| bh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1.06 | 0.771 | 3.39e+38 | 193 ±12% | 86780 |
| bh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 1.5 | 1144 | 14661 |
| bh | bf16 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero | 3.2e+28 | 3.23e+24 | 2.5 | 319 | 52525 |
| bh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 1.5 | 1140 | 14722 |
| bh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists | 121 | 6.4 | 1.31e-38 | 356 ±29% | 47085 |
| bh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 8.91 ULP beyond | 8.91 | 1.28 | 0.777 | 3621 | 4634 |
| bh | bf16 | `exp` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 3.39e+38 | 204 ±8% | 82440 |
| bh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6.08 | 6.08 | 3.05 | — | 198 | 84532 |
| bh | bf16 | `exp2` | `default` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 3.39e+38 | 231 ±13% | 72519 |
| bh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 1.89 | 0.942 | 128 | 820 | 20458 |
| bh | bf16 | `exp_bw` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 3.39e+38 | 600 ±6% | 27973 |
| bh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 251 ±7% | 66798 |
| bh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists | 125 | 5.58 | 2.08 | 814 | 20605 |
| bh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 195 ±21% | 86140 |
| bh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | — | 1178 | 14236 |
| bh | bf16 | `fmod` | `default` | worst pairing 9.14e+35 ULP; mean 8.94e+34 | 9.14e+35 | 8.94e+34 | — | 261 ±5% | 64206 |
| bh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 199 | 84219 |
| bh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | — | 246 | 68185 |
| bh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | — | 246 ±18% | 68129 |
| bh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists | 165 | 9.18 | 2.33e-38 | 351 ±116% | 47761 |
| bh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists | 1.14e+36 | 1.82e+34 | 2.33e-38 | 186 ±8% | 90218 |
| bh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.38; up to 3.5 ULP beyond | 3.5 | 1.37 | 8.38 | 706 | 23778 |
| bh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 190 | 88311 |
| bh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | — | 254 | 66125 |
| bh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | — | 243 | 68902 |
| bh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 198 ±9% | 84824 |
| bh | bf16 | `hardmish` | `default` | faithfully rounded; 2587 of 65024 points took the other neighbour | 1 | 0.913 | 3.39e+38 | 194 ±11% | 86593 |
| bh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 193 ±8% | 87027 |
| bh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 773 | 21718 |
| bh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.557 | 3.39e+38 | 191 ±13% | 87925 |
| bh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 3.39e+38 | 1183 | 14188 |
| bh | bf16 | `hardswish` | `default` | 2 of 65024 points returned inf or zero where a value exists | 2.14 | 0.94 | 2.33e-38 | 212 ±5% | 79254 |
| bh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.13; up to 85.3 ULP beyond | 85.3 | 2.1 | 1.13 | 1655 | 10138 |
| bh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 200 ±6% | 83867 |
| bh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 988 | 16974 |
| bh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 203 ±646% | 82516 |
| bh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 3.39e+38 | 196 | 85792 |
| bh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 199 ±5% | 84237 |
| bh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 52.7 | 1.58 | — | 271 ±10% | 61844 |
| bh | bf16 | `hypot_bw` | `default` | worst pairing 74.6 ULP; mean 2.43 | 74.6 | 2.43 | — | 1515 | 11074 |
| bh | bf16 | `i0` | `default` | accurate to |x| <= 13.6; up to 255 ULP beyond | 255 | 56.3 | 13.6 | 173 ±6% | 96712 |
| bh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.58 | 2.33e-38 | 849 | 19764 |
| bh | bf16 | `i1` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.58 | 2.33e-38 | 442 ±6% | 37985 |
| bh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 198 ±8% | 84805 |
| bh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | — | 262 | 64147 |
| bh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 193 ±13% | 86987 |
| bh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 200 ±6% | 83817 |
| bh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 202 ±49% | 83150 |
| bh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 198 | 84856 |
| bh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 192 ±7% | 87520 |
| bh | bf16 | `l1_loss` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 245 | 68466 |
| bh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | — | 261 | 64170 |
| bh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | — | 251 ±6% | 66914 |
| bh | bf16 | `ldexp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.898 | 0.898 | — | 1203 | 13942 |
| bh | bf16 | `le` | `default` | bit-exact | 0 | 0 | — | 258 ±32% | 64978 |
| bh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | — | 244 ±15% | 68641 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 195 | 86042 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 15341 of 65024 points took the other neighbour | 0.96 | 0.741 | 3.39e+38 | 193 ±1010% | 86884 |
| bh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 192 | 87254 |
| bh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 3.39e+38 | 883 | 18998 |
| bh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 39.9 | 1.77e+03 | 39.9 | — | 328 ±58% | 51077 |
| bh | bf16 | `lerp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.506 | 0.506 | — | 1267 | 13242 |
| bh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 194 | 86416 |
| bh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.412; up to 324 ULP beyond | 324 | 0.846 | 0.412 | 837 | 20042 |
| bh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 800 | 20982 |
| bh | bf16 | `log` | `default` | faithfully rounded; 63 of 32512 points took the other neighbour | 0.78 | 0.534 | 3.39e+38 | 239 | 70150 |
| bh | bf16 | `log10` | `default` | faithfully rounded; 58 of 32512 points took the other neighbour | 0.785 | 0.543 | 3.39e+38 | 252 | 66664 |
| bh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 1.74 | 0.803 | 3.69e+37 | 2427 | 6913 |
| bh | bf16 | `log1p` | `default` | faithfully rounded; 109 of 48640 points took the other neighbour | 0.931 | 0.57 | 3.39e+38 | 257 | 65212 |
| bh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1.42 | 0.744 | 8.47e+37 | 2421 | 6929 |
| bh | bf16 | `log2` | `default` | faithfully rounded; 59 of 32512 points took the other neighbour | 0.766 | 0.542 | 3.39e+38 | 256 | 65504 |
| bh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists | 1.51 | 0.847 | — | 2434 ±17% | 6894 |
| bh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0.498 | 0 | 8.47e+37 | 1849 | 9072 |
| bh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.428; up to 6.39 ULP beyond | 6.39 | 1.38 | 0.428 | 235 | 71413 |
| bh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 1.71 | 0.722 | 3.39e+38 | 2708 | 6196 |
| bh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 326 ±10% | 51470 |
| bh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 385 | 43581 |
| bh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 385 ±9% | 43559 |
| bh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41.4 ULP; mean 3.6 | 41.4 | 3.6 | — | 2510 | 6685 |
| bh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 307 | 54690 |
| bh | bf16 | `logaddexp_bw` | `default` | worst pairing 62.5 ULP; mean 4.9 | 62.5 | 4.9 | — | 2168 ±89% | 7740 |
| bh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | — | 266 ±91% | 63007 |
| bh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | — | 258 ±30% | 65009 |
| bh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 194 ±5% | 86403 |
| bh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 194 ±11% | 86272 |
| bh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | — | 261 ±19% | 64248 |
| bh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | — | 264 ±17% | 63518 |
| bh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | — | 254 ±7% | 66041 |
| bh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.67 | 0.395 | 337 | 49856 |
| bh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.729 | 0.996 | 3204 | 5237 |
| bh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.729 | 0.996 | 3822 | 4390 |
| bh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | — | 249 ±5% | 67430 |
| bh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | — | 253 | 66195 |
| bh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 195 | 86051 |
| bh | bf16 | `mac` | `default` | worst pairing 63.5 ULP; mean 26.4 | 63.5 | 26.4 | — | 331 | 50713 |
| bh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 2501 | 6707 |
| bh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | — | 254 ±21% | 65976 |
| bh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 2538 | 6612 |
| bh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | — | 256 ±34% | 65470 |
| bh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists | 1.49 | 0.666 | 1.95e-38 | 268 ±7% | 62685 |
| bh | bf16 | `mse_loss` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | — | 254 ±12% | 65975 |
| bh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | — | 650 | 25819 |
| bh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists | 724 | 1.67 | 5.59e-17 | 4668 | 3594 |
| bh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.59e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 293 | 5.59e-17 | 3815 | 4398 |
| bh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | — | 248 ±7% | 67666 |
| bh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | — | 246 ±6% | 68279 |
| bh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | — | 246 | 68245 |
| bh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | — | 259 ±45% | 64729 |
| bh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 193 ±8% | 86983 |
| bh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | — | 1457 | 11513 |
| bh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 192 ±11% | 87241 |
| bh | bf16 | `polygamma` | `k=1` | 7 of 49922 points returned inf or zero where a value exists | 1.02e+08 | 2.13e+05 | 0.996 | 347 | 48288 |
| bh | bf16 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 1.21e+06 | 4.47 | 409 | 41002 |
| bh | bf16 | `polygamma` | `k=4` | 142 of 49922 points returned inf or zero where a value exists | 3.22e+09 | 1.66e+07 | 3.5 | 443 | 37848 |
| bh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 1.21e+06 | 4.47 | 3236 | 5185 |
| bh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | — | 407 | 41207 |
| bh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1.69e+38 | 1188 | 14128 |
| bh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists | 0 | 0 | 4.68e-38 | 195 | 86194 |
| bh | bf16 | `rad2deg` | `default` | faithfully rounded; 31760 of 63518 points took the other neighbour | 0.992 | 0.75 | 5.9e+36 | 198 ±13% | 84644 |
| bh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 0.498 | 0 | 8.47e+37 | 195 ±16% | 86062 |
| bh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists | 1.51 | 0.837 | 7.67e-20 | 3158 | 5312 |
| bh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0.498 | 0 | 8.47e+37 | 194 ±18% | 86429 |
| bh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 1.51 | 0.837 | 5.42e-20 | 2392 | 7015 |
| bh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 198 ±7% | 84759 |
| bh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 196 | 85475 |
| bh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1850 ±6% | 9071 |
| bh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 603 | 27822 |
| bh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 192 | 87446 |
| bh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 200 | 84036 |
| bh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 3.39e+38 | 197 ±12% | 85279 |
| bh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 198 | 84905 |
| bh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 191 | 87992 |
| bh | bf16 | `remainder` | `default` | worst pairing 1.08e+36 ULP; mean 1.22e+35 | 1.08e+36 | 1.22e+35 | — | 268 | 62673 |
| bh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 196 ±9% | 85498 |
| bh | bf16 | `rpow` | `exponent=0.5` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 3.39e+38 | 384 ±20% | 43686 |
| bh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 387 ±14% | 43325 |
| bh | bf16 | `rpow` | `exponent=2.0` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 3.39e+38 | 380 ±7% | 44184 |
| bh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | 0 | 1.18e-38 | 1366 | 12282 |
| bh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 3.39e+38 | 248 ±6% | 67538 |
| bh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists | 3.23 | 1.19 | — | 2822 | 5946 |
| bh | bf16 | `rsub` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 248 | 67568 |
| bh | bf16 | `rsub_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 255 ±84% | 65857 |
| bh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 208 ±250% | 80723 |
| bh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 2.62 | 1.06 | 0.00443 | 1470 | 11410 |
| bh | bf16 | `sigmoid` | `default` | faithfully rounded; 482 of 65024 points took the other neighbour | 0.873 | 0.522 | 3.39e+38 | 251 ±7% | 66835 |
| bh | bf16 | `sigmoid_accurate` | `default` | faithfully rounded; 482 of 65024 points took the other neighbour | 0.873 | 0.522 | 3.39e+38 | 253 ±5% | 66316 |
| bh | bf16 | `sigmoid_bw` | `default` | 331 of 65024 points returned inf or zero where a value exists | 125 | 4.81 | 1.76 | 1053 | 15935 |
| bh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 195 ±7% | 85897 |
| bh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 194 ±5% | 86304 |
| bh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists | 0.888 | 0.582 | 2.33e-38 | 168 ±14% | 99808 |
| bh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 285 | 1.3 | 0.863 | 1449 | 11578 |
| bh | bf16 | `sin` | `default` | accurate to |x| <= 2.62e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 73.5 | 2.62e+05 | 235 ±5% | 71320 |
| bh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 1.94e+42 | 1.95e+39 | 1.31e+05 | 678 | 24745 |
| bh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 89 | 186 | 90191 |
| bh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0.501 | 0.501 | 88.5 | 2806 | 5979 |
| bh | bf16 | `softcap` | `beta=50.0` | 1426 of 65024 points returned inf or zero where a value exists | 0.718 | 0.574 | — | 236 | 71143 |
| bh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists | 0.775 | 0.556 | 5.03 | 192 ±7% | 87300 |
| bh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 1.71 | 0.678 | 3.39e+38 | 2015 | 8328 |
| bh | bf16 | `softshrink` | `default` | faithfully rounded; 3968 of 65024 points took the other neighbour | 1 | 0.948 | 3.39e+38 | 203 | 82652 |
| bh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 988 | 16984 |
| bh | bf16 | `softsign` | `default` | 512 of 65024 points returned inf or zero where a value exists | 1 | 0.605 | 8.47e+37 | 196 | 85457 |
| bh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391; up to 81 ULP beyond | 81 | 3.24 | 0.00391 | 622 | 26977 |
| bh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 233 ±11% | 72068 |
| bh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1.14 | 0.699 | 3.39e+38 | 3022 | 5552 |
| bh | bf16 | `square` | `default` | faithfully rounded; 13970 of 48640 points took the other neighbour | 0.973 | 0.688 | 1.84e+19 | 196 | 85576 |
| bh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1.69e+38 | 613 ±25% | 27381 |
| bh | bf16 | `squared_difference` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | — | 248 ±7% | 67751 |
| bh | bf16 | `squared_difference_` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | — | 245 | 68584 |
| bh | bf16 | `squared_difference_bw` | `default` | faithfully rounded; 64842 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 1024 | 16381 |
| bh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | — | 249 | 67352 |
| bh | bf16 | `subtract` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 249 ±5% | 67366 |
| bh | bf16 | `subtract_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 245 ±15% | 68454 |
| bh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists | 0.888 | 0.582 | 2.33e-38 | 176 ±7% | 95084 |
| bh | bf16 | `tan` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 49.1 | 1.31e+05 | 213 ±10% | 78910 |
| bh | bf16 | `tan_bw` | `default` | 22759 of 65024 points returned inf or zero where a value exists | 3.1e+40 | 4.91e+37 | 1.13 | 970 | 17300 |
| bh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.811 | 0.586 | — | 198 | 84781 |
| bh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 1.93 | 17.2 | 618 | 27158 |
| bh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.38 | 1.35e-08 | 180 ±6% | 93293 |
| bh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.52 | 7.45e-09 | 770 | 21775 |
| bh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 198 | 84597 |
| bh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 199 ±28% | 84436 |
| bh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 770 ±21% | 21798 |
| bh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 201 ±14% | 83302 |
| bh | bf16 | `where` | `default` | bit-exact | 0 | 0 | — | 325 | 51656 |
| bh | bf16 | `xielu` | `default` | 1 of 56847 points returned inf or zero where a value exists | 0.5 | 0.5 | 2.33e-38 | 436 | 38480 |
| bh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | — | 257 ±7% | 65373 |
| bh | bf16 | `xlogy_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.78 | 0.78 | — | 4112 | 4080 |
| bh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 351 | 47732 |
| bh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1134 | 14790 |
| bh | fp32 | `acos` | `default` | within 2 ULP everywhere | 1.55 | 0.865 | 1 | 473 ±5% | 35483 |
| bh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.926 | 6841 | 2452 |
| bh | fp32 | `acosh` | `default` | never within 2 ULP; mean 0.76, worst 2.47 | 2.47 | 0.76 | — | 453 | 37023 |
| bh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 724 | 1.13 | 0.996 | 7846 | 2138 |
| bh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | — | 482 ±6% | 34838 |
| bh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | — | 487 ±6% | 34416 |
| bh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | — | 487 ±6% | 34455 |
| bh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | — | 650 | 25800 |
| bh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | — | 649 | 25844 |
| bh | fp32 | `asin` | `default` | within 2 ULP everywhere | 1.77 | 0.795 | 1 | 455 ±7% | 36900 |
| bh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.926 | 6698 | 2505 |
| bh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 1.64 | 0.774 | 3.39e+38 | 602 | 27856 |
| bh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1.98 | 1.06 | 1.84e+19 | 1187 | 14140 |
| bh | fp32 | `atan` | `default` | accurate to |x| <= 0.902; up to 2.34 ULP beyond | 2.34 | 0.806 | 0.902 | 400 | 41949 |
| bh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | — | 505 | 33237 |
| bh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 1.66e+05 | 1.64e+07 | 1.66e+05 | — | 4896 | 3426 |
| bh | fp32 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 9.02e+04 | 3.25e+03 | 1.02 | 1146 | 14636 |
| bh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000229; up to 3.04 ULP beyond | 3.04 | 1.51 | 0.000229 | 343 | 48876 |
| bh | fp32 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 9.02e+04 | 3.31e+03 | 0.681 | 7523 | 2230 |
| bh | fp32 | `bias_gelu` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | — | 480 ±7% | 34956 |
| bh | fp32 | `bias_gelu_` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | — | 480 ±7% | 34985 |
| bh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38; up to 2.55 ULP beyond | 2.55 | 1.76 | 2.26e-38 | 398 ±7% | 42190 |
| bh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 352 | 47614 |
| bh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 3.39e+38 | 416 ±60% | 40282 |
| bh | fp32 | `celu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 2496 | 6722 |
| bh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 359 ±6% | 46763 |
| bh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 350 | 47944 |
| bh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 349 | 48014 |
| bh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | — | 2191 | 7656 |
| bh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 349 | 48009 |
| bh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 346 | 48557 |
| bh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 352 | 47729 |
| bh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | — | 2184 | 7682 |
| bh | fp32 | `cos` | `default` | accurate to |x| <= 92.4; up to 3.3e+12 ULP beyond | 3.3e+12 | 3.33e+09 | 92.4 | 405 ±7% | 41474 |
| bh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 2.68e+51 | 2.8e+48 | 28 | 1496 | 11214 |
| bh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1.35 | 0.791 | 89.1 | 390 | 43022 |
| bh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 2.2 | 1.16 | 0.0155 | 6128 | 2738 |
| bh | fp32 | `deg2rad` | `default` | faithfully rounded; 63542 of 65024 points took the other neighbour | 0.63 | 0.595 | 3.4e+38 | 354 ±12% | 47338 |
| bh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 737 | 22772 |
| bh | fp32 | `digamma_bw` | `default` | 257 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 5.4e-20 | 6580 | 2550 |
| bh | fp32 | `div` | `default` | worst pairing 2.44 ULP; mean 1.44 | 2.44 | 1.44 | — | 490 ±6% | 34229 |
| bh | fp32 | `div_bw` | `default` | worst pairing 7.95e+04 ULP; mean 7.95e+04 | 7.95e+04 | 7.95e+04 | — | 9437 | 1778 |
| bh | fp32 | `div_no_nan` | `default` | worst pairing 9.24e+04 ULP; mean 4.51e+04 | 9.24e+04 | 4.51e+04 | — | 1603 | 10467 |
| bh | fp32 | `divide` | `default` | worst pairing 2.44 ULP; mean 1.44 | 2.44 | 1.44 | — | 492 | 34109 |
| bh | fp32 | `divide_` | `default` | worst pairing 2.44 ULP; mean 1.44 | 2.44 | 1.44 | — | 489 | 34282 |
| bh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 3.39e+38 | 410 | 40918 |
| bh | fp32 | `elu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 2486 | 6749 |
| bh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | — | 482 ±8% | 34824 |
| bh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | — | 486 | 34494 |
| bh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 350 | 47871 |
| bh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 7.38 ULP beyond | 7.38 | 1.36 | 0.000334 | 529 | 31714 |
| bh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.348 | 2197 | 7636 |
| bh | fp32 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero | 2.1e+33 | 1.71e+29 | — | 369 ±10% | 45407 |
| bh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.348 | 2180 | 7696 |
| bh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists | 7.96e+06 | 2.62e+05 | 1.32e-38 | 360 | 46656 |
| bh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456; up to 1.03e+06 ULP beyond | 1.03e+06 | 1.68e+03 | 0.000456 | 6749 | 2486 |
| bh | fp32 | `exp` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 402 ±6% | 41751 |
| bh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | — | 345 | 48604 |
| bh | fp32 | `exp2` | `default` | faithfully rounded; 6044 of 49536 points took the other neighbour | 0.965 | 0.626 | 3.39e+38 | 384 | 43706 |
| bh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 1.85 | 1.09 | 128 | 1506 | 11143 |
| bh | fp32 | `exp_bw` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 1181 | 14202 |
| bh | fp32 | `expm1` | `default` | faithfully rounded; 5257 of 49458 points took the other neighbour | 0.997 | 0.565 | 3.39e+38 | 458 | 36594 |
| bh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists | 4.91e+06 | 1.14e+04 | 1.39 | 1578 | 10629 |
| bh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 349 | 48070 |
| bh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.56e+03 | 4.19e+06 | 1.56e+03 | — | 2266 | 7404 |
| bh | fp32 | `fmod` | `default` | worst pairing 1.61e+45 ULP; mean 2.55e+41 | 1.61e+45 | 2.55e+41 | — | 497 | 33756 |
| bh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 359 ±12% | 46776 |
| bh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | — | 479 | 35010 |
| bh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | — | 481 | 34898 |
| bh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists | 1.71e+08 | 1.7e+05 | 0.208 | 368 | 45597 |
| bh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists | 7.45e+40 | 1.18e+39 | 2.33e-38 | 348 | 48204 |
| bh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0296; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.0296 | 832 | 20175 |
| bh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 353 ±10% | 47576 |
| bh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | — | 485 | 34618 |
| bh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | — | 480 ±8% | 34942 |
| bh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 355 | 47321 |
| bh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 354 ±47% | 47357 |
| bh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 348 | 48144 |
| bh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1477 | 11358 |
| bh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25; up to 4.89e+06 ULP beyond | 4.89e+06 | 3e+03 | 2.25 | 352 | 47601 |
| bh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 2311 | 7259 |
| bh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36; up to 7.34e+06 ULP beyond | 7.34e+06 | 1.41e+03 | 2.36 | 372 ±6% | 45047 |
| bh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.00133 | 3253 | 5158 |
| bh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 355 | 47267 |
| bh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1970 | 8517 |
| bh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 357 | 46977 |
| bh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 3.39e+38 | 356 ±19% | 47081 |
| bh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 350 | 47867 |
| bh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | — | 512 | 32773 |
| bh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 4.86e+06 | 4.48e+04 | — | 2985 | 5620 |
| bh | fp32 | `i0` | `default` | accurate to |x| <= 2.43; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 2.43 | 403 | 41623 |
| bh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00362; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.1e+04 | 0.00362 | 1357 | 12367 |
| bh | fp32 | `i1` | `default` | accurate to |x| <= 0.00362; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.1e+04 | 0.00362 | 568 ±7% | 29552 |
| bh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 353 | 47486 |
| bh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | — | 493 ±6% | 34065 |
| bh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 347 | 48290 |
| bh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 356 | 47120 |
| bh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 351 | 47770 |
| bh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 356 | 47178 |
| bh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 351 | 47817 |
| bh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | — | 478 | 35120 |
| bh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | — | 498 | 33682 |
| bh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | — | 491 | 34175 |
| bh | fp32 | `ldexp_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.742 | 0.742 | — | 2214 | 7578 |
| bh | fp32 | `le` | `default` | bit-exact | 0 | 0 | — | 482 ±6% | 34771 |
| bh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | — | 482 ±9% | 34810 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 354 | 47376 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 31672 of 65024 points took the other neighbour | 0.84 | 0.747 | 3.39e+38 | 347 | 48330 |
| bh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 350 | 47947 |
| bh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 3.39e+38 | 1668 | 10055 |
| bh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | — | 646 | 25964 |
| bh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | — | 2515 | 6670 |
| bh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 355 ±6% | 47228 |
| bh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.0181 | 1336 | 12555 |
| bh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 1517 | 11060 |
| bh | fp32 | `log` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.961 | 0.562 | 3.39e+38 | 407 | 41222 |
| bh | fp32 | `log10` | `default` | accurate to |x| <= 0.336; up to 2.13 ULP beyond | 2.13 | 1.36 | 0.336 | 398 | 42152 |
| bh | fp32 | `log10_bw` | `default` | accurate to |x| <= 4.42e+30; up to 9.02e+04 ULP beyond | 9.02e+04 | 1.73e+03 | 4.42e+30 | 4689 | 3578 |
| bh | fp32 | `log1p` | `default` | faithfully rounded; 20111 of 48640 points took the other neighbour | 0.984 | 0.558 | 3.39e+38 | 392 | 42774 |
| bh | fp32 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 9.02e+04 | 2.96e+03 | 2.02e+31 | 4709 | 3563 |
| bh | fp32 | `log2` | `default` | accurate to |x| <= 0.704; up to 2.43 ULP beyond | 2.43 | 0.511 | 0.704 | 401 ±6% | 41859 |
| bh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists | 9.02e+04 | 1.76e+03 | — | 4700 | 3570 |
| bh | fp32 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 9.02e+04 | 1.72e+03 | 2.02e+31 | 3572 | 4696 |
| bh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | — | 481 | 34856 |
| bh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.163; up to 2.93 ULP beyond | 2.93 | 1.28 | 0.163 | 5210 | 3220 |
| bh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 592 ±7% | 28356 |
| bh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 511 | 32843 |
| bh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 510 | 32869 |
| bh | fp32 | `logaddexp2_bw` | `default` | worst pairing 9.02e+04 ULP; mean 5.6e+04 | 9.02e+04 | 5.6e+04 | — | 4729 | 3548 |
| bh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 592 | 28344 |
| bh | fp32 | `logaddexp_bw` | `default` | worst pairing 8.98e+04 ULP; mean 4.46e+04 | 8.98e+04 | 4.46e+04 | — | 4281 | 3919 |
| bh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | — | 514 ±6% | 32620 |
| bh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | — | 500 ±6% | 33530 |
| bh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 352 | 47680 |
| bh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 374 ±13% | 44904 |
| bh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | — | 493 | 34036 |
| bh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | — | 488 | 34389 |
| bh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | — | 496 | 33824 |
| bh | fp32 | `logit` | `default` | accurate to |x| <= 0.266; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.266 | 367 | 45777 |
| bh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.97e-08; up to 2.28 ULP beyond | 2.28 | 0.789 | 2.97e-08 | 6326 | 2652 |
| bh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.97e-08; up to 2.28 ULP beyond | 2.28 | 0.789 | 2.97e-08 | 7487 | 2241 |
| bh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | — | 498 ±6% | 33691 |
| bh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | — | 472 | 35529 |
| bh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 356 | 47106 |
| bh | fp32 | `mac` | `default` | worst pairing 4.22e+06 ULP; mean 7.31e+05 | 4.22e+06 | 7.31e+05 | — | 654 | 25672 |
| bh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 4874 | 3442 |
| bh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | — | 491 ±6% | 34199 |
| bh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 4881 | 3437 |
| bh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | — | 489 ±5% | 34341 |
| bh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 7 | 2.58 | 1.49e-07 | 460 ±8% | 36511 |
| bh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | — | 484 ±8% | 34690 |
| bh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | — | 1262 | 13290 |
| bh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists | 3.51e+07 | 9.16e+03 | 5.55e-17 | 7980 | 2102 |
| bh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 5.55e-17 | 7375 | 2275 |
| bh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | — | 482 | 34783 |
| bh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | — | 498 ±5% | 33667 |
| bh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | — | 485 ±5% | 34598 |
| bh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | — | 473 | 35489 |
| bh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 351 | 47759 |
| bh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | — | 2892 | 5801 |
| bh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 351 | 47743 |
| bh | fp32 | `polygamma` | `k=1` | 1 of 49922 points returned inf or zero where a value exists | 7.91e+33 | 4.68e+29 | 5.4e-20 | 1032 | 16254 |
| bh | fp32 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 1079 | 15543 |
| bh | fp32 | `polygamma` | `k=4` | 141 of 49922 points returned inf or zero where a value exists | 3.77e+25 | 6.51e+21 | 3.68e-08 | 1092 | 15365 |
| bh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 6621 | 2534 |
| bh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.17 | 5.92e+03 | 3.17 | — | 662 ±9% | 25328 |
| bh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1.69e+38 | 2279 ±26% | 7363 |
| bh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0.5 | 0 | 3.39e+38 | 364 ±39% | 46062 |
| bh | fp32 | `rad2deg` | `default` | faithfully rounded; 63518 of 63518 points took the other neighbour | 0.696 | 0.643 | 5.93e+36 | 347 | 48298 |
| bh | fp32 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 1.49 | 1.19 | 8.47e+37 | 350 | 47930 |
| bh | fp32 | `rdiv_bw` | `scalar=2.0` | 254 of 48490 points returned inf or zero where a value exists | 9.02e+04 | 1.92e+03 | 7.67e-20 | 6089 | 2756 |
| bh | fp32 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 9.02e+04 | 1.72e+03 | 2.02e+31 | 349 | 48126 |
| bh | fp32 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 9.02e+04 | 1.92e+03 | 5.42e-20 | 4561 | 3678 |
| bh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 355 | 47314 |
| bh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 355 | 47295 |
| bh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 3617 | 4639 |
| bh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1148 | 14614 |
| bh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 346 | 48428 |
| bh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 356 | 47074 |
| bh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 3.39e+38 | 350 | 47976 |
| bh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 353 ±10% | 47461 |
| bh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 355 | 47255 |
| bh | fp32 | `remainder` | `default` | worst pairing 8.92e+44 ULP; mean 2.49e+41 | 8.92e+44 | 2.49e+41 | — | 503 | 33370 |
| bh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 359 ±8% | 46744 |
| bh | fp32 | `rpow` | `exponent=0.5` | faithfully rounded; 5624 of 49536 points took the other neighbour | 0.879 | 0.609 | 3.39e+38 | 632 | 26535 |
| bh | fp32 | `rpow` | `exponent=1.0` | 1536 of 49536 points returned inf or zero where a value exists | 0 | 0 | 8.28e+34 | 630 | 26629 |
| bh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists | 0.879 | 0.609 | 8.28e+34 | 631 ±6% | 26568 |
| bh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | 0 | 1.18e-38 | 2598 | 6457 |
| bh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1.12 | 0.834 | 3.39e+38 | 388 | 43246 |
| bh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists | 7.19 | 4.49 | — | 5238 | 3203 |
| bh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | — | 481 | 34855 |
| bh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | — | 479 | 34990 |
| bh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.3, worst 51.3 | 51.3 | 26.3 | — | 433 | 38786 |
| bh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 50.7 | 20.4 | — | 2817 | 5955 |
| bh | fp32 | `sigmoid` | `default` | accurate to |x| <= 1.59e-05; up to 3.31 ULP beyond | 3.31 | 1.59 | 1.59e-05 | 425 ±7% | 39502 |
| bh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 1.59e-05; up to 3.31 ULP beyond | 3.31 | 1.59 | 1.59e-05 | 420 | 39938 |
| bh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists | 8.39e+06 | 2.54e+04 | 0.283 | 2013 | 8333 |
| bh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 349 ±6% | 48068 |
| bh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 345 | 48582 |
| bh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists | 3.61 | 1.78 | 2.01e-05 | 426 ±9% | 39421 |
| bh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 5.72e+06 | 842 | 2.98e-07 | 2804 | 5984 |
| bh | fp32 | `sin` | `default` | accurate to |x| <= 28; up to 2.2e+12 ULP beyond | 2.2e+12 | 2.68e+09 | 28 | 379 | 44290 |
| bh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 1.56e+53 | 2.03e+49 | 92.4 | 1200 | 13987 |
| bh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155; up to 2.2 ULP beyond | 2.2 | 1.16 | 0.0155 | 459 | 36542 |
| bh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1.35 | 0.791 | 88.5 | 5392 | 3111 |
| bh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists | 8.21e+03 | 656 | — | 414 | 40544 |
| bh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.164; up to 2.93 ULP beyond | 2.93 | 1.33 | 0.164 | 3947 ±13% | 4250 |
| bh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 367 ±104% | 45762 |
| bh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1962 | 8551 |
| bh | fp32 | `softsign` | `default` | 512 of 65024 points returned inf or zero where a value exists | 3 | 1.28 | 1.21e-05 | 350 ±9% | 47893 |
| bh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 1.79e-07 | 1167 | 14379 |
| bh | fp32 | `sqrt` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.867 | 0.827 | 3.39e+38 | 376 ±7% | 44668 |
| bh | fp32 | `sqrt_bw` | `default` | never within 2 ULP; mean 1.4, worst 2.19 | 2.19 | 1.4 | — | 5785 | 2900 |
| bh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1.84e+19 | 351 | 47731 |
| bh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1.69e+38 | 1145 | 14649 |
| bh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | — | 486 ±20% | 34489 |
| bh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | — | 479 | 35054 |
| bh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | — | 1943 | 8635 |
| bh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | — | 486 | 34537 |
| bh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | — | 478 | 35106 |
| bh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | — | 477 ±8% | 35177 |
| bh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 3.61 | 1.78 | 2.01e-05 | 434 | 38638 |
| bh | fp32 | `tan` | `default` | accurate to |x| <= 3.92; up to 2.2e+12 ULP beyond | 2.2e+12 | 5.61e+09 | 3.92 | 469 | 35767 |
| bh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists | 2.85e+45 | 5.11e+44 | 0.852 | 1932 | 8682 |
| bh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000946; up to 2.6 ULP beyond | 2.6 | 1.45 | 0.000946 | 395 | 42456 |
| bh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | — | 860 ±44% | 19510 |
| bh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.23e+05 | 1.34e-08 | 360 | 46647 |
| bh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.02e+05 | 7.42e-09 | 1514 | 11082 |
| bh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 358 | 46849 |
| bh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 358 ±7% | 46910 |
| bh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 1486 | 11292 |
| bh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 353 | 47558 |
| bh | fp32 | `where` | `default` | bit-exact | 0 | 0 | — | 650 | 25798 |
| bh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 1.71e-13 | 433 | 38753 |
| bh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | — | 495 ±8% | 33889 |
| bh | fp32 | `xlogy_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.766 | 0.766 | — | 8029 | 2090 |
| wh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 394 ±10% | 42566 |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1307 ±18% | 12834 |
| wh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 680 | 24663 |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949; up to 2.7 ULP beyond | 2.7 | 0.744 | 0.949 | 7469 | 2246 |
| wh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1.41 | 0.596 | 3.39e+38 | 924 | 18153 |
| wh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 3.26 | 0.688 | 1.03 | 8632 | 1944 |
| wh | bf16 | `add` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 534 ±6% | 31443 |
| wh | bf16 | `add_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 555 ±6% | 30222 |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | — | 542 ±7% | 30964 |
| wh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.35e+03 | 2.11e+06 | 2.35e+03 | — | 688 | 24399 |
| wh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.4 | 126 | 19.4 | — | 686 | 24471 |
| wh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists | 0.499 | 0 | — | 675 | 24867 |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 2.76 ULP beyond | 2.76 | 0.787 | 0.938 | 7424 | 2260 |
| wh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1.32 | 0.619 | 3.39e+38 | 1411 ±16% | 11888 |
| wh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1.33 | 0.681 | 1.84e+19 | 1356 | 12369 |
| wh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.527 | 0.513 | — | 602 ±11% | 27882 |
| wh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 2.51 | 200 | 2.51 | — | 550 ±14% | 30496 |
| wh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.88 | 248 | 4.88 | — | 5517 | 3041 |
| wh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 2.23 | 0.867 | 0.23 | 1325 | 12664 |
| wh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists | 2.08 | 0.995 | — | 694 ±11% | 24189 |
| wh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 8.17 | 1.02 | 0.82 | 8437 | 1988 |
| wh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 536 | 31315 |
| wh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 544 | 30824 |
| wh | bf16 | `cbrt` | `default` | faithfully rounded; 846 of 65024 points took the other neighbour | 0.507 | 0.502 | 3.39e+38 | 413 ±8% | 40606 |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 401 ±9% | 41798 |
| wh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 482 ±10% | 34804 |
| wh | bf16 | `celu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 3.39e+38 | 2718 ±24% | 6173 |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 408 | 41077 |
| wh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 394 ±7% | 42567 |
| wh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 396 | 42358 |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | — | 2457 | 6827 |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 403 | 41616 |
| wh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 394 ±6% | 42607 |
| wh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 393 | 42655 |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | — | 2449 | 6851 |
| wh | bf16 | `cos` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 26.5 | 1.31e+05 | 427 ±7% | 39299 |
| wh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 7.45e+42 | 3.63e+39 | 2.62e+05 | 1692 ±6% | 9918 |
| wh | bf16 | `cosh` | `default` | bit-exact | 0.5 | 0 | 89 | 446 ±10% | 37651 |
| wh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0.5 | 0 | 88.5 | 6675 | 2513 |
| wh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.998 | 0.744 | 6.7e-37 | 405 ±7% | 41376 |
| wh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 1183 ±14% | 14179 |
| wh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.32e+04 | 1 | 12072 | 1390 |
| wh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | — | 556 ±9% | 30161 |
| wh | bf16 | `div_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.512 | 0.512 | — | 10489 | 1600 |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | — | 1457 | 11515 |
| wh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | — | 559 ±8% | 29993 |
| wh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | — | 569 ±5% | 29463 |
| wh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 482 ±12% | 34832 |
| wh | bf16 | `elu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 3.39e+38 | 2738 | 6129 |
| wh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | — | 533 | 31480 |
| wh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | — | 542 | 30957 |
| wh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 394 | 42558 |
| wh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1.06 | 0.771 | 3.39e+38 | 568 ±11% | 29562 |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 1.5 | 2444 | 6865 |
| wh | bf16 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero | 3.19e+28 | 3.23e+24 | 2.5 | 818 | 20507 |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 1.5 | 2452 | 6843 |
| wh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists | 121 | 6.4 | 1.31e-38 | 890 | 18856 |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 8.91 ULP beyond | 8.91 | 1.28 | 0.777 | 7856 | 2136 |
| wh | bf16 | `exp` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 3.39e+38 | 424 ±9% | 39563 |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6.08 | 6.08 | 3.05 | — | 396 ±5% | 42343 |
| wh | bf16 | `exp2` | `default` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 3.39e+38 | 426 ±8% | 39381 |
| wh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 1.89 | 0.942 | 128 | 1695 | 9895 |
| wh | bf16 | `exp_bw` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 3.39e+38 | 1353 ±16% | 12398 |
| wh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 461 ±14% | 36384 |
| wh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists | 125 | 5.58 | 2.09 | 1706 ±27% | 9835 |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 412 ±6% | 40729 |
| wh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | — | 2521 | 6654 |
| wh | bf16 | `fmod` | `default` | worst pairing 6.65e+35 ULP; mean 7.8e+34 | 6.65e+35 | 7.8e+34 | — | 655 | 25613 |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 405 ±11% | 41390 |
| wh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | — | 533 | 31458 |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | — | 552 ±25% | 30403 |
| wh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists | 165 | 9.18 | 2.33e-38 | 810 | 20710 |
| wh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists | 1.14e+36 | 1.82e+34 | 2.33e-38 | 402 ±7% | 41774 |
| wh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.44; up to 3.5 ULP beyond | 3.5 | 1.37 | 8.44 | 1775 | 9452 |
| wh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 404 ±9% | 41531 |
| wh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | — | 532 | 31561 |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | — | 551 | 30471 |
| wh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 406 | 41327 |
| wh | bf16 | `hardmish` | `default` | faithfully rounded; 2587 of 65024 points took the other neighbour | 1 | 0.913 | 3.39e+38 | 402 ±6% | 41758 |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 410 ±12% | 40873 |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1676 | 10012 |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.557 | 3.39e+38 | 408 ±6% | 41140 |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 3.39e+38 | 2537 | 6613 |
| wh | bf16 | `hardswish` | `default` | 2 of 65024 points returned inf or zero where a value exists | 2.14 | 0.94 | 2.33e-38 | 417 ±11% | 40234 |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12; up to 85.3 ULP beyond | 85.3 | 2.1 | 1.12 | 3532 | 4751 |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 401 ±6% | 41811 |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 2155 | 7784 |
| wh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 408 ±9% | 41082 |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 3.39e+38 | 422 ±10% | 39757 |
| wh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 415 ±14% | 40458 |
| wh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 52.7 | 1.58 | — | 547 | 30698 |
| wh | bf16 | `hypot_bw` | `default` | worst pairing 74.6 ULP; mean 2.42 | 74.6 | 2.42 | — | 3364 | 4987 |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.6; up to 255 ULP beyond | 255 | 56.3 | 13.6 | 451 ±8% | 37200 |
| wh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.58 | 2.33e-38 | 2086 | 8042 |
| wh | bf16 | `i1` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.58 | 2.33e-38 | 1196 | 14029 |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 403 ±6% | 41589 |
| wh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | — | 550 ±6% | 30490 |
| wh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 407 | 41218 |
| wh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 396 ±7% | 42406 |
| wh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 399 ±7% | 42026 |
| wh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 404 ±8% | 41533 |
| wh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 403 ±5% | 41617 |
| wh | bf16 | `l1_loss` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 543 ±6% | 30911 |
| wh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | — | 548 ±6% | 30625 |
| wh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | — | 558 | 30051 |
| wh | bf16 | `ldexp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.898 | 0.898 | — | 2725 | 6156 |
| wh | bf16 | `le` | `default` | bit-exact | 0 | 0 | — | 544 | 30820 |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | — | 547 ±5% | 30685 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 400 ±7% | 41892 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 15341 of 65024 points took the other neighbour | 0.96 | 0.741 | 3.39e+38 | 397 | 42230 |
| wh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 400 ±6% | 41959 |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 3.39e+38 | 1830 | 9170 |
| wh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 39.9 | 1.77e+03 | 39.9 | — | 683 | 24561 |
| wh | bf16 | `lerp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.506 | 0.506 | — | 2811 | 5969 |
| wh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 399 ±7% | 42095 |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.414; up to 324 ULP beyond | 324 | 0.846 | 0.414 | 2420 | 6934 |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 2074 | 8088 |
| wh | bf16 | `log` | `default` | faithfully rounded; 63 of 32512 points took the other neighbour | 0.78 | 0.534 | 3.39e+38 | 428 ±9% | 39178 |
| wh | bf16 | `log10` | `default` | faithfully rounded; 58 of 32512 points took the other neighbour | 0.785 | 0.543 | 3.39e+38 | 428 ±15% | 39239 |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 1.74 | 0.796 | 3.69e+37 | 5280 | 3178 |
| wh | bf16 | `log1p` | `default` | faithfully rounded; 109 of 48640 points took the other neighbour | 0.931 | 0.57 | 3.39e+38 | 426 ±10% | 39404 |
| wh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1.42 | 0.68 | 8.47e+37 | 5261 | 3189 |
| wh | bf16 | `log2` | `default` | faithfully rounded; 59 of 32512 points took the other neighbour | 0.766 | 0.542 | 3.39e+38 | 420 ±8% | 39933 |
| wh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists | 1.51 | 0.842 | — | 5273 | 3182 |
| wh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0.512 | 0.507 | 8.47e+37 | 3982 | 4213 |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.426; up to 6.39 ULP beyond | 6.39 | 1.38 | 0.426 | 588 ±5% | 28554 |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 1.91 | 0.785 | 3.39e+38 | 5904 | 2842 |
| wh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 814 ±10% | 20603 |
| wh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 898 | 18676 |
| wh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 903 | 18579 |
| wh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41.4 ULP; mean 3.6 | 41.4 | 3.6 | — | 5812 | 2887 |
| wh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 782 | 21462 |
| wh | bf16 | `logaddexp_bw` | `default` | worst pairing 62.5 ULP; mean 4.9 | 62.5 | 4.9 | — | 4772 ±5% | 3516 |
| wh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | — | 546 | 30727 |
| wh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | — | 562 ±11% | 29835 |
| wh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 404 | 41578 |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 404 | 41488 |
| wh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | — | 556 ±58% | 30161 |
| wh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | — | 563 ±5% | 29785 |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | — | 566 | 29633 |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.67 | 0.395 | 888 | 18895 |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.649 | 0.996 | 6992 | 2399 |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.649 | 0.996 | 8234 | 2038 |
| wh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | — | 533 | 31472 |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | — | 547 | 30664 |
| wh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 400 ±8% | 41922 |
| wh | bf16 | `mac` | `default` | worst pairing 63.5 ULP; mean 26.4 | 63.5 | 26.4 | — | 683 | 24551 |
| wh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 5463 | 3071 |
| wh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | — | 548 ±10% | 30642 |
| wh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 5458 | 3074 |
| wh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | — | 530 | 31632 |
| wh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists | 1.49 | 0.667 | 1.95e-38 | 599 ±7% | 27995 |
| wh | bf16 | `mse_loss` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | — | 528 | 31754 |
| wh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | — | 1422 | 11794 |
| wh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists | 724 | 1.67 | 5.55e-17 | 12628 | 1328 |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 293 | 5.55e-17 | 9635 | 1741 |
| wh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | — | 537 | 31259 |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | — | 550 ±5% | 30486 |
| wh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | — | 536 ±6% | 31324 |
| wh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | — | 551 | 30434 |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 406 ±14% | 41296 |
| wh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | — | 3123 | 5371 |
| wh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 402 ±8% | 41700 |
| wh | bf16 | `polygamma` | `k=1` | 7 of 49922 points returned inf or zero where a value exists | 1.02e+08 | 1.43e+05 | 0.996 | 5547 | 3024 |
| wh | bf16 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 4.47 | 5629 | 2980 |
| wh | bf16 | `polygamma` | `k=4` | 142 of 49922 points returned inf or zero where a value exists | 3.22e+09 | 1.09e+07 | 3.48 | 5729 | 2928 |
| wh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 4.47 | 12059 | 1391 |
| wh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | — | 1001 ±10% | 16763 |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1.69e+38 | 2550 | 6580 |
| wh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists | 0 | 0 | 4.68e-38 | 404 ±7% | 41527 |
| wh | bf16 | `rad2deg` | `default` | faithfully rounded; 31760 of 63518 points took the other neighbour | 0.992 | 0.75 | 5.9e+36 | 398 ±6% | 42152 |
| wh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 0.512 | 0.507 | 8.47e+37 | 420 ±9% | 39953 |
| wh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists | 1.51 | 0.835 | 7.67e-20 | 6890 | 2435 |
| wh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0.512 | 0.507 | 8.47e+37 | 404 ±15% | 41483 |
| wh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 1.51 | 0.835 | 5.42e-20 | 5097 | 3292 |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 410 | 40919 |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 401 ±5% | 41875 |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 3937 | 4261 |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 1305 | 12852 |
| wh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 404 ±6% | 41569 |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 409 ±9% | 40972 |
| wh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 3.39e+38 | 402 ±6% | 41728 |
| wh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 404 ±6% | 41514 |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 405 ±9% | 41455 |
| wh | bf16 | `remainder` | `default` | worst pairing 6.65e+35 ULP; mean 1.06e+35 | 6.65e+35 | 1.06e+35 | — | 679 | 24702 |
| wh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 404 ±7% | 41503 |
| wh | bf16 | `rpow` | `exponent=0.5` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 3.39e+38 | 973 ±11% | 17240 |
| wh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 966 | 17360 |
| wh | bf16 | `rpow` | `exponent=2.0` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 3.39e+38 | 956 | 17545 |
| wh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | 0 | 1.18e-38 | 2924 | 5737 |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 3.39e+38 | 426 ±6% | 39346 |
| wh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists | 3.23 | 1.19 | — | 6294 | 2666 |
| wh | bf16 | `rsub` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 543 ±486% | 30921 |
| wh | bf16 | `rsub_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 544 | 30825 |
| wh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 521 ±27% | 32184 |
| wh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 2.62 | 1.06 | 0.00443 | 3122 ±7% | 5373 |
| wh | bf16 | `sigmoid` | `default` | faithfully rounded; 505 of 65024 points took the other neighbour | 0.857 | 0.517 | 3.39e+38 | 473 ±5% | 35478 |
| wh | bf16 | `sigmoid_accurate` | `default` | faithfully rounded; 505 of 65024 points took the other neighbour | 0.857 | 0.517 | 3.39e+38 | 477 ±18% | 35187 |
| wh | bf16 | `sigmoid_bw` | `default` | 329 of 65024 points returned inf or zero where a value exists | 266 | 5.5 | 1.77 | 2252 | 7451 |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 395 ±9% | 42504 |
| wh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 398 ±5% | 42186 |
| wh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists | 0.914 | 0.585 | 2.33e-38 | 479 ±15% | 35033 |
| wh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 285 | 1.31 | 0.867 | 3101 | 5409 |
| wh | bf16 | `sin` | `default` | accurate to |x| <= 2.62e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 73.5 | 2.62e+05 | 410 ±12% | 40896 |
| wh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 1.94e+42 | 1.95e+39 | 1.31e+05 | 1318 | 12734 |
| wh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 89 | 535 ±17% | 31379 |
| wh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0.5 | 0 | 88.5 | 5834 | 2876 |
| wh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists | 0.775 | 0.556 | 5.03 | 478 ±10% | 35129 |
| wh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 1.91 | 0.746 | 3.39e+38 | 4357 | 3851 |
| wh | bf16 | `softshrink` | `default` | faithfully rounded; 3968 of 65024 points took the other neighbour | 1 | 0.948 | 3.39e+38 | 415 ±5% | 40410 |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 2167 | 7743 |
| wh | bf16 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 1 | 0.641 | 8.51e+37 | 427 ±12% | 39320 |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391; up to 81 ULP beyond | 81 | 3.29 | 0.00391 | 1319 | 12717 |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 435 ±8% | 38528 |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1.14 | 0.695 | 3.39e+38 | 6556 | 2559 |
| wh | bf16 | `square` | `default` | faithfully rounded; 13970 of 48640 points took the other neighbour | 0.973 | 0.688 | 1.84e+19 | 393 | 42742 |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1.69e+38 | 1287 | 13035 |
| wh | bf16 | `squared_difference` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | — | 539 | 31103 |
| wh | bf16 | `squared_difference_` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | — | 555 | 30212 |
| wh | bf16 | `squared_difference_bw` | `default` | faithfully rounded; 64842 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 2164 | 7753 |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | — | 531 | 31601 |
| wh | bf16 | `subtract` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 528 | 31780 |
| wh | bf16 | `subtract_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | — | 548 | 30636 |
| wh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists | 0.914 | 0.585 | 2.33e-38 | 495 ±15% | 33888 |
| wh | bf16 | `tan` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 47.6 | 1.31e+05 | 572 ±7% | 29354 |
| wh | bf16 | `tan_bw` | `default` | 12178 of 65024 points returned inf or zero where a value exists | 3.1e+40 | 1.95e+37 | 1.13 | 2203 | 7617 |
| wh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.811 | 0.586 | — | 404 ±8% | 41530 |
| wh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 1.93 | 17.2 | 1465 | 11451 |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.38 | 1.35e-08 | 443 ±6% | 37879 |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.52 | 7.45e-09 | 1683 | 9966 |
| wh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 397 | 42213 |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 417 ±6% | 40215 |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 1658 | 10117 |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 406 ±8% | 41288 |
| wh | bf16 | `where` | `default` | bit-exact | 0 | 0 | — | 680 | 24676 |
| wh | bf16 | `xielu` | `default` | 1 of 56847 points returned inf or zero where a value exists | 0.5 | 0.5 | 2.33e-38 | 1014 | 16546 |
| wh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | — | 550 ±7% | 30488 |
| wh | bf16 | `xlogy_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.78 | 0.78 | — | 8852 | 1895 |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 761 ±157% | 22055 |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 2481 | 6762 |
| wh | fp32 | `acos` | `default` | within 2 ULP everywhere | 1.55 | 0.865 | 1 | 1090 | 15388 |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.926 | 14940 | 1123 |
| wh | fp32 | `acosh` | `default` | never within 2 ULP; mean 0.764, worst 2.76 | 2.76 | 0.764 | — | 1462 ±9% | 11479 |
| wh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 724 | 1.13 | 0.996 | 17165 | 977 |
| wh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | — | 991 ±7% | 16934 |
| wh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | — | 997 | 16828 |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | — | 994 | 16872 |
| wh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | — | 1344 | 12479 |
| wh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | — | 1339 | 12532 |
| wh | fp32 | `asin` | `default` | within 2 ULP everywhere | 1.77 | 0.795 | 1 | 1056 | 15882 |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.926 | 14748 | 1138 |
| wh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 1.72 | 0.774 | 3.39e+38 | 1800 | 9321 |
| wh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1.98 | 1.06 | 1.84e+19 | 2592 | 6472 |
| wh | fp32 | `atan` | `default` | accurate to |x| <= 0.902; up to 2.34 ULP beyond | 2.34 | 0.807 | 0.902 | 900 | 18646 |
| wh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | — | 1009 | 16633 |
| wh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 9.5e+04 | 1.64e+07 | 9.5e+04 | — | 10552 | 1590 |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 1.01; up to 2.8 ULP beyond | 2.8 | 1.23 | 1.01 | 2554 | 6568 |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 4.28e-07; up to 3.11 ULP beyond | 3.11 | 1.55 | 4.28e-07 | 1309 ±10% | 12819 |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.678; up to 2.05e+03 ULP beyond | 2.05e+03 | 1.55 | 0.678 | 16361 | 1025 |
| wh | fp32 | `bias_gelu` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | — | 988 | 16983 |
| wh | fp32 | `bias_gelu_` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | — | 1004 | 16706 |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38; up to 2.55 ULP beyond | 2.55 | 1.76 | 2.26e-38 | 842 | 19934 |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 771 | 21768 |
| wh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 3.39e+38 | 945 | 17755 |
| wh | fp32 | `celu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 5468 | 3068 |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 771 | 21771 |
| wh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 775 | 21657 |
| wh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 770 | 21778 |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | — | 4602 | 3645 |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 769 | 21803 |
| wh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 765 | 21924 |
| wh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 3.39e+38 | 763 | 21989 |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | — | 4613 ±56% | 3637 |
| wh | fp32 | `cos` | `default` | accurate to |x| <= 92.4; up to 3.3e+12 ULP beyond | 3.3e+12 | 3.33e+09 | 92.4 | 856 | 19594 |
| wh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 2.68e+51 | 2.8e+48 | 28 | 3299 | 5086 |
| wh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1.35 | 0.798 | 89.1 | 973 | 17250 |
| wh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 2.21 | 1.17 | 0.0155 | 13544 | 1239 |
| wh | fp32 | `deg2rad` | `default` | faithfully rounded; 63542 of 65024 points took the other neighbour | 0.63 | 0.595 | 3.4e+38 | 771 | 21762 |
| wh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 2628 | 6385 |
| wh | fp32 | `digamma_bw` | `default` | 255 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 5.4e-20 | 17824 | 941 |
| wh | fp32 | `div` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | — | 1015 ±447% | 16534 |
| wh | fp32 | `div_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.84 | 0.84 | — | 20661 | 812 |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP everywhere | 1.79 | 1.52 | — | 3461 | 4847 |
| wh | fp32 | `divide` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | — | 999 | 16790 |
| wh | fp32 | `divide_` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | — | 1017 | 16503 |
| wh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 3.39e+38 | 949 | 17674 |
| wh | fp32 | `elu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 5435 | 3087 |
| wh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | — | 986 | 17019 |
| wh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | — | 1002 | 16743 |
| wh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 766 | 21897 |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 6.48 ULP beyond | 6.48 | 1.35 | 0.000334 | 1198 ±68% | 14000 |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.348 | 4853 | 3457 |
| wh | fp32 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero | 2.1e+33 | 1.71e+29 | — | 1299 ±9% | 12914 |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.348 | 4856 | 3455 |
| wh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists | 7.96e+06 | 2.62e+05 | 1.32e-38 | 1415 ±10% | 11859 |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456; up to 4.98e+05 ULP beyond | 4.98e+05 | 1.32e+03 | 0.000456 | 15351 | 1093 |
| wh | fp32 | `exp` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 895 | 18745 |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | — | 772 | 21744 |
| wh | fp32 | `exp2` | `default` | faithfully rounded; 6044 of 49536 points took the other neighbour | 0.965 | 0.626 | 3.39e+38 | 838 | 20023 |
| wh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 1.85 | 1.09 | 128 | 3314 | 5063 |
| wh | fp32 | `exp_bw` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 3.39e+38 | 2605 | 6440 |
| wh | fp32 | `expm1` | `default` | faithfully rounded; 5257 of 49458 points took the other neighbour | 0.997 | 0.565 | 3.39e+38 | 1042 ±6% | 16103 |
| wh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists | 4.91e+06 | 1.14e+04 | 1.39 | 3495 | 4800 |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 784 | 21410 |
| wh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.77e+03 | 4.19e+06 | 1.77e+03 | — | 4796 | 3498 |
| wh | fp32 | `fmod` | `default` | worst pairing 1.43e+45 ULP; mean 1.86e+41 | 1.43e+45 | 1.86e+41 | — | 1014 ±12% | 16551 |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 775 | 21647 |
| wh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | — | 981 | 17096 |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | — | 1000 | 16780 |
| wh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists | 1.51e+08 | 1.55e+05 | 0.253 | 1389 ±13% | 12081 |
| wh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists | 7.45e+40 | 1.18e+39 | 2.33e-38 | 776 | 21622 |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0296; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.0296 | 2096 | 8004 |
| wh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 776 | 21616 |
| wh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | — | 983 | 17068 |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | — | 1000 | 16778 |
| wh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 769 | 21803 |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 776 ±89% | 21621 |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 772 | 21732 |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 3220 | 5210 |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25; up to 4.89e+06 ULP beyond | 4.89e+06 | 3e+03 | 2.25 | 778 | 21577 |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 5030 | 3336 |
| wh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36; up to 7.34e+06 ULP beyond | 7.34e+06 | 1.41e+03 | 2.36 | 818 | 20513 |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.00133 | 7078 | 2370 |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 779 | 21545 |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 4257 | 3941 |
| wh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 790 | 21237 |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 3.39e+38 | 786 | 21333 |
| wh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 774 | 21662 |
| wh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | — | 1003 | 16731 |
| wh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 4.86e+06 | 4.48e+04 | — | 6337 | 2648 |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.43; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 2.43 | 897 | 18697 |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00626; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.00626 | 3440 | 4878 |
| wh | fp32 | `i1` | `default` | accurate to |x| <= 0.00626; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.00626 | 1736 | 9665 |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 773 ±6% | 21694 |
| wh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | — | 1002 | 16745 |
| wh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 770 | 21786 |
| wh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 774 | 21664 |
| wh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 777 | 21605 |
| wh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 776 | 21608 |
| wh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 773 | 21692 |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | — | 994 | 16880 |
| wh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | — | 997 | 16834 |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | — | 1010 | 16605 |
| wh | fp32 | `ldexp_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.742 | 0.742 | — | 5252 | 3194 |
| wh | fp32 | `le` | `default` | bit-exact | 0 | 0 | — | 987 | 17005 |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | — | 1006 | 16675 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 771 | 21754 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 31672 of 65024 points took the other neighbour | 0.84 | 0.747 | 3.39e+38 | 767 | 21861 |
| wh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 781 | 21472 |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 3.39e+38 | 3609 | 4649 |
| wh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | — | 1342 | 12502 |
| wh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | — | 5370 | 3124 |
| wh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 766 | 21910 |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.0181 | 3746 | 4478 |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 4338 | 3867 |
| wh | fp32 | `log` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.961 | 0.562 | 3.39e+38 | 861 | 19487 |
| wh | fp32 | `log10` | `default` | accurate to |x| <= 0.336; up to 2.13 ULP beyond | 2.13 | 1.36 | 0.336 | 874 ±5% | 19194 |
| wh | fp32 | `log10_bw` | `default` | accurate to |x| <= 2.04e-38; up to 2.1 ULP beyond | 2.1 | 1.41 | 2.04e-38 | 10285 ±33% | 1631 |
| wh | fp32 | `log1p` | `default` | faithfully rounded; 20111 of 48640 points took the other neighbour | 0.984 | 0.558 | 3.39e+38 | 906 | 18520 |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP everywhere | 1.89 | 0.769 | 8.51e+37 | 10279 | 1632 |
| wh | fp32 | `log2` | `default` | accurate to |x| <= 0.704; up to 2.43 ULP beyond | 2.43 | 0.511 | 0.704 | 882 ±20% | 19023 |
| wh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists | 1.9 | 1.3 | — | 10236 | 1639 |
| wh | fp32 | `log_bw` | `default` | faithfully rounded; 64512 of 64514 points took the other neighbour | 0.892 | 0.714 | 8.51e+37 | 7769 | 2160 |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | — | 1091 | 15380 |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.00164; up to 2.95 ULP beyond | 2.95 | 1.28 | 0.00164 | 11486 | 1461 |
| wh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 1486 | 11291 |
| wh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 1302 | 12883 |
| wh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 1312 ±5% | 12786 |
| wh | fp32 | `logaddexp2_bw` | `default` | worst pairing 45.7 ULP; mean 6.68 | 45.7 | 6.68 | — | 11266 | 1489 |
| wh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 1485 | 11296 |
| wh | fp32 | `logaddexp_bw` | `default` | worst pairing 65.7 ULP; mean 8.66 | 65.7 | 8.66 | — | 9386 | 1788 |
| wh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | — | 1005 | 16701 |
| wh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | — | 1025 | 16365 |
| wh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 779 | 21550 |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 773 | 21699 |
| wh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | — | 992 | 16919 |
| wh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | — | 1010 | 16606 |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | — | 1015 | 16527 |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.266; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.266 | 1050 ±230% | 15972 |
| wh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.98e-08; up to 2.36 ULP beyond | 2.36 | 0.839 | 2.98e-08 | 13815 | 1214 |
| wh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.98e-08; up to 2.36 ULP beyond | 2.36 | 0.839 | 2.98e-08 | 16444 | 1020 |
| wh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | — | 990 ±6% | 16946 |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | — | 1002 | 16746 |
| wh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 762 ±17% | 22004 |
| wh | fp32 | `mac` | `default` | worst pairing 4.22e+06 ULP; mean 7.31e+05 | 4.22e+06 | 7.31e+05 | — | 1348 ±10% | 12444 |
| wh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 10450 | 1606 |
| wh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | — | 982 | 17092 |
| wh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 10441 | 1607 |
| wh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | — | 996 | 16837 |
| wh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 7 | 2.49 | 8.61e-06 | 1178 ±6% | 14248 |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | — | 990 | 16948 |
| wh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | — | 2690 | 6236 |
| wh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists | 3.51e+07 | 9.16e+03 | 5.59e-17 | 20796 | 807 |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 5.55e-17 | 20143 | 833 |
| wh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | — | 978 | 17158 |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | — | 1002 | 16744 |
| wh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | — | 995 ±6% | 16862 |
| wh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | — | 1009 | 16634 |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 769 | 21821 |
| wh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | — | 6117 | 2743 |
| wh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 765 | 21927 |
| wh | fp32 | `polygamma` | `k=1` | accurate to |x| <= 5.4e-20; up to 7.91e+33 ULP beyond | 7.91e+33 | 4.68e+29 | 5.4e-20 | 5705 | 2941 |
| wh | fp32 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 5892 | 2848 |
| wh | fp32 | `polygamma` | `k=4` | 141 of 49922 points returned inf or zero where a value exists | 3.77e+25 | 6.51e+21 | 3.68e-08 | 6301 | 2663 |
| wh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 18064 | 929 |
| wh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.17 | 5.92e+03 | 3.17 | — | 2160 ±9% | 7767 |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1.69e+38 | 4956 | 3385 |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | 0 | 3.39e+38 | 770 | 21798 |
| wh | fp32 | `rad2deg` | `default` | faithfully rounded; 63518 of 63518 points took the other neighbour | 0.696 | 0.643 | 5.93e+36 | 767 | 21875 |
| wh | fp32 | `rdiv` | `value=2.0` | 256 of 64770 points returned inf or zero where a value exists | 0.892 | 0.714 | 8.51e+37 | 807 | 20792 |
| wh | fp32 | `rdiv_bw` | `scalar=2.0` | 252 of 48490 points returned inf or zero where a value exists | 1.84 | 1.22 | 7.67e-20 | 13324 | 1259 |
| wh | fp32 | `reciprocal` | `default` | faithfully rounded; 64512 of 64514 points took the other neighbour | 0.892 | 0.714 | 8.51e+37 | 784 | 21399 |
| wh | fp32 | `reciprocal_bw` | `default` | 254 of 48386 points returned inf or zero where a value exists | 1.84 | 1.22 | 5.42e-20 | 10017 | 1675 |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 767 | 21861 |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 772 | 21723 |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 7863 | 2134 |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 2486 | 6749 |
| wh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 774 ±262% | 21666 |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 773 | 21705 |
| wh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 3.39e+38 | 774 | 21685 |
| wh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 771 | 21757 |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 772 | 21720 |
| wh | fp32 | `remainder` | `default` | worst pairing 1.43e+45 ULP; mean 1.68e+41 | 1.43e+45 | 1.68e+41 | — | 1015 | 16523 |
| wh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 778 | 21570 |
| wh | fp32 | `rpow` | `exponent=0.5` | faithfully rounded; 5624 of 49536 points took the other neighbour | 0.879 | 0.609 | 3.39e+38 | 1838 | 9130 |
| wh | fp32 | `rpow` | `exponent=1.0` | 1536 of 49536 points returned inf or zero where a value exists | 0 | 0 | 8.28e+34 | 1838 | 9129 |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists | 0.879 | 0.609 | 8.28e+34 | 1832 | 9159 |
| wh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | 0 | 1.18e-38 | 5694 | 2946 |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1.12 | 0.834 | 3.39e+38 | 846 | 19838 |
| wh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists | 7.19 | 4.49 | — | 11874 | 1413 |
| wh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | — | 985 | 17032 |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | — | 1005 ±9% | 16686 |
| wh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.3, worst 51.3 | 51.3 | 26.3 | — | 975 | 17200 |
| wh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 50.7 | 20.4 | — | 6200 | 2706 |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 0.000345; up to 2.64 ULP beyond | 2.64 | 1.31 | 0.000345 | 1077 | 15574 |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 0.000345; up to 2.64 ULP beyond | 2.64 | 1.31 | 0.000345 | 1072 | 15650 |
| wh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists | 8.39e+06 | 2.71e+04 | 0.447 | 4495 | 3732 |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 765 ±5% | 21941 |
| wh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 765 | 21934 |
| wh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists | 2.94 | 1.55 | 0.000121 | 1092 | 15367 |
| wh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 5.72e+06 | 841 | 2.98e-07 | 6216 | 2699 |
| wh | fp32 | `sin` | `default` | accurate to |x| <= 28; up to 2.2e+12 ULP beyond | 2.2e+12 | 2.68e+09 | 28 | 851 | 19716 |
| wh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 1.56e+53 | 2.03e+49 | 92.4 | 2561 | 6550 |
| wh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155; up to 2.21 ULP beyond | 2.21 | 1.17 | 0.0155 | 1101 | 15241 |
| wh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1.35 | 0.798 | 88.5 | 11881 | 1412 |
| wh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists | 8.21e+03 | 656 | — | 1398 ±12% | 12004 |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.00163; up to 2.95 ULP beyond | 2.95 | 1.34 | 0.00163 | 8683 | 1932 |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 3.39e+38 | 775 ±6% | 21642 |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 4242 | 3955 |
| wh | fp32 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 2.66 | 0.897 | 0.000462 | 822 | 20411 |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 1.79e-07 | 2570 | 6529 |
| wh | fp32 | `sqrt` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.867 | 0.827 | 3.39e+38 | 834 | 20106 |
| wh | fp32 | `sqrt_bw` | `default` | never within 2 ULP; mean 1.44, worst 2.24 | 2.24 | 1.44 | — | 12633 | 1328 |
| wh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1.84e+19 | 764 | 21952 |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1.69e+38 | 2495 | 6724 |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | — | 983 | 17070 |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | — | 998 | 16811 |
| wh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | — | 4211 | 3984 |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | — | 986 | 17009 |
| wh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | — | 997 | 16833 |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | — | 993 | 16898 |
| wh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 2.94 | 1.55 | 0.000121 | 1085 | 15458 |
| wh | fp32 | `tan` | `default` | accurate to |x| <= 3.92; up to 2.2e+12 ULP beyond | 2.2e+12 | 5.61e+09 | 3.92 | 1166 | 14392 |
| wh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists | 2.85e+45 | 5.11e+44 | 0.882 | 4363 | 3846 |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000439; up to 2.79 ULP beyond | 2.79 | 1.49 | 0.000439 | 930 | 18043 |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | — | 1769 | 9482 |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.35e+04 | 1.34e-08 | 1334 ±16% | 12577 |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.01e+05 | 7.42e-09 | 3387 | 4954 |
| wh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 3.39e+38 | 774 | 21669 |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 771 | 21762 |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 3.39e+38 | 3248 | 5166 |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 3.39e+38 | 770 | 21790 |
| wh | fp32 | `where` | `default` | bit-exact | 0 | 0 | — | 1350 | 12428 |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 1.71e-13 | 1381 ±11% | 12151 |
| wh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | — | 985 ±7% | 17033 |
| wh | fp32 | `xlogy_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.766 | 0.766 | — | 17461 ±22% | 961 |
