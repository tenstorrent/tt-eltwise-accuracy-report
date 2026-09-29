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
| bh | bf16 | `a3a9fb4229a` | 0.78.0 | f9adaa7ae3c4 |
| bh | fp32 | `a3a9fb4229a` | 0.78.0 | f9adaa7ae3c4 |
| wh | bf16 | `a3a9fb4229a` | 0.78.0 | 57e9901fdca0 |
| wh | fp32 | `a3a9fb4229a` | 0.78.0 | 57e9901fdca0 |

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
| `N of M points returned inf or zero where a value exists; the rest reach X ULP` | defects > 0. Leads, because those points carry no ULP and every figure below excludes them — but it no longer replaces the accuracy of the points that do score. Two defects once hid 9.02e+04 ULP on the other 64,512 |
| `N of M points returned a value where the reference is zero; the rest reach X ULP` | unflushed > 0. Also unscorable, and otherwise invisible: those points would read bit-exact. Carries the scorable points' worst error for the same reason |
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
7. Do not subtract `max_ulp` or `mean_ulp` from an index that lacks `rounded_frac` against
   one that has it. Those ULP figures are different quantities. `ttnn-accuracy compare`
   already refuses that; a nightly of two pre-`rounded_frac` indexes still scores ULP.

## Results — 953 variants

| Arch | Dtype | Op | Parameters | Verdict | Max ULP | Mean ULP | Rounded | Usable to | µs | Melem/s |
|---|---|---|---|---|---|---|---|---|---|---|
| bh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 ±19% | 87076 |
| bh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 599 | 28026 |
| bh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 279 | 60144 |
| bh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949; up to 2.7 ULP beyond | 2.7 | 0.744 | 0.996 | 0.949 | 3520 | 4766 |
| bh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1.41 | 0.596 | 0.973 | 3.39e+38 | 364 | 46104 |
| bh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists; the rest reach 3.26 ULP | 3.26 | 0.688 | 0.781 | 1.03 | 4025 | 4168 |
| bh | bf16 | `add` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 248 | 67716 |
| bh | bf16 | `add_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 244 ±5% | 68879 |
| bh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | 0.997 | — | 252 ±9% | 66561 |
| bh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.35e+03 | 2.11e+06 | 2.35e+03 | 0.999 | — | 323 | 51949 |
| bh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.4 | 126 | 19.4 | 1 | — | 327 | 51263 |
| bh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists; the rest reach 0.499 ULP | 0.499 | 0 | 1 | — | 279 | 60167 |
| bh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 2.76 ULP beyond | 2.76 | 0.787 | 0.992 | 0.938 | 3428 | 4894 |
| bh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1.32 | 0.615 | 0.991 | 3.39e+38 | 500 | 33542 |
| bh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists; the rest reach 1.33 ULP | 1.33 | 0.681 | 0.918 | 1.84e+19 | 579 | 28956 |
| bh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists; the rest reach 0.527 ULP | 0.527 | 0.513 | 1 | — | 210 | 79971 |
| bh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 3.56 | 200 | 3.56 | 0.901 | — | 255 | 65914 |
| bh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.89 | 248 | 4.89 | 0.508 | — | 2493 | 6728 |
| bh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists; the rest reach 2.23 ULP | 2.23 | 0.867 | 0.849 | 0.23 | 611 | 27453 |
| bh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists; the rest reach 2.08 ULP | 2.08 | 0.995 | 0.175 | — | 273 | 61506 |
| bh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists; the rest reach 8.17 ULP | 8.17 | 1 | 0.79 | 0.82 | 3838 | 4372 |
| bh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | 0.749 | — | 247 ±14% | 68057 |
| bh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | 0.749 | — | 242 | 69310 |
| bh | bf16 | `cbrt` | `default` | faithfully rounded; 846 of 65024 points took the other neighbour | 0.507 | 0.502 | 0.987 | 3.39e+38 | 218 | 76876 |
| bh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 | 87127 |
| bh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 194 ±10% | 86382 |
| bh | bf16 | `celu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1266 | 13250 |
| bh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 ±14% | 85841 |
| bh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85502 |
| bh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 ±8% | 84635 |
| bh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1097 | 15298 |
| bh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 88045 |
| bh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 200 ±9% | 84074 |
| bh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 ±7% | 87735 |
| bh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1096 | 15308 |
| bh | bf16 | `cos` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 26.5 | 0.983 | 1.31e+05 | 263 | 63675 |
| bh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists; the rest reach 7.45e+42 ULP | 7.45e+42 | 3.63e+39 | 0.85 | 2.62e+05 | 819 | 20479 |
| bh | bf16 | `cosh` | `default` | faithfully rounded; 4 of 33894 points took the other neighbour | 0.501 | 0.501 | 1 | 89 | 259 ±10% | 64793 |
| bh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists; the rest reach 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 3065 ±13% | 5474 |
| bh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists; the rest reach 0.998 ULP | 0.998 | 0.744 | 0.516 | 6.7e-37 | 188 | 89091 |
| bh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | 0.173 | — | 394 ±6% | 42558 |
| bh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists; the rest reach 1.02e+08 ULP | 1.02e+08 | 1.36e+04 | 0.674 | 0.996 | 3167 | 5298 |
| bh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 252 | 66671 |
| bh | bf16 | `div_bw` | `default` | bit-exact | 0.498 | 0 | 1 | — | 4884 | 3435 |
| bh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 661 | 25367 |
| bh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 256 | 65604 |
| bh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 252 | 66591 |
| bh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 191 | 87857 |
| bh | bf16 | `elu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 0.994 | 3.39e+38 | 1279 | 13116 |
| bh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 252 | 66644 |
| bh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 245 | 68374 |
| bh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88739 |
| bh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1.06 | 0.771 | 0.729 | 3.39e+38 | 189 | 88552 |
| bh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 0.976 | 1.5 | 1133 | 14810 |
| bh | bf16 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero; the rest reach 3.2e+28 ULP | 3.2e+28 | 3.23e+24 | 0.949 | 2.5 | 314 ±7% | 53473 |
| bh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 0.976 | 1.5 | 1135 | 14778 |
| bh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists; the rest reach 121 ULP | 121 | 6.4 | 0.394 | 1.31e-38 | 350 | 47929 |
| bh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 8.91 ULP beyond | 8.91 | 1.28 | 0.98 | 0.777 | 3612 | 4645 |
| bh | bf16 | `exp` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 0.983 | 3.39e+38 | 194 | 86627 |
| bh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6.08 | 6.08 | 3.05 | 0.314 | — | 190 | 88453 |
| bh | bf16 | `exp2` | `default` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 0.983 | 3.39e+38 | 228 | 73571 |
| bh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists; the rest reach 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 818 | 20502 |
| bh | bf16 | `exp_bw` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 0.983 | 3.39e+38 | 601 | 27912 |
| bh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 255 ±26% | 65777 |
| bh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists; the rest reach 125 ULP | 125 | 5.58 | 0.987 | 2.08 | 809 | 20726 |
| bh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 203 ±6% | 82771 |
| bh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | 0.996 | — | 1153 | 14549 |
| bh | bf16 | `fmod` | `default` | worst pairing 9.14e+35 ULP; mean 8.94e+34 | 9.14e+35 | 8.94e+34 | 0.673 | — | 265 | 63311 |
| bh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 186 | 90203 |
| bh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 244 ±17% | 68836 |
| bh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 245 ±10% | 68495 |
| bh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists; the rest reach 165 ULP | 165 | 9.18 | 0.999 | 2.33e-38 | 345 ±9% | 48606 |
| bh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists; the rest reach 1.14e+36 ULP | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 188 | 89415 |
| bh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.38; up to 3.5 ULP beyond | 3.5 | 1.37 | 0.998 | 8.38 | 712 ±5% | 23558 |
| bh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 187 | 89940 |
| bh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 245 ±13% | 68613 |
| bh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 240 ±20% | 69794 |
| bh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88816 |
| bh | bf16 | `hardmish` | `default` | faithfully rounded; 2587 of 65024 points took the other neighbour | 1 | 0.913 | 0.961 | 3.39e+38 | 191 | 87964 |
| bh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 ±12% | 88343 |
| bh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21882 |
| bh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.557 | 0.986 | 3.39e+38 | 194 | 86586 |
| bh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 1177 | 14252 |
| bh | bf16 | `hardswish` | `default` | 2 of 65024 points returned inf or zero where a value exists; the rest reach 2.14 ULP | 2.14 | 0.94 | 0.95 | 2.33e-38 | 208 | 80526 |
| bh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.13; up to 85.3 ULP beyond | 85.3 | 2.1 | 0.993 | 1.13 | 1650 | 10166 |
| bh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 ±5% | 87912 |
| bh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 980 | 17123 |
| bh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 | 84945 |
| bh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85397 |
| bh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 196 | 85632 |
| bh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 52.7 | 1.58 | 0.947 | — | 260 ±7% | 64573 |
| bh | bf16 | `hypot_bw` | `default` | worst pairing 74.6 ULP; mean 2.43 | 74.6 | 2.43 | 0.835 | — | 1509 | 11121 |
| bh | bf16 | `i0` | `default` | accurate to |x| <= 13.6; up to 255 ULP beyond | 255 | 56.3 | 0.952 | 13.6 | 173 | 97106 |
| bh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists; the rest reach 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 854 | 19656 |
| bh | bf16 | `i1` | `default` | 4 of 33904 points returned inf or zero where a value exists; the rest reach 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 442 | 37952 |
| bh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88397 |
| bh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 255 ±17% | 65808 |
| bh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 197 ±6% | 85058 |
| bh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 ±6% | 86494 |
| bh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 206 ±18% | 81452 |
| bh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 | 88020 |
| bh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88302 |
| bh | bf16 | `l1_loss` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 247 ±5% | 68008 |
| bh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | 0.955 | — | 255 ±10% | 65884 |
| bh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | 0.955 | — | 247 | 67889 |
| bh | bf16 | `ldexp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.898 | 0.898 | 0.983 | — | 1194 | 14046 |
| bh | bf16 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 246 | 68185 |
| bh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 245 | 68583 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88602 |
| bh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 15341 of 65024 points took the other neighbour | 0.96 | 0.741 | 0.761 | 3.39e+38 | 190 | 88457 |
| bh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±6% | 87368 |
| bh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 868 | 19321 |
| bh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 39.9 | 1.77e+03 | 39.9 | 1 | — | 322 | 52105 |
| bh | bf16 | `lerp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.506 | 0.506 | 0.992 | — | 1261 | 13301 |
| bh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88200 |
| bh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.412; up to 324 ULP beyond | 324 | 0.846 | 0.76 | 0.412 | 846 | 19833 |
| bh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | 0.173 | — | 794 | 21120 |
| bh | bf16 | `log` | `default` | faithfully rounded; 63 of 32512 points took the other neighbour | 0.78 | 0.534 | 0.998 | 3.39e+38 | 236 | 71208 |
| bh | bf16 | `log10` | `default` | faithfully rounded; 58 of 32512 points took the other neighbour | 0.785 | 0.543 | 0.998 | 3.39e+38 | 253 | 66276 |
| bh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 1.74 | 0.803 | 0.508 | 3.69e+37 | 2433 | 6895 |
| bh | bf16 | `log1p` | `default` | faithfully rounded; 109 of 48640 points took the other neighbour | 0.931 | 0.57 | 0.998 | 3.39e+38 | 260 | 64417 |
| bh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists; the rest reach 1.42 ULP | 1.42 | 0.744 | 0.982 | 8.47e+37 | 2399 | 6993 |
| bh | bf16 | `log2` | `default` | faithfully rounded; 59 of 32512 points took the other neighbour | 0.766 | 0.542 | 0.998 | 3.39e+38 | 257 ±10% | 65176 |
| bh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists; the rest reach 1.51 ULP | 1.51 | 0.847 | 0.57 | — | 2433 | 6895 |
| bh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists; the rest reach 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 1848 | 9078 |
| bh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.428; up to 6.39 ULP beyond | 6.39 | 1.38 | 0.958 | 0.428 | 224 | 74759 |
| bh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 1.71 | 0.722 | 0.976 | 3.39e+38 | 2678 | 6266 |
| bh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | 0.858 | — | 321 ±36% | 52308 |
| bh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | 0.871 | — | 382 | 43952 |
| bh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | 0.871 | — | 387 | 43351 |
| bh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41.4 ULP; mean 3.6 | 41.4 | 3.6 | 0.978 | — | 2505 | 6696 |
| bh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | 0.858 | — | 316 | 53129 |
| bh | bf16 | `logaddexp_bw` | `default` | worst pairing 62.5 ULP; mean 4.9 | 62.5 | 4.9 | 0.979 | — | 2161 ±65% | 7764 |
| bh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 258 | 65106 |
| bh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 246 | 68080 |
| bh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 | 86506 |
| bh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 196 ±18% | 85631 |
| bh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 251 | 66925 |
| bh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 250 ±13% | 67070 |
| bh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 251 ±8% | 66795 |
| bh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.67 | 0.985 | 0.395 | 334 | 50198 |
| bh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.729 | 0.975 | 0.996 | 3197 | 5247 |
| bh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.729 | 0.975 | 0.996 | 3810 | 4403 |
| bh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 247 | 68033 |
| bh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 243 ±9% | 68981 |
| bh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88280 |
| bh | bf16 | `mac` | `default` | worst pairing 63.5 ULP; mean 26.4 | 63.5 | 26.4 | 1 | — | 328 | 51156 |
| bh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | 1 | — | 2478 | 6771 |
| bh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 247 ±7% | 68033 |
| bh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | 1 | — | 2485 | 6752 |
| bh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 245 | 68577 |
| bh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists; the rest reach 1.49 ULP | 1.49 | 0.666 | 0.988 | 1.95e-38 | 256 | 65436 |
| bh | bf16 | `mse_loss` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 255 ±19% | 65854 |
| bh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 641 | 26172 |
| bh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists; the rest reach 724 ULP | 724 | 1.67 | 0.717 | 5.59e-17 | 4659 | 3601 |
| bh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.59e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 293 | 0.249 | 5.59e-17 | 3790 | 4426 |
| bh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 246 | 68143 |
| bh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 249 ±15% | 67314 |
| bh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 254 ±18% | 65923 |
| bh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 251 ±5% | 66882 |
| bh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 189 | 88952 |
| bh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | 0.56 | — | 1439 | 11662 |
| bh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 191 ±11% | 88025 |
| bh | bf16 | `polygamma` | `k=1` | 7 of 49922 points returned inf or zero where a value exists; the rest reach 1.02e+08 ULP | 1.02e+08 | 2.13e+05 | 0.97 | 0.996 | 334 | 50181 |
| bh | bf16 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists; the rest reach 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 396 | 42395 |
| bh | bf16 | `polygamma` | `k=4` | 142 of 49922 points returned inf or zero where a value exists; the rest reach 3.22e+09 ULP | 3.22e+09 | 1.66e+07 | 0.958 | 3.5 | 442 | 37974 |
| bh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists; the rest reach 3.57e+08 ULP | 3.57e+08 | 1.21e+06 | 0.969 | 4.47 | 3224 | 5204 |
| bh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | 0.982 | — | 391 | 42897 |
| bh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1182 | 14198 |
| bh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists; the rest reach 0 ULP | 0 | 0 | 1 | 4.68e-38 | 197 | 85214 |
| bh | bf16 | `rad2deg` | `default` | faithfully rounded; 31760 of 63518 points took the other neighbour | 0.992 | 0.75 | 0.5 | 5.9e+36 | 192 | 87354 |
| bh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists; the rest reach 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 190 | 88212 |
| bh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists; the rest reach 1.51 ULP | 1.51 | 0.837 | 0.602 | 7.67e-20 | 3158 ±13% | 5313 |
| bh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists; the rest reach 0.498 ULP | 0.498 | 0 | 1 | 8.47e+37 | 190 | 88138 |
| bh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists; the rest reach 1.51 ULP | 1.51 | 0.837 | 0.602 | 5.42e-20 | 2378 | 7055 |
| bh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88147 |
| bh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88118 |
| bh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1820 | 9219 |
| bh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 599 | 28023 |
| bh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 190 | 88360 |
| bh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 201 ±8% | 83314 |
| bh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 192 ±14% | 87158 |
| bh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 193 ±11% | 87088 |
| bh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 ±24% | 86480 |
| bh | bf16 | `remainder` | `default` | worst pairing 1.08e+36 ULP; mean 1.22e+35 | 1.08e+36 | 1.22e+35 | 0.672 | — | 264 ±15% | 63622 |
| bh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 199 ±10% | 84100 |
| bh | bf16 | `rpow` | `exponent=0.5` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 0.975 | 3.39e+38 | 391 | 42917 |
| bh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 378 ±8% | 44392 |
| bh | bf16 | `rpow` | `exponent=2.0` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 0.983 | 3.39e+38 | 375 | 44728 |
| bh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists; the rest reach 0 ULP | 0 | 0 | 1 | 1.18e-38 | 1358 | 12359 |
| bh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 244 | 68663 |
| bh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists; the rest reach 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 2798 | 5996 |
| bh | bf16 | `rsub` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 246 ±6% | 68126 |
| bh | bf16 | `rsub_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 243 | 69129 |
| bh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 207 | 81238 |
| bh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists; the rest reach 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 1458 | 11504 |
| bh | bf16 | `sigmoid` | `default` | faithfully rounded; 482 of 65024 points took the other neighbour | 0.873 | 0.522 | 0.994 | 3.39e+38 | 259 ±8% | 64785 |
| bh | bf16 | `sigmoid_accurate` | `default` | faithfully rounded; 482 of 65024 points took the other neighbour | 0.873 | 0.522 | 0.994 | 3.39e+38 | 252 ±13% | 66470 |
| bh | bf16 | `sigmoid_bw` | `default` | 331 of 65024 points returned inf or zero where a value exists; the rest reach 125 ULP | 125 | 4.81 | 0.988 | 1.76 | 1050 | 15977 |
| bh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 194 | 86534 |
| bh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 ±10% | 85831 |
| bh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists; the rest reach 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 173 | 97153 |
| bh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists; the rest reach 285 ULP | 285 | 1.3 | 0.979 | 0.863 | 1447 | 11593 |
| bh | bf16 | `sin` | `default` | accurate to |x| <= 2.62e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 73.5 | 0.992 | 2.62e+05 | 239 ±6% | 70216 |
| bh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists; the rest reach 1.94e+42 ULP | 1.94e+42 | 1.95e+39 | 0.84 | 1.31e+05 | 660 | 25421 |
| bh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 188 | 89237 |
| bh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists; the rest reach 0.501 ULP | 0.501 | 0.501 | 1 | 88.5 | 2776 | 6044 |
| bh | bf16 | `softcap` | `beta=50.0` | 1426 of 65024 points returned inf or zero where a value exists; the rest reach 0.718 ULP | 0.718 | 0.574 | 0.997 | — | 234 ±111% | 71830 |
| bh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists; the rest reach 0.775 ULP | 0.775 | 0.556 | 0.532 | 5.03 | 193 ±8% | 87100 |
| bh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 1.71 | 0.678 | 0.983 | 3.39e+38 | 2017 | 8316 |
| bh | bf16 | `softshrink` | `default` | faithfully rounded; 3968 of 65024 points took the other neighbour | 1 | 0.948 | 0.941 | 3.39e+38 | 198 | 84776 |
| bh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 985 | 17030 |
| bh | bf16 | `softsign` | `default` | 512 of 65024 points returned inf or zero where a value exists; the rest reach 1 ULP | 1 | 0.605 | 0.907 | 8.47e+37 | 191 ±15% | 87920 |
| bh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391; up to 81 ULP beyond | 81 | 3.24 | 0.856 | 0.00391 | 629 ±7% | 26687 |
| bh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 227 | 74027 |
| bh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1.14 | 0.699 | 0.723 | 3.39e+38 | 3010 | 5574 |
| bh | bf16 | `square` | `default` | faithfully rounded; 13970 of 48640 points took the other neighbour | 0.973 | 0.688 | 0.578 | 1.84e+19 | 192 ±16% | 87337 |
| bh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 608 ±8% | 27615 |
| bh | bf16 | `squared_difference` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 243 | 68947 |
| bh | bf16 | `squared_difference_` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 242 | 69275 |
| bh | bf16 | `squared_difference_bw` | `default` | faithfully rounded; 64842 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 1015 | 16528 |
| bh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | 0.997 | — | 251 | 66970 |
| bh | bf16 | `subtract` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 249 ±7% | 67478 |
| bh | bf16 | `subtract_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 257 ±22% | 65230 |
| bh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists; the rest reach 0.888 ULP | 0.888 | 0.582 | 0.993 | 2.33e-38 | 173 | 96748 |
| bh | bf16 | `tan` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 49.1 | 0.985 | 1.31e+05 | 214 ±9% | 78417 |
| bh | bf16 | `tan_bw` | `default` | 22759 of 65024 points returned inf or zero where a value exists; the rest reach 3.1e+40 ULP | 3.1e+40 | 4.91e+37 | 0.841 | 1.13 | 958 | 17507 |
| bh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists; the rest reach 0.811 ULP | 0.811 | 0.586 | 0.997 | — | 191 | 88069 |
| bh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 1.93 | 0.996 | 17.2 | 617 | 27212 |
| bh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.38 | 0.981 | 1.35e-08 | 171 ±8% | 98158 |
| bh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.52 | 0.937 | 7.45e-09 | 773 | 21700 |
| bh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 195 ±7% | 86166 |
| bh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 200 ±9% | 84055 |
| bh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 761 | 22035 |
| bh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 198 | 84715 |
| bh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 324 | 51744 |
| bh | bf16 | `xielu` | `default` | 1 of 56847 points returned inf or zero where a value exists; the rest reach 0.5 ULP | 0.5 | 0.5 | 1 | 2.33e-38 | 433 | 38704 |
| bh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | 0.11 | — | 258 | 65090 |
| bh | bf16 | `xlogy_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.78 | 0.78 | 0.998 | — | 4107 | 4085 |
| bh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48189 |
| bh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1134 | 14801 |
| bh | fp32 | `acos` | `default` | within 2 ULP everywhere | 1.55 | 0.865 | 0.923 | 1 | 467 | 35901 |
| bh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.988 | 0.926 | 6820 | 2460 |
| bh | fp32 | `acosh` | `default` | never within 2 ULP; mean 0.76, worst 2.47 | 2.47 | 0.76 | 0.782 | — | 451 | 37191 |
| bh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists; the rest reach 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 7800 | 2151 |
| bh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 482 | 34797 |
| bh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 476 | 35227 |
| bh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 482 ±6% | 34781 |
| bh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | 0.852 | — | 649 | 25835 |
| bh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | 0.985 | — | 647 | 25930 |
| bh | fp32 | `asin` | `default` | within 2 ULP everywhere | 1.77 | 0.795 | 0.985 | 1 | 451 | 37186 |
| bh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.988 | 0.926 | 6696 | 2505 |
| bh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 1.64 | 0.774 | 0.881 | 3.39e+38 | 601 | 27902 |
| bh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists; the rest reach 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 1196 | 14033 |
| bh | fp32 | `atan` | `default` | accurate to |x| <= 0.902; up to 2.34 ULP beyond | 2.34 | 0.806 | 0.955 | 0.902 | 402 | 41760 |
| bh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | 0.857 | — | 500 | 33546 |
| bh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 1.66e+05 | 1.64e+07 | 1.66e+05 | 0.587 | — | 4895 | 3428 |
| bh | fp32 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists; the rest reach 9.02e+04 ULP | 9.02e+04 | 3.25e+03 | 0.859 | 1.02 | 1142 | 14685 |
| bh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000229; up to 3.04 ULP beyond | 3.04 | 1.51 | 0.912 | 0.000229 | 335 | 50007 |
| bh | fp32 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists; the rest reach 9.02e+04 ULP | 9.02e+04 | 3.31e+03 | 0.876 | 0.681 | 7511 | 2234 |
| bh | fp32 | `bias_gelu` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | 0.746 | — | 477 | 35138 |
| bh | fp32 | `bias_gelu_` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | 0.746 | — | 470 | 35667 |
| bh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38; up to 2.55 ULP beyond | 2.55 | 1.76 | 0.521 | 2.26e-38 | 383 | 43818 |
| bh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 ±35% | 47800 |
| bh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 0.99 | 3.39e+38 | 410 | 40883 |
| bh | fp32 | `celu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2503 | 6703 |
| bh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48270 |
| bh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 ±10% | 47822 |
| bh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 ±17% | 48078 |
| bh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2173 | 7719 |
| bh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 ±20% | 47950 |
| bh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 354 ±8% | 47418 |
| bh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48543 |
| bh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2170 | 7731 |
| bh | fp32 | `cos` | `default` | accurate to |x| <= 92.4; up to 3.3e+12 ULP beyond | 3.3e+12 | 3.33e+09 | 0.919 | 92.4 | 401 ±9% | 41804 |
| bh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists; the rest reach 2.68e+51 ULP | 2.68e+51 | 2.8e+48 | 0.835 | 28 | 1503 | 11166 |
| bh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1.35 | 0.791 | 0.98 | 89.1 | 392 ±5% | 42764 |
| bh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists; the rest reach 2.2 ULP | 2.2 | 1.16 | 0.943 | 0.0155 | 6126 | 2739 |
| bh | fp32 | `deg2rad` | `default` | faithfully rounded; 63542 of 65024 points took the other neighbour | 0.63 | 0.595 | 0.906 | 3.4e+38 | 348 | 48155 |
| bh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | 0.00543 | — | 728 | 23050 |
| bh | fp32 | `digamma_bw` | `default` | 257 of 64769 points returned inf or zero where a value exists; the rest reach 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.324 | 5.4e-20 | 6562 ±20% | 2557 |
| bh | fp32 | `div` | `default` | worst pairing 2.44 ULP; mean 1.44 | 2.44 | 1.44 | 0.894 | — | 491 | 34158 |
| bh | fp32 | `div_bw` | `default` | worst pairing 7.95e+04 ULP; mean 7.95e+04 | 7.95e+04 | 7.95e+04 | 0.89 | — | 9441 | 1777 |
| bh | fp32 | `div_no_nan` | `default` | worst pairing 9.24e+04 ULP; mean 4.51e+04 | 9.24e+04 | 4.51e+04 | 0.702 | — | 1589 | 10558 |
| bh | fp32 | `divide` | `default` | worst pairing 2.44 ULP; mean 1.44 | 2.44 | 1.44 | 0.894 | — | 494 ±5% | 33952 |
| bh | fp32 | `divide_` | `default` | worst pairing 2.44 ULP; mean 1.44 | 2.44 | 1.44 | 0.894 | — | 495 | 33889 |
| bh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 0.99 | 3.39e+38 | 408 | 41099 |
| bh | fp32 | `elu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 0.999 | 3.39e+38 | 2491 | 6736 |
| bh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 475 | 35341 |
| bh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 474 ±9% | 35372 |
| bh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48420 |
| bh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 7.38 ULP beyond | 7.38 | 1.36 | 0.672 | 0.000334 | 528 | 31798 |
| bh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.956 | 0.348 | 2182 | 7688 |
| bh | fp32 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero; the rest reach 2.1e+33 ULP | 2.1e+33 | 1.71e+29 | 0.329 | — | 363 ±14% | 46272 |
| bh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.956 | 0.348 | 2177 | 7708 |
| bh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists; the rest reach 7.96e+06 ULP | 7.96e+06 | 2.62e+05 | 0.0002 | 1.32e-38 | 359 | 46721 |
| bh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456; up to 1.03e+06 ULP beyond | 1.03e+06 | 1.68e+03 | 0.913 | 0.000456 | 6735 | 2491 |
| bh | fp32 | `exp` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 0.997 | 3.39e+38 | 392 ±6% | 42852 |
| bh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | 0.309 | — | 354 ±7% | 47346 |
| bh | fp32 | `exp2` | `default` | faithfully rounded; 6044 of 49536 points took the other neighbour | 0.965 | 0.626 | 0.992 | 3.39e+38 | 384 | 43732 |
| bh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists; the rest reach 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 1506 | 11141 |
| bh | fp32 | `exp_bw` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 0.997 | 3.39e+38 | 1183 | 14183 |
| bh | fp32 | `expm1` | `default` | faithfully rounded; 5257 of 49458 points took the other neighbour | 0.997 | 0.565 | 0.997 | 3.39e+38 | 456 ±7% | 36775 |
| bh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists; the rest reach 4.91e+06 ULP | 4.91e+06 | 1.14e+04 | 0.985 | 1.39 | 1576 | 10644 |
| bh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 | 48077 |
| bh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.56e+03 | 4.19e+06 | 1.56e+03 | 0.974 | — | 2256 | 7437 |
| bh | fp32 | `fmod` | `default` | worst pairing 1.61e+45 ULP; mean 2.55e+41 | 1.61e+45 | 2.55e+41 | 0.596 | — | 499 | 33635 |
| bh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 354 | 47458 |
| bh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 476 | 35254 |
| bh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 476 | 35278 |
| bh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists; the rest reach 1.71e+08 ULP | 1.71e+08 | 1.7e+05 | 0.963 | 0.208 | 367 | 45685 |
| bh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists; the rest reach 7.45e+40 ULP | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 352 | 47630 |
| bh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0296; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.962 | 0.0296 | 827 | 20291 |
| bh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 359 ±8% | 46748 |
| bh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 478 ±6% | 35065 |
| bh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 477 ±14% | 35175 |
| bh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 352 | 47702 |
| bh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48368 |
| bh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48296 |
| bh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1470 | 11409 |
| bh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25; up to 4.89e+06 ULP beyond | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 348 | 48159 |
| bh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2293 | 7316 |
| bh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36; up to 7.34e+06 ULP beyond | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 372 ±6% | 45101 |
| bh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 3235 | 5187 |
| bh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 364 | 46034 |
| bh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1933 | 8681 |
| bh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 360 ±49% | 46657 |
| bh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 358 ±11% | 46884 |
| bh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48223 |
| bh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | 0.965 | — | 503 | 33380 |
| bh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists; the rest reach 4.86e+06 ULP | 4.86e+06 | 4.48e+04 | 0.826 | — | 2975 | 5639 |
| bh | fp32 | `i0` | `default` | accurate to |x| <= 2.43; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 0.959 | 2.43 | 404 | 41566 |
| bh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00362; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.1e+04 | 0.928 | 0.00362 | 1356 | 12368 |
| bh | fp32 | `i1` | `default` | accurate to |x| <= 0.00362; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.1e+04 | 0.928 | 0.00362 | 557 | 30129 |
| bh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48741 |
| bh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 481 | 34896 |
| bh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47796 |
| bh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48381 |
| bh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 ±18% | 48025 |
| bh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 ±12% | 48542 |
| bh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48421 |
| bh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 475 | 35329 |
| bh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | 0.953 | — | 487 | 34445 |
| bh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | 0.953 | — | 486 | 34537 |
| bh | fp32 | `ldexp_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.742 | 0.742 | 0.996 | — | 2212 | 7584 |
| bh | fp32 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 478 | 35074 |
| bh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 475 | 35323 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47885 |
| bh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 31672 of 65024 points took the other neighbour | 0.84 | 0.747 | 0.868 | 3.39e+38 | 350 | 47884 |
| bh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47884 |
| bh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 1650 | 10170 |
| bh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | 0.962 | — | 646 | 25980 |
| bh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 2501 | 6709 |
| bh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47887 |
| bh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 1336 | 12554 |
| bh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | 0.00543 | — | 1531 ±13% | 10958 |
| bh | fp32 | `log` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.961 | 0.562 | 0.952 | 3.39e+38 | 405 | 41416 |
| bh | fp32 | `log10` | `default` | accurate to |x| <= 0.336; up to 2.13 ULP beyond | 2.13 | 1.36 | 0.641 | 0.336 | 400 ±5% | 41896 |
| bh | fp32 | `log10_bw` | `default` | accurate to |x| <= 4.42e+30; up to 9.02e+04 ULP beyond | 9.02e+04 | 1.73e+03 | 0.667 | 4.42e+30 | 4693 ±7% | 3575 |
| bh | fp32 | `log1p` | `default` | faithfully rounded; 20111 of 48640 points took the other neighbour | 0.984 | 0.558 | 0.982 | 3.39e+38 | 398 | 42151 |
| bh | fp32 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists; the rest reach 9.02e+04 ULP | 9.02e+04 | 2.96e+03 | 0.89 | 2.02e+31 | 4700 | 3570 |
| bh | fp32 | `log2` | `default` | accurate to |x| <= 0.704; up to 2.43 ULP beyond | 2.43 | 0.511 | 0.995 | 0.704 | 398 | 42120 |
| bh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists; the rest reach 9.02e+04 ULP | 9.02e+04 | 1.76e+03 | 0.684 | — | 4689 | 3578 |
| bh | fp32 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists; the rest reach 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 3546 | 4731 |
| bh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | 0.481 | — | 473 | 35503 |
| bh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.163; up to 2.93 ULP beyond | 2.93 | 1.28 | 0.953 | 0.163 | 5184 | 3236 |
| bh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | 0.726 | — | 586 | 28632 |
| bh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | 0.716 | — | 516 ±6% | 32485 |
| bh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | 0.716 | — | 508 | 32994 |
| bh | fp32 | `logaddexp2_bw` | `default` | worst pairing 9.02e+04 ULP; mean 5.6e+04 | 9.02e+04 | 5.6e+04 | 0.959 | — | 4728 | 3548 |
| bh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | 0.726 | — | 591 | 28389 |
| bh | fp32 | `logaddexp_bw` | `default` | worst pairing 8.98e+04 ULP; mean 4.46e+04 | 8.98e+04 | 4.46e+04 | 0.962 | — | 4261 | 3937 |
| bh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 484 | 34631 |
| bh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 487 | 34422 |
| bh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48400 |
| bh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 343 | 48853 |
| bh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 487 | 34459 |
| bh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 485 ±24% | 34588 |
| bh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 484 | 34646 |
| bh | fp32 | `logit` | `default` | accurate to |x| <= 0.266; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.941 | 0.266 | 364 | 46070 |
| bh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.97e-08; up to 2.28 ULP beyond | 2.28 | 0.789 | 0.874 | 2.97e-08 | 6313 | 2657 |
| bh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.97e-08; up to 2.28 ULP beyond | 2.28 | 0.789 | 0.874 | 2.97e-08 | 7486 | 2241 |
| bh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 478 | 35135 |
| bh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 476 | 35250 |
| bh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48177 |
| bh | fp32 | `mac` | `default` | worst pairing 4.22e+06 ULP; mean 7.31e+05 | 4.22e+06 | 7.31e+05 | 0.974 | — | 646 ±709% | 25962 |
| bh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | 1 | — | 4847 | 3462 |
| bh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 475 | 35356 |
| bh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | 1 | — | 4858 | 3453 |
| bh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 482 ±6% | 34811 |
| bh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists; the rest reach 7 ULP | 7 | 2.58 | 0.953 | 1.49e-07 | 454 ±8% | 36958 |
| bh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 480 ±8% | 34938 |
| bh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1255 | 13366 |
| bh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists; the rest reach 3.51e+07 ULP | 3.51e+07 | 9.16e+03 | 0.476 | 5.55e-17 | 7969 | 2105 |
| bh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 0.0102 | 5.55e-17 | 7380 ±27% | 2273 |
| bh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 482 | 34812 |
| bh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 471 | 35590 |
| bh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 476 | 35254 |
| bh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 474 | 35375 |
| bh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48387 |
| bh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | 0.5 | — | 2869 | 5847 |
| bh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 344 | 48739 |
| bh | fp32 | `polygamma` | `k=1` | 1 of 49922 points returned inf or zero where a value exists; the rest reach 7.91e+33 ULP | 7.91e+33 | 4.68e+29 | 0.467 | 5.4e-20 | 1032 | 16261 |
| bh | fp32 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists; the rest reach 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 1071 | 15666 |
| bh | fp32 | `polygamma` | `k=4` | 141 of 49922 points returned inf or zero where a value exists; the rest reach 3.77e+25 ULP | 3.77e+25 | 6.51e+21 | 0.138 | 3.68e-08 | 1089 | 15405 |
| bh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists; the rest reach 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.183 | 1.79e-13 | 6606 | 2540 |
| bh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.17 | 5.92e+03 | 3.17 | 0.992 | — | 652 | 25724 |
| bh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2253 | 7448 |
| bh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 352 | 47730 |
| bh | fp32 | `rad2deg` | `default` | faithfully rounded; 63518 of 63518 points took the other neighbour | 0.696 | 0.643 | 0.858 | 5.93e+36 | 350 | 47880 |
| bh | fp32 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists; the rest reach 1.49 ULP | 1.49 | 1.19 | 0.664 | 8.47e+37 | 348 | 48277 |
| bh | fp32 | `rdiv_bw` | `scalar=2.0` | 254 of 48490 points returned inf or zero where a value exists; the rest reach 9.02e+04 ULP | 9.02e+04 | 1.92e+03 | 0.705 | 7.67e-20 | 6094 | 2753 |
| bh | fp32 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists; the rest reach 9.02e+04 ULP | 9.02e+04 | 1.72e+03 | 0.89 | 2.02e+31 | 347 | 48324 |
| bh | fp32 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists; the rest reach 9.02e+04 ULP | 9.02e+04 | 1.92e+03 | 0.705 | 5.42e-20 | 4550 | 3687 |
| bh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48507 |
| bh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 348 | 48173 |
| bh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3599 | 4662 |
| bh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1145 | 14658 |
| bh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48375 |
| bh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 | 48580 |
| bh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 349 ±10% | 48101 |
| bh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48385 |
| bh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 345 ±6% | 48669 |
| bh | fp32 | `remainder` | `default` | worst pairing 8.92e+44 ULP; mean 2.49e+41 | 8.92e+44 | 2.49e+41 | 0.596 | — | 493 ±7% | 34065 |
| bh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 ±7% | 48386 |
| bh | fp32 | `rpow` | `exponent=0.5` | faithfully rounded; 5624 of 49536 points took the other neighbour | 0.879 | 0.609 | 0.995 | 3.39e+38 | 624 | 26879 |
| bh | fp32 | `rpow` | `exponent=1.0` | 1536 of 49536 points returned inf or zero where a value exists; the rest reach 0 ULP | 0 | 0 | 1 | 8.28e+34 | 621 | 27021 |
| bh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists; the rest reach 0.879 ULP | 0.879 | 0.609 | 0.996 | 8.28e+34 | 628 | 26718 |
| bh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists; the rest reach 0 ULP | 0 | 0 | 1 | 1.18e-38 | 2602 | 6449 |
| bh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1.12 | 0.834 | 0.864 | 3.39e+38 | 383 ±6% | 43764 |
| bh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists; the rest reach 7.19 ULP | 7.19 | 4.49 | 0.293 | — | 5227 | 3210 |
| bh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 478 | 35102 |
| bh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 475 | 35357 |
| bh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.3, worst 51.3 | 51.3 | 26.3 | 0 | — | 428 | 39187 |
| bh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists; the rest reach 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 2817 | 5956 |
| bh | fp32 | `sigmoid` | `default` | accurate to |x| <= 1.59e-05; up to 3.31 ULP beyond | 3.31 | 1.59 | 0.953 | 1.59e-05 | 421 | 39868 |
| bh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 1.59e-05; up to 3.31 ULP beyond | 3.31 | 1.59 | 0.953 | 1.59e-05 | 421 | 39860 |
| bh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists; the rest reach 8.39e+06 ULP | 8.39e+06 | 2.54e+04 | 0.973 | 0.283 | 2016 | 8321 |
| bh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 359 ±14% | 46739 |
| bh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 346 | 48501 |
| bh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists; the rest reach 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 425 | 39474 |
| bh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists; the rest reach 5.72e+06 ULP | 5.72e+06 | 842 | 0.934 | 2.98e-07 | 2796 | 6000 |
| bh | fp32 | `sin` | `default` | accurate to |x| <= 28; up to 2.2e+12 ULP beyond | 2.2e+12 | 2.68e+09 | 0.955 | 28 | 376 | 44639 |
| bh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists; the rest reach 1.56e+53 ULP | 1.56e+53 | 2.03e+49 | 0.802 | 92.4 | 1185 | 14161 |
| bh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155; up to 2.2 ULP beyond | 2.2 | 1.16 | 0.943 | 0.0155 | 463 ±9% | 36272 |
| bh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists; the rest reach 1.35 ULP | 1.35 | 0.791 | 0.98 | 88.5 | 5373 | 3123 |
| bh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists; the rest reach 8.21e+03 ULP | 8.21e+03 | 656 | 0.49 | — | 410 ±6% | 40945 |
| bh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.164; up to 2.93 ULP beyond | 2.93 | 1.33 | 0.954 | 0.164 | 3936 | 4262 |
| bh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 357 | 46985 |
| bh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1948 | 8611 |
| bh | fp32 | `softsign` | `default` | 512 of 65024 points returned inf or zero where a value exists; the rest reach 3 ULP | 3 | 1.28 | 0.863 | 1.21e-05 | 355 | 47226 |
| bh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 1153 | 14554 |
| bh | fp32 | `sqrt` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.867 | 0.827 | 0.869 | 3.39e+38 | 366 | 45786 |
| bh | fp32 | `sqrt_bw` | `default` | never within 2 ULP; mean 1.4, worst 2.19 | 2.19 | 1.4 | 0.708 | — | 5788 | 2899 |
| bh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 344 | 48713 |
| bh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1138 | 14744 |
| bh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 481 | 34861 |
| bh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 483 ±10% | 34726 |
| bh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1926 | 8709 |
| bh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 481 | 34871 |
| bh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 479 | 35021 |
| bh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 471 | 35655 |
| bh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists; the rest reach 3.61 ULP | 3.61 | 1.78 | 0.944 | 2.01e-05 | 425 | 39441 |
| bh | fp32 | `tan` | `default` | accurate to |x| <= 3.92; up to 2.2e+12 ULP beyond | 2.2e+12 | 5.61e+09 | 0.938 | 3.92 | 466 ±6% | 36037 |
| bh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists; the rest reach 2.85e+45 ULP | 2.85e+45 | 5.11e+44 | 0.865 | 0.852 | 1917 | 8754 |
| bh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000946; up to 2.6 ULP beyond | 2.6 | 1.45 | 0.98 | 0.000946 | 389 | 43122 |
| bh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | 0.485 | — | 843 | 19897 |
| bh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.23e+05 | 0.91 | 1.34e-08 | 361 | 46538 |
| bh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.02e+05 | 0.901 | 7.42e-09 | 1506 | 11140 |
| bh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 350 | 47922 |
| bh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 351 | 47789 |
| bh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1475 | 11372 |
| bh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 347 | 48327 |
| bh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 646 | 25983 |
| bh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 426 | 39384 |
| bh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | 1.15e-05 | — | 484 | 34682 |
| bh | fp32 | `xlogy_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.766 | 0.766 | 0.951 | — | 8019 | 2092 |
| wh | bf16 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 ±10% | 42566 |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1307 ±18% | 12834 |
| wh | bf16 | `acos` | `default` | bit-exact | 0.5 | 0 | 1 | 1 | 680 | 24663 |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.949; up to 2.7 ULP beyond | 2.7 | 0.744 | 0.996 | 0.949 | 7469 | 2246 |
| wh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1.41 | 0.596 | 0.973 | 3.39e+38 | 924 | 18153 |
| wh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists; the rest reach 3.26 ULP | 3.26 | 0.688 | 0.781 | 1.03 | 8632 | 1944 |
| wh | bf16 | `add` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 534 ±6% | 31443 |
| wh | bf16 | `add_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 555 ±6% | 30222 |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | 0.997 | — | 542 ±7% | 30964 |
| wh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.35e+03 | 2.11e+06 | 2.35e+03 | 0.999 | — | 688 | 24399 |
| wh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.4 | 126 | 19.4 | 1 | — | 686 | 24471 |
| wh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists | 0.499 | 0 | 1 | — | 675 | 24867 |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 2.76 ULP beyond | 2.76 | 0.787 | 0.992 | 0.938 | 7424 | 2260 |
| wh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1.32 | 0.619 | 0.992 | 3.39e+38 | 1411 ±16% | 11888 |
| wh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists; the rest reach 1.33 ULP | 1.33 | 0.681 | 0.918 | 1.84e+19 | 1356 | 12369 |
| wh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.527 | 0.513 | 1 | — | 602 ±11% | 27882 |
| wh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 2.51 | 200 | 2.51 | 0.998 | — | 550 ±14% | 30496 |
| wh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.88 | 248 | 4.88 | 0.522 | — | 5517 | 3041 |
| wh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists; the rest reach 2.23 ULP | 2.23 | 0.867 | 0.854 | 0.23 | 1325 | 12664 |
| wh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists | 2.08 | 0.995 | 0.081 | — | 694 ±11% | 24189 |
| wh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists; the rest reach 8.17 ULP | 8.17 | 1.02 | 0.84 | 0.82 | 8437 | 1988 |
| wh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | 0.749 | — | 536 | 31315 |
| wh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | 0.749 | — | 544 | 30824 |
| wh | bf16 | `cbrt` | `default` | faithfully rounded; 846 of 65024 points took the other neighbour | 0.507 | 0.502 | 0.987 | 3.39e+38 | 413 ±8% | 40606 |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 ±9% | 41798 |
| wh | bf16 | `celu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 482 ±10% | 34804 |
| wh | bf16 | `celu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2718 ±24% | 6173 |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 408 | 41077 |
| wh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 ±7% | 42567 |
| wh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 | 42358 |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2457 | 6827 |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 | 41616 |
| wh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 ±6% | 42607 |
| wh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 393 | 42655 |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2449 | 6851 |
| wh | bf16 | `cos` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 26.5 | 0.983 | 1.31e+05 | 427 ±7% | 39299 |
| wh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists; the rest reach 7.45e+42 ULP | 7.45e+42 | 3.63e+39 | 0.85 | 2.62e+05 | 1692 ±6% | 9918 |
| wh | bf16 | `cosh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 446 ±10% | 37651 |
| wh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists; the rest reach 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 6675 | 2513 |
| wh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.998 | 0.744 | 0.516 | 6.7e-37 | 405 ±7% | 41376 |
| wh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | 0.173 | — | 1183 ±14% | 14179 |
| wh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists; the rest reach 1.02e+08 ULP | 1.02e+08 | 1.32e+04 | 0.664 | 1 | 12072 | 1390 |
| wh | bf16 | `div` | `default` | bit-exact | 0.498 | 0 | 1 | — | 556 ±9% | 30161 |
| wh | bf16 | `div_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.512 | 0.512 | 0.984 | — | 10489 | 1600 |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0.498 | 0 | 1 | — | 1457 | 11515 |
| wh | bf16 | `divide` | `default` | bit-exact | 0.498 | 0 | 1 | — | 559 ±8% | 29993 |
| wh | bf16 | `divide_` | `default` | bit-exact | 0.498 | 0 | 1 | — | 569 ±5% | 29463 |
| wh | bf16 | `elu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 482 ±12% | 34832 |
| wh | bf16 | `elu_bw` | `default` | faithfully rounded; 734 of 65024 points took the other neighbour | 0.892 | 0.52 | 0.994 | 3.39e+38 | 2738 | 6129 |
| wh | bf16 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 533 | 31480 |
| wh | bf16 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 542 | 30957 |
| wh | bf16 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 394 | 42558 |
| wh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1.06 | 0.771 | 0.729 | 3.39e+38 | 568 ±11% | 29562 |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 0.976 | 1.5 | 2444 | 6865 |
| wh | bf16 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero | 3.19e+28 | 3.23e+24 | 0.969 | 2.5 | 818 | 20507 |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.5; up to 112 ULP beyond | 112 | 4.97 | 0.976 | 1.5 | 2452 | 6843 |
| wh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists | 121 | 6.4 | 0.394 | 1.31e-38 | 890 | 18856 |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 8.91 ULP beyond | 8.91 | 1.28 | 0.98 | 0.777 | 7856 | 2136 |
| wh | bf16 | `exp` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 0.983 | 3.39e+38 | 424 ±9% | 39563 |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6.08 | 6.08 | 3.05 | 0.314 | — | 396 ±5% | 42343 |
| wh | bf16 | `exp2` | `default` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 0.983 | 3.39e+38 | 426 ±8% | 39381 |
| wh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists; the rest reach 1.89 ULP | 1.89 | 0.942 | 0.945 | 128 | 1695 | 9895 |
| wh | bf16 | `exp_bw` | `default` | faithfully rounded; 1153 of 49458 points took the other neighbour | 0.892 | 0.556 | 0.983 | 3.39e+38 | 1353 ±16% | 12398 |
| wh | bf16 | `expm1` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 461 ±14% | 36384 |
| wh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists; the rest reach 125 ULP | 125 | 5.58 | 0.987 | 2.09 | 1706 ±27% | 9835 |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 412 ±6% | 40729 |
| wh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | 0.996 | — | 2521 | 6654 |
| wh | bf16 | `fmod` | `default` | worst pairing 6.65e+35 ULP; mean 7.8e+34 | 6.65e+35 | 7.8e+34 | 0.853 | — | 655 | 25613 |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 405 ±11% | 41390 |
| wh | bf16 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 533 | 31458 |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 552 ±25% | 30403 |
| wh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists | 165 | 9.18 | 0.999 | 2.33e-38 | 810 | 20710 |
| wh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists | 1.14e+36 | 1.82e+34 | 0.501 | 2.33e-38 | 402 ±7% | 41774 |
| wh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.44; up to 3.5 ULP beyond | 3.5 | 1.37 | 0.998 | 8.44 | 1775 | 9452 |
| wh | bf16 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±9% | 41531 |
| wh | bf16 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 532 | 31561 |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 551 | 30471 |
| wh | bf16 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 406 | 41327 |
| wh | bf16 | `hardmish` | `default` | faithfully rounded; 2587 of 65024 points took the other neighbour | 1 | 0.913 | 0.961 | 3.39e+38 | 402 ±6% | 41758 |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 410 ±12% | 40873 |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1676 | 10012 |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.557 | 0.986 | 3.39e+38 | 408 ±6% | 41140 |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0.333 | 0 | 1 | 3.39e+38 | 2537 | 6613 |
| wh | bf16 | `hardswish` | `default` | 2 of 65024 points returned inf or zero where a value exists | 2.14 | 0.94 | 0.95 | 2.33e-38 | 417 ±11% | 40234 |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.12; up to 85.3 ULP beyond | 85.3 | 2.1 | 0.993 | 1.12 | 3532 | 4751 |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 ±6% | 41811 |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2155 | 7784 |
| wh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 408 ±9% | 41082 |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 422 ±10% | 39757 |
| wh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 415 ±14% | 40458 |
| wh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 52.7 | 1.58 | 0.947 | — | 547 | 30698 |
| wh | bf16 | `hypot_bw` | `default` | worst pairing 74.6 ULP; mean 2.42 | 74.6 | 2.42 | 0.833 | — | 3364 | 4987 |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.6; up to 255 ULP beyond | 255 | 56.3 | 0.952 | 13.6 | 451 ±8% | 37200 |
| wh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists; the rest reach 218 ULP | 218 | 8.58 | 0.993 | 2.33e-38 | 2086 | 8042 |
| wh | bf16 | `i1` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.58 | 0.993 | 2.33e-38 | 1196 | 14029 |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 ±6% | 41589 |
| wh | bf16 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 550 ±6% | 30490 |
| wh | bf16 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 407 | 41218 |
| wh | bf16 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 396 ±7% | 42406 |
| wh | bf16 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±7% | 42026 |
| wh | bf16 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±8% | 41533 |
| wh | bf16 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 403 ±5% | 41617 |
| wh | bf16 | `l1_loss` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 543 ±6% | 30911 |
| wh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | 0.955 | — | 548 ±6% | 30625 |
| wh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.4 | 255 | 94.4 | 0.955 | — | 558 | 30051 |
| wh | bf16 | `ldexp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.898 | 0.898 | 0.983 | — | 2725 | 6156 |
| wh | bf16 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 544 | 30820 |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 547 ±5% | 30685 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 400 ±7% | 41892 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 15341 of 65024 points took the other neighbour | 0.96 | 0.741 | 0.761 | 3.39e+38 | 397 | 42230 |
| wh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 400 ±6% | 41959 |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0.16 | 0 | 1 | 3.39e+38 | 1830 | 9170 |
| wh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 39.9 | 1.77e+03 | 39.9 | 1 | — | 683 | 24561 |
| wh | bf16 | `lerp_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.506 | 0.506 | 0.992 | — | 2811 | 5969 |
| wh | bf16 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 399 ±7% | 42095 |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.414; up to 324 ULP beyond | 324 | 0.846 | 0.76 | 0.414 | 2420 | 6934 |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | 0.173 | — | 2074 | 8088 |
| wh | bf16 | `log` | `default` | faithfully rounded; 63 of 32512 points took the other neighbour | 0.78 | 0.534 | 0.998 | 3.39e+38 | 428 ±9% | 39178 |
| wh | bf16 | `log10` | `default` | faithfully rounded; 58 of 32512 points took the other neighbour | 0.785 | 0.543 | 0.998 | 3.39e+38 | 428 ±15% | 39239 |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 1.74 | 0.796 | 0.524 | 3.69e+37 | 5280 | 3178 |
| wh | bf16 | `log1p` | `default` | faithfully rounded; 109 of 48640 points took the other neighbour | 0.931 | 0.57 | 0.998 | 3.39e+38 | 426 ±10% | 39404 |
| wh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists; the rest reach 1.42 ULP | 1.42 | 0.68 | 0.975 | 8.47e+37 | 5261 | 3189 |
| wh | bf16 | `log2` | `default` | faithfully rounded; 59 of 32512 points took the other neighbour | 0.766 | 0.542 | 0.998 | 3.39e+38 | 420 ±8% | 39933 |
| wh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists; the rest reach 1.51 ULP | 1.51 | 0.842 | 0.578 | — | 5273 | 3182 |
| wh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists; the rest reach 0.512 ULP | 0.512 | 0.507 | 0.984 | 8.47e+37 | 3982 | 4213 |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 0.426; up to 6.39 ULP beyond | 6.39 | 1.38 | 0.958 | 0.426 | 588 ±5% | 28554 |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 1.91 | 0.785 | 0.972 | 3.39e+38 | 5904 | 2842 |
| wh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | 0.858 | — | 814 ±10% | 20603 |
| wh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | 0.871 | — | 898 | 18676 |
| wh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | 0.871 | — | 903 | 18579 |
| wh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41.4 ULP; mean 3.6 | 41.4 | 3.6 | 0.978 | — | 5812 | 2887 |
| wh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | 0.858 | — | 782 | 21462 |
| wh | bf16 | `logaddexp_bw` | `default` | worst pairing 62.5 ULP; mean 4.9 | 62.5 | 4.9 | 0.979 | — | 4772 ±5% | 3516 |
| wh | bf16 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 546 | 30727 |
| wh | bf16 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 562 ±11% | 29835 |
| wh | bf16 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 | 41578 |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 404 | 41488 |
| wh | bf16 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 556 ±58% | 30161 |
| wh | bf16 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 563 ±5% | 29785 |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 566 | 29633 |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.67 | 0.985 | 0.395 | 888 | 18895 |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.649 | 0.961 | 0.996 | 6992 | 2399 |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 1.65 | 0.649 | 0.961 | 0.996 | 8234 | 2038 |
| wh | bf16 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 533 | 31472 |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 547 | 30664 |
| wh | bf16 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 400 ±8% | 41922 |
| wh | bf16 | `mac` | `default` | worst pairing 63.5 ULP; mean 26.4 | 63.5 | 26.4 | 1 | — | 683 | 24551 |
| wh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | 1 | — | 5463 | 3071 |
| wh | bf16 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 548 ±10% | 30642 |
| wh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | 1 | — | 5458 | 3074 |
| wh | bf16 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 530 | 31632 |
| wh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists | 1.49 | 0.667 | 0.988 | 1.95e-38 | 599 ±7% | 27995 |
| wh | bf16 | `mse_loss` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 528 | 31754 |
| wh | bf16 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 1422 | 11794 |
| wh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists | 724 | 1.67 | 0.717 | 5.55e-17 | 12628 | 1328 |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 293 | 0.249 | 5.55e-17 | 9635 | 1741 |
| wh | bf16 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 537 | 31259 |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 550 ±5% | 30486 |
| wh | bf16 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 536 ±6% | 31324 |
| wh | bf16 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 551 | 30434 |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 406 ±14% | 41296 |
| wh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | 0.56 | — | 3123 | 5371 |
| wh | bf16 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 ±8% | 41700 |
| wh | bf16 | `polygamma` | `k=1` | 7 of 49922 points returned inf or zero where a value exists | 1.02e+08 | 1.43e+05 | 0.956 | 0.996 | 5547 | 3024 |
| wh | bf16 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 0.945 | 4.47 | 5629 | 2980 |
| wh | bf16 | `polygamma` | `k=4` | 142 of 49922 points returned inf or zero where a value exists | 3.22e+09 | 1.09e+07 | 0.936 | 3.48 | 5729 | 2928 |
| wh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists; the rest reach 3.57e+08 ULP | 3.57e+08 | 6.81e+05 | 0.945 | 4.47 | 12059 | 1391 |
| wh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | 0.982 | — | 1001 ±10% | 16763 |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2550 | 6580 |
| wh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists | 0 | 0 | 1 | 4.68e-38 | 404 ±7% | 41527 |
| wh | bf16 | `rad2deg` | `default` | faithfully rounded; 31760 of 63518 points took the other neighbour | 0.992 | 0.75 | 0.5 | 5.9e+36 | 398 ±6% | 42152 |
| wh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 0.512 | 0.507 | 0.984 | 8.47e+37 | 420 ±9% | 39953 |
| wh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists; the rest reach 1.51 ULP | 1.51 | 0.835 | 0.617 | 7.67e-20 | 6890 | 2435 |
| wh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0.512 | 0.507 | 0.984 | 8.47e+37 | 404 ±15% | 41483 |
| wh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists; the rest reach 1.51 ULP | 1.51 | 0.835 | 0.617 | 5.42e-20 | 5097 | 3292 |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 410 | 40919 |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 401 ±5% | 41875 |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3937 | 4261 |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1305 | 12852 |
| wh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±6% | 41569 |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 409 ±9% | 40972 |
| wh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 402 ±6% | 41728 |
| wh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±6% | 41514 |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 405 ±9% | 41455 |
| wh | bf16 | `remainder` | `default` | worst pairing 6.65e+35 ULP; mean 1.06e+35 | 6.65e+35 | 1.06e+35 | 0.853 | — | 679 | 24702 |
| wh | bf16 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 404 ±7% | 41503 |
| wh | bf16 | `rpow` | `exponent=0.5` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 0.975 | 3.39e+38 | 973 ±11% | 17240 |
| wh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 966 | 17360 |
| wh | bf16 | `rpow` | `exponent=2.0` | faithfully rounded; 1239 of 49536 points took the other neighbour | 0.898 | 0.564 | 0.983 | 3.39e+38 | 956 | 17545 |
| wh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists; the rest reach 0 ULP | 0 | 0 | 1 | 1.18e-38 | 2924 | 5737 |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0.499 | 0 | 1 | 3.39e+38 | 426 ±6% | 39346 |
| wh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists; the rest reach 3.23 ULP | 3.23 | 1.19 | 0.332 | — | 6294 | 2666 |
| wh | bf16 | `rsub` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 543 ±486% | 30921 |
| wh | bf16 | `rsub_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 544 | 30825 |
| wh | bf16 | `selu` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 521 ±27% | 32184 |
| wh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists; the rest reach 2.62 ULP | 2.62 | 1.06 | 0.745 | 0.00443 | 3122 ±7% | 5373 |
| wh | bf16 | `sigmoid` | `default` | faithfully rounded; 505 of 65024 points took the other neighbour | 0.857 | 0.517 | 0.994 | 3.39e+38 | 473 ±5% | 35478 |
| wh | bf16 | `sigmoid_accurate` | `default` | faithfully rounded; 505 of 65024 points took the other neighbour | 0.857 | 0.517 | 0.994 | 3.39e+38 | 477 ±18% | 35187 |
| wh | bf16 | `sigmoid_bw` | `default` | 329 of 65024 points returned inf or zero where a value exists; the rest reach 266 ULP | 266 | 5.5 | 0.988 | 1.77 | 2252 | 7451 |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 395 ±9% | 42504 |
| wh | bf16 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 398 ±5% | 42186 |
| wh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists | 0.914 | 0.585 | 0.993 | 2.33e-38 | 479 ±15% | 35033 |
| wh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists; the rest reach 285 ULP | 285 | 1.31 | 0.979 | 0.867 | 3101 | 5409 |
| wh | bf16 | `sin` | `default` | accurate to |x| <= 2.62e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 73.5 | 0.992 | 2.62e+05 | 410 ±12% | 40896 |
| wh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists; the rest reach 1.94e+42 ULP | 1.94e+42 | 1.95e+39 | 0.84 | 1.31e+05 | 1318 | 12734 |
| wh | bf16 | `sinh` | `default` | bit-exact | 0.5 | 0 | 1 | 89 | 535 ±17% | 31379 |
| wh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists; the rest reach 0.5 ULP | 0.5 | 0 | 1 | 88.5 | 5834 | 2876 |
| wh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists | 0.775 | 0.556 | 0.532 | 5.03 | 478 ±10% | 35129 |
| wh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 1.91 | 0.746 | 0.982 | 3.39e+38 | 4357 | 3851 |
| wh | bf16 | `softshrink` | `default` | faithfully rounded; 3968 of 65024 points took the other neighbour | 1 | 0.948 | 0.941 | 3.39e+38 | 415 ±5% | 40410 |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2167 | 7743 |
| wh | bf16 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 1 | 0.641 | 0.907 | 8.51e+37 | 427 ±12% | 39320 |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00391; up to 81 ULP beyond | 81 | 3.29 | 0.859 | 0.00391 | 1319 | 12717 |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 435 ±8% | 38528 |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1.14 | 0.695 | 0.719 | 3.39e+38 | 6556 | 2559 |
| wh | bf16 | `square` | `default` | faithfully rounded; 13970 of 48640 points took the other neighbour | 0.973 | 0.688 | 0.578 | 1.84e+19 | 393 | 42742 |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 1287 | 13035 |
| wh | bf16 | `squared_difference` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 539 | 31103 |
| wh | bf16 | `squared_difference_` | `default` | worst pairing 2.03 ULP; mean 1.65 | 2.03 | 1.65 | 0.571 | — | 555 | 30212 |
| wh | bf16 | `squared_difference_bw` | `default` | faithfully rounded; 64842 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 2164 | 7753 |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2 | 254 | 2 | 0.997 | — | 531 | 31601 |
| wh | bf16 | `subtract` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 528 | 31780 |
| wh | bf16 | `subtract_` | `default` | faithfully rounded; 64970 of 65026 points took the other neighbour | 0.621 | 0.571 | 0.997 | — | 548 | 30636 |
| wh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists | 0.914 | 0.585 | 0.993 | 2.33e-38 | 495 ±15% | 33888 |
| wh | bf16 | `tan` | `default` | accurate to |x| <= 1.31e+05; up to 4.1e+03 ULP beyond | 4.1e+03 | 47.6 | 0.985 | 1.31e+05 | 572 ±7% | 29354 |
| wh | bf16 | `tan_bw` | `default` | 12178 of 65024 points returned inf or zero where a value exists; the rest reach 3.1e+40 ULP | 3.1e+40 | 1.95e+37 | 0.68 | 1.13 | 2203 | 7617 |
| wh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists | 0.811 | 0.586 | 0.997 | — | 404 ±8% | 41530 |
| wh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 1.93 | 0.996 | 17.2 | 1465 | 11451 |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.38 | 0.981 | 1.35e-08 | 443 ±6% | 37879 |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.52 | 0.937 | 7.45e-09 | 1683 | 9966 |
| wh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 397 | 42213 |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 417 ±6% | 40215 |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 1658 | 10117 |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 406 ±8% | 41288 |
| wh | bf16 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 680 | 24676 |
| wh | bf16 | `xielu` | `default` | 1 of 56847 points returned inf or zero where a value exists | 0.5 | 0.5 | 1 | 2.33e-38 | 1014 | 16546 |
| wh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | 0.11 | — | 550 ±7% | 30488 |
| wh | bf16 | `xlogy_bw` | `default` | faithfully rounded; 65026 of 65026 points took the other neighbour | 0.78 | 0.78 | 0.998 | — | 8852 | 1895 |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 761 ±157% | 22055 |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2481 | 6762 |
| wh | fp32 | `acos` | `default` | within 2 ULP everywhere | 1.55 | 0.865 | 0.923 | 1 | 1090 | 15388 |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.988 | 0.926 | 14940 | 1123 |
| wh | fp32 | `acosh` | `default` | never within 2 ULP; mean 0.764, worst 2.76 | 2.76 | 0.764 | 0.782 | — | 1462 ±9% | 11479 |
| wh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists; the rest reach 724 ULP | 724 | 1.13 | 0.825 | 0.996 | 17165 | 977 |
| wh | fp32 | `add` | `default` | bit-exact | 0.5 | 0 | 1 | — | 991 ±7% | 16934 |
| wh | fp32 | `add_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 997 | 16828 |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 994 | 16872 |
| wh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | 0.901 | — | 1344 | 12479 |
| wh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | 0.985 | — | 1339 | 12532 |
| wh | fp32 | `asin` | `default` | within 2 ULP everywhere | 1.77 | 0.795 | 0.985 | 1 | 1056 | 15882 |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.926; up to 609 ULP beyond | 609 | 1.34 | 0.988 | 0.926 | 14748 | 1138 |
| wh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 1.72 | 0.774 | 0.881 | 3.39e+38 | 1800 | 9321 |
| wh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists; the rest reach 1.98 ULP | 1.98 | 1.06 | 0.927 | 1.84e+19 | 2592 | 6472 |
| wh | fp32 | `atan` | `default` | accurate to |x| <= 0.902; up to 2.34 ULP beyond | 2.34 | 0.807 | 0.955 | 0.902 | 900 | 18646 |
| wh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | 0.882 | — | 1009 | 16633 |
| wh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 9.5e+04 | 1.64e+07 | 9.5e+04 | 0.61 | — | 10552 | 1590 |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 1.01; up to 2.8 ULP beyond | 2.8 | 1.23 | 0.874 | 1.01 | 2554 | 6568 |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 4.28e-07; up to 3.11 ULP beyond | 3.11 | 1.55 | 0.91 | 4.28e-07 | 1309 ±10% | 12819 |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.678; up to 2.05e+03 ULP beyond | 2.05e+03 | 1.55 | 0.891 | 0.678 | 16361 | 1025 |
| wh | fp32 | `bias_gelu` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | 0.746 | — | 988 | 16983 |
| wh | fp32 | `bias_gelu_` | `default` | worst pairing 7.45e+40 ULP; mean 6.51e+39 | 7.45e+40 | 6.51e+39 | 0.746 | — | 1004 | 16706 |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 2.26e-38; up to 2.55 ULP beyond | 2.55 | 1.76 | 0.521 | 2.26e-38 | 842 | 19934 |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21768 |
| wh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 0.99 | 3.39e+38 | 945 | 17755 |
| wh | fp32 | `celu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5468 | 3068 |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21771 |
| wh | fp32 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 775 | 21657 |
| wh | fp32 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21778 |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4602 | 3645 |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21803 |
| wh | fp32 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21924 |
| wh | fp32 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 763 | 21989 |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 4613 ±56% | 3637 |
| wh | fp32 | `cos` | `default` | accurate to |x| <= 92.4; up to 3.3e+12 ULP beyond | 3.3e+12 | 3.33e+09 | 0.919 | 92.4 | 856 | 19594 |
| wh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists; the rest reach 2.68e+51 ULP | 2.68e+51 | 2.8e+48 | 0.835 | 28 | 3299 | 5086 |
| wh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1.35 | 0.798 | 0.98 | 89.1 | 973 | 17250 |
| wh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists; the rest reach 2.21 ULP | 2.21 | 1.17 | 0.943 | 0.0155 | 13544 | 1239 |
| wh | fp32 | `deg2rad` | `default` | faithfully rounded; 63542 of 65024 points took the other neighbour | 0.63 | 0.595 | 0.906 | 3.4e+38 | 771 | 21762 |
| wh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | 0.00566 | — | 2628 | 6385 |
| wh | fp32 | `digamma_bw` | `default` | 255 of 64769 points returned inf or zero where a value exists; the rest reach 7.91e+33 ULP | 7.91e+33 | 3.25e+29 | 0.429 | 5.4e-20 | 17824 | 941 |
| wh | fp32 | `div` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | 0.862 | — | 1015 ±447% | 16534 |
| wh | fp32 | `div_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.84 | 0.84 | 0.902 | — | 20661 | 812 |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP everywhere | 1.79 | 1.52 | 0.714 | — | 3461 | 4847 |
| wh | fp32 | `divide` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | 0.862 | — | 999 | 16790 |
| wh | fp32 | `divide_` | `default` | within 2 ULP everywhere | 1.78 | 1.22 | 0.862 | — | 1017 | 16503 |
| wh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1.36 | 0.799 | 0.99 | 3.39e+38 | 949 | 17674 |
| wh | fp32 | `elu_bw` | `default` | faithfully rounded; 2706 of 65024 points took the other neighbour | 0.866 | 0.578 | 0.999 | 3.39e+38 | 5435 | 3087 |
| wh | fp32 | `eq` | `default` | bit-exact | 0 | 0 | 1 | — | 986 | 17019 |
| wh | fp32 | `eq_` | `default` | bit-exact | 0 | 0 | 1 | — | 1002 | 16743 |
| wh | fp32 | `eqz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21897 |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 6.48 ULP beyond | 6.48 | 1.35 | 0.671 | 0.000334 | 1198 ±68% | 14000 |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.956 | 0.348 | 4853 | 3457 |
| wh | fp32 | `erfc` | `default` | 198 of 65024 points returned a value where the reference is zero | 2.1e+33 | 1.71e+29 | 0.33 | — | 1299 ±9% | 12914 |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.348; up to 65.5 ULP beyond | 65.5 | 3.73 | 0.956 | 0.348 | 4856 | 3455 |
| wh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists | 7.96e+06 | 2.62e+05 | 0.0002 | 1.32e-38 | 1415 ±10% | 11859 |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000456; up to 4.98e+05 ULP beyond | 4.98e+05 | 1.32e+03 | 0.913 | 0.000456 | 15351 | 1093 |
| wh | fp32 | `exp` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 0.997 | 3.39e+38 | 895 | 18745 |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | 0.309 | — | 772 | 21744 |
| wh | fp32 | `exp2` | `default` | faithfully rounded; 6044 of 49536 points took the other neighbour | 0.965 | 0.626 | 0.992 | 3.39e+38 | 838 | 20023 |
| wh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists; the rest reach 1.85 ULP | 1.85 | 1.09 | 0.955 | 128 | 3314 | 5063 |
| wh | fp32 | `exp_bw` | `default` | faithfully rounded; 5447 of 49458 points took the other neighbour | 0.866 | 0.578 | 0.997 | 3.39e+38 | 2605 | 6440 |
| wh | fp32 | `expm1` | `default` | faithfully rounded; 5257 of 49458 points took the other neighbour | 0.997 | 0.565 | 0.997 | 3.39e+38 | 1042 ±6% | 16103 |
| wh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists; the rest reach 4.91e+06 ULP | 4.91e+06 | 1.14e+04 | 0.985 | 1.39 | 3495 | 4800 |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 784 | 21410 |
| wh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.77e+03 | 4.19e+06 | 1.77e+03 | 0.955 | — | 4796 | 3498 |
| wh | fp32 | `fmod` | `default` | worst pairing 1.43e+45 ULP; mean 1.86e+41 | 1.43e+45 | 1.86e+41 | 0.762 | — | 1014 ±12% | 16551 |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 775 | 21647 |
| wh | fp32 | `ge` | `default` | bit-exact | 0 | 0 | 1 | — | 981 | 17096 |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | 0 | 1 | — | 1000 | 16780 |
| wh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists | 1.51e+08 | 1.55e+05 | 0.963 | 0.253 | 1389 ±13% | 12081 |
| wh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists | 7.45e+40 | 1.18e+39 | 0.497 | 2.33e-38 | 776 | 21622 |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0296; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.962 | 0.0296 | 2096 | 8004 |
| wh | fp32 | `gez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 776 | 21616 |
| wh | fp32 | `gt` | `default` | bit-exact | 0 | 0 | 1 | — | 983 | 17068 |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1000 | 16778 |
| wh | fp32 | `gtz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21803 |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 776 ±89% | 21621 |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21732 |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3220 | 5210 |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.25; up to 4.89e+06 ULP beyond | 4.89e+06 | 3e+03 | 0.998 | 2.25 | 778 | 21577 |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 5030 | 3336 |
| wh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36; up to 7.34e+06 ULP beyond | 7.34e+06 | 1.41e+03 | 0.971 | 2.36 | 818 | 20513 |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00133; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.95 | 0.00133 | 7078 | 2370 |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 779 | 21545 |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4257 | 3941 |
| wh | fp32 | `heaviside` | `value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 790 | 21237 |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 786 | 21333 |
| wh | fp32 | `heaviside` | `value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21662 |
| wh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | 0.965 | — | 1003 | 16731 |
| wh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists; the rest reach 4.86e+06 ULP | 4.86e+06 | 4.48e+04 | 0.824 | — | 6337 | 2648 |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.43; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 0.959 | 2.43 | 897 | 18697 |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00626; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.929 | 0.00626 | 3440 | 4878 |
| wh | fp32 | `i1` | `default` | accurate to |x| <= 0.00626; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.929 | 0.00626 | 1736 | 9665 |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 ±6% | 21694 |
| wh | fp32 | `isclose` | `default` | bit-exact | 0 | 0 | 1 | — | 1002 | 16745 |
| wh | fp32 | `isfinite` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21786 |
| wh | fp32 | `isinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21664 |
| wh | fp32 | `isnan` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 777 | 21605 |
| wh | fp32 | `isneginf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 776 | 21608 |
| wh | fp32 | `isposinf` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21692 |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0.5 | 0 | 1 | — | 994 | 16880 |
| wh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | 0.953 | — | 997 | 16834 |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 1.92 | 1.52 | 0.953 | — | 1010 | 16605 |
| wh | fp32 | `ldexp_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.742 | 0.742 | 0.996 | — | 5252 | 3194 |
| wh | fp32 | `le` | `default` | bit-exact | 0 | 0 | 1 | — | 987 | 17005 |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | 0 | 1 | — | 1006 | 16675 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21754 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | faithfully rounded; 31672 of 65024 points took the other neighbour | 0.84 | 0.747 | 0.868 | 3.39e+38 | 767 | 21861 |
| wh | fp32 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 781 | 21472 |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0.24 | 0 | 1 | 3.39e+38 | 3609 | 4649 |
| wh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | 0.962 | — | 1342 | 12502 |
| wh | fp32 | `lerp_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 5370 | 3124 |
| wh | fp32 | `lez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 766 | 21910 |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.0181; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.783 | 0.0181 | 3746 | 4478 |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | 0.00566 | — | 4338 | 3867 |
| wh | fp32 | `log` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.961 | 0.562 | 0.952 | 3.39e+38 | 861 | 19487 |
| wh | fp32 | `log10` | `default` | accurate to |x| <= 0.336; up to 2.13 ULP beyond | 2.13 | 1.36 | 0.641 | 0.336 | 874 ±5% | 19194 |
| wh | fp32 | `log10_bw` | `default` | accurate to |x| <= 2.04e-38; up to 2.1 ULP beyond | 2.1 | 1.41 | 0.674 | 2.04e-38 | 10285 ±33% | 1631 |
| wh | fp32 | `log1p` | `default` | faithfully rounded; 20111 of 48640 points took the other neighbour | 0.984 | 0.558 | 0.982 | 3.39e+38 | 906 | 18520 |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP everywhere | 1.89 | 0.769 | 0.913 | 8.51e+37 | 10279 | 1632 |
| wh | fp32 | `log2` | `default` | accurate to |x| <= 0.704; up to 2.43 ULP beyond | 2.43 | 0.511 | 0.995 | 0.704 | 882 ±20% | 19023 |
| wh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists; the rest reach 1.9 ULP | 1.9 | 1.3 | 0.701 | — | 10236 | 1639 |
| wh | fp32 | `log_bw` | `default` | faithfully rounded; 64512 of 64514 points took the other neighbour | 0.892 | 0.714 | 0.902 | 8.51e+37 | 7769 | 2160 |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | 0.481 | — | 1091 | 15380 |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 0.00164; up to 2.95 ULP beyond | 2.95 | 1.28 | 0.954 | 0.00164 | 11486 | 1461 |
| wh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | 0.726 | — | 1486 | 11291 |
| wh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | 0.716 | — | 1302 | 12883 |
| wh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | 0.716 | — | 1312 ±5% | 12786 |
| wh | fp32 | `logaddexp2_bw` | `default` | worst pairing 45.7 ULP; mean 6.68 | 45.7 | 6.68 | 0.956 | — | 11266 | 1489 |
| wh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | 0.726 | — | 1485 | 11296 |
| wh | fp32 | `logaddexp_bw` | `default` | worst pairing 65.7 ULP; mean 8.66 | 65.7 | 8.66 | 0.962 | — | 9386 | 1788 |
| wh | fp32 | `logical_and` | `default` | bit-exact | 0 | 0 | 1 | — | 1005 | 16701 |
| wh | fp32 | `logical_and_` | `default` | bit-exact | 0 | 0 | 1 | — | 1025 | 16365 |
| wh | fp32 | `logical_not` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 779 | 21550 |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | 0 | 1 | 1 | 773 | 21699 |
| wh | fp32 | `logical_or` | `default` | bit-exact | 0 | 0 | 1 | — | 992 | 16919 |
| wh | fp32 | `logical_or_` | `default` | bit-exact | 0 | 0 | 1 | — | 1010 | 16606 |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | 0 | 1 | — | 1015 | 16527 |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.266; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.94 | 0.266 | 1050 ±230% | 15972 |
| wh | fp32 | `logit_bw` | `default` | accurate to |x| <= 2.98e-08; up to 2.36 ULP beyond | 2.36 | 0.839 | 0.857 | 2.98e-08 | 13815 | 1214 |
| wh | fp32 | `logiteps_bw` | `default` | accurate to |x| <= 2.98e-08; up to 2.36 ULP beyond | 2.36 | 0.839 | 0.857 | 2.98e-08 | 16444 | 1020 |
| wh | fp32 | `lt` | `default` | bit-exact | 0 | 0 | 1 | — | 990 ±6% | 16946 |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | 0 | 1 | — | 1002 | 16746 |
| wh | fp32 | `ltz` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 762 ±17% | 22004 |
| wh | fp32 | `mac` | `default` | worst pairing 4.22e+06 ULP; mean 7.31e+05 | 4.22e+06 | 7.31e+05 | 0.974 | — | 1348 ±10% | 12444 |
| wh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | 1 | — | 10450 | 1606 |
| wh | fp32 | `maximum` | `default` | bit-exact | 0 | 0 | 1 | — | 982 | 17092 |
| wh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | 1 | — | 10441 | 1607 |
| wh | fp32 | `minimum` | `default` | bit-exact | 0 | 0 | 1 | — | 996 | 16837 |
| wh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 7 | 2.49 | 0.954 | 8.61e-06 | 1178 ±6% | 14248 |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 990 | 16948 |
| wh | fp32 | `mul_bw` | `default` | bit-exact | 0 | 0 | 1 | — | 2690 | 6236 |
| wh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists | 3.51e+07 | 9.16e+03 | 0.476 | 5.59e-17 | 20796 | 807 |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 0.0104 | 5.55e-17 | 20143 | 833 |
| wh | fp32 | `multiply` | `default` | bit-exact | 0.5 | 0 | 1 | — | 978 | 17158 |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1002 | 16744 |
| wh | fp32 | `ne` | `default` | bit-exact | 0 | 0 | 1 | — | 995 ±6% | 16862 |
| wh | fp32 | `ne_` | `default` | bit-exact | 0 | 0 | 1 | — | 1009 | 16634 |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 769 | 21821 |
| wh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | 0.5 | — | 6117 | 2743 |
| wh | fp32 | `nez` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21927 |
| wh | fp32 | `polygamma` | `k=1` | accurate to |x| <= 5.4e-20; up to 7.91e+33 ULP beyond | 7.91e+33 | 4.68e+29 | 0.617 | 5.4e-20 | 5705 | 2941 |
| wh | fp32 | `polygamma` | `k=2` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 0.199 | 1.79e-13 | 5892 | 2848 |
| wh | fp32 | `polygamma` | `k=4` | 141 of 49922 points returned inf or zero where a value exists | 3.77e+25 | 6.51e+21 | 0.191 | 3.68e-08 | 6301 | 2663 |
| wh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists; the rest reach 4.45e+30 ULP | 4.45e+30 | 4.42e+26 | 0.199 | 1.79e-13 | 18064 | 929 |
| wh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.17 | 5.92e+03 | 3.17 | 0.992 | — | 2160 ±9% | 7767 |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 4956 | 3385 |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21798 |
| wh | fp32 | `rad2deg` | `default` | faithfully rounded; 63518 of 63518 points took the other neighbour | 0.696 | 0.643 | 0.858 | 5.93e+36 | 767 | 21875 |
| wh | fp32 | `rdiv` | `value=2.0` | 256 of 64770 points returned inf or zero where a value exists | 0.892 | 0.714 | 0.902 | 8.51e+37 | 807 | 20792 |
| wh | fp32 | `rdiv_bw` | `scalar=2.0` | 252 of 48490 points returned inf or zero where a value exists; the rest reach 1.84 ULP | 1.84 | 1.22 | 0.726 | 7.67e-20 | 13324 | 1259 |
| wh | fp32 | `reciprocal` | `default` | faithfully rounded; 64512 of 64514 points took the other neighbour | 0.892 | 0.714 | 0.902 | 8.51e+37 | 784 | 21399 |
| wh | fp32 | `reciprocal_bw` | `default` | 254 of 48386 points returned inf or zero where a value exists; the rest reach 1.84 ULP | 1.84 | 1.22 | 0.726 | 5.42e-20 | 10017 | 1675 |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 767 | 21861 |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21723 |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 7863 | 2134 |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 2486 | 6749 |
| wh | fp32 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 ±262% | 21666 |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 773 | 21705 |
| wh | fp32 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21685 |
| wh | fp32 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21757 |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 772 | 21720 |
| wh | fp32 | `remainder` | `default` | worst pairing 1.43e+45 ULP; mean 1.68e+41 | 1.43e+45 | 1.68e+41 | 0.761 | — | 1015 | 16523 |
| wh | fp32 | `round` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 778 | 21570 |
| wh | fp32 | `rpow` | `exponent=0.5` | faithfully rounded; 5624 of 49536 points took the other neighbour | 0.879 | 0.609 | 0.995 | 3.39e+38 | 1838 | 9130 |
| wh | fp32 | `rpow` | `exponent=1.0` | 1536 of 49536 points returned inf or zero where a value exists | 0 | 0 | 1 | 8.28e+34 | 1838 | 9129 |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists | 0.879 | 0.609 | 0.996 | 8.28e+34 | 1832 | 9159 |
| wh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists; the rest reach 0 ULP | 0 | 0 | 1 | 1.18e-38 | 5694 | 2946 |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1.12 | 0.834 | 0.864 | 3.39e+38 | 846 | 19838 |
| wh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists; the rest reach 7.19 ULP | 7.19 | 4.49 | 0.293 | — | 11874 | 1413 |
| wh | fp32 | `rsub` | `default` | bit-exact | 0.5 | 0 | 1 | — | 985 | 17032 |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 1005 ±9% | 16686 |
| wh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.3, worst 51.3 | 51.3 | 26.3 | 0 | — | 975 | 17200 |
| wh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists; the rest reach 50.7 ULP | 50.7 | 20.4 | 0.235 | — | 6200 | 2706 |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 0.000345; up to 2.64 ULP beyond | 2.64 | 1.31 | 0.96 | 0.000345 | 1077 | 15574 |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 0.000345; up to 2.64 ULP beyond | 2.64 | 1.31 | 0.96 | 0.000345 | 1072 | 15650 |
| wh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists; the rest reach 8.39e+06 ULP | 8.39e+06 | 2.71e+04 | 0.975 | 0.447 | 4495 | 3732 |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 ±5% | 21941 |
| wh | fp32 | `signbit` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 765 | 21934 |
| wh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists | 2.94 | 1.55 | 0.95 | 0.000121 | 1092 | 15367 |
| wh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists; the rest reach 5.72e+06 ULP | 5.72e+06 | 841 | 0.937 | 2.98e-07 | 6216 | 2699 |
| wh | fp32 | `sin` | `default` | accurate to |x| <= 28; up to 2.2e+12 ULP beyond | 2.2e+12 | 2.68e+09 | 0.955 | 28 | 851 | 19716 |
| wh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists; the rest reach 1.56e+53 ULP | 1.56e+53 | 2.03e+49 | 0.802 | 92.4 | 2561 | 6550 |
| wh | fp32 | `sinh` | `default` | accurate to |x| <= 0.0155; up to 2.21 ULP beyond | 2.21 | 1.17 | 0.943 | 0.0155 | 1101 | 15241 |
| wh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists; the rest reach 1.35 ULP | 1.35 | 0.798 | 0.98 | 88.5 | 11881 | 1412 |
| wh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists | 8.21e+03 | 656 | 0.49 | — | 1398 ±12% | 12004 |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 0.00163; up to 2.95 ULP beyond | 2.95 | 1.34 | 0.954 | 0.00163 | 8683 | 1932 |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0.5 | 0 | 1 | 3.39e+38 | 775 ±6% | 21642 |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 4242 | 3955 |
| wh | fp32 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 2.66 | 0.897 | 0.92 | 0.000462 | 822 | 20411 |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 1.79e-07; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 0.791 | 1.79e-07 | 2570 | 6529 |
| wh | fp32 | `sqrt` | `default` | faithfully rounded; 32512 of 32512 points took the other neighbour | 0.867 | 0.827 | 0.869 | 3.39e+38 | 834 | 20106 |
| wh | fp32 | `sqrt_bw` | `default` | never within 2 ULP; mean 1.44, worst 2.24 | 2.24 | 1.44 | 0.702 | — | 12633 | 1328 |
| wh | fp32 | `square` | `default` | bit-exact | 0.5 | 0 | 1 | 1.84e+19 | 764 | 21952 |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | 0 | 1 | 1.69e+38 | 2495 | 6724 |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 983 | 17070 |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 1.91 | 1.78 | 0.896 | — | 998 | 16811 |
| wh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0.5 | 0 | 1 | — | 4211 | 3984 |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0.5 | 0 | 1 | — | 986 | 17009 |
| wh | fp32 | `subtract` | `default` | bit-exact | 0.5 | 0 | 1 | — | 997 | 16833 |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0.5 | 0 | 1 | — | 993 | 16898 |
| wh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 2.94 | 1.55 | 0.95 | 0.000121 | 1085 | 15458 |
| wh | fp32 | `tan` | `default` | accurate to |x| <= 3.92; up to 2.2e+12 ULP beyond | 2.2e+12 | 5.61e+09 | 0.938 | 3.92 | 1166 | 14392 |
| wh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists; the rest reach 2.85e+45 ULP | 2.85e+45 | 5.11e+44 | 0.864 | 0.882 | 4363 | 3846 |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000439; up to 2.79 ULP beyond | 2.79 | 1.49 | 0.979 | 0.000439 | 930 | 18043 |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | 0.485 | — | 1769 | 9482 |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.35e+04 | 0.866 | 1.34e-08 | 1334 ±16% | 12577 |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.01e+05 | 0.901 | 7.42e-09 | 3387 | 4954 |
| wh | fp32 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 774 | 21669 |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 771 | 21762 |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 3248 | 5166 |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | 0 | 1 | 3.39e+38 | 770 | 21790 |
| wh | fp32 | `where` | `default` | bit-exact | 0 | 0 | 1 | — | 1350 | 12428 |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 1.71e-13; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 0.577 | 1.71e-13 | 1381 ±11% | 12151 |
| wh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | 1.15e-05 | — | 985 ±7% | 17033 |
| wh | fp32 | `xlogy_bw` | `default` | faithfully rounded; 65024 of 65024 points took the other neighbour | 0.766 | 0.766 | 0.951 | — | 17461 ±22% | 961 |
