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
| wh | bf16 | `5b8d933` | 0.1.dev30578+g5b8d933 | wh-glx6u-02 |
| wh | fp32 | `f6deef232f7` | 0.1.dev30578+g5b8d933 | wh-glx6u-02 |

The definitions behind every number in `report_index.json` and the report pages. An LLM
answering from the report uses these definitions and no others; the code twin of the
verdict rules is `report/charts.py::verdict`, and the two must change together.

## Fields, per arch/dtype/op/variant

| Field | Definition |
|---|---|
| `max_ulp` | worst defined ULP error over the sweep: `\|y_ref − y\| / ULP(y_ref)`, reference in fp64, ULP by tt-metal's definition at the compute dtype |
| `mean_ulp` | mean over points with a defined, non-zero ULP; `—` means no such point (all exact) |
| `usable_to` | largest \|x\| below which no point exceeds 2 ULP and no point is a defect; unary ops only — `—` on multi-operand entries, where x alone does not determine the output |
| `max_abs` | worst absolute error |
| `ulp_clipped` | count of points past the chart clamp (1000 ULP) |
| `n_inputs` | measured points (fp32 rows are each the worst point of 2¹⁶ consecutive codes) |
| `outcomes` | count per outcome label, below |
| `specials` | device vs golden at ±0, ±inf, NaN, ±min-normal — printed values, no ULP |
| `defects` | points where the device returned inf or zero and the reference is a representable value — wrong answers that carry no ULP, so no other field counts them. A `mismatch` against a NaN reference is a disagreement about the domain, not a defect, and is excluded |
| `verdict` | one of the seven phrases below, precomputed |
| `perf` | `us_median` and `melem_per_s` for one dispatch over 2²⁴ resident elements, with the `host` that took them; `spread_pct` only when the row is not to be trusted. Never scored |
| `rationale` | why this variant's parameters have these values (from `ops/overrides.py`) |
| `_runs.{arch}.{dtype}` | provenance: tt-metal commit, device, versions — every claim is per this build |

## Outcome labels

| Label | Meaning |
|---|---|
| `exact` | bit-identical to the reference |
| `inexact` | differs; ULP defined |
| `flushed` | reference is subnormal — the dtype cannot hold it, nothing to score |
| `zeroed` | hardware returned 0 for a representable reference — real, but ULP cannot express it |
| `overflow` | true result exceeds the dtype |
| `undefined` | reference is NaN — outside the mathematical domain |
| `mismatch` | one side finite, the other not — a defect, not an unscorable point |
| `special` | a special-values row; excluded from every statistic |

## Verdict rules (fixed; computed, never inferred)

| Verdict | Rule |
|---|---|
| `N of M points returned inf or zero where a value exists` | defects > 0 — takes precedence over every rule below, because those points carry no ULP and the figures underneath exclude them |
| `bit-exact` | max_ulp = 0 |
| `within 2 ULP everywhere` | max_ulp ≤ 2 |
| `accurate to \|x\| <= B; up to M ULP beyond` | unary, usable_to = B defined |
| `never within 2 ULP; mean X, worst M` | unary, no point within 2 ULP |
| `worst pairing M ULP; mean X` | multi-operand — the maximum is over sampled pairings |
| `no scorable points` | no defined ULP anywhere |

## Answering rules

- Cite the entry (arch/dtype/op/variant) and its tt-metal commit; never average across archs
  or dtypes, and never extrapolate to an unmeasured cell.
- A sampled sweep's maximum (fp32 pairs, ternary) is a lower bound — the page says so.
- Never compare a timing across hosts, and never call a timing difference a regression.
- A sampled maximum (binary fp32, every ternary) is a lower bound. `ttnn-accuracy refine`
  sweeps the cell the sample drew from: `fp32/pow` publishes 5,918 ULP and holds 8,300.
- Some wrongness carries no ULP and is not in these fields. `ttnn-accuracy check` also
  requires, of the ops it is given, that the answer not change with the tiling, that an
  in-place op write to its own operand, that aliased operands agree with distinct ones,
  and that an op commute wherever its reference does. A page cannot report these; only a
  check run can.
- Not in the index → say it is not measured, and read the reason from the manifest's
  exclusions and refusals, or `analyze-report/uncovered.md`.

## Results — 841 variants

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
| bh | bf16 | `i1` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.95 | 2.33e-38 | 438 ±9% | 38298 |
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
| bh | fp32 | `i1` | `default` | accurate to |x| <= 0.00664; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.1e+04 | 0.00664 | 569 | 29478 |
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
| wh | bf16 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | 348 | 48255 |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1129 | 14860 |
| wh | bf16 | `acos` | `default` | bit-exact | 0 | — | 1 | 747 ±11% | 22446 |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.953; up to 3 ULP beyond | 3 | 1.1 | 0.953 | 6597 | 2543 |
| wh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1 | 0.994 | 3.39e+38 | 814 | 20601 |
| wh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 3 | 1.01 | 1.05 | 7611 | 2204 |
| wh | bf16 | `add` | `default` | within 2 ULP everywhere | 1 | 1 | — | 458 | 36645 |
| wh | bf16 | `add_` | `default` | within 2 ULP everywhere | 1 | 1 | — | 455 | 36870 |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | 459 | 36560 |
| wh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.32e+03 | 2.11e+06 | 2.32e+03 | — | 617 | 27196 |
| wh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.6 | 126 | 19.6 | — | 605 | 27709 |
| wh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists | 0 | — | — | 745 ±10% | 22534 |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 3 ULP beyond | 3 | 1.1 | 0.938 | 6611 | 2538 |
| wh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1 | 0.996 | 3.39e+38 | 1245 | 13470 |
| wh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1 | 1 | 1.84e+19 | 1238 ±6% | 13548 |
| wh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | 616 ±18% | 27238 |
| wh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 2.88 | 200 | 2.88 | — | 476 | 35251 |
| wh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.92 | 248 | 4.92 | — | 4813 | 3486 |
| wh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 2 | 1.04 | 9.19e+18 | 1175 | 14281 |
| wh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists | 2 | 0.997 | — | 745 ±11% | 22510 |
| wh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 8 | 1.16 | 0.82 | 7354 | 2282 |
| wh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 458 | 36602 |
| wh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | 449 | 37394 |
| wh | bf16 | `cbrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 374 ±15% | 44912 |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | 353 | 47530 |
| wh | bf16 | `celu` | `default` | bit-exact | 0 | — | 3.39e+38 | 466 ±20% | 35982 |
| wh | bf16 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | 2386 | 7030 |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 346 | 48460 |
| wh | bf16 | `clamp` | `min=0.0,max=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `clamp` | `min=1.0,max=-1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | — | — | 2128 | 7885 |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 349 | 48041 |
| wh | bf16 | `clip` | `min=0.0,max=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `clip` | `min=1.0,max=-1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | — | — | 2128 | 7882 |
| wh | bf16 | `cos` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | 402 ±16% | 41699 |
| wh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | 1514 | 11081 |
| wh | bf16 | `cosh` | `default` | bit-exact | 0 | — | 89 | 476 ±15% | 35263 |
| wh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0 | — | 88.5 | 5947 | 2821 |
| wh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 0.992 | 6.7e-37 | 346 | 48423 |
| wh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 1045 | 16052 |
| wh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.32e+04 | 0.996 | 10601 | 1583 |
| wh | bf16 | `div` | `default` | bit-exact | 0 | — | — | 470 | 35661 |
| wh | bf16 | `div_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 9301 | 1804 |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0 | — | — | 1244 | 13486 |
| wh | bf16 | `divide` | `default` | bit-exact | 0 | — | — | 470 | 35696 |
| wh | bf16 | `divide_` | `default` | bit-exact | 0 | — | — | 470 | 35689 |
| wh | bf16 | `elu` | `default` | bit-exact | 0 | — | 3.39e+38 | 436 ±17% | 38471 |
| wh | bf16 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | 2407 | 6969 |
| wh | bf16 | `eq` | `default` | bit-exact | 0 | — | — | 454 | 36995 |
| wh | bf16 | `eq_` | `default` | bit-exact | 0 | — | — | 457 | 36732 |
| wh | bf16 | `eqz` | `default` | bit-exact | 0 | — | 3.39e+38 | 351 ±6% | 47752 |
| wh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 593 ±15% | 28286 |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | 2143 | 7829 |
| wh | bf16 | `erfc` | `default` | accurate to |x| <= 2.5; up to 3.19e+28 ULP beyond | 3.19e+28 | 3.23e+24 | 2.5 | 733 | 22883 |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | 2137 | 7850 |
| wh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists | 121 | 6.53 | 1.31e-38 | 811 | 20696 |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 9 ULP beyond | 9 | 1.39 | 0.777 | 6937 | 2418 |
| wh | bf16 | `exp` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | 359 ±8% | 46786 |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6 | 6 | 3.05 | — | 353 ±6% | 47531 |
| wh | bf16 | `exp2` | `default` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | 379 ±7% | 44297 |
| wh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.05 | 128 | 1498 | 11198 |
| wh | bf16 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | 1143 | 14672 |
| wh | bf16 | `expm1` | `default` | bit-exact | 0 | — | 3.39e+38 | 490 | 34252 |
| wh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists | 125 | 5.83 | 2.08 | 1569 ±7% | 10692 |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | 352 | 47628 |
| wh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | — | 2182 | 7688 |
| wh | bf16 | `fmod` | `default` | worst pairing 6.65e+35 ULP; mean 7.8e+34 | 6.65e+35 | 7.8e+34 | — | 596 | 28157 |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | 350 ±5% | 47870 |
| wh | bf16 | `ge` | `default` | bit-exact | 0 | — | — | 459 | 36579 |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | — | — | 457 | 36696 |
| wh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists | 165 | 9.42 | 2.33e-38 | 719 | 23335 |
| wh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists | 1.14e+36 | 1.82e+34 | 2.33e-38 | 350 ±6% | 47973 |
| wh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.38; up to 3 ULP beyond | 3 | 1.52 | 8.38 | 1563 | 10737 |
| wh | bf16 | `gez` | `default` | bit-exact | 0 | — | 3.39e+38 | 352 | 47642 |
| wh | bf16 | `gt` | `default` | bit-exact | 0 | — | — | 457 | 36703 |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | — | — | 457 | 36688 |
| wh | bf16 | `gtz` | `default` | bit-exact | 0 | — | 3.39e+38 | 347 ±8% | 48394 |
| wh | bf16 | `hardmish` | `default` | within 2 ULP everywhere | 1 | 0.993 | 3.39e+38 | 351 | 47756 |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | 347 ±6% | 48410 |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1465 | 11453 |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.661 | 3.39e+38 | 349 | 48032 |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2213 | 7580 |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.13; up to 85 ULP beyond | 85 | 2.36 | 1.13 | 3101 | 5410 |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | 348 | 48273 |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1875 | 8947 |
| wh | bf16 | `heaviside` | `value=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | 347 | 48292 |
| wh | bf16 | `heaviside` | `value=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 53 | 1.46 | — | 481 | 34863 |
| wh | bf16 | `hypot_bw` | `default` | worst pairing 75 ULP; mean 2.44 | 75 | 2.44 | — | 2924 | 5738 |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.9; up to 255 ULP beyond | 255 | 56.4 | 13.9 | 474 ±13% | 35403 |
| wh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.95 | 2.33e-38 | 1834 | 9146 |
| wh | bf16 | `i1` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.95 | 2.33e-38 | 1058 | 15862 |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | 352 ±5% | 47657 |
| wh | bf16 | `isclose` | `default` | bit-exact | 0 | — | — | 470 | 35720 |
| wh | bf16 | `l1_loss` | `default` | within 2 ULP everywhere | 1 | 1 | — | 456 | 36762 |
| wh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | 465 | 36098 |
| wh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | 463 | 36208 |
| wh | bf16 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 2401 | 6986 |
| wh | bf16 | `le` | `default` | bit-exact | 0 | — | — | 459 | 36538 |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | — | — | 459 | 36564 |
| wh | bf16 | `leaky_relu` | `negative_slope=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 351 | 47753 |
| wh | bf16 | `leaky_relu` | `negative_slope=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1609 | 10427 |
| wh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 40 | 1.77e+03 | 40 | — | 605 | 27740 |
| wh | bf16 | `lerp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 2424 | 6920 |
| wh | bf16 | `lez` | `default` | bit-exact | 0 | — | 3.39e+38 | 344 | 48760 |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.439; up to 324 ULP beyond | 324 | 1.17 | 0.439 | 2157 | 7778 |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | 1823 | 9201 |
| wh | bf16 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 400 ±18% | 41910 |
| wh | bf16 | `log10` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 407 ±17% | 41174 |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.69e+37 | 4617 | 3634 |
| wh | bf16 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 389 ±13% | 43155 |
| wh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 4590 | 3655 |
| wh | bf16 | `log2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 405 ±14% | 41386 |
| wh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists | 2 | 1.02 | — | 4622 | 3630 |
| wh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 3501 | 4793 |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 4.12; up to 6 ULP beyond | 6 | 1.51 | 4.12 | 637 ±11% | 26346 |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 2 | 0.929 | 3.39e+38 | 5124 | 3274 |
| wh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 715 | 23456 |
| wh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 811 | 20676 |
| wh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | 811 | 20699 |
| wh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41 ULP; mean 3.82 | 41 | 3.82 | — | 5071 | 3308 |
| wh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | 714 | 23498 |
| wh | bf16 | `logaddexp_bw` | `default` | worst pairing 62 ULP; mean 4.88 | 62 | 4.88 | — | 4111 | 4081 |
| wh | bf16 | `logical_and` | `default` | bit-exact | 0 | — | — | 469 | 35758 |
| wh | bf16 | `logical_and_` | `default` | bit-exact | 0 | — | — | 468 | 35838 |
| wh | bf16 | `logical_not` | `default` | bit-exact | 0 | — | 3.39e+38 | 354 | 47406 |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | 349 | 48096 |
| wh | bf16 | `logical_or` | `default` | bit-exact | 0 | — | — | 469 | 35801 |
| wh | bf16 | `logical_or_` | `default` | bit-exact | 0 | — | — | 463 | 36205 |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | — | — | 467 | 35942 |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.95 | 0.395 | 785 | 21368 |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | 6092 | 2754 |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | 7232 | 2320 |
| wh | bf16 | `lt` | `default` | bit-exact | 0 | — | — | 456 | 36797 |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | — | — | 457 | 36696 |
| wh | bf16 | `ltz` | `default` | bit-exact | 0 | — | 3.39e+38 | 353 | 47552 |
| wh | bf16 | `mac` | `default` | worst pairing 64 ULP; mean 26.4 | 64 | 26.4 | — | 605 | 27708 |
| wh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 4752 | 3531 |
| wh | bf16 | `maximum` | `default` | bit-exact | 0 | — | — | 459 | 36528 |
| wh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | 4754 | 3529 |
| wh | bf16 | `minimum` | `default` | bit-exact | 0 | — | — | 459 | 36571 |
| wh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists | 1 | 0.997 | 1.95e-38 | 631 ±18% | 26602 |
| wh | bf16 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | 458 | 36669 |
| wh | bf16 | `mul_bw` | `default` | bit-exact | 0 | — | — | 1232 | 13621 |
| wh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists | 724 | 1.81 | 5.59e-17 | 11149 | 1505 |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.59e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 294 | 5.59e-17 | 8505 | 1973 |
| wh | bf16 | `multiply` | `default` | bit-exact | 0 | — | — | 463 | 36260 |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0 | — | — | 459 | 36584 |
| wh | bf16 | `ne` | `default` | bit-exact | 0 | — | — | 460 | 36489 |
| wh | bf16 | `ne_` | `default` | bit-exact | 0 | — | — | 457 | 36729 |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | 352 | 47609 |
| wh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | — | 2709 | 6192 |
| wh | bf16 | `nez` | `default` | bit-exact | 0 | — | 3.39e+38 | 353 | 47591 |
| wh | bf16 | `polygamma` | `k=1` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.32e+04 | 0.996 | 4948 | 3391 |
| wh | bf16 | `polygamma` | `k=2` | 75 of 64769 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 4.47 | — | — |
| wh | bf16 | `polygamma` | `k=4` | 142 of 64769 points returned inf or zero where a value exists | 3.22e+09 | 1.09e+07 | 4.47 | — | — |
| wh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 4.47 | 10579 | 1586 |
| wh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | — | 887 | 18918 |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | 2226 | 7536 |
| wh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists | 0 | — | 4.68e-38 | 352 | 47658 |
| wh | bf16 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | 352 | 47617 |
| wh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 363 ±9% | 46212 |
| wh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists | 2 | 1.02 | 7.67e-20 | 6019 | 2788 |
| wh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | 356 ±7% | 47178 |
| wh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 2 | 1.02 | 5.42e-20 | 4510 | 3720 |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | 352 ±6% | 47671 |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | 349 | 48061 |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 3462 | 4846 |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1130 | 14846 |
| wh | bf16 | `relu_max` | `upper_limit=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 351 | 47844 |
| wh | bf16 | `relu_max` | `upper_limit=6.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `relu_min` | `lower_limit=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 347 | 48295 |
| wh | bf16 | `remainder` | `default` | worst pairing 6.65e+35 ULP; mean 1.06e+35 | 6.65e+35 | 1.06e+35 | — | 612 | 27398 |
| wh | bf16 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | 347 ±6% | 48330 |
| wh | bf16 | `rpow` | `exponent=0.5` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | — | — |
| wh | bf16 | `rpow` | `exponent=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `rpow` | `exponent=2.0` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | 849 | 19763 |
| wh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | 2573 | 6521 |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | 400 ±13% | 41910 |
| wh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists | 3 | 1.28 | — | 5505 | 3048 |
| wh | bf16 | `rsub` | `default` | within 2 ULP everywhere | 1 | 1 | — | 459 | 36532 |
| wh | bf16 | `rsub_` | `default` | within 2 ULP everywhere | 1 | 1 | — | 455 | 36878 |
| wh | bf16 | `selu` | `default` | bit-exact | 0 | — | 3.39e+38 | 494 ±18% | 33980 |
| wh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 3 | 1.04 | 0.582 | 2723 | 6161 |
| wh | bf16 | `sigmoid` | `default` | within 2 ULP everywhere | 1 | 0.876 | 3.39e+38 | 450 ±10% | 37256 |
| wh | bf16 | `sigmoid_accurate` | `default` | within 2 ULP everywhere | 1 | 0.876 | 3.39e+38 | 437 ±24% | 38415 |
| wh | bf16 | `sigmoid_bw` | `default` | 329 of 65024 points returned inf or zero where a value exists | 266 | 5.66 | 1.86 | 1988 | 8441 |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | 352 | 47647 |
| wh | bf16 | `signbit` | `default` | bit-exact | 0 | — | 3.39e+38 | 353 | 47539 |
| wh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | 455 ±13% | 36861 |
| wh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 285 | 1.54 | 0.887 | 2756 | 6087 |
| wh | bf16 | `sin` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | 385 ±14% | 43543 |
| wh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | 1174 | 14291 |
| wh | bf16 | `sinh` | `default` | bit-exact | 0 | — | 89 | 506 ±11% | 33160 |
| wh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0 | — | 88.5 | 5253 | 3194 |
| wh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists | 1 | 1 | 5.03 | 463 ±15% | 36209 |
| wh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 2 | 0.897 | 3.39e+38 | 3810 | 4404 |
| wh | bf16 | `softshrink` | `default` | within 2 ULP everywhere | 1 | 0.996 | 3.39e+38 | 345 ±6% | 48640 |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 1877 | 8940 |
| wh | bf16 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 1 | 0.69 | 8.51e+37 | 386 ±8% | 43443 |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00491; up to 81 ULP beyond | 81 | 3.38 | 0.00491 | 1177 ±5% | 14257 |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | 388 ±17% | 43200 |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 5707 | 2940 |
| wh | bf16 | `square` | `default` | within 2 ULP everywhere | 1 | 0.991 | 1.84e+19 | 347 ±6% | 48297 |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | 1116 | 15040 |
| wh | bf16 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | 457 | 36725 |
| wh | bf16 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | 454 | 36928 |
| wh | bf16 | `squared_difference_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 1886 | 8896 |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | 459 | 36571 |
| wh | bf16 | `subtract` | `default` | within 2 ULP everywhere | 1 | 1 | — | 456 | 36758 |
| wh | bf16 | `subtract_` | `default` | within 2 ULP everywhere | 1 | 1 | — | 457 | 36750 |
| wh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | 473 ±15% | 35503 |
| wh | bf16 | `tan` | `default` | 21634 of 65024 points returned inf or zero where a value exists | 3.3e+38 | 1.16e+36 | 1.31e+05 | 583 ±10% | 28786 |
| wh | bf16 | `tan_bw` | `default` | 12178 of 65024 points returned inf or zero where a value exists | 2.53e+38 | 3.12e+35 | 1.31e+05 | 2030 | 8265 |
| wh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | 353 | 47480 |
| wh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 2.23 | 17.2 | 1303 | 12871 |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.5 | 1.35e-08 | 486 ±10% | 34554 |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.71 | 7.45e-09 | 1467 | 11438 |
| wh | bf16 | `threshold` | `threshold=0.0,value=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | 347 ±5% | 48291 |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 1442 | 11634 |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | 349 ±5% | 48089 |
| wh | bf16 | `where` | `default` | bit-exact | 0 | — | — | 607 | 27620 |
| wh | bf16 | `xielu` | `default` | 2 of 56847 points returned inf or zero where a value exists | 1.87e+38 | 3.67e+36 | 2.33e-38 | 894 | 18771 |
| wh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | — | 468 | 35884 |
| wh | bf16 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 7800 | 2151 |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | 687 | 24411 |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2214 | 7579 |
| wh | fp32 | `acos` | `default` | within 2 ULP everywhere | 2 | 1 | 1 | 943 | 17789 |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | 13553 | 1238 |
| wh | fp32 | `acosh` | `default` | never within 2 ULP; mean 1, worst 3 | 3 | 1 | — | 1173 ±16% | 14305 |
| wh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 724 | 1.12 | 0.996 | 15444 | 1086 |
| wh | fp32 | `add` | `default` | bit-exact | 0 | — | — | 881 | 19048 |
| wh | fp32 | `add_` | `default` | bit-exact | 0 | — | — | 877 | 19123 |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0 | — | — | 880 | 19072 |
| wh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | — | 1262 | 13292 |
| wh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | — | 1254 | 13381 |
| wh | fp32 | `asin` | `default` | within 2 ULP everywhere | 2 | 1.08 | 1 | 930 | 18040 |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | 13303 | 1261 |
| wh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | 1588 | 10567 |
| wh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 2 | 1.03 | 1.84e+19 | 2314 | 7250 |
| wh | fp32 | `atan` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.39e+38 | 804 | 20856 |
| wh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | — | 912 | 18390 |
| wh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 9.5e+04 | 1.64e+07 | 9.5e+04 | — | 9474 | 1771 |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 4.08e+03; up to 3 ULP beyond | 3 | 1.19 | 4.08e+03 | 2282 | 7352 |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000852; up to 3 ULP beyond | 3 | 1.56 | 0.000852 | 980 ±19% | 17127 |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.852; up to 2.05e+03 ULP beyond | 2.05e+03 | 1.51 | 0.852 | 14708 | 1141 |
| wh | fp32 | `bias_gelu` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | 878 | 19104 |
| wh | fp32 | `bias_gelu_` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | 877 | 19120 |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 5.77e-38; up to 3 ULP beyond | 3 | 1.82 | 5.77e-38 | 756 | 22184 |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | 691 | 24278 |
| wh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 835 | 20096 |
| wh | fp32 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 4950 | 3389 |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 682 | 24587 |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | — | — | 4141 | 4052 |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | 688 | 24396 |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | — | — | 4135 | 4057 |
| wh | fp32 | `cos` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | 770 | 21787 |
| wh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | 2954 | 5679 |
| wh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1 | 1 | 89 | 873 | 19227 |
| wh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 2 | 1.14 | 88.5 | 12217 | 1373 |
| wh | fp32 | `deg2rad` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 682 | 24593 |
| wh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 2332 | 7193 |
| wh | fp32 | `digamma_bw` | `default` | 255 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 8.58e-05 | 15980 | 1050 |
| wh | fp32 | `div` | `default` | within 2 ULP everywhere | 2 | 1.2 | — | 898 | 18683 |
| wh | fp32 | `div_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 18629 | 901 |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP everywhere | 2 | 1.53 | — | 3097 | 5417 |
| wh | fp32 | `divide` | `default` | within 2 ULP everywhere | 2 | 1.2 | — | 893 | 18785 |
| wh | fp32 | `divide_` | `default` | within 2 ULP everywhere | 2 | 1.2 | — | 898 | 18690 |
| wh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 836 | 20073 |
| wh | fp32 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 4918 | 3411 |
| wh | fp32 | `eq` | `default` | bit-exact | 0 | — | — | 875 | 19178 |
| wh | fp32 | `eq_` | `default` | bit-exact | 0 | — | — | 874 | 19187 |
| wh | fp32 | `eqz` | `default` | bit-exact | 0 | — | 3.39e+38 | 686 | 24463 |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 6 ULP beyond | 6 | 1.22 | 0.000334 | 982 ±20% | 17083 |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | 4353 | 3854 |
| wh | fp32 | `erfc` | `default` | never within 2 ULP; mean 1.71e+29, worst 2.1e+33 | 2.1e+33 | 1.71e+29 | — | 974 ±17% | 17216 |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | 4354 | 3854 |
| wh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists | 7.96e+06 | 2.62e+05 | 1.32e-38 | 954 ±14% | 17580 |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000496; up to 4.98e+05 ULP beyond | 4.98e+05 | 1.32e+03 | 0.000496 | 13700 | 1225 |
| wh | fp32 | `exp` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 791 | 21199 |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | — | 687 | 24406 |
| wh | fp32 | `exp2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 750 | 22366 |
| wh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.17 | 128 | 2964 | 5660 |
| wh | fp32 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 2324 | 7220 |
| wh | fp32 | `expm1` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 920 | 18246 |
| wh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists | 4.91e+06 | 1.14e+04 | 2.08 | 3134 | 5353 |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | 693 | 24204 |
| wh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.77e+03 | 4.19e+06 | 1.77e+03 | — | 4356 | 3851 |
| wh | fp32 | `fmod` | `default` | worst pairing 1.7e+38 ULP; mean 2.72e+36 | 1.7e+38 | 2.72e+36 | — | 914 | 18365 |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | 684 | 24512 |
| wh | fp32 | `ge` | `default` | bit-exact | 0 | — | — | 881 | 19042 |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | — | — | 878 | 19098 |
| wh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists | 1.51e+08 | 1.55e+05 | 0.309 | 997 ±19% | 16820 |
| wh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists | 2.91e+38 | 4.9e+36 | 6e-36 | 681 | 24638 |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0393; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.0393 | 1873 | 8958 |
| wh | fp32 | `gez` | `default` | bit-exact | 0 | — | 3.39e+38 | 685 | 24497 |
| wh | fp32 | `gt` | `default` | bit-exact | 0 | — | — | 879 | 19086 |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | — | — | 876 | 19148 |
| wh | fp32 | `gtz` | `default` | bit-exact | 0 | — | 3.39e+38 | 686 | 24448 |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | — | 3.39e+38 | 687 | 24436 |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | 682 | 24587 |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2885 | 5814 |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.61; up to 4.89e+06 ULP beyond | 4.89e+06 | 1.25e+03 | 2.61 | 685 | 24486 |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 4535 | 3699 |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00179; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.00179 | 6431 | 2609 |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | 687 | 24414 |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 3827 | 4384 |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | 680 | 24678 |
| wh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | — | 911 | 18418 |
| wh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 4.86e+06 | 4.48e+04 | — | 5697 | 2945 |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.56; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 2.56 | 789 | 21266 |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00774; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.00774 | 3068 | 5468 |
| wh | fp32 | `i1` | `default` | accurate to |x| <= 0.00774; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.00774 | 1542 | 10879 |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | 685 | 24477 |
| wh | fp32 | `isclose` | `default` | bit-exact | 0 | — | — | 891 | 18833 |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0 | — | — | 881 | 19054 |
| wh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | 893 | 18778 |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | 890 | 18844 |
| wh | fp32 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 4697 | 3572 |
| wh | fp32 | `le` | `default` | bit-exact | 0 | — | — | 881 | 19046 |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | — | — | 878 | 19113 |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 685 | 24507 |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 3287 | 5104 |
| wh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | — | 1249 | 13429 |
| wh | fp32 | `lerp_bw` | `default` | bit-exact | 0 | — | — | 4805 | 3492 |
| wh | fp32 | `lez` | `default` | bit-exact | 0 | — | 3.39e+38 | 684 | 24544 |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.108; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.108 | 3357 | 4997 |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | 3861 | 4345 |
| wh | fp32 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 775 | 21655 |
| wh | fp32 | `log10` | `default` | within 2 ULP everywhere | 2 | 1.22 | 3.39e+38 | 781 | 21480 |
| wh | fp32 | `log10_bw` | `default` | within 2 ULP everywhere | 2 | 1.41 | 3.69e+37 | 9247 | 1814 |
| wh | fp32 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 798 | 21018 |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP everywhere | 2 | 1.03 | 8.51e+37 | 9280 | 1808 |
| wh | fp32 | `log2` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | 781 | 21477 |
| wh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists | 2 | 1.33 | — | 9246 | 1814 |
| wh | fp32 | `log_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 8.51e+37 | 7029 | 2387 |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | — | 954 | 17581 |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 1.09; up to 3 ULP beyond | 3 | 1.25 | 1.09 | 10294 | 1630 |
| wh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 1353 | 12398 |
| wh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 1171 | 14323 |
| wh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | 1171 | 14322 |
| wh | fp32 | `logaddexp2_bw` | `default` | worst pairing 46 ULP; mean 6.56 | 46 | 6.56 | — | 10094 | 1662 |
| wh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | 1352 | 12407 |
| wh | fp32 | `logaddexp_bw` | `default` | worst pairing 66 ULP; mean 8.57 | 66 | 8.57 | — | 8429 | 1990 |
| wh | fp32 | `logical_and` | `default` | bit-exact | 0 | — | — | 900 | 18636 |
| wh | fp32 | `logical_and_` | `default` | bit-exact | 0 | — | — | 900 | 18648 |
| wh | fp32 | `logical_not` | `default` | bit-exact | 0 | — | 3.39e+38 | 686 | 24448 |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | 686 | 24473 |
| wh | fp32 | `logical_or` | `default` | bit-exact | 0 | — | — | 899 | 18652 |
| wh | fp32 | `logical_or_` | `default` | bit-exact | 0 | — | — | 899 | 18655 |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | — | — | 899 | 18672 |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.332; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.332 | 922 | 18204 |
| wh | fp32 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1.06 | 0.998 | 12525 | 1340 |
| wh | fp32 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1.06 | 0.998 | 14888 | 1127 |
| wh | fp32 | `lt` | `default` | bit-exact | 0 | — | — | 879 | 19090 |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | — | — | 877 | 19125 |
| wh | fp32 | `ltz` | `default` | bit-exact | 0 | — | 3.39e+38 | 684 | 24522 |
| wh | fp32 | `mac` | `default` | worst pairing 4.22e+06 ULP; mean 7.31e+05 | 4.22e+06 | 7.31e+05 | — | 1253 | 13385 |
| wh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 9389 | 1787 |
| wh | fp32 | `maximum` | `default` | bit-exact | 0 | — | — | 879 | 19082 |
| wh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | 9392 | 1786 |
| wh | fp32 | `minimum` | `default` | bit-exact | 0 | — | — | 879 | 19085 |
| wh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 7 | 2.49 | 8.61e-06 | 1002 ±12% | 16745 |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | 882 | 19028 |
| wh | fp32 | `mul_bw` | `default` | bit-exact | 0 | — | — | 2400 | 6991 |
| wh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists | 3.51e+07 | 9.16e+03 | 5.55e-17 | 18705 | 897 |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 5.55e-17 | 18018 | 931 |
| wh | fp32 | `multiply` | `default` | bit-exact | 0 | — | — | 881 | 19048 |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0 | — | — | 879 | 19090 |
| wh | fp32 | `ne` | `default` | bit-exact | 0 | — | — | 880 | 19069 |
| wh | fp32 | `ne_` | `default` | bit-exact | 0 | — | — | 878 | 19118 |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | 687 | 24407 |
| wh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | — | 5583 | 3005 |
| wh | fp32 | `nez` | `default` | bit-exact | 0 | — | 3.39e+38 | 686 | 24459 |
| wh | fp32 | `polygamma` | `k=1` | 255 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 8.58e-05 | 5088 | 3298 |
| wh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | 16207 | 1035 |
| wh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.21 | 5.92e+03 | 3.21 | — | 1933 | 8679 |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | 4426 | 3791 |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | — | 3.39e+38 | 687 | 24438 |
| wh | fp32 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | 687 | 24431 |
| wh | fp32 | `rdiv` | `value=2.0` | 256 of 64770 points returned inf or zero where a value exists | 1 | 1 | 8.51e+37 | 714 | 23502 |
| wh | fp32 | `rdiv_bw` | `scalar=2.0` | 252 of 48490 points returned inf or zero where a value exists | 2 | 1.14 | 7.67e-20 | 11939 | 1405 |
| wh | fp32 | `reciprocal` | `default` | within 2 ULP everywhere | 1 | 1 | 8.51e+37 | 707 | 23732 |
| wh | fp32 | `reciprocal_bw` | `default` | 254 of 48386 points returned inf or zero where a value exists | 2 | 1.14 | 5.42e-20 | 9042 | 1855 |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | 687 | 24433 |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | 686 | 24456 |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 7137 | 2351 |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 2215 | 7574 |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 684 | 24527 |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | 687 | 24405 |
| wh | fp32 | `remainder` | `default` | worst pairing 1.7e+38 ULP; mean 2.68e+36 | 1.7e+38 | 2.68e+36 | — | 918 | 18280 |
| wh | fp32 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | 685 | 24488 |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists | 1 | 1 | 8.28e+34 | 1623 | 10336 |
| wh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | 5098 | 3291 |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 753 | 22290 |
| wh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists | 7 | 4.51 | — | 10612 | 1581 |
| wh | fp32 | `rsub` | `default` | bit-exact | 0 | — | — | 880 | 19073 |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0 | — | — | 879 | 19079 |
| wh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.2, worst 51 | 51 | 26.2 | — | 859 | 19531 |
| wh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 51 | 20.6 | — | 5603 | 2995 |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 16.6; up to 3 ULP beyond | 3 | 1.41 | 16.6 | 941 | 17835 |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 16.6; up to 3 ULP beyond | 3 | 1.41 | 16.6 | 939 | 17872 |
| wh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists | 8.39e+06 | 2.71e+04 | 0.859 | 4015 | 4178 |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | 687 | 24416 |
| wh | fp32 | `signbit` | `default` | bit-exact | 0 | — | 3.39e+38 | 687 | 24438 |
| wh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists | 3 | 1.58 | 0.000928 | 947 | 17710 |
| wh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 5.72e+06 | 841 | 0.000244 | 5556 | 3020 |
| wh | fp32 | `sin` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | 754 | 22257 |
| wh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | 2297 | 7304 |
| wh | fp32 | `sinh` | `default` | within 2 ULP everywhere | 2 | 1.14 | 89 | 954 | 17580 |
| wh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1 | 1 | 88.5 | 10788 | 1555 |
| wh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists | 8.2e+03 | 656 | — | 1060 ±20% | 15826 |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 1.1; up to 3 ULP beyond | 3 | 1.37 | 1.1 | 7833 | 2142 |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | 686 | 24454 |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | 3826 | 4385 |
| wh | fp32 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 3 | 0.89 | 2.53e+07 | 727 | 23078 |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 9.88e-05; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 9.88e-05 | 2300 | 7293 |
| wh | fp32 | `sqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | 734 | 22849 |
| wh | fp32 | `sqrt_bw` | `default` | within 2 ULP everywhere | 2 | 1.41 | 3.39e+38 | 11313 | 1483 |
| wh | fp32 | `square` | `default` | bit-exact | 0 | — | 1.84e+19 | 686 | 24454 |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | 2230 | 7523 |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | 878 | 19110 |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | 878 | 19115 |
| wh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0 | — | — | 3778 | 4440 |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0 | — | — | 882 | 19028 |
| wh | fp32 | `subtract` | `default` | bit-exact | 0 | — | — | 879 | 19076 |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0 | — | — | 878 | 19098 |
| wh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 3 | 1.58 | 0.000928 | 945 | 17754 |
| wh | fp32 | `tan` | `default` | 20110 of 65024 points returned inf or zero where a value exists | 3.34e+38 | 1.43e+36 | 20.3 | 976 ±8% | 17195 |
| wh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists | 3.16e+38 | 9.23e+35 | 1.03 | 3877 | 4327 |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000485; up to 3 ULP beyond | 3 | 1.55 | 0.000485 | 835 | 20094 |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | — | 1586 | 10581 |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.23e+05 | 1.34e-08 | 996 ±20% | 16846 |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.85e+04 | 7.42e-09 | 3030 | 5538 |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | 684 | 24531 |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | 2903 | 5778 |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | 683 | 24572 |
| wh | fp32 | `where` | `default` | bit-exact | 0 | — | — | 1247 | 13455 |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 0.691; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 0.691 | 1092 ±16% | 15363 |
| wh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | — | 892 | 18808 |
| wh | fp32 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | 15735 | 1066 |
