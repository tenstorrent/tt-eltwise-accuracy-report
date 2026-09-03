# Ask an assistant about these measurements

Give your assistant this page's URL, or paste the file, then ask in plain language —
"is `exp` usable on Wormhole in bfloat16?", "which ops are worse than 2 ULP in fp32?".
Everything needed to answer is below: the definitions, then every measured result.

Answering rules, in force for whoever reads this: quote the `Verdict` column rather than
judging the numbers yourself, name the architecture and dtype in every answer, and if a
variant is not in the table say so instead of extrapolating from a neighbouring one. A timing
is only comparable to another taken on the same host, which the "Measured against" table names.


## Measured against

| Arch | Dtype | tt-metal | ttnn |
|---|---|---|---|
| bh | bf16 | `72023534b03` | 0.75.0rc10.dev916+ga5cd86212e1 |
| bh | fp32 | `72023534b03` | 0.75.0rc10.dev916+ga5cd86212e1 |
| wh | bf16 | `812427269cb` | 0.1.dev30578+g5b8d933 |
| wh | fp32 | `812427269cb` | 0.1.dev30578+g5b8d933 |

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
- Not in the index → say it is not measured, and read the reason from the manifest's
  exclusions and refusals, or `analyze-report/uncovered.md`.

## Results — 786 variants

| Arch | Dtype | Op | Parameters | Verdict | Max ULP | Mean ULP | Usable to | µs | Melem/s |
|---|---|---|---|---|---|---|---|---|---|
| bh | bf16 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `acos` | `default` | bit-exact | 0 | — | 1 | — | — |
| bh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.953; up to 3 ULP beyond | 3 | 1.1 | 0.953 | — | — |
| bh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1 | 0.994 | 3.39e+38 | — | — |
| bh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 3 | 1.01 | 1.05 | — | — |
| bh | bf16 | `add` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `add_` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | — | — |
| bh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.21e+03 | 2.11e+06 | 2.21e+03 | — | — | — |
| bh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.6 | 126 | 19.6 | — | — | — |
| bh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists | 0 | — | — | — | — |
| bh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 3 ULP beyond | 3 | 1.1 | 0.938 | — | — |
| bh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1 | 0.998 | 3.39e+38 | — | — |
| bh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1 | 1 | 1.84e+19 | — | — |
| bh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | — | — |
| bh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 3.85 | 200 | 3.85 | — | — | — |
| bh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.92 | 248 | 4.92 | — | — | — |
| bh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 2 | 1.04 | 9.19e+18 | — | — |
| bh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists | 2 | 0.997 | — | — | — |
| bh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 8 | 1.13 | 0.82 | — | — |
| bh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | — | — |
| bh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | — | — |
| bh | bf16 | `cbrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | bf16 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `celu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | — | — |
| bh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `clip_bw` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `cos` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | — | — |
| bh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | — | — |
| bh | bf16 | `cosh` | `default` | within 2 ULP everywhere | 1 | 1 | 89 | — | — |
| bh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0 | — | 88.5 | — | — |
| bh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 0.992 | 6.7e-37 | — | — |
| bh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | — | — |
| bh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.36e+04 | 0.996 | — | — |
| bh | bf16 | `div` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `div_bw` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `div_no_nan` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `divide` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `divide_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `elu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | — | — |
| bh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | — | — |
| bh | bf16 | `erfc` | `default` | accurate to |x| <= 2.5; up to 3.2e+28 ULP beyond | 3.2e+28 | 3.23e+24 | 2.5 | — | — |
| bh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | — | — |
| bh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists | 121 | 6.53 | 1.31e-38 | — | — |
| bh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 9 ULP beyond | 9 | 1.39 | 0.777 | — | — |
| bh | bf16 | `exp` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | — | — |
| bh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6 | 6 | 3.05 | — | — | — |
| bh | bf16 | `exp2` | `default` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | — | — |
| bh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.05 | 128 | — | — |
| bh | bf16 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | — | — |
| bh | bf16 | `expm1` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists | 125 | 5.83 | 2.08 | — | — |
| bh | bf16 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | — | — | — |
| bh | bf16 | `fmod` | `default` | worst pairing 9.14e+35 ULP; mean 8.94e+34 | 9.14e+35 | 8.94e+34 | — | — | — |
| bh | bf16 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `ge_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists | 165 | 9.42 | 2.33e-38 | — | — |
| bh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists | 1.14e+36 | 1.82e+34 | 2.33e-38 | — | — |
| bh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.38; up to 3 ULP beyond | 3 | 1.52 | 8.38 | — | — |
| bh | bf16 | `gt_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `hardmish` | `default` | within 2 ULP everywhere | 1 | 0.993 | 3.39e+38 | — | — |
| bh | bf16 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.661 | 3.39e+38 | — | — |
| bh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `hardswish` | `default` | 2 of 65024 points returned inf or zero where a value exists | 2 | 1.01 | 2.33e-38 | — | — |
| bh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.13; up to 85 ULP beyond | 85 | 2.36 | 1.13 | — | — |
| bh | bf16 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 53 | 1.46 | — | — | — |
| bh | bf16 | `hypot_bw` | `default` | worst pairing 75 ULP; mean 2.44 | 75 | 2.44 | — | — | — |
| bh | bf16 | `i0` | `default` | accurate to |x| <= 13.9; up to 255 ULP beyond | 255 | 56.4 | 13.9 | — | — |
| bh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.95 | 2.33e-38 | — | — |
| bh | bf16 | `i1` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.95 | 2.33e-38 | — | — |
| bh | bf16 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `l1_loss` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | — | — |
| bh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | — | — |
| bh | bf16 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `le_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 40 | 1.77e+03 | 40 | — | — | — |
| bh | bf16 | `lerp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.439; up to 324 ULP beyond | 324 | 1.17 | 0.439 | — | — |
| bh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | — | — |
| bh | bf16 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | bf16 | `log10` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.69e+37 | — | — |
| bh | bf16 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | — | — |
| bh | bf16 | `log2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists | 2 | 1.02 | — | — | — |
| bh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0 | — | 8.47e+37 | — | — |
| bh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 4.12; up to 6 ULP beyond | 6 | 1.51 | 4.12 | — | — |
| bh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 2 | 0.895 | 3.39e+38 | — | — |
| bh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | — | — |
| bh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | — | — |
| bh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | — | — |
| bh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41 ULP; mean 3.82 | 41 | 3.82 | — | — | — |
| bh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | — | — |
| bh | bf16 | `logaddexp_bw` | `default` | worst pairing 62 ULP; mean 4.88 | 62 | 4.88 | — | — | — |
| bh | bf16 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | — | — |
| bh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.95 | 0.395 | — | — |
| bh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | — | — |
| bh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | — | — |
| bh | bf16 | `lt_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `mac` | `default` | worst pairing 64 ULP; mean 26.4 | 64 | 26.4 | — | — | — |
| bh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | — | — |
| bh | bf16 | `maximum` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | — | — |
| bh | bf16 | `minimum` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists | 1 | 0.997 | 1.95e-38 | — | — |
| bh | bf16 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | — | — |
| bh | bf16 | `mul_bw` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists | 724 | 1.81 | 5.59e-17 | — | — |
| bh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.59e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 294 | 5.59e-17 | — | — |
| bh | bf16 | `multiply` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `multiply_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | — | — | — |
| bh | bf16 | `polygamma` | `k=1` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.36e+04 | 0.996 | — | — |
| bh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 1.21e+06 | 4.47 | — | — |
| bh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | — | — | — |
| bh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | — | — |
| bh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists | 0 | — | 4.68e-38 | — | — |
| bh | bf16 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | — | — |
| bh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 0 | — | 8.47e+37 | — | — |
| bh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists | 2 | 1.02 | 7.67e-20 | — | — |
| bh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 0 | — | 8.47e+37 | — | — |
| bh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 2 | 1.02 | 5.42e-20 | — | — |
| bh | bf16 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `remainder` | `default` | worst pairing 1.08e+36 ULP; mean 1.22e+35 | 1.08e+36 | 1.22e+35 | — | — | — |
| bh | bf16 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `rpow` | `exponent=2.0` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | — | — |
| bh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | — | — |
| bh | bf16 | `rsqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists | 3 | 1.28 | — | — | — |
| bh | bf16 | `rsub` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `rsub_` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `selu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 3 | 1.04 | 0.582 | — | — |
| bh | bf16 | `sigmoid` | `default` | within 2 ULP everywhere | 1 | 0.886 | 3.39e+38 | — | — |
| bh | bf16 | `sigmoid_accurate` | `default` | within 2 ULP everywhere | 1 | 0.886 | 3.39e+38 | — | — |
| bh | bf16 | `sigmoid_bw` | `default` | 331 of 65024 points returned inf or zero where a value exists | 125 | 4.97 | 1.86 | — | — |
| bh | bf16 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | — | — |
| bh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 285 | 1.54 | 0.887 | — | — |
| bh | bf16 | `sin` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | — | — |
| bh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | — | — |
| bh | bf16 | `sinh` | `default` | bit-exact | 0 | — | 89 | — | — |
| bh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1 | 1 | 88.5 | — | — |
| bh | bf16 | `softcap` | `beta=50.0` | 1426 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | — | — |
| bh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists | 1 | 1 | 5.03 | — | — |
| bh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 2 | 0.861 | 3.39e+38 | — | — |
| bh | bf16 | `softshrink` | `default` | within 2 ULP everywhere | 1 | 0.996 | 3.39e+38 | — | — |
| bh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `softsign` | `default` | 512 of 65024 points returned inf or zero where a value exists | 1 | 0.642 | 8.47e+37 | — | — |
| bh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00491; up to 81 ULP beyond | 81 | 3.34 | 0.00491 | — | — |
| bh | bf16 | `sqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | bf16 | `square` | `default` | within 2 ULP everywhere | 1 | 0.991 | 1.84e+19 | — | — |
| bh | bf16 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | — | — |
| bh | bf16 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | — | — |
| bh | bf16 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | — | — |
| bh | bf16 | `squared_difference_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | — | — |
| bh | bf16 | `subtract` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `subtract_` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | — | — |
| bh | bf16 | `tan` | `default` | 21634 of 65024 points returned inf or zero where a value exists | 3.3e+38 | 1.16e+36 | 1.31e+05 | — | — |
| bh | bf16 | `tan_bw` | `default` | 22759 of 65024 points returned inf or zero where a value exists | 2.53e+38 | 7.9e+35 | 1.31e+05 | — | — |
| bh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | — | — |
| bh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 2.23 | 17.2 | — | — |
| bh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.5 | 1.35e-08 | — | — |
| bh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.71 | 7.45e-09 | — | — |
| bh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | bf16 | `where` | `default` | bit-exact | 0 | — | — | — | — |
| bh | bf16 | `xielu` | `default` | 1 of 56847 points returned inf or zero where a value exists | 1 | 1 | 2.33e-38 | — | — |
| bh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | — | — | — |
| bh | bf16 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | fp32 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `acos` | `default` | within 2 ULP everywhere | 2 | 1 | 1 | — | — |
| bh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | — | — |
| bh | fp32 | `acosh` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 724 | 1.12 | 0.996 | — | — |
| bh | fp32 | `add` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `add_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | — | — | — |
| bh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | — | — | — |
| bh | fp32 | `asin` | `default` | within 2 ULP everywhere | 2 | 1.08 | 1 | — | — |
| bh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | — | — |
| bh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 2 | 1.03 | 1.84e+19 | — | — |
| bh | fp32 | `atan` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.39e+38 | — | — |
| bh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | — | — | — |
| bh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 1.66e+05 | 1.64e+07 | 1.66e+05 | — | — | — |
| bh | fp32 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 9.02e+04 | 3.25e+03 | 4.08e+03 | — | — |
| bh | fp32 | `atanh` | `default` | accurate to |x| <= 0.00772; up to 3 ULP beyond | 3 | 1.51 | 0.00772 | — | — |
| bh | fp32 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 9.02e+04 | 3.31e+03 | 0.855 | — | — |
| bh | fp32 | `bias_gelu` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | — | — |
| bh | fp32 | `bias_gelu_` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | — | — |
| bh | fp32 | `cbrt` | `default` | accurate to |x| <= 5.77e-38; up to 3 ULP beyond | 3 | 1.82 | 5.77e-38 | — | — |
| bh | fp32 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `clip_bw` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `cos` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | — | — |
| bh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | — | — |
| bh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1 | 1 | 89 | — | — |
| bh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 2 | 1.12 | 88.5 | — | — |
| bh | fp32 | `deg2rad` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | — | — |
| bh | fp32 | `digamma_bw` | `default` | 257 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 5.4e-20 | — | — |
| bh | fp32 | `div` | `default` | within 2 ULP everywhere | 2 | 1.58 | — | — | — |
| bh | fp32 | `div_bw` | `default` | worst pairing 7.95e+04 ULP; mean 7.95e+04 | 7.95e+04 | 7.95e+04 | — | — | — |
| bh | fp32 | `div_no_nan` | `default` | worst pairing 9.24e+04 ULP; mean 4.51e+04 | 9.24e+04 | 4.51e+04 | — | — | — |
| bh | fp32 | `divide` | `default` | within 2 ULP everywhere | 2 | 1.58 | — | — | — |
| bh | fp32 | `divide_` | `default` | within 2 ULP everywhere | 2 | 1.58 | — | — | — |
| bh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 7 ULP beyond | 7 | 1.23 | 0.000334 | — | — |
| bh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | — | — |
| bh | fp32 | `erfc` | `default` | never within 2 ULP; mean 1.71e+29, worst 2.1e+33 | 2.1e+33 | 1.71e+29 | — | — | — |
| bh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | — | — |
| bh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists | 7.96e+06 | 2.62e+05 | 1.32e-38 | — | — |
| bh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000496; up to 1.03e+06 ULP beyond | 1.03e+06 | 1.68e+03 | 0.000496 | — | — |
| bh | fp32 | `exp` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | — | — | — |
| bh | fp32 | `exp2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.17 | 128 | — | — |
| bh | fp32 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `expm1` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists | 4.91e+06 | 1.14e+04 | 2.08 | — | — |
| bh | fp32 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.56e+03 | 4.19e+06 | 1.56e+03 | — | — | — |
| bh | fp32 | `fmod` | `default` | worst pairing 3.19e+38 ULP; mean 4e+36 | 3.19e+38 | 4e+36 | — | — | — |
| bh | fp32 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `ge_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists | 1.71e+08 | 1.7e+05 | 0.301 | — | — |
| bh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists | 2.91e+38 | 4.9e+36 | 6e-36 | — | — |
| bh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0393; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.0393 | — | — |
| bh | fp32 | `gt_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `hardmish` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.61; up to 4.89e+06 ULP beyond | 4.89e+06 | 1.25e+03 | 2.61 | — | — |
| bh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36; up to 7.34e+06 ULP beyond | 7.34e+06 | 1.41e+03 | 2.36 | — | — |
| bh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00179; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.00179 | — | — |
| bh | fp32 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | — | — | — |
| bh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 4.86e+06 | 4.48e+04 | — | — | — |
| bh | fp32 | `i0` | `default` | accurate to |x| <= 2.56; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 2.56 | — | — |
| bh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00664; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.1e+04 | 0.00664 | — | — |
| bh | fp32 | `i1` | `default` | accurate to |x| <= 0.00664; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.1e+04 | 0.00664 | — | — |
| bh | fp32 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `l1_loss` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | — | — |
| bh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | — | — |
| bh | fp32 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| bh | fp32 | `le_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | — | — | — |
| bh | fp32 | `lerp_bw` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.108; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.108 | — | — |
| bh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | — | — |
| bh | fp32 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `log10` | `default` | within 2 ULP everywhere | 2 | 1.22 | 3.39e+38 | — | — |
| bh | fp32 | `log10_bw` | `default` | accurate to |x| <= 8.81e+30; up to 9.02e+04 ULP beyond | 9.02e+04 | 1.73e+03 | 8.81e+30 | — | — |
| bh | fp32 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 9.02e+04 | 2.96e+03 | 2.02e+31 | — | — |
| bh | fp32 | `log2` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists | 9.02e+04 | 1.76e+03 | — | — | — |
| bh | fp32 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 9.02e+04 | 1.72e+03 | 2.02e+31 | — | — |
| bh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | — | — | — |
| bh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 1.09; up to 3 ULP beyond | 3 | 1.25 | 1.09 | — | — |
| bh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | — | — |
| bh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | — | — |
| bh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | — | — |
| bh | fp32 | `logaddexp2_bw` | `default` | worst pairing 9.02e+04 ULP; mean 5.6e+04 | 9.02e+04 | 5.6e+04 | — | — | — |
| bh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | — | — |
| bh | fp32 | `logaddexp_bw` | `default` | worst pairing 8.98e+04 ULP; mean 4.46e+04 | 8.98e+04 | 4.46e+04 | — | — | — |
| bh | fp32 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | — | — |
| bh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `logit` | `default` | accurate to |x| <= 0.375; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.375 | — | — |
| bh | fp32 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1.05 | 0.998 | — | — |
| bh | fp32 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1.05 | 0.998 | — | — |
| bh | fp32 | `lt_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `mac` | `default` | worst pairing 4.22e+06 ULP; mean 7.31e+05 | 4.22e+06 | 7.31e+05 | — | — | — |
| bh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | — | — |
| bh | fp32 | `maximum` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | — | — |
| bh | fp32 | `minimum` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 7 | 2.58 | 1.49e-07 | — | — |
| bh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | — | — |
| bh | fp32 | `mul_bw` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists | 3.51e+07 | 9.16e+03 | 5.55e-17 | — | — |
| bh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 5.55e-17 | — | — |
| bh | fp32 | `multiply` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `multiply_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | — | — | — |
| bh | fp32 | `polygamma` | `k=1` | 257 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 5.4e-20 | — | — |
| bh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | — | — |
| bh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.21 | 5.92e+03 | 3.21 | — | — | — |
| bh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | — | — |
| bh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | — | — |
| bh | fp32 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | — | — |
| bh | fp32 | `rdiv_bw` | `scalar=2.0` | 254 of 48490 points returned inf or zero where a value exists | 9.02e+04 | 1.92e+03 | 7.67e-20 | — | — |
| bh | fp32 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 9.02e+04 | 1.72e+03 | 2.02e+31 | — | — |
| bh | fp32 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 9.02e+04 | 1.92e+03 | 5.42e-20 | — | — |
| bh | fp32 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `remainder` | `default` | worst pairing 3.19e+38 ULP; mean 4.03e+36 | 3.19e+38 | 4.03e+36 | — | — | — |
| bh | fp32 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists | 1 | 1 | 8.28e+34 | — | — |
| bh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | — | — |
| bh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists | 7 | 4.5 | — | — | — |
| bh | fp32 | `rsub` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `rsub_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.2, worst 51 | 51 | 26.2 | — | — | — |
| bh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 51 | 20.6 | — | — | — |
| bh | fp32 | `sigmoid` | `default` | accurate to |x| <= 0.000475; up to 3 ULP beyond | 3 | 1.65 | 0.000475 | — | — |
| bh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 0.000475; up to 3 ULP beyond | 3 | 1.65 | 0.000475 | — | — |
| bh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists | 8.39e+06 | 2.54e+04 | 0.617 | — | — |
| bh | fp32 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists | 4 | 1.82 | 2.63e-05 | — | — |
| bh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 5.72e+06 | 842 | 1.59e-05 | — | — |
| bh | fp32 | `sin` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | — | — |
| bh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | — | — |
| bh | fp32 | `sinh` | `default` | within 2 ULP everywhere | 2 | 1.12 | 89 | — | — |
| bh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1 | 1 | 88.5 | — | — |
| bh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists | 8.2e+03 | 656 | — | — | — |
| bh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 1.09; up to 3 ULP beyond | 3 | 1.36 | 1.09 | — | — |
| bh | fp32 | `softshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `softsign` | `default` | 512 of 65024 points returned inf or zero where a value exists | 3 | 1.28 | 0.000243 | — | — |
| bh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 6.44e-05; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 6.44e-05 | — | — |
| bh | fp32 | `sqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| bh | fp32 | `sqrt_bw` | `default` | within 2 ULP everywhere | 2 | 1.38 | 3.39e+38 | — | — |
| bh | fp32 | `square` | `default` | bit-exact | 0 | — | 1.84e+19 | — | — |
| bh | fp32 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | — | — |
| bh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | — | — |
| bh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | — | — |
| bh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `subtract` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `subtract_` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 4 | 1.82 | 2.63e-05 | — | — |
| bh | fp32 | `tan` | `default` | 20110 of 65024 points returned inf or zero where a value exists | 3.34e+38 | 1.43e+36 | 20.3 | — | — |
| bh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists | 3.16e+38 | 9.23e+35 | 1.01 | — | — |
| bh | fp32 | `tanh` | `default` | accurate to |x| <= 0.0154; up to 3 ULP beyond | 3 | 1.54 | 0.0154 | — | — |
| bh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | — | — | — |
| bh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 1.23e+05 | 1.34e-08 | — | — |
| bh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.94e+04 | 7.42e-09 | — | — |
| bh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| bh | fp32 | `where` | `default` | bit-exact | 0 | — | — | — | — |
| bh | fp32 | `xielu` | `default` | accurate to |x| <= 0.691; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 0.691 | — | — |
| bh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | — | — | — |
| bh | fp32 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `acos` | `default` | bit-exact | 0 | — | 1 | — | — |
| wh | bf16 | `acos_bw` | `default` | accurate to |x| <= 0.953; up to 3 ULP beyond | 3 | 1.1 | 0.953 | — | — |
| wh | bf16 | `acosh` | `default` | within 2 ULP everywhere | 1 | 0.994 | 3.39e+38 | — | — |
| wh | bf16 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 3 | 1.01 | 1.05 | — | — |
| wh | bf16 | `add` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `add_` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `addalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | — | — |
| wh | bf16 | `addcdiv` | `default` | worst pairing 2.11e+06 ULP; mean 2.32e+03 | 2.11e+06 | 2.32e+03 | — | — | — |
| wh | bf16 | `addcmul` | `default` | worst pairing 126 ULP; mean 19.6 | 126 | 19.6 | — | — | — |
| wh | bf16 | `asin` | `default` | 2 of 32258 points returned inf or zero where a value exists | 0 | — | — | — | — |
| wh | bf16 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 3 ULP beyond | 3 | 1.1 | 0.938 | — | — |
| wh | bf16 | `asinh` | `default` | within 2 ULP everywhere | 1 | 0.996 | 3.39e+38 | — | — |
| wh | bf16 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 1 | 1 | 1.84e+19 | — | — |
| wh | bf16 | `atan` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | — | — |
| wh | bf16 | `atan2` | `default` | worst pairing 200 ULP; mean 2.88 | 200 | 2.88 | — | — | — |
| wh | bf16 | `atan2_bw` | `default` | worst pairing 248 ULP; mean 4.92 | 248 | 4.92 | — | — | — |
| wh | bf16 | `atan_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 2 | 1.04 | 9.19e+18 | — | — |
| wh | bf16 | `atanh` | `default` | 2 of 32256 points returned inf or zero where a value exists | 2 | 0.997 | — | — | — |
| wh | bf16 | `atanh_bw` | `default` | 2 of 48386 points returned inf or zero where a value exists | 8 | 1.16 | 0.82 | — | — |
| wh | bf16 | `bias_gelu` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | — | — |
| wh | bf16 | `bias_gelu_` | `default` | worst pairing 1.14e+36 ULP; mean 4.5e+34 | 1.14e+36 | 4.5e+34 | — | — | — |
| wh | bf16 | `bias_gelu_bw` | `default` | worst pairing 2.45e+05 ULP; mean 93.8 | 2.45e+05 | 93.8 | — | — | — |
| wh | bf16 | `cbrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | bf16 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `celu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | — | — |
| wh | bf16 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `clamp_bw` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `clip_bw` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `cos` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | — | — |
| wh | bf16 | `cos_bw` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | — | — |
| wh | bf16 | `cosh` | `default` | bit-exact | 0 | — | 89 | — | — |
| wh | bf16 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0 | — | 88.5 | — | — |
| wh | bf16 | `deg2rad` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 0.992 | 6.7e-37 | — | — |
| wh | bf16 | `digamma` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | — | — |
| wh | bf16 | `digamma_bw` | `default` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.32e+04 | 0.996 | — | — |
| wh | bf16 | `div` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `div_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `div_no_nan` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `divide` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `divide_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `elu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 0.776 | 3.39e+38 | — | — |
| wh | bf16 | `erf` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | bf16 | `erf_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | — | — |
| wh | bf16 | `erfc` | `default` | accurate to |x| <= 2.5; up to 3.19e+28 ULP beyond | 3.19e+28 | 3.23e+24 | 2.5 | — | — |
| wh | bf16 | `erfc_bw` | `default` | accurate to |x| <= 1.75; up to 112 ULP beyond | 112 | 5 | 1.75 | — | — |
| wh | bf16 | `erfinv` | `default` | 29424 of 32256 points returned inf or zero where a value exists | 121 | 6.53 | 1.31e-38 | — | — |
| wh | bf16 | `erfinv_bw` | `default` | accurate to |x| <= 0.777; up to 9 ULP beyond | 9 | 1.39 | 0.777 | — | — |
| wh | bf16 | `exp` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | — | — |
| wh | bf16 | `exp` | `fast_approx` | never within 2 ULP; mean 3.05, worst 6 | 6 | 3.05 | — | — | — |
| wh | bf16 | `exp2` | `default` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | — | — |
| wh | bf16 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.05 | 128 | — | — |
| wh | bf16 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 0.856 | 3.39e+38 | — | — |
| wh | bf16 | `expm1` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `expm1_bw` | `default` | 331 of 49458 points returned inf or zero where a value exists | 125 | 5.83 | 2.08 | — | — |
| wh | bf16 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `floor_div` | `default` | worst pairing 64 ULP; mean 42 | 64 | 42 | — | — | — |
| wh | bf16 | `fmod` | `default` | worst pairing 6.65e+35 ULP; mean 7.8e+34 | 6.65e+35 | 7.8e+34 | — | — | — |
| wh | bf16 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `ge_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `gelu` | `default` | 86 of 65024 points returned inf or zero where a value exists | 165 | 9.42 | 2.33e-38 | — | — |
| wh | bf16 | `gelu` | `fast_approx` | 198 of 65024 points returned inf or zero where a value exists | 1.14e+36 | 1.82e+34 | 2.33e-38 | — | — |
| wh | bf16 | `gelu_bw` | `default` | accurate to |x| <= 8.38; up to 3 ULP beyond | 3 | 1.52 | 8.38 | — | — |
| wh | bf16 | `gt_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `hardmish` | `default` | within 2 ULP everywhere | 1 | 0.993 | 3.39e+38 | — | — |
| wh | bf16 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `hardsigmoid` | `default` | within 2 ULP everywhere | 1 | 0.661 | 3.39e+38 | — | — |
| wh | bf16 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `hardswish` | `default` | 2 of 65024 points returned inf or zero where a value exists | 2 | 1.01 | 2.33e-38 | — | — |
| wh | bf16 | `hardswish_bw` | `default` | accurate to |x| <= 1.13; up to 85 ULP beyond | 85 | 2.36 | 1.13 | — | — |
| wh | bf16 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `hypot` | `default` | 16384 of 65026 points returned inf or zero where a value exists | 53 | 1.46 | — | — | — |
| wh | bf16 | `hypot_bw` | `default` | worst pairing 75 ULP; mean 2.44 | 75 | 2.44 | — | — | — |
| wh | bf16 | `i0` | `default` | accurate to |x| <= 13.9; up to 255 ULP beyond | 255 | 56.4 | 13.9 | — | — |
| wh | bf16 | `i0_bw` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.95 | 2.33e-38 | — | — |
| wh | bf16 | `i1` | `default` | 4 of 33904 points returned inf or zero where a value exists | 218 | 8.95 | 2.33e-38 | — | — |
| wh | bf16 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `l1_loss` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `ldexp` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | — | — |
| wh | bf16 | `ldexp_` | `default` | worst pairing 255 ULP; mean 94.5 | 255 | 94.5 | — | — | — |
| wh | bf16 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `le_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | bf16 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `lerp` | `default` | worst pairing 1.77e+03 ULP; mean 40 | 1.77e+03 | 40 | — | — | — |
| wh | bf16 | `lerp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `lgamma` | `default` | accurate to |x| <= 0.439; up to 324 ULP beyond | 324 | 1.17 | 0.439 | — | — |
| wh | bf16 | `lgamma_bw` | `default` | never within 2 ULP; mean 177, worst 5.57e+05 | 5.57e+05 | 177 | — | — | — |
| wh | bf16 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | bf16 | `log10` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | bf16 | `log10_bw` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.69e+37 | — | — |
| wh | bf16 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | bf16 | `log1p_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | — | — |
| wh | bf16 | `log2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | bf16 | `log2_bw` | `default` | 116 of 64626 points returned inf or zero where a value exists | 2 | 1.02 | — | — | — |
| wh | bf16 | `log_bw` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | — | — |
| wh | bf16 | `log_sigmoid` | `default` | accurate to |x| <= 4.12; up to 6 ULP beyond | 6 | 1.51 | 4.12 | — | — |
| wh | bf16 | `log_sigmoid_bw` | `default` | within 2 ULP everywhere | 2 | 0.929 | 3.39e+38 | — | — |
| wh | bf16 | `logaddexp` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | — | — |
| wh | bf16 | `logaddexp2` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | — | — |
| wh | bf16 | `logaddexp2_` | `default` | 15488 of 65026 points returned inf or zero where a value exists | 3.03e+06 | 4.08e+03 | — | — | — |
| wh | bf16 | `logaddexp2_bw` | `default` | worst pairing 41 ULP; mean 3.82 | 41 | 3.82 | — | — | — |
| wh | bf16 | `logaddexp_` | `default` | 15566 of 65026 points returned inf or zero where a value exists | 8.39e+06 | 4.85e+03 | — | — | — |
| wh | bf16 | `logaddexp_bw` | `default` | worst pairing 62 ULP; mean 4.88 | 62 | 4.88 | — | — | — |
| wh | bf16 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | — | — |
| wh | bf16 | `logical_xor_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `logit` | `default` | accurate to |x| <= 0.395; up to 64 ULP beyond | 64 | 1.95 | 0.395 | — | — |
| wh | bf16 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | — | — |
| wh | bf16 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1 | 0.996 | — | — |
| wh | bf16 | `lt_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `mac` | `default` | worst pairing 64 ULP; mean 26.4 | 64 | 26.4 | — | — | — |
| wh | bf16 | `max_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | — | — |
| wh | bf16 | `maximum` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `min_bw` | `default` | worst pairing 64 ULP; mean 64 | 64 | 64 | — | — | — |
| wh | bf16 | `minimum` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `mish` | `default` | 11 of 65024 points returned inf or zero where a value exists | 1 | 0.997 | 1.95e-38 | — | — |
| wh | bf16 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | — | — |
| wh | bf16 | `mul_bw` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `multigammaln` | `default` | 11536 of 48328 points returned inf or zero where a value exists | 724 | 1.81 | 5.59e-17 | — | — |
| wh | bf16 | `multigammaln_bw` | `default` | accurate to |x| <= 5.59e-17; up to 2.38e+06 ULP beyond | 2.38e+06 | 294 | 5.59e-17 | — | — |
| wh | bf16 | `multiply` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `multiply_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `nextafter` | `default` | worst pairing 1.3e+33 ULP; mean 2.32e+31 | 1.3e+33 | 2.32e+31 | — | — | — |
| wh | bf16 | `polygamma` | `k=1` | 263 of 64769 points returned inf or zero where a value exists | 1.02e+08 | 1.32e+04 | 0.996 | — | — |
| wh | bf16 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 3.57e+08 | 6.81e+05 | 4.47 | — | — |
| wh | bf16 | `pow` | `default` | worst pairing 4.86e+18 ULP; mean 2.08e+14 | 4.86e+18 | 2.08e+14 | — | — | — |
| wh | bf16 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | — | — |
| wh | bf16 | `prelu` | `weight=0.25` | 1 of 65024 points returned inf or zero where a value exists | 0 | — | 4.68e-38 | — | — |
| wh | bf16 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | — | — |
| wh | bf16 | `rdiv` | `value=2.0` | 258 of 64770 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | — | — |
| wh | bf16 | `rdiv_bw` | `scalar=2.0` | 256 of 48492 points returned inf or zero where a value exists | 2 | 1.02 | 7.67e-20 | — | — |
| wh | bf16 | `reciprocal` | `default` | 2 of 64514 points returned inf or zero where a value exists | 1 | 1 | 8.47e+37 | — | — |
| wh | bf16 | `reciprocal_bw` | `default` | 256 of 48386 points returned inf or zero where a value exists | 2 | 1.02 | 5.42e-20 | — | — |
| wh | bf16 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `remainder` | `default` | worst pairing 6.65e+35 ULP; mean 1.06e+35 | 6.65e+35 | 1.06e+35 | — | — | — |
| wh | bf16 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `rpow` | `exponent=2.0` | within 2 ULP everywhere | 1 | 0.843 | 3.39e+38 | — | — |
| wh | bf16 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | — | — |
| wh | bf16 | `rsqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `rsqrt_bw` | `default` | 75 of 21665 points returned inf or zero where a value exists | 3 | 1.28 | — | — | — |
| wh | bf16 | `rsub` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `rsub_` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `selu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 3 | 1.04 | 0.582 | — | — |
| wh | bf16 | `sigmoid` | `default` | within 2 ULP everywhere | 1 | 0.876 | 3.39e+38 | — | — |
| wh | bf16 | `sigmoid_accurate` | `default` | within 2 ULP everywhere | 1 | 0.876 | 3.39e+38 | — | — |
| wh | bf16 | `sigmoid_bw` | `default` | 329 of 65024 points returned inf or zero where a value exists | 266 | 5.66 | 1.86 | — | — |
| wh | bf16 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `silu` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | — | — |
| wh | bf16 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 285 | 1.54 | 0.887 | — | — |
| wh | bf16 | `sin` | `default` | 21275 of 65024 points returned inf or zero where a value exists | 3.38e+38 | 2.53e+36 | 2.62e+05 | — | — |
| wh | bf16 | `sin_bw` | `default` | 21274 of 65024 points returned inf or zero where a value exists | 2.96e+38 | 2.71e+36 | 1.31e+05 | — | — |
| wh | bf16 | `sinh` | `default` | bit-exact | 0 | — | 89 | — | — |
| wh | bf16 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 0 | — | 88.5 | — | — |
| wh | bf16 | `softplus` | `default` | 526 of 65024 points returned inf or zero where a value exists | 1 | 1 | 5.03 | — | — |
| wh | bf16 | `softplus_bw` | `default` | within 2 ULP everywhere | 2 | 0.897 | 3.39e+38 | — | — |
| wh | bf16 | `softshrink` | `default` | within 2 ULP everywhere | 1 | 0.996 | 3.39e+38 | — | — |
| wh | bf16 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 1 | 0.69 | 8.51e+37 | — | — |
| wh | bf16 | `softsign_bw` | `default` | accurate to |x| <= 0.00491; up to 81 ULP beyond | 81 | 3.38 | 0.00491 | — | — |
| wh | bf16 | `sqrt` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `sqrt_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | bf16 | `square` | `default` | within 2 ULP everywhere | 1 | 0.991 | 1.84e+19 | — | — |
| wh | bf16 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | — | — |
| wh | bf16 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | — | — |
| wh | bf16 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.71 | — | — | — |
| wh | bf16 | `squared_difference_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `subalpha` | `alpha=2.0` | worst pairing 254 ULP; mean 2.42 | 254 | 2.42 | — | — | — |
| wh | bf16 | `subtract` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `subtract_` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | bf16 | `swish` | `default` | 13 of 65024 points returned inf or zero where a value exists | 1 | 0.998 | 2.33e-38 | — | — |
| wh | bf16 | `tan` | `default` | 21634 of 65024 points returned inf or zero where a value exists | 3.3e+38 | 1.16e+36 | 1.31e+05 | — | — |
| wh | bf16 | `tan_bw` | `default` | 12178 of 65024 points returned inf or zero where a value exists | 2.53e+38 | 3.12e+35 | 1.31e+05 | — | — |
| wh | bf16 | `tanh` | `default` | 2 of 65024 points returned inf or zero where a value exists | 1 | 1 | — | — | — |
| wh | bf16 | `tanh_bw` | `default` | accurate to |x| <= 17.2; up to 55.5 ULP beyond | 55.5 | 2.23 | 17.2 | — | — |
| wh | bf16 | `tanhshrink` | `default` | accurate to |x| <= 1.35e-08; up to 63.5 ULP beyond | 63.5 | 9.5 | 1.35e-08 | — | — |
| wh | bf16 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.45e-09; up to 63 ULP beyond | 63 | 3.71 | 7.45e-09 | — | — |
| wh | bf16 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | bf16 | `where` | `default` | bit-exact | 0 | — | — | — | — |
| wh | bf16 | `xielu` | `default` | 1 of 56847 points returned inf or zero where a value exists | 1 | 1 | 2.33e-38 | — | — |
| wh | bf16 | `xlogy` | `default` | worst pairing 897 ULP; mean 655 | 897 | 655 | — | — | — |
| wh | bf16 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | fp32 | `abs` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `abs_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `acos` | `default` | within 2 ULP everywhere | 2 | 1 | 1 | — | — |
| wh | fp32 | `acos_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | — | — |
| wh | fp32 | `acosh` | `default` | never within 2 ULP; mean 1, worst 3 | 3 | 1 | — | — | — |
| wh | fp32 | `acosh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 724 | 1.12 | 0.996 | — | — |
| wh | fp32 | `add` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `add_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `addalpha` | `alpha=2.0` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `addcdiv` | `default` | worst pairing 2.19e+12 ULP; mean 8.82e+07 | 2.19e+12 | 8.82e+07 | — | — | — |
| wh | fp32 | `addcmul` | `default` | worst pairing 2.13e+06 ULP; mean 1.59e+04 | 2.13e+06 | 1.59e+04 | — | — | — |
| wh | fp32 | `asin` | `default` | within 2 ULP everywhere | 2 | 1.08 | 1 | — | — |
| wh | fp32 | `asin_bw` | `default` | accurate to |x| <= 0.938; up to 609 ULP beyond | 609 | 1.61 | 0.938 | — | — |
| wh | fp32 | `asinh` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `asinh_bw` | `default` | 15874 of 64514 points returned inf or zero where a value exists | 2 | 1.03 | 1.84e+19 | — | — |
| wh | fp32 | `atan` | `default` | within 2 ULP everywhere | 2 | 1.02 | 3.39e+38 | — | — |
| wh | fp32 | `atan2` | `default` | worst pairing 1.32e+07 ULP; mean 1.26e+05 | 1.32e+07 | 1.26e+05 | — | — | — |
| wh | fp32 | `atan2_bw` | `default` | worst pairing 1.64e+07 ULP; mean 9.5e+04 | 1.64e+07 | 9.5e+04 | — | — | — |
| wh | fp32 | `atan_bw` | `default` | accurate to |x| <= 4.08e+03; up to 3 ULP beyond | 3 | 1.19 | 4.08e+03 | — | — |
| wh | fp32 | `atanh` | `default` | accurate to |x| <= 0.000852; up to 3 ULP beyond | 3 | 1.56 | 0.000852 | — | — |
| wh | fp32 | `atanh_bw` | `default` | accurate to |x| <= 0.852; up to 2.05e+03 ULP beyond | 2.05e+03 | 1.51 | 0.852 | — | — |
| wh | fp32 | `bias_gelu` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | — | — |
| wh | fp32 | `bias_gelu_` | `default` | worst pairing 2.91e+38 ULP; mean 4.94e+36 | 2.91e+38 | 4.94e+36 | — | — | — |
| wh | fp32 | `cbrt` | `default` | accurate to |x| <= 5.77e-38; up to 3 ULP beyond | 3 | 1.82 | 5.77e-38 | — | — |
| wh | fp32 | `ceil` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `celu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `celu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `clamp` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `clamp_bw` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `clip` | `min=-1.0,max=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `clip_bw` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `cos` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | — | — |
| wh | fp32 | `cos_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | — | — |
| wh | fp32 | `cosh` | `default` | within 2 ULP everywhere | 1 | 1 | 89 | — | — |
| wh | fp32 | `cosh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 2 | 1.14 | 88.5 | — | — |
| wh | fp32 | `deg2rad` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `digamma` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | — | — |
| wh | fp32 | `digamma_bw` | `default` | 255 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 8.58e-05 | — | — |
| wh | fp32 | `div` | `default` | within 2 ULP everywhere | 2 | 1.2 | — | — | — |
| wh | fp32 | `div_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | fp32 | `div_no_nan` | `default` | within 2 ULP everywhere | 2 | 1.53 | — | — | — |
| wh | fp32 | `divide` | `default` | within 2 ULP everywhere | 2 | 1.2 | — | — | — |
| wh | fp32 | `divide_` | `default` | within 2 ULP everywhere | 2 | 1.2 | — | — | — |
| wh | fp32 | `elu` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `elu_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `erf` | `default` | accurate to |x| <= 0.000334; up to 6 ULP beyond | 6 | 1.22 | 0.000334 | — | — |
| wh | fp32 | `erf_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | — | — |
| wh | fp32 | `erfc` | `default` | never within 2 ULP; mean 1.71e+29, worst 2.1e+33 | 2.1e+33 | 1.71e+29 | — | — | — |
| wh | fp32 | `erfc_bw` | `default` | accurate to |x| <= 0.996; up to 65 ULP beyond | 65 | 3.72 | 0.996 | — | — |
| wh | fp32 | `erfinv` | `default` | 29420 of 32256 points returned inf or zero where a value exists | 7.96e+06 | 2.62e+05 | 1.32e-38 | — | — |
| wh | fp32 | `erfinv_bw` | `default` | accurate to |x| <= 0.000496; up to 4.98e+05 ULP beyond | 4.98e+05 | 1.32e+03 | 0.000496 | — | — |
| wh | fp32 | `exp` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `exp` | `fast_approx` | never within 2 ULP; mean 2e+05, worst 3.79e+05 | 3.79e+05 | 2e+05 | — | — | — |
| wh | fp32 | `exp2` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `exp2_bw` | `default` | 1 of 49537 points returned inf or zero where a value exists | 2 | 1.17 | 128 | — | — |
| wh | fp32 | `exp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `expm1` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `expm1_bw` | `default` | 138 of 49458 points returned inf or zero where a value exists | 4.91e+06 | 1.14e+04 | 2.08 | — | — |
| wh | fp32 | `floor` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `floor_div` | `default` | worst pairing 4.19e+06 ULP; mean 1.77e+03 | 4.19e+06 | 1.77e+03 | — | — | — |
| wh | fp32 | `fmod` | `default` | worst pairing 1.7e+38 ULP; mean 2.72e+36 | 1.7e+38 | 2.72e+36 | — | — | — |
| wh | fp32 | `frac` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `ge_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `gelu` | `default` | 83 of 65024 points returned inf or zero where a value exists | 1.51e+08 | 1.55e+05 | 0.309 | — | — |
| wh | fp32 | `gelu` | `fast_approx` | 197 of 65024 points returned inf or zero where a value exists | 2.91e+38 | 4.9e+36 | 6e-36 | — | — |
| wh | fp32 | `gelu_bw` | `default` | accurate to |x| <= 0.0393; up to 6.59e+08 ULP beyond | 6.59e+08 | 1.24e+05 | 0.0393 | — | — |
| wh | fp32 | `gt_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `hardmish` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `hardshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `hardshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `hardsigmoid` | `default` | accurate to |x| <= 2.61; up to 4.89e+06 ULP beyond | 4.89e+06 | 1.25e+03 | 2.61 | — | — |
| wh | fp32 | `hardsigmoid_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `hardswish` | `default` | accurate to |x| <= 2.36; up to 7.34e+06 ULP beyond | 7.34e+06 | 1.41e+03 | 2.36 | — | — |
| wh | fp32 | `hardswish_bw` | `default` | accurate to |x| <= 0.00179; up to 1.41e+10 ULP beyond | 1.41e+10 | 6e+06 | 0.00179 | — | — |
| wh | fp32 | `hardtanh` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `hardtanh_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `heaviside` | `value=0.5` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `hypot` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 3.44e+06 | 3.28e+04 | — | — | — |
| wh | fp32 | `hypot_bw` | `default` | 16384 of 65024 points returned inf or zero where a value exists | 4.86e+06 | 4.48e+04 | — | — | — |
| wh | fp32 | `i0` | `default` | accurate to |x| <= 2.56; up to 1.68e+07 ULP beyond | 1.68e+07 | 2.19e+06 | 2.56 | — | — |
| wh | fp32 | `i0_bw` | `default` | accurate to |x| <= 0.00774; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.00774 | — | — |
| wh | fp32 | `i1` | `default` | accurate to |x| <= 0.00774; up to 1.56e+07 ULP beyond | 1.56e+07 | 4.11e+04 | 0.00774 | — | — |
| wh | fp32 | `identity` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `l1_loss` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `ldexp` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | — | — |
| wh | fp32 | `ldexp_` | `default` | within 2 ULP everywhere | 2 | 1.5 | — | — | — |
| wh | fp32 | `ldexp_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
| wh | fp32 | `le_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `leaky_relu` | `negative_slope=0.01` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `leaky_relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `lerp` | `default` | worst pairing 1.46e+08 ULP; mean 9.53e+04 | 1.46e+08 | 9.53e+04 | — | — | — |
| wh | fp32 | `lerp_bw` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `lgamma` | `default` | accurate to |x| <= 0.108; up to 3.27e+07 ULP beyond | 3.27e+07 | 2.98e+03 | 0.108 | — | — |
| wh | fp32 | `lgamma_bw` | `default` | never within 2 ULP; mean 7.77e+12, worst 3.66e+17 | 3.66e+17 | 7.77e+12 | — | — | — |
| wh | fp32 | `log` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `log10` | `default` | within 2 ULP everywhere | 2 | 1.22 | 3.39e+38 | — | — |
| wh | fp32 | `log10_bw` | `default` | within 2 ULP everywhere | 2 | 1.41 | 3.69e+37 | — | — |
| wh | fp32 | `log1p` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `log1p_bw` | `default` | within 2 ULP everywhere | 2 | 1.03 | 8.51e+37 | — | — |
| wh | fp32 | `log2` | `default` | within 2 ULP everywhere | 2 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `log2_bw` | `default` | 112 of 64626 points returned inf or zero where a value exists | 2 | 1.33 | — | — | — |
| wh | fp32 | `log_bw` | `default` | within 2 ULP everywhere | 1 | 1 | 8.51e+37 | — | — |
| wh | fp32 | `log_sigmoid` | `default` | never within 2 ULP; mean 1.82e+04, worst 4.4e+05 | 4.4e+05 | 1.82e+04 | — | — | — |
| wh | fp32 | `log_sigmoid_bw` | `default` | accurate to |x| <= 1.09; up to 3 ULP beyond | 3 | 1.25 | 1.09 | — | — |
| wh | fp32 | `logaddexp` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | — | — |
| wh | fp32 | `logaddexp2` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | — | — |
| wh | fp32 | `logaddexp2_` | `default` | 15488 of 65024 points returned inf or zero where a value exists | 1.54e+09 | 1.05e+07 | — | — | — |
| wh | fp32 | `logaddexp2_bw` | `default` | worst pairing 46 ULP; mean 6.56 | 46 | 6.56 | — | — | — |
| wh | fp32 | `logaddexp_` | `default` | 15566 of 65024 points returned inf or zero where a value exists | 5.23e+08 | 7.28e+06 | — | — | — |
| wh | fp32 | `logaddexp_bw` | `default` | worst pairing 66 ULP; mean 8.57 | 66 | 8.57 | — | — | — |
| wh | fp32 | `logical_not_` | `default` | bit-exact | 0 | — | 1 | — | — |
| wh | fp32 | `logical_xor_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `logit` | `default` | accurate to |x| <= 0.332; up to 4.19e+06 ULP beyond | 4.19e+06 | 261 | 0.332 | — | — |
| wh | fp32 | `logit_bw` | `default` | within 2 ULP everywhere | 2 | 1.06 | 0.998 | — | — |
| wh | fp32 | `logiteps_bw` | `default` | within 2 ULP everywhere | 2 | 1.06 | 0.998 | — | — |
| wh | fp32 | `lt_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `mac` | `default` | worst pairing 4.22e+06 ULP; mean 7.31e+05 | 4.22e+06 | 7.31e+05 | — | — | — |
| wh | fp32 | `max_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | — | — |
| wh | fp32 | `maximum` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `min_bw` | `default` | worst pairing 4.19e+06 ULP; mean 4.19e+06 | 4.19e+06 | 4.19e+06 | — | — | — |
| wh | fp32 | `minimum` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `mish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 7 | 2.49 | 8.61e-06 | — | — |
| wh | fp32 | `mse_loss` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | — | — |
| wh | fp32 | `mul_bw` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `multigammaln` | `default` | 7422 of 50376 points returned inf or zero where a value exists | 3.51e+07 | 9.16e+03 | 5.55e-17 | — | — |
| wh | fp32 | `multigammaln_bw` | `default` | accurate to |x| <= 5.55e-17; up to 1.36e+15 ULP beyond | 1.36e+15 | 2.04e+11 | 5.55e-17 | — | — |
| wh | fp32 | `multiply` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `multiply_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `neg` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `nextafter` | `default` | worst pairing 8.51e+37 ULP; mean 1.34e+36 | 8.51e+37 | 1.34e+36 | — | — | — |
| wh | fp32 | `polygamma` | `k=1` | 255 of 64769 points returned inf or zero where a value exists | 7.91e+33 | 3.25e+29 | 8.58e-05 | — | — |
| wh | fp32 | `polygamma_bw` | `n=1` | 75 of 49922 points returned inf or zero where a value exists | 4.45e+30 | 4.42e+26 | 1.79e-13 | — | — |
| wh | fp32 | `pow` | `default` | worst pairing 5.92e+03 ULP; mean 3.21 | 5.92e+03 | 3.21 | — | — | — |
| wh | fp32 | `pow_bw` | `exponent=2.0` | bit-exact | 0 | — | 1.69e+38 | — | — |
| wh | fp32 | `prelu` | `weight=0.25` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `rad2deg` | `default` | within 2 ULP everywhere | 1 | 1 | 5.9e+36 | — | — |
| wh | fp32 | `rdiv` | `value=2.0` | 256 of 64770 points returned inf or zero where a value exists | 1 | 1 | 8.51e+37 | — | — |
| wh | fp32 | `rdiv_bw` | `scalar=2.0` | 252 of 48490 points returned inf or zero where a value exists | 2 | 1.14 | 7.67e-20 | — | — |
| wh | fp32 | `reciprocal` | `default` | within 2 ULP everywhere | 1 | 1 | 8.51e+37 | — | — |
| wh | fp32 | `reciprocal_bw` | `default` | 254 of 48386 points returned inf or zero where a value exists | 2 | 1.14 | 5.42e-20 | — | — |
| wh | fp32 | `relu` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `relu6` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `relu6_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `relu_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `relu_max` | `upper_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `relu_min` | `lower_limit=1.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `remainder` | `default` | worst pairing 1.7e+38 ULP; mean 2.68e+36 | 1.7e+38 | 2.68e+36 | — | — | — |
| wh | fp32 | `round` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `rpow` | `exponent=2.0` | 1536 of 49536 points returned inf or zero where a value exists | 1 | 1 | 8.28e+34 | — | — |
| wh | fp32 | `rpow_bw` | `exponent=2.0` | 32384 of 64768 points returned inf or zero where a value exists | 0 | — | 1.18e-38 | — | — |
| wh | fp32 | `rsqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `rsqrt_bw` | `default` | 75 of 21666 points returned inf or zero where a value exists | 7 | 4.51 | — | — | — |
| wh | fp32 | `rsub` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `rsub_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `selu` | `default` | never within 2 ULP; mean 26.2, worst 51 | 51 | 26.2 | — | — | — |
| wh | fp32 | `selu_bw` | `default` | 1 of 65024 points returned inf or zero where a value exists | 51 | 20.6 | — | — | — |
| wh | fp32 | `sigmoid` | `default` | accurate to |x| <= 16.6; up to 3 ULP beyond | 3 | 1.41 | 16.6 | — | — |
| wh | fp32 | `sigmoid_accurate` | `default` | accurate to |x| <= 16.6; up to 3 ULP beyond | 3 | 1.41 | 16.6 | — | — |
| wh | fp32 | `sigmoid_bw` | `default` | 140 of 65024 points returned inf or zero where a value exists | 8.39e+06 | 2.71e+04 | 0.859 | — | — |
| wh | fp32 | `sign` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `silu` | `default` | 9 of 65024 points returned inf or zero where a value exists | 3 | 1.58 | 0.000928 | — | — |
| wh | fp32 | `silu_bw` | `default` | 9 of 65024 points returned inf or zero where a value exists | 5.72e+06 | 841 | 0.000244 | — | — |
| wh | fp32 | `sin` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.32e+38 | 1.86e+36 | 1.04e+05 | — | — |
| wh | fp32 | `sin_bw` | `default` | 18190 of 65024 points returned inf or zero where a value exists | 3.21e+38 | 1.45e+36 | 5.23e+04 | — | — |
| wh | fp32 | `sinh` | `default` | within 2 ULP everywhere | 2 | 1.14 | 89 | — | — |
| wh | fp32 | `sinh_bw` | `default` | 2 of 33894 points returned inf or zero where a value exists | 1 | 1 | 88.5 | — | — |
| wh | fp32 | `softplus` | `default` | 30 of 65024 points returned inf or zero where a value exists | 8.2e+03 | 656 | — | — | — |
| wh | fp32 | `softplus_bw` | `default` | accurate to |x| <= 1.1; up to 3 ULP beyond | 3 | 1.37 | 1.1 | — | — |
| wh | fp32 | `softshrink` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `softshrink_bw` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `softsign` | `default` | 510 of 65024 points returned inf or zero where a value exists | 3 | 0.89 | 2.53e+07 | — | — |
| wh | fp32 | `softsign_bw` | `default` | accurate to |x| <= 9.88e-05; up to 8.38e+06 ULP beyond | 8.38e+06 | 1.54e+05 | 9.88e-05 | — | — |
| wh | fp32 | `sqrt` | `default` | within 2 ULP everywhere | 1 | 1 | 3.39e+38 | — | — |
| wh | fp32 | `sqrt_bw` | `default` | within 2 ULP everywhere | 2 | 1.41 | 3.39e+38 | — | — |
| wh | fp32 | `square` | `default` | bit-exact | 0 | — | 1.84e+19 | — | — |
| wh | fp32 | `square_bw` | `default` | bit-exact | 0 | — | 1.69e+38 | — | — |
| wh | fp32 | `squared_difference` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | — | — |
| wh | fp32 | `squared_difference_` | `default` | within 2 ULP everywhere | 2 | 1.94 | — | — | — |
| wh | fp32 | `squared_difference_bw` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `subalpha` | `alpha=2.0` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `subtract` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `subtract_` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `swish` | `default` | 9 of 65024 points returned inf or zero where a value exists | 3 | 1.58 | 0.000928 | — | — |
| wh | fp32 | `tan` | `default` | 20110 of 65024 points returned inf or zero where a value exists | 3.34e+38 | 1.43e+36 | 20.3 | — | — |
| wh | fp32 | `tan_bw` | `default` | 20376 of 65024 points returned inf or zero where a value exists | 3.16e+38 | 9.23e+35 | 1.03 | — | — |
| wh | fp32 | `tanh` | `default` | accurate to |x| <= 0.000485; up to 3 ULP beyond | 3 | 1.55 | 0.000485 | — | — |
| wh | fp32 | `tanh_bw` | `default` | never within 2 ULP; mean 9.01e+03, worst 4.19e+06 | 4.19e+06 | 9.01e+03 | — | — | — |
| wh | fp32 | `tanhshrink` | `default` | accurate to |x| <= 1.34e-08; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.11e+04 | 1.34e-08 | — | — |
| wh | fp32 | `tanhshrink_bw` | `default` | accurate to |x| <= 7.42e-09; up to 4.19e+06 ULP beyond | 4.19e+06 | 9.85e+04 | 7.42e-09 | — | — |
| wh | fp32 | `threshold` | `threshold=0.5,value=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `threshold_bw` | `min=0.5,max=0.0` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `trunc` | `default` | bit-exact | 0 | — | 3.39e+38 | — | — |
| wh | fp32 | `where` | `default` | bit-exact | 0 | — | — | — | — |
| wh | fp32 | `xielu` | `default` | accurate to |x| <= 0.691; up to 1.08e+07 ULP beyond | 1.08e+07 | 256 | 0.691 | — | — |
| wh | fp32 | `xlogy` | `default` | worst pairing 8.84e+07 ULP; mean 6.07e+07 | 8.84e+07 | 6.07e+07 | — | — | — |
| wh | fp32 | `xlogy_bw` | `default` | within 2 ULP everywhere | 1 | 1 | — | — | — |
