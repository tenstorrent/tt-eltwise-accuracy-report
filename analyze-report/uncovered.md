# Uncovered ops

The manifest's 222 ops are covered, in 44 measured variants. Everything else, by what covering it would take:

| Group | Ops | Recorded reason | To cover | Effort |
|---|---|---|---|---|
| integer eltwise | `bitwise_and/or/xor/not`, `bitwise_left/right_shift`, `logical_left/right_shift`, `gcd`, `lcm`, `plus_one` | golden is `not implemented for 'Double'` — INT32 in, INT32 out | an integer sweep: the float path cannot express the inputs, though scoring is exact-match as the predicates already are | medium |
| `where_bw` | `where_bw` | condition must be bool; d/d(condition) undefined | `sweep_axis` field on `Override`, sweep operand 2 | small |
| constant-derivative backwards | `add_bw`, `sub_bw`, `neg_bw`, `ceil_bw`, `floor_bw`, `round_bw`, `trunc_bw`, `sign_bw`, `deg2rad_bw`, `rad2deg_bw`, `frac_bw`, `rsub_bw`, `fmod_bw`, `remainder_bw`, `addalpha_bw`, `subalpha_bw`, `addcdiv_bw`, `addcmul_bw`, `div_no_nan_bw`, `fill_bw` | flat at gradient=ones — zero information | sweep the gradient axis; also catches grad-ignoring kernels in every measured `_bw` op | medium |
| goldenless eltwise | ~120 unscanned names; `divide` already recovered | no golden attached | LLM scan, supplied torch golden per hit | small per op |
| broken goldens | `topk`, `is_imag`, `is_real`, `bitcast`, `polar` | golden itself crashes (`NameError: torch`, …) | file upstream; none is measurable eltwise anyway | — |
| `softcap` | `softcap` | device asserts `arch == BLACKHOLE`; rejected on wh | a BH run | pending, BH runner offline |
| `bias_gelu_bw` | `bias_gelu_bw` | crashes the process while probing, on both architectures | a device-side fix; `charts` drops its stale entry so nothing stale is published | upstream |
| `hardswish` | `hardswish` | TRISC compile fails at tt-metal `9d286803c55` — `hardswish_kernel.cpp:14` reads `get_compile_time_arg_val(0)` and the host passes none | an upstream fix; it measured fine at `f6deef232f7` | upstream regression |

91 goldens refuse the probe outright, each reason recorded under `unprobeable` — but that is
not a 91-op hole. Roughly 75 are ops the probe correctly identified as not elementwise: they
need a `dim`, a 2-D weight or a cache layout, and fail when handed a flat vector. About 10 are
the integer eltwise ops above. Three are goldens broken upstream (`topk` references `torch`
without importing it; `is_imag` and `is_real` call functions torch does not have), which cost
this report nothing and are unusable for anyone else validating those ops.

## Measured, but not in every configuration

| Axis | State |
|---|---|
| row-major layout | 160 of 215 wh ops accept it and none is measured in it — tile only. The largest untested surface in the report |
| sampled maxima | binary fp32 and every ternary sample their operands, so those `max_ulp` figures are lower bounds. `ttnn-accuracy refine` sweeps the cell the sample drew from: `fp32/pow` publishes 5,918 ULP and holds 8,300 |
| shape | one tile-aligned block. `check` requires the answer not to change with the tiling, but no published figure varies it |
| scalar parameters | 26 ops take one; 11 are measured at more than one value. A scalar's boundaries are where the op degenerates — `leaky_relu(0.0)` is `relu`, `clamp(min>max)` is empty — and `polygamma` has defects at k=2 and k=4 that k=1 does not show |

## Permanently out

| Class | Ops | Reason |
|---|---|---|
| complex-valued | `angle`, `conj`, `real`, `imag`, `polar`, their `_bw` | the sweep sends real floats |
| movement · creation · composites | pools, conv, attention, norms, reshapes, KV-cache, `zeros`/`ones`/`full`, `to_*` | not elementwise mathematics |

Predicates used to sit here, on the grounds that ULP between booleans is not a quantity.
That was the wrong conclusion: a predicate answers 0 or 1 exactly, so scoring it as a float
makes ULP degenerate to "right or wrong" and needs no second path. 35 comparison and logical
ops are now measured, and every one so far is bit-exact.

They were kept out by two separate blind spots in the probe, both since fixed. A predicate is
constant across any finite positive range, so `gtz` read as a constant like `zeros_like` until
the probe also tried zero; `isnan` and its four siblings needed inf, −inf and NaN for the same
reason. Anyone generating an op list by probing goldens will hit both.
