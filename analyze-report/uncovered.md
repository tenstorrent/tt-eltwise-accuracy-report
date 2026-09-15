# Uncovered ops

**Status: backlog.** Nothing on this page is measured. 222 ops in 44 variants are;
everything else is here, with what it would take.

## Not measured

| Group | Ops | Why not | To cover | Effort |
|---|---|---|---|---|
| integer eltwise | `bitwise_and/or/xor/not`, `bitwise_left/right_shift`, `logical_left/right_shift`, `gcd`, `lcm`, `plus_one` | golden is `not implemented for 'Double'`. INT32 in, INT32 out | an integer sweep. The float path cannot express the inputs; scoring stays exact-match, as it already is for predicates | medium |
| `where_bw` | `where_bw` | condition must be bool, d/d(condition) undefined | a `sweep_axis` field on `Override`, sweeping operand 2 | small |
| constant-derivative backwards | `add_bw`, `sub_bw`, `neg_bw`, `ceil_bw`, `floor_bw`, `round_bw`, `trunc_bw`, `sign_bw`, `deg2rad_bw`, `rad2deg_bw`, `frac_bw`, `rsub_bw`, `fmod_bw`, `remainder_bw`, `addalpha_bw`, `subalpha_bw`, `addcdiv_bw`, `addcmul_bw`, `div_no_nan_bw`, `fill_bw` | flat at gradient=ones, so zero information | sweep the gradient axis. Also catches grad-ignoring kernels in every measured `_bw` op | medium |
| goldenless eltwise | ~120 unscanned names. `divide` already recovered | no golden attached | scan, then supply a torch golden per hit | small per op |
| broken goldens | `topk`, `is_imag`, `is_real`, `bitcast`, `polar` | the golden itself crashes (`NameError: torch`, and others) | file upstream. None is measurable eltwise anyway | upstream |
| `softcap` | `softcap` | device asserts `arch == BLACKHOLE`, rejected on wh | a BH run | blocked, no BH runner |
| `bias_gelu_bw` | `bias_gelu_bw` | crashes the process while probing, on both architectures | a device-side fix. `charts` drops its stale entry, so nothing stale is published | upstream |
| `hardswish` | `hardswish` | measured again at tt-metal `9f9cd4fd590`. The TRISC compile failure at `9d286803c55`, where `hardswish_kernel.cpp:14` read `get_compile_time_arg_val(0)` and the host passed none, is fixed upstream | nothing | resolved |

91 goldens refuse the probe, each reason recorded under `unprobeable`. That is not a 91-op
hole. About 75 are correctly identified as not elementwise: they need a `dim`, a 2-D weight
or a cache layout, and fail when handed a flat vector. About 10 are the integer ops above.
Three are broken upstream: `topk` references `torch` without importing it, `is_imag` and
`is_real` call functions torch does not have.

## Measured, but not in every configuration

| Axis | State |
|---|---|
| row-major layout | 160 of 215 wh ops accept it. None is measured in it. The largest untested surface in the report |
| sampled maxima | binary fp32 and every ternary sweep sample their operands, so those `max_ulp` figures are lower bounds. `ttnn-accuracy refine` sweeps the cell a sample drew from: `fp32/pow` publishes 5,918 ULP and holds 8,300 |
| shape | one tile-aligned block. `check` requires the answer not to change with the tiling, but no published figure varies it |
| scalar parameters | 26 ops take one, 11 are measured at more than one value. A scalar's boundaries are where the op degenerates: `leaky_relu(0.0)` is `relu`, `clamp(min>max)` is empty. `polygamma` has defects at k=2 and k=4 that k=1 does not show |

## Permanently out

| Class | Ops | Reason |
|---|---|---|
| complex-valued | `angle`, `conj`, `real`, `imag`, `polar`, and their `_bw` | the sweep sends real floats |
| movement, creation, composites | pools, conv, attention, norms, reshapes, KV-cache, `zeros`/`ones`/`full`, `to_*` | not elementwise mathematics |

Predicates used to sit here, on the grounds that ULP between booleans is not a quantity.
That was wrong. A predicate answers 0 or 1 exactly, so scoring it as a float makes ULP
degenerate to right-or-wrong and needs no second path. 35 comparison and logical ops are
now measured and every one is bit-exact.

Two blind spots in the probe had kept them out, both fixed. A predicate is constant across
any finite positive range, so `gtz` read as a constant like `zeros_like` until the probe
also tried zero; `isnan` and its four siblings needed inf, −inf and NaN for the same reason.
Anyone generating an op list by probing goldens will hit both.
