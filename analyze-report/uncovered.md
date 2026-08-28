# Uncovered ops

The manifest's 196 ops are covered. Everything else, by what covering it would take:

| Group | Ops | Recorded reason | To cover | Effort |
|---|---|---|---|---|
| integer eltwise | `bitwise_and/or/xor/not`, `bitwise_left/right_shift`, `logical_left/right_shift`, `gcd`, `lcm`, `plus_one` | `not implemented for 'Double'` / INT32-only | exact-match mode: INT32 sweep, bit identity, no ULP | medium |
| `where_bw` | `where_bw` | condition must be bool; d/d(condition) undefined | `sweep_axis` field on `Override`, sweep operand 2 | small |
| constant-derivative backwards | `add_bw`, `sub_bw`, `neg_bw`, `ceil_bw`, `floor_bw`, `round_bw`, `trunc_bw`, `sign_bw`, `deg2rad_bw`, `rad2deg_bw`, `frac_bw`, `rsub_bw`, `fmod_bw`, `remainder_bw`, `addalpha_bw`, `subalpha_bw`, `addcdiv_bw`, `addcmul_bw`, `div_no_nan_bw`, `fill_bw` | flat at gradient=ones — zero information | sweep the gradient axis; also catches grad-ignoring kernels in every measured `_bw` op | medium |
| goldenless eltwise | ~120 unscanned names; `divide` already recovered | no golden attached | workflow 2: LLM scan, supplied torch golden per hit | small per op |
| broken goldens | `topk`, `is_imag`, `is_real`, `bitcast`, `polar` | golden itself crashes (`NameError: torch`, …) | file upstream; none is measurable eltwise anyway | — |
| `softcap` | `softcap` | device asserts `arch == BLACKHOLE` | the BH run | pending |

Permanently out:

| Class | Ops | Reason |
|---|---|---|
| complex-valued | `angle`, `conj`, `real`, `imag`, `polar`, their `_bw` | the sweep sends real floats |
| predicates | `isnan`, `eq`, `lt`, `signbit`, … | ULP between booleans is not a quantity |
| movement · creation · composites | pools, conv, attention, norms, reshapes, KV-cache, `zeros`/`ones`/`full`, `to_*` | not elementwise mathematics |
