# Findings

**Status: measured.** Every row below is a published result that changed.

_bfc3f70fd49 → bfc3f70fd49_

## Expected (12)

No scored metric got worse on comparable measurements. Either `usable_to` moved, because a different kernel puts the 2 ULP boundary on a neighbouring group; or `n_inputs` moved, which means the swept domain changed and the two runs measure different populations; or a metric is reported here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/digamma_bw/default` | usable_to 0.996 → 1 |
| `bh/bf16/expm1_bw/default` | usable_to 33.5 → 33.2 |
| `bh/bf16/hardswish_bw/default` | usable_to 1.13 → 1.12 |
| `bh/bf16/lgamma/default` | usable_to 0.412 → 0.414 |
| `bh/bf16/log_sigmoid/default` | usable_to 0.428 → 0.426 |
| `bh/bf16/multigammaln/default` | usable_to 5.59e-17 → 5.55e-17 |
| `bh/bf16/multigammaln_bw/default` | usable_to 5.59e-17 → 5.55e-17 |
| `bh/bf16/polygamma/k4` | usable_to 3.5 → 3.48 |
| `bh/bf16/sigmoid_bw/default` | usable_to 12.5 → 12.4 |
| `bh/bf16/silu_bw/default` | usable_to 0.863 → 0.867 |
| `bh/fp32/multigammaln/default` | usable_to 5.55e-17 → 5.59e-17 |
| `bh/fp32/silu_bw/default` | usable_to 2.98e-07 → 2.96e-07 |

## No longer measured (32)

Present in the baseline, absent now.

| Variant | Moved |
|---|---|
| `bh/bf16/bias_gelu/default` | — |
| `bh/bf16/eq/default` | — |
| `bh/bf16/ge/default` | — |
| `bh/bf16/gt/default` | — |
| `bh/bf16/ldexp/default` | — |
| `bh/bf16/le/default` | — |
| `bh/bf16/logaddexp/default` | — |
| `bh/bf16/logaddexp2/default` | — |
| `bh/bf16/logical_and/default` | — |
| `bh/bf16/logical_or/default` | — |
| `bh/bf16/lt/default` | — |
| `bh/bf16/maximum/default` | — |
| `bh/bf16/minimum/default` | — |
| `bh/bf16/ne/default` | — |
| `bh/bf16/remainder/default` | — |
| `bh/bf16/xlogy/default` | — |
| `bh/fp32/bias_gelu/default` | — |
| `bh/fp32/eq/default` | — |
| `bh/fp32/ge/default` | — |
| `bh/fp32/gt/default` | — |
| `bh/fp32/ldexp/default` | — |
| `bh/fp32/le/default` | — |
| `bh/fp32/logaddexp/default` | — |
| `bh/fp32/logaddexp2/default` | — |
| `bh/fp32/logical_and/default` | — |
| `bh/fp32/logical_or/default` | — |
| `bh/fp32/lt/default` | — |
| `bh/fp32/maximum/default` | — |
| `bh/fp32/minimum/default` | — |
| `bh/fp32/ne/default` | — |
| `bh/fp32/remainder/default` | — |
| `bh/fp32/xlogy/default` | — |
