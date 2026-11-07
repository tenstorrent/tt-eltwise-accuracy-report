# Findings

**Status: measured.** Every row below is a published result that changed.

_d80d9482a41 → d80d9482a41_

## Expected (5)

No scored metric got worse on comparable measurements. Either `usable_to` moved, because a different kernel puts the 2 ULP boundary on a neighbouring group; or `n_inputs` moved, which means the swept domain changed and the two runs measure different populations; or a metric is reported here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/lgamma/default` | usable_to 0.414 → 0.412 |
| `bh/bf16/log_sigmoid/default` | usable_to 0.426 → 0.428 |
| `bh/bf16/multigammaln/default` | usable_to 5.55e-17 → 5.59e-17 |
| `bh/bf16/polygamma/k4` | usable_to 3.48 → 3.5 |
| `bh/fp32/multigammaln/default` | usable_to 5.59e-17 → 5.55e-17 |

## No longer measured (2)

Present in the baseline, absent now.

| Variant | Moved |
|---|---|
| `wh/bf16/lerp_bw/default` | — |
| `wh/fp32/lerp_bw/default` | — |
