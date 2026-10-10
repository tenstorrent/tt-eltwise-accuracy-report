# Findings

**Status: measured.** Every row below is a published result that changed.

_de546d3b146 → de546d3b146_

## Expected (7)

No scored metric got worse on comparable measurements. Either `usable_to` moved, because a different kernel puts the 2 ULP boundary on a neighbouring group; or `n_inputs` moved, which means the swept domain changed and the two runs measure different populations; or a metric is reported here for the first time.

| Variant | Moved |
|---|---|
| `bh/bf16/digamma_bw/default` | usable_to 0.996 → 1 |
| `bh/bf16/expm1_bw/default` | usable_to 33.5 → 33.2 |
| `bh/bf16/hardswish_bw/default` | usable_to 1.13 → 1.12 |
| `bh/bf16/multigammaln_bw/default` | usable_to 5.59e-17 → 5.55e-17 |
| `bh/bf16/sigmoid_bw/default` | usable_to 1.76 → 1.77 |
| `bh/bf16/silu_bw/default` | usable_to 0.863 → 0.867 |
| `bh/fp32/silu_bw/default` | usable_to 2.98e-07 → 2.96e-07 |
