# Interpretation contract

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
| `perf` | `us_median` and `melem_per_s` for one dispatch over 2²⁴ resident elements, plus the `host` that took them and, only when the row bounced past the harness's own reproducibility, `spread_pct`. Never an accuracy figure and never scored: two timings may be compared only when their `host` matches, and a `±N%` on a page means that row is not to be trusted |
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
- Never compare a timing to one from another `host`, and never call a timing difference a
  regression: it can be the room. Accuracy is deterministic; timing is not.
- Not in the index → say it is not measured, and read the reason from the manifest's
  exclusions and refusals, or `analyze-report/uncovered.md`.
