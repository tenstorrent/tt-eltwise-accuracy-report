# Scope and sweep strategy

222 eltwise ops (see `analyze-report/uncovered.md` for everything else). Per op, per
architecture, per dtype:

| Arity | Ops | bf16 | fp32 |
|---|---|---|---|
| unary | 79 | exhaustive — all 2¹⁶ | exhaustive — all 2³², 1024 blocks |
| unary_bw | 56 | exhaustive | exhaustive |
| binary | 41 | **exhaustive — all 2¹⁶ × 2¹⁶** | sampled — 2¹⁶ cells × 2¹⁶ cells |
| binary_bw | 12 | exhaustive | sampled |
| ternary | 5 | first operand exhaustive, others strided | sampled + strided |
| ternary_bw | 3 | same | same |

Every sampled page carries an explicit note: a sampled maximum is a lower bound.

## Cost (~63M pairs/s on Wormhole)

| Sweep | Points per op | Time per op |
|---|---|---|
| unary bf16 | 6.6e4 | ~2 s |
| unary fp32 | 4.3e9 | ~2 min |
| binary bf16 · fp32 sampled | 4.2e9 | ~70 s |
| ternary bf16 | 1.1e9 | ~45 s |
| binary fp32 exhaustive | 1.8e19 | ~9,000 years — why sampling exists |
| ternary bf16 exhaustive | 2.8e14 | ~140 days — same |

## Decisions

| Decision | Class | Choice |
|---|---|---|
| ULP definition | forced | tt-metal's own, so the org measures one thing; only the fp32-wide return is ours |
| subnormals | forced | flushed on both sides — the hardware flushes them |
| outcome per point | forced | `exact/inexact/flushed/zeroed/overflow/undefined/mismatch/special`; ULP NaN where undefined |
| max / mean | convergent | over defined, non-trivial points only — else `tanh_bw` reads 1e24 for a 0.003 abs error |
| fp32 grouping | preference | one row = worst of 2¹⁶ codes: max exact, 3 MB/op, chartable |
| binary generation | adopted | all 2³² pairs, streamed 128 second-operands at a time |
| binary reduction | divergent | one row per first operand — 65,536 rows, 128× the other tool's resolution |
| row coherence | forced | a row is one pair: the worst-ULP partner for that x |
| fp32 pair sample | preference | scheme B below, seed fixed forever — a moving sample makes error changes unattributable |
| ternary stride | preference | every 512th code — uniform over exponents, priced like a binary sweep |
| domains | preference | fp64 bisection per dtype, unary only; refusals recorded with reasons |
| domain shape | forced | one interval. `digamma` and `lgamma_bw` are NaN at every negative integer, `acosh_bw` NaN for \|x\| < 1, `multigammaln_bw` below 0.5 — a bisection cannot express a hole, so those points are swept and land as `undefined`. They carry no ULP and are not counted as defects, so nothing is scored wrongly; the cost is sweep time |
| computability bound | forced | `polygamma_bw` swept above −1024 — torch's reference costs O(\|x\|) below zero |

### fp32 pair sample — options considered

| | Scheme | Trade-off |
|---|---|---|
| A | widened bf16 grid | low 16 mantissa bits always zero — the bits that drive rounding |
| **B** | strided exponents + seeded random mantissas | every bf16 cell once, rounding exercised, deterministic |
| C | exhaustive first operand | ~8 days; asymmetric — wrong for `subtract`, `pow`, `atan2` |
| D | skip fp32 binary | honest hole |

## Open

- Report the diluted mean beside the useful one, as ttnn-eltwise-op-tester does?
- Attribute the `tanhshrink` / `digamma` / `digamma_bw` / `softplus` movements — needs one
  run at the old commit; cheap now that provenance is per build.
