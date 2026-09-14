# Scope and sweep strategy

222 eltwise ops, in 44 measured variants. Everything else is in
[uncovered.md](../analyze-report/uncovered.md).

## What each sweep covers

Per op, per architecture, per dtype:

| Arity | Ops | bf16 | fp32 |
|---|---|---|---|
| unary | 79 | exhaustive, all 2¹⁶ | exhaustive, all 2³² in 1,024 blocks |
| unary_bw | 56 | exhaustive | exhaustive |
| binary | 41 | **exhaustive, all 2¹⁶ × 2¹⁶** | sampled, 2¹⁶ cells × 2¹⁶ cells |
| binary_bw | 12 | exhaustive | sampled |
| ternary | 5 | first operand exhaustive, others strided | sampled and strided |
| ternary_bw | 3 | same | same |

A sampled maximum is a lower bound. Every sampled page says so.

## Why sampling exists

At roughly 63M pairs/s on Wormhole:

| Sweep | Points per op | Time per op |
|---|---|---|
| unary bf16 | 6.6e4 | 2 s |
| unary fp32 | 4.3e9 | 2 min |
| binary bf16, and fp32 sampled | 4.2e9 | 70 s |
| ternary bf16 | 1.1e9 | 45 s |
| binary fp32 exhaustive | 1.8e19 | 9,000 years |
| ternary bf16 exhaustive | 2.8e14 | 140 days |

## Decisions

`forced` means there was no real alternative. `convergent` means another tool reached the
same answer independently. `preference` and `divergent` are ours to defend.

| Decision | Class | Choice |
|---|---|---|
| ULP definition | forced | tt-metal's own, so the org measures one thing. Only the fp32-wide return is ours |
| subnormals | forced | flushed on both sides, because the hardware flushes them |
| outcome per point | forced | `exact`, `inexact`, `flushed`, `zeroed`, `overflow`, `undefined`, `mismatch`, `special`. ULP is NaN where undefined |
| max and mean | convergent | over defined, non-trivial points only. Otherwise `tanh_bw` reads 1e24 for an absolute error of 0.003 |
| fp32 grouping | preference | one row is the worst of 2¹⁶ codes: max stays exact, 3 MB per op, chartable |
| binary generation | adopted | all 2³² pairs, streamed 128 second-operands at a time |
| binary reduction | divergent | one row per first operand. 65,536 rows, 128× the resolution of the tool this is based on |
| row coherence | forced | a row is one pair, the worst-ULP partner for that x |
| fp32 pair sample | preference | scheme B below, seed fixed forever. A moving sample makes error changes unattributable |
| ternary stride | preference | every 512th code. Uniform over exponents, priced like a binary sweep |
| domains | preference | fp64 bisection per dtype, unary only. Refusals recorded with reasons |
| domain shape | forced | one interval, so a hole cannot be expressed. `digamma` and `lgamma_bw` are NaN at every negative integer, `acosh_bw` for \|x\| < 1, `multigammaln_bw` below 0.5. Those points are swept and land as `undefined`, carry no ULP and count as no defect, so nothing scores wrongly. The cost is sweep time |
| computability bound | forced | `polygamma_bw` swept above −1024. Torch's reference costs O(\|x\|) below zero |

### fp32 pair sample, options considered

| | Scheme | Trade-off |
|---|---|---|
| A | widened bf16 grid | the low 16 mantissa bits are always zero, and those are the bits that drive rounding |
| **B** | strided exponents, seeded random mantissas | every bf16 cell once, rounding exercised, deterministic |
| C | exhaustive first operand | 8 days, and asymmetric, so wrong for `subtract`, `pow`, `atan2` |
| D | skip fp32 binary | an honest hole |

## Open questions

| Question | What it needs |
|---|---|
| report the diluted mean beside the useful one, as ttnn-eltwise-op-tester does? | a decision, then a column |
| attribute the `tanhshrink`, `digamma`, `digamma_bw` and `softplus` movements | one run at the old commit. Cheap now that provenance is per build |
