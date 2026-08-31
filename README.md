# tt-eltwise-accuracy-report

ULP accuracy of every TT-Metal eltwise op, measured exhaustively against fp64 torch
goldens — per architecture (WH, BH) and dtype (bf16, fp32).

**Browse:** [reports/README.md](reports/README.md) ·
**Design:** [docs/architecture.md](docs/architecture.md) ·
[docs/scope.md](docs/scope.md) · [analyze-report/](analyze-report/README.md) ·
[.github/workflows/](.github/workflows/README.md)

## Structure

```
src/ttnn_accuracy/
  cli.py                # discover | derive | probe | measure | charts | report | compare
  ops/
    introspect.py       # what ttnn registers
    arity.py            # operand count, elementwise, real-valued — by asking the golden
    manifest.py         # builds and holds stats/ops_manifest.json
    plan.py             # manifest → one OpSpec per variant
    overrides.py        # every editorial decision: scalars, variants, supplied goldens, exclusions
  domain/derive.py      # per-dtype input bounds from the golden in float64
  measure/              # metrics · sweeps · schema · store · runner · device
  report/               # charts (CSV → SVG + report_index.json) · pages · compare

.github/workflows/      # the periodic regenerate-and-publish workflow, with its diagrams
analyze-report/         # LLM consumption: workflows, uncovered-op backlog, analysis draft
stats/ops_manifest.json # what discover, derive and probe learned (committed)
stats/runs/             # one provenance record per measurement run (committed)
data/                   # raw CSVs — gitignored; symlink to a fast local disk
reports/                # generated markdown + SVG (committed): by_arch · by_op · by_dtype
report_index.json       # summary stats per arch/dtype/op/variant (committed)
```

## Setup

```bash
source <tt-metal>/python_env/bin/activate   # missing? run <tt-metal>/create_venv.sh
export TT_METAL_HOME=<tt-metal>
uv pip install -e .
ln -s /localdev/$USER/data data             # data/ grows fast
```

## Pipeline

| Command | Does | Device |
|---|---|---|
| `discover` | ttnn registry → classified eltwise manifest, refusal reasons recorded | no |
| `derive` | per-dtype input bounds from each unary golden, in fp64 | no |
| `probe` | which dtype/layout each op accepts, keyed per arch | **yes** |
| `measure` | the sweeps → one CSV per op variant | **yes** |
| `charts` | CSV → SVG + `report_index.json` | no |
| `report` | index + SVG → markdown tree | no |
| `compare` | two indexes → what moved; exit 1 on regression | no |

```bash
ttnn-accuracy discover && ttnn-accuracy derive && ttnn-accuracy probe
for cat in unary unary_bw binary binary_bw ternary ternary_bw; do
  ttnn-accuracy measure --arch wh --category $cat --dtype both
done
ttnn-accuracy charts
ttnn-accuracy report --categories unary,unary_bw,binary,binary_bw,ternary,ternary_bw
```

`--ops exp,gelu` narrows; `--arch bh` targets Blackhole (verified against the device).
Run sweeps in `tmux`; a kill mid-dispatch wedges the device (`tt-smi -glx_reset`).

## Coverage

196 ops — 79 unary · 56 unary_bw · 41 binary · 12 binary_bw · 5 ternary · 3 ternary_bw —
discovered from ttnn's own registry, never a hand list. Parameters, variants, supplied
goldens and exclusions all live in [ops/overrides.py](src/ttnn_accuracy/ops/overrides.py):
covering a new op is one entry. Everything uncovered, with reasons and sketches:
[analyze-report/uncovered.md](analyze-report/uncovered.md).

| Sweep | Points per op | Time |
|---|---|---|
| unary bf16 | 6.6e4 | ~2 s |
| unary fp32 | 4.3e9 | ~2 min |
| binary bf16 / fp32 | 4.2e9 | ~70 s |
| ternary bf16 | 1.1e9 | ~45 s |

## Reading the numbers

| ULP error | Assessment |
|---|---|
| ≤ 1 | bit-accurate |
| ≤ 3 | accurate |
| ≤ 10 | approximate |
| > 10 | poor |

Sampled sweeps (fp32 pairs, ternary) say so on every page — their maxima are lower
bounds. `usable_to` states how far an op stays within 2 ULP, so a cliff (`sin` past 2.6e5)
is not mistaken for a broken op.

## Based on

[ttnn-eltwise-op-tester](https://github.com/nmauriceTT/ttnn-eltwise-op-tester), extended
with automatic discovery, derived domains, per-arch probing, variants, provenance, and
the cross-navigable report tree.
