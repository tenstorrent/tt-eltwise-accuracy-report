# tt-eltwise-accuracy-report

Accuracy measurements for TT-Metal elementwise ops (bf16 and fp32) on Wormhole and Blackhole hardware.

**Browse reports:** [reports/README.md](reports/README.md)

---

## Repository Structure

```
src/ttnn_accuracy/
  cli.py                # ttnn-accuracy measure | charts | report
  paths.py
  ops/registry.py       # Op definitions: variants, domains, golden fns
  measure/
    metrics.py          # ULP / abs / rel — pure, no device
    sweeps.py           # bf16 exhaustive, fp32 in blocks
    runner.py           # ops × dtypes × variants → CSV
    device.py
  report/
    charts.py           # CSV → SVG
    pages.py            # SVG + stats → markdown tree

tests/unit/             # runs without hardware

data/                   # Raw CSVs — gitignored, symlink this to a fast local disk
  {arch}/{dtype}/{op_name}/{variant}.csv

reports/                # Generated markdown + SVG charts (committed)
  README.md             # Top-level navigation
  charts/               # SVG files (embedded in markdown)
  by_arch/              # Browse: arch → dtype → op
  by_op/                # Browse: op → arch/dtype comparison
  by_dtype/             # Browse: dtype → ops × archs
```

## Browsing Reports

Reports are GitHub-browsable markdown. Start at [reports/README.md](reports/README.md):

- **By Architecture** — choose WH or BH, then dtype, then op
- **By Operation** — choose op, see it across all archs and dtypes
- **By Data Type** — choose bf16 or fp32, see all ops on all archs

## Setup

Measurements need a TT device. This package installs into tt-metal's `python_env` — `ttnn`
and `torch` come from there, which is why they are not declared in `pyproject.toml`.

```bash
source <path/to/tt-metal>/python_env/bin/activate
export TT_METAL_HOME=<path/to/tt-metal>
uv pip install -e .
```

If `python_env` does not exist yet, run `./create_venv.sh` in tt-metal first — it is a separate
step from `./build_metal.sh`, and skipping it is what produces
`ModuleNotFoundError: No module named 'ttnn'`.

`data/` is gitignored and grows fast; on a machine with a small home quota, symlink it:
`ln -s /localdev/$USER/data data`.

## Running Measurements

```bash
# All ops, bf16 and fp32, on Wormhole
ttnn-accuracy measure --arch wh --ops all --dtype both

# Specific ops
ttnn-accuracy measure --arch wh --ops exp,gelu,tanh --dtype bf16

# Only unary ops on Blackhole
ttnn-accuracy measure --arch bh --category unary --dtype both
```

`--dtype fp32` walks the whole fp32 code space in 1024 device blocks per op variant. Start with
`bf16`.

## Regenerating Charts and Reports

```bash
ttnn-accuracy charts     # CSVs → SVG, updates report_index.json
ttnn-accuracy report     # SVG + stats → markdown tree
```

Both accept `--arch`, `--dtype`, and `--op`/`--categories` to regenerate a subset.

## Op Coverage

Ops are defined in `src/ttnn_accuracy/ops/registry.py`. Each op entry specifies:
- **Variants** — parameter configurations tested (e.g., `elu alpha=0.5/1.0/2.0`, `gelu default/fast_approx`)
- **Valid input range** — domain restrictions applied during data collection
- **Golden function** — PyTorch/mpmath reference for comparison

| Category | Examples |
|----------|---------|
| Unary | abs, exp, log, sqrt, tanh, sin, cos, gelu, elu (×3 α), selu, relu_max (×3), ... |
| Binary | add, multiply, divide (default + accurate), hypot, pow, atan2, ... |
| Unary backward | exp_bw, log_bw, tanh_bw, gelu_bw, sin_bw, ... |

## Accuracy Thresholds

Charts show ULP (units in last place) error vs input value.

| ULP error | Assessment |
|-----------|-----------|
| ≤ 1 | Excellent (bit-accurate) |
| ≤ 3 | Accurate |
| ≤ 10 | Approximate |
| > 10 | Poor |

## Based On

Inspired by [Nathan Maurice's ttnn-eltwise-op-tester](https://github.com/nmauriceTT/ttnn-eltwise-op-tester),
extended with:
- Valid domain ranges per op (prevents misleading accuracy data outside the meaningful range)
- Multiple parameter variants per op (alpha, beta, approximate modes)
- SVG charts (git-friendly, renders inline on GitHub)
- Multi-arch support (WH and BH)
- Cross-navigable markdown report structure
