# tt-eltwise-accuracy-report

Accuracy measurements for TT-Metal elementwise ops (bf16 and fp32) on Wormhole and Blackhole hardware.

**Browse reports:** [reports/README.md](reports/README.md)

---

## Repository Structure

```
scripts/
  ops_registry.py       # Op definitions: variants, valid input ranges, golden fns
  measure_accuracy.py   # Data collection (run on hardware)
  generate_charts.py    # CSV → SVG charts
  generate_reports.py   # SVG + data → browsable markdown report tree
  requirements.txt

data/                   # Raw CSV measurements (committed)
  {arch}/
    {dtype}/
      {op_name}/
        {variant}.csv

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

## Running Measurements

Measurements require a TT device and a TT-Metal environment.

```bash
cd scripts

# Install Python dependencies (if not already in environment)
pip install -r requirements.txt

# Measure all ops, bf16 and fp32, on Wormhole
python measure_accuracy.py --arch wh --ops all --dtype both

# Measure specific ops
python measure_accuracy.py --arch wh --ops exp,gelu,tanh --dtype bf16

# Measure only unary ops on Blackhole
python measure_accuracy.py --arch bh --category unary --dtype both
```

## Regenerating Charts and Reports

After collecting new data:

```bash
cd scripts

# Generate SVG charts from CSVs
python generate_charts.py

# Regenerate full markdown report tree
python generate_reports.py
```

Both scripts support `--arch`, `--dtype`, and `--op` flags to regenerate a subset.

## Op Coverage

Ops are defined in `scripts/ops_registry.py`. Each op entry specifies:
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
