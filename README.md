# tt-eltwise-accuracy-report

ULP accuracy and kernel timing for every TT-Metal eltwise op, measured exhaustively against
fp64 torch goldens, per architecture (WH, BH) and dtype (bf16, fp32).

[Browse the report](reports/README.md) ·
[Ask an assistant](analyze-report/README.md) ·
[Design](docs/architecture.md) ·
[Scope](docs/scope.md)

## Stats

| | |
|---|---|
| Ops in scope | 222, in 44 measured variants: 92 unary, 56 unary_bw, 54 binary, 12 binary_bw, 5 ternary, 3 ternary_bw |
| Measured | 827 variants: 2 architectures × 2 dtypes, every one also timed |
| Bit-exact | 274 |
| Within 2 ULP | 165 |
| Returning inf or zero where a value exists | 192 variants across 55 ops |
| Points per op | bf16 6.6e4 exhaustive, fp32 4.3e9 in blocks, binary 4.2e9 pairs |

| ULP error | Assessment |
|---|---|
| ≤ 1 | bit-accurate |
| ≤ 3 | accurate |
| ≤ 10 | approximate |
| > 10 | poor |

`usable_to` states how far an op stays within 2 ULP, so a cliff (`sin` past 2.6e5) is not
mistaken for a broken op. Sampled sweeps say so on every page: their maxima are lower bounds.

## Workflows

Everything runs from the Actions tab on the shared org runner pool, in tt-metal's own
container images. No machine of your own, nothing to provision. Only a complete
`accuracy-report` sweep writes to the repository, so no other run can overwrite a published
baseline.

| Workflow | Trigger | Answers | Output |
|---|---|---|---|
| [validate-kernel](.github/workflows/validate-kernel.yml) | dispatch | did my change break or slow anything? | pass/fail and a log |
| [accuracy-report](.github/workflows/accuracy-report.yml) | Sun cron, nightly cron, dispatch | what does every op do, and what happens at a parameter the report lacks? | commits the report, or an artifact |
| [analyze-report](.github/workflows/analyze-report.yml) | after an accuracy-report | what moved? | commits findings, opens an issue |
| [perf-report](.github/workflows/perf-report.yml) | dispatch | how fast, and what did this commit cost? | timings artifact |

Details and the container contract: [.github/workflows/README.md](.github/workflows/README.md).

<details>
<summary><b>validate-kernel</b>: the one to reach for</summary>

Name a tt-metal commit and the ops your change touches. It builds that commit, measures those
ops, diffs them against the published report, times them, and **fails when something got
worse**. Nothing is committed. The report is the baseline, never the output.

| Input | |
|---|---|
| `commit` | tt-metal commit or branch to validate |
| `ops` | ops the change touches, comma separated |
| `arch`, `dtype` | `wh`/`bh`, `bf16`/`fp32`/`both` |
| `max_ulp` | optional absolute bar, independent of the baseline |
| `params` | optional scalar, e.g. `relu_max=6` |

```
exp [default] bf16: 366.689 us, 45753.3 Melem/s (spread 7.9%)
gelu [fast_approx] fp32: 687.075 us, 24418.3 Melem/s (spread 0.9%)
8 variant(s) measured — nothing moved
```

The exit code counts findings: regressions, variants over the bar, slower timings, variants
that produced no data. So it gates a generated-kernel loop directly.

</details>

<details>
<summary><b>accuracy-report</b>: the whole report weekly, an early warning nightly, an answer on demand</summary>

One workflow, because the three uses differ only in how much they measure.

| Use | How | Scope | Output |
|---|---|---|---|
| the report | cron Sunday 02:37 | six categories × both dtypes × wh and bh | commits |
| the early warning | cron Mon–Sat 02:37 | unary, bf16 | `compare` on the run page |
| a question | dispatch `ops` + `params`, e.g. `relu_max=6` | what you name | run page and artifact |

One tt-metal release tag is pinned per run, so both architectures measure the same version.
A complete sweep also discovers, derives domains, probes layouts and times every variant;
12 h per architecture, one at a time because both push the same branch.

**Publishing is derived, not requested.** A run commits only when it measured everything —
`ops` and `params` empty, `dtype: both`, all six categories — because `charts` merges into
the published index, and a partial run would leave it half at one tt-metal commit and half
at another. `publish: false` can withhold publication from a full sweep; nothing can force
it onto a partial one.

</details>

<details>
<summary><b>analyze-report</b>: what moved</summary>

Runs on ubuntu-latest with no device: `compare` is pure JSON over two indexes. Buckets every
change into regressed, improved, expected, new or removed, writes
[findings.md](analyze-report/findings.md), and opens an issue when something regressed.

The page is produced by rules, not by a model, because the buckets already are the
classification. Set `ANTHROPIC_API_KEY` and a model adds one short note on top.

</details>

<details>
<summary><b>perf-report</b>: timing, and the A/B between two builds</summary>

`commit` alone times that build. Add `against` and both are built and timed **on the same
host, back to back**, and the run reports what moved, because a timing is a property of the
machine as much as of the kernel. `ops` and `dtype` narrow it.

Timings never enter the accuracy score: a difference here can be the room.

</details>

## CLI

<details>
<summary><b>Setup</b>: once per machine</summary>

```bash
source <tt-metal>/python_env/bin/activate   # missing? run <tt-metal>/create_venv.sh
export TT_METAL_HOME=<tt-metal>
uv pip install -e .
ln -s /localdev/$USER/data data             # data/ grows fast
```

</details>

<details>
<summary><b>check</b>: validate a kernel locally</summary>

What `validate-kernel` runs, against the tt-metal you already have built.

```bash
ttnn-accuracy check --arch wh --ops exp,gelu --dtype both
ttnn-accuracy check --arch wh --ops exp --perf              # also time them
ttnn-accuracy check --arch wh --ops gelu --max-ulp 2        # fail over an absolute bar
ttnn-accuracy check --arch wh --ops relu_max --params relu_max=6
```

The baseline is read from git, because accuracy is deterministic and re-measuring it
reproduces the same numbers. Timings are compared only when the published ones came from this
same host. An op ttnn has gained since the manifest was built is brought into scope for you,
before the device is taken.

</details>

<details>
<summary><b>The full pipeline</b>: what a complete sweep runs</summary>

```bash
ttnn-accuracy discover && ttnn-accuracy derive && ttnn-accuracy probe
for cat in unary unary_bw binary binary_bw ternary ternary_bw; do
  ttnn-accuracy measure --arch wh --category $cat --dtype both
done
ttnn-accuracy perf --arch wh
ttnn-accuracy charts
ttnn-accuracy report --categories unary,unary_bw,binary,binary_bw,ternary,ternary_bw
```

| Command | Does | Device |
|---|---|---|
| `discover` | ttnn registry to a classified eltwise manifest, refusal reasons recorded | no |
| `derive` | per-dtype input bounds from each unary golden, in fp64 | no |
| `probe` | which dtype and layout each op accepts, keyed per arch | **yes** |
| `measure` | the sweeps, to one CSV per op variant | **yes** |
| `perf` | one resident tensor per op, timed, to `stats/perf/{arch}.json` | **yes** |
| `charts` | CSV to SVG and `report_index.json`, timings attached | no |
| `export` | measured data in the tt-llk harness schema, for its dashboard's Load CSV | no |
| `report` | index and SVG to a markdown tree | no |
| `refine` | sweep a sampled worst point exhaustively; exit counts looser bounds | **yes** |
| `compare` | two indexes to what moved; exit 1 on regression | no |
| `history` | every published index to which build moved a number | no |
| `perf-diff` | two timing files to what got slower; refuses two hosts | no |

```bash
ttnn-accuracy history --op sin        # when did it break, and on whose commit
ttnn-accuracy refine --arch wh --ops pow --dtype fp32   # how loose is a sampled maximum
ttnn-accuracy export --arch wh --op exp                 # then drag into the LLK dashboard
```

The charts are static SVG because GitHub strips scripts from markdown, so `export` writes the
measurements in [tt-llk](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tests/python_tests/accuracy)'s
own 19-column schema and the [SFPU dashboard](https://github.com/tenstorrent/llk-sfpu-dashboard)
renders them with zoom, pan and sub-range recompute. Its aggregator reports the same max ULP
and exact fraction our pages do, from the same points. One op is about 10 MB, so export the
ops you are looking at rather than the tree.

Binary fp32 and every ternary sweep are sampled, so their maxima are lower bounds and the
pages say so. `refine` holds the other operands at a variant's worst point and sweeps one
across its whole space: `fp32/pow` published 5,918 ULP and the cell it was drawn from holds
8,300.

Run sweeps in `tmux`. A kill mid-dispatch wedges the device, recovered with `tt-smi -glx_reset`.

</details>

<details>
<summary><b>Layout</b>: where things live</summary>

```
src/ttnn_accuracy/
  cli.py                # discover | derive | probe | measure | check | perf | perf-diff
                        # | refine | charts | report | compare | history
  ops/
    introspect.py       # what ttnn registers
    arity.py            # operand count, elementwise, real-valued, by asking the golden
    manifest.py         # builds and holds stats/ops_manifest.json
    plan.py             # manifest to one OpSpec per variant
    overrides.py        # every editorial decision: scalars, variants, supplied goldens, exclusions
  domain/derive.py      # per-dtype input bounds from the golden in float64
  measure/              # metrics · sweeps · schema · store · runner · device · perf
  config.py             # every tunable number: sweep sizing, timing samples, chart thresholds
  report/
    score.py            # one CSV to its index entry: stats, percentiles, verdict
    charts.py           # score to SVG (error, CDF, per-bin percentiles) and report_index.json
    index.py            # report_index.json as the pages read it, and every display name
    pages.py            # one op's page · navigation.py: the index pages · ask.py: the ask page
    tree.py             # writes the whole reports/ tree
    compare.py          # two indexes to what moved

analyze-report/         # how an assistant reads it: contract, ask page, uncovered backlog
stats/ops_manifest.json # what discover, derive and probe learned (committed)
stats/runs/             # one provenance record per measurement run (committed)
stats/perf/             # µs per variant, with the host that took them (committed)
data/                   # raw CSVs, gitignored; symlink to a fast local disk
reports/                # generated markdown and SVG (committed): by_arch · by_op · by_dtype
report_index.json       # summary stats per arch/dtype/op/variant (committed)
```

Ops are discovered from ttnn's own registry, never a hand list. Covering a new one is a single
entry in [overrides.py](src/ttnn_accuracy/ops/overrides.py); everything uncovered is listed
with reasons in [uncovered.md](analyze-report/uncovered.md).

</details>

## Based on

[ttnn-eltwise-op-tester](https://github.com/nmauriceTT/ttnn-eltwise-op-tester), extended with
automatic discovery, derived domains, per-arch probing, variants, timing, provenance, and the
cross-navigable report tree.
